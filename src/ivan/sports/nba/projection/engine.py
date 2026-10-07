"""
project_team(): roster + availability + history -> per-player projection.

Inputs are plain records so the same code serves the live slate (NBAPlayer
rows, Rotowire starters, a manual minutes CSV) and the walk-forward backtest
(actual availability and starters from the box score).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fpts import simulate_fpts, summarize, minutes_sd
from .history import HistoryState, STATS
from .minutes import MinutesInputs, allocate_team_minutes, baseline_minutes
from .rates import shrunk_rates, fpts_per_minute, apply_usage_bump, USAGE_RECOVERY
from .market import blend_rates


@dataclass
class RosterPlayer:
    name: str
    available: bool
    starter: bool | None = None       # confirmed starter; None = unknown
    salary: int | None = None
    minutes_override: float | None = None
    market_rates: dict[str, float] | None = None   # stat -> per-minute rate from the props
    rate_mult: float = 1.0                         # e.g. conditional "without X" usage


@dataclass
class ProjectionParams:
    start_shift: float = 6.0
    alloc: str = "headroom"
    cut_tau: float | None = 8.0
    use_play_prob: bool = True
    recovery: float = USAGE_RECOVERY
    prior_minutes: float = 60.0
    n_sims: int = 4000
    dispersion: dict | None = None
    market_weight: float = 1.0


def project_team(
    roster: list[RosterPlayer],
    state: pd.DataFrame,              # HistoryState.before(date), indexed by player_name
    history: HistoryState,
    team_fpts: float | None = None,   # trailing team production; league average when None
    params: ProjectionParams | None = None,
    rng: np.random.Generator | None = None,
    simulate: bool = True,
) -> pd.DataFrame:
    """
    Returns one row per *available* player: minutes, per-stat means, fpts_mean,
    fpts_std, plus baseline minutes and the usage multiplier for inspection.
    Absent players are listed only so the caller can see who was excluded;
    the usage bump works from the team's trailing production, not from them.
    """
    params = params or ProjectionParams()
    rng = rng or np.random.default_rng(0)

    def srow(name):
        return state.loc[name] if name in state.index else None

    # per-player rates and baseline minutes for everyone on the roster
    recs = {}
    for p in roster:
        s = srow(p.name)
        rates = shrunk_rates(s, history.league_rates, params.prior_minutes)
        rates = blend_rates(rates, p.market_rates, params.market_weight)
        if p.rate_mult != 1.0:
            rates = {k: v * p.rate_mult for k, v in rates.items()}
        mi = MinutesInputs(
            name=p.name,
            min_ewm=None if s is None else float(s["min_ewm"]),
            start_ewm=0.0 if s is None else float(s["start_ewm"]),
            starter=p.starter,
            play_prob=1.0 if (s is None or not params.use_play_prob) else float(s["play_ewm"]),
            salary=p.salary, minutes_override=p.minutes_override,
        )
        recs[p.name] = dict(inputs=mi, rates=rates, fpm=fpts_per_minute(rates),
                            priced=bool(p.market_rates),
                            baseline=baseline_minutes(mi, params.start_shift),
                            hist_sd=None if s is None else float(s["min_ewm_sd"]),
                            games=0 if s is None else int(s["games"]),
                            available=p.available)

    avail = [p for p in roster if p.available]
    absent = [p for p in roster if not p.available]
    alloc = allocate_team_minutes([recs[p.name]["inputs"] for p in avail],
                                  start_shift=params.start_shift, alloc=params.alloc,
                                  cut_tau=params.cut_tau)
    alloc["fpm"] = [recs[n]["fpm"] for n in alloc.index]
    team_total = history.league_team_fpts if team_fpts is None else float(team_fpts)
    bump = apply_usage_bump(alloc, team_total, params.recovery)

    rows = []
    for name in alloc.index:
        r = recs[name]
        mins = float(alloc.loc[name, "minutes"])
        rates = {s: r["rates"][s] * float(bump[name]) for s in STATS}
        rec = {"player_name": name, "minutes": mins, "baseline": r["baseline"],
               "pinned": bool(alloc.loc[name, "pinned"]), "usage_mult": float(bump[name]),
               "games": r["games"], "fpm": r["fpm"] * float(bump[name]), "priced": r["priced"]}
        for s in STATS:
            rec[f"proj_{s}"] = mins * rates[s]
        if simulate and mins > 0:
            samples = simulate_fpts(mins, rates, n=params.n_sims,
                                    min_sd=minutes_sd(mins, r["hist_sd"]),
                                    dispersion=params.dispersion, rng=rng)
            rec["fpts_mean"], rec["fpts_std"] = summarize(samples)
        else:
            rec["fpts_mean"] = mins * rec["fpm"]
            rec["fpts_std"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows).set_index("player_name")
