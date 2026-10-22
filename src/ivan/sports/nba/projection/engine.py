"""
project_team(): roster + availability + history -> per-player projection.

Inputs are plain records so the same code serves the live slate (NBAPlayer
rows, Rotowire starters, a manual minutes CSV) and the walk-forward backtest
(actual availability and starters from the box score).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .fpts import simulate_fpts, summarize, minutes_sd
from .history import HistoryState, STATS
from .minutes import MinutesInputs, allocate_team_minutes, baseline_minutes
from .rates import shrunk_rates, fpts_per_minute, apply_usage_bump, USAGE_RECOVERY
from .market import blend_rates
from .team_minutes import load_params as load_team_params, role_of, sample_team_minutes, scratch_probability, tau_from_band

MINUTES_MODELS = ("independent", "team")


@dataclass
class RosterPlayer:
    name: str
    available: bool
    starter: bool | None = None       # confirmed starter; None = unknown
    salary: int | None = None
    minutes_override: float | None = None
    market_rates: dict[str, float] | None = None   # stat -> per-minute rate from the props
    rate_mult: float = 1.0                         # e.g. conditional "without X" usage
    status: str | None = None                      # DK status (Q, GTD, ...): the team model's scratch probability
    fpm_override: float | None = None              # hand-set fantasy points per minute (every stat rate scaled to match)
    minutes_low: float | None = None               # hand-set 10-90 minutes band (team model: the player's own noise scale)
    minutes_high: float | None = None


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
    quantiles: int = 101          # points of the marginal kept for the correlated sampler
    minutes_model: str = "independent"   # "independent": each player's minutes a clipped normal; "team": the joint team draw
    rebalance: bool = False              # True: a pin re-allocates the rest of the team to 241; False: the others keep the model's
                                         # minutes and the difference is left for a person to place (the desk shows it)
    spread: float | None = None          # the team's own line for the team draw (negative = favoured); None = pick'em
    team_params: dict | None = None      # team_minutes.load_params() when None

    @classmethod
    def from_config(cls, config: dict | None = None, **overrides) -> "ProjectionParams":
        """projection.* from config/nba.json (minutes_model, n_sims), then keyword overrides."""
        if config is None:
            from ivan.core.config import load_config
            try:
                config = load_config("nba")
            except Exception:   # noqa: BLE001 - no config file is a plain default
                config = {}
        block = (config or {}).get("projection") or {}
        kw = {k: block[k] for k in ("minutes_model", "n_sims", "market_weight", "recovery", "rebalance") if k in block}
        kw.update(overrides)
        return cls(**kw)


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
    budget = None
    if not params.rebalance and any(p.minutes_override is not None for p in avail):
        # hand-set minutes do not move anyone else: the others keep the model's own allocation, every player is then
        # fixed, and the team total is whatever that adds to (the desk shows the minutes still to place)
        base = allocate_team_minutes([replace(recs[p.name]["inputs"], minutes_override=None) for p in avail],
                                     start_shift=params.start_shift, alloc=params.alloc, cut_tau=params.cut_tau)
        for p in avail:
            if p.minutes_override is None:
                recs[p.name]["inputs"] = replace(recs[p.name]["inputs"], minutes_override=float(base.loc[p.name, "minutes"]))
                recs[p.name]["soft_pin"] = True
        budget = float(sum(recs[p.name]["inputs"].minutes_override for p in avail))
    alloc = allocate_team_minutes([recs[p.name]["inputs"] for p in avail],
                                  start_shift=params.start_shift, alloc=params.alloc,
                                  cut_tau=params.cut_tau)
    if budget is not None:
        alloc["pinned"] = [not recs[n].get("soft_pin") for n in alloc.index]        # only the hand pins count as pins
    alloc["fpm"] = [recs[n]["fpm"] for n in alloc.index]
    alloc["priced"] = [recs[n]["priced"] for n in alloc.index]
    team_total = history.league_team_fpts if team_fpts is None else float(team_fpts)
    bump = apply_usage_bump(alloc, team_total, params.recovery)

    # the team draw: every available player's minutes in every world, jointly (team model only)
    if params.minutes_model not in MINUTES_MODELS:
        raise ValueError(f"unknown minutes_model {params.minutes_model!r}; expected one of {MINUTES_MODELS}")
    by_name = {p.name: p for p in avail}
    team_params = params.team_params or (load_team_params() if params.minutes_model == "team" else None)
    roles, p_out = {}, {}
    for name in alloc.index:
        p = by_name[name]
        st = recs[name]["inputs"]
        mu = float(alloc.loc[name, "minutes"])
        roles[name] = role_of(float(recs[name]["baseline"]), p.starter)
        p_out[name] = 0.0 if (alloc.loc[name, "pinned"] or p.starter) else scratch_probability(p.status, st.play_prob if params.use_play_prob else None, mu, team_params or {"scratch": {}})
    draw = None
    if params.minutes_model == "team" and simulate and params.n_sims > 0 and len(alloc):
        names = list(alloc.index)
        # a pin sets the mean (the allocation already holds it); a band sets the player's own spread; nobody is frozen
        taus = [tau_from_band(by_name[n].minutes_low, by_name[n].minutes_high) for n in names]
        draw = sample_team_minutes(alloc["minutes"].values, [roles[n] for n in names], params.n_sims, rng, spread=params.spread,
                                   p_out=[p_out[n] for n in names], params=team_params, tau_override=taus, budget=budget)

    rows = []
    for j, name in enumerate(alloc.index):
        r = recs[name]
        mins = float(alloc.loc[name, "minutes"])
        rates = {s: r["rates"][s] * float(bump[name]) for s in STATS}
        fpm = r["fpm"] * float(bump[name])
        fpm_override = by_name[name].fpm_override
        if fpm_override is not None and fpm > 0:
            rates = {s: v * float(fpm_override) / fpm for s, v in rates.items()}; fpm = float(fpm_override)
        rec = {"player_name": name, "minutes": mins, "baseline": r["baseline"],
               "pinned": bool(alloc.loc[name, "pinned"]), "usage_mult": float(bump[name]),
               "games": r["games"], "fpm": fpm, "priced": r["priced"], "fpm_pinned": fpm_override is not None,
               "band_pinned": tau_from_band(by_name[name].minutes_low, by_name[name].minutes_high) is not None,
               "role": roles[name], "p_out": p_out[name], "rates": rates}
        for s in STATS:
            rec[f"proj_{s}"] = mins * rates[s]
        if simulate and mins > 0 and params.n_sims > 0:
            if draw is not None:
                samples = simulate_fpts(mins, rates, n=params.n_sims, dispersion=params.dispersion, rng=rng, minutes=draw.minutes[:, j])
                rec["minutes_p10"], rec["minutes_p90"] = (float(x) for x in np.percentile(draw.minutes[:, j], [10, 90]))
            else:
                samples = simulate_fpts(mins, rates, n=params.n_sims,
                                        min_sd=minutes_sd(mins, r["hist_sd"]),
                                        dispersion=params.dispersion, rng=rng)
            rec["fpts_mean"], rec["fpts_std"] = summarize(samples)
            rec["quantiles"] = np.quantile(samples, np.linspace(0, 1, params.quantiles)).round(2).tolist()
        else:
            rec["fpts_mean"] = mins * rec["fpm"]
            rec["fpts_std"] = float("nan")
            rec["quantiles"] = None
        rows.append(rec)
    return pd.DataFrame(rows).set_index("player_name")
