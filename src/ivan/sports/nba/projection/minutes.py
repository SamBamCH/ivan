"""
Minutes: baseline per player, then a team-level allocation so the roster that
actually plays adds up to a game's worth of minutes.

Baseline  mu_i = play_i * min_ewm_i + START_SHIFT * (starter_i - start_ewm_i)
          play_i is how often he has played when dressed (1 for a confirmed
          starter). A confirmed starter who has been coming off the bench
          moves up, a demoted starter moves down. Players with no history get a salary
          prior (rookies, two-way call-ups).

Allocation  The team budget is TEAM_MINUTES (240 plus the expected overtime
          share). When the available players' baselines fall short (someone
          is out) the gap is handed out with weights
              w_i = mu_i * max(CAP - mu_i, 1)
          so a 30-minute player absorbs more than a 5-minute one, but a
          36-minute player has little room left. When they exceed the budget
          (a DK pool lists the whole roster, and a coach plays nine or ten)
          the surplus comes off the bottom of the rotation:
              w_i = mu_i * exp(-mu_i / CUT_TAU)
          so the 4-minute player loses his 4 minutes and the star loses a few
          seconds. Manual overrides are fixed first and the remainder is
          allocated among the rest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TEAM_MINUTES = 241.0   # 240 + average overtime (measured 2025-26 regular season)
CAP = 38.0             # practical ceiling for an allocated starter
START_SHIFT = 6.0      # minutes: confirmed starter vs bench at the same history
FLOOR_FOR_SHARE = 4.0  # players below this baseline absorb nothing unless listed
MAX_ALLOCATED = 42.0
CUT_TAU = 8.0          # minutes scale of the soft rotation cut (surplus case)


@dataclass
class MinutesInputs:
    """One available player. `minutes_override` pins the player."""
    name: str
    min_ewm: float | None       # None when no history
    start_ewm: float            # 0..1 share of recent games started
    starter: bool | None        # confirmed starter flag; None = unknown
    play_prob: float = 1.0      # share of recent dressed games he actually played
    salary: int | None = None
    minutes_override: float | None = None
    salary_w: float = 0.0       # share of the baseline from the salary curve (salary_weight)


SALARY_CURVE_JSON = None   # set below (data/nba/models/salary_minutes.json when fitted by scripts/nba/validate.py fit-salary)
# minutes = a + b*k + c*max(k - knee, 0), k = (salary - 3000) / 1000; fitted on the 2026-10-20/21 slates (316 player-games,
# residual sd 6.4): DK's price is a better minutes predictor than last season's minutes early in a season
SALARY_CURVE = {"a": 10.3, "b": 6.31, "c": -5.34, "knee": 3.0, "floor": 4.0, "cap": 34.0}
# how much of the minutes baseline comes from the salary curve rather than the player's history, by where his
# history is; each weight decays as games arrive (halflife in games this season / on the new team)
# replayed on the 2026-10-20/21 captures: blending toward the curve made the minutes error worse for same-team players
# and left new-team players under-projected (the plateau pulls stars down), so only players with no history use it;
# the weights are here to revisit once a few weeks of scorecards exist
SALARY_BLEND = {"no_history": 1.0, "new_team": 0.0, "new_team_halflife": 3.0, "same_team": 0.0, "same_team_halflife": 6.0}


def _load_salary_curve() -> dict:
    try:
        from ivan.sports.nba.paths import MODELS_DIR
        pth = MODELS_DIR / "salary_minutes.json"
        if pth.exists():
            d = json.loads(pth.read_text())
            return {**SALARY_CURVE, **{k: float(v) for k, v in d.items() if k in SALARY_CURVE}}
    except Exception:   # noqa: BLE001
        pass
    return dict(SALARY_CURVE)


def salary_prior_minutes(salary: int | None, curve: dict | None = None) -> float:
    """Expected minutes from the DraftKings salary alone (the fitted curve; a plateau above $6k)."""
    if salary is None:
        return 6.0
    c = curve or _CURVE
    k = (float(salary) - 3000.0) / 1000.0
    m = c["a"] + c["b"] * k + c["c"] * max(k - c["knee"], 0.0)
    return float(np.clip(m, c["floor"], c["cap"]))


def salary_weight(history_games: int, new_team: bool, games_on_team: int, games_this_season: int, blend: dict | None = None) -> float:
    """Share of the baseline taken from the salary curve: 1 with no history, high on a new team, modest on the same team, all fading with games."""
    b = blend or SALARY_BLEND
    if history_games <= 0:
        return b["no_history"]
    if new_team:
        return float(b["new_team"] * 0.5 ** (games_on_team / b["new_team_halflife"]))
    return float(b["same_team"] * 0.5 ** (games_this_season / b["same_team_halflife"]))


_CURVE = _load_salary_curve()


STARTER_PRIOR_W = 0.6      # a named starter who rarely started: this share of the way from his history to the team's starter minutes
                           # (a usual starter named to the bench: the same share toward the team's bench minutes); start_ewm scales it


def baseline_minutes(p: MinutesInputs, start_shift: float = START_SHIFT, team_starter_min: float | None = None,
                     team_bench_min: float | None = None, prior_w: float = STARTER_PRIOR_W) -> float:
    prior = salary_prior_minutes(p.salary)
    mu = prior if p.min_ewm is None else (1.0 - p.salary_w) * p.min_ewm + p.salary_w * prior
    if not p.starter:
        mu *= float(np.clip(p.play_prob, 0.0, 1.0))
    if p.starter is not None:
        if team_starter_min is not None and team_bench_min is not None:
            # the team's own numbers for the role: a new starter moves most of the way to what this coach gives a starter
            # (Houston 34.8, Golden State 28.0), a benched regular toward the bench's; a player whose history already
            # matches the call barely moves (start_ewm = his trailing share of starts)
            w = prior_w * ((1.0 - p.start_ewm) if p.starter else p.start_ewm)
            mu = (1.0 - w) * mu + w * (team_starter_min if p.starter else team_bench_min)
        else:
            mu += start_shift * (float(p.starter) - p.start_ewm)
    return float(np.clip(mu, 0.0, MAX_ALLOCATED))


def allocate_team_minutes(
    players: list[MinutesInputs],
    team_minutes: float = TEAM_MINUTES,
    cap: float = CAP,
    start_shift: float = START_SHIFT,
    alloc: str = "headroom",
    cut_tau: float | None = CUT_TAU,
) -> pd.DataFrame:
    """
    Returns a frame indexed by name with columns baseline, minutes, pinned.
    alloc: 'headroom' (default), 'prop' (proportional to baseline), 'none'
    (baselines unchanged; used as a backtest control).
    cut_tau: scale of the soft rotation cut for a surplus; None = proportional.
    """
    names = [p.name for p in players]
    base = np.array([baseline_minutes(p, start_shift) for p in players])
    pinned = np.array([p.minutes_override is not None for p in players])
    mins = base.copy()
    mins[pinned] = [p.minutes_override for p in players if p.minutes_override is not None]

    if alloc != "none":
        free = ~pinned
        budget = team_minutes - mins[pinned].sum()
        gap = budget - mins[free].sum()
        # a few passes so clipping at the cap re-distributes the overflow
        for _ in range(8):
            if abs(gap) < 0.05 or not free.any():
                break
            mu = mins[free]
            if gap > 0:
                if alloc == "prop":
                    w = mu.copy()
                else:
                    w = mu * np.maximum(cap - mu, 1.0)
                w[mu < FLOOR_FOR_SHARE] = 0.0
                if w.sum() <= 0:
                    w = np.ones_like(mu)
                new = np.minimum(mu + gap * w / w.sum(), MAX_ALLOCATED)
            else:
                w = mu.copy() if cut_tau is None else mu * np.exp(-mu / cut_tau)
                w[mu <= 0] = 0.0
                if w.sum() <= 0:
                    break
                new = np.maximum(mu + gap * w / w.sum(), 0.0)
            mins[free] = new
            gap = budget - mins[free].sum()

    return pd.DataFrame({"baseline": base, "minutes": mins, "pinned": pinned}, index=pd.Index(names, name="player_name"))
