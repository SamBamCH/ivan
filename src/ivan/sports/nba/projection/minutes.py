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


def salary_prior_minutes(salary: int | None) -> float:
    """Rough minutes prior for a player with no box score history."""
    if salary is None:
        return 6.0
    return float(np.clip((salary - 3000) / 200.0, 4.0, 32.0))


def baseline_minutes(p: MinutesInputs, start_shift: float = START_SHIFT) -> float:
    mu = salary_prior_minutes(p.salary) if p.min_ewm is None else p.min_ewm
    if not p.starter:
        mu *= float(np.clip(p.play_prob, 0.0, 1.0))
    if p.starter is not None:
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
