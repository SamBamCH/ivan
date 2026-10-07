"""
Per-minute stat rates.

A player's rate for each stat is his minute-weighted trailing rate shrunk
toward the league rate with PRIOR_MINUTES of pseudo-weight. The usage bump
then scales every present player's rates by one factor so that a fraction
USAGE_RECOVERY of the gap between the team's trailing fantasy production and
the present roster's projected production is closed. A team still has to
score, rebound and assist when a star sits; the minutes allocation moves the
volume, this moves the per-minute efficiency the rest of the way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .history import STATS

PRIOR_MINUTES = 60.0
USAGE_RECOVERY = 0.5   # backtest 2026-10-06: 0.25-0.5 flat, zero star-out bias at 0.25-0.5

DK_WEIGHTS = {"points": 1.0, "fg3m": 0.5, "rebounds": 1.25, "assists": 1.5,
              "steals": 2.0, "blocks": 2.0, "turnovers": -0.5}


def shrunk_rates(state_row: pd.Series | None, league: dict[str, float],
                 prior_minutes: float = PRIOR_MINUTES) -> dict[str, float]:
    if state_row is None:
        return dict(league)
    eff = float(state_row["eff_min"])
    w = eff / (eff + prior_minutes)
    return {s: w * float(state_row[f"rate_{s}"]) + (1 - w) * league[s] for s in STATS}


def fpts_per_minute(rates: dict[str, float]) -> float:
    return sum(DK_WEIGHTS[s] * rates[s] for s in STATS)


def apply_usage_bump(
    present: pd.DataFrame,          # index name; columns minutes, fpm, priced
    team_fpts: float,               # trailing team fantasy points per game
    recovery: float = USAGE_RECOVERY,
) -> pd.Series:
    """
    One multiplier for the *unpriced* present players so that a fraction
    `recovery` of the gap between the team's trailing production and the
    present roster's projected production is closed. Priced players are
    gospel: the books have already priced their role, so they keep their
    market rate and only the rest of the roster absorbs the gap. Symmetric: a
    roster that projects above the team's usual total is scaled down.
    """
    prod = present["minutes"] * present["fpm"]
    priced = present["priced"].astype(bool) if "priced" in present else pd.Series(False, index=present.index)
    free = prod[~priced].sum()
    total = float(prod.sum())
    mult = pd.Series(1.0, index=present.index)
    if free <= 0 or not np.isfinite(team_fpts):
        return mult
    m = 1.0 + recovery * (team_fpts - total) / free
    mult[~priced] = float(np.clip(m, 0.7, 1.4))
    return mult
