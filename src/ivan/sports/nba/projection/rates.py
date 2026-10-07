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
    present: pd.DataFrame,          # index name; columns minutes, fpm (per-minute fpts)
    team_fpts: float,               # trailing team fantasy points per game
    recovery: float = USAGE_RECOVERY,
) -> pd.Series:
    """
    One multiplier for every present player so that a fraction `recovery` of
    the gap between the team's trailing production and the present roster's
    projected production is closed. Symmetric: a roster that projects above
    the team's usual total (a star back from injury while his teammates'
    trailing rates are still inflated) is scaled down the same way.
    """
    new_prod = float((present["minutes"] * present["fpm"]).sum())
    if new_prod <= 0 or not np.isfinite(team_fpts):
        return pd.Series(1.0, index=present.index)
    mult = 1.0 + recovery * (team_fpts - new_prod) / new_prod
    return pd.Series(float(np.clip(mult, 0.7, 1.4)), index=present.index)
