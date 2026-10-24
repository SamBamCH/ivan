# src/ivan/sports/nba/public_projection.py
"""
A replica of the public's projection (Stokastic, the provider the field model was built
on last season), from public inputs only: the books' player props and box-score history.
The field model uses it; our own projections never enter it.

Priced players   stok ~= PRICED_SCALE * prop_sum + PRICED_SHIFT, where prop_sum is the DK points
                 implied by the consensus prop means (projection.market.market_means): points + 1.25
                 rebounds + 1.5 assists + 0.5 threes + 2 (steals + blocks), minus 0.5 x history's turnovers
Unpriced         a ridge on history-only fantasy points, salary, how often he starts, recent minutes

Fitted 2026-10-23 on 11 days of last season (2025-11-20 .. 2025-12-27; 1,363 player-days with a Stokastic
projection, historical props bought for those days): held out by day, the priced replica is within 1.13 pts
of Stokastic on average (bias 0.00, 761 player-days), the unpriced one within 3.05 (bias +0.03, 602). The raw
prop sum alone, unfitted, is within 1.60 (R^2 0.96): Stokastic is essentially the props where they exist.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ivan.sports.nba.projection.history import STATS, HistoryState
from ivan.sports.nba.projection.market import market_means
from ivan.sports.nba.projection.rates import DK_WEIGHTS, shrunk_rates

PRICED_SCALE, PRICED_SHIFT = 0.886, 3.57
UNPRICED = {"hist_fpts": 0.765, "salary_k": 1.852, "start_ewm": 1.009, "hist_min": 0.025}
UNPRICED_INTERCEPT = -4.96
PROP_WEIGHTS = {"points": 1.0, "rebounds": 1.25, "assists": 1.5, "fg3m": 0.5, "steals": 2.0, "blocks": 2.0}
CORE = ("points", "rebounds", "assists")


def history_features(state_row: Optional[pd.Series], league_rates: Dict[str, float]) -> dict:
    if state_row is None:
        return dict(hist_min=0.0, hist_fpts=0.0, hist_tov=0.0, start_ewm=0.0)
    rates = shrunk_rates(state_row, league_rates)
    mins = float(state_row["min_ewm"]) * float(state_row["play_ewm"])
    return dict(hist_min=mins, hist_fpts=mins * sum(DK_WEIGHTS[k] * rates[k] for k in STATS), hist_tov=mins * rates["turnovers"],
                start_ewm=float(state_row["start_ewm"]))


def prop_sum(means: Dict[str, float], hist_tov: float, hist: Optional[dict] = None) -> Optional[float]:
    """DK points implied by the prop means; None unless points, rebounds and assists are all priced. A missing minor market
    (threes, steals, blocks) counts as zero, as in the fit."""
    if any(means.get(k) is None or pd.isna(means.get(k)) for k in CORE):
        return None
    val = lambda k: 0.0 if means.get(k) is None or pd.isna(means.get(k)) else float(means[k])     # NaN is truthy: test it explicitly
    return float(sum(PROP_WEIGHTS[k] * val(k) for k in PROP_WEIGHTS) - 0.5 * hist_tov)


def project(players: pd.DataFrame, odds: pd.DataFrame, history: HistoryState, on: date) -> pd.Series:
    """
    players: rows with player_name (normalized) and salary. odds: the props table (player, market, sportsbook, side, point,
    price). Returns the public projection per player_name.
    """
    mm = market_means(odds).pivot(index="player", columns="stat", values="mean") if len(odds) else pd.DataFrame()
    st = history.before(on)
    out = {}
    for r in players.itertuples():
        h = history_features(st.loc[r.player_name] if r.player_name in st.index else None, history.league_rates)
        means = mm.loc[r.player_name].to_dict() if r.player_name in mm.index else {}
        ps = prop_sum(means, h["hist_tov"])
        if ps is not None:
            out[r.player_name] = PRICED_SCALE * ps + PRICED_SHIFT
        else:
            x = dict(h, salary_k=float(r.salary) / 1000.0)
            out[r.player_name] = max(0.0, UNPRICED_INTERCEPT + sum(UNPRICED[k] * x[k] for k in UNPRICED))
    return pd.Series(out, dtype=float)
