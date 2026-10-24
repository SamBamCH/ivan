# src/ivan/sports/nba/projection/book_minutes.py
"""
The minutes the sportsbooks are pricing a player for.

A minutes market, when a book offers one (stat "minutes" in the market means), is used as is. Otherwise the
minutes are implied by the counting-stat props: each priced stat's market mean divided by the player's
per-minute rate from history (shrunk to the league, projection.rates.shrunk_rates), the stats averaged with
their rates as weights (points carry the most), times SCALE.

SCALE corrects a structural lean: market means over history rates land ~2 minutes above the minutes played,
on both seasons. Fitted 2026-10-23 on last season's 11 field days (996 priced player-games, historical props),
tested on this season's 7 slates (329 player-games, props at lock): mean miss 4.06 minutes, bias +0.01,
against 4.92 for our minutes model and 5.42 for history alone. Needs at least two of points, rebounds, assists.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .rates import shrunk_rates

SCALE = 0.924
STATS = ("points", "rebounds", "assists")


def book_minutes(means: Dict[str, float], state_row: Optional[pd.Series], league_rates: Dict[str, float]) -> Optional[dict]:
    """{minutes, source ('market' | 'implied'), by_stat {stat: minutes}} or None when the books do not say enough."""
    m = means.get("minutes")
    if m is not None and pd.notna(m) and m > 0:
        return dict(minutes=float(m), source="market", by_stat={"minutes": float(m)})
    if state_row is None:
        return None
    rates = shrunk_rates(state_row, league_rates)
    ks = [k for k in STATS if means.get(k) is not None and pd.notna(means.get(k)) and rates.get(k, 0) > 0]
    if len(ks) < 2:
        return None
    by = {k: float(means[k]) / float(rates[k]) for k in ks}
    implied = float(np.average([by[k] for k in ks], weights=[rates[k] for k in ks]))
    return dict(minutes=SCALE * implied, source="implied", by_stat={k: round(SCALE * v, 1) for k, v in by.items()})
