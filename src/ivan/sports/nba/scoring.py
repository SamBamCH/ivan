"""
DraftKings NBA classic scoring, in one place.

    STATS        the seven counting stats a projection carries, by the names every table and frame uses
    DK_WEIGHTS   points per unit of each
    dk_points    fantasy points from a mapping of those stats, scalars or arrays alike, with the double-double (+1.5) and
                 triple-double (+3, on top) bonuses from the joint values
    from_nba_api a box-score or live block from nba_api / the live feed, re-keyed to STATS
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

STATS = ("points", "rebounds", "assists", "steals", "blocks", "turnovers", "fg3m")
DK_WEIGHTS = {"points": 1.0, "fg3m": 0.5, "rebounds": 1.25, "assists": 1.5, "steals": 2.0, "blocks": 2.0, "turnovers": -0.5}
DOUBLE_DOUBLE, TRIPLE_DOUBLE = 1.5, 3.0
_TENS = ("points", "rebounds", "assists", "steals", "blocks")
NBA_API_KEYS = {"points": "points", "rebounds": "reboundsTotal", "assists": "assists", "steals": "steals", "blocks": "blocks",
                "turnovers": "turnovers", "fg3m": "threePointersMade"}


def dk_points(stats: Mapping[str, float | np.ndarray]):
    """Fantasy points from {stat: value} (any of STATS missing counts 0). Arrays give an array, numbers a float."""
    val = {s: np.asarray(stats.get(s, 0.0), dtype=float) for s in STATS}
    fp = sum(DK_WEIGHTS[s] * val[s] for s in STATS)
    tens = sum((val[s] >= 10).astype(int) for s in _TENS)
    out = fp + DOUBLE_DOUBLE * (tens >= 2) + TRIPLE_DOUBLE * (tens >= 3)
    return float(out) if np.ndim(out) == 0 else out


def from_nba_api(block: Mapping) -> dict:
    """An nba_api box-score row or live-feed statistics block -> {stat: float} by STATS (None and missing read 0)."""
    return {s: float(block.get(k) or 0) for s, k in NBA_API_KEYS.items()}
