"""
Same-game correlation of NBA fantasy points: the measurement.

Measured on 2025-26 walk-forward residuals (scripts/nba/backtest_projection.py --fit-correlation): teammates who both
start ~0.00, starter with bench ~-0.05, bench with bench ~-0.03, opponents ~0.00, and a pair's own history is noise (its
first-half correlation predicts its second half at r = 0.08). The table (data/nba/models/correlation.json) is what the
game simulation's pairwise correlations are checked against (docs/nba/notes/game_sim_fit_2026-11-07.md); nothing
imposes it any more: the sampler is the game (ivan.sports.nba.gamesim.world).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ivan.sports.nba.paths import MODELS_DIR

CORRELATION_JSON = MODELS_DIR / "correlation.json"
STARTER_MINUTES = 24.0         # projected minutes that make a player "S" for the role pair


def role(minutes: float) -> str:
    return "S" if minutes >= STARTER_MINUTES else "B"


def category(same_team: bool, role_a: str, role_b: str) -> str:
    return ("T" if same_team else "O") + "".join(sorted([role_a, role_b]))


def fit_league(pairs: pd.DataFrame) -> dict:
    """pairs: columns cat, za, zb, date. League rho per category plus a split-half check."""
    out = {"league": {}, "split_half": {}}
    mid = pairs["date"].sort_values().iloc[len(pairs) // 2]
    for cat, d in pairs.groupby("cat"):
        out["league"][cat] = {"rho": float(np.corrcoef(d.za, d.zb)[0, 1]), "n": int(len(d))}
        a, b = d[d.date < mid], d[d.date >= mid]
        if len(a) > 50 and len(b) > 50:
            out["split_half"][cat] = [float(np.corrcoef(a.za, a.zb)[0, 1]), float(np.corrcoef(b.za, b.zb)[0, 1])]
    return out
