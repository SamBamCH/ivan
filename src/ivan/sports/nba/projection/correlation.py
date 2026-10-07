"""
Same-game correlation for NBA fantasy points.

Measured on 2025-26 walk-forward residuals (scripts/nba/backtest_projection.py
--fit-correlation), same-game correlations are small: teammates who both
start ~0.00, starter with bench ~-0.05, bench with bench ~-0.03, opponents
~0.00, and pair-specific history is noise (a pair's first-half correlation
predicts its second half at r=0.08). So the structure is a league table by
role pair, with each specific pair's own shared-game history shrunk toward it
by PAIR_SHRINK_GAMES pseudo-games: lineup history is used exactly as much as
it earns, which is very little. Availability effects (Luka out) are in the
means via the minutes allocation, not here.

The sampler is a Gaussian copula per game: correlated normals -> uniforms ->
each player's own marginal quantiles (from the Monte Carlo in fpts.py), so
tails, skew and double-double bumps survive instead of a Normal fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ivan.sports.nba.paths import MODELS_DIR

CORRELATION_JSON = MODELS_DIR / "correlation.json"
RESIDUALS_CSV = MODELS_DIR / "correlation_residuals.csv"
PAIR_SHRINK_GAMES = 200.0      # pseudo pair-games toward the league value
STARTER_MINUTES = 24.0         # projected minutes that make a player "S" for the role pair
DEFAULT_LEAGUE = {"TSS": 0.0, "TBS": -0.05, "TBB": -0.03, "OSS": 0.0, "OBS": 0.0, "OBB": 0.0}


def role(minutes: float) -> str:
    return "S" if minutes >= STARTER_MINUTES else "B"


def category(same_team: bool, role_a: str, role_b: str) -> str:
    return ("T" if same_team else "O") + "".join(sorted([role_a, role_b]))


def load_league() -> dict[str, float]:
    if CORRELATION_JSON.exists():
        try:
            return {k: float(v["rho"]) for k, v in json.loads(CORRELATION_JSON.read_text())["league"].items()}
        except Exception:
            pass
    return dict(DEFAULT_LEAGUE)


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


class PairHistory:
    """Residual z per player-game, for the pair-specific layer."""
    def __init__(self, residuals: pd.DataFrame | None):
        self.by_player: dict[str, pd.Series] = {}
        if residuals is not None and len(residuals):
            for name, d in residuals.groupby("player_name"):
                self.by_player[name] = pd.Series(d.z.values, index=d.game_id.values)

    @classmethod
    def load(cls) -> "PairHistory":
        return cls(pd.read_csv(RESIDUALS_CSV) if RESIDUALS_CSV.exists() else None)

    def pair(self, a: str, b: str) -> tuple[float, int]:
        sa, sb = self.by_player.get(a), self.by_player.get(b)
        if sa is None or sb is None:
            return float("nan"), 0
        common = sa.index.intersection(sb.index)
        if len(common) < 5:
            return float("nan"), int(len(common))
        return float(np.corrcoef(sa.loc[common], sb.loc[common])[0, 1]), int(len(common))


def build_game_correlation(names: list[str], teams: list[str], minutes: list[float],
                           league: dict[str, float], pairs: PairHistory | None = None,
                           shrink: float = PAIR_SHRINK_GAMES) -> np.ndarray:
    n = len(names)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            cat = category(teams[i] == teams[j], role(minutes[i]), role(minutes[j]))
            rho = league.get(cat, 0.0)
            if pairs is not None:
                own, k = pairs.pair(names[i], names[j])
                if k and np.isfinite(own):
                    rho = (k * own + shrink * rho) / (k + shrink)
            R[i, j] = R[j, i] = rho
    return near_spd(R)


def near_spd(A: np.ndarray) -> np.ndarray:
    B = (A + A.T) / 2
    w, V = np.linalg.eigh(B)
    w[w < 1e-8] = 1e-8
    C = (V * w) @ V.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def draw_from_marginals(R: np.ndarray, quantiles: list[np.ndarray | None], means: list[float], sds: list[float],
                        n: int, rng: np.random.Generator, randomness: float = 1.0) -> np.ndarray:
    """(n, players) fantasy points: correlated uniforms through each marginal."""
    L = np.linalg.cholesky(R)
    z = rng.standard_normal((n, R.shape[0])) @ L.T
    u = norm.cdf(z)
    grid = None
    out = np.empty_like(u)
    for j, q in enumerate(quantiles):
        if q is not None:
            q = np.asarray(q, float)
            grid = grid if grid is not None and len(grid) == len(q) else np.linspace(0, 1, len(q))
            x = np.interp(u[:, j], grid, q)
        else:
            x = means[j] + sds[j] * z[:, j]
        if randomness < 1.0:
            x = means[j] + randomness * (x - means[j])
        out[:, j] = np.clip(x, 0.0, None)
    return out
