"""
Salary-anchored volume for stats the books do not price.

Sportsbooks post yardage and reception lines only for players they expect to
have a real role. When a player carries nothing but an anytime-TD line, the
absence of a yards market is itself the signal: the books do not expect the
volume. The old approach filled the gap from game logs of players with a
similar TD rate, which selects lead backs and overprojects backups.

Instead, the expected stat mean for an unpriced player is predicted from two
things the market and DraftKings *do* tell us about him this week:

  * DK salary, which encodes DK's own view of his role, and
  * the consensus anytime-TD probability, which separates a goal-line backup
    from a fullback when both sit at the salary floor.

The prediction is a per-(position, stat) linear model fit on the *same slate's*
market-priced players, whose stat means come straight from their calibrated
lines. Slopes are constrained non-negative so a small sample cannot flip the
sign. When the slate has too few priced players at a position the model falls
back to salary only, and then to constants calibrated on the 2026 week 3 main
slate (DEFAULT_COEFFS).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import nnls

# Players nobody prices carry less volume than priced players at the same
# salary and TD odds: the books post a line when they trust the role. Measured
# 2026-09-17 on the week 3 slate against 2025 per-game actuals, unpriced
# players ran at 0.60-0.85 of the priced curve (RB receiving lowest, TE
# highest) after netting out the priced group's own 0.9 ratio. One factor
# rather than a per-stat table: the per-stat numbers are noisy and partly
# reflect 2025 roles that have since changed.
UNPRICED_SHADE = 0.75

# Minimum priced players needed before trusting a slate fit.
MIN_N_JOINT = 12    # salary + TD probability
MIN_N_SALARY = 6    # salary only

# Calibrated 2026-09-17 on the DK week 3 main slate (51 priced RBs, 76 WRs,
# 33 TEs, 32 QBs for the yardage stats). Values: (intercept, per-$1000-of-
# salary slope, per-unit anytime-TD-probability slope), before UNPRICED_SHADE.
# Used only when the slate cannot support its own fit (small showdown slates).
DEFAULT_COEFFS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("QB", "passing_interceptions"): (0.66, 0.00, 0.00),
    ("QB", "passing_tds"):           (-0.08, 0.28, 0.00),
    ("QB", "passing_yards"):         (124.36, 17.53, 0.00),
    ("QB", "rushing_tds"):           (-0.06, 0.00, 1.36),
    ("QB", "rushing_yards"):         (3.62, 0.00, 92.04),
    ("RB", "receiving_yards"):       (-11.97, 5.92, 0.00),
    ("RB", "receptions"):            (-1.56, 0.73, 0.00),
    ("RB", "rushing_tds"):           (-0.23, 0.05, 1.33),
    ("RB", "rushing_yards"):         (-11.24, 3.65, 108.94),
    ("TE", "receiving_tds"):         (-0.02, 0.00, 1.23),
    ("TE", "receiving_yards"):       (-10.06, 6.78, 94.34),
    ("TE", "receptions"):            (-0.80, 0.73, 6.46),
    ("WR", "receiving_tds"):         (-0.04, 0.01, 1.21),
    ("WR", "receiving_yards"):       (-13.41, 9.72, 52.62),
    ("WR", "receptions"):            (-0.93, 0.80, 2.89),
}


@dataclass
class _Fit:
    coeffs: tuple[float, float, float]   # intercept, per $1k salary, per unit p_td
    n: int
    cap: float                            # upper clamp on predictions
    source: str                           # 'joint' | 'salary' | 'default'


@dataclass
class VolumeModel:
    """Predicts an unpriced stat's mean from salary and anytime-TD probability."""
    obs: dict[tuple[str, str], list[tuple[float, float | None, float]]] = field(default_factory=dict)
    fits: dict[tuple[str, str], _Fit] = field(default_factory=dict)

    def add(self, position: str, stat: str, salary: float, p_td: float | None, mean: float) -> None:
        if salary is None or mean is None or not np.isfinite(mean):
            return
        self.obs.setdefault((position, stat), []).append((float(salary), p_td, float(mean)))

    def fit(self) -> None:
        self.fits = {}
        for key, rows in self.obs.items():
            fit = _fit_one(rows)
            if fit is not None:
                self.fits[key] = fit

    def predict(self, position: str, stat: str, salary: float, p_td: float | None) -> float | None:
        """Expected mean of an *unpriced* stat (UNPRICED_SHADE is applied)."""
        key = (position, stat)
        fit = self.fits.get(key)
        if fit is None:
            coeffs = DEFAULT_COEFFS.get(key)
            if coeffs is None:
                return None
            fit = _Fit(coeffs, 0, float("inf"), "default")
            self.fits[key] = fit
        a, b, c = fit.coeffs
        p = 0.0 if p_td is None else float(p_td)
        if p_td is None and c != 0.0:
            # No TD line: fall back to the salary-only view of the same data.
            a, b = _salary_only(self.obs.get(key, []), fit.coeffs)
            c = 0.0
        pred = (a + b * float(salary) / 1000.0 + c * p) * UNPRICED_SHADE
        return float(np.clip(pred, 0.0, fit.cap))

    def describe(self) -> str:
        lines = []
        for (pos, stat), f in sorted(self.fits.items()):
            a, b, c = f.coeffs
            lines.append(f"{pos:3s} {stat:22s} n={f.n:3d} {f.source:7s} {a:8.2f} + {b:6.2f}/$1k + {c:7.2f}*p_td (cap {f.cap:.1f})")
        return "\n".join(lines)


def _nnls_with_free_intercept(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with slopes >= 0 and an unconstrained intercept.

    Centring both sides removes the intercept from the constrained solve; it
    is recovered afterwards from the means.
    """
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    slopes, _ = nnls(X - x_mean, y - y_mean)
    intercept = y_mean - float(x_mean @ slopes)
    return np.concatenate([[intercept], slopes])


def _fit_one(rows: Iterable[tuple[float, float | None, float]]) -> _Fit | None:
    rows = list(rows)
    sal = np.array([r[0] for r in rows], dtype=float) / 1000.0
    y = np.array([r[2] for r in rows], dtype=float)
    cap = float(np.max(y)) * 1.1 if len(y) else float("inf")

    joint = [(s, p, m) for s, p, m in rows if p is not None and np.isfinite(p)]
    if len(joint) >= MIN_N_JOINT:
        sj = np.array([r[0] for r in joint]) / 1000.0
        pj = np.array([r[1] for r in joint], dtype=float)
        yj = np.array([r[2] for r in joint], dtype=float)
        a, b, c = _nnls_with_free_intercept(np.column_stack([sj, pj]), yj)
        return _Fit((float(a), float(b), float(c)), len(joint), cap, "joint")

    if len(rows) >= MIN_N_SALARY:
        a, b = _nnls_with_free_intercept(sal[:, None], y)
        return _Fit((float(a), float(b), 0.0), len(rows), cap, "salary")

    return None


def _salary_only(rows: list[tuple[float, float | None, float]], fallback: tuple[float, float, float]) -> tuple[float, float]:
    if len(rows) >= MIN_N_SALARY:
        sal = np.array([r[0] for r in rows], dtype=float) / 1000.0
        y = np.array([r[2] for r in rows], dtype=float)
        a, b = _nnls_with_free_intercept(sal[:, None], y)
        return float(a), float(b)
    # Default coefficients carry the TD term's average contribution back into
    # the intercept at a typical probability so the salary-only view is not
    # biased low.
    a, b, c = fallback
    return a + 0.25 * c, b
