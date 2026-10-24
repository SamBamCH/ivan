# src/ivan/sports/nba/projection/solve.py
"""
One team's minutes: the allocation closest to every anchor, that sums to the budget and respects the position groups.

Each available player brings an anchor (mean, sd): the books' implied minutes for priced players (sd ~4), history for
the rest (wider), a desk edit (very tight). Each position group (big / guard / wing) brings a target (the team's trailing
minutes in that group, sd its game-to-game spread). The solve minimises

    sum_i ((x_i - m_i) / s_i)^2  +  sum_g ((sum_{i in g} x_i - T_g) / sd_g)^2

subject to sum_i x_i = budget and 0 <= x_i <= cap_i. It is a small convex quadratic program (one team, <= 18 players):
solved with SLSQP in a few milliseconds. An absent player simply has no row: his group's target still stands, so his
minutes flow to his group first (the position tilt measured 2026-10-23: bigs' minutes to bigs, guards ~4:1), and then to
everyone as the budget demands.

Deep-bench players enter with their expected minutes (play probability x minutes when they play) and a wide sd; they are
never removed: the deep bench's expectation is honest (measured 2026-10-23), only its shape is bimodal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

GROUPS = ("big", "guard", "wing")
BUDGET = 241.0
CAP = 42.0
EDIT_SD = 0.05          # a hand-set number: held almost exactly
BOOK_SD = 4.0           # the books' implied minutes miss by ~4 (2026-10-23)


@dataclass
class Anchor:
    name: str
    mean: float
    sd: float
    group: str                      # big | guard | wing
    cap: float = CAP
    source: str = "history"         # book | history | edit | deep
    floor: float = 0.0


@dataclass
class Solution:
    minutes: Dict[str, float]
    group_minutes: Dict[str, float]
    group_target: Dict[str, float]
    budget: float
    ok: bool
    message: str = ""
    pulls: Dict[str, float] = field(default_factory=dict)     # how far each player moved from his anchor, in his own sd units


def group_of(positions) -> str:
    """DK eligibility -> group: a centre-eligible player is a big, a point-guard-eligible one a guard, the rest wings."""
    e = {str(p).strip().upper() for p in (positions if isinstance(positions, (list, tuple, set)) else str(positions or "").split("/"))}
    return "big" if "C" in e else ("guard" if "PG" in e else "wing")


def solve(anchors: List[Anchor], targets: Optional[Dict[str, tuple]] = None, budget: float = BUDGET) -> Solution:
    """
    anchors: the available players. targets: group -> (target minutes, sd); a group without a target is unconstrained.
    Returns the solved minutes (never below each anchor's floor, never above its cap).
    """
    from scipy.optimize import minimize
    n = len(anchors)
    if n == 0:
        return Solution({}, {}, {}, budget, False, "no players")
    m = np.array([a.mean for a in anchors]); s = np.array([max(a.sd, 1e-3) for a in anchors])
    lo = np.array([max(0.0, a.floor) for a in anchors]); hi = np.array([max(a.cap, lo[i]) for i, a in enumerate(anchors)])
    targets = targets or {}
    G = {g: np.array([a.group == g for a in anchors], dtype=float) for g in GROUPS if g in targets and any(a.group == g for a in anchors)}
    T = {g: (float(targets[g][0]), max(float(targets[g][1]), 1e-3)) for g in G}
    if hi.sum() < budget - 1e-6:
        budget = float(hi.sum())                       # not enough bodies for the budget: everyone at the cap
    if lo.sum() > budget + 1e-6:
        budget = float(lo.sum())

    def f(x):
        v = float((((x - m) / s) ** 2).sum())
        for g, mask in G.items():
            t, sd = T[g]; v += ((mask @ x - t) / sd) ** 2
        return v

    def grad(x):
        gr = 2 * (x - m) / s ** 2
        for g, mask in G.items():
            t, sd = T[g]; gr = gr + 2 * (mask @ x - t) / sd ** 2 * mask
        return gr

    x0 = np.clip(m, lo, hi)
    if x0.sum() > 0:
        x0 = np.clip(x0 * budget / x0.sum(), lo, hi)
    res = minimize(f, x0, jac=grad, method="SLSQP", bounds=list(zip(lo, hi)),
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - budget, "jac": lambda x: np.ones(n)}],
                   options={"maxiter": 300, "ftol": 1e-10})
    x = np.clip(res.x, lo, hi)
    if abs(x.sum() - budget) > 0.05:                   # SLSQP left a tiny residual: spread it over the un-clamped players
        free = (x > lo + 1e-6) & (x < hi - 1e-6)
        if free.any():
            x[free] += (budget - x.sum()) / free.sum()
    mins = {a.name: float(v) for a, v in zip(anchors, x)}
    gm = {g: float(sum(v for a, v in zip(anchors, x) if a.group == g)) for g in GROUPS}
    return Solution(mins, gm, {g: T[g][0] for g in G}, budget, bool(res.success or abs(x.sum() - budget) < 0.1), str(res.message),
                    pulls={a.name: float((v - a.mean) / a.sd) for a, v in zip(anchors, x)})
