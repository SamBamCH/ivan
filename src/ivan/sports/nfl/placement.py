# src/ivan/sports/nfl/placement.py
"""
Where a lineup finishes in a real contest, and what that pays.

The same convention as the simulator: entries are ranked by points, a tie
group shares the pooled prize of the ranks it spans, and the prize for a
rank comes from the contest's payout table (payouts in the database).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy.orm import Session

from ivan.sports.nfl.models import ContestPayout


def load_payout_tiers(session: Session, contest_id: int) -> List[dict]:
    rows = (session.query(ContestPayout).filter(ContestPayout.contest_id == contest_id)
            .order_by(ContestPayout.min_position.asc()).all())
    return [{"min": int(r.min_position), "max": int(r.max_position), "amount": float(r.payout_amount)} for r in rows]


def payout_vector(tiers: Sequence[dict], n: int) -> np.ndarray:
    """pay[r - 1] = prize at rank r, for ranks 1..n."""
    pay = np.zeros(max(int(n), 0), dtype=np.float64)
    for t in tiers:
        lo, hi = int(t["min"]), min(int(t["max"]), n)
        if lo <= hi:
            pay[lo - 1:hi] = float(t["amount"])
    return pay


def places_paid(tiers: Sequence[dict]) -> int:
    return max((int(t["max"]) for t in tiers if float(t["amount"]) > 0), default=0)


def prize_pool(tiers: Sequence[dict]) -> float:
    return float(sum((int(t["max"]) - int(t["min"]) + 1) * float(t["amount"]) for t in tiers))


def place(our_scores: Sequence[float], field_scores: Sequence[float], pay: np.ndarray,
          atol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank and payout of each of our entries against a field that does not
    contain them. Ties (within atol) share the pooled prize of their ranks.
    Returns (ranks, payouts) aligned to our_scores; a pay vector shorter than
    the combined field pays nothing beyond its end.
    """
    ours = np.asarray(our_scores, dtype=np.float64)
    field = np.asarray(field_scores, dtype=np.float64)
    allv = np.concatenate([ours, field])
    n = allv.size
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    pay_full = np.zeros(n, dtype=np.float64)
    k = min(n, pay.size)
    pay_full[:k] = pay[:k]
    csum = np.concatenate([[0.0], np.cumsum(pay_full)])

    order = np.argsort(-allv, kind="stable")
    sorted_v = allv[order]
    breaks = np.nonzero(np.abs(np.diff(sorted_v)) > atol)[0] + 1
    edges = np.concatenate([[0], breaks, [n]])
    ranks = np.zeros(n, dtype=np.int64)
    pays = np.zeros(n, dtype=np.float64)
    for a, b in zip(edges[:-1], edges[1:]):
        idx = order[a:b]
        ranks[idx] = a + 1
        pays[idx] = (csum[b] - csum[a]) / (b - a)
    m = ours.size
    return ranks[:m], pays[:m]


def remove_our_copies(field_scores: Sequence[float], our_scores: Sequence[float], atol: float = 1e-6) -> Tuple[np.ndarray, int]:
    """
    A scraped field that we played in already holds our entries. Remove one
    field entry per our entry with the same score (if there is one), so that
    place() does not count us twice. Returns (field without us, removed count).
    """
    field = np.asarray(field_scores, dtype=np.float64).copy()
    keep = np.ones(field.size, dtype=bool)
    removed = 0
    for s in np.asarray(our_scores, dtype=np.float64):
        cand = np.nonzero(keep & (np.abs(field - s) <= atol))[0]
        if cand.size:
            keep[cand[0]] = False
            removed += 1
    return field[keep], removed


def rank_payout_from_table(rank: int, tie_size: int, pay: np.ndarray) -> float:
    """Prize for an entry at `rank` that shares its rank with tie_size entries."""
    a, b = rank - 1, min(rank - 1 + max(tie_size, 1), pay.size)
    return float(pay[a:b].sum() / max(tie_size, 1)) if a < pay.size else 0.0
