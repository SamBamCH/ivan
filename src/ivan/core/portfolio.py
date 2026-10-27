# src/ivan/core/portfolio.py
"""
Choose a book of lineups on how it does across simulated worlds, not lineup
by lineup. Shared by the NFL and MMA choosers: each candidate carries its
ROI in every simulated world; the greedy adds the lineup that most raises
sum_w log(1 + net_w / bankroll), the book's log return summed over worlds.
A lineup that wins in worlds the book already wins adds less than one that
wins elsewhere, so the picks spread across outcomes; a lineup that won two
of two thousand worlds no longer outranks one that cashes steadily.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

DEFAULT_BANKROLL = 3.0          # x the entries' fees; the book can lose at most 1x, so anything above 1 keeps the log finite
MIN_BANKROLL = 1.05


def pick_portfolio(roi_matrix: np.ndarray, num_lineups: int, bankroll: float = DEFAULT_BANKROLL,
                   q: Optional[np.ndarray] = None, lineup_ids: Optional[Sequence[int]] = None,
                   chunk: int = 2048) -> tuple[List[int], List[dict]]:
    """
    Greedy on the book's log utility over simulated worlds.

    roi_matrix: candidates x worlds, each entry the lineup's ROI in that world
    (-1 = lost the fee). The book's net in world w, in fee units, is the sum of
    its lineups' ROI there; the utility is sum_w log(1 + net_w / B) with
    B = bankroll x num_lineups (the fees at stake, so the argument stays above
    -1 whenever bankroll > 1). Each step adds the candidate with the largest
    utility gain; the trace records that gain as "div" and the utility reached
    as "score". Ties fall to the lower index, so the pick is deterministic.
    """
    R = np.asarray(roi_matrix, dtype=np.float32)
    N, S = R.shape
    if N == 0 or S == 0 or num_lineups <= 0:
        return [], []
    num_lineups = min(num_lineups, N)
    B = max(MIN_BANKROLL, float(bankroll)) * num_lineups
    ids = list(lineup_ids) if lineup_ids is not None else list(range(N))
    qv = np.asarray(q, dtype=float) if q is not None else np.zeros(N)
    net = np.zeros(S, dtype=np.float64)
    utility = 0.0
    available = np.ones(N, dtype=bool)
    selected: List[int] = []
    trace: List[dict] = []
    gain = np.empty(N, dtype=np.float64)
    for step in range(num_lineups):
        base = net / B
        for a in range(0, N, chunk):                      # chunked so the temporary stays small
            block = R[a:a + chunk].astype(np.float64) / B
            block += base[None, :]
            np.maximum(block, -0.999999, out=block)       # numerical guard; unreachable when bankroll > 1
            gain[a:a + chunk] = np.log1p(block).sum(axis=1)
        gain -= utility
        gain[~available] = -np.inf
        j = int(np.argmax(gain))
        g = float(gain[j])
        utility += g
        net += R[j]
        available[j] = False
        selected.append(j)
        trace.append({"pick": step + 1, "lineup_id": ids[j], "q": float(qv[j]), "div": g, "score": float(utility)})
    return selected, trace


def book_summary(roi_rows: np.ndarray) -> Optional[dict]:
    """How a set of lineups does as one book across the simulated worlds, in ROI on its total fees."""
    roi_rows = np.asarray(roi_rows, dtype=np.float64)
    if roi_rows.ndim != 2 or roi_rows.size == 0:
        return None
    k = roi_rows.shape[0]
    book = roi_rows.sum(axis=0) / k
    p = np.percentile(book, [5, 25, 50, 75, 95])
    return {"entries": int(k), "worlds": int(roi_rows.shape[1]), "mean": float(book.mean()),
            "p5": float(p[0]), "p25": float(p[1]), "p50": float(p[2]), "p75": float(p[3]), "p95": float(p[4]),
            "worst": float(book.min()), "p_loss": float((book < 0).mean()), "p_lose_half": float((book <= -0.5).mean()),
            "share_from_top1pct": float(np.sort(book)[-max(1, int(round(0.01 * book.size))):].sum() / max(1e-9, book[book > 0].sum()))
            if (book > 0).any() else 0.0}


# ------------------------------------------------------------------------------ the slate's book, in dollars
def pick_book(net: np.ndarray, bankroll: float, group: np.ndarray, slots: Sequence[int],
              q: Optional[np.ndarray] = None, ids: Optional[Sequence] = None, chunk: int = 2048) -> tuple[List[int], List[dict], dict]:
    """
    Greedy on the whole book's log utility, in dollars, across groups of slots.

    net: candidates x scenarios, each entry's net dollars in that scenario
    (payout - fee). bankroll: dollars. group[i]: the candidate's group (a
    contest), slots[g]: how many picks group g takes. Each step adds, among the
    candidates of groups with a slot left, the one with the largest gain in
    sum_s log(1 + (book_s + net_s) / bankroll); a candidate is picked at most
    once (one entry cannot carry the same lineup twice in one contest).
    The trace records each pick's gain as "div" and the utility reached as
    "score". Returns (selected indices, trace, diagnostics); a scenario where
    the book would lose more than the bankroll is clipped and counted.
    """
    R = np.asarray(net, dtype=np.float32)
    N, S = R.shape
    group = np.asarray(group, dtype=int)
    slots = [int(k) for k in slots]
    total = min(sum(slots), N)
    if N == 0 or S == 0 or total <= 0 or bankroll <= 0:
        return [], [], {"clipped": 0, "steps": 0}
    B = float(bankroll)
    ids = list(ids) if ids is not None else list(range(N))
    qv = np.asarray(q, dtype=float) if q is not None else np.zeros(N)
    left = list(slots)
    book = np.zeros(S, dtype=np.float64)
    utility = 0.0
    available = np.ones(N, dtype=bool)
    selected: List[int] = []
    trace: List[dict] = []
    clipped = 0
    for step in range(total):
        open_groups = np.array([k > 0 for k in left], dtype=bool)
        cand = np.flatnonzero(available & open_groups[group])
        if cand.size == 0:
            break
        base = book / B
        gain = np.empty(cand.size, dtype=np.float64)
        for a in range(0, cand.size, chunk):
            rows = cand[a:a + chunk]
            block = R[rows].astype(np.float64) / B
            block += base[None, :]
            np.maximum(block, -0.999999, out=block)
            gain[a:a + chunk] = np.log1p(block).sum(axis=1)
        gain -= utility
        j = int(cand[int(np.argmax(gain))])
        g = float(gain[int(np.argmax(gain))])
        utility += g
        book += R[j]
        clipped = int((book <= -B).sum())
        available[j] = False
        left[int(group[j])] -= 1
        selected.append(j)
        trace.append({"pick": step + 1, "lineup_id": ids[j], "group": int(group[j]), "q": float(qv[j]), "div": g, "score": float(utility)})
    return selected, trace, {"clipped": clipped, "steps": len(selected)}


def dollar_book_summary(net_rows: np.ndarray, fees: float, bankroll: float) -> Optional[dict]:
    """
    How a book does as one position across scenarios, in dollars: the
    distribution of its net (payouts minus fees), the same as ROI on its fees,
    and the mean log growth of the bankroll it is played from.
    """
    net_rows = np.asarray(net_rows, dtype=np.float64)
    if net_rows.ndim != 2 or net_rows.size == 0 or fees <= 0:
        return None
    book = net_rows.sum(axis=0)
    p = np.percentile(book, [5, 25, 50, 75, 95])
    growth = np.log1p(np.maximum(book / bankroll, -0.999999)) if bankroll > 0 else np.zeros_like(book)
    return {"entries": int(net_rows.shape[0]), "scenarios": int(net_rows.shape[1]), "fees": float(fees), "bankroll": float(bankroll),
            "mean": float(book.mean()), "p5": float(p[0]), "p25": float(p[1]), "p50": float(p[2]), "p75": float(p[3]), "p95": float(p[4]),
            "worst": float(book.min()), "best": float(book.max()),
            "roi_mean": float(book.mean() / fees), "roi_p50": float(p[2] / fees), "roi_p5": float(p[0] / fees),
            "p_loss": float((book < 0).mean()), "p_lose_half": float((book <= -0.5 * fees).mean()),
            "log_growth": float(growth.mean()), "share_from_top1pct": float(np.sort(book)[-max(1, int(round(0.01 * book.size))):].sum()
                                                                            / max(1e-9, book[book > 0].sum())) if (book > 0).any() else 0.0}
