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
