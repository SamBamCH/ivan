# src/ivan/core/chooser.py
"""
The lineup choosers' numerics, shared by the sports: the greedy quality +
diversity pick, the empirical-Bayes shrinkage of every candidate's per-world
ROI row, the cash features, the cross-fit (honest) book expectation and the
sampled pairwise distances the portfolio method reports. The portfolio pick
itself is ivan.core.portfolio. Each sport's choose module reads its own
simulation rows and calls these.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ivan.core.portfolio import book_summary, pick_portfolio


# ------------------------------------------------------------------------------ the two selections
def pick_greedy(q: np.ndarray, dist_mat: np.ndarray, lam: float, num_lineups: int,
                lineup_ids: Optional[Sequence[int]] = None) -> tuple[List[int], List[dict]]:
    """Best quality first, then repeatedly quality + lambda x mean distance to the picks so far."""
    ids = list(lineup_ids) if lineup_ids is not None else list(range(len(q)))
    first = int(np.argmax(q))
    selected = [first]
    trace = [{"pick": 1, "lineup_id": ids[first], "q": float(q[first]), "div": 0.0, "score": float(q[first])}]
    remaining = set(range(len(q))) - set(selected)
    while len(selected) < num_lineups and remaining:
        div = dist_mat[:, selected].mean(axis=1)
        best = max(remaining, key=lambda j: q[j] + lam * div[j])
        selected.append(best)
        remaining.remove(best)
        trace.append({"pick": len(selected), "lineup_id": ids[best], "q": float(q[best]),
                      "div": float(div[best]), "score": float(q[best] + lam * div[best])})
    return selected, trace


class SampledDistances:
    """
    Pairwise fpts-trace distances for a subset of candidates (the picks and the
    quality-only picks) plus a random sample for the pool-wide mean, scaled to
    [0, 1] the way the greedy's full matrix is; indexed like that matrix.
    """

    def __init__(self, fpts_matrix: np.ndarray, keep: set, sample: int = 1500, seed: int = 0):
        N = len(fpts_matrix)
        rng = np.random.default_rng(seed)
        extra = [int(i) for i in rng.choice(N, size=min(N, sample), replace=False)]
        self.rows = sorted(set(keep) | set(extra))
        self.pos = {i: k for k, i in enumerate(self.rows)}
        d = squareform(pdist(fpts_matrix[self.rows], metric="euclidean")) if len(self.rows) > 1 else np.zeros((len(self.rows), len(self.rows)))
        self.d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        self.sample = [self.pos[i] for i in extra]

    def mean_pair(self, idx) -> Optional[float]:
        rows = [self.pos[i] for i in idx if i in self.pos]
        if len(rows) < 2:
            return None
        sub = self.d[np.ix_(rows, rows)]
        return float(sub[np.triu_indices(len(rows), 1)].mean())

    def mean_all(self) -> Optional[float]:
        rows = self.sample
        if len(rows) < 2:
            return None
        sub = self.d[np.ix_(rows, rows)]
        return float(sub[np.triu_indices(len(rows), 1)].mean())


def shrink_rows(R: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Empirical-Bayes shrinkage of every candidate's per-world ROI row toward
    the pool. A row's mean is an estimate with standard error se_c = sd_c /
    sqrt(S); the spread of true means across the pool, tau^2, is estimated
    the random-effects way (DerSimonian-Laird). Each row's deviation from the
    pool's row is then scaled by tau^2 / (tau^2 + se_c^2): a lineup estimated
    at +1000% +- 190% is pulled far more than one at +240% +- 40%, which is
    what stops a chooser from buying the noisiest draws. The world structure
    is kept (the deviation is scaled, not replaced), so the portfolio method
    still sees which worlds a lineup wins. Returns (shrunk rows, summary).
    """
    R = np.asarray(R, dtype=np.float64)
    N, S = R.shape
    if N < 2 or S < 2:
        return R.astype(np.float32), {"tau": None, "lambda_median": 1.0, "se_median": None, "pool_mean": float(R.mean()) if R.size else None}
    mu = R.mean(axis=1)
    se2 = np.maximum(R.var(axis=1, ddof=1) / S, 1e-12)
    # tau^2 by DerSimonian-Laird: precision-weighted, so a few very noisy rows cannot inflate the spread of true means
    w = 1.0 / se2
    mu_w = float((w * mu).sum() / w.sum())
    Q = float((w * (mu - mu_w) ** 2).sum())
    tau2 = max((Q - (N - 1)) / (w.sum() - (w ** 2).sum() / w.sum()), 0.0)
    lam = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros(N)
    pool = R.mean(axis=0, keepdims=True)
    out = pool + lam[:, None] * (R - pool)
    return out.astype(np.float32), {"tau": float(np.sqrt(tau2)), "se_median": float(np.sqrt(np.median(se2))), "lambda_median": float(np.median(lam)),
                                    "lambda_top_decile": float(np.median(lam[mu >= np.percentile(mu, 90)])), "pool_mean": float(mu.mean()),
                                    "top_mean_before": float(np.sort(mu)[-20:].mean()), "top_mean_after": float(np.sort(out.mean(axis=1))[-20:].mean())}


def cash_feature(res) -> float:
    """A candidate's cash probability: the mean of its per-world P(cash) when the sim stored it, else the share of worlds paid."""
    cash = getattr(res, "cash", None)
    if cash:
        return float(np.mean(np.asarray(cash, dtype=float)))
    payouts = np.asarray(res.payout, dtype=float)
    return float((payouts > 0).mean()) if payouts.size else 0.0


def cash_worlds(sims) -> Optional[np.ndarray]:
    """Candidates x worlds P(cash) matrix when every result carries one, else None."""
    if not sims or any(not getattr(res, "cash", None) for res in sims):
        return None
    traces = [np.asarray(res.cash, dtype=np.float32) for res in sims]
    S = min(t.size for t in traces)
    return np.vstack([t[:S] for t in traces]) if S else None


def cross_fit_book(method: str, roi_matrix: np.ndarray, num_lineups: int, bankroll: float, lam: float,
                   dist_mat: Optional[np.ndarray], w_feat: np.ndarray, cash_matrix: Optional[np.ndarray] = None) -> Optional[dict]:
    """
    The book's honest expectation: select on the even worlds, score on the odd
    ones. A chooser that ranks candidates by their sampled ROI picks the ones
    whose rare wins happened to land in the sample, so the ROI it reports for
    its picks (in-sample) overstates what the same sim expects of them on fresh
    worlds (out-of-sample); the difference is selection bias, a property of the
    estimator, not of the sim. Validation compares the out-of-sample number
    with the realized ROI.
    """
    N, S = roi_matrix.shape
    if N == 0 or S < 20:
        return None
    fit = np.arange(S) % 2 == 0
    RA, RB = roi_matrix[:, fit], roi_matrix[:, ~fit]
    RA, _ = shrink_rows(RA)                             # the selection sees shrunk rows, as the real chooser does
    k = min(num_lineups, N)
    if method == "portfolio":
        sel, _ = pick_portfolio(RA, k, bankroll)
    else:
        cash_fit = cash_matrix[:, fit[:cash_matrix.shape[1]]].mean(axis=1) if cash_matrix is not None and cash_matrix.shape[1] >= S else (RA > -1).mean(axis=1)
        raw = np.column_stack([cash_fit, RA.mean(axis=1)])
        norm = (raw - raw.min(axis=0)) / (raw.max(axis=0) - raw.min(axis=0) + 1e-6)
        sel, _ = pick_greedy((norm * w_feat).sum(axis=1), dist_mat, lam, k)
    sel = [int(i) for i in sel]
    ins, oos = book_summary(RA[sel]), book_summary(RB[sel])
    if ins is None or oos is None:
        return None
    return {"worlds_fit": int(fit.sum()), "worlds_score": int((~fit).sum()), "in_sample": ins, "out_of_sample": oos,
            "selection_bias": float(ins["mean"] - oos["mean"])}


def roi_worlds(sims) -> np.ndarray:
    """Candidates x worlds ROI matrix from simulation results, truncated to the shortest trace."""
    traces = [np.asarray(res.roi, dtype=np.float32) for res in sims]
    S = min((t.size for t in traces), default=0)
    if S == 0:
        return np.zeros((len(traces), 0), dtype=np.float32)
    return np.vstack([t[:S] for t in traces])


def fpts_worlds(sims) -> np.ndarray:
    """Candidates x worlds simulated-points matrix from simulation results, truncated to the shortest trace."""
    traces = [np.asarray(res.fpts, dtype=np.float64) for res in sims]
    S = min((t.size for t in traces), default=0)
    if S == 0:
        return np.zeros((len(traces), 0), dtype=np.float64)
    return np.vstack([t[:S] for t in traces])


# ------------------------------------------------------------------------------ what every chooser reports
def quality_features(cash: np.ndarray, roi_matrix: np.ndarray, weights: dict):
    """
    Cash % and (shrunk) mean ROI per candidate, each scaled to [0, 1] across
    the pool and blended by the weights (cash_pct, avg_roi_win). Returns (raw
    N x 2, quality N, the two weights, (mins, maxs) of the scaling).
    """
    cash = np.asarray(cash, dtype=float).reshape(-1)
    N = cash.size
    raw = np.zeros((N, 2), dtype=float)
    raw[:, 0] = cash
    raw[:, 1] = roi_matrix.mean(axis=1) if roi_matrix.size else 0.0
    mins, maxs = (raw.min(axis=0), raw.max(axis=0)) if N else (np.zeros(2), np.zeros(2))
    norm = (raw - mins) / (maxs - mins + 1e-6)
    w_feat = np.array([weights.get("cash_pct", 0), weights.get("avg_roi_win", 1)], dtype=float)
    return raw, (norm * w_feat).sum(axis=1), w_feat, (mins, maxs)


def viz_payload(lineup_ids: Sequence[int], raw, q, dist_mat, fpts_matrix, selected, top_q, trace, lam, weights, norm,
                book=None, method: Optional[str] = None, max_points: int = 1500, seed: int = 0) -> dict:
    """
    What the chooser saw, for plotting: every candidate's cash %, ROI and
    quality; a 2-D map of the simulated-points space (PCA of the fpts traces,
    so distance on the map is roughly the diversity distance); which candidates
    were picked (and in what order) or would have been picked on quality alone;
    the greedy trace; how spread out each set is; and the book's simulated
    return distribution. Candidates are subsampled to max_points for the page,
    keeping every pick and quality-only pick.
    """
    ids = list(lineup_ids)
    N = len(ids)
    fpts_matrix = np.asarray(fpts_matrix, dtype=float)
    Xc = fpts_matrix - fpts_matrix.mean(axis=0, keepdims=True) if fpts_matrix.size else np.zeros((N, 1))
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    coords = U[:, :2] * S[:2] if S.size >= 2 else np.zeros((N, 2))
    var = S ** 2
    evr = (var / var.sum()).tolist() if var.sum() > 0 else [0.0, 0.0]

    pick_order = {idx: k + 1 for k, idx in enumerate(selected)}
    top_set = set(top_q)
    keep = set(selected) | top_set
    pool = [i for i in range(N) if i not in keep]
    if len(keep) + len(pool) > max_points:
        pool = [int(i) for i in np.random.default_rng(seed).choice(pool, size=max(0, max_points - len(keep)), replace=False)]
    points = [{"lineup_id": ids[i], "cash": float(raw[i, 0]), "roi": float(raw[i, 1]), "q": float(q[i]),
               "x": float(coords[i, 0]), "y": float(coords[i, 1]), "pick": pick_order.get(i), "top": i in top_set}
              for i in sorted(keep | set(pool))]

    if isinstance(dist_mat, SampledDistances):
        mean_pair, mean_all = dist_mat.mean_pair, dist_mat.mean_all()
    else:
        def mean_pair(idx) -> Optional[float]:
            idx = list(idx)
            if len(idx) < 2:
                return None
            sub = dist_mat[np.ix_(idx, idx)]
            return float(sub[np.triu_indices(len(idx), 1)].mean())
        mean_all = float(dist_mat[np.triu_indices(N, 1)].mean()) if N > 1 else None

    return {"n_candidates": N, "n_shown": len(points), "weights": dict(weights), "lambda": lam,
            "method": (book or {}).get("method") or method or str(weights.get("method") or "greedy"), "bankroll": (book or {}).get("bankroll"),
            "explained_variance": [float(evr[0]), float(evr[1]) if len(evr) > 1 else 0.0],
            "norm": {"cash": [float(norm[0][0]), float(norm[1][0])], "roi": [float(norm[0][1]), float(norm[1][1])]},
            "points": points, "trace": trace, "book": book,
            "spread": {"chosen": mean_pair(selected), "top_quality": mean_pair(top_q), "all": mean_all}}


def fill_stats(stats, n_results, n_after_roi, n_valid, n_unique, top_raw, sel_raw, selected, weights, n_missing,
               by_lineup=None, book=None, trace=None) -> None:
    """The selection's summary every choose run logs and writes: counts, the picks' and the quality-only picks' mean ROI / cash, the book."""
    if stats is None:
        return
    top_raw, sel_raw = np.asarray(top_raw, dtype=float).reshape(-1, 2), np.asarray(sel_raw, dtype=float).reshape(-1, 2)
    stats.update({
        "results": int(n_results), "after_roi_filter": int(n_after_roi), "valid": int(n_valid), "unique": int(n_unique),
        "missing_players": int(n_missing), "chosen": int(len(selected)), "weights": dict(weights),
        "by_lineup": by_lineup or {},                      # lineup_id -> {cash_pct, avg_roi} of the picks
        "book": book,                                      # method, bankroll, chosen / top_quality return distributions
        "trace": trace or [],
        "top_quality_roi": float(top_raw[:, 1].mean()) if len(top_raw) else None,
        "top_quality_cash": float(top_raw[:, 0].mean()) if len(top_raw) else None,
        "chosen_roi": float(sel_raw[:, 1].mean()) if len(sel_raw) else None,
        "chosen_cash": float(sel_raw[:, 0].mean()) if len(sel_raw) else None,
    })
