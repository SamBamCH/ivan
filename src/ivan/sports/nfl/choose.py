# src/ivan/sports/nfl/choose.py
"""
Pick lineups for one contest from its simulation results: a greedy blend of
quality (cash % and average ROI, weighted) and diversity (distance between
lineups' simulated point traces).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sqlalchemy.orm import Session

from ivan.sports.nfl.models import LineupSimulationResult, NFLPlayer


def choose_lineup_ids(
    session: Session,
    weights: dict[str, float],
    num_lineups: int,
    contest_name: str,
    stats: Optional[dict] = None,
) -> List[int]:
    """
    Select `num_lineups` lineup ids for `contest_name`.

    weights: cash_pct and avg_roi_win blend the two quality features (each
    normalised to [0, 1] across the candidates); diversity_lambda scales the
    average normalised distance to the lineups already picked.

    If `stats` is a dict it is filled with a summary of the selection
    (candidates, filtering, mean ROI / cash % before and after diversity).
    """
    sims = (session.query(LineupSimulationResult)
            .filter(LineupSimulationResult.contest_name == contest_name).all())
    n_results = len(sims)

    # with a big pool, drop lineups that lose money on average
    if len(sims) > 5000:
        sims = [res for res in sims if np.array(res.roi, dtype=float).mean() >= 0.0]
    n_after_roi = len(sims)
    if not sims:
        _fill(stats, n_results, n_after_roi, 0, 0, [], [], [], weights, 0)
        return []

    # drop lineups whose players are no longer on the slate
    all_pids = {pid for res in sims for pid in res.lineup.player_ids_list}
    existing = {pid for (pid,) in session.query(NFLPlayer.id).filter(NFLPlayer.id.in_(all_pids)).all()}
    missing_players = sorted(all_pids - existing)
    sims = [res for res in sims if not (set(res.lineup.player_ids_list) - existing)]
    n_valid = len(sims)

    # exact duplicates by composition
    unique, seen = [], set()
    for res in sims:
        comp = tuple(sorted(res.lineup.player_ids_list))
        if comp not in seen:
            seen.add(comp)
            unique.append(res)
    sims = unique
    N = len(sims)
    if N == 0:
        _fill(stats, n_results, n_after_roi, n_valid, 0, [], [], [], weights, len(missing_players))
        return []

    # quality features: cash % and average ROI, normalised to [0, 1]
    raw = np.zeros((N, 2), dtype=float)
    for i, res in enumerate(sims):
        payouts = np.array(res.payout, dtype=float)
        rois = np.array(res.roi, dtype=float)
        raw[i, 0] = (payouts > 0).mean() if payouts.size else 0.0
        raw[i, 1] = rois.mean() if rois.size else 0.0
    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    norm = (raw - mins) / (maxs - mins + 1e-6)
    w_feat = np.array([weights.get("cash_pct", 0), weights.get("avg_roi_win", 1)], dtype=float)
    q = (norm * w_feat).sum(axis=1)

    # diversity: distance between simulated fpts traces, normalised to [0, 1]
    fpts_matrix = np.vstack([np.array(res.fpts, dtype=float) for res in sims])
    dist_mat = squareform(pdist(fpts_matrix, metric="euclidean"))
    dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min() + 1e-6)
    lam = float(weights.get("diversity_lambda", 0.5))

    # greedy: best quality first, then quality + lambda * mean distance to the picks so far
    num_lineups = min(num_lineups, N)
    first = int(np.argmax(q))
    selected = [first]
    trace = [{"pick": 1, "lineup_id": sims[first].lineup_id, "q": float(q[first]), "div": 0.0, "score": float(q[first])}]
    remaining = set(range(N)) - set(selected)
    while len(selected) < num_lineups and remaining:
        div = dist_mat[:, selected].mean(axis=1)
        best = max(remaining, key=lambda j: q[j] + lam * div[j])
        selected.append(best)
        remaining.remove(best)
        trace.append({"pick": len(selected), "lineup_id": sims[best].lineup_id, "q": float(q[best]),
                      "div": float(div[best]), "score": float(q[best] + lam * div[best])})

    top_q = [int(i) for i in np.argsort(-q)[:num_lineups]]
    _fill(stats, n_results, n_after_roi, n_valid, N, raw[top_q], raw[selected], selected, weights, len(missing_players),
          by_lineup={sims[i].lineup_id: {"cash_pct": float(raw[i, 0]), "avg_roi": float(raw[i, 1])} for i in selected})
    if stats is not None:
        stats["viz"] = _viz(sims, raw, q, dist_mat, fpts_matrix, selected, top_q, trace, lam, weights, (mins, maxs))
    return [sims[i].lineup_id for i in selected]


def sim_roi_traces(session: Session, contest_name: str, lineup_ids) -> dict[int, np.ndarray]:
    """Per-simulation ROI of each lineup in this contest's results: lineup_id -> array of length num_sims."""
    ids = [int(i) for i in lineup_ids]
    out: dict[int, np.ndarray] = {}
    for i in range(0, len(ids), 900):
        rows = (session.query(LineupSimulationResult.lineup_id, LineupSimulationResult.roi)
                .filter(LineupSimulationResult.contest_name == contest_name,
                        LineupSimulationResult.lineup_id.in_(ids[i:i + 900])).all())
        for lid, roi in rows:
            out[int(lid)] = np.asarray(roi, dtype=np.float32)
    return out


def _viz(sims, raw, q, dist_mat, fpts_matrix, selected, top_q, trace, lam, weights, norm, max_points=1500, seed=0) -> dict:
    """
    What the chooser saw, for plotting: every candidate's cash %, ROI and
    quality; a 2-D map of the simulated-points space (PCA of the fpts traces,
    so distance on the map is roughly the diversity distance); which candidates
    were picked (and in what order) or would have been picked on quality alone;
    the greedy trace; and how spread out each set is. Candidates are subsampled
    to max_points for the page, keeping every pick and quality-only pick.
    """
    N = len(sims)
    Xc = fpts_matrix - fpts_matrix.mean(axis=0, keepdims=True)
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
    points = [{"lineup_id": sims[i].lineup_id, "cash": float(raw[i, 0]), "roi": float(raw[i, 1]), "q": float(q[i]),
               "x": float(coords[i, 0]), "y": float(coords[i, 1]), "pick": pick_order.get(i), "top": i in top_set}
              for i in sorted(keep | set(pool))]

    def mean_pair(idx) -> Optional[float]:
        idx = list(idx)
        if len(idx) < 2:
            return None
        sub = dist_mat[np.ix_(idx, idx)]
        return float(sub[np.triu_indices(len(idx), 1)].mean())

    return {"n_candidates": N, "n_shown": len(points), "weights": dict(weights), "lambda": lam,
            "explained_variance": [float(evr[0]), float(evr[1]) if len(evr) > 1 else 0.0],
            "norm": {"cash": [float(norm[0][0]), float(norm[1][0])], "roi": [float(norm[0][1]), float(norm[1][1])]},
            "points": points, "trace": trace,
            "spread": {"chosen": mean_pair(selected), "top_quality": mean_pair(top_q),
                       "all": float(dist_mat[np.triu_indices(N, 1)].mean()) if N > 1 else None}}


def _fill(stats, n_results, n_after_roi, n_valid, n_unique, top_raw, sel_raw, selected, weights, n_missing, by_lineup=None):
    if stats is None:
        return
    top_raw, sel_raw = np.asarray(top_raw, dtype=float).reshape(-1, 2), np.asarray(sel_raw, dtype=float).reshape(-1, 2)
    stats.update({
        "results": n_results, "after_roi_filter": n_after_roi, "valid": n_valid, "unique": n_unique,
        "missing_players": n_missing, "chosen": len(selected), "weights": dict(weights),
        "by_lineup": by_lineup or {},                      # lineup_id -> {cash_pct, avg_roi} of the picks
        "top_quality_roi": float(top_raw[:, 1].mean()) if len(top_raw) else None,
        "top_quality_cash": float(top_raw[:, 0].mean()) if len(top_raw) else None,
        "chosen_roi": float(sel_raw[:, 1].mean()) if len(sel_raw) else None,
        "chosen_cash": float(sel_raw[:, 0].mean()) if len(sel_raw) else None,
    })
