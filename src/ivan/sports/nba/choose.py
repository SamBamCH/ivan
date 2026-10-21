# src/ivan/sports/nba/choose.py
"""
Pick lineups for one contest from its simulation results. The numerics are
ivan.core.chooser (shared with NFL); this module reads the NBA rows.

greedy     a greedy blend of quality (cash % and average ROI, weighted) and
           diversity (distance between lineups' simulated point traces).
portfolio  judge the book, not the lineup: the greedy adds the lineup that most
           raises the book's log utility summed over simulated worlds
           (ivan.core.portfolio), so the picks spread across outcomes.

Every candidate's per-world ROI row is first pulled toward the pool by its
own standard error (chooser.shrink_rows), so neither method ranks on noise.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sqlalchemy.orm import Session

from ivan.core.chooser import SampledDistances, cash_feature, cross_fit_book, pick_greedy, roi_worlds, shrink_rows
from ivan.core.portfolio import DEFAULT_BANKROLL, MIN_BANKROLL, book_summary, pick_portfolio
from ivan.core.utils import coerce_player_ids
from ivan.sports.nba.models import LineupSimulationResult, NBAPlayer

METHODS = ("greedy", "portfolio")
DEFAULT_METHOD = "greedy"
DEFAULT_WEIGHTS = {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1, "method": DEFAULT_METHOD, "bankroll": DEFAULT_BANKROLL}


def method_of(weights: dict) -> str:
    m = str(weights.get("method") or DEFAULT_METHOD).strip().lower()
    if m not in METHODS:
        raise ValueError(f"unknown chooser method {m!r}; expected one of {METHODS}")
    return m


def bankroll_of(weights: dict) -> float:
    return max(MIN_BANKROLL, float(weights.get("bankroll") or DEFAULT_BANKROLL))


def choose_lineup_ids(session: Session, weights: dict, num_lineups: int, contest_id: int, stats: Optional[dict] = None) -> List[int]:
    """
    Select `num_lineups` lineup ids for the contest. weights: method, cash_pct,
    avg_roi_win, diversity_lambda, bankroll (see DEFAULT_WEIGHTS). `stats`,
    when a dict, is filled with the selection's summary.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    sims = session.query(LineupSimulationResult).filter(LineupSimulationResult.contest_id == int(contest_id)).all()
    n_results = len(sims)
    method = method_of(weights)
    if len(sims) > 5000 and method == "greedy":
        sims = [r for r in sims if float(np.mean(np.asarray(r.roi, dtype=float))) >= 0.0]
    n_after_roi = len(sims)
    if not sims:
        _fill(stats, n_results, n_after_roi, 0, 0, [], [], [], weights, 0)
        return []
    pids_of = {r.lineup_id: coerce_player_ids(r.lineup.player_ids_list) for r in sims}
    all_pids = {p for ids in pids_of.values() for p in ids}
    existing = {pid for (pid,) in session.query(NBAPlayer.id).filter(NBAPlayer.id.in_(list(all_pids))).all()}
    missing_players = sorted(all_pids - existing)
    sims = [r for r in sims if not (set(pids_of[r.lineup_id]) - existing)]
    n_valid = len(sims)
    unique, seen = [], set()
    for r in sims:
        comp = tuple(sorted(pids_of[r.lineup_id]))
        if comp not in seen:
            seen.add(comp); unique.append(r)
    sims = unique
    N = len(sims)
    if N == 0:
        _fill(stats, n_results, n_after_roi, n_valid, 0, [], [], [], weights, len(missing_players))
        return []

    roi_raw = roi_worlds(sims)
    roi_matrix, shrink = shrink_rows(roi_raw)
    raw = np.zeros((N, 2), dtype=float)
    for i, r in enumerate(sims):
        raw[i, 0] = cash_feature(r)
    raw[:, 1] = roi_matrix.mean(axis=1) if roi_matrix.size else 0.0
    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    norm = (raw - mins) / (maxs - mins + 1e-6)
    w_feat = np.array([weights.get("cash_pct", 0), weights.get("avg_roi_win", 1)], dtype=float)
    q = (norm * w_feat).sum(axis=1)

    fpts_matrix = np.vstack([np.asarray(r.fpts, dtype=float) for r in sims])
    lam = float(weights.get("diversity_lambda", 0.5))
    num_lineups = min(int(num_lineups), N)
    bankroll = bankroll_of(weights)
    ids = [r.lineup_id for r in sims]
    if method == "portfolio":
        selected, trace = pick_portfolio(roi_matrix, num_lineups, bankroll, q=q, lineup_ids=ids)
        top_q = [int(i) for i in np.argsort(-q)[:num_lineups]]
        dist_mat = SampledDistances(fpts_matrix, keep=set(selected) | set(top_q))
    else:
        dist_mat = squareform(pdist(fpts_matrix, metric="euclidean"))
        dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min() + 1e-6)
        selected, trace = pick_greedy(q, dist_mat, lam, num_lineups, lineup_ids=ids)
        top_q = [int(i) for i in np.argsort(-q)[:num_lineups]]
    book = {"method": method, "bankroll": bankroll, "shrinkage": shrink,
            "chosen": book_summary(roi_matrix[selected]), "top_quality": book_summary(roi_matrix[top_q]),
            "chosen_unshrunk": book_summary(roi_raw[selected]),
            "cross_fit": cross_fit_book(method, roi_raw, num_lineups, bankroll, lam, dist_mat if method != "portfolio" else None, w_feat)}
    _fill(stats, n_results, n_after_roi, n_valid, N, raw[top_q], raw[selected], selected, weights, len(missing_players),
          by_lineup={sims[i].lineup_id: {"cash_pct": float(raw[i, 0]), "avg_roi": float(raw[i, 1])} for i in selected}, book=book, trace=trace)
    return [sims[i].lineup_id for i in selected]


def _fill(stats, n_results, n_after_roi, n_valid, n_unique, top_raw, sel_raw, selected, weights, n_missing, by_lineup=None, book=None, trace=None):
    if stats is None:
        return
    top_raw = np.asarray(top_raw, dtype=float).reshape(-1, 2); sel_raw = np.asarray(sel_raw, dtype=float).reshape(-1, 2)
    stats.update({"results": int(n_results), "after_roi_filter": int(n_after_roi), "valid": int(n_valid), "unique": int(n_unique),
                  "chosen": int(len(selected)), "weights": dict(weights), "missing_players": int(n_missing),
                  "chosen_roi": float(sel_raw[:, 1].mean()) if len(sel_raw) else None, "chosen_cash": float(sel_raw[:, 0].mean()) if len(sel_raw) else None,
                  "top_quality_roi": float(top_raw[:, 1].mean()) if len(top_raw) else None, "top_quality_cash": float(top_raw[:, 0].mean()) if len(top_raw) else None,
                  "by_lineup": by_lineup or {}, "book": book or {}, "trace": trace or []})


def sim_roi_traces(session: Session, contest_id: int, lineup_ids, column: str = "roi") -> Dict[int, np.ndarray]:
    """lineup_id -> its per-simulation ROI (or another stored column) in this contest's results."""
    out: Dict[int, np.ndarray] = {}
    ids = [int(i) for i in lineup_ids]
    for i in range(0, len(ids), 900):
        rows = (session.query(LineupSimulationResult.lineup_id, getattr(LineupSimulationResult, column))
                .filter(LineupSimulationResult.contest_id == int(contest_id), LineupSimulationResult.lineup_id.in_(ids[i:i + 900])).all())
        for lid, trace in rows:
            if trace:
                out[int(lid)] = np.asarray(trace, dtype=np.float32)
    return out
