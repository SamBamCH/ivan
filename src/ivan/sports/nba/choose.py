# src/ivan/sports/nba/choose.py
"""
Pick lineups for one contest from its simulation results. The numerics are
ivan.core.chooser (shared with NFL); this module reads the NBA rows.

greedy     a greedy blend of quality (cash % and average ROI, weighted) and
           diversity (distance between lineups' simulated point traces).
portfolio  judge the book, not the lineup: the greedy adds the lineup that most
           raises the book's log utility summed over simulated worlds
           (ivan.core.portfolio), so the picks spread across outcomes.
slate      the whole slate as one book in dollars (choose_slate,
           ivan.core.slate_book): every contest of the entries file chosen in
           one pass from one bankroll; the candidates are (contest, lineup)
           pairs carrying fee x ROI per world, the worlds the sim run's one
           shared draw (simulate_contest.slate_draw), so world w is the same
           games in each contest. Needs every contest's sim report to say
           contest.draw == "slate" (one sim run over the whole entries file).

Every candidate's per-world ROI row is first pulled toward the pool by its
own standard error (chooser.shrink_rows), so no method ranks on noise.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sqlalchemy.orm import Session

from ivan.core.chooser import (SampledDistances, cash_feature, cross_fit_book, fill_stats, fpts_worlds, pick_greedy, quality_features,
                               roi_worlds, shrink_rows)
from ivan.core.portfolio import DEFAULT_BANKROLL, MIN_BANKROLL, book_summary, pick_portfolio
from ivan.core.slate_book import DEFAULT_BANKROLL_DOLLARS, ContestCandidates, bankroll_dollars_of, choose_book  # noqa: F401  (the slate as one book)
from ivan.core.utils import coerce_player_ids
from ivan.sports.nba.models import ContestSimReport, LineupSimulationResult, NBAPlayer

METHODS = ("greedy", "portfolio", "slate")
DEFAULT_METHOD = "greedy"
DEFAULT_WEIGHTS = {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1, "method": DEFAULT_METHOD, "bankroll": DEFAULT_BANKROLL,
                   "bankroll_dollars": DEFAULT_BANKROLL_DOLLARS}


def method_of(weights: dict) -> str:
    m = str(weights.get("method") or DEFAULT_METHOD).strip().lower()
    if m not in METHODS:
        raise ValueError(f"unknown chooser method {m!r}; expected one of {METHODS}")
    return m


def bankroll_of(weights: dict) -> float:
    return max(MIN_BANKROLL, float(weights.get("bankroll") or DEFAULT_BANKROLL))


def _candidates(session: Session, contest_id: int, roi_filter: bool):
    """
    The contest's simulated lineups, filtered: optionally (greedy, big pool)
    those with mean ROI >= 0, then those whose players are all still on the
    slate, then one per composition. Returns (sims, n_results, n_after_roi,
    n_valid, missing_player_ids).
    """
    sims = session.query(LineupSimulationResult).filter(LineupSimulationResult.contest_id == int(contest_id)).all()
    n_results = len(sims)
    if roi_filter and len(sims) > 5000:
        sims = [r for r in sims if float(np.mean(np.asarray(r.roi, dtype=float))) >= 0.0]
    n_after_roi = len(sims)
    if not sims:
        return [], n_results, n_after_roi, 0, []
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
    return unique, n_results, n_after_roi, n_valid, missing_players


def choose_lineup_ids(session: Session, weights: dict, num_lineups: int, contest_id: int, stats: Optional[dict] = None) -> List[int]:
    """
    Select `num_lineups` lineup ids for the contest. weights: method, cash_pct,
    avg_roi_win, diversity_lambda, bankroll (see DEFAULT_WEIGHTS). `stats`,
    when a dict, is filled with the selection's summary.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    method = method_of(weights)
    if method == "slate":
        raise ValueError("the slate method chooses every contest at once: use choose_slate (lineup_export.choose_entries does)")
    sims, n_results, n_after_roi, n_valid, missing_players = _candidates(session, contest_id, roi_filter=method == "greedy")
    N = len(sims)
    if N == 0:
        _fill(stats, n_results, n_after_roi, n_valid, 0, [], [], [], weights, len(missing_players))
        return []

    roi_raw = roi_worlds(sims)
    roi_matrix, shrink = shrink_rows(roi_raw)
    raw, q, w_feat, (mins, maxs) = quality_features(np.array([cash_feature(r) for r in sims], dtype=float), roi_matrix, weights)

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
    fill_stats(stats, n_results, n_after_roi, n_valid, n_unique, top_raw, sel_raw, selected, weights, n_missing,
               by_lineup=by_lineup, book=book if book is not None else {}, trace=trace)


# ------------------------------------------------------------------------------ the slate as one book
def shared_worlds(session: Session, contest_ids: Sequence[int]) -> dict:
    """
    Check that the contests' sim reports come from one run's shared draw
    (contest.draw == "slate", simulate_contest.slate_draw) with one seed, so
    world w is the same games in each. Raises ValueError when they do not;
    returns what was shared.
    """
    draws, seeds, missing = set(), set(), []
    for cid in contest_ids:
        rep = session.get(ContestSimReport, int(cid))
        r = (rep.report or {}) if rep else None
        if not r:
            missing.append(str(cid))
            continue
        draws.add(str((r.get("contest") or {}).get("draw") or "contest"))
        seeds.add(r.get("seed"))
    if missing:
        raise ValueError("no sim report for contest " + ", ".join(missing) + ": simulate the slate first")
    if draws != {"slate"}:
        raise ValueError(f"the contests were not all simulated on the run's shared draw (contest.draw: {sorted(draws)}); "
                         "re-run the sim over the whole entries file so world w is the same games in every contest")
    if len(seeds) != 1:
        raise ValueError(f"the contests' sims carry {len(seeds)} different draw seeds; re-run the sim over the whole entries file")
    seed = seeds.pop()
    return {"method": "montecarlo", "draw": "slate" if seed is None else f"slate (seed {seed})"}


def _slate_candidates(session: Session, contest: dict) -> ContestCandidates:
    """One contest's simulated lineups as the shared slate book reads them (every candidate: the book wants the steady cashers too)."""
    sims, n_results, n_after_roi, n_valid, missing = _candidates(session, int(contest["contest_id"]), roi_filter=False)
    return ContestCandidates(contest_id=int(contest["contest_id"]), contest_name=str(contest["contest_name"]), fee=float(contest["fee"]),
                             slots=int(contest["entries"]), lineup_ids=[r.lineup_id for r in sims],
                             roi_raw=roi_worlds(sims) if sims else np.zeros((0, 0), dtype=np.float32),
                             fpts=fpts_worlds(sims) if sims else np.zeros((0, 0)),
                             cash=np.array([cash_feature(r) for r in sims], dtype=float),
                             counts=(n_results, n_after_roi, n_valid, len(sims), len(missing)))


def choose_slate(session: Session, contests: Sequence[dict], weights: dict, log=None) -> dict:
    """
    Choose every contest of the slate as one book in dollars
    (ivan.core.slate_book.choose_book).

    contests: [{"contest_id", "contest_name", "entries", "fee"}], the entries
    file grouped by contest. weights: bankroll_dollars (the real bankroll),
    cash_pct / avg_roi_win (only for the quality shown beside the picks).
    Returns {"picks": {contest_id: [lineup_id, ...]}, "stats": {contest_id:
    stats as choose_lineup_ids fills them}, "book": the slate book}.
    """
    log = log or (lambda *_: None)
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    bankroll = bankroll_dollars_of(weights)
    ids = [int(c["contest_id"]) for c in contests]
    shared = shared_worlds(session, ids)
    cands = [_slate_candidates(session, c) for c in contests]
    out = choose_book(cands, bankroll, weights, shared=shared, log=log)
    return {"picks": dict(zip(ids, out["picks"])), "stats": dict(zip(ids, out["stats"])), "book": out["book"]}


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
