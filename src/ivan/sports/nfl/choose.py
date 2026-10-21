# src/ivan/sports/nfl/choose.py
"""
Pick lineups for one contest from its simulation results. Each candidate's
row is its expected ROI in every world (ivan.sports.nfl.conditioned: the
world is a draw of the game latents, the cell the lineup's expected payout
given that world, integrated over its own residual and the field's), so a
row mean is an expectation with only between-world noise and the cross-fit
gap should be near zero. Two methods:

greedy (the original): a greedy blend of quality (cash % and average ROI,
weighted) and diversity (distance between lineups' simulated point traces).
Each lineup is judged on its own; the portfolio has no say.

portfolio: judge the book, not the lineup. Every candidate carries its ROI in
each simulated world; the greedy adds the lineup that most raises the
portfolio's log utility summed over worlds, sum_w log(1 + net_w / bankroll),
where net_w is the book's net return in world w (in entry-fee units) and the
bankroll is `bankroll` x the entries' fees. A second lineup that wins in the
same worlds as the first adds less than the first did, so the method spreads
the book across worlds instead of stacking one outcome; a lineup that wins two
of two thousand worlds no longer outranks one that cashes steadily.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sqlalchemy.orm import Session

from ivan.core.chooser import (SampledDistances, cash_feature, cash_worlds, cross_fit_book,  # noqa: F401  (shared numerics)
                               pick_greedy, roi_worlds, shrink_rows)
from ivan.core.portfolio import DEFAULT_BANKROLL, MIN_BANKROLL, book_summary, pick_portfolio  # noqa: F401  (shared picker)
from ivan.sports.nfl.models import ContestSimReport, LineupSimulationResult, NFLPlayer

METHODS = ("greedy", "portfolio")
DEFAULT_METHOD = "greedy"
DEFAULT_BANKROLL, MIN_BANKROLL = DEFAULT_BANKROLL, MIN_BANKROLL      # re-exported for callers and tests


def method_of(weights: dict) -> str:
    m = str(weights.get("method") or DEFAULT_METHOD).strip().lower()
    if m not in METHODS:
        raise ValueError(f"unknown chooser method {m!r}; expected one of {METHODS}")
    return m


def bankroll_of(weights: dict) -> float:
    return max(MIN_BANKROLL, float(weights.get("bankroll") or DEFAULT_BANKROLL))


def choose_lineup_ids(
    session: Session,
    weights: dict,
    num_lineups: int,
    contest_name: str,
    stats: Optional[dict] = None,
) -> List[int]:
    """
    Select `num_lineups` lineup ids for `contest_name`.

    weights: method picks the selection ("greedy" or "portfolio"). cash_pct and
    avg_roi_win blend the two quality features (each normalised to [0, 1]
    across the candidates); diversity_lambda scales the average normalised
    distance to the lineups already picked (greedy); bankroll is the log
    utility's scale in multiples of the entries' fees (portfolio).

    If `stats` is a dict it is filled with a summary of the selection
    (candidates, filtering, mean ROI / cash % before and after diversity, and
    the book's simulated return distribution).
    """
    sims = (session.query(LineupSimulationResult)
            .filter(LineupSimulationResult.contest_name == contest_name).all())
    n_results = len(sims)

    # greedy, with a big pool: drop lineups that lose money on average. In a large field that keeps only the
    # lineups that won at least one simulated world, so the portfolio method (which wants the steady cashers
    # too, whatever their mean) sees the whole pool.
    method = method_of(weights)
    if len(sims) > 5000 and method == "greedy":
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

    # every candidate's ROI per simulated world (one sim run: the same worlds), each row pulled toward the pool
    # by its own standard error (empirical Bayes), so the quality and the book never rank on noise
    roi_raw = roi_worlds(sims)
    roi_matrix, shrink = shrink_rows(roi_raw)

    # quality features: cash % and (shrunk) average ROI, normalised to [0, 1]
    raw = np.zeros((N, 2), dtype=float)
    for i, res in enumerate(sims):
        raw[i, 0] = cash_feature(res)
    raw[:, 1] = roi_matrix.mean(axis=1) if roi_matrix.size else 0.0
    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    norm = (raw - mins) / (maxs - mins + 1e-6)
    w_feat = np.array([weights.get("cash_pct", 0), weights.get("avg_roi_win", 1)], dtype=float)
    q = (norm * w_feat).sum(axis=1)

    # diversity: distance between simulated fpts traces, normalised to [0, 1]. The greedy needs every pair;
    # the portfolio method only reports distances, so it measures them on the picks and a sample (the full
    # matrix of a 10k pool is a gigabyte).
    fpts_matrix = np.vstack([np.array(res.fpts, dtype=float) for res in sims])
    lam = float(weights.get("diversity_lambda", 0.5))
    num_lineups = min(num_lineups, N)
    bankroll = bankroll_of(weights)
    ids = [res.lineup_id for res in sims]
    cash_matrix = cash_worlds(sims)
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
            "cross_fit": cross_fit_book(method, roi_raw, num_lineups, bankroll, lam, dist_mat if method != "portfolio" else None, w_feat,
                                        cash_matrix=cash_matrix)}
    rooting = rooting_guide(session, contest_name, roi_matrix[selected])
    _fill(stats, n_results, n_after_roi, n_valid, N, raw[top_q], raw[selected], selected, weights, len(missing_players),
          by_lineup={sims[i].lineup_id: {"cash_pct": float(raw[i, 0]), "avg_roi": float(raw[i, 1])} for i in selected}, book=book)
    if stats is not None:
        stats["viz"] = _viz(sims, raw, q, dist_mat, fpts_matrix, selected, top_q, trace, lam, weights, (mins, maxs), book)
        stats["viz"]["rooting"] = rooting
    return [sims[i].lineup_id for i in selected]


# the numerics (pick_greedy, SampledDistances, shrink_rows, cash_feature, cash_worlds, cross_fit_book, roi_worlds)
# live in ivan.core.chooser and are shared with the other sports; imported above.


def sim_roi_traces(session: Session, contest_name: str, lineup_ids, column: str = "roi") -> dict[int, np.ndarray]:
    """
    Per-world ROI of each lineup in this contest's results: lineup_id -> array
    of length num_worlds. column="roi" is the expectation per world (what the
    chooser ranks on); column="roi_draw" one realized outcome per world (what
    the luck and payout-ladder tests need). Lineups without the column are left out.
    """
    ids = [int(i) for i in lineup_ids]
    col = getattr(LineupSimulationResult, column)
    out: dict[int, np.ndarray] = {}
    for i in range(0, len(ids), 900):
        rows = (session.query(LineupSimulationResult.lineup_id, col)
                .filter(LineupSimulationResult.contest_name == contest_name,
                        LineupSimulationResult.lineup_id.in_(ids[i:i + 900])).all())
        for lid, roi in rows:
            if roi:
                out[int(lid)] = np.asarray(roi, dtype=np.float32)
    return out


# ------------------------------------------------------------------------------ what to root for
POINT_BINS = ((None, 13), (14, 20), (21, 27), (28, 34), (35, None))


def rooting_guide(session: Session, contest_name: str, book_rows: np.ndarray) -> Optional[dict]:
    """
    The book's expected ROI by game state, from the worlds the sim stored with
    its report (each world is a draw of every team's points). For each team:
    the book's mean ROI in the worlds where that team scores in each bin, the
    correlation of the book's ROI with the team's points, and the team's mean
    points in the book's best tenth of worlds against all worlds. Read it as
    "what to root for": a team with a strongly positive lean is one the book
    needs to score.
    """
    rep = (session.query(ContestSimReport).filter(ContestSimReport.contest_name == contest_name)
           .order_by(ContestSimReport.created_at.desc()).first())
    worlds = (rep.report or {}).get("worlds") if rep else None
    if not worlds or book_rows.size == 0:
        return None
    pts = np.asarray(worlds.get("team_points") or [], dtype=float)
    teams = list(worlds.get("teams") or [])
    if pts.ndim != 2 or pts.shape[1] != len(teams):
        return None
    S = min(pts.shape[0], book_rows.shape[1])
    if S < 20:
        return None
    book = book_rows[:, :S].astype(float).mean(axis=0)              # book ROI per world
    pts = pts[:S]
    best = book >= np.percentile(book, 90)
    out = {"worlds": int(S), "bins": [f"<= {hi}" if lo is None else (f"{lo}+" if hi is None else f"{lo}-{hi}") for lo, hi in POINT_BINS],
           "teams": []}
    for j, team in enumerate(teams):
        p = pts[:, j]
        if not np.isfinite(p).any():
            continue
        rows = []
        for lo, hi in POINT_BINS:
            k = np.ones(S, dtype=bool)
            if lo is not None:
                k &= p >= lo
            if hi is not None:
                k &= p <= hi
            rows.append({"n": int(k.sum()), "roi": float(book[k].mean()) if k.any() else None})
        sd = p.std()
        corr = float(np.corrcoef(p, book)[0, 1]) if sd > 0 and book.std() > 0 else 0.0
        out["teams"].append({"team": team, "mean": float(p.mean()), "best_mean": float(p[best].mean()), "corr": corr,
                             "lean": "root for" if corr > 0.1 else ("root against" if corr < -0.1 else "neutral"), "bins": rows})
    return out


def _viz(sims, raw, q, dist_mat, fpts_matrix, selected, top_q, trace, lam, weights, norm, book=None, max_points=1500, seed=0) -> dict:
    """
    What the chooser saw, for plotting: every candidate's cash %, ROI and
    quality; a 2-D map of the simulated-points space (PCA of the fpts traces,
    so distance on the map is roughly the diversity distance); which candidates
    were picked (and in what order) or would have been picked on quality alone;
    the greedy trace; how spread out each set is; and the book's simulated
    return distribution. Candidates are subsampled to max_points for the page,
    keeping every pick and quality-only pick.
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
            "method": (book or {}).get("method", method_of(weights)), "bankroll": (book or {}).get("bankroll"),
            "explained_variance": [float(evr[0]), float(evr[1]) if len(evr) > 1 else 0.0],
            "norm": {"cash": [float(norm[0][0]), float(norm[1][0])], "roi": [float(norm[0][1]), float(norm[1][1])]},
            "points": points, "trace": trace, "book": book,
            "spread": {"chosen": mean_pair(selected), "top_quality": mean_pair(top_q), "all": mean_all}}


def _fill(stats, n_results, n_after_roi, n_valid, n_unique, top_raw, sel_raw, selected, weights, n_missing, by_lineup=None, book=None):
    if stats is None:
        return
    top_raw, sel_raw = np.asarray(top_raw, dtype=float).reshape(-1, 2), np.asarray(sel_raw, dtype=float).reshape(-1, 2)
    stats.update({
        "results": n_results, "after_roi_filter": n_after_roi, "valid": n_valid, "unique": n_unique,
        "missing_players": n_missing, "chosen": len(selected), "weights": dict(weights),
        "by_lineup": by_lineup or {},                      # lineup_id -> {cash_pct, avg_roi} of the picks
        "book": book,                                      # method, bankroll, chosen / top_quality return distributions
        "top_quality_roi": float(top_raw[:, 1].mean()) if len(top_raw) else None,
        "top_quality_cash": float(top_raw[:, 0].mean()) if len(top_raw) else None,
        "chosen_roi": float(sel_raw[:, 1].mean()) if len(sel_raw) else None,
        "chosen_cash": float(sel_raw[:, 0].mean()) if len(sel_raw) else None,
    })
