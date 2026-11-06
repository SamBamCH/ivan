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

slate: the whole slate as one book, in dollars (choose_slate). Every contest
in the entries file is chosen in one pass: the candidates are (contest,
lineup) pairs carrying their net dollars (fee x ROI) in every world, the
worlds are the slate's shared draw so world w is the same game in each
contest, and the greedy adds the pair that most raises sum_w log(1 + book_w /
bankroll) with the bankroll in dollars (weights["bankroll_dollars"]). The
per-contest methods above optimize each contest alone; this one knows that
every entry on the slate lives or dies on the same game.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sqlalchemy.orm import Session

from ivan.core.chooser import (SampledDistances, cash_feature, cash_worlds, cross_fit_book,  # noqa: F401  (shared numerics)
                               fill_stats, fpts_worlds, pick_greedy, quality_features, roi_worlds, shrink_rows, viz_payload)
from ivan.core.portfolio import (DEFAULT_BANKROLL, MIN_BANKROLL, book_summary, dollar_book_summary,  # noqa: F401  (shared picker)
                                 pick_book, pick_portfolio)
from ivan.core.slate_book import DEFAULT_BANKROLL_DOLLARS, ContestCandidates, bankroll_dollars_of, choose_book  # noqa: F401  (the slate as one book)
from ivan.sports.nfl.models import ContestSimReport, LineupSimulationResult, NFLPlayer

METHODS = ("greedy", "portfolio", "slate")
DEFAULT_METHOD = "greedy"
DEFAULT_BANKROLL, MIN_BANKROLL = DEFAULT_BANKROLL, MIN_BANKROLL      # re-exported for callers and tests


def method_of(weights: dict) -> str:
    m = str(weights.get("method") or DEFAULT_METHOD).strip().lower()
    if m not in METHODS:
        raise ValueError(f"unknown chooser method {m!r}; expected one of {METHODS}")
    return m


def bankroll_of(weights: dict) -> float:
    return max(MIN_BANKROLL, float(weights.get("bankroll") or DEFAULT_BANKROLL))


def _candidates(session: Session, contest_name: str, roi_filter: bool):
    """
    The contest's simulated lineups, filtered: optionally (greedy, big pool)
    those with mean ROI >= 0, then those whose players are all still on the
    slate, then one per composition. Returns (sims, n_results, n_after_roi,
    n_valid, missing_player_ids).
    """
    sims = (session.query(LineupSimulationResult)
            .filter(LineupSimulationResult.contest_name == contest_name).all())
    n_results = len(sims)
    if roi_filter and len(sims) > 5000:
        sims = [res for res in sims if np.array(res.roi, dtype=float).mean() >= 0.0]
    n_after_roi = len(sims)
    if not sims:
        return [], n_results, n_after_roi, 0, []
    all_pids = {pid for res in sims for pid in res.lineup.player_ids_list}
    existing = {pid for (pid,) in session.query(NFLPlayer.id).filter(NFLPlayer.id.in_(all_pids)).all()}
    missing_players = sorted(all_pids - existing)
    sims = [res for res in sims if not (set(res.lineup.player_ids_list) - existing)]
    n_valid = len(sims)
    unique, seen = [], set()
    for res in sims:
        comp = tuple(sorted(res.lineup.player_ids_list))
        if comp not in seen:
            seen.add(comp)
            unique.append(res)
    return unique, n_results, n_after_roi, n_valid, missing_players


def _quality(sims, roi_matrix: np.ndarray, weights: dict):
    """Cash % and (shrunk) mean ROI per candidate, each scaled to [0, 1] across the pool, blended by the weights (chooser.quality_features)."""
    return quality_features(np.array([cash_feature(res) for res in sims], dtype=float), roi_matrix, weights)


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
    method = method_of(weights)
    if method == "slate":
        raise ValueError("the slate method chooses every contest at once: use choose_slate (lineup_export.choose_entries does)")
    # greedy, with a big pool: drop lineups that lose money on average. In a large field that keeps only the
    # lineups that won at least one simulated world, so the portfolio method (which wants the steady cashers
    # too, whatever their mean) sees the whole pool.
    sims, n_results, n_after_roi, n_valid, missing_players = _candidates(session, contest_name, roi_filter=method == "greedy")
    N = len(sims)
    if N == 0:
        _fill(stats, n_results, n_after_roi, n_valid, 0, [], [], [], weights, len(missing_players))
        return []

    # every candidate's ROI per simulated world (one sim run: the same worlds), each row pulled toward the pool
    # by its own standard error (empirical Bayes), so the quality and the book never rank on noise
    roi_raw = roi_worlds(sims)
    roi_matrix, shrink = shrink_rows(roi_raw)

    # quality features: cash % and (shrunk) average ROI, normalised to [0, 1]
    raw, q, w_feat, (mins, maxs) = _quality(sims, roi_matrix, weights)

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


# ------------------------------------------------------------------------------ the slate as one book
def shared_worlds(session: Session, contest_names: Sequence[str]) -> dict:
    """
    Check that the contests' latest sim reports share one world axis, so
    world w is the same game in each: the Monte Carlo path must have used the
    slate's shared draw (contest.draw == "slate"), the conditioned path one
    world seed. Raises ValueError when they do not; returns what was shared.
    """
    kinds, draws, seeds, missing = set(), set(), set(), []
    for name in contest_names:
        rep = (session.query(ContestSimReport).filter(ContestSimReport.contest_name == name)
               .order_by(ContestSimReport.created_at.desc()).first())
        r = (rep.report or {}) if rep else None
        if not r:
            missing.append(name)
            continue
        c, sim = r.get("contest") or {}, r.get("simulation") or {}
        kind = str(c.get("method") or sim.get("method") or "montecarlo")
        kinds.add(kind)
        if kind == "conditioned":
            seeds.add((r.get("worlds") or {}).get("seed"))
        else:
            draws.add(str(c.get("draw") or "contest"))
    if missing:
        raise ValueError("no sim report for " + "; ".join(missing) + ": simulate the slate first")
    if len(kinds) > 1:
        raise ValueError(f"the contests were simulated by different methods ({sorted(kinds)}); re-run the sim for the whole slate")
    kind = kinds.pop()
    if kind == "conditioned" and len(seeds) != 1:
        raise ValueError(f"the conditioned sims use {len(seeds)} different world seeds; re-run the sim for the whole slate")
    if kind != "conditioned" and draws != {"slate"}:
        raise ValueError("the Monte Carlo sims did not all use the slate's shared draw (contest.draw: "
                         f"{sorted(draws)}); re-run the sim for the whole slate so world w is the same game in every contest")
    return {"method": kind, "draw": "slate" if kind != "conditioned" else f"seed {seeds.pop()}"}


def _slate_candidates(session: Session, contest: dict) -> ContestCandidates:
    """One contest's simulated lineups as the shared slate book reads them (every candidate: the book wants the steady cashers too)."""
    sims, n_results, n_after_roi, n_valid, missing = _candidates(session, str(contest["contest_name"]), roi_filter=False)
    return ContestCandidates(contest_id=int(contest.get("contest_id") or 0), contest_name=str(contest["contest_name"]), fee=float(contest["fee"]),
                             slots=int(contest["entries"]), lineup_ids=[res.lineup_id for res in sims],
                             roi_raw=roi_worlds(sims) if sims else np.zeros((0, 0), dtype=np.float32),
                             fpts=fpts_worlds(sims) if sims else np.zeros((0, 0)),
                             cash=np.array([cash_feature(res) for res in sims], dtype=float),
                             counts=(n_results, n_after_roi, n_valid, len(sims), len(missing)))


def choose_slate(session: Session, contests: Sequence[dict], weights: dict, log=None) -> dict:
    """
    Choose every contest of the slate as one book in dollars
    (ivan.core.slate_book.choose_book: shrink per contest, stack the
    (contest, lineup) pairs as net dollars per shared world, greedy on the
    slate's log utility, then the exchange pass).

    contests: [{"contest_id", "contest_name", "entries", "fee"}], the entries
    file grouped by contest. weights: bankroll_dollars (the real bankroll),
    cash_pct / avg_roi_win (only for the quality shown on the page).

    The shared world axis is checked first (shared_worlds). Returns {"picks":
    {contest_name: [lineup_id, ...]}, "stats": {contest_name: stats as
    choose_lineup_ids fills them, the viz carrying the slate book and the
    rooting guide}, "book": the slate book}.
    """
    log = log or (lambda *_: None)
    bankroll = bankroll_dollars_of(weights)
    names = [str(c["contest_name"]) for c in contests]
    shared = shared_worlds(session, names)
    cands = [_slate_candidates(session, c) for c in contests]
    out = choose_book(cands, bankroll, weights, shared=shared, log=log)
    picks = dict(zip(names, out["picks"]))
    stats = dict(zip(names, out["stats"]))
    for name, c, st in zip(names, cands, out["stats"]):
        if st.get("viz") is not None and picks[name]:
            sel = [c.lineup_ids.index(lid) for lid in picks[name]]
            roi_m, _ = shrink_rows(c.roi_raw)
            st["viz"]["rooting"] = rooting_guide(session, name, roi_m[sel])
    return {"picks": picks, "stats": stats, "book": out["book"]}


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
    """What the chooser saw, for plotting (chooser.viz_payload on these results' lineup ids)."""
    return viz_payload([res.lineup_id for res in sims], raw, q, dist_mat, fpts_matrix, selected, top_q, trace, lam, weights, norm,
                       book=book, method=method_of(weights), max_points=max_points, seed=seed)


def _fill(stats, n_results, n_after_roi, n_valid, n_unique, top_raw, sel_raw, selected, weights, n_missing, by_lineup=None, book=None):
    fill_stats(stats, n_results, n_after_roi, n_valid, n_unique, top_raw, sel_raw, selected, weights, n_missing, by_lineup=by_lineup, book=book)
