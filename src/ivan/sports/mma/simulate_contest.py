# src/ivan/sports/mma/simulate_contest.py
"""
Simulate one contest: sample the field from the candidate pool, run every
fight n_sims times, score and rank every entry (tied entries share the
pooled prize of their rank range, as the NFL simulator does), and store one
result row per unique lineup in the field.

The contest is either a DraftKings contest in the database (contest id,
after get_contest_ids.py / get_payouts.py) or a hand-described one: a
payouts CSV plus the real entry count and fee. The real field is far bigger
than the simulated one, so the payout table is scaled to `field_size`
entries with the same prize share and shape (common.payouts).
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core.seed import seed_for
from ivan.sports.common.payouts import scale_payouts_correctly
from ivan.sports.mma.ingest import load_fights
from ivan.sports.mma.models import HistoricalContest, Lineup, LineupSimulationResult, MMAPlayer
from ivan.sports.mma.payouts import parse_payout_csv, payouts_from_db
from ivan.sports.mma.sample_field import DEFAULTS as _FIELD_DEFAULTS, sample_field

DEFAULT_MODEL_PATH = _FIELD_DEFAULTS["dupe_model_path"]


def player_matrix(fights, n_sims: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[int, int]]:
    """P[s, j] = fantasy points of fighter j in simulation s; plus fighter id -> column."""
    cols: dict[int, int] = {}
    series = []
    for f in fights:
        a, b = f.simulate(n_sims, rng)
        cols[f.f1.id] = len(series)
        series.append(a)
        cols[f.f2.id] = len(series)
        series.append(b)
    return np.column_stack(series) if series else np.zeros((n_sims, 0)), cols


def rank_and_pay(scores: np.ndarray, pay_arr: np.ndarray, entry_fee: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For scores (n_sims x M entries): payout, roi and rank per entry. Entries
    with equal scores occupy consecutive ranks and split those prizes equally.
    """
    n, M = scores.shape
    payouts = np.zeros_like(scores)
    ranks = np.zeros((n, M), dtype=np.int32)
    csum = np.cumsum(pay_arr)
    for i in range(n):
        order = np.argsort(-scores[i], kind="stable")
        sc = scores[i][order]
        edges = np.concatenate(([0], np.nonzero(np.abs(np.diff(sc)) > 1e-8)[0] + 1, [M]))
        start, end = edges[:-1] + 1, edges[1:]
        pools = csum[end - 1] - np.where(start >= 2, csum[np.maximum(start - 2, 0)], 0.0)
        per_entry = pools / (end - start + 1)
        for g in range(len(start)):
            idx = order[edges[g]:edges[g + 1]]
            payouts[i, idx] = per_entry[g]
            ranks[i, idx] = start[g]
    rois = (payouts - entry_fee) / entry_fee if entry_fee else np.zeros_like(payouts)
    return payouts, rois, ranks


def find_contest_id(session: Session, text: str, days: int = 2) -> int:
    """The one upcoming contest whose name contains `text`; the biggest if several, error if none."""
    from datetime import timedelta
    rows = (session.query(HistoricalContest).filter(HistoricalContest.date >= datetime.now() - timedelta(days=days))
            .order_by(HistoricalContest.total_entries.desc()).all())
    hits = [r for r in rows if text.lower() in (r.name or "").lower()]
    if not hits:
        raise SystemExit(f"no upcoming contest name contains {text!r}; run scripts/mma/ingest.py or get_contest_ids.py")
    if len(hits) > 1:
        print(f"{len(hits)} contests match {text!r}; using the biggest: {hits[0].name!r} ({hits[0].total_entries:,} entries)")
    return int(hits[0].contest_id)


def resolve_contest(session: Session, cfg: dict, *, contest_id: Optional[int] = None, payouts_csv: str | Path | None = None,
                    total_entries: Optional[int] = None, entry_fee: Optional[float] = None,
                    contest_name: Optional[str] = None) -> dict:
    """What the sim needs about the contest, from the DB row or from the manual description."""
    c = cfg.get("contest", {})
    if contest_id is not None:
        row = session.query(HistoricalContest).filter_by(contest_id=int(contest_id)).one_or_none()
        if row is None:
            raise SystemExit(f"contest {contest_id} is not in the database; run scripts/mma/get_contest_ids.py")
        table = payouts_from_db(session, int(contest_id))
        if table.empty:
            raise SystemExit(f"contest {contest_id} has no payout table; run scripts/mma/get_payouts.py")
        return {"contest_id": int(contest_id), "name": contest_name or row.name or str(contest_id),
                "total_entries": int(row.total_entries), "entry_fee": float(row.entry_fee), "payouts": table}
    path = payouts_csv or cfg.get("paths", {}).get("payouts_csv")
    table = parse_payout_csv(path)
    if table.empty:
        raise SystemExit(f"no payout rows parsed from {path}")
    return {"contest_id": None, "name": contest_name or c.get("name") or Path(path).stem,
            "total_entries": int(total_entries or c.get("total_entries", 10000)),
            "entry_fee": float(entry_fee if entry_fee is not None else c.get("entry_fee", 0.5)), "payouts": table}


def simulate_contest(session: Session, cfg: dict, contest: dict, *, field_size: Optional[int] = None,
                     n_sims: Optional[int] = None, store: bool = True) -> dict:
    """Run it. Returns a report: contest, field diagnostics, exposures, invariants, timing."""
    t0 = time.time()
    field_cfg = cfg.get("field", {})
    field_size = int(field_size or field_cfg.get("size", 5000))
    n_sims = int(n_sims or cfg.get("simulation", {}).get("n_sims", 2000))
    cap = int(cfg.get("lineup_generation", {}).get("salary_cap", 50000))

    players = {p.id: p for p in session.query(MMAPlayer).all()}
    fights = load_fights(session, cfg.get("scoring"))
    lineups = session.query(Lineup).order_by(Lineup.id).all()
    if not lineups:
        raise SystemExit("no candidate lineups; run scripts/mma/generate_lineups.py first")
    ids = [list(lu.player_ids_list) for lu in lineups]
    missing = {p for lu in ids for p in lu} - set(players)
    if missing:
        raise SystemExit(f"{len(missing)} fighter id(s) in the lineup pool are not in the slate; regenerate lineups")

    rng = np.random.default_rng(seed_for("mma.simulate"))
    salary = {pid: p.salary for pid, p in players.items()}
    win_prob = {pid: (p.win_prob if p.win_prob is not None else 0.5) for pid, p in players.items()}
    fight_of = {pid: p.fight_id for pid, p in players.items()}
    main_event = {pid for f in fights if f.rounds == 5 for pid in (f.f1.id, f.f2.id)}
    counts, field_meta = sample_field(ids, salary, win_prob, field_size, field_cfg, rng, cap, fights=fight_of, main_event=main_event)
    in_field = np.nonzero(counts)[0]
    predicted = None                                                             # expected copies in the real field, per candidate
    if str(field_cfg.get("model", "original")).lower() == "fitted":
        from ivan.sports.mma.sample_field import expected_copies, load_dupe_model
        model = load_dupe_model(field_cfg.get("dupe_model_path", DEFAULT_MODEL_PATH))
        if model is not None:
            predicted = expected_copies(ids, salary, win_prob, model, contest["total_entries"], fight_of, cap, main_event=main_event)

    # payout table scaled to the simulated field
    scaled = scale_payouts_correctly(field_size, contest["payouts"], contest["total_entries"], contest["entry_fee"])
    pay_arr = np.zeros(field_size, dtype=float)
    for r, a in scaled.items():
        if 1 <= r <= field_size:
            pay_arr[r - 1] = float(a)

    # score every unique lineup, expand to entries, rank
    P, col = player_matrix(fights, n_sims, rng)
    U = np.array([[col[p] for p in ids[i]] for i in in_field])                 # unique lineups in the field x 6
    scores_u = P[:, U].sum(axis=2)                                               # n_sims x unique
    entry_of = np.repeat(np.arange(len(in_field)), counts[in_field])             # entry -> unique index
    first_entry = np.searchsorted(entry_of, np.arange(len(in_field)))            # a representative entry per lineup
    payouts_e, rois_e, ranks_e = rank_and_pay(scores_u[:, entry_of], pay_arr, contest["entry_fee"])
    payouts_u, rois_u, ranks_u = payouts_e[:, first_entry], rois_e[:, first_entry], ranks_e[:, first_entry]

    pool_err = float(np.abs(payouts_e.sum(axis=1) - pay_arr.sum()).max())
    if store:
        session.query(LineupSimulationResult).filter(LineupSimulationResult.contest_name == contest["name"]).delete(synchronize_session=False)
        rows = []
        for k, i in enumerate(in_field):
            rows.append(LineupSimulationResult(
                lineup_id=lineups[i].id, contest_id=contest["contest_id"], contest_name=contest["name"], dupes=int(counts[i]),
                predicted_dupes=(float(predicted[i]) if predicted is not None else None),
                fpts=np.round(scores_u[:, k], 2).tolist(), payout=np.round(payouts_u[:, k], 4).tolist(),
                roi=np.round(rois_u[:, k], 4).tolist(), rank=ranks_u[:, k].tolist(),
                avg_fpts=float(scores_u[:, k].mean()), avg_payout=float(payouts_u[:, k].mean()), avg_roi=float(rois_u[:, k].mean()),
                cash_pct=float((payouts_u[:, k] > 0).mean()), win_pct=float((ranks_u[:, k] == 1).mean())))
        session.bulk_save_objects(rows)
        session.commit()

    # exposures of the sampled field
    exp_counts: dict[int, int] = {}
    for i in in_field:
        for p in ids[i]:
            exp_counts[p] = exp_counts.get(p, 0) + int(counts[i])
    exposures = sorted(({"id": pid, "name": players[pid].name, "salary": players[pid].salary, "win_prob": win_prob[pid],
                         "exposure": n / field_size} for pid, n in exp_counts.items()), key=lambda e: -e["exposure"])

    return {"contest": {k: v for k, v in contest.items() if k != "payouts"}, "field": field_meta, "exposures": exposures,
            "n_sims": n_sims, "candidates": len(lineups), "scaled_prize_pool": float(pay_arr.sum()),
            "paying_ranks": int((pay_arr > 0).sum()), "pool_max_abs_error": pool_err,
            "fpts_mean": float(scores_u.mean()), "fpts_p90": float(np.percentile(scores_u, 90)),
            "seconds": time.time() - t0, "finished_at": datetime.now().isoformat(timespec="seconds")}


def field_frame(session: Session, contest_name: str) -> pd.DataFrame:
    """The simulated field as the original exported it: F1..F6, Avg Payout, Avg ROI, Cash %, Salary Remaining, Dupe Count."""
    from ivan.sports.mma.choose import lineup_rows
    results = (session.query(LineupSimulationResult).filter(LineupSimulationResult.contest_name == contest_name)
               .order_by(LineupSimulationResult.dupes.desc(), LineupSimulationResult.avg_roi.desc()).all())
    return lineup_rows(session, results)
