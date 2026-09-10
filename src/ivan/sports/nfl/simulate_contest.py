# src/ivan/sports/nfl/simulate_contest.py
"""
Simulate one DraftKings contest against the generated candidate field.

    field   sample_field() draws an opposing field of `field_cap` entries from
            the candidate lineups, with duplication learned from history
    scores  one correlated draw per simulation for every player on the slate
    payouts the contest's real payout table scaled to the simulated field size
    results per unique lineup: fpts / payout / ROI / rank per simulation,
            aggregated over its copies, replacing any earlier results for the
            contest, plus a ContestSimReport documenting the run (see sim_report)
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core.utils import coerce_player_ids as _coerce_player_ids
from ivan.sports.common.payouts import scale_payouts_by_rank_average
from ivan.sports.nfl import sim_report
from ivan.sports.nfl.correlated_samples import generate_correlated_samples_gaussian
from ivan.sports.nfl.models import ContestPayout, HistoricalContest, Lineup, LineupSimulationResult
from ivan.sports.nfl.paths import DIAG_DIR, MODELS_DIR
from ivan.sports.nfl.sample_field import sample_field

MAX_WORKERS = 8
MAX_CHUNKS = 12


def clean_old_diagnostics(diagnostics_root: str | Path = DIAG_DIR, days_to_keep: int = 7) -> int:
    """Remove per-contest diagnostic folders older than days_to_keep. Returns how many."""
    diagnostics_root = Path(diagnostics_root)
    if not diagnostics_root.exists():
        return 0
    cutoff = datetime.now() - timedelta(days=days_to_keep)
    removed = 0
    for folder in diagnostics_root.iterdir():
        if folder.is_dir() and datetime.fromtimestamp(folder.stat().st_mtime) < cutoff:
            shutil.rmtree(folder)
            removed += 1
    return removed


sanitize_contest_name = sim_report.sanitize_contest_name


# ------------------------------------------------------------------------- field
def _expand_sampled_field(session: Session, counts_by_lineup: Dict[int, int]):
    """
    Lay the sampled field out as M entries (one per copy of a lineup).
    Returns (u_lids, pid_lists_by_lid, lineup_ix, unique_pids, pid_to_idx, starts, ends):
    entries [starts[u], ends[u]) are the copies of u_lids[u]; lineup_ix[m] holds
    the player column indexes of entry m.
    """
    u_lids = sorted(lid for lid, c in counts_by_lineup.items() if c > 0)
    cnts = np.array([int(counts_by_lineup[lid]) for lid in u_lids], dtype=np.int64)
    starts = np.cumsum(np.r_[0, cnts[:-1]]) if len(cnts) else np.array([], dtype=np.int64)
    ends = starts + cnts
    M = int(cnts.sum())

    pid_lists_by_lid: Dict[int, List[int]] = {}
    for i in range(0, len(u_lids), 900):                       # SQLite variable limit
        for lid, j in session.query(Lineup.id, Lineup.player_ids_list).filter(Lineup.id.in_(u_lids[i:i + 900])).all():
            pid_lists_by_lid[int(lid)] = _coerce_player_ids(j)

    unique_pids = sorted({p for lid in u_lids for p in pid_lists_by_lid.get(lid, [])})
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    lineup_ix: List[List[int]] = [None] * M
    for u, lid in enumerate(u_lids):
        idxs = [pid_to_idx[p] for p in pid_lists_by_lid.get(lid, [])]
        for pos in range(int(starts[u]), int(ends[u])):
            lineup_ix[pos] = idxs
    return u_lids, pid_lists_by_lid, lineup_ix, unique_pids, pid_to_idx, starts, ends


# ------------------------------------------------------------------------- payouts
def _payout_tiers(session: Session, contest_id: int) -> List[dict]:
    rows = (session.query(ContestPayout).filter(ContestPayout.contest_id == contest_id)
            .order_by(ContestPayout.min_position.asc()).all())
    if not rows:
        raise ValueError(f"No payouts found for contest_id={contest_id}; run get_payouts.py")
    return [{"min": int(r.min_position), "max": int(r.max_position), "amount": float(r.payout_amount)} for r in rows]


def _scale_payouts(tiers: List[dict], num_field_lineups: int, original_field_size: int) -> Dict[int, float]:
    df = pd.DataFrame([(t["min"], t["max"], t["amount"]) for t in tiers], columns=["Start Rank", "End Rank", "Payout"])
    return scale_payouts_by_rank_average(df, original_field_size=original_field_size, num_field_lineups=num_field_lineups)


# ------------------------------------------------------------------------- samples
def _player_samples_matrix(session: Session, pid_to_idx: Dict[int, int], n_samples: int,
                           model_dir: str | Path = MODELS_DIR):
    """
    Scores P[num_sims, num_players] aligned to pid_to_idx from one correlated
    draw per simulation. Also returns the raw per-player samples, the sampler's
    diagnostics and the field players that had no samples (zero columns).
    """
    raw, diag = generate_correlated_samples_gaussian(session, model_dir=model_dir, n_samples=n_samples,
                                                     return_diagnostics=True)
    P = np.zeros((n_samples, len(pid_to_idx)), dtype=np.float32)
    missing = []
    for pid, col in pid_to_idx.items():
        v = raw.get(pid)
        if not isinstance(v, np.ndarray) or v.size == 0:
            missing.append(pid)
            continue
        v = v.astype(np.float32, copy=False)
        if v.shape[0] < n_samples:
            v = np.tile(v, int(math.ceil(n_samples / v.shape[0])))
        P[:, col] = v[:n_samples]
    return P, raw, diag, missing


# ------------------------------------------------------------------------- simulate
def _simulate_chunk(start: int, end: int, lineup_ix: List[List[int]], P_chunk: np.ndarray,
                    entry_fee: float, scaled_payouts: Dict[int, float]) -> Dict[str, object]:
    """
    Score every entry for sims [start, end), rank them (ties share the pooled
    prize of their rank range) and pay out. Runs in a worker process; the
    invariant violations it finds come back as counts, not prints.
    """
    slice_len = end - start
    M = len(lineup_ix)
    scores = np.zeros((slice_len, M), dtype=np.float64)
    payouts = np.zeros_like(scores)
    rois = np.zeros_like(scores)
    ranks = np.zeros((slice_len, M), dtype=np.int32)

    pay_arr = np.zeros(M, dtype=np.float64)                    # index 0 -> rank 1
    for r, a in scaled_payouts.items():
        if 1 <= r <= M:
            pay_arr[r - 1] = float(a)
    pay_csum = np.cumsum(pay_arr)
    expected_pool = float(pay_arr.sum())
    top_prize = float(pay_arr[0]) if M else 0.0
    atol = 1e-8

    stats = {"max_abs_error": 0.0, "sims_off": 0, "tie_anomalies": 0, "over_top": 0}

    for i in range(slice_len):
        row = P_chunk[i]
        for m in range(M):
            idxs = lineup_ix[m]
            if idxs:
                scores[i, m] = row[idxs].sum(dtype=np.float64)

        order = np.argsort(-scores[i], kind="stable")
        sc_sorted = scores[i][order]
        edges = np.concatenate(([0], np.nonzero(np.abs(np.diff(sc_sorted)) > atol)[0] + 1, [M]))
        start_rank, end_rank = edges[:-1] + 1, edges[1:]       # 1-indexed, inclusive
        sizes = end_rank - start_rank + 1
        j1m1 = start_rank - 2
        pools = pay_csum[end_rank - 1] - np.where(j1m1 >= 0, pay_csum[np.maximum(j1m1, 0)], 0.0)
        per_entry = np.divide(pools, sizes, out=np.zeros_like(pools), where=sizes > 0)

        for g in range(len(start_rank)):
            idx = order[edges[g]:edges[g + 1]]
            payouts[i, idx] = per_entry[g]
            ranks[i, idx] = int(start_rank[g])
            sl = pay_arr[start_rank[g] - 1:end_rank[g]]
            if sl.size and sl.min() != sl.max() and not (sl.min() - 1e-8 <= per_entry[g] <= sl.max() + 1e-8):
                stats["tie_anomalies"] += 1
        rois[i] = (payouts[i] - entry_fee) / entry_fee

        err = abs(float(payouts[i].sum()) - expected_pool)
        if not np.isfinite(err) or err > 1e-4:
            stats["sims_off"] += 1
        stats["max_abs_error"] = max(stats["max_abs_error"], err if np.isfinite(err) else float("inf"))
        if float(payouts[i].max()) > top_prize + 1e-8:
            stats["over_top"] += 1

    return {"start": start, "end": end, "scores": scores.astype(np.float32), "payouts": payouts.astype(np.float32),
            "rois": rois.astype(np.float32), "ranks": ranks, "stats": stats}


def simulate_one_contest(
    session: Session,
    contest_id: int,
    contest_name: str,
    config_path: str,
    field_cap: int,
    num_sims: int = 1000,
    use_classic_dupes: bool = False,
) -> dict:
    """
    Simulate a contest end to end and persist the per-lineup results and the
    run's report. Returns the report (see sim_report for its layout).
    """
    t0 = time.time()
    cfg = json.load(open(config_path, encoding="utf-8"))
    contest = session.query(HistoricalContest).filter_by(contest_id=contest_id).one()
    mode = "showdown" if contest.contest_type == "Showdown Captain Mode" else "classic"
    field_sz = int(contest.total_entries)
    capped = min(field_cap, field_sz)
    entry_fee = float(contest.entry_fee)
    n_candidates = session.query(Lineup).count()

    report: dict = {
        "version": sim_report.REPORT_VERSION,
        "contest": {"id": contest_id, "name": contest_name, "mode": mode, "contest_type": contest.contest_type,
                    "entry_fee": entry_fee, "total_entries": field_sz, "field_cap": field_cap, "simulated_size": capped,
                    "num_sims": num_sims, "use_classic_dupes": use_classic_dupes, "candidates": n_candidates,
                    "start": contest.date, "started_at": datetime.now(),
                    "config": {k: cfg.get(k) for k in ("field", "sampling", "simulation", "randomness") if k in cfg}},
    }

    # 1) the opposing field, with the contest's real size and fee driving the duplication model
    counts_by_lineup, meta = sample_field(session, field_size=capped, config=cfg, mode=mode,
                                          contest_id=contest_id, use_classic_dupes=use_classic_dupes,
                                          total_entries=field_sz, entry_fee=entry_fee)
    M = sum(counts_by_lineup.values())
    if M == 0:
        raise RuntimeError(f"field sampler returned no entries for contest {contest_id} ({meta.get('error', 'no lineups?')})")
    if M != capped:
        raise RuntimeError(f"field sampler returned {M} entries, expected {capped}")
    u_lids, pid_lists_by_lid, lineup_ix, unique_pids, pid_to_idx, starts, ends = _expand_sampled_field(session, counts_by_lineup)
    players = sim_report.player_meta(session, unique_pids)

    # 2) payouts scaled from the real field to the simulated one
    tiers = _payout_tiers(session, contest_id)
    scaled = _scale_payouts(tiers, M, field_sz)
    report["payouts"] = sim_report.payout_section(tiers, scaled, entry_fee, field_sz, M)
    pool_in_field = float(sum(a for r, a in scaled.items() if 1 <= r <= M))
    paid_places = max((r for r, a in scaled.items() if 1 <= r <= M and a > 0), default=0)

    # 3) one correlated draw per simulation
    P, raw, diag, missing = _player_samples_matrix(session, pid_to_idx, num_sims)
    players.update(sim_report.player_meta(session, [p for g in diag["games"] for p in g["player_ids"]]
                                          + [d["id"] for d in diag["independent"]]))
    report["field"] = sim_report.field_section(counts_by_lineup, pid_lists_by_lid, players, meta, n_candidates)
    report["correlation"] = sim_report.correlation_section(raw, diag, players)

    # 4) simulate in parallel chunks
    t_sim = time.time()
    chunk = math.ceil(num_sims / min(MAX_CHUNKS, num_sims))
    full_scores = np.zeros((num_sims, M), dtype=np.float32)
    full_payouts = np.zeros_like(full_scores)
    full_rois = np.zeros_like(full_scores)
    full_ranks = np.zeros((num_sims, M), dtype=np.int32)
    chunk_stats = {"max_abs_error": 0.0, "sims_off": 0, "tie_anomalies": 0, "over_top": 0,
                   "workers": min(MAX_WORKERS, os.cpu_count() or 1)}
    with ProcessPoolExecutor(max_workers=chunk_stats["workers"]) as exe:
        futures = [exe.submit(_simulate_chunk, s, min(s + chunk, num_sims), lineup_ix, P[s:min(s + chunk, num_sims)],
                              entry_fee, scaled) for s in range(0, num_sims, chunk)]
        for fut in as_completed(futures):
            part = fut.result()
            s, e = part["start"], part["end"]
            full_scores[s:e], full_payouts[s:e], full_rois[s:e], full_ranks[s:e] = (
                part["scores"], part["payouts"], part["rois"], part["ranks"])
            st = part["stats"]
            chunk_stats["max_abs_error"] = max(chunk_stats["max_abs_error"], st["max_abs_error"])
            for k in ("sims_off", "tie_anomalies", "over_top"):
                chunk_stats[k] += st[k]

    report["simulation"] = sim_report.simulation_section(
        full_scores, full_payouts, full_rois, full_ranks, starts, ends, u_lids, pid_lists_by_lid, players,
        entry_fee, pool_in_field, paid_places, chunk_stats, time.time() - t_sim, missing)

    # 5) persist per-unique-lineup aggregates, replacing earlier results for this contest
    session.query(LineupSimulationResult).filter(LineupSimulationResult.contest_id == contest_id).delete(synchronize_session=False)
    now = datetime.utcnow()
    rows = []
    for u, lid in enumerate(u_lids):
        a, b = int(starts[u]), int(ends[u])
        rows.append(dict(lineup_id=int(lid), contest_id=contest_id, contest_name=contest_name, created_at=now,
                         fpts=full_scores[:, a:b].mean(axis=1).tolist(),
                         payout=full_payouts[:, a:b].sum(axis=1).tolist(),     # total over the lineup's copies
                         roi=full_rois[:, a:b].mean(axis=1).tolist(),          # mean ROI per entry
                         rank=full_ranks[:, a:b].mean(axis=1).tolist()))
    session.bulk_insert_mappings(LineupSimulationResult, rows)

    report["contest"]["finished_at"] = datetime.now()
    report["contest"]["seconds"] = time.time() - t0
    report["checks"] = sim_report.build_checks(report)
    report["check_counts"] = sim_report.summarise_checks(report["checks"])
    sim_report.save_report(session, report)
    session.commit()
    return sim_report.jsonable(report)
