# src/ivan/sports/nfl/simulate_contest.py
"""
Simulate or evaluate one DraftKings contest for every candidate lineup.

Two methods, picked by config `sim.method` (default "montecarlo"):

montecarlo  the sampled field of `field_cap` entries, one correlated draw per
            simulation for every player, ranks and payouts counted, the payout
            table scaled to the sampled field (simulate_one_contest_mc).
conditioned the evaluation below, in the lab until its count model handles a
            field whose lineups are deterministic given the world:

    worlds  one draw per world of the slate's game latents (team points), shared
            by every contest of the run (ivan.sports.nfl.conditioned.build_world_model)
    field   the field sampler's lineup weights give each candidate its expected
            copies in the contest's real field; nothing is sampled
    payouts the contest's real payout table, unscaled
    results per candidate lineup: expected payout / ROI / rank / cash probability
            in every world, computed in closed form from the lineup's conditional
            score distribution against the field's (conditioned.evaluate_contest),
            plus one realized draw per world for the calibration tests; the rows
            replace any earlier results for the contest, and a ContestSimReport
            documents the run (see sim_report)
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core.utils import coerce_player_ids as _coerce_player_ids
from ivan.core.seed import seed_for
from ivan.database.session import add_missing_columns
from ivan.sports.common.payouts import scale_payouts_by_rank_average
from ivan.sports.nfl import sim_report
from ivan.sports.nfl.conditioned import (Ladder, LineupPool, WorldModel, build_lineup_pool, build_world_model, contest_tables,
                                         describe_world_model, evaluate_contest, field_quantiles, lineup_moments, pool_conservation)
from ivan.sports.nfl.correlated_samples import generate_correlated_samples_gaussian
from ivan.sports.nfl.models import ContestPayout, HistoricalContest, Lineup, LineupSimulationResult, NFLBase
from ivan.sports.nfl.paths import DIAG_DIR
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
                           model_dir: str | Path | None = None, seed: int | None = None):
    """
    Scores P[num_sims, num_players] aligned to pid_to_idx from one correlated
    draw per simulation. Also returns the raw per-player samples, the sampler's
    diagnostics and the field players that had no samples (zero columns).
    """
    raw, diag = generate_correlated_samples_gaussian(session, model_dir=model_dir, n_samples=n_samples, seed=seed,
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



# ------------------------------------------------------------------------- samples (diagnostic draw)
def _player_samples(session: Session, pids, n_samples: int, model_dir: str | Path | None = None, seed: int | None = None):
    """
    One correlated draw per world for the correlation report and the players'
    sampled quantiles in the artifact. The contest itself is not scored from
    these draws (see simulate_one_contest); they document what the game model
    says about each player.
    """
    raw, diag = generate_correlated_samples_gaussian(session, model_dir=model_dir, n_samples=n_samples, seed=seed,
                                                     return_diagnostics=True)
    missing = [pid for pid in pids if not isinstance(raw.get(pid), np.ndarray) or raw[pid].size == 0]
    return raw, diag, missing


# ------------------------------------------------------------------------- the slate's worlds
@dataclass
class SlateWorlds:
    """The world model and the candidate pool, shared by every contest of a run."""
    world: WorldModel
    pool: LineupPool
    n_worlds: int
    seed: int

    @classmethod
    def build(cls, session: Session, n_worlds: int, seed: int | None = None, log=print) -> "SlateWorlds":
        if seed is None:
            seed = seed_for("sim/worlds")
        if seed is None:                                            # no IVAN_SEED: draw one and record it with the run
            seed = int(np.random.SeedSequence().entropy % (2 ** 32))
        seed = int(seed)
        t0 = time.time()
        world = build_world_model(session, n_worlds=n_worlds, seed=seed)
        lineups = {int(lid): _coerce_player_ids(j) for lid, j in session.query(Lineup.id, Lineup.player_ids_list).all()}
        pool = build_lineup_pool(world, lineups)
        pool.m, pool.s = lineup_moments(world, pool.L)
        log(describe_world_model(world))
        log(f"pool: {len(pool.lineup_ids):,} candidate lineups, conditional sd {pool.s.min():.1f}-{pool.s.max():.1f} "
            f"({time.time() - t0:.1f}s)" + (f"; {len(pool.missing_players)} player id(s) in lineups have no model row" if pool.missing_players else ""))
        return cls(world=world, pool=pool, n_worlds=int(n_worlds), seed=seed)


# ------------------------------------------------------------------------- simulate
def simulate_one_contest_conditioned(
    session: Session,
    contest_id: int,
    contest_name: str,
    config_path: str,
    field_cap: int = 0,
    num_sims: int = 1000,
    use_classic_dupes: bool = False,
    artifact_dir: str | Path | None = None,
    worlds: SlateWorlds | None = None,
) -> dict:
    """
    Evaluate a contest end to end and persist the per-lineup results and the
    run's report. Returns the report (see sim_report for its layout).

    The contest is the real field (its full size and its real payout table):
    the field sampler's lineup weights give every candidate its expected
    copies, the slate's worlds (ivan.sports.nfl.conditioned) give every
    candidate a conditional score distribution in each world, and the
    conditioned evaluation turns the two into each lineup's expected payout,
    cash probability and rank per world, plus one realized draw per world for
    the calibration tests. `field_cap` is accepted for the old call sites and
    ignored: nothing is sampled or scaled any more. With `artifact_dir` the
    run's assumptions are also written there for the validation layer; a
    failure to write them never sinks the evaluation.
    """
    t0 = time.time()
    cfg = json.load(open(config_path, encoding="utf-8"))
    contest = session.query(HistoricalContest).filter_by(contest_id=contest_id).one()
    mode = "showdown" if contest.contest_type == "Showdown Captain Mode" else "classic"
    F = int(contest.total_entries)
    entry_fee = float(contest.entry_fee)
    n_candidates = session.query(Lineup).count()
    if worlds is None:
        worlds = SlateWorlds.build(session, n_worlds=num_sims)
    pool, world = worlds.pool, worlds.world
    W = worlds.n_worlds

    report: dict = {
        "version": sim_report.REPORT_VERSION,
        "contest": {"id": contest_id, "name": contest_name, "mode": mode, "contest_type": contest.contest_type,
                    "entry_fee": entry_fee, "total_entries": F, "field_cap": None, "simulated_size": F,
                    "num_sims": W, "use_classic_dupes": use_classic_dupes, "candidates": n_candidates,
                    "start": contest.date, "started_at": datetime.now(), "method": "conditioned",
                    "config": {k: cfg.get(k) for k in ("field", "randomness") if k in cfg}},
    }

    # 1) the field: every candidate's expected copies from the sampler's weights (the sampled counts only feed the report)
    counts_by_lineup, meta = sample_field(session, field_size=min(F, 10_000), config=cfg, mode=mode, contest_id=contest_id,
                                          use_classic_dupes=use_classic_dupes, total_entries=F, entry_fee=entry_fee)
    sampler_arrays = meta.pop("_arrays", None) or {}
    if "lineup_id" not in sampler_arrays or "combined_weights" not in sampler_arrays:
        raise RuntimeError(f"field sampler returned no lineup weights for contest {contest_id} ({meta.get('error', 'no lineups?')})")
    w_by_lid = dict(zip(np.asarray(sampler_arrays["lineup_id"]).tolist(), np.asarray(sampler_arrays["combined_weights"], dtype=float).tolist()))
    weights = np.asarray([w_by_lid.get(int(l), 0.0) for l in pool.lineup_ids], dtype=np.float64)
    if weights.sum() <= 0:
        raise RuntimeError(f"field sampler weights do not cover the candidate pool for contest {contest_id}")
    own = weights / weights.sum() * F                                   # expected copies of each candidate in the field
    u_lids = [int(l) for l in pool.lineup_ids]
    pid_lists_by_lid = {lid: pool.players[lid] for lid in u_lids}
    unique_pids = sorted({p for lid in u_lids for p in pid_lists_by_lid[lid]})
    players = sim_report.player_meta(session, unique_pids)

    # 2) the real payout table
    tiers = _payout_tiers(session, contest_id)
    pay_by_rank = np.zeros(F)
    for t in tiers:
        lo, hi = max(1, int(t["min"])), min(F, int(t["max"]))
        if hi >= lo:
            pay_by_rank[lo - 1:hi] = float(t["amount"])
    ladder = Ladder(pay_by_rank, F)
    scaled = {r: float(pay_by_rank[r - 1]) for r in range(1, F + 1) if pay_by_rank[r - 1] > 0}
    report["payouts"] = sim_report.payout_section(tiers, scaled, entry_fee, F, F)
    paid_places = ladder.paid_places

    # 3) the diagnostic draw: correlation report and player quantiles (the contest is not scored from it)
    sim_seed = seed_for(f"sim/{contest_id}")
    raw, diag, missing = _player_samples(session, unique_pids, min(W, 2000), seed=sim_seed)
    players.update(sim_report.player_meta(session, [p for g in diag["games"] for p in g["player_ids"]]
                                          + [d["id"] for d in diag["independent"]]))
    report["field"] = sim_report.field_section(counts_by_lineup, pid_lists_by_lid, players, meta, n_candidates)
    report["field"]["expected_copies"] = {"sum": float(own.sum()), "max": float(own.max()), "p50": float(np.median(own)),
                                          "p90": float(np.percentile(own, 90)), "lineups_with_copies_ge_1": int((own >= 1).sum())}
    report["correlation"] = sim_report.correlation_section(raw, diag, players)

    # 4) the conditioned evaluation
    t_sim = time.time()
    rng = np.random.default_rng(seed_for(f"sim/draw/{contest_id}"))
    tables = contest_tables(pool.m, pool.s, own, ladder)
    ev = evaluate_contest(pool.m, pool.s, own, ladder, tables, rng=rng)
    roi = ev.roi(entry_fee)
    roi_draw = ev.roi_draw(entry_fee)
    conservation = pool_conservation(ev, own, ladder)
    score_pct, cash_line, winning = field_quantiles(tables.lam, tables.grid, F, paid_places)
    report["simulation"] = sim_report.conditioned_section(
        ev, roi, own, pool, players, entry_fee, ladder, conservation, score_pct, cash_line, winning,
        time.time() - t_sim, missing, tables)
    report["worlds"] = {"n": W, "seed": worlds.seed, "teams": list(world.teams),
                        "team_points": np.round(world.team_points, 1).tolist(),
                        "games": [g for g in world.games if "teams" in g]}

    # 5) persist per-lineup rows, replacing earlier results for this contest
    session.query(LineupSimulationResult).filter(LineupSimulationResult.contest_id == contest_id).delete(synchronize_session=False)
    now = datetime.utcnow()
    copies = 1.0 + own
    for a in range(0, len(u_lids), 1000):
        rows = []
        for i in range(a, min(a + 1000, len(u_lids))):
            rows.append(dict(lineup_id=u_lids[i], contest_id=contest_id, contest_name=contest_name, created_at=now,
                             fpts=np.round(pool.m[i], 1).tolist(),                      # conditional mean score per world
                             payout=np.round(ev.pay[i] * copies[i], 2).tolist(),        # expected prize, total over the lineup's copies
                             roi=np.round(roi[i], 4).tolist(),                           # expected ROI of one entry per world
                             rank=np.round(ev.rank[i], 0).tolist(),                      # expected rank per world
                             cash=np.round(ev.cash[i], 4).tolist(),                      # P(cash) per world
                             roi_draw=np.round(roi_draw[i], 3).tolist()))                # one realized ROI per world
        session.bulk_insert_mappings(LineupSimulationResult, rows)

    report["contest"]["finished_at"] = datetime.now()
    report["contest"]["seconds"] = time.time() - t0
    report["checks"] = sim_report.build_checks(report)
    report["check_counts"] = sim_report.summarise_checks(report["checks"])
    sim_report.save_report(session, report)
    session.commit()

    if artifact_dir is not None:
        try:
            from ivan.core.artifacts import contest_artifact_paths
            from ivan.sports.nfl.run_artifacts import build_conditioned_artifact, write_sim_artifact
            arrays, art_meta = build_conditioned_artifact(
                contest={"id": contest_id, "name": contest_name, "mode": mode, "contest_type": contest.contest_type,
                         "entry_fee": entry_fee, "total_entries": F, "field_cap": None, "start": contest.date,
                         "candidates": n_candidates},
                lineup_ids=pool.lineup_ids, pid_lists_by_lid=pid_lists_by_lid, own=own, ev=ev, roi=roi, mean_score=pool.m.mean(axis=1),
                lineup_sd=pool.s, raw_samples=raw, players=players, pay_by_rank=pay_by_rank, tiers=tiers, paid_places=paid_places,
                pool_in_field=ladder.pool, score_pct=score_pct, cash_line=cash_line, winning_score=winning,
                sampler_arrays=sampler_arrays, sampler_meta=meta, seed=worlds.seed, world=world)
            npz_path, json_path = contest_artifact_paths(Path(artifact_dir), contest_id)
            write_sim_artifact(npz_path, json_path, arrays, art_meta)
            report["artifact"] = str(npz_path)
        except Exception as e:                      # never sink a run over its paper trail
            print(f"  WARNING: could not write the run artifact for contest {contest_id}: {type(e).__name__}: {e}")
    return sim_report.jsonable(report)


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


def simulate_one_contest_mc(
    session: Session,
    contest_id: int,
    contest_name: str,
    config_path: str,
    field_cap: int,
    num_sims: int = 1000,
    use_classic_dupes: bool = False,
    artifact_dir: str | Path | None = None,
) -> dict:
    """
    The Monte Carlo contest simulation (the live default): the sampled field of
    `field_cap` entries, one correlated draw per simulation, ranks and payouts
    counted, the payout table scaled to the sampled field. Persists the
    per-lineup results and the run's report, returns the report. With
    `artifact_dir` the run's assumptions are also written there for the
    validation layer (ivan.sports.nfl.run_artifacts); a failure to write
    them never sinks the simulation.
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
                    "config": {k: cfg.get(k) for k in ("field", "randomness") if k in cfg}},
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
    sampler_arrays = meta.pop("_arrays", None)
    u_lids, pid_lists_by_lid, lineup_ix, unique_pids, pid_to_idx, starts, ends = _expand_sampled_field(session, counts_by_lineup)
    players = sim_report.player_meta(session, unique_pids)

    # 2) payouts scaled from the real field to the simulated one
    tiers = _payout_tiers(session, contest_id)
    scaled = _scale_payouts(tiers, M, field_sz)
    report["payouts"] = sim_report.payout_section(tiers, scaled, entry_fee, field_sz, M)
    pool_in_field = float(sum(a for r, a in scaled.items() if 1 <= r <= M))
    paid_places = max((r for r, a in scaled.items() if 1 <= r <= M and a > 0), default=0)

    # 3) one correlated draw per simulation
    sim_seed = seed_for(f"sim/{contest_id}")
    P, raw, diag, missing = _player_samples_matrix(session, pid_to_idx, num_sims, seed=sim_seed)
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
                   "workers": max(1, min(MAX_WORKERS, int(os.environ.get("IVAN_SIM_WORKERS") or (os.cpu_count() or 1))))}   # IVAN_SIM_WORKERS: a replay worker's share of the cores
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
                         rank=full_ranks[:, a:b].mean(axis=1).tolist(),
                         cash=(full_payouts[:, a] > 0).astype(float).tolist(),  # cashed in that simulation (one copy)
                         roi_draw=full_rois[:, a:b].mean(axis=1).tolist()))     # a counted outcome is its own realization
    session.bulk_insert_mappings(LineupSimulationResult, rows)

    report["contest"]["finished_at"] = datetime.now()
    report["contest"]["seconds"] = time.time() - t0
    report["checks"] = sim_report.build_checks(report)
    report["check_counts"] = sim_report.summarise_checks(report["checks"])
    sim_report.save_report(session, report)
    session.commit()

    if artifact_dir is not None:
        try:
            from ivan.core.artifacts import contest_artifact_paths
            from ivan.sports.nfl.run_artifacts import build_sim_artifact, write_sim_artifact
            pay_by_rank = np.array([float(scaled.get(r, 0.0)) for r in range(1, M + 1)])
            arrays, art_meta = build_sim_artifact(
                contest={"id": contest_id, "name": contest_name, "mode": mode, "contest_type": contest.contest_type,
                         "entry_fee": entry_fee, "total_entries": field_sz, "field_cap": field_cap, "start": contest.date,
                         "candidates": n_candidates},
                u_lids=u_lids, starts=starts, ends=ends, pid_lists_by_lid=pid_lists_by_lid, full_scores=full_scores,
                full_payouts=full_payouts, full_rois=full_rois, full_ranks=full_ranks, raw_samples=raw, players=players,
                pay_by_rank=pay_by_rank, tiers=tiers, paid_places=paid_places, pool_in_field=pool_in_field,
                sampler_arrays=sampler_arrays, sampler_meta=meta, seed=sim_seed)
            npz_path, json_path = contest_artifact_paths(Path(artifact_dir), contest_id)
            write_sim_artifact(npz_path, json_path, arrays, art_meta)
            report["artifact"] = str(npz_path)
        except Exception as e:                      # never sink a simulation over its paper trail
            print(f"  WARNING: could not write the run artifact for contest {contest_id}: {type(e).__name__}: {e}")
    return sim_report.jsonable(report)




# ------------------------------------------------------------------------- dispatch
SIM_METHODS = ("montecarlo", "conditioned")
DEFAULT_SIM_METHOD = "montecarlo"


def sim_method(cfg: dict) -> str:
    """config `sim.method`: "montecarlo" (the live default) or "conditioned" (ivan.sports.nfl.conditioned, lab use)."""
    m = str(((cfg or {}).get("sim") or {}).get("method") or DEFAULT_SIM_METHOD).strip().lower()
    if m not in SIM_METHODS:
        raise ValueError(f"unknown sim method {m!r}; expected one of {SIM_METHODS}")
    return m


def simulate_one_contest(session: Session, contest_id: int, contest_name: str, config_path: str, field_cap: int = 10000,
                         num_sims: int = 1000, use_classic_dupes: bool = False, artifact_dir: str | Path | None = None,
                         worlds: "SlateWorlds | None" = None) -> dict:
    """One contest by the configured method (sim_method)."""
    cfg = json.load(open(config_path, encoding="utf-8"))
    if sim_method(cfg) == "conditioned":
        return simulate_one_contest_conditioned(session, contest_id, contest_name, config_path, field_cap, num_sims,
                                                use_classic_dupes, artifact_dir, worlds)
    return simulate_one_contest_mc(session, contest_id, contest_name, config_path, field_cap or 10000, num_sims,
                                   use_classic_dupes, artifact_dir)


# ----------------------------------------------------------------------------- the field alone
def sample_contest_field(session: Session, contest_id: int, contest_name: str, config_path: str, field_cap: int,
                         artifact_dir: str | Path) -> dict:
    """
    The simulator's first step on its own: sample the contest's field from the
    generated pool and write it as a field artifact (run_artifacts.build_field_artifact),
    with no outcomes drawn and nothing persisted to the database. The replay's
    "field" stage (ivan.core.replay) uses it so the field model can be scored
    against the real field without paying for the simulations.
    """
    t0 = time.time()
    cfg = json.load(open(config_path, encoding="utf-8"))
    contest = session.query(HistoricalContest).filter_by(contest_id=contest_id).one()
    mode = "showdown" if contest.contest_type == "Showdown Captain Mode" else "classic"
    field_sz = int(contest.total_entries)
    capped = min(field_cap, field_sz)
    entry_fee = float(contest.entry_fee)
    n_candidates = session.query(Lineup).count()
    counts_by_lineup, meta = sample_field(session, field_size=capped, config=cfg, mode=mode, contest_id=contest_id,
                                          total_entries=field_sz, entry_fee=entry_fee)
    M = sum(counts_by_lineup.values())
    if M == 0:
        raise RuntimeError(f"field sampler returned no entries for contest {contest_id} ({meta.get('error', 'no lineups?')})")
    sampler_arrays = meta.pop("_arrays", None)
    u_lids, pid_lists_by_lid, _, unique_pids, _, starts, ends = _expand_sampled_field(session, counts_by_lineup)
    players = sim_report.player_meta(session, unique_pids)
    tiers = _payout_tiers(session, contest_id)
    scaled = _scale_payouts(tiers, M, field_sz)
    paid_places = max((r for r, a in scaled.items() if 1 <= r <= M and a > 0), default=0)
    pay_by_rank = np.array([float(scaled.get(r, 0.0)) for r in range(1, M + 1)])
    from ivan.core.artifacts import contest_artifact_paths
    from ivan.sports.nfl.run_artifacts import build_field_artifact, write_sim_artifact
    arrays, art_meta = build_field_artifact(
        contest={"id": contest_id, "name": contest_name, "mode": mode, "contest_type": contest.contest_type, "entry_fee": entry_fee,
                 "total_entries": field_sz, "field_cap": field_cap, "start": contest.date, "candidates": n_candidates},
        u_lids=u_lids, starts=starts, ends=ends, pid_lists_by_lid=pid_lists_by_lid, players=players, pay_by_rank=pay_by_rank,
        tiers=tiers, paid_places=paid_places, sampler_arrays=sampler_arrays, sampler_meta=meta, seed=seed_for(f"sim/{contest_id}"))
    npz_path, json_path = contest_artifact_paths(Path(artifact_dir), contest_id)
    write_sim_artifact(npz_path, json_path, arrays, art_meta)
    return {"contest_id": contest_id, "sampled": int(M), "unique": len(u_lids), "paid_places": int(paid_places),
            "seconds": round(time.time() - t0, 1), "artifact": str(npz_path)}


def sample_entries_fields(session, entries_csv, config_path, field_cap: int, artifact_dir: str | Path, log=print) -> dict:
    """sample_contest_field for every contest in the entries file; a failure is logged and the others still run."""
    contests = contests_in_entries(entries_csv)
    log(f"Sampling the field of {len(contests)} contest(s): cap {field_cap:,}; artifacts -> {artifact_dir}")
    out = {"contests": int(len(contests)), "sampled": 0, "failed": []}
    for i, (cid, name) in enumerate(zip(contests["Contest ID"], contests["Contest Name"]), 1):
        try:
            r = sample_contest_field(session, int(cid), name, str(config_path), field_cap, artifact_dir)
            out["sampled"] += 1
            log(f"--- {i}/{len(contests)} {name}: {r['sampled']:,} entries, {r['unique']:,} unique lineups, {r['paid_places']} paid, {r['seconds']}s")
        except Exception as e:   # noqa: BLE001
            import traceback
            session.rollback()
            out["failed"].append({"contest_id": int(cid), "name": name, "error": f"{type(e).__name__}: {e}"})
            log(f"FAILED contest {cid} ('{name}'): {e}\n{traceback.format_exc()}")
    return out


# ----------------------------------------------------------------------------- every contest in an entries file
def contests_in_entries(entries_csv) -> pd.DataFrame:
    """The distinct contests of a DraftKings entries file (one row per contest name, highest id kept)."""
    from ivan.sports.nfl.lineup_export import read_entries
    df = read_entries(entries_csv)[["Contest ID", "Contest Name"]].copy()
    df["Contest Name"] = df["Contest Name"].str.strip()
    return df.sort_values("Contest ID").drop_duplicates(subset=["Contest Name"], keep="last").reset_index(drop=True)


def simulate_entries(session, entries_csv, config_path, n_sims: int, field_cap: int, use_classic_dupes: bool = False,
                     artifact_dir: str | Path | None = None, log=print) -> dict:
    """
    Simulate every contest in the entries file (simulate_one_contest each; a failure is
    logged and the others still run). Returns {"contests", "simulated", "failed", "reports"}.
    """
    contests = contests_in_entries(entries_csv)
    art = Path(artifact_dir) if artifact_dir is not None else None
    method = sim_method(json.load(open(config_path, encoding="utf-8")))
    if method == "conditioned":
        log(f"Evaluating {len(contests)} contest(s) over {n_sims} worlds (conditioned: expected payout per world, real field size and payouts)"
            + (f"; run artifacts -> {art}" if art is not None else ""))
    else:
        log(f"Simulating {len(contests)} contest(s): {n_sims} sims each, field cap {field_cap:,}"
            + (f"; run artifacts -> {art}" if art is not None else ""))
    out = {"contests": int(len(contests)), "simulated": 0, "failed": [], "reports": {}}
    try:
        add_missing_columns(session.get_bind(), NFLBase.metadata)        # cash / roi_draw on an older database
    except Exception as e:   # noqa: BLE001
        log(f"WARNING: could not add missing columns: {e}")
    worlds = SlateWorlds.build(session, n_worlds=n_sims, log=log) if method == "conditioned" else None
    for i, (cid, name) in enumerate(zip(contests["Contest ID"], contests["Contest Name"]), 1):
        log(f"--- {i}/{len(contests)} ---")
        try:
            report = simulate_one_contest(session=session, contest_id=int(cid), contest_name=name, config_path=str(config_path),
                                          field_cap=field_cap, num_sims=n_sims, use_classic_dupes=use_classic_dupes, artifact_dir=art,
                                          worlds=worlds)
            out["reports"][int(cid)] = report
            out["simulated"] += 1
            log(sim_report.format_summary(report))
        except Exception as e:   # noqa: BLE001 - one contest failing must not lose the others
            import traceback
            session.rollback()
            out["failed"].append({"contest_id": int(cid), "name": name, "error": f"{type(e).__name__}: {e}"})
            log(f"FAILED contest {cid} ('{name}'): {e}\n{traceback.format_exc()}")
    log(f"Done: {out['simulated']} contest(s) simulated" + (f", {len(out['failed'])} failed" if out["failed"] else ""))
    return out
