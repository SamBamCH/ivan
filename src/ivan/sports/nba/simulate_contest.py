# src/ivan/sports/nba/simulate_contest.py
"""
Simulate one DraftKings classic contest for every candidate lineup.

field     the field sampler (sample_field) draws `field_cap` entries from the
          generated pool; the candidates are the pool's unique lineups
payouts   the contest's payout table, scaled to the sampled field
          (ivan.sports.common.payouts.scale_payouts_correctly)
scores    one correlated draw per simulation for every player on the field
          (pre_lock.player.generate_player_samples: a Gaussian copula per
          game on each player's fitted marginal), lineups scored and ranked,
          ties sharing the pooled payout
results   per unique lineup: fpts / payout / ROI / rank in every simulation,
          as lineup_simulation_results rows (the contest's earlier rows are
          replaced); a sim artifact (<artifact_dir>/contest_<id>.npz + .json,
          ivan.core.artifacts) with what the field and the players looked
          like; and a report (contest_sim_report row + diagnostics/<contest>/
          report.json) with the field's shape, the payout scaling, the score
          distribution and the conservation checks.

simulate_entries runs it for every contest in a DraftKings entries file.
scripts/nba/sim.py is argparse around it.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core.artifacts import contest_artifact_paths
from ivan.core.seed import seed_for
from ivan.sports.common.payouts import scale_payouts_correctly
from ivan.sports.nba import contests as C
from ivan.sports.nba.models import ContestPayout, ContestSimReport, HistoricalContest, Lineup, LineupSimulationResult, NBAPlayer
from ivan.sports.nba.paths import DIAG_DIR
from ivan.sports.nba.pre_lock.player import generate_player_samples
from ivan.sports.nba.sample_field import expand_sampled_field, sample_field
from ivan.sports.nba.utils import _cfg

Log = Callable[[str], None]
MAX_WORKERS = 8
MAX_CHUNKS = 12
QGRID = (1, 5, 10, 25, 50, 75, 90, 95, 99)
REPORT_VERSION = 1
TOP_N = 10


# ----------------------------------------------------------------------------- helpers
def _f(x) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def jsonable(obj):
    """numpy scalars/arrays, tuples, datetimes and NaN -> plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return _f(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="seconds")
    if isinstance(obj, Path):
        return str(obj)
    return obj


def sanitize_contest_name(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    return re.sub(r"_+", "_", safe)[:80].strip("_")


def clean_old_diagnostics(diagnostics_root: str | Path = DIAG_DIR, days_to_keep: int = 7) -> int:
    """Remove per-contest diagnostic folders older than days_to_keep. Returns how many."""
    root = Path(diagnostics_root)
    if not root.exists():
        return 0
    cutoff = datetime.now() - timedelta(days=days_to_keep)
    removed = 0
    for d in root.iterdir():
        if d.is_dir() and datetime.fromtimestamp(d.stat().st_mtime) < cutoff:
            shutil.rmtree(d, ignore_errors=True); removed += 1
    return removed


def player_meta(session: Session, pids) -> Dict[int, dict]:
    pids = [int(p) for p in pids]
    meta: Dict[int, dict] = {}
    for i in range(0, len(pids), 900):
        for r in session.query(NBAPlayer).filter(NBAPlayer.id.in_(pids[i:i + 900])).all():
            meta[int(r.id)] = {"id": int(r.id), "name": r.name, "team": r.team, "opponent": r.opponent, "pos": r.position,
                               "salary": r.salary, "fpts_mean": _f(r.fpts_mean), "fpts_std": _f(r.fpts_std), "minutes_proj": _f(r.minutes_proj)}
    return meta


# ----------------------------------------------------------------------------- payouts
def payout_tiers(session: Session, contest_id: int) -> List[dict]:
    rows = (session.query(ContestPayout).filter(ContestPayout.contest_id == int(contest_id))
            .order_by(ContestPayout.min_position.asc()).all())
    return [{"min_position": int(r.min_position), "max_position": int(r.max_position), "payout": float(r.payout_amount)} for r in rows]


def scale_payouts(tiers: List[dict], num_field_lineups: int, original_field_size: int, entry_fee: float) -> Dict[int, float]:
    df = pd.DataFrame([(t["min_position"], t["max_position"], t["payout"]) for t in tiers], columns=["Start Rank", "End Rank", "Payout"])
    return scale_payouts_correctly(num_field_lineups=int(num_field_lineups), payouts_df=df,
                                   original_field_size=int(original_field_size), entry_fee=float(entry_fee))


def load_scaled_payouts(session: Session, contest_id: int, num_field_lineups: int, original_field_size: int, entry_fee: float) -> Dict[int, float]:
    tiers = payout_tiers(session, contest_id)
    if not tiers:
        raise ValueError(f"no payouts stored for contest {contest_id}")
    return scale_payouts(tiers, num_field_lineups, original_field_size, entry_fee)


# ----------------------------------------------------------------------------- scoring
def player_samples_matrix(session: Session, pid_to_idx: Dict[int, int], n_samples: int, seed: Optional[int] = None):
    """(P: simulations x players, raw: pid -> samples) from the correlated player sampler."""
    raw = generate_player_samples(session, n_samples=int(n_samples), randomness=1.0, seed=seed)
    ns = next((int(v.shape[0]) for v in raw.values() if isinstance(v, np.ndarray)), 0)
    P = np.zeros((ns, len(pid_to_idx)), dtype=np.float32)
    missing = []
    for pid, col in pid_to_idx.items():
        v = raw.get(pid)
        if not isinstance(v, np.ndarray):
            missing.append(int(pid)); continue
        vv = v.astype(np.float32, copy=False)
        if vv.shape[0] < ns:
            vv = np.tile(vv, int(np.ceil(ns / vv.shape[0])))[:ns]
        P[:, col] = vv[:ns]
    return P, ns, raw, missing


def _simulate_chunk(start: int, end: int, lineup_ix: List[List[int]], P_chunk: np.ndarray, entry_fee: float,
                    scaled_payouts: Dict[int, float]) -> Dict[str, np.ndarray]:
    """Score every field entry in simulations start..end, rank them, pay the tiers (ties share the pooled payout)."""
    n = end - start; M = len(lineup_ix)
    scores = np.zeros((n, M), dtype=np.float64); payouts = np.zeros_like(scores); ranks = np.zeros((n, M), dtype=np.int32)
    pay_arr = np.zeros(M, dtype=np.float64)
    for r, a in scaled_payouts.items():
        if 1 <= r <= M:
            pay_arr[r - 1] = float(a)
    pay_csum = np.cumsum(pay_arr)
    pool = float(pay_arr.sum()); sims_off = 0
    # lineup x player membership as a padded index matrix: one matmul-free gather per simulation
    width = max((len(ix) for ix in lineup_ix), default=0)
    ix = np.zeros((M, max(width, 1)), dtype=np.int64); mask = np.zeros((M, max(width, 1)), dtype=np.float64)
    for m, idxs in enumerate(lineup_ix):
        ix[m, :len(idxs)] = idxs; mask[m, :len(idxs)] = 1.0
    for i in range(n):
        row = P_chunk[i].astype(np.float64)
        scores[i] = (row[ix] * mask).sum(axis=1)
        order = np.argsort(-scores[i], kind="stable")
        sc = scores[i][order]
        edges = np.concatenate(([0], np.nonzero(np.abs(np.diff(sc)) > 1e-8)[0] + 1, [M]))
        start_rank = edges[:-1] + 1; end_rank = edges[1:]
        sizes = end_rank - start_rank + 1
        pools = pay_csum[end_rank - 1] - np.where(start_rank - 2 >= 0, pay_csum[np.maximum(start_rank - 2, 0)], 0.0)
        per_entry = np.divide(pools, sizes, out=np.zeros_like(pools, dtype=np.float64), where=sizes > 0)
        for g in range(len(start_rank)):
            idx = order[edges[g]:edges[g + 1]]
            payouts[i, idx] = per_entry[g]; ranks[i, idx] = int(start_rank[g])
        if abs(payouts[i].sum() - pool) > 1e-6 * max(pool, 1.0):
            sims_off += 1
    rois = (payouts - entry_fee) / entry_fee
    return {"start": start, "end": end, "scores": scores.astype(np.float32), "payouts": payouts.astype(np.float32),
            "rois": rois.astype(np.float32), "ranks": ranks, "stats": {"sims_off": sims_off, "pool": pool}}


# ----------------------------------------------------------------------------- one contest
def simulate_one_contest(session: Session, contest_id: int, contest_name: str, config_path: str, field_cap: int = 10000,
                         num_sims: int = 1000, artifact_dir: str | Path | None = None, log: Log = print) -> dict:
    """Sample the field, score it, store the results, the artifact and the report. Returns the report."""
    t0 = time.time()
    cfg = json.load(open(config_path, encoding="utf-8"))
    contest = C.ensure_contest(session, int(contest_id), log=log)
    if contest is None:
        raise ValueError(f"contest {contest_id} is not in the database and DraftKings has nothing for it")
    if not payout_tiers(session, contest_id):
        C.fetch_payouts(session, [int(contest_id)], log=log)
    field_sz = int(contest.total_entries); entry_fee = float(contest.entry_fee)
    capped = min(int(field_cap or 10000), field_sz)
    n_candidates = session.query(Lineup).count()
    if n_candidates == 0:
        raise ValueError("no lineups in the database; run the generate step first")
    contest_params = {"entry_fee": entry_fee, "total_entries": field_sz, "multi_entry": int(contest.multi_entry or 150)}
    rng = np.random.default_rng(seed_for(f"field/{contest_id}"))
    counts, field_meta = sample_field(session, capped, cfg, contest_params, rng=rng)
    u_lids, pid_lists_by_lid, lineup_ix, unique_pids, pid_to_idx, starts, ends = expand_sampled_field(session, counts)
    M = len(lineup_ix); U = len(u_lids)
    tiers = payout_tiers(session, contest_id)
    if not tiers:
        raise ValueError(f"no payouts for contest {contest_id}")
    scaled = scale_payouts(tiers, M, field_sz, entry_fee)
    pay_by_rank = np.array([float(scaled.get(r, 0.0)) for r in range(1, M + 1)])
    paid_places = int(np.count_nonzero(pay_by_rank))
    seed = seed_for(f"sim/{contest_id}")
    P, ns, raw_samples, missing = player_samples_matrix(session, pid_to_idx, num_sims, seed=seed)
    if ns == 0:
        raise ValueError("the player sampler returned nothing; project the slate first")
    if missing:
        log(f"WARNING: {len(missing)} player id(s) on the field have no projection; they score 0")
    chunk = math.ceil(ns / min(MAX_CHUNKS, ns))
    full_scores = np.zeros((ns, M), dtype=np.float32); full_payouts = np.zeros_like(full_scores)
    full_rois = np.zeros_like(full_scores); full_ranks = np.zeros((ns, M), dtype=np.int32)
    sims_off = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futs = [exe.submit(_simulate_chunk, s, min(s + chunk, ns), lineup_ix, P[s:min(s + chunk, ns)], entry_fee, scaled)
                for s in range(0, ns, chunk)]
        for fut in as_completed(futs):
            part = fut.result(); a, b = part["start"], part["end"]
            full_scores[a:b] = part["scores"]; full_payouts[a:b] = part["payouts"]; full_rois[a:b] = part["rois"]; full_ranks[a:b] = part["ranks"]
            sims_off += part["stats"]["sims_off"]

    # results: one row per unique lineup, the copies averaged (the payout summed, as the copies share)
    session.query(LineupSimulationResult).filter(LineupSimulationResult.contest_id == int(contest_id)).delete(synchronize_session=False)
    session.commit()
    cash_pct = np.zeros(U); avg_roi = np.zeros(U); mean_rank = np.zeros(U); mean_score = np.zeros(U); win_pct = np.zeros(U)
    dupes = np.asarray([int(ends[u] - starts[u]) for u in range(U)], dtype=np.int64)
    for u, lid in enumerate(u_lids):
        a, b = int(starts[u]), int(ends[u])
        cash_pct[u] = (full_payouts[:, a] > 0).mean(); avg_roi[u] = full_rois[:, a:b].mean()
        mean_rank[u] = full_ranks[:, a:b].mean(); mean_score[u] = full_scores[:, a:b].mean()
        win_pct[u] = (full_ranks[:, a:b] == 1).any(axis=1).mean()
        session.add(LineupSimulationResult(lineup_id=int(lid), contest_id=int(contest_id), contest_name=contest_name,
                                           fpts=full_scores[:, a:b].mean(axis=1).tolist(), payout=full_payouts[:, a:b].sum(axis=1).tolist(),
                                           roi=full_rois[:, a:b].mean(axis=1).tolist(), rank=full_ranks[:, a:b].mean(axis=1).tolist()))
    session.commit()

    players = player_meta(session, unique_pids)
    exposure: Dict[int, int] = {}
    for u, lid in enumerate(u_lids):
        for pid in pid_lists_by_lid.get(lid, []):
            exposure[pid] = exposure.get(pid, 0) + int(dupes[u])
    contest_info = {"id": int(contest_id), "name": contest_name, "entry_fee": entry_fee, "total_entries": field_sz, "field_cap": int(field_cap or 0),
                    "multi_entry": contest_params["multi_entry"], "start": contest.date, "candidates": n_candidates}
    if artifact_dir is not None:
        arrays, meta = build_sim_artifact(contest=contest_info, u_lids=u_lids, dupes=dupes, pid_lists_by_lid=pid_lists_by_lid,
                                          cash_pct=cash_pct, avg_roi=avg_roi, mean_rank=mean_rank, mean_score=mean_score, win_pct=win_pct,
                                          full_scores=full_scores, raw_samples=raw_samples, players=players, exposure=exposure, M=M,
                                          pay_by_rank=pay_by_rank, tiers=tiers, paid_places=paid_places, field_meta=field_meta, seed=seed)
        npz, js = contest_artifact_paths(Path(artifact_dir), contest_id)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **arrays); js.write_text(json.dumps(jsonable(meta), indent=1, default=str))
    report = build_report(contest=contest_info, field_meta=field_meta, u_lids=u_lids, dupes=dupes, pid_lists_by_lid=pid_lists_by_lid,
                          players=players, exposure=exposure, M=M, tiers=tiers, scaled=scaled, pay_by_rank=pay_by_rank, paid_places=paid_places,
                          full_scores=full_scores, full_payouts=full_payouts, cash_pct=cash_pct, avg_roi=avg_roi, win_pct=win_pct,
                          sims_off=sims_off, n_sims=ns, missing=missing, seed=seed, seconds=time.time() - t0)
    save_report(session, report)
    return report


# ----------------------------------------------------------------------------- artifact + report
def build_sim_artifact(*, contest, u_lids, dupes, pid_lists_by_lid, cash_pct, avg_roi, mean_rank, mean_score, win_pct, full_scores,
                       raw_samples, players, exposure, M, pay_by_rank, tiers, paid_places, field_meta, seed):
    width = max((len(pid_lists_by_lid.get(int(l), [])) for l in u_lids), default=8)
    lineup_players = np.full((len(u_lids), width), -1, dtype=np.int64)
    for i, l in enumerate(u_lids):
        ids = pid_lists_by_lid.get(int(l), []); lineup_players[i, :len(ids)] = ids
    pids = sorted(set(int(p) for p in raw_samples) | set(exposure))
    q = np.full((len(pids), len(QGRID)), np.nan); mean = np.full(len(pids), np.nan); sd = np.full(len(pids), np.nan)
    for i, pid in enumerate(pids):
        v = raw_samples.get(pid)
        if isinstance(v, np.ndarray) and v.size:
            q[i] = np.percentile(v, QGRID); mean[i] = v.mean(); sd[i] = v.std()
    exp_pct = np.asarray([100.0 * exposure.get(pid, 0) / max(1, M) for pid in pids])
    score_pct = np.percentile(full_scores, QGRID, axis=1).T.astype(np.float32)
    sorted_scores = -np.sort(-full_scores, axis=1)
    cash_line = sorted_scores[:, min(max(paid_places, 1), M) - 1] if M else np.zeros(full_scores.shape[0])
    arrays = {"lineup_id": np.asarray(u_lids, dtype=np.int64), "dupes": dupes, "cash_pct": cash_pct, "avg_roi": avg_roi, "mean_rank": mean_rank,
              "mean_score": mean_score, "win_pct": win_pct, "lineup_players": lineup_players,
              "player_id": np.asarray(pids, dtype=np.int64), "player_q": q, "player_mean": mean, "player_sd": sd, "player_exposure_pct": exp_pct,
              "score_pct": score_pct, "cash_line": cash_line.astype(np.float32), "winning_score": full_scores.max(axis=1).astype(np.float32),
              "pay_by_rank": np.asarray(pay_by_rank, dtype=np.float64), "qgrid": np.asarray(QGRID, dtype=np.int64)}
    meta = {"version": 1, "created_at": datetime.now().isoformat(timespec="seconds"), "seed": seed, "contest": contest,
            "n_sims": int(full_scores.shape[0]), "simulated_size": int(M), "unique_lineups": int(len(u_lids)), "paid_places": int(paid_places),
            "tiers": tiers, "players": [{"id": pid, **{k: players.get(pid, {}).get(k) for k in ("name", "team", "pos", "salary", "fpts_mean")}} for pid in pids],
            "sampler": field_meta}
    return arrays, meta


def build_report(*, contest, field_meta, u_lids, dupes, pid_lists_by_lid, players, exposure, M, tiers, scaled, pay_by_rank, paid_places,
                 full_scores, full_payouts, cash_pct, avg_roi, win_pct, sims_off, n_sims, missing, seed, seconds) -> dict:
    def label(pid):
        m = players.get(pid, {})
        return f"{m.get('name', pid)} ({m.get('team', '?')} {m.get('pos', '?')})"
    top_exp = sorted(exposure.items(), key=lambda kv: -kv[1])[:15]
    pool_orig = sum((t["max_position"] - t["min_position"] + 1) * t["payout"] for t in tiers)
    pool_scaled = float(pay_by_rank.sum())
    paid_per_sim = full_payouts.sum(axis=1)
    best = np.argsort(-avg_roi)[:TOP_N]
    pct = lambda a: {f"p{p}": _f(np.percentile(a, p)) for p in (10, 50, 90)}   # noqa: E731
    report = {
        "version": REPORT_VERSION, "created_at": datetime.now().isoformat(timespec="seconds"), "seed": seed, "seconds": round(seconds, 1),
        "contest": contest,
        "field": {"sampled": int(M), "unique": int(len(u_lids)), "avg_copies": _f(M / max(1, len(u_lids))), "max_copies": int(dupes.max()) if len(dupes) else 0,
                  "sampler": field_meta, "players_without_projection": missing,
                  "top_exposure": [{"player": label(pid), "pct": _f(100.0 * n / max(1, M))} for pid, n in top_exp]},
        "payouts": {"tiers": len(tiers), "pool_original": _f(pool_orig), "pool_scaled": _f(pool_scaled), "paid_places": int(paid_places),
                    "paid_share": _f(paid_places / max(1, M)), "first_place": _f(pay_by_rank[0]) if M else None,
                    "min_cash": _f(min((a for a in scaled.values() if a > 0), default=0.0)), "rake": _f(1 - pool_orig / (contest["total_entries"] * contest["entry_fee"]))
                    if contest["entry_fee"] else None},
        "simulation": {"n_sims": int(n_sims), "score": pct(full_scores), "winning_score": pct(full_scores.max(axis=1)),
                       "cash_line": pct((-np.sort(-full_scores, axis=1))[:, min(max(paid_places, 1), M) - 1]) if M else None,
                       "cash_pct": pct(cash_pct), "avg_roi": pct(avg_roi),
                       "top_lineups": [{"lineup_id": int(u_lids[i]), "avg_roi": _f(avg_roi[i]), "cash_pct": _f(cash_pct[i]), "win_pct": _f(win_pct[i]),
                                        "copies": int(dupes[i]), "players": [label(p) for p in pid_lists_by_lid.get(int(u_lids[i]), [])]} for i in best]},
        "checks": [
            {"name": "pool conserved every simulation", "ok": sims_off == 0, "detail": f"{sims_off} of {n_sims} simulations paid a different total"},
            {"name": "payout per simulation equals the scaled pool", "ok": bool(np.allclose(paid_per_sim, pool_scaled, rtol=1e-4)),
             "detail": f"paid {paid_per_sim.mean():.2f} on average vs pool {pool_scaled:.2f}"},
            {"name": "mean ROI of the field is minus the rake", "ok": bool(abs(full_payouts.mean() / contest["entry_fee"] - pool_scaled / (M * contest["entry_fee"])) < 1e-3)
             if contest["entry_fee"] and M else True, "detail": f"field mean ROI {(full_payouts.mean() / contest['entry_fee'] - 1):+.3f}" if contest["entry_fee"] else ""},
            {"name": "every field player projected", "ok": not missing, "detail": f"{len(missing)} missing"},
        ],
    }
    return report


def save_report(session: Session, report: dict, diag_dir: str | Path = DIAG_DIR) -> Path:
    cid = int(report["contest"]["id"]); name = str(report["contest"]["name"])
    doc = jsonable(report)
    row = session.get(ContestSimReport, cid)
    if row is None:
        session.add(ContestSimReport(contest_id=cid, contest_name=name, report=doc))
    else:
        row.contest_name = name; row.report = doc; row.created_at = datetime.utcnow()
    session.commit()
    out = Path(diag_dir) / f"{sanitize_contest_name(name)}_{cid}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(doc, indent=1))
    return out / "report.json"


def format_summary(report: dict) -> str:
    c, f, p, s = report["contest"], report["field"], report["payouts"], report["simulation"]
    lines = [f"{c['name']} (#{c['id']}): fee ${c['entry_fee']:.2f}, {c['total_entries']:,} entries, simulated {f['sampled']:,} "
             f"({f['unique']:,} unique, {f['avg_copies']:.1f}x copies) over {s['n_sims']} sims in {report['seconds']}s",
             f"  payouts: {p['paid_places']} paid ({100 * p['paid_share']:.1f}%), first ${p['first_place']:,.0f}, min cash ${p['min_cash']:.2f}, pool ${p['pool_scaled']:,.0f}",
             f"  scores: median {s['score']['p50']:.1f}, cash line p50 {s['cash_line']['p50'] if s['cash_line'] else float('nan'):.1f}, winning p50 {s['winning_score']['p50']:.1f}",
             f"  lineups: cash % p50 {100 * s['cash_pct']['p50']:.1f} (p90 {100 * s['cash_pct']['p90']:.1f}), ROI p50 {s['avg_roi']['p50']:+.2f} (p90 {s['avg_roi']['p90']:+.2f})",
             "  top exposure: " + ", ".join(f"{t['player']} {t['pct']:.0f}%" for t in f["top_exposure"][:6])]
    for chk in report["checks"]:
        lines.append(f"  [{'ok' if chk['ok'] else 'FAIL'}] {chk['name']}: {chk['detail']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- every contest in an entries file
def contests_in_entries(entries_csv) -> pd.DataFrame:
    """The distinct contests of a DraftKings entries file (one row per contest id)."""
    from ivan.sports.nba.lineup_export import read_entries
    df = read_entries(entries_csv)[["Contest ID", "Contest Name"]].copy()
    df["Contest Name"] = df["Contest Name"].str.strip()
    return df.drop_duplicates(subset=["Contest ID"], keep="last").sort_values("Contest ID").reset_index(drop=True)


def simulate_entries(session: Session, entries_csv, config_path, n_sims: int, field_cap: int,
                     artifact_dir: str | Path | None = None, log: Log = print) -> dict:
    """simulate_one_contest for every contest in the entries file; a failure is logged and the others still run."""
    contests = contests_in_entries(entries_csv)
    art = Path(artifact_dir) if artifact_dir is not None else None
    log(f"Simulating {len(contests)} contest(s): {n_sims} sims each, field cap {field_cap:,}" + (f"; run artifacts -> {art}" if art else ""))
    out = {"contests": int(len(contests)), "simulated": 0, "failed": [], "reports": {}}
    for i, (cid, name) in enumerate(zip(contests["Contest ID"], contests["Contest Name"]), 1):
        log(f"--- {i}/{len(contests)} ---")
        try:
            report = simulate_one_contest(session, int(cid), name, str(config_path), field_cap=field_cap, num_sims=n_sims, artifact_dir=art, log=log)
            out["reports"][int(cid)] = report; out["simulated"] += 1
            log(format_summary(report))
        except Exception as e:   # noqa: BLE001 - one contest failing must not lose the others
            import traceback
            session.rollback()
            out["failed"].append({"contest_id": int(cid), "name": name, "error": f"{type(e).__name__}: {e}"})
            log(f"FAILED contest {cid} ('{name}'): {e}\n{traceback.format_exc()}")
    log(f"Done: {out['simulated']} contest(s) simulated" + (f", {len(out['failed'])} failed" if out["failed"] else ""))
    return out
