# src/ivan/sports/nba/contest/simulate.py
"""
Simulate every DraftKings classic contest in the entries file for every candidate lineup.

field     the public's lineups (public_field.slate_field: the two-group field on the public projection), built once
          per run at the largest size any contest needs and sliced per contest to min(field_cap, entries) - 1
payouts   the contest's payout table, scaled to the simulated field (ivan.sports.common.payouts.scale_payouts_correctly)
scores    one draw of every player's points per run (slate_draw -> gamesim.draw.generate_player_samples: the game
          simulation), so world w is the same games in every contest and the slate chooser can add the contests'
          results world by world; every candidate is scored against the field in every world (simulate_vs_field),
          ranking one below the field entries that outscore it and sharing the payout with its exact copies
results   per candidate (the best KEEP_RESULTS by mean ROI): fpts / payout / ROI / rank in every world, as
          lineup_simulation_results rows (the slate's earlier rows are replaced); a sim artifact
          (<artifact_dir>/contest_<id>.npz + .json, ivan.core.artifacts) with the field and its exposures; a report
          (contest_sim_report row + diagnostics/<contest>/report.json) with the field's shape and the checks

scripts/nba/sim.py is argparse around simulate_entries.
"""
from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core.artifacts import contest_artifact_paths, jsonable
from ivan.core.seed import seed_for
from ivan.sports.common.payouts import scale_payouts_correctly
from ivan.sports.nba.data_fetch import contests as C
from ivan.sports.nba.models import ContestPayout, ContestSimReport, Lineup, LineupSimulationResult, NBAPlayer
from ivan.sports.nba.paths import DIAG_DIR
from ivan.sports.nba.gamesim.draw import generate_player_samples
from ivan.sports.nba.field.public_field import field_params, slate_field, write_ownership

Log = Callable[[str], None]
REPORT_VERSION = 1


# ----------------------------------------------------------------------------- helpers
def _f(x) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def sanitize_contest_name(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    return re.sub(r"_+", "_", safe)[:80].strip("_")


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




# ----------------------------------------------------------------------------- scoring
def slate_draw(session: Session, n_samples: int, log: Log = print) -> dict:
    """
    The run's one draw of every player's points: {"raw": pid -> samples, "n": worlds, "seed"}. Drawn once per sim run
    (simulate_entries) and handed to every contest, so world w is the same games in each contest and the slate chooser
    can add the contests' results world by world (ivan.core.slate_book). Seeded from IVAN_SEED when set.
    """
    seed = seed_for("sim/slate")
    raw = generate_player_samples(session, n_samples=int(n_samples), randomness=1.0, seed=seed)
    ns = next((int(v.shape[0]) for v in raw.values() if isinstance(v, np.ndarray)), 0)
    log(f"one shared draw of {len(raw)} players x {ns} worlds for every contest of the run" + (f" (seed {seed})" if seed is not None else ""))
    return {"raw": raw, "n": ns, "seed": seed}


def player_samples_matrix(session: Session, pid_to_idx: Dict[int, int], n_samples: int, seed: Optional[int] = None, raw: Optional[dict] = None):
    """(P: simulations x players, raw: pid -> samples) from the correlated player sampler, or from the run's shared draw (`raw`)."""
    if raw is None:
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




# ----------------------------------------------------------------------------- one contest


# ----------------------------------------------------------------------------- artifact + report


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




# ----------------------------------------------------------------------------- every contest in an entries file
def contests_in_entries(entries_csv) -> pd.DataFrame:
    """The distinct contests of a DraftKings entries file (one row per contest id)."""
    from ivan.sports.nba.contest.export import read_entries
    df = read_entries(entries_csv)[["Contest ID", "Contest Name"]].copy()
    df["Contest Name"] = df["Contest Name"].str.strip()
    return df.drop_duplicates(subset=["Contest ID"], keep="last").sort_values("Contest ID").reset_index(drop=True)


def simulate_entries(session: Session, entries_csv, config_path, n_sims: int, field_cap: int,
                     artifact_dir: str | Path | None = None, log: Log = print) -> dict:   # config_path: kept for the callers; the field reads its own file
    """simulate_vs_field for every contest in the entries file; a failure is logged and the others still run."""
    contests = contests_in_entries(entries_csv)
    art = Path(artifact_dir) if artifact_dir is not None else None
    # every run starts from nothing for the loaded slate: its per-lineup results and contest reports of earlier runs are gone
    # (2026-10-26: a sim run must never sit beside stale results of another run); other slates' are kept for them
    from ivan.sports.nba.field.generate import pool_draft_group
    dg = pool_draft_group(session)
    if dg is None:
        n_old = session.query(LineupSimulationResult).delete(synchronize_session=False)
        n_rep = session.query(ContestSimReport).delete(synchronize_session=False)
    else:
        from sqlalchemy import text as _t
        n_old = session.execute(_t("DELETE FROM lineup_simulation_results WHERE lineup_id IN (SELECT id FROM lineup WHERE draft_group_id = :d OR draft_group_id IS NULL)"
                                   " OR lineup_id NOT IN (SELECT id FROM lineup)").bindparams(d=dg)).rowcount
        n_rep = session.execute(_t("DELETE FROM contest_sim_report WHERE contest_id IN (SELECT contest_id FROM historical_contest_information WHERE draft_group_id = :d)"
                                   " OR contest_id NOT IN (SELECT contest_id FROM historical_contest_information)").bindparams(d=dg)).rowcount
    session.commit()
    if n_old or n_rep:
        log(f"cleared the previous sim: {n_old:,} per-lineup results, {n_rep} contest reports")
    log(f"Simulating {len(contests)} contest(s): {n_sims} sims each, field cap {field_cap:,}" + (f"; run artifacts -> {art}" if art else ""))
    out = {"contests": int(len(contests)), "simulated": 0, "failed": [], "reports": {}, "skipped": []}
    biggest: dict = {}
    # an entries file of another slate's contests cannot be simulated on this pool: those contests are skipped (each contest's
    # draft group as stored, or fetched once from DraftKings when it is new)
    from ivan.sports.nba.field.generate import pool_draft_group
    dg_pool = pool_draft_group(session)
    if dg_pool is not None:
        keep, skipped = [], []
        for cid, name in zip(contests["Contest ID"], contests["Contest Name"]):
            try:
                c = C.ensure_contest(session, int(cid), log=log)
            except Exception:   # noqa: BLE001 - unknown: the contest's own run reports it
                c = None
            if c is not None and c.draft_group_id is not None and int(c.draft_group_id) != int(dg_pool):
                skipped.append(dict(contest_id=int(cid), name=name, draft_group=int(c.draft_group_id)))
            else:
                keep.append(int(cid))
        for x in skipped:
            log(f"SKIPPED contest {x['contest_id']} ('{x['name']}'): it belongs to slate {x['draft_group']}, the pool is slate {dg_pool}")
        out["skipped"] = skipped
        contests = contests[contests["Contest ID"].astype(int).isin(keep)].reset_index(drop=True)
    prebuild_field(session, contests["Contest ID"], field_cap, log=log)
    # one draw of the players' points for the whole run: every contest scores its field and candidates in the same worlds,
    # which is what lets the slate chooser read the contests' results as one book (world w = the same games everywhere)
    draw = slate_draw(session, n_sims, log=log) if len(contests) else None
    for i, (cid, name) in enumerate(zip(contests["Contest ID"], contests["Contest Name"]), 1):
        log(f"--- {i}/{len(contests)} ---")
        try:
            own: dict = {}
            report = simulate_vs_field(session, int(cid), name, field_cap=field_cap, num_sims=n_sims, log=log, own_out=own, artifact_dir=art, draw=draw)
            out["reports"][int(cid)] = report; out["simulated"] += 1
            if own.get("total_entries", 0) >= biggest.get("total_entries", 0):
                biggest = own
        except Exception as e:   # noqa: BLE001 - one contest failing must not lose the others
            import traceback
            session.rollback()
            out["failed"].append({"contest_id": int(cid), "name": name, "error": f"{type(e).__name__}: {e}"})
            log(f"FAILED contest {cid} ('{name}'): {e}\n{traceback.format_exc()}")
    if biggest:
        n = write_ownership(session, {k: v for k, v in biggest.items() if k != "total_entries"})
        log(f"ownership written for {n} players from the largest contest's field ({biggest['total_entries']:,} entries)")
    log(f"Done: {out['simulated']} contest(s) simulated" + (f", {len(out['failed'])} failed" if out["failed"] else ""))
    return out


def field_sizes(total_entries: List[int], field_cap: int) -> List[int]:
    """The field each contest is simulated against: min(cap, entries) less one (our entry), at least one."""
    return [max(1, min(int(field_cap or 10000), int(n)) - 1) for n in total_entries]


def prebuild_field(session: Session, contest_ids, field_cap: int, log: Log = print) -> int:
    """
    Build the public field ONCE at the largest size any contest in the run needs, so every contest slices the same cached
    field (slate_field caches by key and serves any smaller size from it). Before this (2026-10-27) a run built the field
    at the first contest's size and again, from scratch, when a bigger contest came (89s + 142s of a 403s sim).
    Returns the size built, 0 when nothing could be sized (the contests then build their own).
    """
    sizes = []
    for cid in contest_ids:
        try:
            c = C.ensure_contest(session, int(cid), log=log)
        except Exception as e:   # noqa: BLE001 - the contest's own simulation reports the failure
            log(f"{cid}: could not size the field ({type(e).__name__}: {e})"); c = None
        if c is not None:
            sizes.append(field_sizes([int(c.total_entries)], field_cap)[0])
    if not sizes:
        return 0
    n = max(sizes)
    try:
        slate_field(session, n, log=log)
    except Exception as e:   # noqa: BLE001
        log(f"public field not prebuilt at {n:,}: {type(e).__name__}: {e}"); return 0
    return n


# ----------------------------------------------------------------------------- the two-group field
# The field is its own set of lineups, generated from the fitted two-group model (field.model.simulate_field with
# data/nba/models/field_two_group.json) on the public projection: a quarter of the entrants take the optimal lineup on
# one of a few providers' projections through a little personal noise, the rest build loosely. It never touches the
# candidate table. Every candidate is then scored against it in every world: its rank is one more than the field
# entries that outscore it, and it splits the payout with any exact copies of itself in the field.
KEEP_RESULTS = 6000          # candidates whose per-world results are stored (the best by mean ROI): what the chooser reads
CAND_CHUNK = 4000


def pay_against(field_sorted: np.ndarray, scores: np.ndarray, copies: np.ndarray, csum: np.ndarray):
    """
    One world: (payout, rank) for each candidate score against the field's scores (ascending). The candidate ranks one
    below the field entries that outscore it; it shares the places with the field entries tied with it (at least its own
    exact copies), so it gets the mean payout of those places. csum[k] is the total payout of ranks 1..k.
    """
    n = len(field_sorted)
    right = np.searchsorted(field_sorted, scores, side="right"); left = np.searchsorted(field_sorted, scores, side="left")
    higher = n - right
    tie = np.maximum(right - left, copies) + 1
    top = np.minimum(higher + tie, n + 1)
    return (csum[top] - csum[higher]) / tie, higher + 1


def simulate_vs_field(session: Session, contest_id: int, contest_name: str, field_cap: int = 10000, num_sims: int = 1000,
                      log: Log = print, own_out: Optional[dict] = None, artifact_dir: str | Path | None = None,
                      draw: Optional[dict] = None) -> dict:
    """Score every candidate lineup against the two-group field; store per-world results for the best KEEP_RESULTS. Returns a report.
    With artifact_dir, writes contest_<id>.npz / .json there: the sampled field (unique lineups x copies), its player exposures and
    every candidate's mean ROI and cash rate, which the desk's exposures tab reads (a sampled field per simmed contest).
    `draw` is the run's shared player draw (slate_draw), so world w is the same games in every contest of the run; without it the
    contest draws its own (report contest.draw = "contest", which the slate chooser refuses)."""
    from collections import Counter
    from ivan.core.utils import coerce_player_ids
    t0 = time.time()
    contest = C.ensure_contest(session, int(contest_id), log=log)
    if contest is None:
        raise ValueError(f"contest {contest_id} is not in the database and DraftKings has nothing for it")
    if not payout_tiers(session, contest_id):
        C.fetch_payouts(session, [int(contest_id)], log=log)
    tiers = payout_tiers(session, contest_id)
    if not tiers:
        raise ValueError(f"no payouts for contest {contest_id}")
    field_sz = int(contest.total_entries); entry_fee = float(contest.entry_fee)
    n = field_sizes([field_sz], field_cap)[0]                                       # the field around one of our entries
    field = slate_field(session, n, log=log)[:n]
    n = len(field)
    from ivan.sports.nba.field.generate import slate_lineups
    cands = [(int(lid), coerce_player_ids(raw)) for lid, raw in slate_lineups(session, Lineup.id, Lineup.player_ids_list).all()]
    if not cands:
        raise ValueError("no lineups in the database; run the generate step first")
    copies = Counter(field)
    pids = sorted({p for lu in field for p in lu} | {p for _, ids in cands for p in ids})
    col = {pid: i for i, pid in enumerate(pids)}
    seed = draw["seed"] if draw else seed_for(f"sim/{contest_id}")
    P, ns, raw_samples, missing = player_samples_matrix(session, col, num_sims, seed=seed, raw=draw["raw"] if draw else None)
    if ns == 0:
        raise ValueError("the player sampler returned nothing; project the slate first")
    F = np.sort(P[:, np.array([[col[p] for p in lu] for lu in field])].sum(axis=2), axis=1)       # worlds x field, ascending
    scaled = scale_payouts(tiers, n + 1, field_sz, entry_fee)
    pay = np.zeros(n + 2)
    for r, a in scaled.items():
        if 1 <= r <= n + 1:
            pay[r] = float(a)
    csum = np.cumsum(pay)                                                           # csum[k] = pay for ranks 1..k
    tie_extra = np.array([copies.get(tuple(sorted(ids)), 0) for _, ids in cands])

    def score(idx):
        ix = np.array([[col[p] for p in cands[i][1]] for i in idx]); s = P[:, ix].sum(axis=2)        # worlds x chunk
        payout = np.zeros_like(s, dtype=np.float64); rank = np.zeros(s.shape, dtype=np.int32)
        extra = tie_extra[idx]
        for w in range(ns):
            payout[w], rank[w] = pay_against(F[w], s[w], extra, csum)
        return s, payout, rank

    means = np.zeros(len(cands)); cashes = np.zeros(len(cands))
    for a in range(0, len(cands), CAND_CHUNK):
        idx = np.arange(a, min(a + CAND_CHUNK, len(cands)))
        _, payout, _ = score(idx)
        means[idx] = (payout.mean(axis=0) - entry_fee) / entry_fee; cashes[idx] = (payout > 0).mean(axis=0)
    keep = np.argsort(-means)[:KEEP_RESULTS]
    session.query(LineupSimulationResult).filter(LineupSimulationResult.contest_id == int(contest_id)).delete(synchronize_session=False)
    session.commit()
    for a in range(0, len(keep), CAND_CHUNK):
        idx = keep[a:a + CAND_CHUNK]
        s, payout, rank = score(idx)
        roi = (payout - entry_fee) / entry_fee
        for j, i in enumerate(idx):
            session.add(LineupSimulationResult(lineup_id=int(cands[i][0]), contest_id=int(contest_id), contest_name=contest_name,
                                               fpts=s[:, j].astype(float).tolist(), payout=payout[:, j].tolist(), roi=roi[:, j].tolist(), rank=rank[:, j].tolist()))
        session.commit()
    exposure = Counter(p for lu in field for p in lu)
    if own_out is not None:
        own_out.update({int(pid): 100.0 * c / n for pid, c in exposure.items()}); own_out["total_entries"] = field_sz
    if artifact_dir is not None:
        uniq = list(copies.keys())
        arrays = dict(lineup_players=np.array([list(lu) for lu in uniq], dtype=np.int64).reshape(len(uniq), -1), dupes=np.array([copies[lu] for lu in uniq], dtype=np.int32),
                      player_id=np.array(sorted(exposure), dtype=np.int64), player_exposure_pct=np.array([100.0 * exposure[p] / n for p in sorted(exposure)]),
                      lineup_id=np.array([lid for lid, _ in cands], dtype=np.int64), avg_roi=means.astype(np.float32), cash_pct=cashes.astype(np.float32),
                      in_field=tie_extra.astype(np.int32))
        meta = {"version": REPORT_VERSION, "created_at": datetime.now().isoformat(timespec="seconds"), "seed": seed, "n_sims": int(ns), "simulated_size": int(n),
                "unique_lineups": len(uniq), "contest": {"id": int(contest_id), "name": contest_name, "entry_fee": entry_fee, "total_entries": field_sz,
                                                        "field_cap": int(field_cap or 0), "candidates": len(cands)}, "sampler": {"sampler": "two_group", **{k: v for k, v in field_params().items() if k != "notes"}}}
        npz, js = contest_artifact_paths(Path(artifact_dir), contest_id)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **arrays); js.write_text(json.dumps(jsonable(meta), indent=1, default=str))
    players = player_meta(session, list(exposure)[:2000])
    label = lambda pid: f"{players.get(pid, {}).get('name', pid)} ({players.get(pid, {}).get('team', '?')})"   # noqa: E731
    fs = np.array([sum(float(players.get(p, {}).get('fpts_mean') or 0.0) for p in lu) for lu in field])
    report = {"version": REPORT_VERSION, "created_at": datetime.now().isoformat(timespec="seconds"), "seed": seed, "seconds": round(time.time() - t0, 1),
              "contest": {"id": int(contest_id), "name": contest_name, "entry_fee": entry_fee, "total_entries": field_sz, "field_cap": int(field_cap or 0),
                          "multi_entry": int(contest.multi_entry or 150), "start": contest.date, "candidates": len(cands),
                          "draw": "slate" if draw else "contest"},
              "field": {"sampler": {"sampler": "two_group", **{k: v for k, v in field_params().items() if k != "notes"}}, "sampled": int(n),
                        "unique": len(copies), "dup_share": _f(sum(c for c in copies.values() if c > 1) / n), "max_copies": int(max(copies.values())),
                        "median_projection": _f(np.median(fs)), "players_without_projection": missing,
                        "top_exposure": [{"player": label(pid), "pct": _f(100.0 * c / n)} for pid, c in exposure.most_common(15)]},
              "simulation": {"n_sims": int(ns), "kept": int(len(keep)), "avg_roi": {f"p{p}": _f(np.percentile(means, p)) for p in (10, 50, 90, 99)},
                             "best_roi": _f(means[keep[0]]) if len(keep) else None, "best_cash": _f(cashes[keep[0]]) if len(keep) else None,
                             "candidates_in_field": int((tie_extra > 0).sum())},
              "checks": [{"name": "every field player projected", "ok": not missing, "detail": f"{len(missing)} missing"}]}
    save_report(session, report)
    log(f"{contest_name}: field {n:,} ({len(copies):,} unique, {report['field']['dup_share']:.0%} in duplicated lineups), "
        f"{len(cands):,} candidates scored, best mean ROI {report['simulation']['best_roi']:+.1%} in {report['seconds']:.0f}s")
    return report
