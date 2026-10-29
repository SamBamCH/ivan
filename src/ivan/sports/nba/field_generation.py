# src/ivan/sports/nba/field_generation.py
"""
The generate step: the pool of candidate lineups the simulator picks from.

generate_field() splits the count over worker processes that each open their
own session, draw one correlated universe of player scores per sample
(pre_lock.player.generate_player_samples), and solve one classic lineup per
jittered sample (pre_lock.lineup_generation.generate_single_classic_lineup),
then removes duplicate compositions. Every worker's draws are seeded from
IVAN_SEED when set (ivan.core.seed), so a replay repeat is reproducible.
scripts/nba/generate_lineups.py is argparse around it.

Config (config/nba.json): generate.n_samples, generate.randomness,
generate.jitter, generate.min_fpts, generate.player_min_fpts; the solver
reads max_salary / max_salary_per_team / solver.* from the same file.
"""
from __future__ import annotations

import json
import random
import numpy as np
import time
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from typing import Callable, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.core.seed import seed_for
from ivan.core.utils import coerce_player_ids
from ivan.database.session import get_session, init_db
from ivan.sports.nba.models import Lineup, LineupPool, NBAPlayer
from ivan.sports.nba.pre_lock.lineup_generation import FieldPlayer, generate_single_classic_lineup
from ivan.sports.nba.pre_lock.player import generate_player_samples
from ivan.sports.nba.utils import _cfg, _coerce_positions

Log = Callable[[str], None]
BATCH_COMMIT_SIZE = 100
STALL_WINDOW = 300         # solves; a worker stops when fewer than STALL_MIN_NEW of the last STALL_WINDOW solves were new lineups
STALL_MIN_NEW = 6          # (2%): on a two-game slate 25,000 lineups do not exist and the dedupe would otherwise spin for hours
PROGRESS_EVERY = 30.0      # seconds between progress lines when the bars are off (the desk reads the log)
DEFAULTS = {"n_samples": 5000, "randomness": 0.25, "jitter": 0.05, "min_fpts": None, "min_fpts_frac": 0.9, "player_min_fpts": None}
# randomness scales each sampled world's spread around the mean projection: at 1.0 the lineup that is optimal in a
# random world sits a median 45 points (14%) below the mean-optimal lineup (2026-10-21 slate); at 0.25 it sits ~20
# below and the field's chalk runs 40-50%, like a real field. min_fpts_frac floors every lineup at that share of
# the mean-optimal total; min_fpts (absolute) overrides it when given.


def quotas(n_lineups: int, jobs: int) -> List[int]:
    """n_lineups split over jobs workers, the remainder on the first ones."""
    jobs = max(1, int(jobs))
    base, rem = divmod(int(n_lineups), jobs)
    return [base + (1 if i < rem else 0) for i in range(jobs)]


def generate_settings(config: dict, min_fpts: Optional[float] = None, randomness: Optional[float] = None) -> dict:
    s = {k: _cfg(config, f"generate.{k}", v) for k, v in DEFAULTS.items()}
    if s["player_min_fpts"] is None:                      # the older top-level key, then the solver's old floor
        s["player_min_fpts"] = _cfg(config, "player_min_fpts", 15.0)
    if min_fpts is not None:
        s["min_fpts"] = float(min_fpts)
    if randomness is not None:
        s["randomness"] = float(randomness)
    return s


def optimal_total(players: List[FieldPlayer], config: dict) -> Optional[float]:
    """The mean-projection optimal lineup's total: the anchor the pool is measured against."""
    means = {p.id: float(p.fpts_mean or 0.0) for p in players}
    ids = generate_single_classic_lineup(players=players, base_scores=means, config=config, max_salary_per_team=_cfg(config, "max_salary_per_team", 25000))
    return float(sum(means[i] for i in ids)) if ids else None


def player_pool(session: Session, player_min_fpts: float = 15.0) -> List[FieldPlayer]:
    """The solver's pool: every projected player above player_min_fpts, positions as a tuple."""
    out = []
    for rp in session.query(NBAPlayer).filter(NBAPlayer.fpts_mean > float(player_min_fpts)).all():
        team, opp = rp.team or "", rp.opponent
        out.append(FieldPlayer(id=int(rp.id), name=rp.name, salary=int(rp.salary), team=team, opponent=opp,
                               position=tuple(_coerce_positions(rp.position or [])), fpts_mean=float(rp.fpts_mean),
                               game_id="-".join(sorted([team.upper(), opp.upper()])) if team and opp else None,
                               gametime=rp.gametime))
    return out


def worker(worker_id: int, num_lineups: int, config_path: str, min_fpts: Optional[float], q: Queue, randomness: Optional[float] = None) -> None:
    """One generator process: its own session and seed; progress messages on q."""
    session = get_session(init_db("nba"))
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    s = generate_settings(config, min_fpts, randomness)
    seed = seed_for(f"generate/{worker_id}")
    if seed is not None:
        random.seed(seed)
    q.put(("start", worker_id, num_lineups, f"samples={s['n_samples']}"))
    try:
        dg = pool_draft_group(session)
        players = player_pool(session, s["player_min_fpts"])
        if not players:
            q.put(("done", worker_id, 0, "no players")); return
        if s["min_fpts"] is None and s.get("min_fpts_frac"):
            opt = optimal_total(players, config)
            if opt:
                s["min_fpts"] = float(s["min_fpts_frac"]) * opt
        samples = generate_player_samples(session, n_samples=int(s["n_samples"]), randomness=float(s["randomness"]), seed=seed)
        n_samples = int(s["n_samples"]); jitter = float(s["jitter"])
        max_team = _cfg(config, "max_salary_per_team", 25000)
        generated: List[List[int]] = []; seen = set(); pending: List[List[int]] = []
        cursor = worker_id * 1000; attempts = 0; max_attempts = num_lineups * 10
        window: List[int] = []                                 # 1 per attempt that found a new lineup, 0 otherwise, last STALL_WINDOW
        stalled = ""
        while len(generated) < num_lineups and attempts < max_attempts:
            attempts += 1
            k = cursor % n_samples; cursor += 1
            scores = {pid: float(vec[k]) * (1.0 + random.uniform(-jitter, jitter)) for pid, vec in samples.items()}
            ids = generate_single_classic_lineup(players=players, base_scores=scores, config=config, min_fpts=s["min_fpts"],
                                                 max_salary_per_team=max_team)
            key = tuple(sorted(ids)) if ids else None
            new_one = key is not None and key not in seen
            window.append(1 if new_one else 0)
            if len(window) > STALL_WINDOW:
                window.pop(0)
            if len(window) == STALL_WINDOW and sum(window) < STALL_MIN_NEW:
                stalled = f"stalled: {sum(window)} new lineups in the last {STALL_WINDOW} solves, the slate cannot fill the request"
                break
            if not new_one:
                continue
            seen.add(key); generated.append(ids); pending.append(ids)
            q.put(("advance", worker_id, 1, ""))
            if len(pending) >= BATCH_COMMIT_SIZE:
                _store(session, pending, dg); pending = []
        _store(session, pending, dg)
        q.put(("done", worker_id, len(generated), f"created={len(generated)} attempts={attempts}" + (f"; {stalled}" if stalled else "")))
    except Exception as e:   # noqa: BLE001 - the parent must hear from every worker
        q.put(("done", worker_id, 0, f"error: {type(e).__name__}: {e}"))
    finally:
        session.close()


def _store(session: Session, lineups: List[List[int]], draft_group_id: Optional[int] = None) -> None:
    if not lineups:
        return
    now = datetime.utcnow()
    for ids in lineups:
        session.add(Lineup(player_ids_list=list(ids), created_at=now, draft_group_id=draft_group_id))
    session.commit()


def pool_draft_group(session: Session) -> Optional[int]:
    row = session.query(NBAPlayer.draft_group_id).filter(NBAPlayer.draft_group_id.isnot(None)).first()
    return None if row is None else int(row[0])


def slate_lineups(session: Session, *columns):
    """The loaded slate's lineups (Lineup.draft_group_id = the pool's draft group): what generate, the field sampler and the
    sim work on. Other slates' lineups stay in the table (ingest.isolate_slate keeps the recent ones) and are never read
    as this slate's. Rows without a draft group (older than the column) count as the loaded slate's."""
    q = session.query(*columns) if columns else session.query(Lineup)
    dg = pool_draft_group(session)
    return q if dg is None else q.filter((Lineup.draft_group_id == dg) | (Lineup.draft_group_id.is_(None)))


def snapshot_pool(session: Session) -> Optional[int]:
    """Write (replace) the lineup_pool rows for the current pool's slate: what the exports read names and positions from."""
    dg = pool_draft_group(session)
    if dg is None:
        return None
    players = session.query(NBAPlayer).filter(NBAPlayer.draft_group_id == dg).all()
    lock = min((p.gametime for p in players if p.gametime), default=None)
    session.query(LineupPool).filter(LineupPool.draft_group_id == dg).delete(synchronize_session=False)
    session.bulk_insert_mappings(LineupPool, [dict(draft_group_id=dg, player_id=int(p.id), name=p.name, team=p.team, position=p.position,
                                                   salary=p.salary, lock=lock, written_at=datetime.now()) for p in players])
    session.commit()
    return dg


def run_workers(quota: List[int], config_path: str, min_fpts: Optional[float], log: Log = print, progress: bool = True,
                randomness: Optional[float] = None) -> int:
    """One worker per quota entry; a rich progress bar per worker when `progress`, else a line per start/finish. Returns lineups created."""
    q: Queue = Queue()
    procs = [Process(target=worker, args=(i, n, config_path, min_fpts, q, randomness)) for i, n in enumerate(quota)]
    for p in procs:
        p.start()
    created, done = 0, 0
    bars = _ProgressBars() if progress else None
    advanced = 0; total = sum(int(n) for n in quota); last_line = time.time(); stalled = 0
    while done < len(procs):
        try:
            kind, proc_no, val, note = q.get(timeout=0.25)
        except Empty:
            kind = None
        if kind == "start":
            if bars:
                bars.start(proc_no, int(val), note)
            else:
                log(f"[generate][{proc_no}] starting: {val} lineups ({note})")
        elif kind == "advance":
            advanced += int(val)
            if bars:
                bars.advance(proc_no, int(val))
        elif kind == "done":
            created += int(val); done += 1
            if "stalled" in note:
                stalled += 1
            if bars:
                bars.done(proc_no, note)
            else:
                log(f"[generate][{proc_no}] finished ({note})")
            if note.startswith("error") and bars:
                log(f"[generate][{proc_no}] {note}")
        if not bars and time.time() - last_line >= PROGRESS_EVERY:
            log(f"[generate] {advanced:,} / {total:,} lineups so far, {done}/{len(procs)} workers finished")
            last_line = time.time()
    if bars:
        bars.stop()
    if stalled:
        log(f"[generate] STOPPED EARLY: {stalled} of {len(procs)} workers ran out of new lineups; {created:,} of {total:,} made. "
            f"The slate is too small for that many unique lineups: ask for fewer, or raise randomness / lower min fpts to widen the pool.")
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    q.close(); q.join_thread()
    return created


class _ProgressBars:
    """rich progress, one bar per worker."""

    def __init__(self):
        from rich.console import Console
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
        self.progress = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(),
                                 TextColumn("{task.percentage:>5.1f}%"), TimeRemainingColumn(), console=Console(),
                                 transient=False, expand=True, refresh_per_second=4)
        self.progress.start(); self.tasks = {}

    def start(self, proc_no: int, total: int, note: str) -> None:
        self.tasks[proc_no] = self.progress.add_task(f"Worker {proc_no}" + (f" ({note})" if note else ""), total=total)

    def advance(self, proc_no: int, delta: int) -> None:
        tid = self.tasks.get(proc_no)
        if tid is None:
            tid = self.tasks[proc_no] = self.progress.add_task(f"Worker {proc_no}", total=delta)
        self.progress.advance(tid, delta)

    def done(self, proc_no: int, note: str) -> None:
        tid = self.tasks.get(proc_no)
        if tid is not None:
            self.progress.update(tid, completed=self.progress.tasks[tid].total, description=f"Worker {proc_no} done {note}")

    def stop(self) -> None:
        self.progress.stop()


def deduplicate_lineups(session: Session) -> Tuple[int, int]:
    """Remove Lineup rows that repeat an earlier row's player set (any order). Returns (removed, kept)."""
    seen = set(); dupes = []
    for lid, raw in slate_lineups(session, Lineup.id, Lineup.player_ids_list).order_by(Lineup.id).all():
        key = tuple(sorted(coerce_player_ids(raw)))
        if key in seen:
            dupes.append(int(lid))
        else:
            seen.add(key)
    for i in range(0, len(dupes), 900):
        session.query(Lineup).filter(Lineup.id.in_(dupes[i:i + 900])).delete(synchronize_session=False)
    if dupes:
        session.commit()
    return len(dupes), len(seen)


def clear_lineups(session: Session) -> Tuple[int, int]:
    """Empty the loaded slate's lineups and their simulation results (a fresh generate); other slates' stay. Returns the rows
    removed from each."""
    dg = pool_draft_group(session)
    if dg is None:
        n_sims = session.execute(text("DELETE FROM lineup_simulation_results")).rowcount
        n_lineups = session.execute(text("DELETE FROM lineup")).rowcount
    else:
        mine = "SELECT id FROM lineup WHERE draft_group_id = :d OR draft_group_id IS NULL"
        n_sims = session.execute(text(f"DELETE FROM lineup_simulation_results WHERE lineup_id IN ({mine})").bindparams(d=dg)).rowcount
        n_lineups = session.execute(text("DELETE FROM lineup WHERE draft_group_id = :d OR draft_group_id IS NULL").bindparams(d=dg)).rowcount
    session.commit()
    return int(n_lineups or 0), int(n_sims or 0)


def pool_report(session: Session, config: dict, log: Log = print) -> dict:
    """Where the pool sits against the mean-optimal lineup: median and p90 gap, share within 5/10%, top exposures."""
    players = player_pool(session, generate_settings(config)["player_min_fpts"])
    opt = optimal_total(players, config)
    means = {p.id: float(p.fpts_mean or 0.0) for p in players}; names = {p.id: p.name for p in players}
    L = [coerce_player_ids(x) for (x,) in slate_lineups(session, Lineup.player_ids_list).all()]
    if not L or not opt:
        return {}
    import numpy as np
    from collections import Counter
    tot = np.array([sum(means.get(i, 0.0) for i in l) for l in L])
    exp = Counter(i for l in L for i in l)
    rep = dict(optimal=round(opt, 1), lineups=len(L), median_gap=round(float(opt - np.median(tot)), 1), p90_gap=round(float(opt - np.percentile(tot, 90)), 1),
               within_5pct=round(float((tot >= 0.95 * opt).mean()), 2), within_10pct=round(float((tot >= 0.90 * opt).mean()), 2), players_used=len(exp),
               top_exposure=[(names.get(i, str(i)), round(100.0 * n / len(L), 1)) for i, n in exp.most_common(8)])
    log(f"pool vs the mean-optimal {opt:.1f}: median {rep['median_gap']} below, p90 {rep['p90_gap']} below, {100 * rep['within_10pct']:.0f}% within 10%, "
        f"{rep['players_used']} players used; top exposure " + ", ".join(f"{n.split()[-1]} {x:.0f}%" for n, x in rep["top_exposure"][:5]))
    return rep


def write_field_ownership(session: Session, config: dict, log: Log = print) -> dict:
    """
    Store a generic field's exposure as every player's ownership (not part of the generate step since 2026-10-28: it cost
    a public-field build after every generate). With the two-group sampler the field is the public one at
    public_field.ownership_field_size entries; the projection_model sampler draws field.cap entries from the pool.
    """
    from ivan.sports.nba.sample_field import DEFAULT_CONTEST, field_ownership, write_ownership
    try:
        if str(_cfg(config, "field.sampler", "projection_model")) == "two_group":
            from ivan.sports.nba.public_field import ownership_field_size
            size = ownership_field_size()
        else:
            size = int(_cfg(config, "field.cap", 10000))
        own = field_ownership(session, config, size, DEFAULT_CONTEST, rng=np.random.default_rng(seed_for("ownership")))
    except Exception as e:   # noqa: BLE001 - ownership is a by-product; the pool is the step
        log(f"ownership not written: {type(e).__name__}: {e}")
        return {}
    n = write_ownership(session, own)
    names = {int(i): nm for i, nm in session.query(NBAPlayer.id, NBAPlayer.name).all()}
    top = sorted(own.items(), key=lambda kv: -kv[1])[:6]
    log(f"ownership from the sampled field written for {n} players; top " + ", ".join(f"{names.get(pid, pid).split()[-1]} {x:.0f}%" for pid, x in top))
    return {int(pid): round(float(x), 2) for pid, x in own.items()}


def generate_field(n_lineups: int, jobs: int, config_path: str | Path, log: Log = print, progress: bool = True,
                   min_fpts: Optional[float] = None, session: Optional[Session] = None, randomness: Optional[float] = None) -> dict:
    """The whole generate step: workers, dedupe, the pool report (no field, no ownership). `session` (opened from the environment when None) is used for the dedupe. Returns counts."""
    own = session is None
    session = session or get_session(init_db("nba"))
    try:
        dg = snapshot_pool(session)                      # the exports read names and positions from this, not from the live pool
        log(f"pool snapshot for slate {dg}: {session.query(LineupPool).filter(LineupPool.draft_group_id == dg).count()} players" if dg else "no draft group on the pool")
        created = run_workers(quotas(n_lineups, jobs), str(config_path), min_fpts, log, progress, randomness)
        log(f"generated {created:,} lineups across {max(1, int(jobs))} workers")
        session.expire_all()
        removed, kept = deduplicate_lineups(session)
        log(f"deduplicator: removed {removed} duplicate lineups, kept {kept} unique")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        rep = pool_report(session, config, log)
        # no ownership write here (2026-10-28): it cost a public-field build after every generate; the sim writes the
        # column from the largest contest's field, the desk's Field + ownership button builds it on demand
        return {"requested": int(n_lineups), "created": int(created), "removed": int(removed), "unique": int(kept), "pool": rep}
    finally:
        if own:
            session.close()
