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
from ivan.sports.nba.models import Lineup, NBAPlayer
from ivan.sports.nba.pre_lock.lineup_generation import FieldPlayer, generate_single_classic_lineup
from ivan.sports.nba.pre_lock.player import generate_player_samples
from ivan.sports.nba.utils import _cfg, _coerce_positions

Log = Callable[[str], None]
BATCH_COMMIT_SIZE = 100
DEFAULTS = {"n_samples": 5000, "randomness": 1.0, "jitter": 0.05, "min_fpts": None, "player_min_fpts": None}


def quotas(n_lineups: int, jobs: int) -> List[int]:
    """n_lineups split over jobs workers, the remainder on the first ones."""
    jobs = max(1, int(jobs))
    base, rem = divmod(int(n_lineups), jobs)
    return [base + (1 if i < rem else 0) for i in range(jobs)]


def generate_settings(config: dict, min_fpts: Optional[float] = None) -> dict:
    s = {k: _cfg(config, f"generate.{k}", v) for k, v in DEFAULTS.items()}
    if s["player_min_fpts"] is None:                      # the older top-level key, then the solver's old floor
        s["player_min_fpts"] = _cfg(config, "player_min_fpts", 15.0)
    if min_fpts is not None:
        s["min_fpts"] = float(min_fpts)
    return s


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


def worker(worker_id: int, num_lineups: int, config_path: str, min_fpts: Optional[float], q: Queue) -> None:
    """One generator process: its own session and seed; progress messages on q."""
    session = get_session(init_db("nba"))
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    s = generate_settings(config, min_fpts)
    seed = seed_for(f"generate/{worker_id}")
    if seed is not None:
        random.seed(seed)
    q.put(("start", worker_id, num_lineups, f"samples={s['n_samples']}"))
    try:
        players = player_pool(session, s["player_min_fpts"])
        if not players:
            q.put(("done", worker_id, 0, "no players")); return
        samples = generate_player_samples(session, n_samples=int(s["n_samples"]), randomness=float(s["randomness"]), seed=seed)
        n_samples = int(s["n_samples"]); jitter = float(s["jitter"])
        max_team = _cfg(config, "max_salary_per_team", 25000)
        generated: List[List[int]] = []; seen = set(); pending: List[List[int]] = []
        cursor = worker_id * 1000; attempts = 0; max_attempts = num_lineups * 10
        while len(generated) < num_lineups and attempts < max_attempts:
            attempts += 1
            k = cursor % n_samples; cursor += 1
            scores = {pid: float(vec[k]) * (1.0 + random.uniform(-jitter, jitter)) for pid, vec in samples.items()}
            ids = generate_single_classic_lineup(players=players, base_scores=scores, config=config, min_fpts=s["min_fpts"],
                                                 max_salary_per_team=max_team)
            if not ids:
                continue
            key = tuple(sorted(ids))
            if key in seen:
                continue
            seen.add(key); generated.append(ids); pending.append(ids)
            q.put(("advance", worker_id, 1, ""))
            if len(pending) >= BATCH_COMMIT_SIZE:
                _store(session, pending); pending = []
        _store(session, pending)
        q.put(("done", worker_id, len(generated), f"created={len(generated)} attempts={attempts}"))
    except Exception as e:   # noqa: BLE001 - the parent must hear from every worker
        q.put(("done", worker_id, 0, f"error: {type(e).__name__}: {e}"))
    finally:
        session.close()


def _store(session: Session, lineups: List[List[int]]) -> None:
    if not lineups:
        return
    now = datetime.utcnow()
    for ids in lineups:
        session.add(Lineup(player_ids_list=list(ids), created_at=now))
    session.commit()


def run_workers(quota: List[int], config_path: str, min_fpts: Optional[float], log: Log = print, progress: bool = True) -> int:
    """One worker per quota entry; a rich progress bar per worker when `progress`, else a line per start/finish. Returns lineups created."""
    q: Queue = Queue()
    procs = [Process(target=worker, args=(i, n, config_path, min_fpts, q)) for i, n in enumerate(quota)]
    for p in procs:
        p.start()
    created, done = 0, 0
    bars = _ProgressBars() if progress else None
    while done < len(procs):
        try:
            kind, proc_no, val, note = q.get(timeout=0.25)
        except Empty:
            continue
        if kind == "start":
            if bars:
                bars.start(proc_no, int(val), note)
            else:
                log(f"[generate][{proc_no}] starting: {val} lineups ({note})")
        elif kind == "advance":
            if bars:
                bars.advance(proc_no, int(val))
        elif kind == "done":
            created += int(val); done += 1
            if bars:
                bars.done(proc_no, note)
            else:
                log(f"[generate][{proc_no}] finished ({note})")
            if note.startswith("error") and bars:
                log(f"[generate][{proc_no}] {note}")
    if bars:
        bars.stop()
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
    for lid, raw in session.query(Lineup.id, Lineup.player_ids_list).order_by(Lineup.id).all():
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
    """Empty the working tables (simulation results, then lineups). Returns the rows removed from each."""
    n_sims = session.execute(text("DELETE FROM lineup_simulation_results")).rowcount
    n_lineups = session.execute(text("DELETE FROM lineup")).rowcount
    session.commit()
    return int(n_lineups or 0), int(n_sims or 0)


def generate_field(n_lineups: int, jobs: int, config_path: str | Path, log: Log = print, progress: bool = True,
                   min_fpts: Optional[float] = None, session: Optional[Session] = None) -> dict:
    """The whole generate step: workers, dedupe. `session` (opened from the environment when None) is used for the dedupe. Returns counts."""
    own = session is None
    session = session or get_session(init_db("nba"))
    try:
        created = run_workers(quotas(n_lineups, jobs), str(config_path), min_fpts, log, progress)
        log(f"generated {created:,} lineups across {max(1, int(jobs))} workers")
        session.expire_all()
        removed, kept = deduplicate_lineups(session)
        log(f"deduplicator: removed {removed} duplicate lineups, kept {kept} unique")
        return {"requested": int(n_lineups), "created": int(created), "removed": int(removed), "unique": int(kept)}
    finally:
        if own:
            session.close()
