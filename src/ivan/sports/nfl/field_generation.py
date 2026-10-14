# src/ivan/sports/nfl/field_generation.py
"""
The generate step: the pool of candidate lineups the simulator picks from.

generate_field() checks the dependence structure once in the parent (the
per-game correlation check), splits the count over worker processes that each
run generate_lineups_logic.generate_and_store_field_lineups on their own
session, then removes duplicate compositions. Every worker's jitter is seeded
from IVAN_SEED when set (ivan.core.seed), so a replay repeat is reproducible.
scripts/nfl/generate_lineups.py is argparse around it; the replay adapter
calls it in process.
"""
from __future__ import annotations

import json
import random
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from typing import Callable, List, Tuple

from ivan.core.seed import seed_for
from ivan.core.utils import coerce_player_ids
from ivan.database.session import get_session, init_db
from ivan.sports.nfl.generate_lineups_logic import generate_and_store_field_lineups
from ivan.sports.nfl.models import Lineup

Log = Callable[[str], None]
BATCH_COMMIT_SIZE = 100          # commit every n lineups


def quotas(n_lineups: int, jobs: int) -> List[int]:
    """n_lineups split over jobs workers, the remainder on the first ones."""
    jobs = max(1, int(jobs))
    base, rem = divmod(int(n_lineups), jobs)
    return [base + (1 if i < rem else 0) for i in range(jobs)]


def worker(worker_id: int, num_lineups: int, config_path: str, q: Queue) -> None:
    """One generator process: its own session and seed; progress messages on q."""
    session = get_session(init_db("nfl"))
    seed = seed_for(f"generate/{worker_id}")      # the jitter and shuffles in generate_lineups_logic use `random`
    if seed is not None:
        random.seed(seed)
    with open(config_path, "r") as f:
        config = json.load(f)
    q.put(("start", worker_id, num_lineups, ""))
    created = generate_and_store_field_lineups(
        session, config, num_lineups=num_lineups, batch_size=BATCH_COMMIT_SIZE,
        on_progress=lambda delta, note=None: q.put(("advance", worker_id, delta, note or "")))
    q.put(("done", worker_id, created, f"created={created}"))


def run_workers(quota: List[int], config_path: str, log: Log = print, progress: bool = True) -> int:
    """Run one worker per quota entry; a rich progress bar per worker when `progress`, else a line per start/finish. Returns lineups created."""
    q: Queue = Queue()
    procs = [Process(target=worker, args=(i, n, config_path, q)) for i, n in enumerate(quota)]
    for p in procs:
        p.start()
    created = 0
    bars = _ProgressBars() if progress else None
    done = 0
    while done < len(procs):
        try:
            kind, proc_no, val, note = q.get(timeout=0.25)
        except Empty:
            continue
        if kind == "start":
            if bars:
                bars.start(proc_no, int(val), note)
            else:
                log(f"[generate][{proc_no}] starting: {val} lineups")
        elif kind == "advance":
            if bars:
                bars.advance(proc_no, int(val))
        elif kind == "done":
            created += int(val)
            done += 1
            if bars:
                bars.done(proc_no, note)
            else:
                log(f"[generate][{proc_no}] finished ({note})")
    if bars:
        bars.stop()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    q.close()
    q.join_thread()
    return created


class _ProgressBars:
    """rich progress, one bar per worker."""

    def __init__(self):
        from rich.console import Console
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
        self.progress = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(),
                                 TextColumn("{task.percentage:>5.1f}%"), TimeRemainingColumn(), console=Console(),
                                 transient=False, expand=True, refresh_per_second=4)
        self.progress.start()
        self.tasks = {}

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
            total = self.progress.tasks[tid].total
            self.progress.update(tid, completed=total, description=f"Worker {proc_no} done {note}")

    def stop(self) -> None:
        self.progress.stop()


def deduplicate_lineups(session) -> Tuple[int, int]:
    """Remove Lineup rows that repeat an earlier row's player set (any order). Returns (removed, kept)."""
    seen = set()
    dupes = []
    for lu in session.query(Lineup).all():
        key = tuple(sorted(coerce_player_ids(lu.player_ids_list)))
        if key in seen:
            dupes.append(lu)
        else:
            seen.add(key)
    for d in dupes:
        session.delete(d)
    if dupes:
        session.commit()
    return len(dupes), len(seen)


def correlation_check(session, n_samples: int = 4000) -> str:
    """The per-game correlation check the generator prints before the workers draw their own samples."""
    from ivan.sports.nfl.correlated_samples import generate_correlated_samples_gaussian, summarize_sampling
    samples, diag = generate_correlated_samples_gaussian(session, n_samples=n_samples, return_diagnostics=True)
    return summarize_sampling(samples, diag)


def generate_field(n_lineups: int, jobs: int, config_path: str | Path, log: Log = print, progress: bool = True,
                   check_correlation: bool = True, session=None) -> dict:
    """
    The whole generate step: correlation check, workers, dedupe. `session` (the
    working database; opened from the environment when None) is used for the
    check and the dedupe; the workers open their own. Returns counts.
    """
    own = session is None
    session = session or get_session(init_db("nfl"))
    try:
        if check_correlation:
            log(correlation_check(session))
        created = run_workers(quotas(n_lineups, jobs), str(config_path), log, progress)
        log(f"generated {created:,} lineups across {max(1, int(jobs))} workers")
        session.expire_all()
        removed, kept = deduplicate_lineups(session)
        log(f"deduplicator: removed {removed} duplicate lineups, kept {kept} unique")
        return {"requested": int(n_lineups), "created": int(created), "removed": int(removed), "unique": int(kept)}
    finally:
        if own:
            session.close()
