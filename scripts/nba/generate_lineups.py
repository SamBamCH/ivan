import json, os, argparse, sys, re

from multiprocessing import Process, freeze_support, Queue
from queue import Empty
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn,MofNCompleteColumn

from ivan.database.session import init_db, get_session
from ivan.sports.nba.lineup_generation import generate_and_store_field_lineups

from ivan.sports.nba.models import Lineup
from ivan.sports.nba.utils import _coerce_player_ids

BATCH_COMMIT_SIZE=100 #how frequently we write to the DB, every 100 lineups if this is set to 100 for example. 

def worker(worker_id: int, num_lineups: int, config_path: str, total_workers: int, q: Queue):
    engine = init_db('nba')
    session = get_session(engine)

    with open(config_path, "r") as f:
        config = json.load(f)

    q.put(("start", worker_id, num_lineups, "seed=42"))

    def on_progress(delta: int, note: str | None = None):
        q.put(("advance", worker_id, delta, note or ""))

    created = generate_and_store_field_lineups(
        session, 
        config,
        num_lineups=num_lineups,
        batch_size=BATCH_COMMIT_SIZE,
        on_progress=on_progress
    )

    q.put(("done", worker_id, created, f"created={created}"))

def _run_with_progress(quotas, config_path, total_workers):
    q = Queue()
    procs = []
    for proc_no, qnty in enumerate(quotas):
        p = Process(
            target=worker,
            args=(proc_no, qnty, config_path, total_workers, q)
        )
        p.start()
        procs.append(p)

    n_workers = len(procs)

    console = Console()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=True,
        refresh_per_second=4,
    ) as progress:
        task_id = {}
        done_count = 0

        while done_count < n_workers:
            try:
                kind, proc_no, val, note = q.get(timeout=0.25)
            except Empty:
                continue

            if kind == "start":
                # Show worker config in initial description
                desc = f"Worker {proc_no} ({note})"
                task_id[proc_no] = progress.add_task(desc, total=int(val))
            elif kind == "advance":
                tid = task_id.get(proc_no)
                if tid is None:
                    tid = task_id[proc_no] = progress.add_task(
                        f"Worker {proc_no}", total=int(val)
                    )
                progress.advance(tid, int(val))
                # Don't update description on every advance - causes flicker
            elif kind == "done":
                tid = task_id.get(proc_no)
                if tid is not None:
                    total = progress.tasks[tid].total or int(val)
                    progress.update(
                        tid,
                        completed=total,
                        description=f"Worker {proc_no} ✅ {note}",
                    )
                done_count += 1

        progress.stop()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    q.close(); q.join_thread()

def deduplicate_lineups():
    """
    Remove any saved Lineup records that share the same set of player IDs, 
    regardless of order, keeping only the first occurrence.
    """

    engine = init_db('nba')
    session = get_session(engine)

    seen_keys = set()
    duplicates = []

    for lu in session.query(Lineup).all():
        ids = _coerce_player_ids(lu.player_ids_list)
        key = tuple(sorted(ids))
        if key in seen_keys:
            duplicates.append(lu)
        else:
            seen_keys.add(key)

    if duplicates:
        for dup in duplicates:
            session.delete(dup)
        session.commit()
    
    print(f"Deduplicator: removed {len(duplicates)} duplicate lineups, kept {len(seen_keys)} unique.")

def main():
    p = argparse.ArgumentParser(description="generate and store nba lineups")
    p.add_argument("num_lineups", type=int)
    p.add_argument("num_jobs", type=int)
    p.add_argument("-c", "--config", default="config/nba.json")

    args = p.parse_args()

    base = args.num_lineups // args.num_jobs
    remainder = args.num_lineups % args.num_jobs
    quotas = [base + (1 if i < remainder else 0) for i in range(args.num_jobs)]

    _run_with_progress(quotas, args.config, total_workers=args.num_jobs)

    print(f"\n✅ Generated {args.num_lineups:,} lineups across {args.num_jobs} workers")
    return 0

if __name__ == "__main__":
    freeze_support()
    code = main()
    deduplicate_lineups()
    sys.exit(code)


    