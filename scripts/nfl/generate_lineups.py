import json, os, argparse, sys

from queue import Empty

from multiprocessing import Process, freeze_support, Queue
from pathlib import Path

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn,MofNCompleteColumn
from rich.console import Console

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import Lineup
from ivan.sports.nfl.generate_lineups_logic import generate_and_store_field_lineups

BATCH_COMMIT_SIZE = 100 # commit every n lineups

SCRIPT_NAME = Path(sys.argv[0]).stem

def _prefix(script: str, proc_no: int, sub_no: int) -> str:
    return f"[{script}][{proc_no}][{sub_no}]"

def _coerce_player_ids(val): 
    if val is None:
        return []
    if isinstance(val, list):
        return [int(x) for x in val]
    if isinstance(val, str):
        try:
            data = json.loads(val)
            if isinstance(data, list):
                return [int(x) for x in data]
            
        except Exception:
            return []
        
    return []

def worker(worker_id: int, num_lineups: int, config_path: str, q: Queue):
    # init db
    engine = init_db('nfl')
    session = get_session(engine)

    # load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # tell parent our total target
    q.put(("start", worker_id, num_lineups, f"target={num_lineups}"))

    # progress callback from the generator
    def on_progress(delta: int, note: str | None = None):
        q.put(("advance", worker_id, delta, note or ""))

    # run
    created = generate_and_store_field_lineups(
        session,
        config,
        num_lineups=num_lineups,
        batch_size=BATCH_COMMIT_SIZE,
        on_progress=on_progress,   # <--- NEW
    )

    # final report
    q.put(("done", worker_id, created, f"created={created}"))

def _run_with_progress(quotas, config_path):
    q = Queue()
    procs = []
    for proc_no, qnty in enumerate(quotas):
        p = Process(target=worker, args=(proc_no, qnty, config_path, q))
        p.start()
        procs.append(p)

    n_workers = len(procs)

    # ---------- Fallback (no rich) ----------
    if Progress is None:
        totals = quotas[:]
        done = [0]*n_workers
        done_count = 0

        while done_count < n_workers:
            try:
                kind, proc_no, val, note = q.get(timeout=0.25)
            except Empty:
                # no message; continue polling / let workers progress
                continue

            if kind == "start":
                print(f"[{SCRIPT_NAME}][{proc_no}][1] starting… target={val}")
            elif kind == "advance":
                done[proc_no] += int(val)
                print(
                    f"[{SCRIPT_NAME}][{proc_no}][1] {done[proc_no]}/{totals[proc_no]} {note or ''}",
                    end="\r",
                    flush=True,
                )
            elif kind == "done":
                print()  # seal the carriage-returned line
                print(f"[{SCRIPT_NAME}][{proc_no}][1] ✅ finished ({note})")
                done_count += 1

        # clean shutdown
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        q.close(); q.join_thread()
        return

    # ---------- Rich multi-bar ----------
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
        refresh_per_second=20,
    ) as progress:
        task_id = {}
        done_count = 0

        while done_count < n_workers:
            try:
                kind, proc_no, val, note = q.get(timeout=0.25)
            except Empty:
                # no message; keep the UI alive and loop
                continue

            if kind == "start":
                desc = f"[{SCRIPT_NAME}][{proc_no}][1] initializing…"
                task_id[proc_no] = progress.add_task(desc, total=int(val))
            elif kind == "advance":
                tid = task_id.get(proc_no)
                if tid is None:
                    tid = task_id[proc_no] = progress.add_task(
                        f"[{SCRIPT_NAME}][{proc_no}][1] initializing…", total=int(val)
                    )
                progress.advance(tid, int(val))
                if note:
                    progress.update(
                        tid,
                        description=f"[{SCRIPT_NAME}][{proc_no}][1] building lineups… {note}",
                    )
            elif kind == "done":
                tid = task_id.get(proc_no)
                if tid is not None:
                    # snap to complete and print a nice final description
                    total = progress.tasks[tid].total or int(val)
                    progress.update(
                        tid,
                        completed=total,
                        description=f"[{SCRIPT_NAME}][{proc_no}][1] ✅ finished ({note})",
                    )
                done_count += 1

        # end of loop — make extra sure progress stops flushing
        progress.stop()

    # clean shutdown
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

    engine = init_db('nfl')
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
    p = argparse.ArgumentParser(description="Generate and store NFL field lineups (multi-job)")
    p.add_argument("num_lineups", type=int, help="Total number of lineups to generate")
    p.add_argument("num_jobs", type=int, help="how many parallel jobs to split across")
    p.add_argument("-c", "--config", default="config/nfl.json", help="path to nfl json config file")
    args = p.parse_args()

    base = args.num_lineups // args.num_jobs
    remainder = args.num_lineups % args.num_jobs
    quotas = [base + (1 if i < remainder else 0) for i in range(args.num_jobs)]

    _run_with_progress(quotas, args.config)

    print(f"{_prefix(SCRIPT_NAME, 'all', 0)} ✅ All done: generated a total of {args.num_lineups} lineups in {args.num_jobs} jobs.")
    return 0

if __name__ == "__main__":
    freeze_support()
    code = main()
    deduplicate_lineups()
    sys.exit(code)

