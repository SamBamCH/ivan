import json, os, argparse, sys

from multiprocessing import Process, freeze_support

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import Lineup
from ivan.sports.nfl.generate_lineups_logic import generate_and_store_field_lineups

BATCH_COMMIT_SIZE = 100 # commit every n lineups

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

def worker(num_lineups: int, config_path: str):
    """
    Each worker spins up its own DB session and generates 'num_lineups',
    committing every BATCH_COMMIT_SIZE.
    """

    # 1) init db and connect session
    engine = init_db('nfl')
    session = get_session(engine)

    # 2) load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # 3) call NFL generator (same batching idea as MLB)
    created = generate_and_store_field_lineups(
        session,
        config, 
        num_lineups=num_lineups,
        batch_size = BATCH_COMMIT_SIZE
    )
    print(f"[PID {os.getpid()}] generated {created} lineups")

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

    procs = []
    for q in quotas:
        proc = Process(target=worker, args=(q, args.config))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    print(f"✅ All done: generated a total of {args.num_lineups} lineups in {args.num_jobs} jobs.")
    return 0

if __name__ == "__main__":
    freeze_support()
    code = main()
    deduplicate_lineups()
    sys.exit(code)

