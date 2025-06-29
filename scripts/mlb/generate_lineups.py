#!/usr/bin/env python
import sys
import json
import argparse
from multiprocessing import Process
from math import ceil
import os

from ivan.database.session import init_db, get_session
from ivan.sports.mlb.generate_lineups import generate_and_store_field_lineups

BATCH_COMMIT_SIZE = 100  # commit every 100 lineups

def worker(num_lineups: int, config_path: str):
    """
    Each worker spins up its own DB session and
    generates `num_lineups`, committing every BATCH_COMMIT_SIZE.
    """
    # 1) init db & session
    engine  = init_db("mlb")
    session = get_session(engine)

    # 2) load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # 3) call your existing generator, but with batch_size override
    created = generate_and_store_field_lineups(
        session,
        config,
        num_lineups=num_lineups,
        batch_size=BATCH_COMMIT_SIZE
    )
    print(f"[PID {os.getpid()}] generated {created} lineups")


def main():
    p = argparse.ArgumentParser(
        description="Generate and store MLB field lineups (multi-job)"
    )
    p.add_argument("num_lineups", type=int,
                   help="Total number of lineups to generate")
    p.add_argument("num_jobs", type=int,
                   help="How many parallel jobs to split across")
    p.add_argument("-c", "--config", default="config/mlb.json",
                   help="Path to MLB JSON configuration file")
    args = p.parse_args()

    # determine per‐job quotas
    base      = args.num_lineups // args.num_jobs
    remainder = args.num_lineups % args.num_jobs

    quotas = []
    for i in range(args.num_jobs):
        # distribute the remainder among the first `remainder` jobs
        extra = 1 if i < remainder else 0
        quotas.append(base + extra)

    # launch all workers
    procs = []
    for q in quotas:
        proc = Process(target=worker, args=(q, args.config))
        proc.start()
        procs.append(proc)

    # wait for them all
    for proc in procs:
        proc.join()

    print(f"✅ All done: generated a total of {args.num_lineups} lineups "
          f"in {args.num_jobs} jobs.")

if __name__ == "__main__":
    sys.exit(main())
