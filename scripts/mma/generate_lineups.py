#!/usr/bin/env python3
# scripts/mma/generate_lineups.py
"""
Generate the candidate lineup pool the field is sampled from.

    scripts/mma/generate_lineups.py [n_candidates] [n_jobs] [--append]

Clears the previous pool and its simulation results first unless --append.
"""
import argparse
import os
import time

from ivan.core.config import config_path, load_config
from ivan.database.session import get_session, init_db
from ivan.sports.mma.lineup_generation import generate_candidates
from ivan.sports.mma.models import MMABase


def main():
    p = argparse.ArgumentParser(description="Generate MMA candidate lineups")
    p.add_argument("n_candidates", type=int, nargs="?", default=None, help="MIP solves to run (default: lineup_generation.n_candidates)")
    p.add_argument("n_jobs", type=int, nargs="?", default=max(1, min(4, os.cpu_count() or 1)), help="worker processes")
    p.add_argument("--append", action="store_true", help="keep the existing pool and add to it")
    p.add_argument("--universe", action="store_true", help="every valid lineup of the slate instead of MIP solves (the fitted field model sets frequencies)")
    p.add_argument("--config", default=str(config_path("mma")))
    args = p.parse_args()

    cfg = load_config("mma", args.config)
    engine = init_db("mma", cfg.get("database_url"))
    MMABase.metadata.create_all(engine)
    session = get_session(engine)

    t0 = time.time()
    if args.universe:
        from ivan.sports.mma.lineup_generation import generate_universe
        r = generate_universe(session, cfg, fresh=not args.append)
        print(f"universe: {r['requested']:,} valid lineups, {r['new_unique']:,} new, pool now {r['pool']:,} ({time.time() - t0:.0f}s)")
        return
    r = generate_candidates(session, cfg, n_candidates=args.n_candidates, n_jobs=args.n_jobs, fresh=not args.append)
    print(f"{r['solved']:,} solves, {r['new_unique']:,} new unique lineups, pool now {r['pool']:,} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
