#!/usr/bin/env python3
"""
Project player ownership for the current slate from field sampling, then snapshot
projections for model training.

This runs automatically at the end of ingest.py; use it on its own to re-project
without re-fetching anything (e.g. after tuning config["ownership"]).
"""
import argparse
import logging

from ivan.core.config import config_path, load_config
from ivan.database.session import get_session, init_db
from ivan.sports.nfl.ownership import project_and_snapshot


def main():
    p = argparse.ArgumentParser(description="Project ownership from field sampling and snapshot projections")
    p.add_argument("--config", default=str(config_path("nfl")), help="path to nfl json config file")
    p.add_argument("--num-lineups", type=int, default=None,
                   help="lineups sampled per run (default: config ownership.num_lineups, 1000)")
    p.add_argument("--sims", type=int, default=None,
                   help="independent runs to average (default: config ownership.num_simulations, 1)")
    p.add_argument("--no-snapshot", action="store_true", help="skip the projection snapshot")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config("nfl", args.config)
    session = get_session(init_db("nfl", cfg.get("database_url")))
    ownership = project_and_snapshot(session, cfg, num_lineups=args.num_lineups,
                                     num_simulations=args.sims, snapshot=not args.no_snapshot)
    print(f"projected ownership for {len(ownership)} players")


if __name__ == "__main__":
    main()
