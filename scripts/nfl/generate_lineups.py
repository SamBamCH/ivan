#!/usr/bin/env python3
"""
scripts/nfl/generate_lineups.py

Generate the field of candidate lineups: `<lineups> <parallel workers>`. The
correlation check, the worker processes and the dedupe are
ivan.sports.nfl.field_generation.generate_field; this is the command line.
"""
import argparse
from multiprocessing import freeze_support

from ivan.core.config import config_path
from ivan.sports.nfl.field_generation import generate_field


def main() -> int:
    p = argparse.ArgumentParser(description="Generate and store NFL field lineups (multi-job)")
    p.add_argument("num_lineups", type=int, help="Total number of lineups to generate")
    p.add_argument("num_jobs", type=int, help="how many parallel jobs to split across")
    p.add_argument("-c", "--config", default=str(config_path("nfl")), help="path to nfl json config file")
    p.add_argument("--no-progress", action="store_true", help="plain log lines instead of progress bars")
    a = p.parse_args()
    generate_field(a.num_lineups, a.num_jobs, a.config, progress=not a.no_progress)
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
