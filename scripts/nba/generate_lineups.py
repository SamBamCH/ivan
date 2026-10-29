#!/usr/bin/env python3
"""
scripts/nba/generate_lineups.py

Generate the pool of candidate classic lineups: `<lineups> <parallel workers>`.
The worker processes and the dedupe are ivan.sports.nba.field_generation.generate_field;
this is the command line.

    scripts/nba/generate_lineups.py 25000 8
    scripts/nba/generate_lineups.py 25000 8 --fresh          # empty the lineup and sim tables first
    scripts/nba/generate_lineups.py 5000 4 --min-fpts 260    # only lineups projecting at least 260
"""
import argparse
from multiprocessing import freeze_support

from ivan.core.config import config_path
from ivan.database.session import get_session, init_db
from ivan.sports.nba.field_generation import clear_lineups, generate_field
from ivan.sports.nba.ingest import ensure_schema


def main() -> int:
    p = argparse.ArgumentParser(description="Generate and store NBA classic field lineups (multi-job)")
    p.add_argument("num_lineups", type=int, help="total number of lineups to generate")
    p.add_argument("num_jobs", type=int, help="how many parallel jobs to split across")
    p.add_argument("-c", "--config", default=str(config_path("nba")), help="path to the nba json config file")
    p.add_argument("--fresh", action="store_true", help="delete the loaded slate's lineups and their simulation results first (other slates' stay)")
    p.add_argument("--min-fpts", type=float, default=None, help="minimum projected total for a lineup (default: config generate.min_fpts_frac x the optimal)")
    p.add_argument("--randomness", type=float, default=None, help="spread of the sampled worlds around the mean, 0-1 (default: config generate.randomness, 0.25)")
    p.add_argument("--no-progress", action="store_true", help="plain log lines instead of progress bars")
    a = p.parse_args()
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    if a.fresh:
        n_lineups, n_sims = clear_lineups(session)
        print(f"cleared {n_lineups} lineups and {n_sims} simulation results")
    generate_field(a.num_lineups, a.num_jobs, a.config, progress=not a.no_progress, min_fpts=a.min_fpts, session=session, randomness=a.randomness)
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
