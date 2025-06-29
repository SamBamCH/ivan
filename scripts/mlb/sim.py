#!/usr/bin/env python
# scripts/mlb/run_simulations.py

import os
from datetime import date
import argparse
from ivan.sports.mlb.simulation import run_simulations

def parse_args():
    p = argparse.ArgumentParser(
        description="Run MLB DFS simulations for a given slate."
    )
    p.add_argument(
        'num_lineups',
        type=int,
        help="Number of lineups to generate"
    )
    p.add_argument(
        'num_sims',
        type=int,
        help="Number of simulation runs per lineup"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # adjust slate as needed
    slate = date(2025, 5, 21)
    here  = os.path.dirname(__file__)
    cfg   = os.path.abspath(os.path.join(here, '../../config/mlb.json'))

    print(f"Running {args.num_lineups} lineups × {args.num_sims} sims on {slate}")
    run_simulations(
        slate_date = slate,
        n_lineups  = args.num_lineups,
        num_sims   = args.num_sims,
        config_path= cfg
    )

if __name__ == '__main__':
    main()
