#!/usr/bin/env python3
"""
Standalone script to plot diagnostics for *all* lineups in the database,
using the same histograms and stack‐probability output as after sampling.
"""

import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter

from ivan.database.session import init_db, get_session
from ivan.sports.mlb.models import Lineup, MLBPlayer
from ivan.sports.mlb.sample_lineups import fetch_latest_map, PITCHER_POS

def main():
    parser = argparse.ArgumentParser(
        description="Plot ownership, projection, and stack distributions "
                    "for all lineups in the database"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/mlb.json",
        help="Path to MLB JSON config (must include `database_url`)"
    )
    args = parser.parse_args()

    # Load config & init DB/session
    cfg    = json.load(open(args.config, encoding="utf-8"))
    engine = init_db("mlb", cfg.get("database_url"))
    session = get_session(engine)

    # Fetch *all* lineups
    all_lineups = session.query(Lineup).all()
    if not all_lineups:
        print("No lineups found in the database.")
        return

    # Fetch latest player rows for every player in those lineups
    all_pids = {pid for lup in all_lineups for pid in lup.player_ids_list}
    latest_map = fetch_latest_map(session, all_pids)

    # Compute per‐lineup metrics
    ownership_avgs = []
    proj_sums      = []
    stack_types    = []

    for lup in all_lineups:
        pids = lup.player_ids_list

        # total projection
        proj = sum(latest_map[pid].fpts_projection for pid in pids)
        proj_sums.append(proj)

        # average ownership
        own_values = [latest_map[pid].ownership / 100 for pid in pids]
        ownership_avgs.append(sum(own_values) / len(own_values))

        # identify top two non‐pitcher counts to form a 'stack' type
        counts = Counter()
        for pid in pids:
            player = latest_map[pid]
            if not any(pos in PITCHER_POS for pos in player.position):
                counts[player.team] += 1
        top_two = sorted(counts.values(), reverse=True)[:2]
        c1 = top_two[0] if top_two else 0
        c2 = top_two[1] if len(top_two) > 1 else 0
        stack_types.append(f"{c1}-{c2}")

    # Plot histogram: average ownership
    plt.figure()
    plt.hist(ownership_avgs, bins=20)
    plt.title("All Lineups: Average Player Ownership")
    plt.xlabel("Avg Ownership")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Plot histogram: projection sums
    plt.figure()
    plt.hist(proj_sums, bins=20)
    plt.title("All Lineups: Total Projected FPTS")
    plt.xlabel("Projection Sum")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Print stack‐type probabilities
    total = len(stack_types)
    stack_counts = Counter(stack_types)
    print("\nStack Type Probabilities (non‐pitcher counts):")
    for stype, cnt in stack_counts.most_common():
        print(f"  {stype}: {cnt/total:.3f}")

if __name__ == "__main__":
    main()
