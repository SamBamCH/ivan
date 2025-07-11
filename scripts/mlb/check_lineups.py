#!/usr/bin/env python3
"""
Standalone script to plot diagnostics for *all* lineups in the database,
using combined histograms and then reporting exposures vs. projected ownership.
"""

import argparse
import json
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from ivan.database.session import init_db, get_session
from ivan.sports.mlb.models import Lineup
from ivan.sports.mlb.sample_lineups import fetch_latest_map, PITCHER_POS


def main():
    parser = argparse.ArgumentParser(
        description="Plot ownership, projection, and stack distributions "
                    "for all lineups and print player exposures vs. projected ownership"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/mlb.json",
        help="Path to MLB JSON config (must include `database_url`)"
    )
    args = parser.parse_args()

    # Load config & init DB/session
    cfg     = json.load(open(args.config, encoding="utf-8"))
    engine  = init_db("mlb", cfg.get("database_url"))
    session = get_session(engine)

    # Fetch *all* lineups
    all_lineups = session.query(Lineup).all()
    if not all_lineups:
        print("No lineups found in the database.")
        return

    # Fetch latest player rows for every player in those lineups
    all_pids   = {pid for lup in all_lineups for pid in lup.player_ids_list}
    latest_map = fetch_latest_map(session, all_pids)

    # Compute per‐lineup metrics
    ownership_avgs = []
    proj_sums      = []
    stack_types    = []

    for lup in all_lineups:
        pids = lup.player_ids_list

        # total projection
        proj_sums.append(
            sum(latest_map[pid].fpts_projection for pid in pids)
        )

        # average ownership (convert % → fraction)
        owns = [latest_map[pid].ownership / 100 for pid in pids]
        ownership_avgs.append(sum(owns) / len(owns))

        # stack‐type (top two non‐pitcher counts)
        cnt = Counter(
            latest_map[pid].team
            for pid in pids
            if not any(pos in PITCHER_POS for pos in latest_map[pid].position)
        )
        top_two = sorted(cnt.values(), reverse=True)[:2]
        c1, c2 = (top_two + [0, 0])[:2]
        stack_types.append(f"{c1}-{c2}")

    # Convert to numpy arrays
    own_arr  = np.array(ownership_avgs)
    proj_arr = np.array(proj_sums)
    labels   = stack_types

    # ––– Plot all metrics side by side –––
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Ownership histogram
    axes[0].hist(own_arr, bins=20, edgecolor="black")
    axes[0].set_title("Average Ownership per Lineup")
    axes[0].set_xlabel("Avg Ownership")
    axes[0].set_ylabel("Count of Lineups")

    # 2) Projection‐sum histogram
    axes[1].hist(proj_arr, bins=20, edgecolor="black")
    axes[1].set_title("Total Projection per Lineup")
    axes[1].set_xlabel("Sum of fpts_projection")
    axes[1].set_ylabel("Count of Lineups")

    # 3) Stack‐type bar chart
    stack_counts = Counter(labels)
    types, counts = zip(*sorted(stack_counts.items(), key=lambda x: x[1], reverse=True))
    axes[2].bar(types, counts)
    axes[2].set_title("Stack Types (Top-2 Non-Pitcher Counts)")
    axes[2].set_xlabel("Stack Type (c1-c2)")
    axes[2].set_ylabel("Count of Lineups")
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # ––– Compute player exposures –––
    exposures = Counter(pid for lup in all_lineups for pid in lup.player_ids_list)
    total_lineups = len(all_lineups)

    pitchers = {}
    batters  = {}

    for pid, count in exposures.items():
        exposure_frac = count / total_lineups
        projected_own = latest_map[pid].ownership / 100
        if any(pos in PITCHER_POS for pos in latest_map[pid].position):
            pitchers[pid] = (exposure_frac, projected_own)
        else:
            batters[pid]  = (exposure_frac, projected_own)

    def print_exposures(title, exp_dict):
        print(f"\n{title}")
        print(f"{'Name':<20}  {'Team':<5}  {'Exposure':>10}  {'ProjOwn':>10}")
        for pid, (expv, projv) in sorted(exp_dict.items(), key=lambda x: x[1][0], reverse=True):
            player = latest_map[pid]
            print(f"{player.name:<20}  {player.team:<5}  {expv:>9.2%}  {projv:>9.2%}")

    print_exposures("Pitcher Exposures vs. Projected Ownership", pitchers)
    print_exposures("Batter Exposures vs. Projected Ownership", batters)

   # ––– Top 5 absolute deviations from projected ownership –––
    def print_top_deviators(title, exp_dict, n=5):
        print(f"\n{title}")
        print(f"{'Name':<20}  {'Team':<5}  {'Exposure':>10}  {'ProjOwn':>10}  {'Dev':>8}")
        # build list of (pid, deviation, exposure, proj)
        devs = [
            (pid, abs(expv - projv), expv, projv)
            for pid, (expv, projv) in exp_dict.items()
        ]
        for pid, dev, expv, projv in sorted(devs, key=lambda x: x[1], reverse=True)[:n]:
            player = latest_map[pid]
            print(f"{player.name:<20}  {player.team:<5}  {expv:>9.2%}  {projv:>9.2%}  {dev:>7.2%}")

    print_top_deviators("Top 5 Deviations — Pitchers", pitchers)
    print_top_deviators("Top 5 Deviations — Batters", batters)


if __name__ == "__main__":
    main()
