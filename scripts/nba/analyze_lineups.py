#!/usr/bin/env python3
"""
Analyze all lineups in the database:
1. Calculate player exposures (how many lineups contain each player)
2. Calculate the mean FPTS sum for each lineup
3. Display distribution of lineup mean FPTS sums
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics

from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import Lineup, NBAPlayer


def analyze_lineups(session) -> Tuple[Dict[int, int], List[float], Dict[str, any]]:
    """
    Analyze all lineups in the database.

    Returns:
        - player_exposures: Dict mapping player_id -> number of lineups they appear in
        - lineup_fpts_sums: List of mean FPTS sums for each lineup
        - stats: Dictionary with summary statistics
    """
    lineups = session.query(Lineup).all()

    if not lineups:
        print("No lineups found in database.")
        return {}, [], {}

    player_exposures: Dict[int, int] = defaultdict(int)
    lineup_fpts_sums: List[float] = []

    # Get all unique player IDs to fetch their data once
    all_player_ids = set()
    for lineup in lineups:
        all_player_ids.update(lineup.player_ids_list)

    # Fetch all players
    players_data = session.query(NBAPlayer).filter(NBAPlayer.id.in_(all_player_ids)).all()
    player_fpts_map = {p.id: p.fpts_mean for p in players_data}

    # Analyze each lineup
    for lineup in lineups:
        player_ids = lineup.player_ids_list

        # Update exposures
        for pid in player_ids:
            player_exposures[pid] += 1

        # Calculate mean FPTS sum for this lineup
        fpts_sum = sum(player_fpts_map.get(pid, 0.0) for pid in player_ids)
        lineup_fpts_sums.append(fpts_sum)

    # Calculate statistics
    stats = {
        "total_lineups": len(lineups),
        "unique_players": len(player_exposures),
        "mean_fpts_sum": statistics.mean(lineup_fpts_sums),
        "median_fpts_sum": statistics.median(lineup_fpts_sums),
        "stdev_fpts_sum": statistics.stdev(lineup_fpts_sums) if len(lineup_fpts_sums) > 1 else 0.0,
        "min_fpts_sum": min(lineup_fpts_sums),
        "max_fpts_sum": max(lineup_fpts_sums),
    }

    return dict(player_exposures), lineup_fpts_sums, stats


def print_exposures(player_exposures: Dict[int, int], session, stats, top_n: int = 20):
    """Print top N most exposed players."""
    if not player_exposures:
        print("No player exposures to display.")
        return

    # Get player names
    player_ids = list(player_exposures.keys())
    players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(player_ids)).all()
    player_name_map = {p.id: p.name for p in players}

    # Sort by exposure
    sorted_exposures = sorted(
        player_exposures.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n{'='*60}")
    print(f"Top {min(top_n, len(sorted_exposures))} Most Exposed Players")
    print(f"{'='*60}")
    print(f"{'Player Name':<30} {'Exposure':<15} {'%':<10}")
    print(f"{'-'*60}")

    total_lineups = stats['total_lineups']

    for i, (pid, count) in enumerate(sorted_exposures[:top_n]):
        name = player_name_map.get(pid, f"Player {pid}")
        pct = (count / total_lineups * 100) if total_lineups > 0 else 0
        print(f"{name:<30} {count:<15} {pct:>6.1f}%")


def print_fpts_distribution(lineup_fpts_sums: List[float], stats: Dict, num_buckets: int = 10):
    """Print a histogram-style distribution of lineup FPTS sums."""
    if not lineup_fpts_sums:
        print("No FPTS sums to display.")
        return

    print(f"\n{'='*60}")
    print(f"Distribution of Lineup Mean FPTS Sums")
    print(f"{'='*60}")

    # Calculate bucket boundaries
    min_val = stats["min_fpts_sum"]
    max_val = stats["max_fpts_sum"]
    bucket_width = (max_val - min_val) / num_buckets

    # Create buckets
    buckets = [0] * num_buckets
    for fpts_sum in lineup_fpts_sums:
        bucket_idx = min(int((fpts_sum - min_val) / bucket_width), num_buckets - 1)
        buckets[bucket_idx] += 1

    # Print distribution
    print(f"\nStatistics:")
    print(f"  Mean:     {stats['mean_fpts_sum']:>8.2f}")
    print(f"  Median:   {stats['median_fpts_sum']:>8.2f}")
    print(f"  Std Dev:  {stats['stdev_fpts_sum']:>8.2f}")
    print(f"  Min:      {stats['min_fpts_sum']:>8.2f}")
    print(f"  Max:      {stats['max_fpts_sum']:>8.2f}")

    print(f"\nHistogram ({num_buckets} buckets):")
    print(f"{'Range':<20} {'Count':<10} {'%':<10} {'Distribution'}")
    print(f"{'-'*60}")

    max_count = max(buckets) if buckets else 1

    for i in range(num_buckets):
        lower = min_val + i * bucket_width
        upper = lower + bucket_width
        count = buckets[i]
        pct = (count / len(lineup_fpts_sums) * 100) if lineup_fpts_sums else 0
        bar_width = int((count / max_count) * 40) if max_count > 0 else 0
        bar = "█" * bar_width

        print(f"{lower:>7.1f}-{upper:<7.1f}  {count:<10} {pct:>6.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze lineups: exposures and FPTS distributions"
    )
    parser.add_argument(
        "-t", "--top",
        type=int,
        default=20,
        help="Number of top exposed players to display (default: 20)"
    )
    parser.add_argument(
        "-b", "--buckets",
        type=int,
        default=10,
        help="Number of histogram buckets (default: 10)"
    )

    args = parser.parse_args()

    engine = init_db('nba')
    session = get_session(engine)

    print("Analyzing lineups...")
    player_exposures, lineup_fpts_sums, stats = analyze_lineups(session)

    if not player_exposures:
        print("No lineups to analyze.")
        return 1

    print(f"\n✅ Analysis complete!")
    print(f"   Total lineups: {stats['total_lineups']}")
    print(f"   Unique players: {stats['unique_players']}")

    print_exposures(player_exposures, session, stats, top_n=args.top)
    print_fpts_distribution(lineup_fpts_sums, stats, num_buckets=args.buckets)

    return 0


if __name__ == "__main__":
    exit(main())
