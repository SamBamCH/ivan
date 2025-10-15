#!/usr/bin/env python3
"""
Calibrate role factor parameters using historical game data.

The role factor mechanism uses anytime TD odds to estimate a player's usage.
This script compares predicted vs actual stats to tune ROLE_MIN, ROLE_MAX, ROLE_BETA.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sqlalchemy import text
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLStat, NFLOdds


def gather_historical_role_data(session, lookback_weeks=8):
    """
    For each player-game where we have:
    1. Historical odds (anytime TD)
    2. Actual game stats

    Calculate what role factor would have been derived and compare to actual usage.
    """

    # Get recent games with stats
    query = text("""
        SELECT
            player_id,
            player_display_name,
            position_group,
            season,
            week,
            recent_team as team,
            rushing_yards,
            rushing_tds,
            receiving_yards,
            receiving_tds,
            receptions,
            targets
        FROM nfl_stats
        WHERE position_group IN ('RB', 'WR', 'TE')
        ORDER BY season DESC, week DESC
        LIMIT :limit
    """)

    stats_df = pd.read_sql(query, session.bind, params={'limit': 1000})

    if stats_df.empty:
        print("No historical stats found")
        return pd.DataFrame()

    print(f"Found {len(stats_df)} player-games")

    # Position baselines from your model
    # These are the baseline scrim TD expectations per position
    baseline_td = {
        'RB': 0.50,
        'WR': 0.35,
        'TE': 0.30,
    }

    # For each stat, define what % of baseline a player at each role level should have
    # We'll calculate actual usage as: actual_stat / position_avg_stat

    rows = []
    for _, game in stats_df.iterrows():
        pos = game['position_group']
        if pos not in baseline_td:
            continue

        # Calculate actual "role" based on touches/targets (usage proxy)
        if pos == 'RB':
            # For RBs, use rushing attempts as usage (we don't have carries in this view, use yards as proxy)
            # Better: use (rushing_tds + receiving_tds) / baseline as realized TD rate
            actual_tds = (game['rushing_tds'] or 0) + (game['receiving_tds'] or 0)
            actual_role_tds = actual_tds / max(baseline_td[pos], 0.01)

            # Volume stats (normalize by typical game)
            rush_yds = game['rushing_yards'] or 0
            rec_yds = game['receiving_yards'] or 0
            recs = game['receptions'] or 0

            # Typical RB game: 60 rush yds, 20 rec yds, 2 rec
            # A backup might be: 10 rush yds, 5 rec yds, 1 rec
            actual_role_volume = (rush_yds / 60 + rec_yds / 20 + recs / 2) / 3

        elif pos in ['WR', 'TE']:
            actual_tds = game['receiving_tds'] or 0
            actual_role_tds = actual_tds / max(baseline_td[pos], 0.01)

            rec_yds = game['receiving_yards'] or 0
            recs = game['receptions'] or 0
            targets = game['targets'] or 0

            # Typical WR game: 70 rec yds, 5 rec, 7 targets
            # Typical TE game: 50 rec yds, 4 rec, 5 targets
            if pos == 'WR':
                actual_role_volume = (rec_yds / 70 + recs / 5 + targets / 7) / 3
            else:  # TE
                actual_role_volume = (rec_yds / 50 + recs / 4 + targets / 5) / 3
        else:
            continue

        rows.append({
            'player': game['player_display_name'],
            'position': pos,
            'season': game['season'],
            'week': game['week'],
            'team': game['team'],
            'actual_role_tds': actual_role_tds,
            'actual_role_volume': actual_role_volume,
            'rushing_yards': game['rushing_yards'] or 0,
            'receiving_yards': game['receiving_yards'] or 0,
            'receptions': game['receptions'] or 0,
            'total_tds': actual_tds,
        })

    calibration_df = pd.DataFrame(rows)

    print("\n" + "="*100)
    print("HISTORICAL ROLE FACTOR ANALYSIS")
    print("="*100)

    print("\nActual Role Distribution (based on realized TDs vs baseline):")
    print(calibration_df['actual_role_tds'].describe())

    print("\nActual Role Distribution (based on volume stats):")
    print(calibration_df['actual_role_volume'].describe())

    # Show breakdown by position
    print("\n" + "-"*100)
    print("BY POSITION")
    print("-"*100)
    for pos in ['RB', 'WR', 'TE']:
        pos_data = calibration_df[calibration_df['position'] == pos]
        if len(pos_data) > 0:
            print(f"\n{pos} (n={len(pos_data)}):")
            print(f"  TD-based role: mean={pos_data['actual_role_tds'].mean():.3f}, "
                  f"median={pos_data['actual_role_tds'].median():.3f}, "
                  f"p10={pos_data['actual_role_tds'].quantile(0.10):.3f}, "
                  f"p90={pos_data['actual_role_tds'].quantile(0.90):.3f}")
            print(f"  Volume-based role: mean={pos_data['actual_role_volume'].mean():.3f}, "
                  f"median={pos_data['actual_role_volume'].median():.3f}, "
                  f"p10={pos_data['actual_role_volume'].quantile(0.10):.3f}, "
                  f"p90={pos_data['actual_role_volume'].quantile(0.90):.3f}")

    # Compare current parameters
    print("\n" + "="*100)
    print("CURRENT PARAMETER ANALYSIS")
    print("="*100)
    print("\nYour current settings:")
    print("  ROLE_MIN = 0.01")
    print("  ROLE_MAX = 0.50")
    print("  ROLE_BETA = 1.25")

    # Calculate what % of actual games fall outside your bounds
    volume_below_min = (calibration_df['actual_role_volume'] < 0.01).sum()
    volume_above_max = (calibration_df['actual_role_volume'] > 0.50).sum()
    volume_in_range = len(calibration_df) - volume_below_min - volume_above_max

    print(f"\nBased on {len(calibration_df)} historical games:")
    print(f"  Below ROLE_MIN (0.01): {volume_below_min} ({volume_below_min/len(calibration_df)*100:.1f}%)")
    print(f"  Within bounds: {volume_in_range} ({volume_in_range/len(calibration_df)*100:.1f}%)")
    print(f"  Above ROLE_MAX (0.50): {volume_above_max} ({volume_above_max/len(calibration_df)*100:.1f}%)")

    # Suggest new ROLE_MAX based on percentiles
    print("\n" + "-"*100)
    print("SUGGESTED ADJUSTMENTS")
    print("-"*100)

    p75_role = calibration_df['actual_role_volume'].quantile(0.75)
    p80_role = calibration_df['actual_role_volume'].quantile(0.80)
    p85_role = calibration_df['actual_role_volume'].quantile(0.85)
    p90_role = calibration_df['actual_role_volume'].quantile(0.90)

    print(f"\nRole percentiles from historical data:")
    print(f"  75th percentile: {p75_role:.3f}")
    print(f"  80th percentile: {p80_role:.3f}")
    print(f"  85th percentile: {p85_role:.3f}")
    print(f"  90th percentile: {p90_role:.3f}")

    print(f"\nRecommendations:")
    if p90_role < 0.50:
        print(f"  ⚠️  Your ROLE_MAX (0.50) is higher than 90th percentile ({p90_role:.3f})")
        print(f"  → Consider lowering ROLE_MAX to {p85_role:.2f} - {p90_role:.2f}")
        print(f"     This will more aggressively shrink projections for backup players")

    # Check if cheap players are getting too high role factors
    print("\n" + "-"*100)
    print("LOW-USAGE PLAYER ANALYSIS (potential backups)")
    print("-"*100)

    low_usage = calibration_df[calibration_df['actual_role_volume'] < 0.15].sort_values('actual_role_volume')
    print(f"\nPlayers with <15% of typical usage (n={len(low_usage)}):")
    print(low_usage[['player', 'position', 'actual_role_volume', 'total_tds', 'receptions', 'rushing_yards', 'receiving_yards']].head(20).to_string(index=False))

    return calibration_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-url', default='sqlite:///data/nfl/nfl_history.sqlite')
    parser.add_argument('--lookback-weeks', type=int, default=8)
    parser.add_argument('--output', default=None, help='Save to CSV')
    args = parser.parse_args()

    engine = init_db('nfl', args.db_url)
    session = get_session(engine)

    df = gather_historical_role_data(session, args.lookback_weeks)

    if args.output and not df.empty:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")
