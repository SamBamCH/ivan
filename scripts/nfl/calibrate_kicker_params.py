#!/usr/bin/env python3
"""
Calibrate Kicker distribution parameters from historical data.

This script fits Gamma distribution parameters to historical kicker performance,
replacing the hardcoded KICKER_DEFAULT_SHAPE and KICKER_DEFAULT_SCALE values.

Outputs:
- Fitted Gamma distribution parameters (shape, scale)
- Saves to data/nfl/models/kicker_params.json
- Prints diagnostic statistics and fit quality metrics
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.optimize import minimize
from sqlalchemy import text

from ivan.database.session import init_db, get_session


def load_kicker_data(session, min_season: int = 2020, min_attempts: int = 1) -> pd.DataFrame:
    """
    Load historical kicker fantasy points from nfl_stats table.

    Args:
        session: Database session
        min_season: Earliest season to include (default: 2020)
        min_attempts: Minimum field goal attempts to include game (filters out zero-snap games)

    Returns:
        DataFrame with columns: player_display_name, recent_team, season, week, fantasy_points
    """
    query = text("""
        SELECT
            player_display_name,
            team,
            season,
            week,
            fantasy_points,
            fantasy_points_ppr as fpts_ppr
        FROM nfl_stats
        WHERE position_group = 'K'
          AND season >= :min_season
          AND fantasy_points IS NOT NULL
          AND fantasy_points >= 0
        ORDER BY season DESC, week DESC
    """)

    df = pd.read_sql(query, session.bind, params={"min_season": min_season})

    if df.empty:
        raise RuntimeError(f"No kicker data found for seasons >= {min_season}")

    print(f"Loaded {len(df):,} kicker game records")
    print(f"  Seasons: {df['season'].min()}-{df['season'].max()}")
    print(f"  Unique players: {df['player_display_name'].nunique()}")
    print(f"  Games per player: {len(df) / df['player_display_name'].nunique():.1f} avg")

    return df


def fit_gamma_distribution(data: np.ndarray) -> Tuple[float, float, Dict]:
    """
    Fit Gamma distribution to kicker fantasy points using MLE.

    Args:
        data: Array of fantasy points values

    Returns:
        (shape, scale, diagnostics_dict)
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]  # Gamma requires positive values

    if len(data) < 10:
        raise ValueError(f"Insufficient data points ({len(data)}) to fit distribution")

    # Fit using scipy's MLE
    shape, loc, scale = st.gamma.fit(data, floc=0)  # Fix loc=0 for standard Gamma

    # Compute goodness-of-fit statistics
    ks_stat, ks_pval = st.kstest(data, 'gamma', args=(shape, 0, scale))

    # Compare to actual data statistics
    actual_mean = float(np.mean(data))
    actual_std = float(np.std(data, ddof=1))
    fitted_mean = float(shape * scale)
    fitted_std = float(np.sqrt(shape * scale**2))

    diagnostics = {
        "n_samples": int(len(data)),
        "actual_mean": actual_mean,
        "actual_std": actual_std,
        "actual_median": float(np.median(data)),
        "actual_p25": float(np.percentile(data, 25)),
        "actual_p75": float(np.percentile(data, 75)),
        "fitted_mean": fitted_mean,
        "fitted_std": fitted_std,
        "mean_error_pct": 100 * abs(fitted_mean - actual_mean) / actual_mean,
        "std_error_pct": 100 * abs(fitted_std - actual_std) / actual_std,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
    }

    return float(shape), float(scale), diagnostics


def fit_by_team(df: pd.DataFrame, min_games: int = 20) -> Dict[str, Tuple[float, float]]:
    """
    Fit Gamma distribution per team (for team-specific kicker environments).

    Args:
        df: Kicker data DataFrame
        min_games: Minimum games required to fit team-specific params

    Returns:
        {team: (shape, scale)} for teams with sufficient data
    """
    team_params = {}

    for team, group in df.groupby('team'):
        if len(group) >= min_games:
            try:
                shape, scale, _ = fit_gamma_distribution(group['fantasy_points'].values)
                team_params[team] = (shape, scale)
            except Exception as e:
                print(f"  Warning: Failed to fit {team} ({len(group)} games): {e}")

    return team_params


def main():
    parser = argparse.ArgumentParser(description="Calibrate kicker distribution parameters")
    parser.add_argument("--config", default="config/nfl.json", help="Path to config file")
    parser.add_argument("--min-season", type=int, default=2020, help="Earliest season to include")
    parser.add_argument("--output", default="data/nfl/models/kicker_params.json", help="Output path")
    parser.add_argument("--by-team", action="store_true", help="Fit separate params per team")
    args = parser.parse_args()

    # Load config and initialize DB
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        db_url = config.get("database_url")
    else:
        db_url = None

    engine = init_db("nfl", db_url)
    session = get_session(engine)

    print("=" * 80)
    print("KICKER PARAMETER CALIBRATION")
    print("=" * 80)
    print()

    # Load historical data
    df = load_kicker_data(session, min_season=args.min_season)
    print()

    # Fit league-wide distribution
    print("-" * 80)
    print("LEAGUE-WIDE DISTRIBUTION")
    print("-" * 80)

    shape, scale, diagnostics = fit_gamma_distribution(df['fantasy_points'].values)

    print(f"\n✓ Fitted Gamma distribution:")
    print(f"  Shape (α): {shape:.4f}")
    print(f"  Scale (β): {scale:.4f}")
    print(f"\n  Actual vs Fitted:")
    print(f"    Mean:   {diagnostics['actual_mean']:.2f} vs {diagnostics['fitted_mean']:.2f} "
          f"(error: {diagnostics['mean_error_pct']:.1f}%)")
    print(f"    Std:    {diagnostics['actual_std']:.2f} vs {diagnostics['fitted_std']:.2f} "
          f"(error: {diagnostics['std_error_pct']:.1f}%)")
    print(f"    Median: {diagnostics['actual_median']:.2f}")
    print(f"\n  Goodness of fit:")
    print(f"    KS statistic: {diagnostics['ks_statistic']:.4f}")
    print(f"    KS p-value:   {diagnostics['ks_pvalue']:.4f}")

    if diagnostics['ks_pvalue'] < 0.05:
        print(f"    ⚠️  Warning: Gamma fit may not be ideal (p < 0.05)")
    else:
        print(f"    ✓ Good fit (p >= 0.05)")

    # Compare to old hardcoded values
    OLD_SHAPE, OLD_SCALE = 7.3, 1.2
    old_mean = OLD_SHAPE * OLD_SCALE
    new_mean = shape * scale

    print(f"\n  Comparison to old hardcoded values:")
    print(f"    Old: Shape={OLD_SHAPE:.2f}, Scale={OLD_SCALE:.2f}, Mean={old_mean:.2f}")
    print(f"    New: Shape={shape:.2f}, Scale={scale:.2f}, Mean={new_mean:.2f}")
    print(f"    Mean difference: {new_mean - old_mean:+.2f} pts ({100*(new_mean/old_mean - 1):+.1f}%)")

    # Prepare output
    output_data = {
        "league_wide": {
            "shape": shape,
            "scale": scale,
            "diagnostics": diagnostics,
        },
        "metadata": {
            "min_season": args.min_season,
            "n_games": int(len(df)),
            "n_players": int(df['player_display_name'].nunique()),
            "old_hardcoded": {"shape": OLD_SHAPE, "scale": OLD_SCALE},
        }
    }

    # Optional: fit by team
    if args.by_team:
        print(f"\n" + "-" * 80)
        print("TEAM-SPECIFIC DISTRIBUTIONS")
        print("-" * 80)

        team_params = fit_by_team(df, min_games=20)

        print(f"\n✓ Fitted {len(team_params)} teams with sufficient data:")

        # Sort teams by mean projection
        team_means = [(team, shape * scale) for team, (shape, scale) in team_params.items()]
        team_means.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  Top 5 kicker-friendly teams:")
        for i, (team, mean) in enumerate(team_means[:5], 1):
            shape_t, scale_t = team_params[team]
            print(f"    {i}. {team:3s}: {mean:.2f} pts (shape={shape_t:.2f}, scale={scale_t:.2f})")

        print(f"\n  Bottom 5 teams:")
        for i, (team, mean) in enumerate(team_means[-5:], 1):
            shape_t, scale_t = team_params[team]
            print(f"    {i}. {team:3s}: {mean:.2f} pts (shape={shape_t:.2f}, scale={scale_t:.2f})")

        output_data["by_team"] = {
            team: {"shape": float(shape), "scale": float(scale)}
            for team, (shape, scale) in team_params.items()
        }

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n" + "=" * 80)
    print(f"✓ Saved calibrated parameters to: {output_path}")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review the fit statistics above")
    print("  2. Update fpts_distribution_calculations.py to load these params")
    print("  3. Run backtests to validate improvement")
    print()


if __name__ == "__main__":
    main()
