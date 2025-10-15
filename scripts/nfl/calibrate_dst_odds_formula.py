#!/usr/bin/env python3
"""
Calibrate DST projection formula from historical odds and actual outcomes.

This script fits a regression model to predict DST fantasy points from game odds,
replacing the hardcoded coefficients in the current formula:
    mean = 11.0 - 0.25*opp_tt + 0.15*max(-team_spread, 0) + 0.05*(46.0 - total)

Outputs:
- Fitted regression coefficients for odds-based DST formula
- Standard deviation (residual std from regression)
- Saves to data/nfl/models/dst_odds_params.json
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import text

from ivan.database.session import init_db, get_session


def load_dst_odds_data(session, min_season: int = 2020) -> pd.DataFrame:
    """
    Load historical DST fantasy points with corresponding game odds.

    Joins nfl_dst_game with game_odds to get:
    - DST fantasy points (actual outcome)
    - Game total (over/under)
    - Team spread (positive if underdog, negative if favorite)
    - Opponent implied team total

    Args:
        session: Database session
        min_season: Earliest season to include

    Returns:
        DataFrame with columns: season, week, team, opponent, dst_points,
                                 total, home_spread, is_home, team_spread, opp_total
    """
    # First, get all DST game data
    dst_query = text("""
        SELECT
            season,
            week,
            team,
            opponent,
            dst_points
        FROM nfl_dst_game
        WHERE season >= :min_season
          AND dst_points IS NOT NULL
        ORDER BY season DESC, week DESC
    """)

    dst_df = pd.read_sql(dst_query, session.bind, params={"min_season": min_season})

    if dst_df.empty:
        raise RuntimeError(f"No DST data found for seasons >= {min_season}")

    print(f"Loaded {len(dst_df):,} DST game records")

    # Now join with game odds
    # Note: game_odds uses home_abbr/away_abbr, we need to match with team/opponent
    odds_query = text("""
        SELECT DISTINCT
            season,
            week,
            home_abbr,
            away_abbr,
            market,
            side,
            point,
            last_update
        FROM game_odds
        WHERE market IN ('totals', 'spreads')
          AND season >= :min_season
        ORDER BY last_update DESC
    """)

    odds_df = pd.read_sql(odds_query, session.bind, params={"min_season": min_season})

    if odds_df.empty:
        print("⚠️  Warning: No game odds data found. DST calibration requires odds data.")
        return pd.DataFrame()

    print(f"Loaded {len(odds_df):,} game odds records")

    # Get latest totals per game
    totals = (odds_df[odds_df['market'] == 'totals']
              .sort_values('last_update', ascending=False)
              .groupby(['season', 'week', 'home_abbr', 'away_abbr'], as_index=False)
              .first()[['season', 'week', 'home_abbr', 'away_abbr', 'point']]
              .rename(columns={'point': 'total'}))

    # Get latest home spreads per game
    spreads_home = (odds_df[(odds_df['market'] == 'spreads') & (odds_df['side'] == 'home')]
                    .sort_values('last_update', ascending=False)
                    .groupby(['season', 'week', 'home_abbr', 'away_abbr'], as_index=False)
                    .first()[['season', 'week', 'home_abbr', 'away_abbr', 'point']]
                    .rename(columns={'point': 'home_spread'}))

    # Merge totals and spreads
    game_odds = totals.merge(spreads_home, on=['season', 'week', 'home_abbr', 'away_abbr'], how='inner')

    print(f"Matched {len(game_odds):,} games with both total and spread")

    # Now match DST data to odds (team can be home or away)
    merged = []

    for _, dst_row in dst_df.iterrows():
        season, week, team, opponent = dst_row['season'], dst_row['week'], dst_row['team'], dst_row['opponent']

        # Try matching as home team
        home_match = game_odds[
            (game_odds['season'] == season) &
            (game_odds['week'] == week) &
            (game_odds['home_abbr'] == team) &
            (game_odds['away_abbr'] == opponent)
        ]

        if not home_match.empty:
            odds_row = home_match.iloc[0]
            total = float(odds_row['total'])
            home_spread = float(odds_row['home_spread'])
            is_home = True
            team_spread = home_spread
            # Implied totals: home_tt = (total - home_spread) / 2, away_tt = (total + home_spread) / 2
            opp_total = (total + home_spread) / 2

            merged.append({
                'season': season, 'week': week, 'team': team, 'opponent': opponent,
                'dst_points': dst_row['dst_points'],
                'total': total, 'home_spread': home_spread, 'is_home': is_home,
                'team_spread': team_spread, 'opp_total': opp_total
            })
            continue

        # Try matching as away team
        away_match = game_odds[
            (game_odds['season'] == season) &
            (game_odds['week'] == week) &
            (game_odds['home_abbr'] == opponent) &
            (game_odds['away_abbr'] == team)
        ]

        if not away_match.empty:
            odds_row = away_match.iloc[0]
            total = float(odds_row['total'])
            home_spread = float(odds_row['home_spread'])
            is_home = False
            team_spread = -home_spread  # Away spread is negative of home spread
            opp_total = (total - home_spread) / 2

            merged.append({
                'season': season, 'week': week, 'team': team, 'opponent': opponent,
                'dst_points': dst_row['dst_points'],
                'total': total, 'home_spread': home_spread, 'is_home': is_home,
                'team_spread': team_spread, 'opp_total': opp_total
            })

    result = pd.DataFrame(merged)

    print(f"\n✓ Matched {len(result):,} DST games with odds data ({100*len(result)/len(dst_df):.1f}% of DST games)")

    return result


def fit_dst_odds_formula(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Fit regression model: dst_points ~ opp_total + team_favored + (game_total - 46)

    Current formula:
        mean = 11.0 - 0.25*opp_tt + 0.15*max(-team_spread, 0) + 0.05*(46.0 - total)

    We'll fit:
        dst_points = β0 + β1*opp_total + β2*team_favored + β3*total_deviation + ε

    Where:
        opp_total = opponent's implied team total
        team_favored = max(-team_spread, 0)  [positive when team is favored]
        total_deviation = (46.0 - total)  [positive when game total is low]

    Args:
        df: DataFrame with dst_points, opp_total, team_spread, total

    Returns:
        (coefficients_dict, diagnostics_dict)
    """
    if df.empty or len(df) < 50:
        raise ValueError(f"Insufficient data ({len(df)} games) to fit DST formula")

    # Create features
    X = pd.DataFrame({
        'opp_total': df['opp_total'],
        'team_favored': np.maximum(-df['team_spread'], 0),  # Positive when favored
        'total_deviation': 46.0 - df['total'],
    })

    y = df['dst_points'].values

    # Fit linear regression
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # Get coefficients
    intercept = float(model.intercept_)
    coef_opp_total = float(model.coef_[0])
    coef_favored = float(model.coef_[1])
    coef_total_dev = float(model.coef_[2])

    # Predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Diagnostics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    residual_std = float(np.std(residuals, ddof=1))

    coefficients = {
        "intercept": intercept,
        "opp_total": coef_opp_total,
        "team_favored": coef_favored,
        "total_deviation": coef_total_dev,
        "residual_std": residual_std,
    }

    diagnostics = {
        "n_games": int(len(df)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "residual_std": float(residual_std),
        "actual_mean": float(y.mean()),
        "actual_std": float(y.std(ddof=1)),
        "predicted_mean": float(y_pred.mean()),
        "predicted_std": float(y_pred.std(ddof=1)),
    }

    return coefficients, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Calibrate DST odds formula")
    parser.add_argument("--config", default="config/nfl.json", help="Path to config file")
    parser.add_argument("--min-season", type=int, default=2020, help="Earliest season to include")
    parser.add_argument("--output", default="data/nfl/models/dst_odds_params.json", help="Output path")
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
    print("DST ODDS FORMULA CALIBRATION")
    print("=" * 80)
    print()

    # Load historical data with odds
    df = load_dst_odds_data(session, min_season=args.min_season)

    if df.empty:
        print("\n❌ Error: No data available for calibration")
        print("   Make sure you have both nfl_dst_game and game_odds tables populated")
        return

    print()
    print("-" * 80)
    print("DATA SUMMARY")
    print("-" * 80)
    print(f"  DST points:     {df['dst_points'].mean():.2f} ± {df['dst_points'].std():.2f}")
    print(f"  Opponent total: {df['opp_total'].mean():.2f} ± {df['opp_total'].std():.2f}")
    print(f"  Game total:     {df['total'].mean():.2f} ± {df['total'].std():.2f}")
    print(f"  Team favored:   {(df['team_spread'] < 0).sum()} games ({100*(df['team_spread'] < 0).sum()/len(df):.1f}%)")

    # Fit formula
    print()
    print("-" * 80)
    print("REGRESSION MODEL FIT")
    print("-" * 80)

    coefficients, diagnostics = fit_dst_odds_formula(df)

    print(f"\n✓ Fitted regression model:")
    print(f"  DST_points = {coefficients['intercept']:.3f}")
    print(f"             + ({coefficients['opp_total']:.3f}) * opp_total")
    print(f"             + ({coefficients['team_favored']:.3f}) * max(-team_spread, 0)")
    print(f"             + ({coefficients['total_deviation']:.3f}) * (46 - game_total)")
    print(f"             + ε  [where σ(ε) = {coefficients['residual_std']:.2f}]")

    print(f"\n  Model performance:")
    print(f"    R² = {diagnostics['r2']:.3f}")
    print(f"    MAE = {diagnostics['mae']:.2f} pts")
    print(f"    RMSE = {diagnostics['rmse']:.2f} pts")
    print(f"    Residual std = {diagnostics['residual_std']:.2f} pts")

    # Compare to old hardcoded formula
    print()
    print("-" * 80)
    print("COMPARISON TO OLD HARDCODED FORMULA")
    print("-" * 80)

    OLD_INTERCEPT = 11.0
    OLD_OPP_TOTAL = -0.25
    OLD_FAVORED = 0.15
    OLD_TOTAL_DEV = 0.05
    OLD_STD = 5.0

    print(f"\n  Old formula (hardcoded):")
    print(f"    DST_points = {OLD_INTERCEPT:.2f} + ({OLD_OPP_TOTAL:.2f})*opp_tt + ({OLD_FAVORED:.2f})*favored + ({OLD_TOTAL_DEV:.2f})*(46-total)")
    print(f"    σ = {OLD_STD:.2f}")

    print(f"\n  New formula (calibrated):")
    print(f"    DST_points = {coefficients['intercept']:.2f} + ({coefficients['opp_total']:.2f})*opp_tt + ({coefficients['team_favored']:.2f})*favored + ({coefficients['total_deviation']:.2f})*(46-total)")
    print(f"    σ = {coefficients['residual_std']:.2f}")

    # Compute predictions with both formulas for comparison
    df_copy = df.copy()
    df_copy['team_favored'] = np.maximum(-df_copy['team_spread'], 0)
    df_copy['total_dev'] = 46.0 - df_copy['total']

    old_pred = (OLD_INTERCEPT +
                OLD_OPP_TOTAL * df_copy['opp_total'] +
                OLD_FAVORED * df_copy['team_favored'] +
                OLD_TOTAL_DEV * df_copy['total_dev'])
    old_pred = np.clip(old_pred, -1.5, 14.0)  # Original formula has clipping

    new_pred = (coefficients['intercept'] +
                coefficients['opp_total'] * df_copy['opp_total'] +
                coefficients['team_favored'] * df_copy['team_favored'] +
                coefficients['total_deviation'] * df_copy['total_dev'])

    old_mae = mean_absolute_error(df_copy['dst_points'], old_pred)
    new_mae = diagnostics['mae']

    print(f"\n  MAE comparison:")
    print(f"    Old formula: {old_mae:.2f} pts")
    print(f"    New formula: {new_mae:.2f} pts")
    print(f"    Improvement: {old_mae - new_mae:+.2f} pts ({100*(old_mae - new_mae)/old_mae:+.1f}%)")

    # Save output
    output_data = {
        "coefficients": coefficients,
        "diagnostics": diagnostics,
        "old_hardcoded": {
            "intercept": OLD_INTERCEPT,
            "opp_total": OLD_OPP_TOTAL,
            "team_favored": OLD_FAVORED,
            "total_deviation": OLD_TOTAL_DEV,
            "std": OLD_STD,
        },
        "metadata": {
            "min_season": args.min_season,
            "n_games": int(len(df)),
            "seasons": f"{df['season'].min()}-{df['season'].max()}",
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n" + "=" * 80)
    print(f"✓ Saved calibrated parameters to: {output_path}")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review the fit statistics and coefficient changes above")
    print("  2. Update fpts_distribution_calculations.py to load these params")
    print("  3. Run backtests to validate improvement")
    print()


if __name__ == "__main__":
    main()
