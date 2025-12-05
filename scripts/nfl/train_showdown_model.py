#!/usr/bin/env python3
"""
Two-Stage Hierarchical Showdown Duplication Model

This approach separates contest-level and lineup-level factors:

Stage 1: Contest Distribution Model
  - Input: Contest-level parameters (field_size, ownership_concentration, etc.)
  - Output: Predicted duplication distribution quantiles (Q25, Q50, Q75, Q90, Q95, Q99)
  - This defines the "landscape" for each contest

Stage 2: Lineup Positioning Model
  - Input: Lineup features (salary_remaining_pct, ownership_sum, cpt_ownership_pct)
  - Output: Percentile rank (0.0 to 1.0) of where this lineup sits in the distribution
  - This places each lineup on the landscape

At inference:
  1. Given a contest, Stage 1 predicts its quantile distribution
  2. For each lineup, Stage 2 predicts its percentile rank
  3. Interpolate: percentile rank + quantile distribution → final duplicate count
"""

import argparse
from datetime import datetime
import pickle
from pathlib import Path
from typing import Tuple, Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.special import logit, expit
from sqlalchemy import func

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import (
    ShowdownLineup,
    HistoricalContest,
    ShowdownContestMetadata,
    ProjectionSnapshot,
)


def calculate_gini(values: np.ndarray) -> float:
    """Calculate Gini coefficient for ownership concentration."""
    if len(values) < 2:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    gini = 1.0 - (2.0 * cumsum[-1] - cumsum[0]) / ((n - 1) * cumsum[-1])
    return float(np.clip(gini, 0.0, 1.0))


def build_contest_level_data(session, min_entries: int = 5000) -> pd.DataFrame:
    """
    For each contest, compute:
    - Macro contest features (total_entries, entry_fee, ownership_gini)
    - Empirical quantiles of duplication distribution
    """
    
    # Pull total_entries and entry_fee from HistoricalContest (source of truth)
    query = (
        session.query(
            HistoricalContest.contest_id,
            HistoricalContest.total_entries,
            HistoricalContest.entry_fee,
            ShowdownContestMetadata.unique_lineup_count,
            ShowdownContestMetadata.max_dupe_count,
            ShowdownContestMetadata.median_dupe_count
        )
        .join(
            HistoricalContest,
            ShowdownContestMetadata.contest_id == HistoricalContest.contest_id,
        )
        .filter(HistoricalContest.total_entries >= min_entries)
    )
    
    contests = query.all()
    if not contests:
        return pd.DataFrame()
    
    contest_quantiles = []
    
    for contest_id, total_entries, entry_fee, unique_lineups, max_dupes, median_dupes, own_gini in contests:
        # Get all lineups for this contest
        lineup_query = (
            session.query(ShowdownLineup.duplicate_count)
            .filter(ShowdownLineup.contest_id == contest_id)
        )
        dupes = np.array([x[0] for x in lineup_query.all()], dtype=int)

        print(dupes)
        
        if len(dupes) < 10:  # Need minimum data
            continue
        
        # Empirical quantiles (targets for Stage 1)
        quantiles = {
            'q25': float(np.percentile(dupes, 25)),
            'q50': float(np.percentile(dupes, 50)),
            'q75': float(np.percentile(dupes, 75)),
            'q90': float(np.percentile(dupes, 90)),
            'q95': float(np.percentile(dupes, 95)),
            'q99': float(np.percentile(dupes, 99)),
        }
        
        contest_quantiles.append({
            'contest_id': contest_id,
            'total_entries': total_entries,
            'entry_fee': entry_fee if entry_fee is not None else 5.0,
            'ownership_gini': own_gini if own_gini is not None else 0.5,
            'max_duplicate_count': max_dupes,
            'median_duplicate_count': median_dupes,
            **quantiles
        })
    
    df_contest = pd.DataFrame(contest_quantiles)
    return df_contest


def build_lineup_level_data(session, min_entries: int = 5000) -> pd.DataFrame:
    """
    For each lineup, compute:
    - Micro lineup features (captain ownership, projection, salary)
    - Contest context
    - Empirical percentile rank within its contest
    
    Key insight: We're training on data GROUPED BY CONTEST, meaning the model
    learns how lineup features correlate with duplication WITHIN each contest.
    This is correct because at runtime we're also predicting within a single contest.
    """
    
    # Get all lineups from contests >= min_entries
    # Pull only the minimal lineup fields available, and filter contests
    lineup_rows = (
        session.query(
            ShowdownLineup.id,
            ShowdownLineup.contest_id,
            ShowdownLineup.dupe_count,
            ShowdownLineup.salary_remaining,
            ShowdownLineup.cpt_player_id,
            ShowdownLineup.flex_player_ids,
            HistoricalContest.total_entries,
        )
        .join(
            HistoricalContest,
            ShowdownLineup.contest_id == HistoricalContest.contest_id
        )
        .filter(HistoricalContest.total_entries >= min_entries)
    ).all()

    if not lineup_rows:
        return pd.DataFrame()

    # Collect all roster_slot_ids appearing in selected lineups
    all_pids: set[int] = set()
    for _, _, _, _, cpt_id, flex_ids, _ in lineup_rows:
        all_pids.add(int(cpt_id))
        for pid in (flex_ids or []):
            all_pids.add(int(pid))

    if not all_pids:
        return pd.DataFrame()

    # Fetch ProjectionSnapshot for these roster_slot_ids
    # Use the latest snapshot per roster_slot_id
    snapshot_rows = (
        session.query(
            ProjectionSnapshot.roster_slot_id,
            ProjectionSnapshot.snapshot_date,
            ProjectionSnapshot.fpts_mean,
            ProjectionSnapshot.ownership_proj,
        )
        .filter(ProjectionSnapshot.roster_slot_id.in_(list(all_pids)))
        .all()
    )

    if not snapshot_rows:
        return pd.DataFrame()

    # Reduce to latest snapshot per roster slot
    latest_by_slot: dict[int, tuple[float, float]] = {}
    seen_date: dict[int, datetime] = {}
    for slot_id, snap_dt, fpts_mean, own_proj in snapshot_rows:
        prev = seen_date.get(slot_id)
        if prev is None or (snap_dt and snap_dt > prev):
            seen_date[slot_id] = snap_dt
            latest_by_slot[slot_id] = (
                float(fpts_mean or 0.0),
                float(own_proj or 0.0),
            )

    # Build feature rows; drop any lineup missing snapshot data for any player id
    feat_rows: list[dict] = []
    for lineup_id, contest_id, dupe_count, salary_remaining, cpt_id, flex_ids, total_entries in lineup_rows:
        player_ids = [int(cpt_id)] + [int(x) for x in (flex_ids or [])]
        # Ensure we have snapshot for all players
        if any(pid not in latest_by_slot for pid in player_ids):
            continue

        # Build projection and ownership vectors
        fpts_vec = [latest_by_slot[pid][0] for pid in player_ids]
        own_vec = [latest_by_slot[pid][1] for pid in player_ids]

        projection_sum = float(np.sum(fpts_vec))
        cpt_ownership_pct = float(own_vec[0])  # first is CPT
        # salary remaining percent of cap (assume DK Showdown cap=50000)
        salary_remaining_pct = float(np.clip(salary_remaining, 0, 50000)) / 50000.0

        feat_rows.append({
            'lineup_id': lineup_id,
            'contest_id': contest_id,
            'duplicate_count': dupe_count,
            'total_entries': total_entries,
            'salary_remaining_pct': salary_remaining_pct,
            'projection_sum': projection_sum,
            'cpt_ownership_pct': cpt_ownership_pct,
        })

    df = pd.DataFrame(feat_rows)
    if df.empty:
        return df

    # Normalize projection within each contest to 0-1 range
    def _normalize_group(series: pd.Series) -> pd.Series:
        if series.empty:
            return series
        mn = series.min()
        mx = series.max()
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - mn) / (mx - mn)

    df['projection_normalized'] = df.groupby('contest_id')['projection_sum'].transform(_normalize_group)
    
    # Calculate percentile rank within each contest using transform (preserves index)
    def compute_percentile_rank(x):
        """Compute percentile rank for a group."""
        if len(x) <= 1:
            return pd.Series([0.5] * len(x), index=x.index)
        ranks = x.rank(pct=True)
        # Normalize to [0, 1] range more carefully
        return ranks
    
    df['percentile_rank'] = df.groupby('contest_id')['duplicate_count'].transform(
        compute_percentile_rank
    )
    df['percentile_rank'] = df['percentile_rank'].clip(0.0, 1.0)
    
    # Add log contest size
    df['log_total_entries'] = np.log10(df['total_entries'].clip(lower=1))
    
    # Drop rows with missing critical values
    df = df.dropna(subset=[
        'salary_remaining_pct', 'projection_normalized', 'cpt_ownership_pct',
        'percentile_rank'
    ])
    
    return df


def train_stage1_model(
    df_contest: pd.DataFrame,
    output_dir: Path,
) -> Tuple[MultiOutputRegressor, StandardScaler, list]:
    """
    Train Stage 1: Contest Distribution Model
    
    Predicts quantiles (Q25, Q50, Q75, Q90, Q95, Q99) from macro contest features.
    
    Features:
      1. total_entries - Contest size
      2. entry_fee - Entry fee (contest difficulty proxy)
      3. ownership_gini - Ownership concentration
    
    Returns:
        model: Trained MultiOutputRegressor
        scaler: StandardScaler for features
        feature_cols: List of feature column names
    """
    
    print(f"\n{'='*70}")
    print("STAGE 1: CONTEST DISTRIBUTION MODEL (MACRO FEATURES)")
    print(f"{'='*70}")
    print(f"Training on {len(df_contest):,} contests")
    
    feature_cols = [
        'total_entries',
        'entry_fee',
        'ownership_gini',
    ]
    
    target_cols = ['q25', 'q50', 'q75', 'q90', 'q95', 'q99']
    
    print(f"\nFeatures: {feature_cols}")
    print(f"Targets: {target_cols}")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    for col in feature_cols:
        print(f"  {col}:")
        print(f"    Mean: {df_contest[col].mean():.3f}")
        print(f"    Std:  {df_contest[col].std():.3f}")
        print(f"    Min:  {df_contest[col].min():.3f}")
        print(f"    Max:  {df_contest[col].max():.3f}")
    
    # Target statistics
    print(f"\nTarget Statistics (duplication quantiles):")
    for col in target_cols:
        print(f"  {col}: mean={df_contest[col].mean():.1f}, "
              f"min={df_contest[col].min():.1f}, max={df_contest[col].max():.1f}")
    
    # Prepare data
    X = df_contest[feature_cols].values
    Y = df_contest[target_cols].values
    
    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(X_train):,} contests")
    print(f"  Test:  {len(X_test):,} contests")
    
    # Train multi-output model (one model per quantile)
    print(f"\nTraining MultiOutputRegressor (predicting 6 quantiles)...")
    model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
            verbose=0,
        )
    )
    
    model.fit(X_train_scaled, Y_train)
    
    # Evaluate
    print(f"\n{'='*70}")
    print("STAGE 1 PERFORMANCE")
    print(f"{'='*70}")
    
    Y_train_pred = model.predict(X_train_scaled)
    Y_test_pred = model.predict(X_test_scaled)
    
    print(f"\nTrain MAE (per quantile):")
    for i, col in enumerate(target_cols):
        mae = mean_absolute_error(Y_train[:, i], Y_train_pred[:, i])
        print(f"  {col}: {mae:.2f} duplicates")
    
    print(f"\nTest MAE (per quantile):")
    for i, col in enumerate(target_cols):
        mae = mean_absolute_error(Y_test[:, i], Y_test_pred[:, i])
        print(f"  {col}: {mae:.2f} duplicates")
    
    print(f"\nOverall Test R²: {r2_score(Y_test, Y_test_pred):.4f}")
    
    # Save models
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'showdown_stage1_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(output_dir / 'showdown_stage1_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_dir / 'showdown_stage1_feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"\n✓ Stage 1 models saved to {output_dir}")
    
    return model, scaler, feature_cols


def train_stage2_model(
    df_lineup: pd.DataFrame,
    output_dir: Path,
) -> Tuple[GradientBoostingRegressor, StandardScaler, list]:
    """
    Train Stage 2: Lineup Positioning Model
    
    Predicts percentile rank (0.0-1.0) from 3 lineup-specific features only.
    This tells us where each lineup sits in the duplication landscape,
    independent of contest size (contest effects handled by Stage 1).
    
    Features (3 only):
      1. cpt_ownership_pct - Captain ownership percentage
      2. projection_normalized - Lineup projection (0-1 normalized within contest)
      3. salary_remaining_pct - % of salary cap remaining
    
    Note: Using projection_normalized instead of ownership_sum to capture
    lineup quality/favorability rather than just popularity. At runtime,
    we'll use INDEPENDENT samples (not correlated) to reduce QB captain bias.
    
    Returns:
        model: Trained GradientBoostingRegressor
        scaler: StandardScaler for features
        feature_cols: List of feature column names
    """
    
    print(f"\n{'='*70}")
    print("STAGE 2: LINEUP POSITIONING MODEL (MICRO FEATURES)")
    print(f"{'='*70}")
    print(f"Training on {len(df_lineup):,} lineups from {df_lineup['contest_id'].nunique()} contests")
    
    feature_cols = [
        'cpt_ownership_pct',
        'projection_normalized',
        'salary_remaining_pct',
    ]
    
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Target: percentile_rank (0.0-1.0) within each contest")
    print(f"        [Contest-level effects handled by Stage 1]")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    for col in feature_cols:
        print(f"  {col}:")
        print(f"    Mean: {df_lineup[col].mean():.3f}")
        print(f"    Std:  {df_lineup[col].std():.3f}")
        print(f"    Min:  {df_lineup[col].min():.3f}")
        print(f"    Max:  {df_lineup[col].max():.3f}")
    
    # Target statistics
    print(f"\nTarget Statistics (percentile rank):")
    print(f"  Mean:   {df_lineup['percentile_rank'].mean():.3f}")
    print(f"  Median: {df_lineup['percentile_rank'].median():.3f}")
    print(f"  Std:    {df_lineup['percentile_rank'].std():.3f}")
    print(f"  Min:    {df_lineup['percentile_rank'].min():.3f}")
    print(f"  Max:    {df_lineup['percentile_rank'].max():.3f}")
    
    # Prepare data
    X = df_lineup[feature_cols].values
    y = df_lineup['percentile_rank'].values
    
    # Apply logit transform for better calibration (clip to avoid infinities)
    y_logit = logit(np.clip(y, 0.001, 0.999))
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_logit, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(X_train):,} lineups")
    print(f"  Test:  {len(X_test):,} lineups")
    
    # Train model
    print(f"\nTraining GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_leaf=8,
        random_state=42,
        verbose=0,
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print(f"\n{'='*70}")
    print("STAGE 2 PERFORMANCE")
    print(f"{'='*70}")
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Convert back from logit space for metrics
    y_train_actual = expit(y_train)
    y_test_actual = expit(y_test)
    y_train_pred_actual = expit(y_train_pred)
    y_test_pred_actual = expit(y_test_pred)
    
    print(f"\nLogit-Space Metrics (what model predicts):")
    print(f"  Train MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"  Test MAE:  {mean_absolute_error(y_test, y_test_pred):.4f}")
    print(f"  Train R²:  {r2_score(y_train, y_train_pred):.4f}")
    print(f"  Test R²:   {r2_score(y_test, y_test_pred):.4f}")
    
    print(f"\nPercentile-Space Metrics (interpretable):")
    print(f"  Train MAE: {mean_absolute_error(y_train_actual, y_train_pred_actual):.4f}")
    print(f"  Test MAE:  {mean_absolute_error(y_test_actual, y_test_pred_actual):.4f}")
    
    # Feature importance
    print(f"\nFeature Importance:")
    importances = model.feature_importances_
    for i, (col, imp) in enumerate(sorted(zip(feature_cols, importances), 
                                         key=lambda x: x[1], reverse=True)):
        print(f"  {i+1}. {col}: {imp:.3f}")
    
    # Calibration analysis: distribution of predicted vs actual percentiles
    print(f"\nCalibration Analysis (Test Set):")
    print(f"  Predicted percentile range: [{y_test_pred_actual.min():.3f}, {y_test_pred_actual.max():.3f}]")
    print(f"  Actual percentile range:    [{y_test_actual.min():.3f}, {y_test_actual.max():.3f}]")
    
    # Save models
    with open(output_dir / 'showdown_stage2_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(output_dir / 'showdown_stage2_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_dir / 'showdown_stage2_feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"\n✓ Stage 2 models saved to {output_dir}")
    print(f"  Features: {feature_cols}")
    
    return model, scaler, feature_cols


def analyze_dataset(df_contest: pd.DataFrame, df_lineup: pd.DataFrame):
    """Print comprehensive diagnostic information."""
    
    print(f"\n{'='*70}")
    print("TRAINING DATA ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nContest-Level Data:")
    print(f"  Total contests: {len(df_contest):,}")
    print(f"  Entry fee range: ${df_contest['entry_fee'].min():.2f} - ${df_contest['entry_fee'].max():.2f}")
    print(f"  Contest size range: {df_contest['total_entries'].min():,} - {df_contest['total_entries'].max():,}")
    print(f"  Ownership Gini range: {df_contest['ownership_gini'].min():.3f} - {df_contest['ownership_gini'].max():.3f}")
    print(f"  Average max dupes per contest: {df_contest['max_duplicate_count'].mean():.1f}")
    
    print(f"\nLineup-Level Data:")
    print(f"  Total lineups: {len(df_lineup):,}")
    print(f"  Unique contests: {df_lineup['contest_id'].nunique()}")
    print(f"  Duplicate count range: {df_lineup['duplicate_count'].min():.0f} to {df_lineup['duplicate_count'].max():.0f}")
    
    print(f"\nPercentile Rank Distribution (among lineups within contest):")
    quantiles = df_lineup['percentile_rank'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
    for q, val in quantiles.items():
        print(f"  {q*100:.0f}th percentile: {val:.3f}")
    
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Train two-stage hierarchical showdown duplication model"
    )
    parser.add_argument('--db-url', default=None,
                       help='Database URL override')
    parser.add_argument('--model-dir', default='data/nfl/models',
                       help='Directory to save models')
    parser.add_argument('--min-entries', type=int, default=5000,
                       help='Minimum contest size to include in training')
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    print("↓ Loading historical showdown data")
    engine = init_db("nfl", args.db_url)
    session = get_session(engine)
    
    # Check data availability
    lineup_count = session.query(func.count(ShowdownLineup.id)).scalar()
    contest_count = session.query(func.count(ShowdownContestMetadata.contest_id)).scalar()
    
    if lineup_count == 0 or contest_count == 0:
        print("ERROR: No showdown data found - run contest ingestion first")
        return
    
    print(f"  Found {lineup_count:,} showdown lineups")
    print(f"  Found {contest_count:,} contest metadata records")
    
    # Build training datasets
    print("↓ Building contest-level data")
    df_contest = build_contest_level_data(session, min_entries=args.min_entries)
    
    if df_contest.empty or len(df_contest) < 10:
        print(f"ERROR: Insufficient contest data ({len(df_contest)} contests)")
        print(f"       Need at least 10 contests with {args.min_entries}+ entries")
        return
    
    print("↓ Building lineup-level data")
    df_lineup = build_lineup_level_data(session, min_entries=args.min_entries)
    
    if df_lineup.empty or len(df_lineup) < 100:
        print(f"ERROR: Insufficient lineup data ({len(df_lineup)} lineups)")
        print(f"       Need at least 100 lineups from contests with {args.min_entries}+ entries")
        return
    
    # Analyze
    analyze_dataset(df_contest, df_lineup)
    
    # Train both stages
    print("↓ Training Stage 1: Contest Distribution Model")
    model1, scaler1, features1 = train_stage1_model(df_contest, model_dir)
    
    print("\n↓ Training Stage 2: Lineup Positioning Model")
    model2, scaler2, features2 = train_stage2_model(df_lineup, model_dir)
    
    print(f"\n{'='*70}")
    print("✓ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print("\nTwo-Stage Model Summary:")
    print("\n  STAGE 1: Contest Distribution Model")
    print("    - Predicts: Quantiles (Q25, Q50, Q75, Q90, Q95, Q99)")
    print("    - From: Contest features (field_size, ownership_concentration, etc.)")
    print("    - Uses: MultiOutputRegressor with GradientBoosting")
    print("\n  STAGE 2: Lineup Positioning Model")
    print("    - Predicts: Percentile rank (0.0-1.0)")
    print("    - From: Lineup features (salary_remaining_pct, ownership_sum, etc.)")
    print("    - Uses: GradientBoostingRegressor with logit transform")
    print("\n  At Inference:")
    print("    1. Stage 1 predicts quantile distribution for contest")
    print("    2. Stage 2 predicts percentile rank for each lineup")
    print("    3. Interpolate: rank + quantiles → duplicate count prediction")
    print("\nModels saved to: " + str(model_dir))


if __name__ == '__main__':
    main()