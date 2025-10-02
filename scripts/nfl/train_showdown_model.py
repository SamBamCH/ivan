#!/usr/bin/env python3
"""
scripts/nfl/train_showdown_model.py

Train showdown field models from historical lineup data.
Uses individual lineup records from ShowdownLineup table.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import (
    ShowdownLineup,
    ShowdownContestMetadata,
    HistoricalContest,
)


def build_training_dataset(session, min_entries: int = 1000) -> pd.DataFrame:
    """
    Build lineup-level training data directly from ShowdownLineup table.
    Each row is a unique lineup with its features and duplicate count.
    """
    
    # Get all lineups with contest context
    query = (
        session.query(
            ShowdownLineup.contest_id,
            ShowdownLineup.id,
            ShowdownLineup.duplicate_count,
            # Dimension 1: Correlations
            ShowdownLineup.correlation_score,
            ShowdownLineup.same_team_count,
            ShowdownLineup.qb_wr_stack,
            ShowdownLineup.rb_dst_negative,
            ShowdownLineup.stack_pattern,
            ShowdownLineup.game_stack_type,
            # Dimension 2: Captain ownership
            ShowdownLineup.cpt_ownership_pct,
            ShowdownLineup.cpt_ownership_percentile,
            # Dimension 3: Projection
            ShowdownLineup.projection_percentile,
            # Dimension 4: Ownership
            ShowdownLineup.ownership_percentile,
            ShowdownLineup.ownership_sum,
            # Dimension 5: Salary
            ShowdownLineup.salary_remaining,
            ShowdownLineup.salary_remaining_pct,
            # Contest info
            ShowdownContestMetadata.total_entries,
            ShowdownContestMetadata.entry_fee,
            ShowdownContestMetadata.unique_lineup_count,
            ShowdownContestMetadata.ownership_gini,
        )
        .join(
            ShowdownContestMetadata,
            ShowdownLineup.contest_id == ShowdownContestMetadata.contest_id
        )
        .filter(ShowdownContestMetadata.total_entries >= min_entries)
    )
    
    # Execute query and convert to DataFrame manually
    results = query.all()
    
    if not results:
        return pd.DataFrame()
    
    # Convert to DataFrame
    columns = [
        'contest_id', 'id', 'duplicate_count',
        'correlation_score', 'same_team_count', 'qb_wr_stack', 'rb_dst_negative',
        'stack_pattern', 'game_stack_type',
        'cpt_ownership_pct', 'cpt_ownership_percentile',
        'projection_percentile',
        'ownership_percentile', 'ownership_sum',
        'salary_remaining', 'salary_remaining_pct',
        'total_entries', 'entry_fee', 'unique_lineup_count', 'ownership_gini'
    ]
    
    df = pd.DataFrame(results, columns=columns)
    
    if df.empty:
        return df
    
    # Add derived features
    # Contest-level features (log scale for better model performance)
    df['log_total_entries'] = np.log10(df['total_entries'].clip(lower=1))
    
    # Interaction features
    df['cpt_own_x_total_own'] = df['cpt_ownership_percentile'] * df['ownership_percentile']
    df['near_cap'] = 1.0 - df['salary_remaining_pct']  # Higher when close to cap
    df['chalk_score'] = (
        df['cpt_ownership_percentile'] * 
        df['ownership_percentile'] * 
        df['projection_percentile']
    )
    
    # Target: selection probability (how often was this lineup chosen?)
    df['selection_probability'] = df['duplicate_count'] / df['total_entries']
    
    # Log transform of duplicate count for better distribution
    df['log_duplicate_count'] = np.log1p(df['duplicate_count'])
    
    return df


def train_duplication_model(
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[GradientBoostingRegressor, StandardScaler, list]:
    """
    Train duplication model that predicts duplicate count given lineup features.
    
    Returns:
        model: Trained GradientBoostingRegressor
        scaler: Feature scaler
        feature_cols: List of feature column names
    """
    
    print(f"\nTraining on {len(df)} lineups from {df['contest_id'].nunique()} contests")
    
    # Define feature columns
    feature_cols = [
        # Contest-level features
        'log_total_entries',
        'entry_fee',
        'ownership_gini',
        
        # Dimension 1: Correlation
        'correlation_score',
        'same_team_count',
        'qb_wr_stack',
        'rb_dst_negative',
        
        # Dimension 2: Captain ownership
        'cpt_ownership_percentile',
        
        # Dimension 3: Projection
        'projection_percentile',
        
        # Dimension 4: Total ownership
        'ownership_percentile',
        
        # Dimension 5: Salary
        'salary_remaining_pct',
        
        # Interactions
        'cpt_own_x_total_own',
        'near_cap',
        'chalk_score',
    ]
    
    # Prepare data
    X = df[feature_cols].fillna(0).values
    y = df['duplicate_count'].values  # Predict actual duplicate count
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=None
    )
    
    # Train model
    print("\nTraining duplication model...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate MAE for interpretability
    train_mae = np.mean(np.abs(train_pred - y_train))
    test_mae = np.mean(np.abs(test_pred - y_test))
    
    print(f"\n  Train R²: {train_score:.3f} | MAE: {train_mae:.2f} duplicates")
    print(f"  Test R²:  {test_score:.3f} | MAE: {test_mae:.2f} duplicates")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Diagnostics: prediction vs actual by percentile
    print("\nPrediction accuracy by duplicate count range:")
    df_test = pd.DataFrame({'actual': y_test, 'predicted': test_pred})
    
    for q, label in [(0.5, 'Median'), (0.75, '75th'), (0.90, '90th'), (0.95, '95th')]:
        threshold = np.quantile(y_test, q)
        mask = y_test >= threshold
        if mask.sum() > 0:
            mae_range = np.mean(np.abs(test_pred[mask] - y_test[mask]))
            print(f"  {label} percentile (>={threshold:.0f} dupes): MAE = {mae_range:.2f}")
    
    # Save models
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'showdown_duplication_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(output_dir / 'showdown_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_dir / 'showdown_feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"\nModels saved to {output_dir}")
    
    return model, scaler, feature_cols


def analyze_dataset(df: pd.DataFrame):
    """Print diagnostic information about the training data."""
    
    print("\n" + "="*70)
    print("TRAINING DATA ANALYSIS")
    print("="*70)
    
    print(f"\nDataset size: {len(df):,} unique lineups")
    print(f"Contests: {df['contest_id'].nunique()}")
    
    print(f"\nContest sizes:")
    print(f"  Mean entries: {df['total_entries'].mean():,.0f}")
    print(f"  Range: {df['total_entries'].min():,} - {df['total_entries'].max():,}")
    
    print(f"\nDuplicate count distribution:")
    print(f"  Mean: {df['duplicate_count'].mean():.1f}")
    print(f"  Median: {df['duplicate_count'].median():.0f}")
    print(f"  Max: {df['duplicate_count'].max():,.0f}")
    
    quantiles = df['duplicate_count'].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    print(f"\nDuplicate count percentiles:")
    for q, val in quantiles.items():
        print(f"  {q*100:.0f}th: {val:.0f}")
    
    print(f"\nFeature completeness:")
    for col in ['correlation_score', 'cpt_ownership_percentile', 'projection_percentile', 
                'ownership_percentile', 'salary_remaining_pct']:
        pct_valid = (df[col].notna()).mean() * 100
        print(f"  {col}: {pct_valid:.1f}% valid")
    
    # Check for any red flags
    issues = []
    if df['duplicate_count'].min() < 1:
        issues.append("Some lineups have duplicate_count < 1")
    if (df['cpt_ownership_percentile'] < 0).any() or (df['cpt_ownership_percentile'] > 1).any():
        issues.append("Captain ownership percentile out of [0,1] range")
    if df['salary_remaining_pct'].isna().mean() > 0.1:
        issues.append("More than 10% missing salary data")
    
    if issues:
        print(f"\nWarnings:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print(f"\n✓ No data quality issues detected")
    
    print("="*70)


def build_showdown_models(db_url: str = None, model_dir: str | Path = "data/nfl/models"):
    """
    Callable function for integration into build_models.py pipeline.
    """
    model_dir = Path(model_dir)
    
    engine = init_db("nfl", db_url)
    session = get_session(engine)
    
    # Check if we have any showdown lineup data
    lineup_count = session.query(func.count(ShowdownLineup.id)).scalar()
    
    if lineup_count == 0:
        print("  No showdown lineups found - skipping showdown model training")
        print("  Run get_contest_data.py first to collect historical data")
        return
    
    print(f"  Found {lineup_count:,} showdown lineups in database")
    
    # Build training dataset
    df = build_training_dataset(session, min_entries=1000)
    
    if df.empty or len(df) < 100:
        print(f"  Insufficient training data ({len(df)} lineups) - skipping model training")
        print("  Need at least 100 lineups from contests with 1000+ entries")
        return
    
    # Analyze data quality
    analyze_dataset(df)
    
    # Train and save model
    model, scaler, feature_cols = train_duplication_model(df, model_dir)
    
    print(f"\n  ✓ Showdown models trained on {len(df):,} lineups")
    session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-url', default=None)
    parser.add_argument('--model-dir', default='data/nfl/models')
    parser.add_argument('--min-entries', type=int, default=1000,
                       help='Minimum contest size to include in training')
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    print("→ Loading historical showdown lineup data")
    engine = init_db("nfl", args.db_url)
    session = get_session(engine)
    
    # Build training dataset
    print("→ Building training dataset from ShowdownLineup table")
    df = build_training_dataset(session, min_entries=args.min_entries)
    
    if df.empty:
        print("ERROR: No training data found. Run historical contest pipeline first.")
        print("       (python scripts/nfl/get_contest_data.py)")
        return
    
    # Analyze dataset
    analyze_dataset(df)
    
    # Train models
    print("\n→ Training showdown field model")
    model, scaler, feature_cols = train_duplication_model(df, model_dir)
    
    print("\n✓ Training complete!")
    print("\nNext steps:")
    print("  1. Models are stored in data/nfl/models/")
    print("  2. Enable learned sampling in config/nfl.json:")
    print('     {"showdown": {"use_learned_model": true}}')
    print("  3. Run simulations: python scripts/nfl/sim.py")
    print("  4. Retrain weekly as you collect more historical data")


if __name__ == '__main__':
    main()