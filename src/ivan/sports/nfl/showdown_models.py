"""
Helper functions for building the 5-dimensional lineup prediction model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sqlalchemy.orm import Session
from ivan.sports.nfl.models import (
    ShowdownLineup,
    ShowdownContestMetadata,
    ShowdownPlayer,
)


def build_training_dataset(session: Session, min_entries: int = 1000) -> pd.DataFrame:
    """
    Build a training dataset from all processed contests.
    
    Returns a DataFrame where each row is a unique lineup with:
    - The 5 input dimensions
    - Target variables (duplicate_count, selection probability)
    
    Args:
        session: Database session
        min_entries: Only include contests with at least this many entries
    """
    # Get all lineups with contest metadata
    query = (
        session.query(
            ShowdownLineup,
            ShowdownContestMetadata.total_entries,
            ShowdownContestMetadata.entry_fee,
            ShowdownContestMetadata.unique_lineup_count,
        )
        .join(
            ShowdownContestMetadata,
            ShowdownLineup.contest_id == ShowdownContestMetadata.contest_id
        )
        .filter(ShowdownContestMetadata.total_entries >= min_entries)
    )
    
    records = []
    for lineup, total_entries, entry_fee, unique_count in query.all():
        # Calculate selection probability for this lineup
        # (how many times was this lineup used out of total entries)
        selection_prob = lineup.duplicate_count / total_entries
        
        # Calculate duplication rate (for selected lineups)
        # (if this lineup is selected, how many duplicates on average)
        expected_dupes = lineup.duplicate_count - 1  # Exclude original
        
        records.append({
            # Identifiers
            'contest_id': lineup.contest_id,
            'lineup_id': lineup.id,
            
            # Input features (5 dimensions)
            # 1. Correlation features
            'correlation_score': lineup.correlation_score,
            'same_team_count': lineup.same_team_count,
            'qb_wr_stack': int(lineup.qb_wr_stack),
            'rb_dst_negative': int(lineup.rb_dst_negative),
            'stack_pattern': lineup.stack_pattern,
            'game_stack_type': lineup.game_stack_type,
            
            # 2. Captain ownership (already normalized 0-1)
            'cpt_ownership_pct': lineup.cpt_ownership_pct,
            'cpt_ownership_percentile': lineup.cpt_ownership_percentile,
            
            # 3. Lineup projection (normalized 0-1)
            'projection_normalized': lineup.projection_normalized,
            'projection_percentile': lineup.projection_percentile,
            
            # 4. Lineup ownership (normalized 0-1)
            'ownership_normalized': lineup.ownership_normalized,
            'ownership_percentile': lineup.ownership_percentile,
            
            # 5. Salary remaining
            'salary_remaining': lineup.salary_remaining,
            'salary_remaining_pct': lineup.salary_remaining_pct,
            
            # Target variables
            'selection_probability': selection_prob,
            'duplicate_count': lineup.duplicate_count,
            'expected_dupes': expected_dupes,
            'was_selected': 1 if lineup.duplicate_count > 0 else 0,
            
            # Contest context (for stratification)
            'total_entries': total_entries,
            'entry_fee': entry_fee,
            'unique_lineup_count': unique_count,
            'field_size_category': categorize_field_size(total_entries),
        })
    
    df = pd.DataFrame(records)
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['stack_pattern', 'game_stack_type', 'field_size_category'])
    
    return df


def categorize_field_size(entries: int) -> str:
    """Categorize contest size for stratification."""
    if entries < 1000:
        return 'small'
    elif entries < 10000:
        return 'medium'
    elif entries < 100000:
        return 'large'
    else:
        return 'massive'


def get_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Extract feature matrix and target vectors for modeling.
    
    Returns:
        X: Feature matrix (5 core dimensions + derived features)
        y_selection: Binary target (was lineup selected?)
        y_dupes: Regression target (expected duplicates | selected)
    """
    # Define core feature columns (5 dimensions)
    feature_cols = [
        # Dimension 1: Correlation
        'correlation_score',
        'same_team_count',
        'qb_wr_stack',
        'rb_dst_negative',
        
        # Dimension 2: Captain ownership
        'cpt_ownership_percentile',
        
        # Dimension 3: Projection
        'projection_percentile',
        
        # Dimension 4: Ownership
        'ownership_percentile',
        
        # Dimension 5: Salary
        'salary_remaining_pct',
    ]
    
    # Add one-hot encoded categorical features
    categorical_cols = [c for c in df.columns if c.startswith(('stack_pattern_', 'game_stack_type_'))]
    feature_cols.extend(categorical_cols)
    
    X = df[feature_cols].values
    y_selection = df['was_selected'].values
    y_dupes = df['expected_dupes'].values
    
    return X, y_selection, y_dupes


def analyze_dimension_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how each of the 5 dimensions correlates with duplication.
    """
    dimensions = {
        'Correlation Score': 'correlation_score',
        'Captain Ownership %ile': 'cpt_ownership_percentile',
        'Projection %ile': 'projection_percentile',
        'Ownership %ile': 'ownership_percentile',
        'Salary Remaining %': 'salary_remaining_pct',
    }
    
    results = []
    for dim_name, col in dimensions.items():
        # Correlation with duplicate count
        corr = df[[col, 'duplicate_count']].corr().iloc[0, 1]
        
        # Mean duplicates by quartile
        df['quartile'] = pd.qcut(df[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        quartile_means = df.groupby('quartile')['duplicate_count'].mean()
        
        results.append({
            'dimension': dim_name,
            'correlation': corr,
            'Q1_mean_dupes': quartile_means.get('Q1', 0),
            'Q2_mean_dupes': quartile_means.get('Q2', 0),
            'Q3_mean_dupes': quartile_means.get('Q3', 0),
            'Q4_mean_dupes': quartile_means.get('Q4', 0),
        })
    
    return pd.DataFrame(results)


def get_contest_summary_stats(session: Session, contest_id: int) -> Dict:
    """
    Get summary statistics for a specific contest.
    """
    metadata = session.query(ShowdownContestMetadata).get(contest_id)
    
    if not metadata:
        return {}
    
    lineups = (
        session.query(ShowdownLineup)
        .filter_by(contest_id=contest_id)
        .all()
    )
    
    return {
        'contest_id': contest_id,
        'date': metadata.date,
        'total_entries': metadata.total_entries,
        'unique_lineups': metadata.unique_lineup_count,
        'uniqueness_rate': metadata.unique_lineup_count / metadata.total_entries,
        'max_duplicates': metadata.max_duplicate_count,
        'median_duplicates': metadata.median_duplicate_count,
        
        # Dimension ranges
        'ownership_range': (metadata.min_ownership, metadata.max_ownership),
        'projection_range': (metadata.min_projection, metadata.max_projection),
        'cpt_ownership_range': (metadata.min_cpt_ownership, metadata.max_cpt_ownership),
        
        # Popular patterns
        'top_stacks': metadata.top_stack_patterns,
        'ownership_gini': metadata.ownership_gini,
    }


def find_contrarian_opportunities(
    session: Session,
    contest_id: int,
    min_projection_percentile: float = 0.6,
    max_cpt_ownership: float = 10.0,
    min_salary_remaining: int = 500,
) -> List[Dict]:
    """
    Find lineups that meet contrarian criteria but still have good projections.
    
    These are lineups that historically have low duplication rates but solid projections.
    """
    contrarian_lineups = (
        session.query(ShowdownLineup)
        .filter(
            ShowdownLineup.contest_id == contest_id,
            ShowdownLineup.projection_percentile >= min_projection_percentile,
            ShowdownLineup.cpt_ownership_pct <= max_cpt_ownership,
            ShowdownLineup.salary_remaining >= min_salary_remaining,
        )
        .order_by(ShowdownLineup.duplicate_count.asc())
        .limit(20)
        .all()
    )
    
    results = []
    for lu in contrarian_lineups:
        results.append({
            'captain': lu.captain_name,
            'flex_players': lu.flex_players,
            'projection_percentile': lu.projection_percentile,
            'cpt_ownership': lu.cpt_ownership_pct,
            'salary_remaining': lu.salary_remaining,
            'duplicate_count': lu.duplicate_count,
            'correlation_score': lu.correlation_score,
        })
    
    return results


# Example usage for model training
def example_train_model():
    """
    Example of how to train a model using this data.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    from ivan.database.session import init_db, get_session
    
    engine = init_db("nfl")
    session = get_session(engine)
    
    # Build dataset
    print("Building training dataset...")
    df = build_training_dataset(session, min_entries=1000)
    print(f"Dataset shape: {df.shape}")
    
    # Analyze dimensions
    print("\nDimension importance analysis:")
    importance_df = analyze_dimension_importance(df)
    print(importance_df)
    
    # Prepare features
    X, y_selection, y_dupes = get_feature_matrix(df)
    
    # Split data
    X_train, X_test, y_sel_train, y_sel_test, y_dup_train, y_dup_test = train_test_split(
        X, y_selection, y_dupes, test_size=0.2, random_state=42
    )
    
    # Model 1: Selection probability (classification)
    print("\nTraining selection model...")
    selection_model = RandomForestClassifier(n_estimators=100, random_state=42)
    selection_model.fit(X_train, y_sel_train)
    sel_score = selection_model.score(X_test, y_sel_test)
    print(f"Selection model accuracy: {sel_score:.3f}")
    
    # Model 2: Duplication count (regression, only for selected lineups)
    print("\nTraining duplication model...")
    selected_mask = y_sel_train == 1
    duplication_model = RandomForestRegressor(n_estimators=100, random_state=42)
    duplication_model.fit(X_train[selected_mask], y_dup_train[selected_mask])
    
    test_selected_mask = y_sel_test == 1
    dup_score = duplication_model.score(X_test[test_selected_mask], y_dup_test[test_selected_mask])
    print(f"Duplication model R²: {dup_score:.3f}")
    
    return selection_model, duplication_model, df