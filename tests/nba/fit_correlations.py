"""
Fit rank-order (Spearman) correlations between NBA stats by position.

This script:
1. Loads historical box score data from NBAStat table
2. Groups by position (G, F, C)
3. Calculates Spearman rank correlations between stats
4. Saves correlations.pkl to /data/nba/models/

Usage:
    python tests/nba/fit_correlations.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import select

from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import NBAStat


# Role configurations: separate thresholds for starters vs backups
ROLE_CONFIGS = {
    'starters': {
        'min_threshold': 25,  # Starters play >25 min
        'description': 'Primary players (>25 min)',
    },
    'backups': {
        'min_threshold': (15, 25),  # Backups play 15-25 min
        'description': 'Backup players (15-25 min)',
    },
}

POSITION_CONFIGS = {
    'G': {
        'stats': ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks'],
    },
    'F': {
        'stats': ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks'],
    },
    'C': {
        'stats': ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks'],
    },
}


def fit_position_correlations(session, season: str = "2024-25", role: str = "starters") -> dict:
    """
    Fit Spearman correlation matrices for each position, filtered by role.

    Args:
        session: Database session
        season: NBA season to fit (e.g., "2024-25")
        role: "starters" (>25 min) or "backups" (15-25 min)

    Returns:
        {
            'G': np.ndarray (6x6 correlation matrix),
            'F': np.ndarray (6x6),
            'C': np.ndarray (6x6)
        }
    """
    results = {}

    if role not in ROLE_CONFIGS:
        raise ValueError(f"Invalid role: {role}. Must be one of {list(ROLE_CONFIGS.keys())}")

    role_config = ROLE_CONFIGS[role]

    with session.bind.connect() as conn:
        # Load all stats for the season
        stmt = select(NBAStat).where(NBAStat.season == season)
        df = pd.read_sql(stmt, conn)

    print(f"Loaded {len(df)} player-game records for season {season}")

    # Filter to only players with position data
    df = df[df['position'].isin(['G', 'F', 'C'])]
    print(f"Filtered to {len(df)} records with valid positions (G/F/C)")

    # Apply role filter
    if isinstance(role_config['min_threshold'], tuple):
        # Backups: between min and max
        min_val, max_val = role_config['min_threshold']
        df = df[(df['minutes'] > min_val) & (df['minutes'] <= max_val)]
        print(f"Filtered to {len(df)} records with minutes in ({min_val}, {max_val}]\n")
    else:
        # Starters: above min
        min_val = role_config['min_threshold']
        df = df[df['minutes'] > min_val]
        print(f"Filtered to {len(df)} records with minutes > {min_val}\n")

    for pos_group, config in POSITION_CONFIGS.items():
        print(f"Fitting {role} correlations for position: {pos_group}")

        # Filter by position
        pos_df = df[df['position'] == pos_group]
        print(f"  {len(pos_df)} games for {pos_group}")

        # Extract stat columns
        stat_cols = config['stats']
        data_matrix = pos_df[stat_cols].dropna().values

        if data_matrix.shape[0] < 10:
            print(f"  ⚠ Not enough data ({data_matrix.shape[0]} rows), using identity matrix")
            corr_matrix = np.eye(len(stat_cols))
        else:
            # Compute Spearman rank correlation
            corr_matrix, _ = spearmanr(data_matrix)

            # Ensure it's a 2D array (spearmanr returns scalar if only 1 variable)
            if corr_matrix.ndim == 0:
                corr_matrix = np.array([[1.0]])

            print(f"  ✓ Correlation matrix shape: {corr_matrix.shape}")
            print(f"  Sample correlations:")
            for i, stat1 in enumerate(stat_cols[:3]):  # Show first 3
                for j, stat2 in enumerate(stat_cols[:3]):
                    if i < j:
                        print(f"    {stat1:12s} <-> {stat2:12s}: {corr_matrix[i, j]:6.3f}")

        results[pos_group] = corr_matrix
        print()

    return results


def main():
    """Main entry point."""
    # Initialize database
    engine = init_db("nba")
    session = get_session(engine)

    output_dir = Path("/home/sam-pratt/ivan/data/nba/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit correlations for each role
    for role in ['starters', 'backups']:
        print("="*60)
        print(f"Fitting NBA rank-order correlations for {role}")
        print("="*60)
        correlations = fit_position_correlations(session, season="2024-25", role=role)

        # Save to disk
        output_path = output_dir / f"correlations_{role}.pkl"
        joblib.dump(correlations, output_path)

        print("="*60)
        print(f"✓ Saved {role} correlations to: {output_path}")
        print("="*60)

        # Print summary
        print("Summary:")
        for pos, corr_matrix in correlations.items():
            print(f"  {pos}: {corr_matrix.shape} correlation matrix")
        print()

    session.close()


if __name__ == "__main__":
    main()
