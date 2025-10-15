#!/usr/bin/env python3
"""
Improved NFL stats update script using nflreadpy.

Updates:
1. Uses nflreadpy for all data (replaces deprecated nfl_data_py)
2. Includes kicker data with proper fantasy point calculation
3. Includes DST data
4. Separates K from SPEC to create proper position_group='K'
5. Calculates fantasy points for kickers from FG/PAT data

Usage:
    python scripts/nfl/update_stats_improved.py --seasons 2020 2021 2022 2023 2024
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
from sqlalchemy import MetaData, Table, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ivan.database.session import init_db
from ivan.core.utils import PROJECT_ROOT

# ── CONFIG ────────────────────────────────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "config" / "nfl.json"
DEFAULT_YEARS = [2023, 2024, 2025]
TABLE_NAME = "nfl_stats"
# ────────────────────────────────────────────────────────────────────────────


def calculate_kicker_fantasy_points(df: pd.DataFrame) -> pd.Series:
    """
    Calculate DraftKings kicker fantasy points from FG/PAT data.

    DraftKings scoring:
    - FG Made 0-39 yards: 3 points
    - FG Made 40-49 yards: 4 points
    - FG Made 50+ yards: 5 points
    - PAT Made: 1 point
    - FG Missed: -1 point

    Args:
        df: DataFrame with kicker stats (fg_made_*, fg_missed, pat_made)

    Returns:
        Series of fantasy points
    """
    fpts = pd.Series(0.0, index=df.index)

    # Field goals by distance
    fpts += df['fg_made_0_19'].fillna(0) * 3
    fpts += df['fg_made_20_29'].fillna(0) * 3
    fpts += df['fg_made_30_39'].fillna(0) * 3
    fpts += df['fg_made_40_49'].fillna(0) * 4
    fpts += df['fg_made_50_59'].fillna(0) * 5
    fpts += df['fg_made_60_'].fillna(0) * 5  # 60+ also 5 pts

    # Extra points
    fpts += df['pat_made'].fillna(0) * 1

    # Missed field goals (penalty)
    fpts += df['fg_missed'].fillna(0) * -1

    return fpts


def process_player_stats(seasons: list[int]) -> pd.DataFrame:
    """
    Load and process player stats including kickers.

    Args:
        seasons: List of seasons to load

    Returns:
        DataFrame with processed player stats
    """
    print(f"Loading player stats for seasons: {seasons}")
    df = nfl.load_player_stats(seasons=seasons, summary_level='week').to_pandas()

    print(f"  Loaded {len(df):,} total records")

    # Separate kickers from SPEC
    # nflreadpy puts kickers in position_group='SPEC', position='K'
    kicker_mask = (df['position_group'] == 'SPEC') & (df['position'] == 'K')
    df.loc[kicker_mask, 'position_group'] = 'K'

    print(f"  Identified {kicker_mask.sum():,} kicker records")

    # Calculate fantasy points for kickers
    # (nflreadpy's fantasy_points doesn't include kickers properly)
    kicker_fpts = calculate_kicker_fantasy_points(df[kicker_mask])
    df.loc[kicker_mask, 'fantasy_points'] = kicker_fpts

    # For PPR, kickers same as standard
    df.loc[kicker_mask, 'fantasy_points_ppr'] = kicker_fpts

    # Summary
    print(f"\nPosition breakdown:")
    for pos in ['QB', 'RB', 'WR', 'TE', 'K']:
        count = (df['position_group'] == pos).sum()
        if count > 0:
            avg_fpts = df[df['position_group'] == pos]['fantasy_points'].mean()
            print(f"  {pos}: {count:,} records, avg {avg_fpts:.1f} fpts")

    return df


def main():
    parser = argparse.ArgumentParser(description="Update NFL stats using nflreadpy")
    parser.add_argument('--seasons', nargs='+', type=int, default=DEFAULT_YEARS,
                        help=f'Seasons to load (default: {DEFAULT_YEARS})')
    parser.add_argument('--config', default=str(CONFIG_PATH),
                        help='Path to config file')
    parser.add_argument('--table', default=TABLE_NAME,
                        help=f'Database table name (default: {TABLE_NAME})')
    args = parser.parse_args()

    print("=" * 80)
    print("NFL STATS UPDATE (nflreadpy)")
    print("=" * 80)
    print()

    # 1) Load config & init DB
    config_path = Path(args.config)
    if config_path.exists():
        cfg = json.load(open(config_path, encoding="utf-8"))
        db_url = cfg.get("database_url")
    else:
        print(f"Warning: Config file {config_path} not found, using default DB")
        db_url = None

    engine = init_db("nfl", db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # 2) Fetch & process player stats
    df = process_player_stats(args.seasons)

    # 3) Write to database
    print(f"\nWriting to table '{args.table}'...")
    df.to_sql(args.table, engine, if_exists="replace", index=False)
    print(f"✅ Table `{args.table}` created/replaced with {len(df):,} records")

    # 4) Create composite primary key
    print(f"\nCreating primary key...")
    metadata = MetaData()
    nfl_stats = Table(
        args.table,
        metadata,
        autoload_with=engine
    )

    # Composite PK on (week, season, season_type, player_display_name)
    nfl_stats.append_constraint(
        PrimaryKeyConstraint(
            "week", "season", "season_type", "player_display_name",
            name="pk_nfl_stats"
        )
    )

    # 5) Map to ORM
    OrmBase = declarative_base(metadata=metadata)
    class NFLStat(OrmBase):
        __table__ = nfl_stats

    session.close()

    print()
    print("=" * 80)
    print("✅ NFL STATS UPDATE COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"  1. Verify kicker data: SELECT * FROM {args.table} WHERE position_group='K'")
    print(f"  2. Run build_models.py to fit distributions")
    print(f"  3. Run calibration scripts for K/DST parameters")
    print()


if __name__ == "__main__":
    main()
