import pandas as pd

def calculate_kicker_pts(df: pd.DataFrame) -> pd.Series:
    """
    Calculate simple kicker points: FGs (3 pts each) + PATs (1 pt each).
    
    Args:
        df: DataFrame with kicker stats (fg_made, pat_made)
    
    Returns:
        Series of kicker points
    """
    kicker_pts = pd.Series(0.0, index=df.index)
    kicker_pts += df['fg_made'].fillna(0) * 3  # Total FG made × 3
    kicker_pts += df['pat_made'].fillna(0) * 1  # PAT made × 1
    return kicker_pts

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