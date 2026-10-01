import pandas as pd

def calculate_kicker_pts(df: pd.DataFrame) -> pd.Series:
    """
    Kicking points as the sportsbooks define them (the player_kicking_points
    market): 3 per field goal made, 1 per extra point made, no distance bonus.
    """
    kicker_pts = pd.Series(0.0, index=df.index)
    kicker_pts += df['fg_made'].fillna(0) * 3
    kicker_pts += df['pat_made'].fillna(0) * 1
    return kicker_pts

def calculate_kicker_fantasy_points(df: pd.DataFrame) -> pd.Series:
    """
    DraftKings kicker fantasy points from FG/PAT data.

    DraftKings scoring:
    - FG Made 0-39 yards: 3 points
    - FG Made 40-49 yards: 4 points
    - FG Made 50+ yards: 5 points
    - PAT Made: 1 point
    DraftKings does not deduct for missed kicks.
    """
    fpts = pd.Series(0.0, index=df.index)
    fpts += df['fg_made_0_19'].fillna(0) * 3
    fpts += df['fg_made_20_29'].fillna(0) * 3
    fpts += df['fg_made_30_39'].fillna(0) * 3
    fpts += df['fg_made_40_49'].fillna(0) * 4
    fpts += df['fg_made_50_59'].fillna(0) * 5
    fpts += df['fg_made_60_'].fillna(0) * 5
    fpts += df['pat_made'].fillna(0) * 1
    return fpts
