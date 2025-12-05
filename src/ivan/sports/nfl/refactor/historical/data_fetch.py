import nflreadpy as nfl
from ivan.sports.nfl.refactor.historical.data_clean import calculate_kicker_fantasy_points, calculate_kicker_pts
import pandas as pd
from ivan.sports.nfl.models import NFLStat
from ivan.core.utils import normalize_name

def fetch_boxscores(session):
    df = nfl.load_player_stats(seasons=[2022, 2023, 2024, 2025]).to_pandas()
    kicker_mask = (df['position_group'] == 'SPEC') & (df['position'] == 'K')
    df.loc[kicker_mask, 'position_group'] = 'K'
    
    # Calculate kicker metrics
    kicker_fpts = calculate_kicker_fantasy_points(df[kicker_mask])
    kicker_pts_simple = calculate_kicker_pts(df[kicker_mask])
    
    df.loc[kicker_mask, 'fantasy_points'] = kicker_fpts
    df.loc[kicker_mask, 'fantasy_points_ppr'] = kicker_fpts
    df.loc[kicker_mask, 'kicker_pts'] = kicker_pts_simple
    
    # Normalize player_display_name
    df['player_display_name'] = df['player_display_name'].astype(str).map(normalize_name)
    
    # Upsert to NFLStat table
    for _, row in df.iterrows():
        stat = session.query(NFLStat).filter_by(
            player_id=row['player_id'],
            season=row['season'],
            week=row['week']
        ).first()
        
        if stat:
            for col in df.columns:
                setattr(stat, col, row[col])
        else:
            stat = NFLStat(**row.to_dict())
            session.add(stat)
    
    session.commit()

    

