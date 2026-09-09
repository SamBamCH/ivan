import nflreadpy as nfl
from ivan.sports.nfl.refactor.historical.data_clean import calculate_kicker_fantasy_points, calculate_kicker_pts
import pandas as pd
from ivan.sports.nfl.models import NFLStat
from ivan.core.utils import normalize_name

from sqlalchemy import inspect as sa_inspect

def fetch_boxscores(session):
    df = nfl.load_player_stats(seasons=[2022, 2023, 2024, 2025]).to_pandas()
    kicker_mask = (df['position_group'] == 'SPEC') & (df['position'] == 'K')
    df.loc[kicker_mask, 'position_group'] = 'K'

    kicker_fpts = calculate_kicker_fantasy_points(df[kicker_mask])
    kicker_pts_simple = calculate_kicker_pts(df[kicker_mask])

    df.loc[kicker_mask, 'fantasy_points'] = kicker_fpts
    df.loc[kicker_mask, 'fantasy_points_ppr'] = kicker_fpts
    df.loc[kicker_mask, 'kicker_pts'] = kicker_pts_simple

    df['player_display_name'] = df['player_display_name'].astype(str).map(normalize_name)

    # Only keep columns that actually exist on the model
    valid_cols = {c.key for c in sa_inspect(NFLStat).mapper.column_attrs}
    df = df[[c for c in df.columns if c in valid_cols]]

    # NaN -> None so the DB gets NULL instead of float('nan')
    df = df.astype(object).where(pd.notnull(df), None)

    for _, row in df.iterrows():
        data = row.to_dict()
        stat = session.query(NFLStat).filter_by(
            player_id=data['player_id'],
            season=data['season'],
            week=data['week'],
        ).first()

        if stat:
            for col, val in data.items():
                setattr(stat, col, val)
        else:
            session.add(NFLStat(**data))

    session.commit()

    

