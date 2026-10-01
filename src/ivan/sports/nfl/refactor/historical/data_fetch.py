"""
Pull nflverse reference data into the history database.

Everything here is a full refresh of a reference table: seasons run from
FIRST_SEASON through the season in progress, so a weekly run picks up the
latest completed week without anyone editing a season list.
"""
from datetime import date

import nflreadpy as nfl
import pandas as pd
from sqlalchemy import inspect as sa_inspect, text

from ivan.core.utils import normalize_name
from ivan.sports.nfl.models import NFLStat
from ivan.sports.nfl.refactor.historical.data_clean import (
    calculate_kicker_fantasy_points, calculate_kicker_pts,
)

FIRST_SEASON = 2022


def current_season(today: date | None = None) -> int:
    """NFL season in progress: the calendar year from August on, else the previous one."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def seasons_to_fetch(today: date | None = None) -> list[int]:
    return list(range(FIRST_SEASON, current_season(today) + 1))


def fetch_boxscores(session, seasons: list[int] | None = None) -> int:
    """Upsert nflverse weekly player stats into nfl_stats. Returns rows seen."""
    seasons = seasons or seasons_to_fetch()
    df = nfl.load_player_stats(seasons=seasons).to_pandas()
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
    return len(df)


SCHEDULE_COLS = ['game_id', 'season', 'game_type', 'week', 'gameday', 'home_team', 'away_team',
                 'home_score', 'away_score', 'spread_line', 'total_line']


def fetch_schedules(session, seasons: list[int] | None = None) -> int:
    """
    Rebuild nfl_schedule from nflverse: one row per game with the final score
    and the closing market (spread_line is the home team's expected margin,
    positive when the home side is favored; total_line is the game total).
    Unplayed games of the current season are kept with NULL scores so the
    upcoming week's lines are available too. Returns rows written.
    """
    seasons = seasons or seasons_to_fetch()
    df = nfl.load_schedules(seasons).to_pandas()
    df = df[SCHEDULE_COLS].copy()
    df['gameday'] = df['gameday'].astype(str)
    engine = session.get_bind()
    with engine.begin() as con:
        df.to_sql('nfl_schedule', con, if_exists='replace', index=False)
        con.execute(text("CREATE INDEX IF NOT EXISTS ix_nfl_schedule_season_week ON nfl_schedule(season, week)"))
    return len(df)
