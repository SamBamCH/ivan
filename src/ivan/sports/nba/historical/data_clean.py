import pandas as pd
from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.scoring import dk_points, from_nba_api
import re
from datetime import datetime, timezone

def safe_int(value, default=0):
    """Safely convert value to int, handling None and NaN."""
    if pd.isna(value) or value is None:
        return default
    return int(value)

def safe_float(value, default=0.0):
    """Safely convert value to float, handling None and NaN."""
    if pd.isna(value) or value is None:
        return default
    return float(value)

def _transfer_boxscore_to_model(boxscore_df, game_date, season='2025-26'):
    stats_list = []

    for _, row in boxscore_df.iterrows():
        player_name = normalize_name(f"{row.get('firstName', '')} {row.get('familyName', '')}")
        # DNP rows carry an empty minutes string. Without the else branch the
        # previous player's minutes leaked into every DNP row (fixed 2026-10-06).
        min_str = row.get('minutes') or ''
        if ':' in str(min_str):
            parts = str(min_str).split(':')
            minutes = float(parts[0]) + float(parts[1]) / 60
        else:
            minutes = 0.0

        # Get position - try START_POSITION first (starters), then POSITION (bench)
        position = row.get('position', '').strip()        
        # If still no position, use 'BENCH' as placeholder
        if not position:
            position = 'BENCH'

        stat_dict = {
            'game_id': row['gameId'],
            'game_date': game_date.date() if hasattr(game_date, 'date') else game_date,
            'season': season,
            'player_name': player_name,
            'position': position,
            'team_abbreviation': row['teamTricode'],
            'minutes': round(minutes, 1),
            'points': safe_int(row.get('points')),
            'rebounds': safe_int(row.get('reboundsTotal')),
            'rebounds_off': safe_int(row.get('reboundsOffensive')),
            'rebounds_def': safe_int(row.get('reboundsDefensive')),
            'assists': safe_int(row.get('assists')),
            'steals': safe_int(row.get('steals')),
            'blocks': safe_int(row.get('blocks')),
            'turnovers': safe_int(row.get('turnovers')),
            'fgm': safe_int(row.get('fieldGoalsMade')),
            'fga': safe_int(row.get('fieldGoalsAttempted')),
            'fg_pct': safe_float(row.get('fieldGoalsPercentage')),
            'fg3m': safe_int(row.get('threePointersMade')),
            'fg3a': safe_int(row.get('threePointersAttempted')),
            'fg3_pct': safe_float(row.get('threePointersPercentage')),
            'ftm': safe_int(row.get('freeThrowsMade')),
            'fta': safe_int(row.get('freeThrowsAttempted')),
            'ft_pct': safe_float(row.get('freeThrowsPercentage')),
        }

        stat_dict['dk_fpts'] = round(dk_points(from_nba_api(row)), 2)

        stats_list.append(stat_dict)

    return stats_list

def parse_sd(sd_str: str) -> datetime:
    """
    Parse timestamps of the form '/Date(1747782300000)/' into a UTC datetime.
    """
    m = re.search(r"/Date\((\d+)\)/", sd_str or "")
    if not m:
        return None
    ts = int(m.group(1)) / 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc)
