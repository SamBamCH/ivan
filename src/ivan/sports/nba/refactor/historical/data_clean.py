import pandas as pd
from ivan.core.utils import normalize_name
from pathlib import Path
from datetime import timedelta
import shutil
import re
from datetime import datetime, timezone
from ivan.sports.nba.models import ContestPayout
from ivan.sports.nba.models import NBAStat

def _calculate_dk_fpts(row):
    """
    Calculate DraftKings fantasy points for NBA.
    Scoring: PTS=1, 3PM=0.5, REB=1.25, AST=1.5, STL=2, BLK=2, TO=-0.5
    Double-double bonus: +1.5, Triple-double bonus: +3
    """
    pts = row.get('points', 0) or 0
    fg3m = row.get('threePointersMade', 0) or 0
    reb = row.get('reboundsTotal', 0) or 0
    ast = row.get('assists', 0) or 0
    stl = row.get('steals', 0) or 0
    blk = row.get('blocks', 0) or 0
    to = row.get('turnovers', 0) or 0

    fpts = (pts * 1.0) + (fg3m * 0.5) + (reb * 1.25) + (ast * 1.5) + (stl * 2.0) + (blk * 2.0) + (to * -0.5)

    # Double-double and triple-double bonuses
    stats = [pts, reb, ast, stl, blk]
    double_digit_stats = sum(1 for s in stats if s >= 10)

    if double_digit_stats >= 2:
        fpts += 1.5  # Double-double bonus
    if double_digit_stats >= 3:
        fpts += 1.5  # Triple-double bonus (additional 1.5, total 3.0)

    return round(fpts, 2)


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
        min_str = row.get('minutes', '0:00') or '0.00'
        if ':' in str(min_str):
            parts = str(min_str).split(':')
            minutes = float(parts[0]) + float(parts[1]) / 60

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

        stat_dict['dk_fpts'] = _calculate_dk_fpts(row)

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

def upsert_payouts(session, cid, payload):
    detail  = payload.get("contestDetail", {})
    summary = detail.get("payoutSummary", []) or []

    rows_to_insert = []
    for tier in summary:
        min_pos = tier.get("minPosition")
        max_pos = tier.get("maxPosition")

        pd_list = tier.get("payoutDescriptions", "")
        if not pd_list:
            continue
        
        first_pd    = pd_list[0]
        val         = first_pd.get("value", 0.0)
        text_form   = first_pd.get("payoutDescription", "")
        currency    = "USD"

        rows_to_insert.append(
            ContestPayout(
                contest_id      = cid,
                min_position    = min_pos,
                max_position    = max_pos,
                payout_amount   = float(val),
                payout_text     = text_form,
                currency        = currency,
            )
        )
    session.bulk_save_objects(rows_to_insert)
    session.commit()

def normalize_player_names(session):
    """Normalize all player names in NBAStat table."""
    
    stats = session.query(NBAStat).all()
    for stat in stats:
        stat.player_name = normalize_name(stat.player_name)
    
    session.commit()

def clean_old_diagnostics(
    diagnostics_root: str | Path = "data/nfl/diagnostics",
    days_to_keep: int = 7
):
    """Remove diagnostic folders older than days_to_keep."""
    diagnostics_root = Path(diagnostics_root)
    if not diagnostics_root.exists():
        return
    
    cutoff = datetime.now() - timedelta(days=days_to_keep)
    removed = 0
    
    for folder in diagnostics_root.iterdir():
        if not folder.is_dir():
            continue
        
        folder_mtime = datetime.fromtimestamp(folder.stat().st_mtime)
        
        if folder_mtime < cutoff:
            print(f"  Removing old diagnostics: {folder.name} (age: {(datetime.now() - folder_mtime).days} days)")
            shutil.rmtree(folder)
            removed += 1
    
    if removed:
        print(f"  Cleaned up {removed} old diagnostic folder(s)")
