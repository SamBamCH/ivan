from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.static import teams
import pandas as pd
from sqlalchemy.dialects.sqlite import insert
import time
from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import NBAStat
from ivan.core.utils import normalize_name


def calculate_dk_fpts(row):
    """
    Calculate DraftKings fantasy points for NBA.
    Scoring: PTS=1, 3PM=0.5, REB=1.25, AST=1.5, STL=2, BLK=2, TO=-0.5
    Double-double bonus: +1.5, Triple-double bonus: +3
    """
    pts = row.get('PTS', 0) or 0
    fg3m = row.get('FG3M', 0) or 0
    reb = row.get('REB', 0) or 0
    ast = row.get('AST', 0) or 0
    stl = row.get('STL', 0) or 0
    blk = row.get('BLK', 0) or 0
    to = row.get('TO', 0) or 0

    fpts = (pts * 1.0) + (fg3m * 0.5) + (reb * 1.25) + (ast * 1.5) + (stl * 2.0) + (blk * 2.0) + (to * -0.5)

    # Double-double and triple-double bonuses
    stats = [pts, reb, ast, stl, blk]
    double_digit_stats = sum(1 for s in stats if s >= 10)

    if double_digit_stats >= 2:
        fpts += 1.5  # Double-double bonus
    if double_digit_stats >= 3:
        fpts += 1.5  # Triple-double bonus (additional 1.5, total 3.0)

    return round(fpts, 2)


def fetch_games_for_date_range(start_date, end_date, season="2023-24"):
    """
    Fetch all games in a date range.
    Returns dict mapping game_id to game_date.
    """
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable='00',
        season_type_nullable='Regular Season'
    )

    games_df = gamefinder.get_data_frames()[0]

    # Filter by date range
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
    games_df = games_df[
        (games_df['GAME_DATE'] >= start_date) &
        (games_df['GAME_DATE'] <= end_date)
    ]

    # Create dict of game_id -> game_date
    games_dict = {}
    for _, row in games_df.iterrows():
        game_id = row['GAME_ID']
        if game_id not in games_dict:  # Avoid duplicates (home/away teams)
            games_dict[game_id] = row['GAME_DATE']

    return games_dict


def fetch_boxscore_for_game(game_id, max_retries=3):
    """
    Fetch box score data for a single game with exponential backoff.
    Returns a list of player stat dictionaries.
    """
    for attempt in range(max_retries):
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            player_stats = boxscore.get_data_frames()[0]

            # Add game_id to the dataframe
            player_stats['GAME_ID'] = game_id

            return player_stats

        except Exception as e:
            wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
            print(f"Error fetching game {game_id} (attempt {attempt+1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch game {game_id} after {max_retries} attempts")
                return None


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


def transform_boxscore_to_model(boxscore_df, game_date, season):
    """
    Transform API response to NBAStat model format.
    Includes both starters and bench players.
    """
    stats_list = []

    for _, row in boxscore_df.iterrows():
        # Parse minutes (format: "MM:SS")
        min_str = row.get('MIN', '0:00') or '0:00'
        if ':' in str(min_str):
            parts = str(min_str).split(':')
            minutes = float(parts[0]) + float(parts[1]) / 60
        else:
            minutes = 0.0

        # Skip players with 0 minutes (didn't play)
        if minutes == 0:
            continue

        # Determine if home game
        matchup = row.get('MATCHUP', '')
        is_home = 1 if 'vs.' in matchup else 0

        # Extract opponent from matchup (e.g., "LAL vs. BOS" -> "BOS")
        opponent = None
        if 'vs.' in matchup:
            opponent = matchup.split('vs.')[1].strip()
        elif '@' in matchup:
            opponent = matchup.split('@')[1].strip()

        # Get position - try START_POSITION first (starters), then POSITION (bench)
        position = row.get('START_POSITION', '').strip()
        if not position:
            # For bench players, try the POSITION field
            position = row.get('POSITION', '').strip()
        
        # If still no position, use 'BENCH' as placeholder
        if not position:
            position = 'BENCH'
        
        # Validate position is valid
        if position not in ['G', 'F', 'C', 'BENCH']:
            position = 'BENCH'

        stat_dict = {
            'game_id': row['GAME_ID'],
            'game_date': game_date.date() if hasattr(game_date, 'date') else game_date,
            'season': season,
            'player_id': str(row['PLAYER_ID']),
            'player_name': normalize_name(row['PLAYER_NAME']),
            'team_abbreviation': row['TEAM_ABBREVIATION'],
            'opponent_team': opponent,
            'is_home': is_home,
            'position': position,
            'minutes': round(minutes, 1),
            'points': safe_int(row.get('PTS')),
            'rebounds': safe_int(row.get('REB')),
            'assists': safe_int(row.get('AST')),
            'steals': safe_int(row.get('STL')),
            'blocks': safe_int(row.get('BLK')),
            'turnovers': safe_int(row.get('TO')),
            'fgm': safe_int(row.get('FGM')),
            'fga': safe_int(row.get('FGA')),
            'fg_pct': safe_float(row.get('FG_PCT')),
            'fg3m': safe_int(row.get('FG3M')),
            'fg3a': safe_int(row.get('FG3A')),
            'fg3_pct': safe_float(row.get('FG3_PCT')),
            'ftm': safe_int(row.get('FTM')),
            'fta': safe_int(row.get('FTA')),
            'ft_pct': safe_float(row.get('FT_PCT')),
            'plus_minus': safe_int(row.get('PLUS_MINUS')),
        }

        # Calculate DraftKings fantasy points
        stat_dict['dk_fpts'] = calculate_dk_fpts(row)

        stats_list.append(stat_dict)

    return stats_list


def upsert_stats(session, stats_list):
    """
    Upsert stats to database using SQLite's INSERT OR REPLACE.
    """
    if not stats_list:
        return

    for stat in stats_list:
        stmt = insert(NBAStat).values(**stat)
        stmt = stmt.on_conflict_do_update(
            index_elements=['game_id', 'player_id'],
            set_={k: stmt.excluded[k] for k in stat.keys() if k not in ['id']}
        )
        session.execute(stmt)

    session.commit()
    print(f"Upserted {len(stats_list)} player stats")


def game_exists_in_db(session, game_id):
    """Check if a game already exists in the database."""
    from sqlalchemy import select
    stmt = select(NBAStat).where(NBAStat.game_id == game_id).limit(1)
    result = session.execute(stmt).first()
    return result is not None


def main(start_date, end_date, season="2023-24"):
    """
    Main function to fetch and store NBA box scores.

    Args:
        start_date: datetime object for start date
        end_date: datetime object for end date
        season: NBA season string (e.g., "2023-24")
    """
    # Initialize database
    engine = init_db("nba")
    NBAStat.metadata.create_all(engine)
    session = get_session(engine)

    print(f"Fetching games from {start_date.date()} to {end_date.date()}...")

    # Fetch game IDs with dates
    games_dict = fetch_games_for_date_range(start_date, end_date, season)
    print(f"Found {len(games_dict)} games")

    # Filter out games that already exist in database
    games_to_fetch = {gid: gdate for gid, gdate in games_dict.items()
                      if not game_exists_in_db(session, gid)}

    already_fetched = len(games_dict) - len(games_to_fetch)
    if already_fetched > 0:
        print(f"Skipping {already_fetched} games already in database")
    print(f"Fetching {len(games_to_fetch)} remaining games")

    # Fetch and store box scores for each game
    for i, (game_id, game_date) in enumerate(games_to_fetch.items()):
        print(f"Processing game {i+1}/{len(games_to_fetch)}: {game_id} (Date: {game_date.date()})")

        boxscore_df = fetch_boxscore_for_game(game_id)

        if boxscore_df is not None and not boxscore_df.empty:
            stats_list = transform_boxscore_to_model(boxscore_df, game_date, season)
            upsert_stats(session, stats_list)

        # Rate limiting - NBA API can be strict
        # Increased to 1 second to be more conservative
        time.sleep(1.0)

    session.close()
    print("Done!")


if __name__ == "__main__":
    # Example: Fetch all games from Jan 1, 2024 to Jan 7, 2024
    start = datetime(2024, 11, 1)
    end = datetime(2025, 6, 30)

    main(start, end, season="2024-25")

    # To fetch entire 2024 season (uncomment):
    # main(datetime(2024, 1, 1), datetime(2024, 12, 31), season="2023-24")
