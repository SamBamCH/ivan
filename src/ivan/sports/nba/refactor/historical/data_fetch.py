from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from datetime import datetime, timedelta
import time
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3
from bs4 import BeautifulSoup
import requests
import random
from requests.exceptions import ReadTimeout, ConnectionError, RequestException
import pandas as pd
from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import StartingLineups, NBAStat
from ivan.core.utils import normalize_name

from ivan.sports.nba.refactor.historical.data_clean import _transfer_boxscore_to_model

# Logging setup
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s",
)
logger = logging.getLogger(__name__)


def _upsert_stats(session, stats_list):
    """Upsert a list of stat dicts into the `nba_stats` table.

    Uses SQLite `INSERT ... ON CONFLICT DO UPDATE` to keep a single
    authoritative row per (game_id, player_id).
    """
    if not stats_list:
        return

    for stat in stats_list:
        ins = sqlite_insert(NBAStat).values(**stat)
        stmt = ins.on_conflict_do_update(
            index_elements=["game_id", "player_name"],
            set_={k: ins.excluded[k] for k in stat.keys() if k != "id"},
        )
        session.execute(stmt)

    session.commit()
    logger.info("Upserted %d player stats", len(stats_list))

def _game_in_db(session, game_id):
    """Return True if at least one NBAStat row exists for `game_id`."""
    stmt = select(NBAStat).where(NBAStat.game_id == game_id).limit(1)
    result = session.execute(stmt).first()
    return result is not None

def _fetch_games():
    """
    Fetch all games for a season.
    returns dict mapping game_id to game_date.
    """
    # Request only NBA games from the API (league_id '00' = NBA)
    season_arg = '2025-26'
    league_arg = '00'
    logger.debug("Calling LeagueGameFinder with season=%s league_id=%s", season_arg, league_arg)
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season_arg,
        league_id_nullable=league_arg,
    )
    games_df = gamefinder.get_data_frames()[0]  # TODO: could just get json from gamefinder object?
    logger.debug("Raw games_df rows=%d columns=%s", len(games_df), list(games_df.columns))

    # Parse GAME_DATE to datetimes and filter to yesterday or earlier
    games_dict = {}
    if 'GAME_DATE' in games_df.columns:
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'], errors='coerce')
        before_dropna = len(games_df)
        games_df = games_df[~games_df['GAME_DATE'].isna()]
        after_dropna = len(games_df)
        logger.debug("Dropped %d rows with invalid GAME_DATE", before_dropna - after_dropna)

        yesterday = (datetime.utcnow().date() - timedelta(days=1))
        logger.debug("Filtering games to dates <= %s (yesterday, UTC)", yesterday)
        filtered_before = len(games_df)
        games_df = games_df[games_df['GAME_DATE'].dt.date <= yesterday]
        filtered_after = len(games_df)
        logger.info("games found %d (NBA only, up to yesterday). Filtered out %d after date check.",
                    filtered_after, filtered_before - filtered_after)

        if not games_df.empty:
            logger.debug("GAME_DATE range: min=%s max=%s",
                         games_df['GAME_DATE'].min(), games_df['GAME_DATE'].max())
            # show a small sample to help debugging
            try:
                logger.debug("Sample rows:\n%s", games_df.head(5).to_dict(orient='records'))
            except Exception:
                logger.debug("Could not stringify sample rows for games_df")

    else:
        logger.warning("No GAME_DATE column in games_df returned by LeagueGameFinder")

    for _, row in games_df.iterrows():
        game_id = row['GAME_ID']
        if game_id in games_dict:
            logger.debug("Duplicate GAME_ID %s encountered; skipping duplicate", game_id)
            continue

        raw_date = row.get('GAME_DATE')
        # Ensure we store a Python date object so SQLAlchemy/SQLite Date accepts it.
        try:
            parsed = pd.to_datetime(raw_date, errors='coerce')
            if pd.isna(parsed):
                logger.warning("Could not parse GAME_DATE for game %s: %s", game_id, raw_date)
                continue
            games_dict[game_id] = parsed.date()
        except Exception as e:
            logger.exception("Unexpected error parsing GAME_DATE for game %s: %s", game_id, raw_date)
            continue

    logger.debug("Returning %d games from _fetch_games", len(games_dict))
    return games_dict

def _fetch_boxscore_for_game(game_id, max_retries=1):
    """Fetch boxscore using NBA Stats V3 `boxscoretraditionalv3` endpoint.

    Returns a pandas DataFrame containing player-level rows (if any).
    On repeated failure returns None.
    """
    for attempt in range(max_retries):
        try:
            timeout_seconds = 10
            logger.info("Fetching boxscore for game %s (attempt %d/%d)", game_id, attempt + 1, max_retries)
            boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=timeout_seconds)
            player_stats = boxscore.get_data_frames()[0]

            if player_stats is None:
                logger.info("No player rows found in boxscore for game %s", game_id)
                return pd.DataFrame()

            player_stats["GAME_ID"] = game_id
            return player_stats

        except (ReadTimeout, ConnectionError) as e:
            wait_time = (2 ** attempt) * 2 + random.uniform(0, 1)
            logger.warning("Network error fetching %s: %s; retrying after %.1fs", game_id, e, wait_time)
            if attempt < max_retries - 1:
                time.sleep(wait_time)
                continue
            logger.error("Network fetch failed for game %s after %d attempts", game_id, max_retries)
            return None

        except RequestException as e:
            logger.error("Request exception fetching %s: %s", game_id, e)
            return None

        except Exception as e:
            logger.exception("Unexpected error fetching boxscore for %s", game_id)
            return None


def fetch_boxscores(session: Session):
    """
    ARGS:
        start_date: datetime object for start date
        end_date: datetime object for end date
        season: NBA season string (i.e. "2023-24")
    """
    games_dict = _fetch_games()

    logger.info("Total games returned by _fetch_games: %d", len(games_dict))

    games_to_fetch = {gid: gdate for gid, gdate in games_dict.items() if not _game_in_db(session, gid)}
    already_fetched = len(games_dict) - len(games_to_fetch)
    logger.info("Skipping %d games already in DB", already_fetched)
    logger.info("Fetching %d remaining games.", len(games_to_fetch))

    for game_id, game_date in games_to_fetch.items():
        boxscore_df = _fetch_boxscore_for_game(game_id)
        print(boxscore_df)

        if boxscore_df is None:
            logger.warning("Boxscore fetch returned None for game %s", game_id)
            continue

        if boxscore_df.empty:
            logger.info("No player rows in boxscore for game %s; skipping upsert", game_id)
            continue

        try:
            stats_list = _transfer_boxscore_to_model(boxscore_df, game_date)
        except Exception:
            logger.exception("Error transforming boxscore for game %s", game_id)
            stats_list = []

        if stats_list:
            try:
                _upsert_stats(session, stats_list)
            except Exception:
                logger.exception("Error upserting stats for game %s", game_id)
        else:
            logger.info("No stats to upsert for game %s (empty stats_list)", game_id)

        # small delay to be polite to the API
        time.sleep(1.0)

    logger.info("NBA stats upsert run completed")

def fetch_starting_lineups(session: Session):
    """Parse Rotowire NBA starters from lineup__list containers (name, team, position)."""
    url = "https://www.rotowire.com/basketball/nba-lineups.php"
    html = requests.get(url, timeout=15).text
    soup = BeautifulSoup(html, "html.parser")

    names: list[str] = []
    teams: list[str] = []
    positions: list[str] = []

    # Find all game containers
    for game_container in soup.select(".lineup"):
        # Each game has visit and home lineup lists
        for lineup_ul in game_container.select("ul.lineup__list"):
            # Determine team abbreviation from data attributes in buttons
            team_abbr = None
            button = lineup_ul.select_one("button[data-team]")
            if button and button.get("data-team"):
                team_abbr = button.get("data-team")
            
            if not team_abbr:
                continue
            
            # Parse starters - only players with is-pct-play-100 before "MAY NOT PLAY" section
            in_may_not_play = False
            for li in lineup_ul.find_all("li", recursive=False):
                # Check if we've hit the "MAY NOT PLAY" section
                if "lineup__title" in li.get("class", []):
                    in_may_not_play = True
                    continue
                
                # Only process starter players (is-pct-play-100)
                if "lineup__player" in li.get("class", []) and not in_may_not_play:
                    
                    # Extract position
                    pos_el = li.select_one(".lineup__pos")
                    position = pos_el.get_text(strip=True) if pos_el else ""
                    
                    # Extract name from <a title="...">
                    a = li.find("a")
                    name = a.get("title", "").strip() if a else ""
                    
                    if not name or not position:
                        continue
                    
                    names.append(name)
                    teams.append(team_abbr)
                    positions.append(position)

    df = pd.DataFrame({"name": names, "team": teams, "position": positions})
    
    # Remove duplicates (keep first occurrence in case of scraping issues)
    df = df.drop_duplicates(subset=["name"], keep="first")

    # Clear existing lineups (this is a daily snapshot)
    deleted_count = session.query(StartingLineups).delete()
    session.flush()
    
    # Insert fresh data
    for _, row in df.iterrows():
        player_name = normalize_name(row["name"])
        team = row["team"]
        position = row["position"]
        
        new = StartingLineups(player_name=player_name, team=team, position=position)
        session.add(new)

    session.commit()
    session.close()