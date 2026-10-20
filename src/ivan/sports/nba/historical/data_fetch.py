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

from ivan.sports.nba.historical.data_clean import _transfer_boxscore_to_model

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

def current_season(today=None) -> str:
    """NBA season string for a date: October onwards is the season that starts that year."""
    from datetime import date as _date
    d = today or _date.today()
    y = d.year if d.month >= 10 else d.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def _fetch_games(season: str | None = None):
    """
    Fetch all games for a season.
    returns dict mapping game_id to game_date.
    """
    # Request only NBA games from the API (league_id '00' = NBA)
    season_arg = season or current_season()
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


def fetch_boxscores(session: Session, season: str | None = None, max_games: int | None = None):
    """
    Store box scores for every game of `season` (default: the current one) that
    is not in nba_stats yet. max_games caps one run (nba_api rate limits).
    """
    season = season or current_season()
    games_dict = _fetch_games(season)

    logger.info("Total games returned by _fetch_games: %d", len(games_dict))

    games_to_fetch = {gid: gdate for gid, gdate in games_dict.items() if not _game_in_db(session, gid)}
    already_fetched = len(games_dict) - len(games_to_fetch)
    logger.info("Skipping %d games already in DB", already_fetched)
    logger.info("Fetching %d remaining games.", len(games_to_fetch))

    for n_done, (game_id, game_date) in enumerate(games_to_fetch.items()):
        if max_games is not None and n_done >= max_games:
            logger.info("stopping after %d games (max_games)", max_games)
            break
        boxscore_df = _fetch_boxscore_for_game(game_id)

        if boxscore_df is None:
            logger.warning("Boxscore fetch returned None for game %s", game_id)
            continue

        if boxscore_df.empty:
            logger.info("No player rows in boxscore for game %s; skipping upsert", game_id)
            continue

        try:
            stats_list = _transfer_boxscore_to_model(boxscore_df, game_date, season=season)
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

ROTOWIRE_URL = "https://www.rotowire.com/basketball/nba-lineups.php"


def fetch_rotowire_html(timeout: int = 15) -> str:
    return requests.get(ROTOWIRE_URL, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}).text


def parse_rotowire(html: str) -> pd.DataFrame:
    """
    Starters from the Rotowire NBA lineups page: one row per listed starter with
    columns name, team, position, confirmed (Rotowire's is-confirmed flag).
    Players under the "may not play" title are skipped.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for game_container in soup.select(".lineup"):
        for lineup_ul in game_container.select("ul.lineup__list"):
            button = lineup_ul.select_one("button[data-team]")
            team_abbr = button.get("data-team") if button else None
            if not team_abbr:
                continue
            status_li = lineup_ul.select_one("li.lineup__status")
            confirmed = bool(status_li and "is-confirmed" in status_li.get("class", []))
            in_may_not_play = False
            for li in lineup_ul.find_all("li", recursive=False):
                if "lineup__title" in li.get("class", []):
                    in_may_not_play = True
                    continue
                if "lineup__player" in li.get("class", []) and not in_may_not_play:
                    pos_el = li.select_one(".lineup__pos")
                    position = pos_el.get_text(strip=True) if pos_el else ""
                    a = li.find("a")
                    name = a.get("title", "").strip() if a else ""
                    if name and position:
                        rows.append({"name": name, "team": team_abbr, "position": position, "confirmed": confirmed})
    df = pd.DataFrame(rows, columns=["name", "team", "position", "confirmed"])
    return df.drop_duplicates(subset=["name"], keep="first")


def store_starting_lineups(session: Session, df: pd.DataFrame, fetched_at=None) -> int:
    """Replace starting_lineups with this page's starters."""
    from datetime import datetime as _dt
    fetched_at = fetched_at or _dt.now()
    session.query(StartingLineups).delete()
    session.flush()
    n = 0
    for _, row in df.iterrows():
        session.add(StartingLineups(player_name=normalize_name(row["name"]), team=row["team"],
                                    position=row["position"], confirmed=bool(row["confirmed"]), fetched_at=fetched_at))
        n += 1
    session.commit()
    return n


def fetch_starting_lineups(session: Session, raw_sport: str | None = "nba") -> int:
    """Fetch, keep the page as raw rotowire/lineups, parse and store."""
    html = fetch_rotowire_html()
    if raw_sport:
        from ivan.core import raw
        from datetime import date as _date
        raw.store(raw_sport, "rotowire/lineups", _date.today().isoformat(), html, ext="html")
    return store_starting_lineups(session, parse_rotowire(html))
