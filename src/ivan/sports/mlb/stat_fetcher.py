import requests
from datetime import date, timedelta
import logging
from datetime import datetime as dt
from requests.exceptions import HTTPError
import statsapi
import json
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
import time
from pprint import pprint

from ivan.core.utils import normalize_name, PROJECT_ROOT
from ivan.database.session import init_db, get_session
from ivan.sports.mlb.models import MLBStat, HistoricalContest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options



logger = logging.getLogger(__name__)

# maximum days per single schedule call
MAX_DAYS_PER_CALL = 100
RETRIES = 3
BACKOFF_BASE = 2 #exponential backoff factor

# DraftKings scoring weights
SCORING_DK = {
    "single":   3,
    "double":   5,
    "triple":   8,
    "home_run": 10,
    "rbi":      2,
    "run":      2,
    "bb":       2,
    "hbp":      2,
    "sb":       5,
}


def get_game_ids(days_back: int = 730) -> list[int]:
    """
    Return all FINAL gamePk values for the last `days_back` days,
    split into ≤MAX_DAYS_PER_CALL windows, with minimal hydration
    and retries on timeout.
    """
    today = datetime.now().date() - timedelta(days=1)
    start = today - timedelta(days=days_back - 1)
    all_ids = []

    cur = start
    while cur <= today:
        end = min(cur + timedelta(days=MAX_DAYS_PER_CALL - 1), today)
        logger.info(f"Scheduling games from {cur} through {end}")

        # try up to RETRIES times
        for attempt in range(1, RETRIES + 1):
            try:
                sched = statsapi.schedule(
                    start_date=cur.isoformat(),
                    end_date=end.isoformat(),
                    sportId=1,
                )
                break
            except HTTPError as e:
                logger.warning(
                    f"Attempt {attempt}/{RETRIES} timeout scheduling {cur}→{end}: {e}"
                )
                if attempt < RETRIES:
                    sleep_time = BACKOFF_BASE ** (attempt - 1)
                    logger.info(f"Sleeping {sleep_time}s before retry")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Giving up on window {cur}→{end}")
                    sched = {"dates": []}

        for g in sched:
            all_ids.append(g["game_id"])

        cur = end + timedelta(days=1)

    return sorted(set(all_ids))


def extract_player_stats(game_id: int, original_date: str) -> list[dict]:
    """
    Pull boxscore for one game, build one dict per batter with all needed fields.
    Does *not* write to the DB.
    """
    box = statsapi.boxscore_data(game_id)
    rows = []

    for side in ("home", "away"):
        for pid, pdata in box[side]["players"].items():
            try:
                bat = pdata.get("stats", {}).get("batting", {})
                if not bat or "battingOrder" not in pdata:
                    continue

                bo = int(pdata["battingOrder"]) // 10

                # raw counts
                runs    = bat.get("runs",        0) or 0
                hits    = bat.get("hits",        0) or 0
                doubles = bat.get("doubles",     0) or 0
                triples = bat.get("triples",     0) or 0
                hr      = bat.get("homeRuns",    0) or 0
                rbi     = bat.get("rbi",         0) or 0
                bb      = bat.get("baseOnBalls", 0) or 0
                hbp     = bat.get("hitByPitch",  0) or 0
                sb      = bat.get("stolenBases", 0) or 0
                ab      = bat.get("atBats",      0) or 0
                pa      = ab + bb + hbp

                # team
                team = pdata.get("parentTeamId")
                if team is None:
                    raise KeyError("parentTeamId")

                # compute DK fantasy points
                singles = hits - doubles - triples - hr
                dk_fpts = (
                    singles*SCORING_DK["single"] +
                    doubles*SCORING_DK["double"] +
                    triples*SCORING_DK["triple"] +
                    hr*SCORING_DK["home_run"] +
                    rbi*SCORING_DK["rbi"] +
                    runs*SCORING_DK["run"] +
                    bb*SCORING_DK["bb"] +
                    hbp*SCORING_DK["hbp"] +
                    sb*SCORING_DK["sb"]
                )

                # build the dict
                row = {
                    "date"          : datetime.fromisoformat(original_date).date(),
                    "player_id"     : pid,
                    "name"          : pdata["person"]["fullName"],
                    "team"          : team,
                    "position"      : pdata.get("position",{}).get("abbreviation",""),
                    "batting_order" : bo,
                    "pa"            : pa,
                    "ab"            : ab,
                    "runs"          : runs,
                    "hits"          : hits,
                    "doubles"       : doubles,
                    "triples"       : triples,
                    "hr"            : hr,
                    "rbi"           : rbi,
                    "bb"            : bb,
                    "sb"            : sb,
                    "hbp"           : hbp,
                    "dk_fpts"       : float(dk_fpts),
                    "fd_fpts"       : 0.0,   # fill in later
                }
                rows.append(row)

            except KeyError as ke:
                logger.warning(f"Game {game_id} / player {pid}: missing key {ke}; skipping")
            except Exception:
                logger.exception(f"Game {game_id} / player {pid}: unexpected error")
    return rows

def _as_date(s: str):
    # StatsAPI gives 'YYYY-MM-DD' → convert to python date (model column is Date)
    return dt.strptime(s, "%Y-%m-%d").date()

def _upsert_mlb_stats(sess, rows: list[dict]) -> None:
    """Insert-or-update on (date, player_id). Works for Postgres & SQLite."""
    if not rows:
        return
    table = MLBStat.__table__
    dialect = sess.bind.dialect.name

    def update_map(stmt):
        # update all columns except the PK
        return {c.name: getattr(stmt.excluded, c.name)
                for c in table.columns if c.name != "id"}

    if dialect == "postgresql":
        stmt = pg_insert(table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["date", "player_id"],
            set_=update_map(stmt),
        )
        sess.execute(stmt)
    elif dialect == "sqlite":
        stmt = sqlite_insert(table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["date", "player_id"],
            set_=update_map(stmt),
        )
        sess.execute(stmt)
    else:
        # Portable fallback (slower): row-by-row get-or-update
        for r in rows:
            existing = (sess.query(MLBStat)
                        .filter(MLBStat.date == r["date"],
                                MLBStat.player_id == r["player_id"])
                        .one_or_none())
            if existing:
                for k, v in r.items():
                    if k != "id":
                        setattr(existing, k, v)
            else:
                sess.add(MLBStat(**r))


def fetch_and_store(days_back: int = 730):
    """
    Backfill MLBStat with idempotent upserts on (date, player_id).
    """
    # 1) init DB/session
    cfg_path = PROJECT_ROOT / 'config' / 'mlb.json'
    cfg      = json.load(open(cfg_path, encoding='utf-8'))
    engine   = init_db('mlb', cfg.get('database_url'))
    sess     = get_session(engine)

    # 2) get all game IDs (chunked)
    game_ids = get_game_ids(days_back=days_back)
    logger.info(f"Backfilling {len(game_ids)} games (last {days_back} days)")

    # 3) loop & upsert per game
    for idx, gid in enumerate(game_ids, start=1):
        gd = statsapi.get("game", {"gamePk": gid})["gameData"]["datetime"]["originalDate"]
        logger.info(f"[{idx}/{len(game_ids)}] Game {gid} on {gd}")
        rows = extract_player_stats(gid, gd)  # expects a list[dict]

        # Prepare rows for upsert
        upsert_payload = []
        game_date = _as_date(gd)

        for data in rows:
            # ensure required key fields
            data["date"] = game_date
            data["name"] = normalize_name(data["name"])

            # calc OBP & SLG
            pa, ab = data["pa"], data["ab"]
            hits, bb, hbp = data["hits"], data["bb"], data["hbp"]
            doubles, triples, hr = data["doubles"], data["triples"], data["hr"]

            obp = (hits + bb + hbp) / pa if pa else 0.0
            tb  = hits + doubles + 2*triples + 3*hr
            slg = (tb / ab) if ab else 0.0
            data["obp"] = obp
            data["slg"] = slg

            upsert_payload.append(data)

        try:
            _upsert_mlb_stats(sess, upsert_payload)
            sess.commit()
        except IntegrityError as e:
            sess.rollback()
            logger.warning(f"Rolled back game {gid} upsert: {e}")

    logger.info("✅ Done backfilling mlb_stats (idempotent on (date, player_id))!")


