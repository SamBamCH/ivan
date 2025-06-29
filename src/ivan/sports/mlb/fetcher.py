import requests
from datetime import date, timedelta
import logging
from datetime import datetime, timedelta
from requests.exceptions import HTTPError
import statsapi
import json
from sqlalchemy.exc import IntegrityError
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


def fetch_and_store(days_back: int = 730):
    """
    Thin orchestrator for backfilling the MLBStat table:
      1) init DB/session
      2) chunked get_game_ids
      3) extract_player_stats → normalize name, calc OBP/SLG → MLBStat(**) → add/commit
    """
    # 1) init DB/session
    cfg_path = PROJECT_ROOT / 'config' / 'mlb.json'
    cfg      = json.load(open(cfg_path, encoding='utf-8'))
    engine   = init_db('mlb', cfg.get('database_url'))
    sess     = get_session(engine)

    # 2) get all game IDs (chunked)
    game_ids = get_game_ids(days_back=days_back)
    logger.info(f"Backfilling {len(game_ids)} games (last {days_back} days)")

    # 3) loop & write
    for idx, gid in enumerate(game_ids, start=1):
        gd = statsapi.get("game", {"gamePk": gid})["gameData"]["datetime"]["originalDate"]
        logger.info(f"[{idx}/{len(game_ids)}] Game {gid} on {gd}")
        rows = extract_player_stats(gid, gd)

        for data in rows:
            # normalize the name
            data["name"] = normalize_name(data["name"])

            # calc OBP & SLG
            pa   = data["pa"]
            ab   = data["ab"]
            hits = data["hits"]
            bb   = data["bb"]
            hbp  = data["hbp"]
            doubles = data["doubles"]
            triples = data["triples"]
            hr       = data["hr"]

            obp = (hits + bb + hbp) / pa if pa else 0.0
            tb  = hits + doubles + 2*triples + 3*hr
            slg = tb / ab    if ab else 0.0
            data["obp"] = obp
            data["slg"] = slg

            sess.add(MLBStat(**data))

        try:
            sess.commit()
        except IntegrityError:
            sess.rollback()
            logger.warning(f"Rolled back game {gid} due to duplicate or bad rows")

    logger.info("✅ Done backfilling mlb_stats!")


def get_upcoming_contest_ids(
    sess: requests.Session,
    sport: str = "MLB",
    days_ahead: int = 1
) -> list[str]:
    # 1) fetch raw JSON
    start = date.today().isoformat()
    end   = (date.today() + timedelta(days=days_ahead)).isoformat()
    params = {"sport": sport.lower(), "startDate": start, "endDate": end}
    url = "https://www.draftkings.com/lobby/getcontests"

    resp = sess.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    # 2) locate the contests array
    if isinstance(data, list):
        contests_list = data
    elif "contests" in data:
        contests_list = data["contests"]
    elif "Contests" in data:
        contests_list = data["Contests"]
    else:
        print("!!! Unexpected JSON keys:", list(data.keys()))
        raise KeyError("Could not find contests array in response")

    # 3) filter by IsGuaranteed & (m>2000 or a>50)
    filtered = []
    for c in contests_list:
        is_g = c.get("attr", {}).get("IsGuaranteed", "").lower() == "true"
        m    = c.get("m", 0)
        a    = c.get("a", 0)

        # keep guaranteed contests where either
        #  • m > 200,  OR
        #  • a > 50  AND  m > 100
        if is_g and (m > 200 or (a > 50 and m > 100)):
            filtered.append(c)

    # 4) debug: print first 5 of your filtered contests
    print("\n=== First 5 GUARANTEED contests matching your criteria ===")
    for c in filtered[:5]:
        print(f"dg: {c.get('id')}   a: {c.get('a')}   m: {c.get('m')}   IsGuaranteed: {c.get('attr',{}).get('IsGuaranteed')}")
        pprint(c, width=80)
        print("-" * 60)

    # 5) return the IDs you actually want to download
    #    here 'dg' is the DraftGroupId shorthand in this payload
    return filtered

def get_dk_session_with_selenium(user_data_dir: str=None, profile: str="Default"):
    """
    Launch Chrome, let you complete login+2FA, then harvest cookies into a Session.
    """
    # (optional) reuse your existing profile so some cookies persist
    opts = Options()
    if user_data_dir:
        opts.add_argument(f"--user-data-dir={user_data_dir}")
        opts.add_argument(f"--profile-directory={profile}")

    driver = webdriver.Chrome(options=opts)
    driver.get("https://www.draftkings.com/account")     # DK will redirect to login

    print("\n→ Please complete login + 2FA in the browser window.")
    while "myaccount.draftkings.com" not in driver.current_url:
        time.sleep(1)
    print("✓ Login complete, extracting cookies…")

    sess = requests.Session()
    for c in driver.get_cookies():
        sess.cookies.set(c['name'], c['value'],
                         domain=c['domain'],
                         path=c['path'],
                         secure=c.get('secure', False),
                         rest={"HttpOnly": c.get('httpOnly', False)},
                         expires=c.get('expiry', None))
    driver.quit()
    return sess

def fetch_contest_json_requests(
    sess: requests.Session,
    contest_id: int,
    draft_group_id: int
) -> dict:
    """
    Calls DraftKings' getContestStandings JSON endpoint and returns a dict.
    """
    url = "https://www.draftkings.com/lobby/getContestStandings"
    params = {
        "contestId":    contest_id,
        "draftGroupId": draft_group_id,
        "startRow":     1,
        "endRow":       50000,
    }
    resp = sess.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

