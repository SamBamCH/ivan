from __future__ import annotations   # annotations may name selenium types that are imported lazily
import time
import io
import zipfile
import logging
import json
import sys
from datetime import date, timedelta
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
from pandas.errors import EmptyDataError
from http.client import RemoteDisconnected
from requests.exceptions import RequestException
import pytz
import re
from ivan.core.utils import normalize_name
from dateutil import parser

logger = logging.getLogger(__name__)


# Base URLs
LOBBY_URL = "https://www.draftkings.com/lobby/getcontests"
API_BASE  = "https://api.draftkings.com/draftgroups/v1"
DK_CONTESTDETAIL_URL = "https://api.draftkings.com/contests/v1/contests/{cid}"
_TEAM_RE = re.compile(r"\b[A-Z]{2,3}\b")
# Define your local timezone (Central Time)
LOCAL_TZ = pytz.timezone('America/Chicago')


def get_upcoming_contest_ids(
    sport: str = "NBA",
    days_ahead: int = 1,
    entry_filter: int = 100
) -> list[str]:
    # 1) fetch raw JSON
    sess = requests.Session()
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
        if is_g and (m > entry_filter or (a > 50 and m > 100)):
            filtered.append(c)

    # 5) return the IDs you actually want to download
    #    here 'dg' is the DraftGroupId shorthand in this payload
    return filtered

def build_requests_session(profile_dir: str, headless: bool = False) -> requests.Session:
    """
    Launch Chrome with your real profile, ensure you're logged in,
    copy cookies + UA into a requests.Session.
    """
    # Selenium is only needed for the logged-in DK scraper; keep it optional on servers.
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    opts = Options()
    # run non-headless so Chrome can fully load the persisted profile/cookies
    if headless:
        opts.add_argument("--headless=new")  # if you must, but prefer False
    opts.add_argument(f"--user-data-dir={profile_dir}")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--lang=en-US")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)

    time.sleep(10)
    try:
        driver.get("https://www.draftkings.com/")
        # Give time for any SSO redirect
        time.sleep(3)

        # Wait (briefly) until a DK cookie that only exists when logged in appears.
        # Adjust names if your cookies differ; examples: 'dkng', 'dkng_session', 'api_token'
        wanted = {"dkng", "dkng_session", "api_token", "csrf_token"}
        for _ in range(20):
            names = {c["name"] for c in driver.get_cookies()}
            if wanted & names:
                break
            time.sleep(0.25)
        else:
            print("[warn] Did not detect an authenticated cookie. You may not be logged in for this profile.")

        user_agent = driver.execute_script("return navigator.userAgent")
        cookies    = driver.get_cookies()
    finally:
        driver.quit()

    sess = requests.Session()
    # Copy cookies verbatim for the right domain
    for c in cookies:
        # requests needs domain like '.draftkings.com' or 'www.draftkings.com'
        sess.cookies.set(c["name"], c["value"], domain=c.get("domain") or "www.draftkings.com")

    # Browser-like headers
    sess.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive",
    })
    return sess

def _cookies_to_requests_session(driver: webdriver.Chrome) -> requests.Session:
    """Copy all visible cookies + user agent from Selenium into a requests.Session."""
    # Selenium is only needed for the logged-in DK scraper; keep it optional on servers.
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    sess = requests.Session()

    # exact browser UA matters for some bot defenses
    ua = driver.execute_script("return navigator.userAgent")
    sess.headers.update({
        "User-Agent": ua,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })

    # copy cookies with original domain/path where possible
    for c in driver.get_cookies():
        # Some cookies specify domain like ".draftkings.com" (keep it),
        # others omit domain; default to main site.
        domain = c.get("domain") or "www.draftkings.com"
        path   = c.get("path") or "/"
        secure = bool(c.get("secure", False))
        # requests requires a "create_cookie" to set path/domain
        cookie = requests.cookies.create_cookie(
            name=c["name"],
            value=c.get("value", ""),
            domain=domain,
            path=path,
            secure=secure,
        )
        try:
            sess.cookies.set_cookie(cookie)
        except Exception as e:
            logging.debug(f"skip cookie {c['name']}: {e}")

    return sess

def build_requests_session_from_chrome(user_data_dir: str,
                                       profile_directory: str = "Default",
                                       headless: bool = False) -> requests.Session:
    """
    Launch Chrome using your real profile. Navigate to the lobby first to trigger SSO/bot cookies,
    then copy cookies+UA into a requests.Session and return it.
    """
    # Selenium is only needed for the logged-in DK scraper; keep it optional on servers.
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    opts = Options()
    # IMPORTANT: Use your actual profile so you're already logged in
    opts.add_argument(f"--user-data-dir={user_data_dir}")
    opts.add_argument(f"--profile-directory={profile_directory}")
    if headless:
        # headless can miss some device-bound cookies; prefer non-headless when possible
        opts.add_argument("--headless=new")

    # helps avoid "automation" flags that some sites key on
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)

    try:
        # 1) Hit the main site first (sets core cookies)
        driver.get("https://www.draftkings.com/")
        time.sleep(2)

        # 2) Hit the lobby webapp (ensures DK sets akamai/bot cookies for this session)
        driver.get("https://www.draftkings.com/lobby")   # sport tab doesn’t really matter
        time.sleep(4)

        # quick auth heuristic: look for a cookie that only appears post-login
        have_auth_cookie = False
        auth_markers = {"SINFN", "SSIDN", "SIDN", "dk_session", "knownUser"}  # varies by account/geo
        names = {c["name"] for c in driver.get_cookies()}
        if auth_markers & names:
            have_auth_cookie = True

        # Also make sure Akamai cookies are present (anti-bot)
        akamai_ok = {"ak_bmsc", "bm_sv", "bm_sz"} & names

        if not have_auth_cookie:
            logging.warning("Did not find a clear 'logged-in' cookie in this profile (found: %s)", list(names)[:10])
        if not akamai_ok:
            logging.warning("Akamai cookies not seen yet; make sure you visited a real DK page (not just the API).")

        # Build session from whatever the browser has right now
        sess = _cookies_to_requests_session(driver)

    finally:
        driver.quit()

    return sess

def download_contest_dataframe(sess: requests.Session, contest_id: int,
                               max_retries: int = 3, timeout: int = 20) -> pd.DataFrame | None:
    gc_url = f"https://www.draftkings.com/contest/gamecenter/{contest_id}"
    export_url = f"https://www.draftkings.com/contest/exportfullstandingscsv/{contest_id}"

    for attempt in range(1, max_retries + 1):
        try:
            # Touch gamecenter first (sets important cookies)
            r1 = sess.get(gc_url, timeout=timeout, allow_redirects=True,
                          headers={"Referer": "https://www.draftkings.com/"})
            r1.raise_for_status()

            # Some flows use a CSRF cookie/header; add if present
            csrf = (sess.cookies.get("_csrf") or
                    sess.cookies.get("csrf_token") or
                    sess.cookies.get("XSRF-TOKEN"))
            hdrs = {
                "Referer": gc_url,
                "Origin": "https://www.draftkings.com",
            }
            if csrf:
                hdrs["X-CSRF-Token"] = csrf

            r2 = sess.get(export_url, timeout=timeout, allow_redirects=True, headers=hdrs)
            r2.raise_for_status()

            ctype = r2.headers.get("Content-Type", "").lower()
            if "zip" not in ctype and "csv" not in ctype:
                # Almost certainly an HTML interstitial (login/bot wall)
                snippet = r2.text[:300].replace("\n", " ")
                print(f"[warn] export returned non-zip content-type={ctype}; html head: {snippet}")
                raise zipfile.BadZipFile("export did not return a zip/csv")

            # Some DK exports return raw CSV (no zip); handle both
            if "csv" in ctype and "zip" not in ctype:
                text = r2.text or ""
                if not text.strip():
                    print(f"[warn] Contest {contest_id} returned an empty CSV body.")
                    raise EmptyDataError("empty CSV body")
                try:
                    return pd.read_csv(io.StringIO(text))
                except EmptyDataError as e:
                    print(f"[warn] Contest {contest_id} CSV parse failed: {e}")
                    raise

            z = zipfile.ZipFile(io.BytesIO(r2.content))
            csv_name = next(f for f in z.namelist() if f.endswith(".csv"))
            try:
                return pd.read_csv(z.open(csv_name))
            except EmptyDataError as e:
                print(f"[warn] Contest {contest_id} ZIP CSV is empty: {e}")
                raise

        except (RemoteDisconnected, zipfile.BadZipFile, RequestException, EmptyDataError) as e:
            print(f"[warn] Contest {contest_id} attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(1.5)

    print(f"[error] Contest {contest_id} skipped after {max_retries} attempts.")
    return None

def fetch_all_contests(sport: str = "MLB") -> List[Dict[str, Any]]:
    """
    Hit the Lobby endpoint and return the raw list of contests.
    """
    logging.info(f"[draftkings] Fetching contests for sport={sport} from {LOBBY_URL}")
    try:
        r = requests.get(
            LOBBY_URL,
            params={"sport": sport},
            timeout=15,  # <--- ADD THIS
        )
        logging.info(f"[draftkings] HTTP status={r.status_code} url={r.url}")
        r.raise_for_status()
    except requests.exceptions.Timeout:
        logging.error("[draftkings] Timeout while calling lobby endpoint.")
        raise
    except requests.exceptions.RequestException as e:
        logging.error(f"[draftkings] RequestException while calling lobby endpoint: {e}")
        raise

    try:
        data = r.json()
    except Exception as e:
        logging.error(f"[draftkings] Failed to parse JSON: {e}; first 300 chars: {r.text[:300]!r}")
        raise

    if isinstance(data, dict) and "Contests" in data:
        return data["Contests"]
    if isinstance(data, dict) and "contests" in data:
        return data["contests"]
    if isinstance(data, list):
        return data

    logging.warning("Unexpected lobby JSON keys: %s", list(data.keys()))
    return []


def summarize_slates(contests: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Build a mapping:
      draftGroupId -> {
        gameType: str,
        sdstring: str,
        name:     str,
        count:    int  # how many contest entries share this dg
      }
    """
    summary: Dict[int, Dict[str, Any]] = {}
    for c in contests:
        dg = c.get("dg") or c.get("id")
        if not dg:
            continue

        if dg not in summary:
            summary[dg] = {
                "gameType": c.get("gameType"),
                "sdstring": c.get("sdstring"),
                "name":     c.get("n"),
                "count":    1
            }
        else:
            summary[dg]["count"] += 1

    return summary


def get_upcoming_slate_ids(sport: str = "MLB") -> List[int]:
    """
    Returns all unique draftGroupIds for upcoming slates.
    """
    contests = fetch_all_contests(sport)
    seen: List[int] = []
    for c in contests:
        dg = c.get("dg")
        if dg and dg not in seen:
            seen.append(dg)

    if not seen:
        raise RuntimeError(f"No upcoming slates for '{sport}'")
    logging.info("Found %d slates: %s", len(seen), seen)
    return seen


def get_players_for_slate(draft_group_id: int) -> List[Dict[str, Any]]:
    """
    Fetch the draftables for one slate, tagging each with `draftGroupId`
    and pretty‑printing the first payload for inspection.
    """
    url = f"{API_BASE}/draftgroups/{draft_group_id}/draftables"
    logging.debug("GET %s", url)
    r = requests.get(url, timeout=5)
    r.raise_for_status()

    payload = r.json()
    draftables = payload.get("draftables", [])
    # tag for easy upsert
    for p in draftables:
        p["draftGroupId"] = draft_group_id

    return draftables

def get_upcoming_payouts(cid):
    url = DK_CONTESTDETAIL_URL.format(cid=cid)
    try:
        resp = requests.get(url, params={"format": "json"})
        resp.raise_for_status()
        payload = resp.json()
        return payload
    except requests.RequestException as e:
        logger.error(f"HTTP error for contest {cid}: {e}")
        return None
    
def _extract_matchup_teams(comp: dict[str, object]) -> tuple[str|None, str|None]:
    """
    Try to extract (away, home) from the competition payload.
    Priority 1: comp['name'] like 'TEN @ DEN'
    Fallback:   scan for two team tokens in comp['name'] or nameDisplay
    """
    away = home = None

    name = (comp or {}).get("name") or ""
    # normalize common separators to '@'
    normalized = (
        name.replace(" vs. ", " @ ")
            .replace(" vs ", " @ ")
            .replace(" VS ", " @ ")
            .replace(" at ", " @ ")
            .replace(" AT ", " @ ")
    )
    if "@" in normalized:
        parts = [p.strip() for p in normalized.split("@")]
        if len(parts) == 2:
            away, home = parts[0], parts[1]

    if not (away and home):
        # Fallback: try regex on name
        tokens = _TEAM_RE.findall(name)
        if len(tokens) >= 2:
            away, home = tokens[0], tokens[-1]
        else:
            # Last fallback: nameDisplay array
            nd = (comp or {}).get("nameDisplay") or []
            vals = []
            for item in nd:
                if isinstance(item, dict):
                    v = (item.get("value") or "").strip()
                else:
                    v = str(item).strip()
                vals.extend(_TEAM_RE.findall(v))
            if len(vals) >= 2:
                away, home = vals[0], vals[-1]

    # sanity: return None if they look invalid
    if away == home:
        return None, None
    return away, home
    
def _derive_opponent_for_player(team: str|None, comp: dict) -> str|None:
    if not team or not comp:
        return None
    away, home = _extract_matchup_teams(comp)
    if not (away and home):
        return None
    if team == away:
        return home
    if team == home:
        return away
    # If team didn't match either side (rare formatting issues), give up gracefully
    return None

def _parse_gametime_to_local(start_time_str: str|None) -> object|None:
    """
    Parse ISO format datetime string and convert to local timezone.
    Assumes the input is in UTC if no timezone info is present.
    """
    if not start_time_str:
        return None
    
    try:
        # Parse the datetime string
        dt = parser.isoparse(start_time_str)
        
        # If no timezone info, assume UTC
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        
        # Convert to local timezone
        local_dt = dt.astimezone(LOCAL_TZ)
        
        return local_dt
    except Exception as e:
        logging.warning(f"Failed to parse datetime '{start_time_str}': {e}")
        return None
    
class Player:
    def __init__(self, dk_id, name, team, opponent, position, roster_slot_id, gametime, salary, status):
        self.dk_id = dk_id
        self.name = name
        self.team = team
        self.opponent = opponent
        self.position = position
        self.roster_slot_id = roster_slot_id
        self.gametime = gametime
        self.salary = salary
        self.status = status

    
SUPPORTED_GAME_TYPES = ("Classic", "Showdown Captain Mode")


def list_slates(sport) -> list[dict]:
    """
    Upcoming DraftKings slates (draft groups) for a sport, in DK's order.
    Each item: {"id", "game_type", "contests", "start", "name"}.
    """
    slate_info = summarize_slates(fetch_all_contests(sport))
    return [{"id": dg, "game_type": info["gameType"], "contests": info["count"],
             "start": info["sdstring"], "name": info["name"]} for dg, info in slate_info.items()]


def _prompt_for_slate(sport, slates: list[dict]) -> int:
    """Interactive slate selection for terminal use; exits on Q or bad input."""
    print(f"             Available {sport.upper()} Slates:")
    for i, s in enumerate(slates, start=1):
        print(f"                {i:2}. DG={s['id']} | {s['game_type']:8} | entries={s['contests']:2} | at {s['start']} | '{s['name']}'")
    print()
    choice = input(f"Select a slate 1–{len(slates)} (or Q to quit): ").strip().upper()
    if choice == "Q":
        sys.exit(0)
    try:
        return slates[int(choice) - 1]["id"]
    except Exception:
        sys.exit(1)


def ingest_players(sport, dedupe_by_name=True, draft_group_id=None):
    """
    pull all draftables from draftkings for a draft group id. (slate)
    returns player objects for upsert in main script

    Args:
        sport: Sport key ('nba', 'nfl', etc.)
        dedupe_by_name: If True, only keep first occurrence of each player name.
                        If False, keep all roster slots (e.g., NFL showdown CPT/FLEX).
        draft_group_id: Ingest this slate without prompting (web UI / scripted runs).
                        When None, list the slates and ask on stdin as before.
    """
    if draft_group_id is None:
        dg = _prompt_for_slate(sport, list_slates(sport))
    else:
        dg = int(draft_group_id)

    players = get_players_for_slate(dg)

    relevant_teams = set()
    seen_names = set()
    processed_players = set()

    for p in players:
        team = p.get("teamAbbreviation")
        comp = p.get("competition") or {}
        opp = _derive_opponent_for_player(team, comp)
        player_name = normalize_name(p.get("displayName"))
        
        # Optional deduplication by name
        if dedupe_by_name:
            if player_name in seen_names:
                continue
            seen_names.add(player_name)

        if team: relevant_teams.add(team)
        if opp: relevant_teams.add(opp)
        gametime = _parse_gametime_to_local(comp.get("startTime"))
        draftable_id    = p.get("draftableId")
        position        = p.get("position")
        salary          = p.get("salary")
        roster_slot_id  = p.get("rosterSlotId")

        if draftable_id is None:
            continue
        if position is None:
            continue
        if salary is None:
            continue
        
        player = Player(
            dk_id=draftable_id,
            name=player_name,
            team=team,
            opponent=opp,
            position=position,
            roster_slot_id=roster_slot_id,
            gametime=gametime,
            salary=salary,
            status=p.get("status"),
        )
        processed_players.add(player)

    return processed_players
        



