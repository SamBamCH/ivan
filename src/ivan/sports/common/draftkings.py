import time
import io
import zipfile
import logging
import json
from datetime import date, timedelta
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from http.client import RemoteDisconnected
from requests.exceptions import RequestException


# Base URLs
LOBBY_URL = "https://www.draftkings.com/lobby/getcontests"
API_BASE  = "https://api.draftkings.com/draftgroups/v1"

def get_upcoming_contest_ids(
    sess: requests.Session,
    sport: str = "MLB",
    days_ahead: int = 1,
    entry_filter: int = 200
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
        if is_g and (m > entry_filter or (a > 50 and m > 100)):
            filtered.append(c)

    # 5) return the IDs you actually want to download
    #    here 'dg' is the DraftGroupId shorthand in this payload
    return filtered

def build_requests_session(profile_dir: str, headless: bool = True) -> requests.Session:
    """
    Launch Chrome with your real profile, ensure you're logged in,
    copy cookies + UA into a requests.Session.
    """
    opts = Options()
    # run non-headless so Chrome can fully load the persisted profile/cookies
    if headless:
        opts.add_argument("--headless=new")  # if you must, but prefer False
    opts.add_argument(f"--user-data-dir={profile_dir}")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--lang=en-US")

    driver = webdriver.Chrome(options=opts)
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
                return pd.read_csv(io.StringIO(r2.text))

            z = zipfile.ZipFile(io.BytesIO(r2.content))
            csv_name = next(f for f in z.namelist() if f.endswith(".csv"))
            return pd.read_csv(z.open(csv_name))

        except (RemoteDisconnected, zipfile.BadZipFile, RequestException) as e:
            print(f"[warn] Contest {contest_id} attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(1.5)

    print(f"[error] Contest {contest_id} skipped after {max_retries} attempts.")
    return None

def fetch_all_contests(sport: str = "MLB") -> List[Dict[str, Any]]:
    """
    Hit the Lobby endpoint and return the raw list of contests.
    """
    r = requests.get(LOBBY_URL, params={"sport": sport})
    r.raise_for_status()
    data = r.json()

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
    r = requests.get(url)
    r.raise_for_status()

    payload = r.json()
    draftables = payload.get("draftables", [])

    # dump the first one so you know all available keys
    if draftables:
        logging.info("=== First draftable payload ===\n%s", json.dumps(draftables[0], indent=2))
    else:
        logging.warning("No draftables for DG=%s", draft_group_id)

    # tag for easy upsert
    for p in draftables:
        p["draftGroupId"] = draft_group_id

    return draftables

