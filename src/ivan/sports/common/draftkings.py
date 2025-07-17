import requests
import time
from datetime import date, timedelta
from pprint import pprint
from selenium.webdriver.chrome.options import Options
from selenium import webdriver

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

