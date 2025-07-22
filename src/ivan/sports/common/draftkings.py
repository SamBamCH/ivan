import requests
from selenium import webdriver
import requests
import pandas as pd
import io, zipfile
from datetime import date, timedelta
from pprint import pprint
import pandas as pd



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

def build_requests_session(profile_dir: str) -> requests.Session:
    opts = webdriver.ChromeOptions()
    opts.add_argument(f"--user-data-dir={profile_dir}")
    driver = webdriver.Chrome(options=opts)

    # hit home to ensure you’re signed‑in
    driver.get("https://www.draftkings.com")
    # collect cookies
    sess = requests.Session()
    for c in driver.get_cookies():
        sess.cookies.set(c["name"], c["value"], domain=c["domain"], path=c["path"])
    # set headers once
    sess.headers.update({
        "User-Agent": driver.execute_script("return navigator.userAgent"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    driver.quit()
    return sess


def get_contest_dataframe(sess: requests.Session, contest_id: int) -> pd.DataFrame:
    # 1) hit the gamecenter page so it sets the uc=… cookie for this contest
    gc_url = f"https://www.draftkings.com/contest/gamecenter/{contest_id}"
    resp = sess.get(gc_url)
    resp.raise_for_status()

    # 2) inject CSRF from cookies
    csrf = sess.cookies.get("_csrf") or sess.cookies.get("csrf_token")
    if csrf:
        sess.headers["X-CSRF-Token"] = csrf

    # 3) download the ZIP
    export_url = f"https://www.draftkings.com/contest/exportfullstandingscsv/{contest_id}"
    resp = sess.get(export_url, headers={"Referer": gc_url})
    resp.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = next(f for f in z.namelist() if f.endswith(".csv"))
    return pd.read_csv(z.open(csv_name))

