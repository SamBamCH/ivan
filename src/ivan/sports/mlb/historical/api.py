# ─── src/ivan/sports/mlb/historical/api.py ───────────────────────────────────────

import math
import time
from sqlalchemy import func

from ivan.sports.mlb.models import HistoricalContest
from ivan.sports.mlb.models import BatterProjection, PitcherProjection
from dateutil.parser import isoparse
import requests
import zipfile, io, csv, re

# ─── src/ivan/sports/mlb/historical/api.py ───────────────────────────────────────

def upsert_contest_metadata(db, contest_id: int, session_http):
    """
    Calls GET /contests/v1/contests/{contest_id}?format=json,
    then upserts into HistoricalContest.
    This version reads from j["contestDetail"].
    Returns: (draft_group_id: int, contest_date: str).
    """
    url = f"https://api.draftkings.com/contests/v1/contests/{contest_id}"
    resp = session_http.get(url, params={"format": "json"})
    resp.raise_for_status()
    j = resp.json()
    cd = j.get("contestDetail", {})

    # extract the fields from contestDetail
    dg_id    = int(cd["draftGroupId"])
    # the contest start time is now under contestDetail["contestStartTime"]
    contest_dt = isoparse(cd["contestStartTime"])
    # entry fee is under contestDetail["entryFee"]
    entry_fee  = float(cd.get("entryFee", 0.0))
    # total entries available in this contest
    total_entries = int(cd.get("entries", 0))

    # upsert into HistoricalContest
    existing = db.query(HistoricalContest).filter_by(contest_id=contest_id).one_or_none()
    if existing:
        existing.draft_group_id = dg_id
        existing.date           = contest_dt
        existing.entry_fee      = entry_fee
        existing.total_entries  = total_entries
        existing.contest_type   = cd.get("playTypeId") or existing.contest_type
    else:
        hc = HistoricalContest(
            contest_id     = contest_id,
            draft_group_id = dg_id,
            date           = contest_dt,
            entry_fee      = entry_fee,
            total_entries  = total_entries,
            contest_type   = cd.get("playTypeId"),
            multi_entry    = int(cd.get("maximumEntriesPerUser", 1))
        )
        db.add(hc)

    db.commit()
    return dg_id, contest_dt


import io
import csv
import re
import zipfile
import requests

def fetch_all_lineup_names(draft_group_id: int, contest_id: int, api_session: requests.Session) -> list[list[str]]:
    # 1) Hit the DK export endpoint, but don’t follow the redirect
    url = f"https://www.draftkings.com/contest/exportfullstandingscsv/{contest_id}"
    resp1 = api_session.get(url, allow_redirects=False)
    if resp1.status_code == 302:
        # 2) Grab the signed-S3 URL
        s3_url = resp1.headers["Location"]
        # 3) Re-GET it without cookies (just carry over a User-Agent)
        resp = requests.get(
            s3_url,
            headers={"User-Agent": api_session.headers["User-Agent"]},
            allow_redirects=True
        )
    else:
        # fallback if somehow there’s no redirect
        print(f"[DEBUG] no redirect from {url}")
        resp = resp1

    resp.raise_for_status()
    raw = resp.content

    # unzip if it really is a .zip
    if raw[:2] == b"PK":
        zf = zipfile.ZipFile(io.BytesIO(raw))
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(csv_name).decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")

    # parse the TSV
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    split_re = re.compile(r'\b(?:1B|2B|3B|C|SS|OF|P)\s+')

    lineups: list[list[str]] = []
    for row in reader:
        lineup = (row.get("Lineup") or "").strip()
        if not lineup:
            continue
        names = [nm.strip() for nm in split_re.split(lineup) if nm.strip()]
        lineups.append(names)

    print(f"[DEBUG] extracted {len(lineups)} lineups; sample: {lineups[:2]}")
    return lineups

# ③ Fetch one entry’s roster
def fetch_roster_for_entry(draft_group_id: int, entry_key: int, session_http):
    """
    GET /scores/v2/entries/{draft_group_id}/{entry_key}?embed=roster
    Return a flat list of all player display names in that entry.
    """
    url = f"https://api.draftkings.com/scores/v2/entries/{draft_group_id}/{entry_key}"
    r = session_http.get(url, params={"format": "json", "embed": "roster"})
    r.raise_for_status()

    data = r.json()
    names = []
    for entry in data.get("entries", []):
        # drill into roster → scorecards
        scorecards = entry.get("roster", {}).get("scorecards", [])
        names.extend(card.get("displayName") for card in scorecards if "displayName" in card)
    return names
