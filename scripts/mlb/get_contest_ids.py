#!/usr/bin/env python
import os
import re
import json
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from ivan.database.session import init_db, get_session
from ivan.sports.mlb.models import HistoricalContest
from ivan.sports.common.draftkings import get_upcoming_contest_ids


def parse_sd(sd_str: str) -> datetime:
    """
    Parse timestamps of the form '/Date(1747782300000)/' into a UTC datetime.
    """
    m = re.search(r"/Date\((\d+)\)/", sd_str or "")
    if not m:
        return None
    ts = int(m.group(1)) / 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc)

def main():
    engine = init_db("mlb")
    db = get_session(engine)

    # 2) 'naked' request, no auth needed for this endpoint
    sess = requests.Session()

    # 3) Fetch & filter contests
    filtered = get_upcoming_contest_ids(sess, sport="MLB", days_ahead=1)

    # 4) Upsert each into historical_contests
    for c in filtered:
        # use 'id' as contest_id
        cid = str(c["id"])
        existing = db.query(HistoricalContest).filter_by(contest_id=cid).one_or_none()

        contest_date = parse_sd(c.get("sd")) or datetime.utcnow()
        entry_fee    = float(c.get("a", 0))
        total_entries= int(c.get("m", 0))

        if not existing:
            hc = HistoricalContest(
                contest_id    = cid,
                draft_group_id= c.get("dg"),
                name          = c.get("n"),
                date          = contest_date,
                entry_fee     = entry_fee,
                contest_type  = c.get("gameType"),
                total_entries = total_entries,
                multi_entry   = c.get("mec")
            )
            db.add(hc)

    db.commit()
    print(f"✅ Saved {len(filtered)} upcoming contests to the database.")

if __name__ == "__main__":
    main()
