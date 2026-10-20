#!/usr/bin/env python3
"""
scripts/mma/get_contest_data.py

Export the full standings of played MMA contests from DraftKings and keep
them raw (data/mma/raw/dk/standings/<contest id>/<stamp>.csv.gz), the way
the NFL settle job does. The export needs a logged-in DraftKings session:
Chrome opens on the saved profile (chrome-selenium/dk-profile) and, if the
session has expired, waits for you to log in there.

    scripts/mma/get_contest_data.py --contest-ids 196337922,196337924
    scripts/mma/get_contest_data.py --played          # every contest in the database whose start has passed

DraftKings keeps standings about 14 days, so scrape soon after the card.
The exports are parsed by ivan.sports.mma.standings (real lineups, fighter
actuals and drafted %) for the dupe and field models.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

from ivan.core import raw
from ivan.database.session import get_session, init_db
from ivan.sports.common.draftkings import build_requests_session_from_chrome, download_contest_dataframe
from ivan.sports.mma.models import HistoricalContest
from ivan.sports.nfl.paths import CHROME_PROFILE_DIR          # the one DraftKings login profile, shared across sports

log = logging.getLogger("mma.get_contest_data")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export played MMA contests' standings from DraftKings into the raw store")
    ap.add_argument("--contest-ids", default=None, help="comma-separated DK contest ids")
    ap.add_argument("--played", action="store_true", help="every contest in the database that has started")
    ap.add_argument("--min-entries", type=int, default=0, help="with --played: skip contests smaller than this")
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    session = get_session(init_db("mma"))
    raw.ensure_tables(session.get_bind())
    ids: list[int] = []
    if a.contest_ids:
        ids = [int(x) for x in a.contest_ids.split(",") if x.strip()]
    elif a.played:
        rows = session.query(HistoricalContest).filter(HistoricalContest.date < datetime.now()).all()
        ids = [int(r.contest_id) for r in rows if int(r.total_entries or 0) >= a.min_entries]
    if not ids:
        print("nothing to export: pass --contest-ids or --played")
        return 1
    have = {int(x.key) for x in raw.index("mma") if x.kind == "dk/standings"}
    todo = [i for i in ids if i not in have]
    print(f"{len(ids)} contest(s), {len(ids) - len(todo)} already exported")
    if not todo:
        return 0

    sess = build_requests_session_from_chrome(str(CHROME_PROFILE_DIR), headless=a.headless)
    ok = 0
    for cid in todo:
        df = download_contest_dataframe(sess, cid, raw_sport="mma")
        if df is None or df.empty:
            print(f"  {cid}: no export (not finished yet, purged, or not logged in)")
            continue
        ok += 1
        print(f"  {cid}: {len(df):,} rows")
    print(f"exported {ok} of {len(todo)}; raw artifacts indexed: {raw.sync_index(session, 'mma')} new")
    return 0 if ok == len(todo) else 2


if __name__ == "__main__":
    raise SystemExit(main())
