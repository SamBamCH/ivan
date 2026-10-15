#!/usr/bin/env python3
"""
scripts/nfl/dupe_report.py

The dupe model's predicted copies against the copies real fields held
(ivan.sports.nfl.dupe_analysis): per contest, calibration by predicted copies,
residual by feature (captain, split, stack, salary left, ownership sum) and
the most duplicated lineups. The web UI's Dupes page shows the same.

    dupe_report.py                         the latest scraped slate
    dupe_report.py --since 2026-10-01      every scraped contest since a date
    dupe_report.py --contests 1,2,3        specific contest ids
    dupe_report.py --list                  the scraped contests
"""
from __future__ import annotations

import argparse

import pandas as pd

from ivan.database.session import get_session, init_db
from ivan.sports.nfl import dupe_analysis as DA


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contests", default=None, help="comma-separated contest ids")
    ap.add_argument("--since", default=None, help="every scraped contest from this date (YYYY-MM-DD)")
    ap.add_argument("--showdown-only", action="store_true")
    ap.add_argument("--list", action="store_true", help="list the scraped contests and exit")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--db-url", default=None)
    a = ap.parse_args()
    session = get_session(init_db("nfl", a.db_url))
    sc = DA.scraped_contests(session)
    if a.list:
        with pd.option_context("display.width", 220, "display.max_rows", None, "display.max_colwidth", 70):
            print(sc.to_string(index=False) if len(sc) else "no scraped fields")
        return 0
    if a.contests:
        ids = [int(x) for x in a.contests.split(",") if x.strip()]
    elif a.since:
        ids = sc[sc.date >= pd.Timestamp(a.since)].contest_id.astype(int).tolist()
    else:
        ids = sc[sc.draft_group_id == sc.draft_group_id.iloc[0]].contest_id.astype(int).tolist() if len(sc) else []
    if a.showdown_only:
        ids = [i for i in ids if bool(sc.loc[sc.contest_id == i, "is_showdown"].any())]
    if not ids:
        print("no contests selected (see --list)")
        return 1
    print(DA.format_report(DA.analyze(session, ids, top_n=a.top)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
