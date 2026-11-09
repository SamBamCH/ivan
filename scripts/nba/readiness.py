#!/usr/bin/env python3
"""
scripts/nba/readiness.py

Which past days are ready to backtest (ivan.sports.nba.backtest.readiness): what the calendar and the picker colour.

    scripts/nba/readiness.py                 every day with a held past slate: ready / pending / failed and why
    scripts/nba/readiness.py --slate 156176  one slate: the box scores, the money, the scrape record

ready = box scores on file for every team of the slate and, where we played, a final standings export of every contest of
ours; failed = a standings scrape recorded failures (scripts/nba/standings.py scrape), or the box scores are still missing
36 hours after lock; pending = not attempted yet.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

from ivan.sports.nba.backtest import readiness as R
from ivan.sports.nba import slates as S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    now = datetime.now()
    if a.slate:
        st = R.slate_status(int(a.slate), now)
        print(json.dumps(st, indent=1, default=str))
        return 0 if st["state"] != "failed" else 2
    days = R.day_status(now, force=True)
    if a.json:
        print(json.dumps(days, indent=1, default=str)); return 0
    if not days:
        print("no held slate of a past day"); return 0
    for d, st in days.items():
        print(f"{d}  {st['state']:8}  slates {', '.join(str(x) for x in st['slates'])}" + (f"  {st['why']}" if st["why"] else ""))
    return 0 if not any(s["state"] == "failed" for s in days.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
