#!/usr/bin/env python3
"""
scripts/nfl/backtest_dupes.py

The season test of the showdown dupe model (ivan.sports.nfl.dupe_backtest): every
slate in the window scored by a model fitted only on the fields locked before it,
the live model file scored on the slates after its fitted_on date, the live terms
against ownership-only / this-season-only / plus the entry-cap term, calibration by
predicted copies and by the contest's entries-per-user cap, and our own entries'
predicted copies against the copies the field held.

    backtest_dupes.py                                   slates since 2026-09-01, every field before each as history
    backtest_dupes.py --since 2026-10-16 --until 2026-11-03
    backtest_dupes.py --train-since 2026-09-01           the "asof season" variant's start (default 2026-09-01)
    backtest_dupes.py --model data/nfl/models/asof/2026-10-11/dupe_model_v2.json
    backtest_dupes.py --no-actual-own                    skip the actual-ownership rerun (halves the time)
    backtest_dupes.py --no-write                         print only

Output: data/nfl/diagnostics/dupe_backtest/<stamp>/ (report.txt, slates.csv, summary.csv, calibration.csv, by_entry_max.csv, ours.csv, summary.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ivan.database.session import get_session, init_db
from ivan.sports.nfl import dupe_backtest as DB
from ivan.sports.nfl import dupe_model as DM


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=DB.SEASON_START, help="first slate scored (YYYY-MM-DD)")
    ap.add_argument("--until", default=None, help="last slate scored (YYYY-MM-DD, inclusive)")
    ap.add_argument("--train-since", default=None, help="the this-season variant fits only on fields from this date (default the season start)")
    ap.add_argument("--model", default=None, help="a dupe_model_v2.json to score as the fixed file (default the live model)")
    ap.add_argument("--user", default=None, help="DK username whose entries count as ours (default IVAN_DK_USERNAME, else every scraped entry of ours)")
    ap.add_argument("--no-actual-own", action="store_true", help="skip the calibration from actual ownership")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=None, help="output root (default data/nfl/diagnostics/dupe_backtest)")
    ap.add_argument("--db-url", default=None)
    a = ap.parse_args(argv)
    session = get_session(init_db("nfl", a.db_url))
    model = None
    if a.model:
        raw = json.loads(Path(a.model).read_text())
        model = DM.load(Path(a.model).parent) if Path(a.model).name == DM.FILE else DM.ProductDupeModel(params=raw["params"], fitted_on=raw.get("fitted_on"), contests=raw.get("contests") or {})
    r = DB.run(session, since=a.since, until=a.until, train_since=a.train_since, model=model, username=a.user, actual_own=not a.no_actual_own)
    print(DB.format_report(r))
    if not a.no_write:
        out = DB.write(r, Path(a.out) if a.out else None)
        print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
