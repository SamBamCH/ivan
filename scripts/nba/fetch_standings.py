#!/usr/bin/env python3
"""
scripts/nba/fetch_standings.py

Download the live standings export of every contest in the entries file and keep it (raw
dk/standings/<contest id>/<stamp>.csv.gz). The late swap's fetch step, apart from the swap:
run it a minute after a tip (the export then shows that game's players as locked), and run
the swap (scripts/nba/lateswap.py, --source auto) as often as the projections move; it uses
the stored export until the next tip. Needs the logged-in DraftKings Chrome profile; the
desk's late swap tab has the same button (Fetch standings).

    scripts/nba/fetch_standings.py                         # every contest in the entries file
    scripts/nba/fetch_standings.py --contest 197015856     # one contest (repeatable)
"""
import argparse
import os
from datetime import datetime

from ivan.core.config import load_config
from ivan.database.session import get_session, init_db
from ivan.sports.nba.ingest import ensure_schema
from ivan.sports.nba.paths import CHROME_PROFILE_DIR, DEFAULT_ENTRIES_CSV
from ivan.sports.nba.post_lock import lateswap as LS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", default=None, help="entries CSV (default: paths.entries_csv from config)")
    ap.add_argument("--contest", type=int, action="append", default=None, help="only this contest id (repeatable)")
    ap.add_argument("--headless", action="store_true", help="headless Chrome (DraftKings answers 403 to headless sessions; only for testing)")
    ap.add_argument("--now", default=None, help="pretend it is this time (ISO), for the lock clock in the log")
    a = ap.parse_args()
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    entries = a.entries
    if entries is None:
        try:
            entries = load_config("nba")["paths"]["entries_csv"]
        except Exception:   # noqa: BLE001
            entries = str(DEFAULT_ENTRIES_CSV)
    now = datetime.fromisoformat(a.now) if a.now else datetime.now()
    from ivan.sports.common.draftkings import build_requests_session_from_chrome
    headless = a.headless or not os.environ.get("DISPLAY")
    if headless and not a.headless:
        print("no DISPLAY: Chrome runs headless (a logged-out profile fails; log in from a terminal session first)", flush=True)
    try:
        http = build_requests_session_from_chrome(str(CHROME_PROFILE_DIR), headless=headless)
    except RuntimeError as e:
        print(str(e), flush=True)
        return 1
    got = LS.fetch_standings(session, entries, http, only=a.contest, log=lambda m: print(m, flush=True), now=now)
    print(f"{len(got)} export(s) stored" + ("; the late swap (--source auto) uses them until the next tip" if got else ""), flush=True)
    return 0 if got else 1


if __name__ == "__main__":
    raise SystemExit(main())
