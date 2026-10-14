#!/usr/bin/env python3
"""
Settle the slates we played: standings for the contests we entered and the
draft group's largest public contests, the week's box scores, the derived
lineup files, and a status row per slate (ivan.sports.nfl.settle).

    scripts/nfl/settle.py [--days 3 | --since 2026-10-01 --until 2026-10-12] [--username NAME]
                          [--public 3] [--no-scrape] [--no-box-scores] [--headless] [--dry-run]

Idempotent: it only fetches what is still missing. The scrape needs the
saved DraftKings Chrome profile; without a display pass --headless (which
fails with 403 once the profile's login cookies are stale; the slate is then
marked login_needed). The overnight job runs this with --days 3.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

from ivan.database.session import get_session, init_db
from ivan.sports.nfl.settle import settle


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=3, help="settle slates locked in the last N days (default 3)")
    ap.add_argument("--since", default=None); ap.add_argument("--until", default=None)
    ap.add_argument("--username", default=os.environ.get("IVAN_DK_USERNAME"))
    ap.add_argument("--public", type=int, default=3, help="largest public contests per slate to settle as well (default 3)")
    ap.add_argument("--no-scrape", action="store_true"); ap.add_argument("--no-box-scores", action="store_true")
    ap.add_argument("--headless", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.since and a.until:
        since, until = datetime.fromisoformat(a.since), datetime.fromisoformat(a.until) + timedelta(days=1) - timedelta(seconds=1)
    else:
        until = datetime.now(); since = until - timedelta(days=a.days)
    if not a.no_scrape and not a.headless and not os.environ.get("DISPLAY") and sys.platform.startswith("linux") and not a.dry_run:
        print("[settle] no display for Chrome: pass --headless (after a login in the profile) or --no-scrape")
        return 2
    session = get_session(init_db("nfl"))
    items = settle(session, since, until, username=a.username, do_scrape=not a.no_scrape, headless=a.headless,
                   fetch_box_scores=not a.no_box_scores, public_n=a.public, dry_run=a.dry_run)
    waiting = [s for s in items if s.status in ("waiting", "login_needed")]
    return 0 if not waiting else 1


if __name__ == "__main__":
    raise SystemExit(main())
