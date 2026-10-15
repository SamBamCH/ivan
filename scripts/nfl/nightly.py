#!/usr/bin/env python3
"""
scripts/nfl/nightly.py

The data job that keeps every played slate backtestable: schedules (scores and
closing lines) and box scores from nflverse, the DST game table, then the
settle job for the slates locked in the last days (standings of the contests
we entered and the slate's largest public contests, derived field files, a
status row per slate). It writes nothing else and deletes nothing, so it is
safe on a timer; settle is idempotent, so running it several times a day only
fetches what is still missing.

    nightly.py [--days 3] [--no-scrape] [--headless]

The DraftKings scrape uses the saved Chrome profile (chrome-selenium/dk-profile)
and IVAN_DK_USERNAME from .env so our finishes are recorded. With a display it
opens Chrome and, when the login has expired, waits up to ten minutes for a
login in that window; without one it runs headless, and an expired login marks
the slate login_needed (the Backtest page shows it; settle again after logging
in). deploy/ivan-nightly.timer runs this three times a day with catch-up, so a
laptop asleep overnight runs it when it wakes.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime, timedelta

from ivan.database.session import get_session, init_db
from ivan.sports.nfl.dst_scoring import build_dst_game_table
from ivan.sports.nfl.refactor.historical.data_fetch import fetch_boxscores, fetch_schedules, seasons_to_fetch
from ivan.sports.nfl.settle import settle


def log(msg: str) -> None:
    print(f"[nightly {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=3, help="settle slates locked in the last N days (default 3)")
    ap.add_argument("--no-scrape", action="store_true", help="box scores and status only; no DraftKings scrape")
    ap.add_argument("--headless", action="store_true", help="force headless Chrome (the default when there is no display)")
    ap.add_argument("--public", type=int, default=3, help="largest public contests per slate to settle as well (default 3)")
    a = ap.parse_args()
    session = get_session(init_db("nfl"))
    rc = 0

    log("schedules and box scores from nflverse")
    try:
        log(f"nfl_schedule: {fetch_schedules(session)} games")
        log(f"nfl_stats: {fetch_boxscores(session)} player-weeks")
        log(f"nfl_dst_game: {build_dst_game_table(None, years=seasons_to_fetch(), session=session)} team-games")
    except Exception as e:   # noqa: BLE001 - a network failure must not stop the settlement of what we already have
        rc = 1
        log(f"box score fetch FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")

    username = os.environ.get("IVAN_DK_USERNAME")
    headless = a.headless or not os.environ.get("DISPLAY")
    until = datetime.now()
    since = until - timedelta(days=a.days)
    log(f"settle slates locked since {since:%Y-%m-%d %H:%M}" + (" (no scrape)" if a.no_scrape else f" (scrape{' headless' if headless else ''}"
        + (f", username {username}" if username else ", NO USERNAME: our finishes will not be recorded") + ")"))
    try:
        items = settle(session, since, until, username=username, do_scrape=not a.no_scrape, headless=headless,
                       fetch_box_scores=False, public_n=a.public, echo=log)
    except Exception as e:   # noqa: BLE001
        log(f"settle FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return 1
    for s in items:
        log(f"  {s.slate_key} {s.mode} lock {s.lock:%Y-%m-%d %H:%M}: {s.status}" + (f" ({'; '.join(s.missing)})" if s.missing else ""))
    pending = [s for s in items if s.status in ("waiting", "login_needed")]
    if pending:
        log(f"{len(pending)} slate(s) still {', '.join(sorted({s.status for s in pending}))}")
    log("done" if rc == 0 and not pending else "done with issues")
    return rc or (1 if pending else 0)


if __name__ == "__main__":
    sys.exit(main())
