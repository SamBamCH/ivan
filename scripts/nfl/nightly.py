#!/usr/bin/env python3
"""
scripts/nfl/nightly.py

The data job that keeps every played slate backtestable: schedules (scores and
closing lines) and box scores from nflverse, the DST game table, then the
settle job for the slates locked in the last days (standings of the contests
we entered and the slate's largest public contests, derived field files, a
status row per slate), then the retention sweep (scripts/nfl/retention.py:
worker clones of the replay database a day after their last write, a run's
repeats, validation csvs and database copy 30 days after its last write;
never an input, a manifest or a report). Settle is idempotent, so running it
several times a day only fetches what is still missing.

    nightly.py [--days 13] [--no-scrape] [--headless] [--no-notify] [--no-retention]

The DraftKings scrape uses the saved Chrome profile (chrome-selenium/dk-profile)
and IVAN_DK_USERNAME from .env so our finishes are recorded. With a display it
opens Chrome and, when the login has expired, waits up to ten minutes for a
login in that window; without one it runs headless, and an expired login marks
the slate login_needed (the Backtest page shows it; settle again after logging
in). Anything still pending after a run raises a desktop notification, so a
stale login is seen the same day and not after DraftKings has expired the
contest data. deploy/ivan-nightly.timer runs this three times a day with
catch-up, so a laptop asleep overnight runs it when it wakes; the window
keeps retrying a missed slate until DraftKings purges it (14 days).
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
from ivan.sports.nfl import retention
from ivan.sports.nfl.settle import settle


def log(msg: str) -> None:
    print(f"[nightly {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=13, help="settle slates locked in the last N days (default 13: DraftKings purges contest data after 14, so a missed slate is retried until then)")
    ap.add_argument("--no-notify", action="store_true", help="no desktop notification when a slate is still pending")
    ap.add_argument("--no-scrape", action="store_true", help="box scores and status only; no DraftKings scrape")
    ap.add_argument("--headless", action="store_true", help="force headless Chrome (the default when there is no display)")
    ap.add_argument("--public", type=int, default=3, help="largest public contests per slate to settle as well (default 3)")
    ap.add_argument("--no-retention", action="store_true", help="skip the retention sweep of replay clones and old runs' heavy files")
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
        note = "; ".join(s.missing)
        if s.status not in ("open",) and not s.played:
            note = (note + "; " if note else "") + "no chooser output for this slate on this machine (only the public contests are scraped)"
        log(f"  {s.slate_key} {s.mode} lock {s.lock:%Y-%m-%d %H:%M}: {s.status}" + (f" ({note})" if note else ""))
    pending = [s for s in items if s.status in ("waiting", "login_needed", "no_contests")]
    if pending:
        log(f"{len(pending)} slate(s) still {', '.join(sorted({s.status for s in pending}))}")
    if not a.no_retention:
        try:
            sw = retention.sweep(log=log)
            log(f"retention freed {sw.freed / 1e9:.1f} GB")
        except Exception as e:   # noqa: BLE001 - housekeeping must not turn a settled night into a failure
            log(f"retention sweep FAILED: {type(e).__name__}: {e}")
    if (pending or rc) and not a.no_notify:
        notify(pending, rc)
    log("done" if rc == 0 and not pending else "done with issues")
    return rc or (1 if pending else 0)


def notify(pending, rc: int) -> None:
    """A desktop notification (notify-send) so a login_needed or a failed fetch is seen the same day; silent without a display."""
    import shutil
    import subprocess
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return
    if not shutil.which("notify-send"):
        return
    lines = [f"{s.lock:%a %m-%d %H:%M} {s.mode} {s.slate_key}: {s.status}" for s in pending]
    if rc:
        lines.append("box score fetch failed (see data/nfl/logs/nightly.log)")
    urgency = "critical" if any(s.status == "login_needed" for s in pending) or rc else "normal"
    body = "\n".join(lines[:8]) + ("\n..." if len(lines) > 8 else "")
    if any(s.status == "login_needed" for s in pending):
        body += "\nLog in to DraftKings in the scraper's Chrome window, then settle again (Backtest page)."
    try:
        subprocess.run(["notify-send", "-u", urgency, "-a", "Ivan", "Ivan NFL: slates not settled", body], timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


if __name__ == "__main__":
    sys.exit(main())
