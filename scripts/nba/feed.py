#!/usr/bin/env python3
"""
scripts/nba/feed.py

The NBA data feed between two Ivans (ivan.core.feed).

Producer (the machine that runs the capture loop and fetches from DK once):
    scripts/nba/feed.py publish [--which live|reference|both]     write data/feed/nba/*; the capture tick and the
                                                                   nightly job call this themselves
    scripts/nba/feed.py push [DEST]                               rsync the bundles to the consumer's feed directory
                                                                   (DEST or $IVAN_FEED_PUSH, e.g. ivan-server:/opt/ivan/app/data/feed);
                                                                   the capture tick and the nightly job push too
Consumer (another machine):
    scripts/nba/feed.py pull [--url file:///opt/ivan/app/data/feed] [--which both] [--no-project]
        a pushed-to machine pulls from its own disk (file:// URL, no token); a machine reaching a feed server
        uses --url http://<feed host>:8790 --token ...; both
        fetches whatever changed, replaces the feed's tables in this database (edits, rotations, lineups and sims
        here are untouched) and re-projects the slate with this machine's edits so the desk and the generator see
        its own projections. --url and --token default to $IVAN_FEED_URL / $IVAN_FEED_TOKEN (.env).
    scripts/nba/feed.py pull --loop 15                            keep pulling every 15 minutes
    scripts/nba/feed.py apply DIR                                 seed this database from bundles copied by hand
                                                                   (DIR holds live.sql.gz / reference.sql.gz / files/)
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from ivan.core import feed
from ivan.core.utils import PROJECT_ROOT


def do_pull(a) -> dict:
    url = a.url or os.environ.get("IVAN_FEED_URL"); token = a.token or os.environ.get("IVAN_FEED_TOKEN")
    if not url:
        raise SystemExit("give --url or set IVAN_FEED_URL in .env")
    changed = feed.pull(url, token, "nba", PROJECT_ROOT / "data" / "feed-state", which=a.which, log=print)
    if changed and not a.no_project and "live" in changed:
        from ivan.database.session import get_session, init_db
        from ivan.sports.nba.ingest import ensure_schema, project_slate
        engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
        try:
            info = project_slate(session, n_sims=2000, write_capture=False, write_csv=False, log=print)
            print(f"re-projected {info.players} players with this machine's edits")
        except Exception as e:   # noqa: BLE001
            print(f"re-projection skipped: {type(e).__name__}: {e}")
    if not changed:
        print("feed unchanged")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish"); p.add_argument("--which", choices=("live", "reference", "both"), default="both")
    q = sub.add_parser("pull"); q.add_argument("--url", default=None); q.add_argument("--token", default=None)
    q.add_argument("--which", choices=("live", "reference", "both"), default="both"); q.add_argument("--no-project", action="store_true")
    q.add_argument("--loop", type=int, default=None, help="minutes between pulls; omit for one pull")
    r = sub.add_parser("apply"); r.add_argument("dir")
    u = sub.add_parser("push"); u.add_argument("dest", nargs="?", default=None)
    a = ap.parse_args()
    if a.cmd == "push":
        ok = feed.push("nba", a.dest, log=print)
        if not ok and not (a.dest or os.environ.get(feed.PUSH_ENV)):
            print("give DEST or set IVAN_FEED_PUSH in .env")
        return 0 if ok else 1
    if a.cmd == "apply":
        import shutil
        d = Path(a.dir)
        for part in ("live", "reference"):
            if (d / f"{part}.sql.gz").exists():
                feed.apply("nba", d / f"{part}.sql.gz", log=print)
        if (d / "files").is_dir():
            shutil.copytree(d / "files", PROJECT_ROOT, dirs_exist_ok=True); print("reference files copied")
        return 0
    if a.cmd == "publish":
        m = feed.publish("nba", which=a.which, log=print)
        print(f"manifest: live {m.get('live', {}).get('stamp')} reference {m.get('reference', {}).get('stamp')} -> {feed.FEED_DIR / 'nba'}")
        return 0
    if a.loop:
        while True:
            try:
                do_pull(a)
            except Exception as e:   # noqa: BLE001
                print(f"pull failed: {type(e).__name__}: {e}")
            time.sleep(a.loop * 60)
    do_pull(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
