#!/usr/bin/env python3
"""
scripts/nba/feed.py
The NBA data feed between two Ivans (ivan.core.feed). The consumer pulls; the producer never pushes.
Producer (the laptop: runs the desk, fetches from DK and the odds API):
    scripts/nba/feed.py publish [--which live|reference|both] [--quiet]
        write data/feed/nba/*: the live bundle (the pool, props, lines, lineups, headlines, the published edits, the
        projection store), the reference bundle (history, models, captures). The consumer's pull asks for a fresh live
        bundle itself (deploy/feed_serve.sh runs this); the nightly writes both.
Consumer (the server):
    scripts/nba/feed.py pull [--no-project]
        what the desk's "Refresh from laptop" does (ivan.core.feed.pull_from_producer): with IVAN_FEED_SSH=<user>@<host>
        in .env, one ssh command has the laptop publish its live bundle, rsync copies its data/feed here (only what
        changed), and the bundles are applied: the feed's tables replaced in this database (edits, lineups and sims here
        untouched), the laptop's pins and grids as the published layer, its projection store mirrored; then only the
        slate's teams this machine edited are re-projected on top (--no-project skips that). The older
        IVAN_FEED_URL (a directory or a feed server, with IVAN_FEED_TOKEN) still works when IVAN_FEED_SSH is unset.
    scripts/nba/feed.py pull --loop 15                            keep pulling every 15 minutes
    scripts/nba/feed.py apply DIR                                 seed this database from bundles copied by hand
                                                                   (DIR holds live.sql.gz / reference.sql.gz / files/)
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from ivan.core import feed
from ivan.core.utils import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")        # IVAN_FEED_SSH (or IVAN_FEED_URL / IVAN_FEED_TOKEN)


def do_pull(a) -> dict:
    src = feed.producer_source()
    if a.url:
        src = ("url", a.url)
    if src is None:
        raise SystemExit(f"set {feed.SSH_ENV}=<user>@<host> (or {feed.URL_ENV}) in .env, or give --url")
    from ivan.database.session import get_session, init_db
    from ivan.sports.nba.ingest import after_pull, ensure_schema
    from ivan.sports.nba.models import NBAPlayer
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    row = session.query(NBAPlayer.draft_group_id).first()
    previous = int(row[0]) if row and row[0] else None                 # the slate held before the pull: its edits are filed if the slate moves
    session.close()
    state_dir = PROJECT_ROOT / "data" / "feed-state"
    if src[0] == "ssh":
        changed = feed.pull_ssh(src[1], "nba", state_dir, log=print)
    else:
        changed = feed.pull(src[1], a.token or os.environ.get("IVAN_FEED_TOKEN"), "nba", state_dir, which=a.which, log=print)
    if not {k: v for k, v in changed.items() if k != "published"}:
        print("nothing new")
    if changed and not a.no_project and "live" in changed:
        # the published projection stands as pulled; this machine's own edits of the old slate are filed if the slate moved, and only
        # the slate's teams it edited (its own pins or grids) are re-projected on top (ingest.after_pull)
        session = get_session(engine)
        try:
            r = after_pull(session, previous, log=print)
            if r["teams"]:
                print(f"re-projected {r['players']} players of {len(r['teams'])} team(s) with this machine's own edits on top of the published projection")
            else:
                print("the published projection stands (no edits of this machine's own on this slate)")
        except Exception as e:   # noqa: BLE001
            print(f"re-projection skipped: {type(e).__name__}: {e}")
        finally:
            session.close()
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish"); p.add_argument("--which", choices=("live", "reference", "both"), default="both")
    p.add_argument("--quiet", action="store_true", help="one line of output (the forced command's answer to the consumer)")
    q = sub.add_parser("pull"); q.add_argument("--url", default=None, help="a feed directory or server instead of the .env source"); q.add_argument("--token", default=None)
    q.add_argument("--which", choices=("live", "reference", "both"), default="both"); q.add_argument("--no-project", action="store_true")
    q.add_argument("--loop", type=int, default=None, help="minutes between pulls; omit for one pull")
    r = sub.add_parser("apply"); r.add_argument("dir")
    a = ap.parse_args()
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
        from ivan.sports.nba import autopilot
        m = feed.publish("nba", which=a.which, log=(lambda s: None) if a.quiet else print, producer=autopilot.producer_info("desk"))
        print(f"live {m.get('live', {}).get('stamp')} reference {m.get('reference', {}).get('stamp')}" + ("" if a.quiet else f" -> {feed.FEED_DIR / 'nba'}"))
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
