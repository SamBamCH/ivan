"""
The feed's command line, one per sport (scripts/<sport>/feed.py calls main(sport, after_pull=...)).

    feed.py publish [--which live|reference|both] [--slate DG] [--quiet]   write data/feed/<sport>/* (the producer; the consumer's pull asks for it)
    feed.py pull [--no-project] [--slate DG] [--url ...] [--loop MIN]       the consumer: ssh publish + rsync + apply (IVAN_FEED_SSH), or from --url / IVAN_FEED_URL
    feed.py apply DIR                                                      seed this database from bundles copied by hand

`before_pull()` returns what the sport wants to remember from before the pull (the NBA: the slate it held, so a slate move
files that slate's edits); `after_pull(changed, before)` is its own step after a pull that brought new tables (the NBA
re-projects the teams this machine edited). `before_publish(args)` runs on the producer before the bundle is written
(the NBA: `--slate DG` fetches that slate when it is not held and publishes its bundle; the consumer's `pull --slate DG`
asks for it over ssh, `publish <sport> DG`). All None for a sport that only reads what it pulled.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from ivan.core import feed
from ivan.core.utils import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")        # IVAN_FEED_SSH (or IVAN_FEED_URL / IVAN_FEED_TOKEN)


def do_pull(sport: str, a, before_pull: Optional[Callable[[], object]], after_pull: Optional[Callable[[dict, object], None]]) -> dict:
    src = feed.producer_source()
    if a.url:
        src = ("url", a.url)
    if src is None:
        raise SystemExit(f"set {feed.SSH_ENV}=<user>@<host> (or {feed.URL_ENV}) in .env, or give --url")
    before = before_pull() if before_pull is not None else None
    state_dir = PROJECT_ROOT / "data" / "feed-state"
    slate = getattr(a, "slate", None)
    if src[0] == "ssh":
        changed = feed.pull_ssh(src[1], sport, state_dir, log=print, slate=slate)
    elif slate is not None:
        raise SystemExit(f"--slate needs an ssh producer ({feed.SSH_ENV}): a feed directory or server serves what the laptop holds")
    else:
        changed = feed.pull(src[1], a.token or os.environ.get("IVAN_FEED_TOKEN"), sport, state_dir, which=a.which, log=print)
    tables = {k: v for k, v in changed.items() if k != "published"}
    if not tables:
        print("nothing new")
    elif after_pull is not None and not getattr(a, "no_project", False):
        after_pull(changed, before)
    return changed


def main(sport: str, before_pull: Optional[Callable[[], object]] = None, after_pull: Optional[Callable[[dict, object], None]] = None, doc: str = "",
         before_publish: Optional[Callable[[argparse.Namespace], None]] = None) -> int:
    ap = argparse.ArgumentParser(description=doc or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish"); p.add_argument("--which", choices=("live", "reference", "both"), default="both")
    p.add_argument("--quiet", action="store_true", help="the stamps on the last line only (the forced command's answer to the consumer)")
    p.add_argument("--slate", type=int, default=None, help="publish this slate's bundle only (fetched first when not held: the sport's before_publish)")
    q = sub.add_parser("pull"); q.add_argument("--url", default=None, help="a feed directory or server instead of the .env source"); q.add_argument("--token", default=None)
    q.add_argument("--which", choices=("live", "reference", "both"), default="both")
    q.add_argument("--slate", type=int, default=None, help="ask the producer to load this draft group first (ssh only): both desks move to it")
    q.add_argument("--no-project", action="store_true", help="apply the bundles only; skip the sport's own step after a pull")
    q.add_argument("--loop", type=int, default=None, help="minutes between pulls; omit for one pull")
    r = sub.add_parser("apply"); r.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "apply":
        import shutil
        d = Path(a.dir)
        for part in feed.parts_of(sport):
            if (d / f"{part}.sql.gz").exists():
                feed.apply(sport, d / f"{part}.sql.gz", log=print)
        if (d / "files").is_dir():
            shutil.copytree(d / "files", PROJECT_ROOT, dirs_exist_ok=True); print("reference files copied")
        return 0
    if a.cmd == "publish":
        if a.slate is not None:
            if before_publish is None:
                raise SystemExit(f"--slate: the {sport} feed has no slate to load (only the NBA holds one slate at a time)")
            before_publish(a)
        producer = None
        if sport == "nba":
            from ivan.sports.nba.slate import autopilot
            producer = autopilot.producer_info("desk")
        m = feed.publish(sport, which=a.which, log=(lambda s: None) if a.quiet else print, producer=producer, slate=a.slate)
        stamps = " ".join(f"{part} {m.get(part, {}).get('stamp')}" for part in feed.parts_of(sport))
        print(stamps + ("" if a.quiet else f" -> {feed.FEED_DIR / sport}"))
        return 0
    if a.loop:
        while True:
            try:
                do_pull(sport, a, before_pull, after_pull)
            except Exception as e:   # noqa: BLE001
                print(f"pull failed: {type(e).__name__}: {e}")
            time.sleep(a.loop * 60)
    do_pull(sport, a, before_pull, after_pull)
    return 0
