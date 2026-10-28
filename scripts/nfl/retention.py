#!/usr/bin/env python3
"""
scripts/nfl/retention.py

Let go of what a replay can rebuild, keep everything a backtest needs
(ivan.sports.nfl.retention has the rules): worker clones of the replay
database a day after their last write, and a run's heavy files (per-repeat
outputs, validation csvs, its database copy, logs) 30 days after its last
write. Manifests, reports, projections and run.json stay, so the per-slate
run index keeps every build. The nightly job runs this; by hand:

    retention.py --dry-run            # what would go, per run
    retention.py                      # do it
    retention.py --days 60            # a longer window for the heavy files
    retention.py --pin NAME           # keep a run's heavy files (a hub run name, models/<component>/<stamp>, a replay root or a path)
    retention.py --unpin NAME
"""
from __future__ import annotations

import argparse
import sys

from ivan.sports.nfl import retention


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list what would be removed; remove nothing")
    ap.add_argument("--days", type=int, default=retention.HEAVY_DAYS, help=f"heavy files of a run go this many days after its last write (default {retention.HEAVY_DAYS})")
    ap.add_argument("--pin", default=None, help="mark a run pinned: its heavy files are kept")
    ap.add_argument("--unpin", default=None, help="remove the pin")
    a = ap.parse_args(argv)
    if a.pin or a.unpin:
        root = retention.find_run(a.pin or a.unpin)
        if root is None:
            print(f"no run named {a.pin or a.unpin!r}", file=sys.stderr)
            return 2
        retention.set_pinned(root, bool(a.pin))
        print(f"{root}: {'pinned' if a.pin else 'unpinned'}")
        return 0
    s = retention.sweep(dry_run=a.dry_run, heavy_days=a.days)
    if not a.dry_run:
        print(f"freed {s.freed / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
