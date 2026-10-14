#!/usr/bin/env python3
"""
Projected vs actual for every component of the NFL pipeline (see
ivan.core.validation and ivan.sports.nfl.validate).

    scripts/nfl/validate.py --since 2026-09-01 --until 2026-10-12            # live runs (data/nfl/output, output/web)
    scripts/nfl/validate.py --replay data/nfl/output/replay_repeats [--repeat 0]   # a replayed window
    scripts/nfl/validate.py --dirs path1 path2 ...                           # any directories with run artifacts

Reads the run artifacts (projections.*.csv, sim/<stamp>/contest_*.npz,
chosen_entries.lineups.*.csv) and the actuals in the database, writes
players.csv, field.csv, lineups.csv, pool.csv, pool_deciles.csv and
summary.json under --out (default data/nfl/output/validation/<stamp>/ or
<replay root>/nfl/validation/), and prints the check list.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from ivan.core.config import load_config
from ivan.core.replay import repeat_dirs, replayed_dirs
from ivan.database.session import get_session, init_db
from ivan.sports.nfl.backtest_tax import local_to_utc
from ivan.sports.nfl.paths import OUTPUT_DIR
from ivan.sports.nfl.validate import validate_window


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=None, help="first slate lock (local date)")
    ap.add_argument("--until", default=None, help="last slate lock (local date, inclusive)")
    ap.add_argument("--replay", default=None, help="a replay output root (its slates' repeat directories are the run directories)")
    ap.add_argument("--repeat", type=int, default=0, help="which repeat of a replay to validate (default 0)")
    ap.add_argument("--dirs", nargs="*", default=None, help="run directories (default data/nfl/output and output/web)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--in-field", default=None, choices=["auto", "yes", "no"],
                    help="whether scraped fields already contain our entries (default: no for replays, auto for live)")
    ap.add_argument("--no-snapshots", action="store_true", help="do not add projection_snapshot rows for slates without artifacts")
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    since = local_to_utc(datetime.fromisoformat(a.since)) if a.since else None
    until = local_to_utc(datetime.fromisoformat(a.until) + timedelta(days=1) - timedelta(seconds=1)) if a.until else None
    if a.replay:
        root = Path(a.replay)
        slates = replayed_dirs(root, "nfl")
        reps = repeat_dirs(slates).get(a.repeat, [])
        dirs = list(reps)
        in_field = a.in_field or "no"
        out = Path(a.out) if a.out else root / "nfl" / "validation" / f"rep_{a.repeat:03d}"
        title = a.title or f"replay {root.name} repeat {a.repeat}"
        include_snapshots = False if a.no_snapshots else bool(a.since and a.until)
    else:
        dirs = [Path(d) for d in a.dirs] if a.dirs else [OUTPUT_DIR, OUTPUT_DIR / "web"]
        in_field = a.in_field or "auto"
        out = Path(a.out) if a.out else OUTPUT_DIR / "validation" / datetime.now().strftime("%Y%m%d-%H%M%S")
        title = a.title or f"live {a.since or ''}..{a.until or ''}"
        include_snapshots = not a.no_snapshots
    cfg = load_config("nfl")
    engine = init_db("nfl", a.db_url or cfg.get("database_url"))
    session = get_session(engine)
    print(f"{len(dirs)} run directories; writing {out}")
    validate_window(session, dirs, out, since=since, until=until, in_field=in_field, include_snapshots=include_snapshots, title=title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
