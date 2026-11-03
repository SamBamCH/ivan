"""
Snapshot: record the NBA projection as it stands (the desk's Snapshot button, what Publish became).

The pool (nba_player) is the projection: every edit on a team page, every refresh and project.py write it, and the
projection store (data/nba/derived/projection/<TEAM>.pkl) holds each team's frame. This records that state without
projecting anything: an nba_projection_capture row per player (source "ours", the rows the scorecard scores) and the
projections CSV under data/nba/output/projections/. Generate, Sim and the late swap take one first from the desk.

  venv/bin/python scripts/nba/snapshot.py             # capture rows + CSV
  venv/bin/python scripts/nba/snapshot.py --no-csv    # capture rows only
"""
from __future__ import annotations

import argparse

from ivan.database.session import get_session, init_db
from ivan.sports.nba.ingest import IngestError, capture_pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-csv", action="store_true", help="capture rows only, no projections CSV")
    args = ap.parse_args()
    session = get_session(init_db("nba"))
    try:
        info = capture_pool(session, log=print, write_csv=not args.no_csv)
    except IngestError as e:
        raise SystemExit(f"snapshot failed: {e}")
    print(f"snapshot: {info.players} players, {info.captures} capture rows" + (f", {info.projections_path}" if info.projections_path else ""))


if __name__ == "__main__":
    main()
