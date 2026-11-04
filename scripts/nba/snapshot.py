"""
Snapshot: record the NBA projection as it stands (what the desk's Generate, Sim and Late swap do first; no button of its own).

The pool (nba_player) is the projection: every edit on a team page, every refresh and project.py write it, and the slate's
projection store (slates/<dg>/projection/<TEAM>.pkl) holds each team's frame. This records that state without projecting
anything: an nba_projection_capture row per player (source "ours", the rows the scorecard scores) and the projections CSV
under the slate's output/projections/. From a terminal, for a record between runs (a belief about to change, say).

  venv/bin/python scripts/nba/snapshot.py             # capture rows + CSV
  venv/bin/python scripts/nba/snapshot.py --no-csv    # capture rows only
"""
from __future__ import annotations

import argparse

from ivan.sports.nba.ingest import IngestError, capture_pool
from ivan.sports.nba import slates as S


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("--no-csv", action="store_true", help="capture rows only, no projections CSV")
    args = ap.parse_args()
    dg, session = S.open(args.slate)
    try:
        info = capture_pool(session, log=print, write_csv=not args.no_csv)
    except IngestError as e:
        raise SystemExit(f"snapshot failed: {e}")
    print(f"snapshot: {info.players} players, {info.captures} capture rows" + (f", {info.projections_path}" if info.projections_path else ""))


if __name__ == "__main__":
    main()
