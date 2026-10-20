#!/usr/bin/env python3
"""
Ingest an NBA slate: DraftKings pool, Rotowire starters, player props (--update-odds),
then project it and record a timestamped projection capture. Steps live in
ivan.sports.nba.ingest.

  python scripts/nba/ingest.py --list                       # today's classic draft groups
  python scripts/nba/ingest.py --draft-group 154575 --update-odds
  python scripts/nba/ingest.py --draft-group 154575 --no-odds       # DK + Rotowire only (free)
  python scripts/nba/ingest.py --project-only                       # re-project the stored pool
  python scripts/nba/ingest.py --stok-csv data/nba/input/stok.csv   # record Stokastic as a yardstick
"""
import argparse
import logging

from ivan.database.session import get_session, init_db
from ivan.sports.nba.ingest import IngestError, classic_slates, ensure_schema, project_slate, record_external_projections, run_ingest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list today's classic draft groups and exit")
    ap.add_argument("--draft-group", type=int, default=None, help="DK draft group id to ingest")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--update-odds", action="store_true", help="fetch player props (costs odds-api credits)")
    g.add_argument("--no-odds", action="store_true", help="skip props (default)")
    ap.add_argument("--no-starters", action="store_true", help="skip the Rotowire fetch")
    ap.add_argument("--no-project", action="store_true", help="store only; do not project")
    ap.add_argument("--project-only", action="store_true", help="re-project the stored pool, no fetch")
    ap.add_argument("--stok-csv", default=None, help="record a Stokastic CSV as yardstick capture rows")
    ap.add_argument("--sims", type=int, default=4000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    engine = init_db("nba")
    ensure_schema(engine)
    session = get_session(engine)
    try:
        if args.list:
            for s in classic_slates():
                print(f"{s['id']}  {s['start']}  {s['name']}  ({s['contests']} contests)")
            return 0
        if args.stok_csv:
            record_external_projections(session, args.stok_csv)
            return 0
        if args.project_only:
            project_slate(session, n_sims=args.sims)
            return 0
        if args.draft_group is None:
            ap.error("--draft-group is required (see --list)")
        run_ingest(session, args.draft_group, update_odds=args.update_odds, project=not args.no_project,
                   starters=not args.no_starters, n_sims=args.sims)
        return 0
    except IngestError as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
