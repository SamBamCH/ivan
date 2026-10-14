#!/usr/bin/env python3
"""
scripts/nfl/ingest.py

Ingest the slate: DraftKings pool (-p), player props and game lines from the-odds-api
(--update-odds), fit every player's FPTS distribution, project ownership and snapshot it,
write the projection artifacts. The steps live in ivan.sports.nfl.ingest.
"""
import argparse
import logging

from ivan.database.session import add_missing_columns, get_session, init_db
from ivan.sports.nfl.ingest import IngestError, run_ingest
from ivan.sports.nfl.models import NFLBase


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    p = argparse.ArgumentParser(description="ingest and update player odds")
    p.add_argument("-p", action='store_true', help='Prelock flag, determines if we need to fetch DK payload or not.')
    p.add_argument("--update-odds", type=str2bool, nargs='?', const=True, default=True, help="Update player odds")
    p.add_argument("--draft-group", type=int, default=None, help="DK draft group id of the slate to ingest; skips the interactive slate prompt")
    p.add_argument("--no-ownership", action='store_true', help="Skip ownership projection after fitting distributions")
    p.add_argument("--no-snapshot", action='store_true', help="Skip the projection snapshot used for model training")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    engine = init_db('nfl')
    for added in add_missing_columns(engine, NFLBase.metadata):
        print(f'schema: added column {added}')
    session = get_session(engine)
    try:
        run_ingest(session, prelock=args.p, update_odds=args.update_odds, draft_group_id=args.draft_group,
                   ownership=not args.no_ownership, snapshot=not args.no_snapshot)
    except IngestError as e:
        raise SystemExit(str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
