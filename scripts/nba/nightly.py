#!/usr/bin/env python3
"""
Nightly NBA data job: box scores for the current season (nba_api), capped per run
because of nba_api rate limits; the timer runs it three times a day so it catches up.
Replaces the box-score half of overnight.py (which runs on import; do not use it).
"""
import argparse
import logging
import sys

from ivan.database.session import get_session, init_db
from ivan.sports.nba.historical.data_fetch import current_season, fetch_boxscores
from ivan.sports.nba.ingest import ensure_schema


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=None, help="e.g. 2026-27 (default: current)")
    ap.add_argument("--max-games", type=int, default=60)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    season = args.season or current_season()
    logging.info(f"box scores for {season}")
    fetch_boxscores(session, season=season, max_games=args.max_games)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
