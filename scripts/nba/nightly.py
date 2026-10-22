#!/usr/bin/env python3
"""
Nightly NBA data job: box scores for the current season (nba_api), capped per run
because of nba_api rate limits; the timer runs it three times a day so it catches up.
Then the projection scorecard for every slate whose box scores have arrived
(ivan.sports.nba.validation.run_pending_scorecards) -> data/nba/diagnostics/scorecard/.
"""
import argparse
import logging
import sys

from ivan.database.session import get_session, init_db
from ivan.sports.nba.historical.data_fetch import current_season, fetch_boxscores
from ivan.sports.nba.ingest import ensure_schema
from ivan.sports.nba.validation import run_pending_scorecards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=None, help="e.g. 2026-27 (default: current)")
    ap.add_argument("--max-games", type=int, default=60)
    ap.add_argument("--no-scorecard", action="store_true", help="skip the projection scorecards")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    season = args.season or current_season()
    logging.info(f"box scores for {season}")
    fetch_boxscores(session, season=season, max_games=args.max_games)
    if not args.no_scorecard:
        try:
            cards = run_pending_scorecards(session, log=logging.info)
            logging.info(f"scorecards written: {len(cards)}")
        except Exception as e:   # noqa: BLE001 - the box scores are the job; the scorecard must not fail it
            logging.warning(f"scorecard failed: {type(e).__name__}: {e}")
        try:
            from ivan.sports.nba.standings import derive_pending
            derived = derive_pending(session, log=logging.info)
            logging.info(f"standings derived: {len(derived)}")
        except Exception as e:   # noqa: BLE001
            logging.warning(f"standings derive failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
