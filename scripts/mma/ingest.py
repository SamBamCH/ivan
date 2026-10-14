#!/usr/bin/env python3
# scripts/mma/ingest.py
"""
Build today's MMA slate with no arguments:

    scripts/mma/ingest.py

picks the Classic MMA slate on DraftKings, pulls its fighter pool, fills Win
odds from the-odds-api (ODDS_API_KEY in .env), fetches the slate's contests
and payout tables, and writes the slate. On a new card it also writes
data/mma/input/fight_odds.csv and player_odds.csv with the names and Win
odds in; fill in the finish markets and volume lines from the sportsbook
and run it again. Those two files are read as overrides whenever they
describe the current card.

Options: --draft-group <id> for a specific slate, --players-csv <DK export>
to skip the DK API, --no-odds-api, --no-contests, --main-events N.
"""
import argparse
import logging
import os

from ivan.core.config import config_path, load_config
from ivan.database.session import add_missing_columns, get_session, init_db
from ivan.sports.mma.ingest import format_summary, ingest_slate
from ivan.sports.mma.models import MMABase


def main():
    p = argparse.ArgumentParser(description="Ingest an MMA slate: DK pool, moneylines, contests, hand-entered markets")
    p.add_argument("--config", default=str(config_path("mma")))
    p.add_argument("--draft-group", type=int, default=None, help="DK draft group id (default: the Classic MMA slate with the most contests)")
    p.add_argument("--players-csv", default=None, help="DraftKings player pool export; skips the DK API for the pool")
    p.add_argument("--fight-odds", default=None, help="fight-level odds CSV (default: paths.fight_odds_csv)")
    p.add_argument("--player-odds", default=None, help="fighter-level odds CSV (default: paths.player_odds_csv)")
    p.add_argument("--main-events", type=int, default=None, help="how many of the latest fights are 5 rounds (default: slate.num_5_round_fights)")
    p.add_argument("--no-odds-api", action="store_true", help="do not fetch moneylines from the-odds-api")
    p.add_argument("--no-contests", action="store_true", help="do not fetch the slate's contests and payout tables")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("urllib3", "requests", "ivan.sports.common.draftkings"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger().handlers[0].addFilter(lambda r: not r.getMessage().startswith("[draftkings]"))
    cfg = load_config("mma", args.config)
    engine = init_db("mma", cfg.get("database_url"))
    MMABase.metadata.create_all(engine)
    for added in add_missing_columns(engine, MMABase.metadata):
        print(f"schema: added column {added}")
    session = get_session(engine)

    key = None if args.no_odds_api else os.environ.get("ODDS_API_KEY")
    if key is None and not args.no_odds_api:
        print("ODDS_API_KEY is not set: Win odds will come from player_odds.csv only")
    summary = ingest_slate(session, cfg, players_csv=args.players_csv, draft_group_id=args.draft_group,
                           fight_odds_csv=args.fight_odds, player_odds_csv=args.player_odds, num_5_round_fights=args.main_events,
                           odds_api_key=key, fetch_contests=not args.no_contests and args.players_csv is None)
    print()
    print(format_summary(summary))


if __name__ == "__main__":
    main()
