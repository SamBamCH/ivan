#!/usr/bin/env python3
"""
scripts/mlb/ingest.py

1) Discover all upcoming MLB draft groups (slates)
2) Retrieve contest info (gameType, sdstring, contest name)
3) Prompt user to choose one by draftGroupId
4) Fetch that slate’s players and upsert into mlb_player
"""

import sys
import logging
from datetime import datetime
from dateutil import parser
from ivan.database.session import init_db, get_session
from sqlalchemy import delete
from ivan.sports.nfl.models import NFLPlayer
from ivan.sports.common.draftkings import (
    fetch_all_contests,
    summarize_slates,
    get_players_for_slate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    # initialize database
    engine = init_db("nfl", None)
    session = get_session(engine)

    session.execute(delete(NFLPlayer))

    # 1) fetch contests
    contests = fetch_all_contests("nfl")
    if not contests:
        logging.error("No contests returned from lobby endpoint")
        sys.exit(1)

    # 2) summarize slates
    slate_info = summarize_slates(contests)

    # 3) print menu for user
    print("Available MLB Slates:")
    for i, (dg, info) in enumerate(slate_info.items(), start=1):
        print(f"{i:2}. DG={dg} | {info['gameType']:8} | entries={info['count']:2} | at {info['sdstring']} | '{info['name']}'")
    print()

    # 4) prompt user
    choice = input(f"Select a slate 1–{len(slate_info)} (or Q to quit): ").strip().upper()
    if choice == "Q":
        sys.exit(0)
    try:
        idx = int(choice) - 1
        dg = list(slate_info.keys())[idx]
    except Exception:
        logging.error("Invalid selection, exiting.")
        sys.exit(1)

    # 5) fetch & upsert players
    players = get_players_for_slate(dg)
    logging.info(f"Pulled {len(players)} players for DG={dg}")

    for p in players:

        if p.get("rosterSlotId") == 70: #this is to remove duplicates. Not sure why, but DK has duplicate players keyed under unique ids. I think the second one is FLEX
            logging.debug(f"Skipping {p.get('displayName')} (rosterSlotId=70)")
            continue
        dbp = NFLPlayer(
            id             = p.get("draftableId"),
            name           = p.get("displayName"),
            team           = p.get("teamAbbreviation"),
            opponent       = None, #TODO: extract this, need a function probably
            position       = p.get("position"),
            roster_slot_id = p.get("rosterSlotId"),
            gametime       = parser.isoparse(p["competition"]["startTime"]),
            salary         = p.get("salary"),
            status         = p.get("status")
        )
        session.merge(dbp)

    session.commit()
    logging.info("✔ Done ingesting players for selected slate.")


if __name__ == "__main__":
    main()
