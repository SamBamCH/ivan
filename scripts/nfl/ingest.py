#!/usr/bin/env python3
"""
scripts/mlb/ingest.py

1) Discover all upcoming MLB draft groups (slates)
2) Retrieve contest info (gameType, sdstring, contest name)
3) Prompt user to choose one by draftGroupId
4) Fetch that slate’s players and upsert into mlb_player
5) Track relevant teams and fetch odds for them
"""

import sys
import re
from pathlib import Path
import logging
from datetime import datetime
from dateutil import parser
from ivan.database.session import init_db, get_session
from sqlalchemy import delete
from ivan.sports.nfl.models import NFLPlayer, NFLOdds
from ivan.sports.common.draftkings import (
    fetch_all_contests,
    summarize_slates,
    get_players_for_slate,
)
from ivan.sports.nfl.fpts_distribution_calculations import generate_and_fit_all
from ivan.sports.common.odds import get_odds, parse_odds  # ← import your odds function
from ivan.core.utils import normalize_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_TEAM_RE = re.compile(r"\b[A-Z]{2,3}\b")

def _extract_matchup_teams(comp: dict[str, object]) -> tuple[str|None, str|None]:
    """
    Try to extract (away, home) from the competition payload.
    Priority 1: comp['name'] like 'TEN @ DEN'
    Fallback:   scan for two team tokens in comp['name'] or nameDisplay
    """
    away = home = None

    name = (comp or {}).get("name") or ""
    # normalize common separators to '@'
    normalized = (
        name.replace(" vs. ", " @ ")
            .replace(" vs ", " @ ")
            .replace(" VS ", " @ ")
            .replace(" at ", " @ ")
            .replace(" AT ", " @ ")
    )
    if "@" in normalized:
        parts = [p.strip() for p in normalized.split("@")]
        if len(parts) == 2:
            away, home = parts[0], parts[1]

    if not (away and home):
        # Fallback: try regex on name
        tokens = _TEAM_RE.findall(name)
        if len(tokens) >= 2:
            away, home = tokens[0], tokens[-1]
        else:
            # Last fallback: nameDisplay array
            nd = (comp or {}).get("nameDisplay") or []
            vals = []
            for item in nd:
                if isinstance(item, dict):
                    v = (item.get("value") or "").strip()
                else:
                    v = str(item).strip()
                vals.extend(_TEAM_RE.findall(v))
            if len(vals) >= 2:
                away, home = vals[0], vals[-1]

    # sanity: return None if they look invalid
    if away == home:
        return None, None
    return away, home


def _derive_opponent_for_player(team: str|None, comp: dict) -> str|None:
    if not team or not comp:
        return None
    away, home = _extract_matchup_teams(comp)
    if not (away and home):
        return None
    if team == away:
        return home
    if team == home:
        return away
    # If team didn't match either side (rare formatting issues), give up gracefully
    return None


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
    print("Available NFL Slates:")
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

    # ← ADDED: track all teams we pulled
    relevant_teams = set()

    for p in players:
        # de-dupe some roster slots if needed
        if p.get("rosterSlotId") == 70:
            logging.debug(f"Skipping {p.get('displayName')} (rosterSlotId=70)")
            continue

        team = p.get("teamAbbreviation")
        comp = p.get("competition") or {}
        opp  = _derive_opponent_for_player(team, comp)

        if team: relevant_teams.add(team)
        if opp:  relevant_teams.add(opp)

        dbp = NFLPlayer(
            id             = p.get("draftableId"),
            name           = normalize_name(p.get("displayName")),
            team           = team,
            opponent       = opp,  # <-- now populated for all players (including DST)
            position       = p.get("position"),
            roster_slot_id = p.get("rosterSlotId"),
            gametime       = parser.isoparse(comp["startTime"]) if comp.get("startTime") else None,
            salary         = p.get("salary"),
            status         = p.get("status"),
        )
        session.merge(dbp)

    session.commit()
    logging.info("✔ Done ingesting players for selected slate.")
    logging.info(f"Relevant teams to fetch odds for: {sorted(relevant_teams)} ({len(relevant_teams) // 2} games)")

    

    
    

if __name__ == "__main__":
    main()
