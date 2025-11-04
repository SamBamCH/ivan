from ivan.database.session import init_db, get_session
from ivan.sports.common.draftkings import(
    fetch_all_contests, 
    summarize_slates, 
    get_players_for_slate
)
from ivan.sports.nba.models import NBAPlayer, NBAOdds
from ivan.core.utils import normalize_name
import sys
import logging
from dateutil import parser
import re
from sqlalchemy import delete
from datetime import timezone, datetime
import pytz


logging.basicConfig(level=logging.DEBUG, format="%(message)s %(asctime)s", datefmt="%Y-%m-%d %H:%M:%S")   

_TEAM_RE = re.compile(r"\b[A-Z]{2,3}\b")

# Define your local timezone (Central Time)
LOCAL_TZ = pytz.timezone('America/Chicago')

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

def _parse_gametime_to_local(start_time_str: str|None) -> object|None:
    """
    Parse ISO format datetime string and convert to local timezone.
    Assumes the input is in UTC if no timezone info is present.
    """
    if not start_time_str:
        return None
    
    try:
        # Parse the datetime string
        dt = parser.isoparse(start_time_str)
        
        # If no timezone info, assume UTC
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        
        # Convert to local timezone
        local_dt = dt.astimezone(LOCAL_TZ)
        
        return local_dt
    except Exception as e:
        logging.warning(f"Failed to parse datetime '{start_time_str}': {e}")
        return None

def main():
    engine = init_db("nba", None)
    session = get_session(engine)

    # Get current time in UTC (database likely stores UTC)
    now = datetime.now(timezone.utc)
    
    # Check for unlocked games (games that haven't started yet)
    unlocked_games = session.query(NBAPlayer).filter(NBAPlayer.gametime > now).all()
    
    if unlocked_games:
        # Only delete unlocked games
        logging.info(f"Found {len(unlocked_games)} unlocked games. Deleting only unlocked games.")
        session.query(NBAPlayer).filter(NBAPlayer.gametime > now).delete(synchronize_session=False)
    else:
        # All games are locked, clear everything
        logging.info("All games are locked. Clearing all NBAPlayer records.")
        session.execute(delete(NBAPlayer))
    
    session.commit()
    # Clear the session to remove references to deleted objects
    session.expunge_all()

    contests = fetch_all_contests("nba")
    slate_info = summarize_slates(contests)

    print("             Available NBA Slates:")
    for i, (dg, info) in enumerate(slate_info.items(), start=1):
        print(f"                {i:2}. DG={dg} | {info['gameType']:8} | entries={info['count']:2} | at {info['sdstring']} | '{info['name']}'")
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

    relevant_teams = set()
    seen_names = set()

    for p in players:

        team = p.get("teamAbbreviation")
        comp = p.get("competition") or {}
        opp  = _derive_opponent_for_player(team, comp)
        player_name = normalize_name(p.get("displayName"))

        # Skip duplicate player names
        if player_name in seen_names:
            logging.debug(f"Skipping duplicate player: {player_name}")
            continue

        seen_names.add(player_name)

        if team: relevant_teams.add(team)
        if opp:  relevant_teams.add(team)

        # Parse gametime and convert to local timezone
        gametime = _parse_gametime_to_local(comp.get("startTime"))

        dbp = NBAPlayer(
            id              = p.get("draftableId"),
            name            = player_name,
            team            = team,
            opponent        = opp,
            position        = p.get("position"),
            roster_slot_id  = p.get("rosterSlotId"),
            gametime        = gametime,
            salary          = p.get("salary"),
            status          = p.get("status"),
        )
        session.merge(dbp)

    session.commit()



main()