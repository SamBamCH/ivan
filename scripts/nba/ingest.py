from ivan.database.session import init_db, get_session
from ivan.sports.common.draftkings import(
    fetch_all_contests, 
    summarize_slates, 
    get_players_for_slate
)
from ivan.sports.nba.models import NBAPlayer, NBAOdds, ProjectionSnapshot
from ivan.core.utils import normalize_name
import sys
import logging
from rich.console import Console
from dateutil import parser
import re
from sqlalchemy import delete
from datetime import timezone, datetime
import pytz


logging.basicConfig(level=logging.DEBUG, format="%(message)s %(asctime)s", datefmt="%Y-%m-%d %H:%M:%S")  
console = Console()

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

    # Get current time in Central timezone (naive) to compare against gametime (which is stored naive but represents Central time)
    now = datetime.now(LOCAL_TZ).replace(tzinfo=None)
    
    # Check for locked games (games that have already started or are starting now)
    locked_players = session.query(NBAPlayer).filter(NBAPlayer.gametime <= now).all()
    
    if locked_players:
        # Create ProjectionSnapshot entries for each locked player before deleting
        logging.info(f"Found {len(locked_players)} locked games. Creating snapshots before deletion.")
        # Use current Central Time (naive) to align with downstream queries that assume CT-naive datetimes
        snapshot_date = datetime.now(LOCAL_TZ).replace(tzinfo=None)

        for player in locked_players:
            snapshot = ProjectionSnapshot(
                id=player.id,
                snapshot_date=snapshot_date,
                player_name=player.name,
                team=player.team,
                opponent=player.opponent,
                position=player.position,
                roster_slot_id=player.roster_slot_id,
                gametime=player.gametime,
                salary=player.salary,
                status=player.status,
                fpts_mean=player.fpts_mean,
                fpts_std=player.fpts_std,
                fpts_updated_at=player.fpts_updated_at,
            )
            session.add(snapshot)

        session.commit()
        logging.info(f"Created {len(locked_players)} projection snapshots (snapshot_date in CT).")

        # Now delete the locked games, logging a reason for each removal
        for lp in locked_players:
            console.print(f"[yellow]Excluding locked player[/yellow] {lp.name} (id={lp.id}) - gametime {lp.gametime} <= now {now}")
        session.query(NBAPlayer).filter(NBAPlayer.gametime <= now).delete(synchronize_session=False)
    else:
        # No locked games yet
        logging.info("No locked games to clean up.")
    
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
    kept_by_name: dict[str, dict] = {}

    for p in players:

        team = p.get("teamAbbreviation")
        comp = p.get("competition") or {}
        opp  = _derive_opponent_for_player(team, comp)
        player_name = normalize_name(p.get("displayName"))

        # Skip duplicate player names, but log which record was kept vs skipped
        if player_name in seen_names:
            kept = kept_by_name.get(player_name, {})
            kept_id = kept.get("draftableId")
            kept_team = kept.get("teamAbbreviation")
            kept_pos = kept.get("position")
            skipped_id = p.get("draftableId")
            skipped_pos = p.get("position")
            console.print(
                f"[yellow]Excluding duplicate player[/yellow] {player_name} - already processed. "
                f"kept id={kept_id} team={kept_team} pos={kept_pos} | skipped id={skipped_id} pos={skipped_pos}"
            )
            continue

        seen_names.add(player_name)
        kept_by_name[player_name] = p

        if team: relevant_teams.add(team)
        if opp:  relevant_teams.add(team)

        # Parse gametime and convert to local timezone
        gametime = _parse_gametime_to_local(comp.get("startTime"))

        # Validate essential fields; log and skip if missing
        draftable_id = p.get("draftableId")
        position     = p.get("position")
        salary       = p.get("salary")

        if draftable_id is None:
            console.print(f"[yellow]Excluding player[/yellow] {player_name} - missing draftableId")
            continue
        if position is None:
            console.print(f"[yellow]Excluding player[/yellow] {player_name} (id={draftable_id}) - missing position")
            continue
        if salary is None:
            console.print(f"[yellow]Excluding player[/yellow] {player_name} (id={draftable_id}) - missing salary")
            continue
        if gametime is None:
            console.print(f"[yellow]Player {player_name} (id={draftable_id}) has missing/invalid gametime; retaining but review data source.")

        dbp = NBAPlayer(
            id              = draftable_id,
            name            = player_name,
            team            = team,
            opponent        = opp,
            position        = position,
            roster_slot_id  = p.get("rosterSlotId"),
            gametime        = gametime,
            salary          = salary,
            status          = p.get("status"),
        )
        session.merge(dbp)

    session.commit()

    # Debug: verify presence of specific problematic names after ingest
    try:
        targets = ["craig porter", "craig porter jr"]
        for t in targets:
            rows = (
                session.query(NBAPlayer)
                .filter(NBAPlayer.name.like(f"{t}%"))
                .all()
            )
            if rows:
                for r in rows:
                    console.print(
                        f"[green]Present after ingest[/green]: name={r.name} id={r.id} team={r.team} pos={r.position} salary={r.salary}"
                    )
            else:
                console.print(f"[red]Missing after ingest[/red]: pattern '{t}%'")
    except Exception as _e:
        # Non-fatal
        pass



main()