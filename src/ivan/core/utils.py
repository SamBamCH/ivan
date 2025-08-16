from datetime import datetime, timedelta
import pytz
import math
from collections import defaultdict
from sqlalchemy.orm import Session
from pathlib import Path
import re
import unicodedata
import pandas as pd
import matplotlib as plt

from ivan.sports.mlb.models import MLBPlayer
from sqlalchemy.exc import NoResultFound
from ivan.sports.mlb.models import HistoricalContest

def parse_game_time(game_info, timezone="US/Eastern", lock_offset_hours=0):
    """
    Parse game time from the Game Info string and adjust it to include a lock offset.
    
    :param game_info: String containing game info (e.g., "GB@PHI 09/06/2024 08:15PM ET").
    :param timezone: Timezone of the game times (default: US/Eastern).
    :param lock_offset_hours: Number of hours to subtract for lock time.
    :return: Timezone-aware datetime object adjusted for lock time, or None if unparseable.
    """
    # Split into parts and ensure we have at least date, time, and zone
    parts = game_info.split()
    if len(parts) < 3:
        # e.g., "Postponed" or other non-standard entries
        return None
    date_part, time_part, _ = parts[-3:]
    try:
        eastern = pytz.timezone(timezone)
        gametime = datetime.strptime(f"{date_part} {time_part}", "%m/%d/%Y %I:%M%p")
        gametime = eastern.localize(gametime)
        lock_time = gametime - timedelta(hours=lock_offset_hours)
        return lock_time
    except Exception as e:
        # if parsing fails for other reasons, surface an error
        raise ValueError(f"Error parsing game info '{game_info}': {e}")
    
def get_config_value(config, key_path, default=None):
    """
    Retrieve a nested configuration value from a dictionary using a dot-delimited key path.
    
    Args:
        config (dict): The configuration dictionary.
        key_path (str): Dot-delimited path to the desired key (e.g., "paths.db").
        default: The value to return if the key isn't found.
        
    Returns:
        The value at the specified key path or the default if not found.
    """
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            print(f"config value not found for {key_path}")
            return default
    return value


def project_root() -> Path:
    """
    Returns the root of the repository, i.e. the folder that contains
    src/, scripts/, config/, data/, pyproject.toml, etc.
    """
    # utils.py is at <root>/src/ivan/core/utils.py
    # so go up 3 levels to hit the project root
    return Path(__file__).resolve().parents[3]

_NAME_OVERRIDES = {
    'benjamin williamson': 'ben williamson',
    'timothy elko': 'tim elko',
    'nicholas kurtz': 'nick kurtz',
    'mike soroka': 'michael soroka',
    'zachary dezenzo': 'zach dezenzo',
    'leonardo rivas': 'leo rivas',
    'lance mccullers jr': 'lance mccullers',
    'jeffrey caglianone': 'jac caglianone',
    'joey cantillo': 'joseph cantillo',
    'alexander freeland': 'alex freeland',
    'andrew gilbert': 'drew gilbert',
    'collin kayfus': 'cj kayfus'
    # add more mappings as needed...
}

_TEAM_RENAME_DICT = {
            'Miami Dolphins': 'MIA',
            'San Francisco 49ers': 'SF',
            'Buffalo Bills': 'BUF',
            'Chicago Bears': 'CHI',
            'Houston Texans': 'HOU',
            'Baltimore Ravens': 'BAL',
            'Las Vegas Raiders': 'LV',
            'Carolina Panthers': 'CAR',
            'Los Angeles Chargers': 'LAC',
            'Jacksonville Jaguars': 'JAX',
            'Cleveland Browns': 'CLE',
            'Dallas Cowboys': 'DAL',
            'New Orleans Saints': 'NO',
            'Detroit Lions': 'DET',
            'Tampa Bay Buccaneers': 'TB',
            'Green Bay Packers': 'GB',
            'Indianapolis Colts': 'IND',
            'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE',
            'Seattle Seahawks': 'SEA',
            'Washington Commanders': 'WAS',
            'New York Giants': 'NYG',
            'Tennessee Titans': 'TEN',
            'New York Jets': 'NYJ',
            'Arizona Cardinals': 'ARI',
            'Los Angeles Rams': 'LAR',
            'Kansas City Chiefs': 'KC',
            'Cincinnati Bengals': 'CIN',
            'Denver Broncos': 'DEN',
            'Pittsburgh Steelers': 'PIT',
            'Atlanta Falcons': 'ATL',
            'Philadelphia Eagles': 'PHI'
        }

def normalize_name(name: str) -> str:
    """
    Normalize a player’s name (or any source name) by:
      1. Stripping leading/trailing whitespace
      2. Removing accents/diacritics
      3. Dropping punctuation
      4. Collapsing multiple spaces
      5. Lower‑casing everything
      6. Applying any special overrides (e.g. 'benjamin williamson' -> 'ben williamson')
    """
    # 1. strip
    name = name.strip()

    # 2. remove accents
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()

    # 3. drop anything that isn’t a letter, digit, or whitespace
    name = re.sub(r'[^\w\s]', '', name)

    # 4. collapse runs of whitespace into a single space
    name = re.sub(r'\s+', ' ', name)

    # 5. lower‑case
    name = name.lower()

    # 6. override special cases
    return _NAME_OVERRIDES.get(name, name)

import importlib
from collections import defaultdict
from sqlalchemy.orm import Session

def show_player_exposures(
    session: Session,
    sport: str,
    lineup_id_lists: list[list[int]]
) -> None:
    """
    Prints each player’s exposure vs. target ownership:
      Player     Team     Owner%    Expo%

    - session: SQLAlchemy session
    - sport: one of your sport folder names, e.g. "mlb", "lol"
    - lineup_id_lists: list of lineups, each a list of player‐IDs (ints)
    """
    total = len(lineup_id_lists)
    if total == 0:
        print("No lineups to show.")
        return

    # 1) count exposures per player ID
    exposure_count: dict[int,int] = defaultdict(int)
    for lineup in lineup_id_lists:
        for pid in lineup:
            exposure_count[pid] += 1

    # 2) dynamically import the correct Player class
    module      = importlib.import_module(f"ivan.sports.{sport}.models")
    cls_name    = sport.upper() + "Player"
    PlayerClass = getattr(module, cls_name)

    # 3) bulk-fetch all those players
    players = (
        session
        .query(PlayerClass)
        .filter(PlayerClass.id.in_(exposure_count.keys()))
        .all()
    )
    player_map = {p.id: p for p in players}

    # 4) print table
    header = f"{'Player':<25} {'Team':<10} {'Owner%':>7} {'Expo%':>7}"
    print("\n" + header)
    print("-" * len(header))

    for pid, count in sorted(
        exposure_count.items(),
        key=lambda kv: kv[1],
        reverse=True
    ):
        p = player_map.get(pid)
        if p:
            name      = p.name
            team      = getattr(p, "team", "")
            owner_pct = getattr(p, "ownership", 0.0)
        else:
            name, team, owner_pct = f"<unknown {pid}>", "", 0.0

        expo_pct = count / total * 100
        print(f"{name:<25} {team:<10} {owner_pct:7.1f} {expo_pct:7.1f}")


PITCHER_POS = {"SP", "RP", "P"}

def show_team_exposures(
    session: Session,
    lineup_id_lists: list[list[int]],
    stack_size: int = 3
) -> None:
    """
    Prints each team's stack‐exposure:
      Team       Count   Expo%
    Only counts a lineup if it contains at least `stack_size`
    non-pitchers from that team.
    """
    total = len(lineup_id_lists)
    if total == 0:
        print("No lineups to show.")
        return

    # 1) bulk fetch all players we'll need
    all_pids = {pid for lineup in lineup_id_lists for pid in lineup}
    players = (
        session
        .query(MLBPlayer)
        .filter(MLBPlayer.id.in_(all_pids))
        .all()
    )
    player_map = {p.id: p for p in players}

    # 2) for each lineup, count non-pitchers per team,
    #    then only increment that team if count >= stack_size
    team_count = defaultdict(int)
    for lineup in lineup_id_lists:
        per_team = defaultdict(int)
        for pid in lineup:
            p = player_map.get(pid)
            if p and p.position not in PITCHER_POS:
                per_team[p.team] += 1
        for team, cnt in per_team.items():
            if cnt >= stack_size:
                team_count[team] += 1

    # 3) print results
    header = f"{'Team':<10} {'Count':>6} {'Expo%':>7}"
    print(f"\n stack counts for each team counting only {stack_size} and larger: ")
    print("\n" + header)
    print("-" * len(header))
    for team, cnt in sorted(team_count.items(), key=lambda kv: kv[1], reverse=True):
        expo_pct = cnt / total * 100
        print(f"{team:<10} {cnt:6d} {expo_pct:7.1f}")

PROJECT_ROOT: Path = project_root()

def match_contest_id(session, orig_id: int, name: str) -> int:
    """
    1) Try to find a HistoricalContest row for `orig_id`.
    2) If not found, try to find the most recent row where name == `name`,
       ordering by HistoricalContest.date DESC.
    3) If still not found, raise ValueError.

    Returns the contest_id that should be used for lookup (e.g. payouts).
    """
    # 1) force a plain Python int
    orig_cid = int(orig_id)

    # 2) direct lookup by ID
    row = (
        session
        .query(HistoricalContest)
        .filter(HistoricalContest.contest_id == orig_cid)
        .one_or_none()
    )
    if row is not None:
        return orig_cid

    # 3) fallback by name only → pick the “most recent” by date descending
    fallback = (
        session
        .query(HistoricalContest)
        .filter(HistoricalContest.name == name)
        .order_by(HistoricalContest.date.desc())
        .first()
    )
    if fallback is not None:
        print(
            f"⚠️  contest_id={orig_cid} not found; "
            f"using HistoricalContest ID={fallback.contest_id} "
            f"(latest match by name='{name}')."
        )
        return fallback.contest_id

    # 4) neither direct nor name‐based match succeeded
    raise ValueError(
        f"No HistoricalContest for contest_id={orig_cid}, "
        f"and no rows with name='{name}'."
    )

def map_team_name_to_abbreviation(full_team_name):
    """
    Convert full team name to its abbreviation using the mapping.
    If the team name is not found, print it and return the original team name.
    """


    abbreviation = _TEAM_RENAME_DICT.get(full_team_name)
    if not abbreviation:
        print(f"Missing team abbreviation for: {full_team_name}")
    return abbreviation or full_team_name

