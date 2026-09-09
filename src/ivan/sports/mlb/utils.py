# src/ivan/sports/mlb/utils.py
"""MLB-specific helpers (moved out of ivan.core.utils so core no longer imports a sport)."""
from collections import defaultdict
from sqlalchemy.orm import Session
from ivan.sports.mlb.models import MLBPlayer, HistoricalContest


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

