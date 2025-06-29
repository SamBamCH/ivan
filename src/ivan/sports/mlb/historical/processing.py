# ─── src/ivan/sports/mlb/historical/processing.py ───────────────────────────────

from collections import Counter
from ivan.sports.mlb.models import BatterProjection, PitcherProjection
from sqlalchemy import cast, Date, func

from ivan.core.utils import normalize_name  # your name‐normalizer

import datetime

from datetime import timezone
from sqlalchemy import cast, Date, func
from pprint import pprint

def to_naive_utc(dt_or_str):
    """
    Turn an ISO‐8601 timestamp (with Z or offset) or an aware datetime
    into a naive UTC datetime.
    """
    if isinstance(dt_or_str, str):
        # support strings ending in 'Z' or '+00:00'
        s = dt_or_str.rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt_or_str

    # normalize to UTC
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    # drop tzinfo
    return dt.replace(tzinfo=None)

def lookup_projection_and_ownership(db, normalized_name: str, contest_date):
    # parse contest_date → a date
    if isinstance(contest_date, datetime.datetime):
        slate_date = contest_date.date()
    else:
        slate_date = datetime.date.fromisoformat(contest_date.split("T",1)[0])

    def _latest_for(model, name_field, fpts_field, own_field, team_field):
        return (
            db.query(model)
              .filter(
                  func.lower(getattr(model, name_field)) == normalized_name.lower(),
                  # SQLite: date(created_at) == 'YYYY-MM-DD'
                  func.date(model.created_at) == slate_date.isoformat()
              )
              .order_by(model.created_at.desc())
              .first()
        )

    bat = _latest_for(BatterProjection, "name", "fpts", "pown", "team")
    if bat:
        return bat.fpts or 0.0, bat.pown or 0.0, bat.team

    pit = _latest_for(PitcherProjection, "player", "fpts", "ownership", "team")
    if pit:
        return pit.fpts or 0.0, pit.ownership or 0.0, pit.team
    
    print(f'no player found for {normalized_name} on date {slate_date}')

    return 0.0, 0.0, None


def process_entry_for_metrics(
    db,
    draft_group_id: int,
    entry_key: int,
    contest_date: str,
    api_session
):
    """
    1) Fetch roster JSON for this entry_key
    2) For each player:
        • normalize name,
        • lookup projection & ownership%,
        • accumulate projection_sum, ownership_sum, teams list
        • increment player_counter[player_id]
    3) Classify stack_type from teams
    4) Return (projection_sum, ownership_sum, stack_type, total_player_ownership)
    """
    from ivan.sports.mlb.historical.api import fetch_roster_for_entry

    roster = fetch_roster_for_entry(draft_group_id, entry_key, api_session)
    print(roster)
    projection_sum = 0.0
    ownership_sum  = 0.0
    teams = []
    total_own = 0.0

    for raw_name in roster:
        norm_name = normalize_name(raw_name)
        proj_fpts, own_pct, team_abbrev = lookup_projection_and_ownership(db, norm_name, contest_date)

        projection_sum += proj_fpts
        ownership_sum  += own_pct
        total_own       = ownership_sum  # in this design, total_own and ownership_sum are identical

        teams.append(team_abbrev)

        # (Optionally) increment a global player_counter here if you want to store per‐player counts

    # 3) Classify stack_type from 'teams'
    team_counts = Counter([t for t in teams if t])
    if not team_counts:
        stack_type = "NoHitters"
    else:
        most = team_counts.most_common()
        top_team, top_cnt = most[0]
        if top_cnt >= 4:
            stack_type = f"{top_cnt}-man_{top_team}"
        elif top_cnt == 3:
            second_cnt = most[1][1] if len(most) > 1 else 0
            if second_cnt == 3:
                second_team = most[1][0]
                stack_type = f"3-3_{top_team}_{second_team}"
            else:
                stack_type = f"3-man_{top_team}"
        elif top_cnt == 2:
            doubles = [t for (t,c) in most if c == 2]
            if len(doubles) >= 2:
                stack_type = f"2-2_{doubles[0]}_{doubles[1]}"
            else:
                stack_type = f"Double_{top_team}"
        else:
            stack_type = "Mixed"

    return projection_sum, ownership_sum, stack_type, total_own
