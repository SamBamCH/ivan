#!/usr/bin/env python3
"""
scripts/nba/considerations.py

What is unusual about each team's game on the loaded slate (ivan.sports.nba.considerations): rest, regular starters
out, a new starting five, players back from a missed game, new team, a big spread, the opponent's starters out. The
team card's "special considerations" box shows the same list with the desk's hand starter calls in it; this script
uses Rotowire's starters only.

    scripts/nba/considerations.py              every team
    scripts/nba/considerations.py --team DET
"""
import argparse

from ivan.database.session import get_session, init_db
from ivan.sports.nba import considerations as CN
from ivan.sports.nba import experience as X
from ivan.sports.nba.ingest import ensure_schema
from ivan.sports.nba.projection.live import LiveSlate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default=None)
    a = ap.parse_args()
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    slate = LiveSlate(session)
    exp = X.load(); year = X.season_start_year()
    played_before = set(slate.box[(slate.box.game_date < __import__("datetime").date(year, 10, 1)) & slate.box.played].player_name)
    by_team = {}
    for p in slate.players:
        by_team.setdefault(p.team, []).append(p)
    for team in sorted(by_team):
        if a.team and team != a.team.upper():
            continue
        ps = by_team[team]
        opp = next((p.opponent for p in ps if p.opponent), None)
        items = CN.team_considerations(team, ps, slate.box, slate.state, slate.starters.get(team), confirmed=slate.starters_confirmed.get(team, False),
                                       spread=slate.spread(team), opponent=opp, opp_players=by_team.get(opp, []),
                                       is_rookie=lambda n, t=team: X.is_rookie(n, year, exp, n in played_before, team=t))
        print(f"{team}" + (f" vs {opp}" if opp else ""))
        for x in items:
            print(f"  {x['level']:4} {x['kind']:9} {x['text']}")
        if not items:
            print("  nothing unusual on file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
