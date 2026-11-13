#!/usr/bin/env python3
"""
scripts/nba/game_context.py

Which of a team's played games are a picture of its usual rotation (ivan.sports.nba.projection.rotation.game_context):
a game that was not close (six or more minutes of garbage time, or a final margin of 20 or more), a regular who did not
play (averaged 24+ minutes over at least three of the team's five games before it), a regular who left early. The team page
shades these games in the played-game picker and in every player's recent-games hover; this prints the same read.

    scripts/nba/game_context.py --team DET              the team's games on file, newest first
    scripts/nba/game_context.py --team DET --last 15
    scripts/nba/game_context.py                         every team of the slate, its last 10
"""
import argparse

from ivan.sports.nba.projection import rotation as ROT
from ivan.sports.nba.projection.live import LiveSlate
from ivan.sports.nba import slates as S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("--team", default=None)
    ap.add_argument("--last", type=int, default=None, help="how many games back (default: all for one team, 10 for the slate)")
    a = ap.parse_args()
    dg, session = S.open(a.slate)
    slate = LiveSlate(session)
    teams = [a.team.upper()] if a.team else sorted(slate.teams())
    n = a.last or (None if a.team else 10)
    for team in teams:
        ctx = ROT.game_context(session, slate.box, team)
        b = slate.box[slate.box.team == team]
        games = b[["game_date", "game_id"]].drop_duplicates().sort_values(["game_date", "game_id"], ascending=False)
        opp = {str(g): next((t for t in sub.team.unique() if t != team), "") for g, sub in slate.box[slate.box.game_id.isin(set(b.game_id))].groupby("game_id")}
        rows = list(games.itertuples())[:n] if n else list(games.itertuples())
        clean = sum(1 for r in rows if not (ctx.get(str(r.game_id)) or {}).get("shade"))
        print(f"{team}: {len(rows)} games, {clean} ordinary")
        for r in rows:
            c = ctx.get(str(r.game_id)) or {}
            mark = {"": "   ", "g-blow": "B  ", "g-inj": " I ", "g-blow g-inj": "BI "}[c.get("shade", "")]
            m = c.get("margin")
            print(f"  {mark} {r.game_date} vs {opp.get(str(r.game_id), ''):3} {m:+4d}  {c.get('why', '')}" if m is not None else f"  {mark} {r.game_date} vs {opp.get(str(r.game_id), ''):3}       {c.get('why', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
