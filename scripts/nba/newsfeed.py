#!/usr/bin/env python3
"""
scripts/nba/newsfeed.py

Player news from Rotowire (ivan.sports.nba.newsfeed): fetch on demand, store what is new, read what is on file.

    scripts/nba/newsfeed.py check                 fetch the page, print the new headlines
    scripts/nba/newsfeed.py show [--hours 24]     the headlines on file for the players of the loaded slate
    scripts/nba/newsfeed.py show --all            every headline on file in the window
"""
import argparse

from ivan.database.session import get_session, init_db
from ivan.sports.nba import newsfeed as NF
from ivan.sports.nba.ingest import ensure_schema
from ivan.sports.nba.models import NBAPlayer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sh = sub.add_parser("show"); sh.add_argument("--hours", type=float, default=24.0); sh.add_argument("--all", action="store_true")
    a = ap.parse_args()
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    if a.cmd == "check":
        r = NF.check(session)
        pool = {p.name for p in session.query(NBAPlayer).all()}
        for x in r["new"]:
            tag = " *" if x.player_name in pool else ""
            print(f"  {x.team or '':3} {x.player_raw}: {x.headline}{tag}")
        print(f"{len(r['new'])} new of {r['total']} ({sum(1 for x in r['new'] if x.player_name in pool)} on the loaded slate, marked *)")
        return 0
    from datetime import datetime, timedelta
    from ivan.sports.nba.models import NBANews
    q = session.query(NBANews).filter(NBANews.first_seen >= datetime.now() - timedelta(hours=a.hours))
    if not a.all:
        pool = {p.name for p in session.query(NBAPlayer).all()}
        q = q.filter(NBANews.player_name.in_(sorted(pool)))
    for x in sorted(q.all(), key=lambda r: (r.first_seen, r.id), reverse=True):
        print(f"  {NF.age(x.first_seen):>9}  {x.team or '':3} {x.player_raw}: {x.headline}" + (f" [{x.injury}]" if x.injury else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
