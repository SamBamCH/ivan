#!/usr/bin/env python3
"""
scripts/nba/newsfeed.py

Player news from Rotowire (ivan.sports.nba.newsfeed): fetch on demand, store what is new, read what is on file.

    scripts/nba/newsfeed.py check                 fetch the page, print the new headlines
    scripts/nba/newsfeed.py show [--hours 24]     the headlines on file for the players of the loaded slate
    scripts/nba/newsfeed.py show --all            every headline on file in the window
    scripts/nba/newsfeed.py slate                 the desk's news table: the open questions, the players out tonight who
                                                  played their team's last game, the standing outs; each with his last headline
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
    sub.add_parser("slate")
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
    if a.cmd == "slate":
        return print_slate(session)
    from datetime import datetime, timedelta
    from ivan.sports.nba.models import NBANews
    q = session.query(NBANews).filter(NBANews.first_seen >= datetime.now() - timedelta(hours=a.hours))
    if not a.all:
        pool = {p.name for p in session.query(NBAPlayer).all()}
        q = q.filter(NBANews.player_name.in_(sorted(pool)))
    for x in sorted(q.all(), key=lambda r: (r.first_seen, r.id), reverse=True):
        print(f"  {NF.age(x.first_seen):>9}  {x.team or '':3} {x.player_raw}: {x.headline}" + (f" [{x.injury}]" if x.injury else ""))
    return 0


def print_slate(session) -> int:
    """The three lists of newsfeed.slate_news (what the desk's news table shows), as text."""
    from ivan.core.utils import normalize_name
    from ivan.sports.nba.projection.live import LiveSlate, read_overrides_full
    slate = LiveSlate(session)
    calls = {}
    for n, ov in read_overrides_full().items():
        if ov.get("minutes") is not None:
            calls[n] = "scratched" if ov["minutes"] <= 0 else f"pinned {ov['minutes']:g} min"
        elif ov.get("starter") is not None:
            calls[n] = "starts" if ov["starter"] > 0 else "benched"
    minutes = {normalize_name(p.name): float(p.minutes_proj) for p in slate.players if p.minutes_proj is not None}
    t = NF.slate_news(session, slate, calls=calls, minutes=minutes)

    def line(r, kind):
        tip = r["tip"].strftime("%H:%M") if r["tip"] else "--:--"
        usual = "" if r["hist"] is None else f"usual {r['hist']:.0f}"
        if kind == "q":
            mid = f"now {r['minutes']:.1f}" if r["minutes"] is not None else "now 0"
            mid += f" · {r['call']}" if r["call"] else " · no call"
        else:
            mid = f"{r['last_date'][5:]} · " + (f"{r['last_min']:.0f} min" if r["played_last"] else "did not play") if r["last_date"] else "no game on file"
        news = f"{r['news']['headline']} ({r['news']['age']})" + (" NEW" if r["news"]["new"] else "") if r["news"] else "no news on file"
        return f"  {tip}  {r['team']:3} {r['raw']:26} {r['status']:4} ${r['salary']:<5} {usual:9} {mid:28} {news}"

    print(f"questionable ({len(t['questionable'])}, {sum(1 for r in t['questionable'] if not r['call'])} with no call):")
    for r in t["questionable"]:
        print(line(r, "q"))
    print(f"\nout tonight, played last game ({len(t['new_out'])}):")
    for r in t["new_out"]:
        print(line(r, "new"))
    print(f"\nalready out ({len(t['out'])}):")
    for r in t["out"]:
        print(line(r, "old"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
