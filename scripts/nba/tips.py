#!/usr/bin/env python3
"""
scripts/nba/tips.py

The record at each game's tip (ivan.sports.nba.tips): data/nba/slates/<dg>/tips/<YYYYMMDD-HHMM>/, written by the archive
timer (scripts/nba/archive_at_lock.py) and read by every backtest of that game's models.

    scripts/nba/tips.py list --slate 156078            the tips on file, with what each holds
    scripts/nba/tips.py backfill --slate 156078        tips from the archive's at_tip captures (slates archived before 2026-11-05)
    scripts/nba/tips.py backfill --all                 every held slate
    scripts/nba/tips.py show --slate 156078 --tip 2026-11-04T18:00   one tip's meta and the players tipping
"""
import argparse
import json
from datetime import datetime

from ivan.sports.nba import slates as S
from ivan.sports.nba import tips as T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    b = sub.add_parser("backfill"); b.add_argument("--all", action="store_true")
    sh = sub.add_parser("show"); sh.add_argument("--tip", required=True)
    a = ap.parse_args()
    if a.cmd == "backfill" and a.all:
        n = 0
        for dg in S.held():
            dg, session = S.open(dg)
            try:
                n += T.backfill(dg, session)
            finally:
                session.close()
        print(f"{n} tip(s) written"); return 0
    dg, session = S.open(a.slate)
    try:
        if a.cmd == "list":
            tips = T.tips_of(dg)
            if not tips:
                print(f"slate {dg}: no tips on file (the archive timer writes them from 2026-11-05; backfill from the captures with `backfill`)"); return 0
            for t in tips:
                m = T.read_tip(dg, t)["meta"]
                print(f"{t:%Y-%m-%d %H:%M}  {','.join(m['teams_tipping']) or '-':16} players {m['players']:<4} projected {m['projected']:<4} props {m['odds_rows']:<6} "
                      f"lines {m['line_rows']:<4} news {m['news_rows']:<4} edits {'yes' if m['edits'].get('overrides') or m['edits'].get('rotations') else 'no ':3}  {m['source']}")
        elif a.cmd == "backfill":
            print(f"{T.backfill(dg, session)} tip(s) written")
        else:
            got = T.read_tip(dg, datetime.fromisoformat(a.tip))
            if got is None:
                print("no such tip"); return 1
            print(json.dumps(got["meta"], indent=1))
            for p in got["players"]:
                if p.get("team") in got["meta"]["teams_tipping"]:
                    print(f"  {p['team']:3} {p['name']:26} {p.get('starter') or '':9} min {p.get('minutes_proj') or 0:5.1f} fpts {p.get('fpts_mean') or 0:5.1f} own {p.get('ownership') or 0:5.1f}")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
