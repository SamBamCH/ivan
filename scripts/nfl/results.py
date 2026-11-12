#!/usr/bin/env python3
"""
scripts/nfl/results.py

The NFL money, from the finish DraftKings gave each of our entries in the scraped standings (contest_entry_result,
written by the settle job's scrape) against the entry fees (ivan.sports.nfl.results): what the NFL picker's season
strip and day headings, the NFL home tile and the slate's backtest page show, in the terminal.

    scripts/nfl/results.py                    every slate summed per day and overall
    scripts/nfl/results.py --slate 156014     one slate (a draft group), per contest
    scripts/nfl/results.py --contest 197536097   our entries in one contest, best first
"""
import argparse
from datetime import datetime

from ivan.database.session import get_session, init_db
from ivan.sports.nfl import results as R


def money(x) -> str:
    return "—" if x is None else f"{'-' if x < 0 else '+'}${abs(x):,.2f}"


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:+.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slate", help="one slate: the DraftKings draft group id")
    ap.add_argument("--contest", type=int, help="one contest: our entries in it")
    a = ap.parse_args()
    session = get_session(init_db("nfl"))
    try:
        if a.contest:
            cd = R.contest_entries(session, a.contest)
            if cd is None:
                print(f"no entries of ours recorded for contest {a.contest}")
                return 1
            c = cd["contest"]
            print(f"{c['name']} · ${c['entry_fee']:g} · {c['field']:,} entries · ours {c['found']} · {money(c['net'])} net ({pct(c['roi'])})")
            for e in cd["entries"]:
                pts = f"{e['points']:.2f}" if e["points"] is not None else ""
                print(f"  {e['rank']:>7,} {('T' + str(e['tie_size'])) if e['tie_size'] > 1 else '':>6} {pts:>7} {money(e['net']):>10}  {e['lineup']}")
            return 0
        if a.slate:
            res = R.slate_results(session, a.slate)
            sm = res["summary"]
            print(f"slate {a.slate}: {money(sm['net'])} net ({pct(sm['roi'])}) on ${sm['fees']:,.2f}; {sm['cashed']}/{sm['entries']} cashed; "
                  f"{sm['contests']} contests, {sm['unscored']} unscored (${sm['fees_unscored']:,.2f})")
            for c in res["contests"]:
                if c["final"]:
                    print(f"  {c['name'][:58]:58} ${c['entry_fee']:<6g} x{c['found']:<3} field {c['field']:<7} best {c['best_rank'] or '-':<7} cashed {c['cashed']:<3} net {money(c['net']):>10}")
                else:
                    print(f"  {c['name'][:58]:58} ${c['entry_fee']:<6g} x{c['ours']:<3} unscored: {c['scored']}")
            return 0
        d = R.ledger_days(datetime.now(), session)
    finally:
        session.close()
    ov = d["overall"]
    print(f"overall: {money(ov['net'])} net ({pct(ov['roi'])}) on ${ov['fees']:,.2f} in fees; {ov['cashed']}/{ov['entries']} entries cashed; {ov['contests']} contests; "
          f"{ov['days']} days, {ov['slates']} slates, {ov['incomplete']} with contests not scored (${ov['fees_unscored']:,.2f} in fees not counted)")
    for day, g in d["days"].items():
        if g["contests"]:
            print(f"  {day}: {money(g['net']):>10} ({pct(g['roi']):>5}) on ${g['fees']:8.2f}; {g['cashed']:>4}/{g['entries']:<4} cashed; {g['contests']} contests; slates {', '.join(g['slates'])}"
                  + (f"; {g['incomplete']} not scored" if g["incomplete"] else ""))
        else:
            print(f"  {day}: unscored, ${g['fees_unscored']:.2f} in fees; slates {', '.join(g['slates'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
