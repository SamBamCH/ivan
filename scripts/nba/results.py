#!/usr/bin/env python3
"""
scripts/nba/results.py

The money, from DraftKings' final standings exports and the payout tables (ivan.sports.nba.backtest.results): what the picker's
season strip, the day headings and the home tile show, in the terminal.

    scripts/nba/results.py                       every held past slate summed per day and overall (the cached ledgers; incomplete slates re-checked)
    scripts/nba/results.py --slate 156078        one slate's ledger, per contest
    scripts/nba/results.py --slate 156078 --force   recompute it (after a scrape, or when in doubt)
"""
import argparse
from datetime import datetime

from ivan.sports.nba.backtest import results as R
from ivan.sports.nba import slates as S


def money(x) -> str:
    return "—" if x is None else f"{'-' if x < 0 else '+'}${abs(x):.2f}"


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:+.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("--force", action="store_true", help="recompute the slate's ledger")
    a = ap.parse_args()
    now = datetime.now()
    if a.slate:
        led = R.slate_ledger(a.slate, now, force=a.force)
        sm = led["summary"]
        print(f"slate {led['draft_group']} {led['day']}: {money(sm['net'])} net ({pct(sm['roi'])}) on ${sm['fees']:.2f}; {sm['cashed']}/{sm['entries']} cashed; "
              f"{sm['contests']} contests final, {sm['unscored']} unscored (${sm['fees_unscored']:.2f}); {'complete' if led['complete'] else 'INCOMPLETE'}; user {led['user']}")
        for c in led["contests"]:
            if c["final"]:
                print(f"  {c['name'][:42]:42} ${c['entry_fee']:<6g} x{c['found']:<3} field {c['field']:<6} best {c['best_rank'] or '-':<6} cashed {c['cashed']:<3} net {money(c['net']):>9}")
            else:
                print(f"  {c['name'][:42]:42} ${c['entry_fee']:<6g} x{c['ours']:<3} {c['scored']}")
        return 0
    d = R.ledger_days(now)
    ov = d["overall"]
    print(f"overall: {money(ov['net'])} net ({pct(ov['roi'])}) on ${ov['fees']:.2f} in fees; {ov['cashed']}/{ov['entries']} entries cashed; {ov['contests']} contests; "
          f"{ov['days']} days, {ov['slates']} slates, {ov['incomplete']} incomplete (${ov['fees_unscored']:.2f} in fees not counted)")
    for day, g in d["days"].items():
        print(f"  {day}: {money(g['net']):>10} ({pct(g['roi']):>5}) on ${g['fees']:7.2f}; {g['cashed']:>3}/{g['entries']:<3} cashed; {g['contests']} contests; slates {g['slates']}"
              + (f"; {g['incomplete']} not final" if g["incomplete"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
