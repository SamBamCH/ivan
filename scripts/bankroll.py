#!/usr/bin/env python3
"""
scripts/bankroll.py

The bankroll the slate book is chosen from (ivan.core.bankroll): config/bankroll.json's start plus what our entries
made in every sport's results ledger (the ledgers of scripts/pnl.py), on the days from `since` on. A choose run with
no --bankroll-dollars reads this figure, so it grows and shrinks with the results.

    scripts/bankroll.py [--start 1000] [--since 2026-11-01] [--sports nba,nfl]      (the flags try a figure; the file is not written)
"""
import argparse
from datetime import datetime

from ivan.core import bankroll


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=float, default=None, help="the starting figure (config/bankroll.json start)")
    ap.add_argument("--since", default=None, help="count the results from this day on (config/bankroll.json since; default every result)")
    ap.add_argument("--sports", default=None, help="default: nba, nfl and every other sport the imported DraftKings entry history holds")
    a = ap.parse_args()
    cfg = bankroll.load_config()
    if a.start is not None:
        cfg["start"] = a.start
    if a.since:
        cfg["since"] = a.since[:10]
    if a.sports:
        cfg["sports"] = [s.strip().lower() for s in a.sports.split(",") if s.strip()]
    info = bankroll.current(datetime.now(), cfg)
    print(bankroll.describe(info))
    for s, v in sorted(info["sports"].items()):
        print(f"  {s.upper():4} {'-' if v < 0 else '+'}${abs(v):,.2f}")
    if info["bankroll"] <= 0:
        print("the results have used up the bankroll: a choose run will refuse until start / since in config/bankroll.json say what you play from")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
