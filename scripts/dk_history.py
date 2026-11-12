#!/usr/bin/env python3
"""
scripts/dk_history.py

DraftKings' own record of what we played (ivan.core.dk_history): the account's entry-history export, one row per entry
of every sport with its fee and winnings. Imported, it is the money of every contest it lists: the sports' results
ledgers (scripts/nba/results.py, scripts/nfl/results.py), the tiles and the home page's total with its tax
(scripts/pnl.py) read it in place of the scraped standings, and a sport with no results code (MMA, ...) gets its
ledger from it alone.

    scripts/dk_history.py                              what is stored: per sport and year, entries, fees, won, net
    scripts/dk_history.py import [PATH] [--since 2026-01-01 | --all]
                                                       merge an export into data/dk/entry_history.csv. PATH defaults to the newest
                                                       draftkings-contest-entry-history*.csv in ~/Downloads; --since defaults to
                                                       January 1 of this year, --all takes every row

The export: draftkings.com -> My Contests -> History -> "Download Entry History" (the browser leaves it in ~/Downloads).
"""
import argparse
from datetime import date, datetime
from pathlib import Path

from ivan.core import dk_history as H


def money(x) -> str:
    return f"{'-' if x < 0 else '+'}${abs(x):,.2f}"


def show() -> int:
    df = H.entries()
    if df.empty:
        print(f"nothing imported yet ({H.STORE}): scripts/dk_history.py import [PATH]")
        return 1
    until = H.covered_until()
    print(f"{len(df):,} entries, {df.date.min():%Y-%m-%d} .. {df.date.max():%Y-%m-%d}; exported {until:%Y-%m-%d %H:%M}" if until else f"{len(df):,} entries")
    g = df.assign(year=df.date.dt.year, sport=df.sport.str.upper()).groupby(["year", "sport"])
    for (year, sport), d in g:
        fees, won = d.entry_fee.sum(), d.winnings.sum()
        print(f"  {int(year)} {sport:5} {len(d):>6,} entries in {d.contest_id.nunique():>4} contests; ${fees:>10,.2f} in fees; won ${won:>10,.2f}; "
              f"{money(won - fees):>11} ({100 * (won - fees) / fees:+.0f}%)" if fees else f"  {int(year)} {sport:5} {len(d):>6,} entries, no fees")
    fees, won = df.entry_fee.sum(), df.winnings.sum()
    print(f"  all        {len(df):>6,} entries in {df.contest_id.nunique():>4} contests; ${fees:>10,.2f} in fees; won ${won:>10,.2f}; {money(won - fees):>11}")
    return 0


def do_import(path: Path, a) -> int:
    since = None if a.all else (date.fromisoformat(a.since) if a.since else date(datetime.now().year, 1, 1))
    m = H.import_file(path, since=since)
    print(f"imported {path}: {m['read_rows']:,} entries{f' since {since}' if since else ''}, {m['new_rows']:,} new; the store holds {m['rows']:,} "
          f"({(m['first'] or '')[:10]} .. {(m['last'] or '')[:10]}); by sport {m['sports']}")
    return show()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("import")
    p.add_argument("path", nargs="?", help="the export (default: the newest in ~/Downloads)")
    p.add_argument("--since", default=None, help="keep entries from this day on (default: January 1 of this year)")
    p.add_argument("--all", action="store_true", help="keep every entry of the export")
    a = ap.parse_args()
    if a.cmd == "import":
        path = Path(a.path) if a.path else H.newest_export()
        if path is None or not path.exists():
            print(f"no export found ({a.path or '~/Downloads/' + H.EXPORT_GLOB}): draftkings.com -> My Contests -> History -> Download Entry History")
            return 1
        return do_import(path, a)
    return show()


if __name__ == "__main__":
    raise SystemExit(main())
