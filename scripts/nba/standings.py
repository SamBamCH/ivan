#!/usr/bin/env python3
"""
scripts/nba/standings.py

Real contest fields from DraftKings (ivan.sports.nba.standings).

    scripts/nba/standings.py status [--days 3]                 played slates and which contests are exported / derived
    scripts/nba/standings.py scrape [--days 3] [--public 3] [--contest-ids 1,2] [--headless]
    scripts/nba/standings.py derive [--contest-ids 1,2]        parse every final export without a derived folder

An export taken while games were on (the late swap's standings fetch) shows the
unplayed slots as LOCKED: it is not a field. status lists it as mid-slate, scrape
exports it again, derive refuses it and removes a derived folder built from one.

scrape needs the logged-in DraftKings Chrome profile (chrome-selenium/dk-profile);
without --headless, Chrome opens and waits for you to log in when the session
has expired. Per played slate it exports every contest we played (the slate's
own entries file, data/nba/slates/<dg>/input/entries.csv; --entries FILE for a
slate this machine does not hold) plus the --public largest public classic
contests. DraftKings keeps standings about two weeks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from ivan.core import raw
from ivan.sports.nba import standings as S
from ivan.sports.nba import slates as SL


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    SL.add_arg(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("status", "scrape"):
        p = sub.add_parser(name)
        p.add_argument("--days", type=int, default=3, help="slates locked in the last N days (default 3)")
        p.add_argument("--public", type=int, default=3, help="largest public classic contests per slate (default 3)")
        p.add_argument("--entries", default=None, help="a DraftKings entries file naming our contests, for a slate this machine does not hold")
    sub.choices["scrape"].add_argument("--contest-ids", default=None, help="export these ids instead")
    sub.choices["scrape"].add_argument("--headless", action="store_true")
    sub.choices["scrape"].add_argument("--no-derive", action="store_true")
    d = sub.add_parser("derive"); d.add_argument("--contest-ids", default=None)
    a = ap.parse_args()
    dg, session = SL.open(a.slate)
    raw.ensure_tables(session.get_bind())
    now = datetime.now()

    if a.cmd in ("status", "scrape") and not getattr(a, "contest_ids", None):
        plan = S.contests_to_scrape(session, now - timedelta(days=a.days), now, public_n=a.public,
                                    entries_csv=Path(a.entries) if a.entries else None, now=now)
        derived = S.derived_ids() - S.partial_derived_ids()
        for s in plan:
            print(f"slate {s['slate_key']} locked {s['lock']:%a %m-%d %H:%M}: {len(s['contests'])} contests ({len(s['ours'])} ours), "
                  f"{len(s['scraped'])} exported final, {sum(1 for c in s['contests'] if c in derived)} derived, {len(s['to_scrape'])} to scrape")
            if not s["ours"]:
                print(f"    (no entries file for this slate: only the public contests; --entries FILE names ours)")
            for c in s["contests"]:
                state = "final    " if c in s["scraped"] else ("mid-slate" if S.has_standings(c) else "pending  ")
                print(f"    {c} {state} {'derived' if c in derived else '       '} {'ours  ' if c in s['ours'] else 'public'}  {s['names'].get(c, '')}")
        if not plan:
            print("no played slates in the window (a slate counts once its lock is 5 hours past)")
        else:
            print("mid-slate: the only export was taken while games were on (the late swap's fetch); it is not a field and is scraped again")
        if a.cmd == "status":
            return 0
        ids = [c for s in plan for c in s["to_scrape"]]
    elif a.cmd == "scrape":
        ids = [int(x) for x in a.contest_ids.split(",") if x.strip()]
    else:
        ids = []

    if a.cmd == "scrape":
        if not ids:
            print("nothing to scrape"); return 0
        print(f"exporting {len(ids)} contest(s)...")
        r = S.scrape(ids, headless=a.headless, log=print)
        print(f"exported {len(r['ok'])}, failed {len(r['failed'])}" + (" (login needed: run without --headless and log in)" if r["login_needed"] else ""))
        raw.sync_index(session, "nba")
        if a.no_derive or not r["ok"]:
            return 0 if not r["failed"] else 2
        for cid in r["ok"]:
            S.derive_contest(session, cid, log=print)
        return 0 if not r["failed"] else 2

    if a.contest_ids:
        for cid in [int(x) for x in a.contest_ids.split(",") if x.strip()]:
            S.derive_contest(session, cid, log=print)
    else:
        out = S.derive_pending(session, log=print)
        print(f"derived {len(out)} contest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
