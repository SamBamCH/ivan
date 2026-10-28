#!/usr/bin/env python3
"""
scripts/nba/field_ownership.py

Build the public field on the stored public projection and write every player's ownership from it
(ingest.refresh_field): the desk's "Field + ownership" button. Nothing else builds it on a slate day
except the sim, which builds the field it needs (and writes ownership from the largest contest's);
the refresh and the generate step leave it alone (2026-10-28). Cached by the public projection's key,
so a second run after no change is instant; about 45 seconds for 3,000 entries, 160 for 10,000.

    scripts/nba/field_ownership.py                 # config field.ownership_entries (default 3,000)
    scripts/nba/field_ownership.py --entries 10000
"""
import argparse

from ivan.database.session import get_session, init_db
from ivan.sports.nba.ingest import ensure_schema
from ivan.sports.nba.public_field import refresh_field_ownership


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", type=int, default=None, help="field entries (default: config field.ownership_entries, 3,000)")
    a = ap.parse_args()
    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    r = refresh_field_ownership(session, log=lambda m: print(m, flush=True), size=a.entries)
    print(f"ownership for {r['owned']} players in {r['seconds']}s", flush=True)
    return 0 if r["owned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
