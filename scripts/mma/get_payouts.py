#!/usr/bin/env python
"""Payout tables for MMA contests that lack one. ingest.py does this for its slate; this is the standalone step."""
import argparse

from ivan.database.session import get_session, init_db
from ivan.sports.mma.contests import fetch_payouts

ap = argparse.ArgumentParser(description="Fetch DK payout tables for MMA contests missing one")
ap.add_argument("--days", type=int, default=2, help="only contests starting within the last N days or later (default 2)")
ap.add_argument("--all", action="store_true", help="every contest without payouts, however old")
a = ap.parse_args()
r = fetch_payouts(get_session(init_db("mma")), days=None if a.all else a.days)
print(f"{r['missing']} contests lacked payouts: {r['fetched']} fetched, {r['failed']} failed")
