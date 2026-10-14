#!/usr/bin/env python
"""Upcoming DraftKings MMA contests into the contest table. ingest.py does this too; this is the standalone step."""
from ivan.database.session import get_session, init_db
from ivan.sports.mma.contests import fetch_contests
from ivan.sports.mma.models import MMABase

engine = init_db("mma")
MMABase.metadata.create_all(engine)
r = fetch_contests(get_session(engine))
print(f"{r['upcoming']} upcoming MMA contests, {r['new']} new")
