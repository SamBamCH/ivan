#!/usr/bin/env python
"""Delete the slate's generated state: lineups and simulation results."""
from sqlalchemy import text

from ivan.database.session import get_session, init_db

session = get_session(init_db("mma"))
for table in ("lineup_simulation_results", "lineup"):
    n = session.execute(text(f"DELETE FROM {table}")).rowcount
    print(f"{table}: {n} rows deleted")
session.commit()
