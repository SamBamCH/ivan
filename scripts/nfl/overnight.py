import os
import subprocess

from ivan.database.session import init_db, get_session
from ivan.core.utils import PROJECT_ROOT
from sqlalchemy import text

from ivan.sports.nfl.refactor.historical.data_fetch import fetch_boxscores, fetch_schedules, seasons_to_fetch
from ivan.sports.nfl.dst_scoring import build_dst_game_table

#init db
engine = init_db('nfl')
session = get_session(engine)
#fetch stats, schedule (scores + closing lines) for every season through the current one
n = fetch_schedules(session)
print(f"[overnight] nfl_schedule: {n} games")
n = fetch_boxscores(session)
print(f"[overnight] nfl_stats: {n} player-weeks")
# DST points per team-game feed the team correlation model at ingest, so
# they must track the stats table.
n = build_dst_game_table(None, years=seasons_to_fetch(), session=session)
print(f"[overnight] nfl_dst_game: {n} team-games")

#clear old data
#clear NFLPlayer -> ProjectionSnapshot.p[p]
#clear sim results
session.execute(text("DELETE FROM lineup_simulation_results"))
session.commit()
#clear lineups
session.execute(text("DELETE FROM lineup"))
session.commit()
#clean_old_files
#get_contest_ids
#get_payouts
#get_contest_data last n days
#build models needed

# The DELETEs above leave free pages behind; SQLite never returns them to the
# OS on its own (the file hit 18 GB with ~600 MB of live data). Reclaim once
# the freelist is a meaningful share of the file. VACUUM cannot run inside a
# transaction, hence the autocommit connection.
page_count, freelist = (
    session.execute(text("PRAGMA page_count")).scalar(),
    session.execute(text("PRAGMA freelist_count")).scalar(),
)
if page_count and freelist / page_count > 0.25:
    print(f"[overnight] VACUUM: {freelist}/{page_count} pages free")
    session.close()
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("VACUUM"))

# Push reference tables + model pickles to the server. No-op unless
# IVAN_SYNC_HOST is set (in .env or the environment); the server's per-slate
# working tables are never touched, see scripts/nfl/sync_to_server.sh.
sync_host = os.environ.get("IVAN_SYNC_HOST")
if sync_host:
    print(f"[overnight] syncing reference data to {sync_host}")
    subprocess.run([str(PROJECT_ROOT / "scripts" / "nfl" / "sync_to_server.sh"), sync_host], check=True)
