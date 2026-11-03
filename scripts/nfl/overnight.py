import os
import sys
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

# Settle the slates we played in the last days: scrape their contests (with our
# username) so every slate stays backtestable. Needs the saved DraftKings Chrome
# profile, which wants a display (headless fails once its login cookies age).
if os.environ.get("IVAN_DK_USERNAME") and os.environ.get("DISPLAY"):
    print("[overnight] settling recent slates (standings, box scores, derived files, status rows)")
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "nfl" / "settle.py"), "--days", "3"], check=False)
else:
    print("[overnight] settling recent slates without the scrape (set IVAN_DK_USERNAME and run with a display to scrape the contests we played)")
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "nfl" / "settle.py"), "--days", "3", "--no-scrape"], check=False)

# The reference bundle for the server (ivan.core.feed): every table but the working set, plus data/nfl/models. Nothing is
# pushed: the server's NFL desk pulls it over ssh when its "Refresh from laptop" is pressed (scripts/nfl/feed.py pull), and
# that pull also asks for a fresh bundle, so this nightly publish only saves it the wait.
print("[overnight] publishing the NFL reference bundle (data/feed/nfl)")
subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "nfl" / "feed.py"), "publish", "--quiet"], check=False)
