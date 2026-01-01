from ivan.database.session import init_db, get_session
from sqlalchemy import text

from ivan.sports.nfl.refactor.historical.data_fetch import fetch_boxscores

#init db
engine = init_db('nfl')
session = get_session(engine)
#fetch_stats
fetch_boxscores(session)

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

