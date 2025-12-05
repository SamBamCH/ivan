from ivan.database.session import init_db, get_session

from ivan.sports.nfl.refactor.historical.data_fetch import fetch_boxscores

#init db
engine = init_db('nfl')
session = get_session(engine)
#fetch_stats
fetch_boxscores(session)

