from ivan.database.session import get_session, init_db
from ivan.sports.nfl.refactor.slate import Slate
from ivan.sports.nfl.models import NFLPlayer
from ivan.sports.nfl.u

#init db
engine = init_db('nfl')
session = get_session(engine)
#read all players
players = session.query(NFLPlayer).all() #TODO: add min_fpts filtering here?

for player in players:
    pos = getattr(rp, "position", None)
    pos_tuple = coerce_positions(pos or [])
slate = Slate(field_players, session)
####GENERATE LINEUPS
slate.generate_lineups_for_slate()
####SIM
####CHOOSE
