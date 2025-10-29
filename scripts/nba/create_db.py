from ivan.database.session import init_db
from ivan.sports.nba.models import NBABase

engine = init_db("nba")

NBABase.metadata.create_all(bind=engine)