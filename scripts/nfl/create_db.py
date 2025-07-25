from ivan.database.session import init_db
from ivan.sports.nfl.models import NFLBase

engine = init_db("nfl")

NFLBase.metadata.create_all(bind=engine)