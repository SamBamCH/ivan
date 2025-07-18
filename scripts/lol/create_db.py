from ivan.database.session import init_db
from ivan.sports.lol.models import LOLBase

engine = init_db("lol")
# this will create *only* the tables defined on LOLBase
LOLBase.metadata.create_all(bind=engine)

