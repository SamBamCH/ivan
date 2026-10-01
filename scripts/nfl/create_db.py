from ivan.database.session import init_db, add_missing_columns
from ivan.sports.nfl.models import NFLBase

engine = init_db("nfl")

NFLBase.metadata.create_all(bind=engine)
for added in add_missing_columns(engine, NFLBase.metadata):
    print(f'added column {added}')
