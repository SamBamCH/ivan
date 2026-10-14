from ivan.database.session import add_missing_columns, init_db
from ivan.sports.mma.models import MMABase

engine = init_db("mma")
MMABase.metadata.create_all(bind=engine)
for added in add_missing_columns(engine, MMABase.metadata):
    print(f"added column {added}")
