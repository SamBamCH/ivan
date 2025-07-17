from ivan.database.session import init_db, Base
from ivan.sports.lol.models import LOLPlayer, HistoricalContest, ContestPayout, LOLStat

engine = init_db("lol")
Base.metadata.create_all(
    bind=engine,
    tables=[
        LOLPlayer.__table__,
        HistoricalContest.__table__,
        ContestPayout.__table__,
        LOLStat.__table__,
    ]
)
