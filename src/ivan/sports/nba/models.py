from datetime import datetime
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

nba_metadata = MetaData()
NBABase = declarative_base(MetaData=nba_metadata)

class NBAStat(NBABase):
    