# src/ivan/sports/mlb/models.py
from datetime import datetime
from typing import Optional, List, Any, Tuple
from sqlalchemy.orm import Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column,
    Integer,
    String,
    JSON,
    DateTime,
    func,
    Float,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    MetaData
)
from dataclasses import dataclass
from ivan.core.player import BasePlayer
from ivan.core.lineup import BaseLineup

nfl_metadata = MetaData()
NFLBase = declarative_base(metadata=nfl_metadata)


class NFLPlayer(NFLBase, BasePlayer):
    __tablename__ = 'nfl_player'

    id              = Column(Integer,   primary_key=True)
    name            = Column(String,    nullable=False)
    team            = Column(String,    nullable=False)
    opponent        = Column(String,    nullable=True)
    position        = Column(String,    nullable=False)
    roster_slot_id  = Column(Integer,   nullable=False)
    gametime        = Column(DateTime,  nullable=False) 
    salary          = Column(Integer,   nullable=False)
    status          = Column(String,    nullable=True)

@dataclass
class FieldPlayer:
    dk_player_id: int
    name: str
    team_abbrev: Optional[str]
    position: str
    salary: int
    roster_position: Optional[str]
    draft_group_id: int
    fppg: Optional[float]
    injury_status: Optional[str]

    def __eq__(self, other):
        return isinstance(other, FieldPlayer) and self.dk_player_id == other.dk_player_id

    def __hash__(self):
        return hash(self.dk_player_id)



class HistoricalContest(NFLBase):
    __tablename__ = "historical_contest_information"

    contest_id          = Column(Integer, primary_key=True)
    draft_group_id      = Column(Integer, nullable=False)
    name                = Column(String, nullable=True)
    date                = Column(DateTime, nullable=True)
    entry_fee           = Column(Float, nullable=False)
    contest_type        = Column(String, nullable=True)
    total_entries       = Column(Integer, nullable=False)
    multi_entry         = Column(Integer, nullable=False)

class ContestPayout(NFLBase):
    __tablename__ = "payouts"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    contest_id    = Column(
        Integer,
        ForeignKey("historical_contest_information.contest_id"),
        nullable=False,
        index=True
    )
    min_position  = Column(Integer, nullable=False)
    max_position  = Column(Integer, nullable=False)
    payout_amount = Column(Float, nullable=False)   # e.g. 1500.0
    payout_text   = Column(String(32), nullable=True)  # e.g. "$1,500.00"
    currency      = Column(String(16), default="USD")  # “USD”


   