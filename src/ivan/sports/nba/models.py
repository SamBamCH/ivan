from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from ivan.database.session import Base
from sqlalchemy.orm import Session, relationship
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
    MetaData,
    Table,
    PrimaryKeyConstraint,
    Date
)
from typing import Optional, List, Any, Tuple


nba_metadata = MetaData()
NBABase = declarative_base(metadata=nba_metadata)
class NBAStat(NBABase):
    __tablename__ = "nba_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Game identifiers
    game_id = Column(String, nullable=False, index=True)
    game_date = Column(Date, nullable=False, index=True)
    season = Column(String, nullable=False)  # e.g., "2023-24"

    # Player identifiers
    player_id = Column(String, nullable=False, index=True)
    player_name = Column(String, nullable=False)

    # Team info
    team_abbreviation = Column(String, nullable=False)
    opponent_team = Column(String)
    is_home = Column(Integer)  # 1 for home, 0 for away
    position = Column(String)  # G, F, or C from START_POSITION

    # Box score stats
    minutes = Column(Float)
    points = Column(Integer, default=0)
    rebounds = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    steals = Column(Integer, default=0)
    blocks = Column(Integer, default=0)
    turnovers = Column(Integer, default=0)

    # Shooting stats
    fgm = Column(Integer, default=0)  # field goals made
    fga = Column(Integer, default=0)  # field goals attempted
    fg_pct = Column(Float)
    fg3m = Column(Integer, default=0)  # 3-pointers made
    fg3a = Column(Integer, default=0)  # 3-pointers attempted
    fg3_pct = Column(Float)
    ftm = Column(Integer, default=0)  # free throws made
    fta = Column(Integer, default=0)  # free throws attempted
    ft_pct = Column(Float)

    # Additional stats
    plus_minus = Column(Integer)

    # Fantasy points
    dk_fpts = Column(Float, default=0.0)

    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_nba_stats_game_player"),
    )

class NBAPlayer(NBABase):
    __tablename__ = "nba_player"

    id              = Column(Integer,   primary_key=True)
    name            = Column(String,    nullable=False, unique=True)
    team            = Column(String,    nullable=False)
    opponent        = Column(String,    nullable=True)
    position        = Column(String,    nullable=False)
    roster_slot_id  = Column(Integer,   nullable=False)
    gametime        = Column(DateTime,  nullable=False)
    salary          = Column(Integer,   nullable=False)
    status          = Column(String,    nullable=True)
    fpts_mean       = Column(Float,     nullable=True)
    fpts_std        = Column(Float,     nullable=True)
    fpts_updated_at = Column(DateTime,  nullable=True)


class NBAOdds(NBABase):
    __tablename__ = "odds"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player          = Column(String,  nullable=True)
    market          = Column(String,  nullable=True)
    sportsbook      = Column(String,  nullable=True)
    side            = Column(String,  nullable=True)
    point           = Column(Float,   nullable=True)
    price           = Column(Integer, nullable=True)
    last_update     = Column(String,  nullable=True)

class Lineup(NBABase):
    __tablename__ = "lineup"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    player_ids_list  = Column(JSON,    nullable=False)
    created_at       = Column(DateTime, server_default=func.now(), nullable=False)

    # simulation_results = relationship(
    #     "LineupSimulationResult",
    #     backref="lineup",
    #     cascade="all, delete-orphan",
    #     lazy="select"
    # )

    def player_ids(self) -> List[int]:
        return list(self.player_ids_list)

    def to_dict(self, session: Session) -> dict:
        players = (
            session.query(NBAPlayer)
                   .filter(NBAPlayer.id.in_(self.player_ids_list))
                   .all()
        )
        pitchers = [p.id for p in players if p.position.upper()=='P']
        batters  = [p.id for p in players if p.position.upper()!='P']
        if len(pitchers)!=1 or len(batters)!=9:
            raise ValueError("Lineup must have 1 pitcher and 9 batters.")
        out = {'pitcher_id':pitchers[0]}
        for idx,b_id in enumerate(batters,1): out[f'batter{idx}_id'] = b_id
        return out

    def save(self, session: Session) -> None:
        session.add(self)
        session.commit()

class HistoricalContest(NBABase): 
    __tablename__ = "historical_contest_information"

    contest_id          = Column(Integer, primary_key=True)
    draft_group_id      = Column(Integer, nullable=False)
    name                = Column(String, nullable=True)
    date                = Column(DateTime, nullable=True)
    entry_fee           = Column(Float, nullable=False)
    contest_type        = Column(String, nullable=True)
    total_entries       = Column(Integer, nullable=False)
    multi_entry         = Column(Integer, nullable=False)

class ContestPayout(NBABase):
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

class LineupSimulationResult(NBABase):
    __tablename__ = "lineup_simulation_results"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    lineup_id  = Column(Integer, ForeignKey("lineup.id"), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # store per-simulation arrays; each JSON list has length = num_simulations
    fpts    = Column(JSON, nullable=False)
    payout  = Column(JSON, nullable=False)
    roi     = Column(JSON, nullable=False)
    rank    = Column(JSON, nullable=False)

    # attach each sim‐result row to exactly one contest
    contest_id = Column(
        Integer,
        ForeignKey("historical_contest_information.contest_id"),
        nullable=False,
        index=True
    )
    contest_name = Column(
        String, 
        ForeignKey("historical_contest_information.name"),
        nullable=False,
        index=True
    )