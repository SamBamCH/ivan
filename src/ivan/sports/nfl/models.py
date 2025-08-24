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
    MetaData,
    Table,
    PrimaryKeyConstraint
)
from sqlalchemy.ext.automap import automap_base

from dataclasses import dataclass
import json
from pathlib import Path
from ivan.core.player import BasePlayer
from ivan.core.lineup import BaseLineup
from ivan.core.utils          import PROJECT_ROOT
from ivan.database.session import init_db

nfl_metadata = MetaData()
NFLBase = declarative_base(metadata=nfl_metadata)

# ── 1) load your DB URL & init engine ────────────────────────────────────────
cfg    = json.load(open(Path(PROJECT_ROOT) / "config" / "nfl.json", encoding="utf-8"))
engine = init_db("nfl", cfg["database_url"])

# ── 2) reflect only the nfl_stats table ────────────────────────────────────
metadata = MetaData()
nfl_stats_table = Table(
    "nfl_stats",
    metadata,
    autoload_with=engine
)

# ── 3) declare a composite PK so the mapper knows how to identify rows ────
nfl_stats_table.append_constraint(
        PrimaryKeyConstraint("week", "season", "season_type", "player_display_name", name="pk_nfl_stats")
    )

# ── 4) automap that table into an ORM class ────────────────────────────────
AutomapBase = automap_base(metadata=metadata)
AutomapBase.prepare()

# Now NFLStat is a normal mapped class you can import elsewhere:
NFLStat = AutomapBase.classes.nfl_stats


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

    fpts_dist_name   = Column(String, nullable=True)    # e.g. 'gamma'
    fpts_dist_params = Column(JSON, nullable=True)  # list/tuple of params

    fpts_n          = Column(Integer, nullable=True)
    fpts_mean       = Column(Float, nullable=True)
    fpts_std        = Column(Float, nullable=True)
    fpts_min        = Column(Float, nullable=True)
    fpts_max        = Column(Float, nullable=True)
    fpts_skew       = Column(Float, nullable=True)
    fpts_kurtosis   = Column(Float, nullable=True)
    fpts_p01        = Column(Float, nullable=True)
    fpts_p05        = Column(Float, nullable=True)

    fpts_p10        = Column(Float, nullable=True)
    fpts_p25        = Column(Float, nullable=True)
    fpts_p50        = Column(Float, nullable=True)
    fpts_p75        = Column(Float, nullable=True)
    fpts_p90        = Column(Float, nullable=True)
    fpts_p95        = Column(Float, nullable=True)
    fpts_p99        = Column(Float, nullable=True)    

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

class NFLOdds(NFLBase):
    __tablename__ = "odds"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player          = Column(String,  nullable=False)
    market          = Column(String,  nullable=False)
    sportsbook      = Column(String,  nullable=False)
    side            = Column(String,  nullable=False)
    point           = Column(Float,   nullable=False)
    price           = Column(Integer, nullable=False)
    last_update     = Column(String,  nullable=True)

class Lineup(NFLBase, BaseLineup):
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
            session.query(NFLPlayer)
                   .filter(NFLPlayer.id.in_(self.player_ids_list))
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

    # def get_sim_arrays(self) -> Tuple[List[float],List[float],List[float],List[int]]:
    #     """
    #     Return the (fpts, payout, roi, rank) lists from the first sim result.
    #     """
    #     if not self.simulation_results:
    #         return [], [], [], []
    #     res = self.simulation_results[0]
    #     return res.fpts, res.payout, res.roi, res.rank


class NFLDSTGame(NFLBase):
    __tablename__ = "nfl_dst_game"
    season   = Column(Integer, primary_key=True)
    week     = Column(Integer, primary_key=True)
    team     = Column(String,  primary_key=True)
    opponent = Column(String)
    dst_points = Column(Float)

class LineupSimulationResult(NFLBase):
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

   