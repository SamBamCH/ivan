# src/ivan/sports/mma/models.py
"""
MMA tables. One SQLite file per sport (data/mma/mma_history.sqlite); the
layout follows the other sports: a player pool for the current slate, the
contests we know about and their payout tables, the generated lineup pool,
and one simulation-result row per lineup per simulated contest.

MMA-specific: a fight table. A DraftKings MMA slate is a card of fights and
a lineup may hold at most one fighter per fight, so fights are first-class.
"""
from datetime import datetime
from typing import List

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer, MetaData, String, func)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from ivan.core.lineup import BaseLineup
from ivan.core.player import BasePlayer

mma_metadata = MetaData()
MMABase = declarative_base(metadata=mma_metadata)


class MMAFight(MMABase):
    """One bout on the card. Fight-level odds describe how the fight ends."""
    __tablename__ = "mma_fight"

    id          = Column(String,   primary_key=True)        # "<abbrev>-<abbrev>", the two DK team abbreviations sorted
    fighter1_id = Column(Integer,  nullable=False)
    fighter2_id = Column(Integer,  nullable=False)
    rounds      = Column(Integer,  nullable=False, default=3)   # 3, or 5 for the main-event fights
    gametime    = Column(DateTime, nullable=True)
    u60_odds    = Column(Float,    nullable=True)   # American odds: fight ends inside 60 seconds
    r1_odds     = Column(Float,    nullable=True)   # fight ends in round 1
    itd_odds    = Column(Float,    nullable=True)   # fight ends inside the distance
    created_at  = Column(DateTime, server_default=func.now(), nullable=False)

    def has_odds(self) -> bool:
        return any(v is not None for v in (self.u60_odds, self.r1_odds, self.itd_odds))


class MMAPlayer(MMABase, BasePlayer):
    """A fighter in the current slate's DraftKings pool, with their markets and projection."""
    __tablename__ = "mma_player"

    id          = Column(Integer,  primary_key=True)        # DraftKings draftable id
    name        = Column(String,   nullable=False)
    team        = Column(String,   nullable=False)          # DK's "team" for a fighter: their abbreviation (last name)
    opponent    = Column(String,   nullable=True)
    fight_id    = Column(String,   ForeignKey("mma_fight.id"), nullable=True, index=True)
    gametime    = Column(DateTime, nullable=True)
    salary      = Column(Integer,  nullable=False)
    status      = Column(String,   nullable=True)
    avg_fpts    = Column(Float,    nullable=True)           # DK's AvgPointsPerGame, informational

    # Fighter markets (American odds) and volume projections
    win_odds     = Column(Float, nullable=True)
    itd_odds     = Column(Float, nullable=True)
    r1_win_odds  = Column(Float, nullable=True)
    sig_strikes  = Column(Float, nullable=True)    # expected significant strikes, full fight
    takedowns    = Column(Float, nullable=True)    # expected takedowns, full fight

    # Derived at ingest
    win_prob    = Column(Float, nullable=True)
    projection  = Column(Float, nullable=True)
    created_at  = Column(DateTime, server_default=func.now(), nullable=False)

    def has_odds(self) -> bool:
        return any(v is not None for v in (self.win_odds, self.itd_odds, self.r1_win_odds))


class HistoricalContest(MMABase):
    __tablename__ = "historical_contest_information"

    contest_id     = Column(Integer, primary_key=True)
    draft_group_id = Column(Integer, nullable=False)
    name           = Column(String, nullable=True)
    date           = Column(DateTime, nullable=True)
    entry_fee      = Column(Float, nullable=False)
    contest_type   = Column(String, nullable=True)
    total_entries  = Column(Integer, nullable=False)
    multi_entry    = Column(Integer, nullable=False)


class ContestPayout(MMABase):
    __tablename__ = "payouts"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    contest_id    = Column(Integer, ForeignKey("historical_contest_information.contest_id"), nullable=False, index=True)
    min_position  = Column(Integer, nullable=False)
    max_position  = Column(Integer, nullable=False)
    payout_amount = Column(Float, nullable=False)
    payout_text   = Column(String(32), nullable=True)
    currency      = Column(String(16), default="USD")


class Lineup(MMABase, BaseLineup):
    """A candidate lineup of six fighters, one per fight at most."""
    __tablename__ = "lineup"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_ids_list = Column(JSON, nullable=False)
    created_at      = Column(DateTime, server_default=func.now(), nullable=False)

    simulation_results = relationship("LineupSimulationResult", backref="lineup",
                                      cascade="all, delete-orphan", lazy="select")

    def player_ids(self) -> List[int]:
        return list(self.player_ids_list)


class LineupSimulationResult(MMABase):
    """
    One row per lineup per simulated contest. The JSON columns hold one value
    per simulation; the summary columns are what the chooser reads.
    """
    __tablename__ = "lineup_simulation_results"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    lineup_id    = Column(Integer, ForeignKey("lineup.id"), index=True, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)
    contest_id   = Column(Integer, nullable=True, index=True)     # DK contest id when the contest came from DK
    contest_name = Column(String,  nullable=False, index=True)
    dupes        = Column(Integer, nullable=False, default=1)     # copies of this lineup in the sampled field

    fpts   = Column(JSON, nullable=False)
    payout = Column(JSON, nullable=False)
    roi    = Column(JSON, nullable=False)
    rank   = Column(JSON, nullable=False)

    avg_fpts   = Column(Float, nullable=True)
    avg_payout = Column(Float, nullable=True)
    avg_roi    = Column(Float, nullable=True)
    cash_pct   = Column(Float, nullable=True)     # share of simulations in the money, 0..1
    win_pct    = Column(Float, nullable=True)     # share of simulations finishing first, 0..1
