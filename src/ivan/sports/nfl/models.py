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
    slot_label      = Column(String,    nullable=True)  # e.g. 'RB1', 'WR2', 'FLEX', 'DST'
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
    ownership_proj  = Column(Float, nullable=True, default=0.0)

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

    simulation_results = relationship(
        "LineupSimulationResult",
        backref="lineup",
        cascade="all, delete-orphan",
        lazy="select"
    )

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

class NFLTeamWeekPositionalFPTS(NFLBase):
    __tablename__ = "nfl_team_week_positional_fpts"

    season        = Column(Integer, primary_key=True)
    season_type   = Column(String,  primary_key=True)   # e.g., 'REG', 'POST', 'PRE'
    week          = Column(Integer, primary_key=True)
    team          = Column(String,  primary_key=True)   # team code (same as 'recent_team' normalization)
    opponent      = Column(String,  nullable=True)

    # Own team
    QB   = Column(Float, nullable=True)
    RB1  = Column(Float, nullable=True)
    RB2  = Column(Float, nullable=True)
    WR1  = Column(Float, nullable=True)
    WR2  = Column(Float, nullable=True)
    WR3  = Column(Float, nullable=True)
    TE1  = Column(Float, nullable=True)
    TE2  = Column(Float, nullable=True)
    K    = Column(Float, nullable=True)   # placeholder for future backfill
    DST  = Column(Float, nullable=True)   # placeholder for future backfill

    # Opponent
    OPPQB   = Column(Float, nullable=True)
    OPPRB1  = Column(Float, nullable=True)
    OPPRB2  = Column(Float, nullable=True)
    OPPWR1  = Column(Float, nullable=True)
    OPPWR2  = Column(Float, nullable=True)
    OPPWR3  = Column(Float, nullable=True)
    OPPTE1  = Column(Float, nullable=True)
    OPPTE2  = Column(Float, nullable=True)
    OPPK    = Column(Float, nullable=True)  # placeholder
    OPPDST  = Column(Float, nullable=True)  # placeholder

class NFLGameOdds(NFLBase):
    __tablename__ = "game_odds"

    id               = Column(Integer, primary_key=True, autoincrement=True)

    # Event-level identifiers
    game_id          = Column(String,  nullable=False, index=True)   # odds API 'id'
    sport_key        = Column(String,  nullable=False, index=True)
    commence_time    = Column(DateTime, nullable=False)

    # Teams (as full names from the API) + handy abbreviations
    home_team        = Column(String,  nullable=False)
    away_team        = Column(String,  nullable=False)
    home_abbr        = Column(String,  nullable=True)
    away_abbr        = Column(String,  nullable=True)

    # Book & market
    bookmaker        = Column(String,  nullable=False, index=True)   # e.g., 'draftkings'
    market           = Column(String,  nullable=False, index=True)   # 'h2h' | 'spreads' | 'totals'
    last_update      = Column(String,  nullable=True)                # ISO string from API

    # Outcome (normalized)
    outcome_name     = Column(String,  nullable=False)   # team name for h2h/spreads, 'Over'/'Under' for totals
    side             = Column(String,  nullable=True)    # optional helper (e.g., 'home','away','over','under')
    point            = Column(Float,   nullable=True)    # spread or total line (NULL for h2h)
    price            = Column(Integer, nullable=False)   # american odds

    # Helpful uniqueness to avoid dup rows on re-pulls at the same timestamp
    __table_args__ = (
        UniqueConstraint(
            "game_id", "bookmaker", "market", "outcome_name", "point", "last_update",
            name="uq_game_odds_snapshot"
        ),
    )

class ShowdownPlayer(NFLBase):
    """
    One row per player per contest.
    Stores player-level information for building correlation features.
    """
    __tablename__ = "showdown_player"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    contest_id = Column(Integer, ForeignKey("historical_contest_information.contest_id"), 
                       nullable=False, index=True)
    
    # Player identification
    player_name = Column(String, nullable=False)
    normalized_name = Column(String, nullable=False, index=True)
    team = Column(String, nullable=True)  # Team abbreviation
    position = Column(String, nullable=True)  # QB, RB, WR, TE, DST
    roster_slot = Column(String, nullable=False)  # CPT or FLEX
    
    # DraftKings data
    salary = Column(Integer, nullable=False)
    ownership_pct = Column(Float, nullable=False)  # 0-100
    fpts_actual = Column(Float, nullable=True)  # Actual points scored
    
    # For correlation analysis
    game_id = Column(String, nullable=True)  # Links players in same game
    opponent = Column(String, nullable=True)

    ownership_proj = Column(Float, nullable=True)
    
    __table_args__ = (
        UniqueConstraint('contest_id', 'player_name', 'roster_slot', 
                        name='uq_showdown_player'),
    )


class ShowdownLineup(NFLBase):
    """
    One row per unique lineup in a contest.
    This is your core training data for the 5-dimensional model.
    """
    __tablename__ = "showdown_lineup"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    contest_id = Column(Integer, ForeignKey("historical_contest_information.contest_id"),
                       nullable=False, index=True)
    
    # Lineup composition
    captain_name = Column(String, nullable=False)
    flex_players = Column(JSON, nullable=False)  # List of player names
    
    # Core features for your 5-dimensional model
    
    # 1. Player correlations (quantified)
    stack_pattern = Column(String, nullable=True)  # e.g., "QB-WR-WR", "RB-DST"
    same_team_count = Column(Integer, nullable=True)  # Players from same team
    game_stack_type = Column(String, nullable=True)  # e.g., "5-1", "4-2" (home-away split)
    qb_wr_stack = Column(Boolean, default=False)  # Has QB+WR from same team
    rb_dst_negative = Column(Boolean, default=False)  # RB facing own DST
    correlation_score = Column(Float, nullable=True)  # Custom correlation metric
    
    # 2. Captain ownership
    cpt_ownership_pct = Column(Float, nullable=False)  # 0-100
    cpt_ownership_percentile = Column(Float, nullable=True)  # 0-1 within contest
    
    # 3. Lineup projection (normalized 0-1)
    projection_raw = Column(Float, nullable=True)  # Raw projection sum
    projection_normalized = Column(Float, nullable=True)  # 0-1 scaled within contest
    projection_percentile = Column(Float, nullable=True)  # Percentile within contest
    
    # 4. Lineup ownership (normalized 0-1)
    ownership_sum = Column(Float, nullable=False)  # Sum of all player ownership
    ownership_normalized = Column(Float, nullable=True)  # 0-1 scaled within contest
    ownership_percentile = Column(Float, nullable=True)  # Percentile within contest
    
    # 5. Salary remaining
    total_salary = Column(Integer, nullable=False)
    salary_remaining = Column(Integer, nullable=False)  # 50000 - total_salary
    salary_remaining_pct = Column(Float, nullable=False)  # Percent of cap unused
    
    # Target variables
    duplicate_count = Column(Integer, nullable=False, default=1)  # How many times this lineup appeared
    unique_rank = Column(Integer, nullable=True)  # Rank among unique lineups
    best_score = Column(Float, nullable=True)  # Best score this lineup achieved
    best_rank = Column(Integer, nullable=True)  # Best rank this lineup achieved

    cpt_ownership_actual = Column(Float, nullable=True)
    ownership_sum_actual = Column(Float, nullable=True)
    
    # Metadata
    first_seen_rank = Column(Integer, nullable=True)  # First occurrence rank
    
    __table_args__ = (
        UniqueConstraint('contest_id', 'captain_name', 'flex_players',
                        name='uq_showdown_lineup'),
    )


class ShowdownContestMetadata(NFLBase):
    """
    Enhanced contest-level metadata for normalization and modeling context.
    One row per contest.
    """
    __tablename__ = "showdown_contest_metadata"
    
    contest_id = Column(Integer, ForeignKey("historical_contest_information.contest_id"),
                       primary_key=True)
    
    # Contest info (denormalized for convenience)
    date = Column(DateTime, nullable=False)
    total_entries = Column(Integer, nullable=False)
    entry_fee = Column(Float, nullable=False)
    
    # Game information
    home_team = Column(String, nullable=True)
    away_team = Column(String, nullable=True)
    game_time = Column(DateTime, nullable=True)
    
    # Normalization bounds for the 5 dimensions
    min_projection = Column(Float, nullable=True)
    max_projection = Column(Float, nullable=True)
    median_projection = Column(Float, nullable=True)
    
    min_ownership = Column(Float, nullable=True)
    max_ownership = Column(Float, nullable=True)
    median_ownership = Column(Float, nullable=True)
    
    min_cpt_ownership = Column(Float, nullable=True)
    max_cpt_ownership = Column(Float, nullable=True)
    
    # Aggregate patterns (for model training context)
    unique_lineup_count = Column(Integer, nullable=False)
    max_duplicate_count = Column(Integer, nullable=False)
    median_duplicate_count = Column(Integer, nullable=False)
    
    # Popular correlation patterns in this contest
    top_stack_patterns = Column(JSON, nullable=True)  # {"QB-WR-WR": 234, ...}
    
    # Distribution summaries
    ownership_gini = Column(Float, nullable=True)  # Ownership concentration
    projection_std = Column(Float, nullable=True)
    
    processed_at = Column(DateTime, default=datetime.utcnow)

class ProjectionSnapshot(NFLBase):
    """
    Frozen snapshot of player projections at a specific time.
    Simple design: just timestamp + player + projection.
    """
    __tablename__ = "projection_snapshot"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # When this snapshot was taken (UTC)
    snapshot_date = Column(DateTime, nullable=False, index=True)
    
    # Player identification
    player_name = Column(String, nullable=False)
    normalized_name = Column(String, nullable=False, index=True)
    team = Column(String, nullable=False)
    opponent = Column(String, nullable=True)
    position = Column(String, nullable=False)
    
    # DraftKings data (from NFLPlayer at snapshot time)
    salary = Column(Integer, nullable=False)
    roster_slot_id = Column(Integer, nullable=False)
    
    # Projection (from your model)
    fpts_mean = Column(Float, nullable=False)
    fpts_median = Column(Float, nullable=True)
    fpts_std = Column(Float, nullable=True)

    ownership_proj = Column(Float, nullable=False)

class ClassicContestMetadata(NFLBase):
    """
    Contest-level metadata and distribution parameters for Classic contests.
    Stores normalized histogram data for projection and ownership distributions.
    Used to train models that predict field distributions based on contest parameters.
    """
    __tablename__ = "classic_contest_metadata"

    contest_id = Column(Integer, ForeignKey("historical_contest_information.contest_id"),
                       primary_key=True)

    # Contest info
    date = Column(DateTime, nullable=False)
    total_entries = Column(Integer, nullable=False)
    entry_fee = Column(Float, nullable=False)
    num_games = Column(Integer, nullable=True)  # Number of games on slate

    # Ownership concentration metric
    ownership_gini = Column(Float, nullable=True)  # Gini coefficient [0-1]

    # Normalization bounds (for converting raw values to 0-1)
    min_projection = Column(Float, nullable=True)
    max_projection = Column(Float, nullable=True)
    min_ownership = Column(Float, nullable=True)
    max_ownership = Column(Float, nullable=True)

    # Histogram bins (BINS=20 fixed-width bins [0-1])
    # Each entry is a count, normalized to sum to 1.0
    hist_proj_counts = Column(JSON, nullable=True)  # 20-bin normalized projection distribution
    hist_own_counts = Column(JSON, nullable=True)   # 20-bin normalized ownership distribution

    # Distribution statistics
    projection_mean = Column(Float, nullable=True)
    projection_std = Column(Float, nullable=True)
    ownership_mean = Column(Float, nullable=True)
    ownership_std = Column(Float, nullable=True)

    # Field statistics
    salary_remaining_mean = Column(Float, nullable=True)
    top_stack_patterns = Column(JSON, nullable=True)

    processed_at = Column(DateTime, default=datetime.utcnow)
    


   