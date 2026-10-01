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
from ivan.core.player import BasePlayer
from ivan.core.lineup import BaseLineup

nfl_metadata = MetaData()
NFLBase = declarative_base(metadata=nfl_metadata)


class NFLStat(NFLBase):
    __tablename__ = 'nfl_stats'

    id                          = Column(Integer, primary_key=True, autoincrement=True)
    
    # Player info
    player_id                   = Column(String, nullable=True)
    player_name                 = Column(String, nullable=True)
    player_display_name         = Column(String, nullable=True)
    position                    = Column(String, nullable=True)
    position_group              = Column(String, nullable=True)
    headshot_url                = Column(String, nullable=True)
    
    # Game info
    season                      = Column(Integer, nullable=True)
    week                        = Column(Integer, nullable=True)
    season_type                 = Column(String, nullable=True)
    team                        = Column(String, nullable=True)
    opponent_team               = Column(String, nullable=True)
    
    # Passing stats
    completions                 = Column(Float, nullable=True)
    attempts                    = Column(Float, nullable=True)
    passing_yards               = Column(Float, nullable=True)
    passing_tds                 = Column(Float, nullable=True)
    passing_interceptions       = Column(Float, nullable=True)
    sacks_suffered              = Column(Float, nullable=True)
    sack_yards_lost             = Column(Float, nullable=True)
    sack_fumbles                = Column(Float, nullable=True)
    sack_fumbles_lost           = Column(Float, nullable=True)
    passing_air_yards           = Column(Float, nullable=True)
    passing_yards_after_catch   = Column(Float, nullable=True)
    passing_first_downs         = Column(Float, nullable=True)
    passing_epa                 = Column(Float, nullable=True)
    passing_cpoe                = Column(Float, nullable=True)
    passing_2pt_conversions     = Column(Float, nullable=True)
    pacr                        = Column(Float, nullable=True)
    
    # Rushing stats
    carries                     = Column(Float, nullable=True)
    rushing_yards               = Column(Float, nullable=True)
    rushing_tds                 = Column(Float, nullable=True)
    rushing_fumbles             = Column(Float, nullable=True)
    rushing_fumbles_lost        = Column(Float, nullable=True)
    rushing_first_downs         = Column(Float, nullable=True)
    rushing_epa                 = Column(Float, nullable=True)
    rushing_2pt_conversions     = Column(Float, nullable=True)
    
    # Receiving stats
    receptions                  = Column(Float, nullable=True)
    targets                     = Column(Float, nullable=True)
    receiving_yards             = Column(Float, nullable=True)
    receiving_tds               = Column(Float, nullable=True)
    receiving_fumbles           = Column(Float, nullable=True)
    receiving_fumbles_lost      = Column(Float, nullable=True)
    receiving_air_yards         = Column(Float, nullable=True)
    receiving_yards_after_catch = Column(Float, nullable=True)
    receiving_first_downs       = Column(Float, nullable=True)
    receiving_epa               = Column(Float, nullable=True)
    receiving_2pt_conversions   = Column(Float, nullable=True)
    racr                        = Column(Float, nullable=True)
    target_share                = Column(Float, nullable=True)
    air_yards_share             = Column(Float, nullable=True)
    wopr                        = Column(Float, nullable=True)
    
    # Special teams stats
    special_teams_tds           = Column(Float, nullable=True)
    
    # Defensive stats
    def_tackles_solo            = Column(Float, nullable=True)
    def_tackles_with_assist     = Column(Float, nullable=True)
    def_tackle_assists          = Column(Float, nullable=True)
    def_tackles_for_loss        = Column(Float, nullable=True)
    def_tackles_for_loss_yards  = Column(Float, nullable=True)
    def_fumbles_forced          = Column(Float, nullable=True)
    def_sacks                   = Column(Float, nullable=True)
    def_sack_yards              = Column(Float, nullable=True)
    def_qb_hits                 = Column(Float, nullable=True)
    def_interceptions           = Column(Float, nullable=True)
    def_interception_yards      = Column(Float, nullable=True)
    def_pass_defended           = Column(Float, nullable=True)
    def_tds                     = Column(Float, nullable=True)
    def_fumbles                 = Column(Float, nullable=True)
    def_safeties                = Column(Float, nullable=True)
    
    # Miscellaneous stats
    misc_yards                  = Column(Float, nullable=True)
    fumble_recovery_own         = Column(Float, nullable=True)
    fumble_recovery_yards_own   = Column(Float, nullable=True)
    fumble_recovery_opp         = Column(Float, nullable=True)
    fumble_recovery_yards_opp   = Column(Float, nullable=True)
    fumble_recovery_tds         = Column(Float, nullable=True)
    penalties                   = Column(Float, nullable=True)
    penalty_yards               = Column(Float, nullable=True)
    
    # Return stats
    punt_returns                = Column(Float, nullable=True)
    punt_return_yards           = Column(Float, nullable=True)
    kickoff_returns             = Column(Float, nullable=True)
    kickoff_return_yards        = Column(Float, nullable=True)
    
    # Field goal stats
    fg_made                     = Column(Float, nullable=True)
    fg_att                      = Column(Float, nullable=True)
    fg_missed                   = Column(Float, nullable=True)
    fg_blocked                  = Column(Float, nullable=True)
    fg_long                     = Column(Float, nullable=True)
    fg_pct                      = Column(Float, nullable=True)
    fg_made_0_19                = Column(Float, nullable=True)
    fg_made_20_29               = Column(Float, nullable=True)
    fg_made_30_39               = Column(Float, nullable=True)
    fg_made_40_49               = Column(Float, nullable=True)
    fg_made_50_59               = Column(Float, nullable=True)
    fg_made_60_                 = Column(Float, nullable=True)
    fg_missed_0_19              = Column(Float, nullable=True)
    fg_missed_20_29             = Column(Float, nullable=True)
    fg_missed_30_39             = Column(Float, nullable=True)
    fg_missed_40_49             = Column(Float, nullable=True)
    fg_missed_50_59             = Column(Float, nullable=True)
    fg_missed_60_               = Column(Float, nullable=True)
    fg_made_list                = Column(JSON, nullable=True)
    fg_missed_list              = Column(JSON, nullable=True)
    fg_blocked_list             = Column(JSON, nullable=True)
    fg_made_distance            = Column(JSON, nullable=True)
    fg_missed_distance          = Column(JSON, nullable=True)
    fg_blocked_distance         = Column(JSON, nullable=True)
    
    # PAT stats
    pat_made                    = Column(Float, nullable=True)
    pat_att                     = Column(Float, nullable=True)
    pat_missed                  = Column(Float, nullable=True)
    pat_blocked                 = Column(Float, nullable=True)
    pat_pct                     = Column(Float, nullable=True)
    
    # Game-winning FG stats
    gwfg_made                   = Column(Float, nullable=True)
    gwfg_att                    = Column(Float, nullable=True)
    gwfg_missed                 = Column(Float, nullable=True)
    gwfg_blocked                = Column(Float, nullable=True)
    gwfg_distance               = Column(Float, nullable=True)
    
    # Kicker points (simple calculation)
    kicker_pts                  = Column(Float, nullable=True)  # FG made × 3 + PAT made × 1
    
    # Fantasy points
    fantasy_points              = Column(Float, nullable=True)
    fantasy_points_ppr          = Column(Float, nullable=True)



class NFLPlayer(NFLBase, BasePlayer):
    __tablename__ = 'nfl_player'

    id              = Column(Integer,   primary_key=True)
    roster_slot_id  = Column(Integer,   nullable=False)  # Composite PK with id
    name            = Column(String,    nullable=False)
    team            = Column(String,    nullable=False)
    opponent        = Column(String,    nullable=True)
    position        = Column(String,    nullable=False)
    slot_label      = Column(String,    nullable=True)  # e.g. 'RB1', 'WR2', 'FLEX', 'DST'
    gametime        = Column(DateTime,  nullable=False) 
    salary          = Column(Integer,   nullable=False)
    status          = Column(String,    nullable=True)

    fpts_dist_name   = Column(String, nullable=True)    # e.g. 'gamma'
    fpts_dist_params = Column(JSON, nullable=True)  # list/tuple of params

    fpts_n          = Column(Integer, nullable=True)
    fpts_mean       = Column(Float, nullable=True)
    fpts_std        = Column(Float, nullable=True)
    fpts_std_model  = Column(Float, nullable=True)   # spread of the fitted distribution before the predictive-sd calibration
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
    """DraftKings DST points per team-game; rebuilt by dst_scoring.build_dst_game_table."""
    __tablename__ = "nfl_dst_game"
    season   = Column(Integer, primary_key=True)
    week     = Column(Integer, primary_key=True)
    team     = Column(String,  primary_key=True)
    season_type = Column(String, nullable=True)
    opponent = Column(String)
    dst_points = Column(Float)
    points_allowed = Column(Float, nullable=True)
    sacks     = Column(Float, nullable=True)
    takeaways = Column(Float, nullable=True)
    blocks    = Column(Float, nullable=True)
    def_tds   = Column(Float, nullable=True)
    safeties  = Column(Float, nullable=True)

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

class ContestSimReport(NFLBase):
    """
    One row per simulated contest: everything simulate_one_contest knew about
    the run (field sample, payout scaling, invariants, correlation check) as a
    JSON document, so a person can verify the sim after the fact. Per-slate
    working data, like lineup_simulation_results; never synced between machines.
    """
    __tablename__ = "contest_sim_report"

    contest_id   = Column(Integer, primary_key=True)
    contest_name = Column(String, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)
    report       = Column(JSON, nullable=False)


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
    Permanent storage for Showdown player data.
    One row per (contest_id, player_id) where player_id is the DK roster slot ID.
    Replaces ProjectionSnapshot after contest is processed.
    """
    __tablename__ = "showdown_player"
    
    contest_id = Column(Integer, ForeignKey("historical_contest_information.contest_id"), 
                       primary_key=True)
    player_id = Column(Integer, primary_key=True)  # DK player ID for this roster slot
    roster_slot_id = Column(Integer, nullable=False)  # 511=CPT, 512=FLEX
    
    # Player metadata (from ProjectionSnapshot)
    salary = Column(Integer, nullable=False)
    position = Column(String, nullable=False)
    team = Column(String, nullable=False)
    
    # Actual ownership from contest results (for model fitting)
    ownership_actual = Column(Float, nullable=False)  # 0-100, calculated from lineups
    
    # Our projections (from ProjectionSnapshot)
    ownership_proj = Column(Float, nullable=True)
    fpts_proj = Column(Float, nullable=True)
    
    # Actual performance
    fpts_actual = Column(Float, nullable=True)


class ShowdownLineup(NFLBase):
    """
    Minimal lineup-level data for Showdown contests.
    Stores only player IDs and duplication count.
    All other features calculated at runtime by joining to ShowdownPlayer/ProjectionSnapshot.
    """
    __tablename__ = "showdown_lineup"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    contest_id = Column(Integer, ForeignKey("historical_contest_information.contest_id"),
                       nullable=False, index=True)
    
    # Lineup composition (DK player IDs for roster slots)
    cpt_player_id = Column(Integer, nullable=False)  # CPT roster slot player ID
    flex_player_ids = Column(JSON, nullable=False)  # List[int] of 5 FLEX roster slot IDs
    
    # Salary
    salary_remaining = Column(Integer, nullable=False)  # 50000 - total
    
    # Target variable
    dupe_count = Column(Integer, nullable=False, default=1)
    
    __table_args__ = (
        UniqueConstraint('contest_id', 'cpt_player_id', 'flex_player_ids',
                        name='uq_showdown_lineup'),
    )


class ShowdownContestMetadata(NFLBase):
    """
    Contest-level duplication distribution stats.
    Target variables for phase 1 of showdown dupe model (predict contest quantiles).
    """
    __tablename__ = "showdown_contest_metadata"
    
    contest_id = Column(Integer, ForeignKey("historical_contest_information.contest_id"),
                       primary_key=True)
    
    # Duplication distribution (target variables for phase 1 model)
    unique_lineup_count = Column(Integer, nullable=False)
    max_dupe_count = Column(Integer, nullable=False)
    median_dupe_count = Column(Integer, nullable=False)
    p10_dupe_count = Column(Integer, nullable=True)
    p25_dupe_count = Column(Integer, nullable=True)
    p75_dupe_count = Column(Integer, nullable=True)
    p90_dupe_count = Column(Integer, nullable=True)

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
    fpts_std_model = Column(Float, nullable=True)   # model spread before calibration (null in older snapshots)

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
    


   