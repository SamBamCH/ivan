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
    game_date = Column(Date, nullable=True, index=True)
    season = Column(String, nullable=True)  # e.g., "2023-24"

    # Player identifiers
    player_name = Column(String, nullable=False) #normalized on input
    position = Column(String, nullable=True)  # G, F, or C from START_POSITION

    # Team info
    team_abbreviation = Column(String, nullable=False)

    # Box score stats
    minutes = Column(Float)
    points = Column(Integer)
    rebounds = Column(Integer)
    rebounds_off = Column(Integer)
    rebounds_def = Column(Integer)
    assists = Column(Integer)
    steals = Column(Integer)
    blocks = Column(Integer)
    turnovers = Column(Integer)

    # Shooting stats
    fgm = Column(Integer)  # field goals made
    fga = Column(Integer)  # field goals attempted
    fg_pct = Column(Float)
    fg3m = Column(Integer)  # 3-pointers made
    fg3a = Column(Integer)  # 3-pointers attempted
    fg3_pct = Column(Float)
    ftm = Column(Integer)  # free throws made
    fta = Column(Integer)  # free throws attempted
    ft_pct = Column(Float)

    # Fantasy points
    dk_fpts = Column(Float, default=0.0)

    __table_args__ = (
        UniqueConstraint("game_id", "player_name", name="uq_nba_stats_game_player"),
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
    # Projected ownership percentage (0-100): the public field's exposure (public_field.py)
    ownership       = Column(Float,     nullable=True, default=0.0)
    pub_fpts        = Column(Float,     nullable=True)      # the public's projection (Stokastic replica from the props); the field is built on it, never on ours
    pub_updated_at  = Column(DateTime,  nullable=True)
    # Minutes behind fpts_mean (minutes-anchored projection); None for CSV projections
    minutes_proj    = Column(Float,     nullable=True)
    # Marginal distribution as 101 quantiles (JSON list); the correlated sampler
    # maps uniforms through it. None -> Normal(fpts_mean, fpts_std).
    fpts_quantiles  = Column(JSON,      nullable=True)
    # DraftKings' own AvgPointsPerGame from the draftables payload (free yardstick)
    dk_avg_fpts     = Column(Float,     nullable=True)
    draft_group_id  = Column(Integer,   nullable=True)
    # the team minutes model's inputs (projection/team_minutes.py), written with the projection
    minutes_role    = Column(String,    nullable=True)   # star | starter | rotation | deep
    p_out           = Column(Float,     nullable=True)   # scratch probability in the draw
    rates           = Column(JSON,      nullable=True)   # stat -> per-minute rate after the usage bump
    minutes_p10     = Column(Float,     nullable=True)
    minutes_p90     = Column(Float,     nullable=True)
    rotation_cells  = Column(JSON,      nullable=True)   # his row of the rotation grid (48 x 0/1), when the rotation model projected him
    boundary_sd     = Column(Float,     nullable=True)   # spread of his substitution times, minutes


class NBAProjectionCapture(NBABase):
    """
    One row per player per capture: what we (or a yardstick source) projected
    at that moment, with the lock time, so accuracy can be read by offset
    before lock. Appended by every ingest/capture tick; never updated.
    """
    __tablename__ = "nba_projection_capture"

    id              = Column(Integer,  primary_key=True, autoincrement=True)
    captured_at     = Column(DateTime, nullable=False, index=True)
    draft_group_id  = Column(Integer,  nullable=True, index=True)
    lock            = Column(DateTime, nullable=True)
    source          = Column(String,   nullable=False, default="ours")   # ours | stokastic | dk_avg
    player_id       = Column(Integer,  nullable=True)
    player_name     = Column(String,   nullable=False, index=True)
    team            = Column(String,   nullable=True)
    opponent        = Column(String,   nullable=True)
    gametime        = Column(DateTime, nullable=True)
    salary          = Column(Integer,  nullable=True)
    status          = Column(String,   nullable=True)
    starter         = Column(String,   nullable=True)    # confirmed | expected | bench | unknown
    minutes_proj    = Column(Float,    nullable=True)
    fpts_mean       = Column(Float,    nullable=True)
    fpts_std        = Column(Float,    nullable=True)
    priced          = Column(Boolean,  nullable=True)
    pinned          = Column(Boolean,  nullable=True)
    # why the minutes are what they are (added 2026-10-21; older captures have NULLs)
    baseline_min    = Column(Float,    nullable=True)    # history baseline before the team allocation
    hist_team       = Column(String,   nullable=True)    # team of the last game in his history (a change = new role)
    hist_games      = Column(Integer,  nullable=True)    # games behind the history (0 = salary prior)
    hist_min        = Column(Float,    nullable=True)    # trailing minutes x play rate
    start_ewm       = Column(Float,    nullable=True)    # trailing share of games started
    play_ewm        = Column(Float,    nullable=True)    # trailing share of dressed games played
    fpm             = Column(Float,    nullable=True)    # projected fantasy points per minute (after the usage bump)
    usage_mult      = Column(Float,    nullable=True)
    market_fpts     = Column(Float,    nullable=True)    # the books' implied fantasy points when priced
    minutes_model   = Column(String,   nullable=True)    # independent | team
    minutes_p10     = Column(Float,    nullable=True)    # the minutes draw's band (team model)
    minutes_p90     = Column(Float,    nullable=True)
    minutes_role    = Column(String,   nullable=True)
    p_out           = Column(Float,    nullable=True)


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

class NBAMarketRate(NBABase):
    """Per-minute rate implied by the props at fetch time (see projection/market.py)."""
    __tablename__ = "nba_market_rate"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_name     = Column(String,  nullable=False, index=True)   # normalized
    stat            = Column(String,  nullable=False)               # points, rebounds, ...
    market_mean     = Column(Float,   nullable=False)               # consensus mean for the game
    rate            = Column(Float,   nullable=False)               # market_mean / basis_minutes
    basis_minutes   = Column(Float,   nullable=False)               # projected minutes when priced
    books           = Column(Integer, nullable=True)
    line            = Column(Float,   nullable=True)
    fetched_at      = Column(DateTime, nullable=True)


class LineupPool(NBABase):
    """
    The player pool a slate's lineups were generated from: id, name, team, position, salary per player, written once per
    generate run. Lineups are exported from this, so a slate switch (the laptop's feed replacing nba_player under the
    server) never makes them unreadable (2026-10-24).
    """
    __tablename__ = "lineup_pool"
    __table_args__ = (UniqueConstraint("draft_group_id", "player_id", name="uq_lineup_pool"),)

    id              = Column(Integer, primary_key=True, autoincrement=True)
    draft_group_id  = Column(Integer, nullable=False, index=True)
    player_id       = Column(Integer, nullable=False)
    name            = Column(String,  nullable=False)
    team            = Column(String,  nullable=True)
    position        = Column(String,  nullable=True)
    salary          = Column(Integer, nullable=True)
    lock            = Column(DateTime, nullable=True)
    written_at      = Column(DateTime, nullable=False, default=datetime.now)


class Lineup(NBABase):
    __tablename__ = "lineup"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    player_ids_list  = Column(JSON,    nullable=False)
    created_at       = Column(DateTime, server_default=func.now(), nullable=False)
    draft_group_id   = Column(Integer, nullable=True, index=True)      # the slate it was generated for (older rows: inferred, see nba_desk.backfill_lineup_slates)

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

class ContestSimReport(NBABase):
    """
    One row per simulated contest: everything simulate_one_contest knew about
    the run (field sample, payout scaling, invariants) as a JSON document, so
    a person can verify the sim after the fact. Per-slate working data, like
    lineup_simulation_results.
    """
    __tablename__ = "contest_sim_report"

    contest_id   = Column(Integer, primary_key=True)
    contest_name = Column(String, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)
    report       = Column(JSON, nullable=False)


class NBAGameOdds(NBABase):
    """
    Game lines (moneyline, spread, total) per book and outcome, the shape of
    ivan.sports.common.odds.parse_game_odds. Replaced per game on every fetch;
    the history lives in the raw payloads (odds/lines) and the archive captures.
    projection/lines.py reduces them to one consensus row per game.
    """
    __tablename__ = "game_odds"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    game_id       = Column(String,  nullable=False, index=True)
    sport_key     = Column(String,  nullable=True)
    commence_time = Column(DateTime, nullable=True)
    home_team     = Column(String,  nullable=True)
    away_team     = Column(String,  nullable=True)
    home_abbr     = Column(String,  nullable=True, index=True)
    away_abbr     = Column(String,  nullable=True, index=True)
    bookmaker     = Column(String,  nullable=True)
    market        = Column(String,  nullable=True)
    last_update   = Column(String,  nullable=True)
    outcome_name  = Column(String,  nullable=True)
    side          = Column(String,  nullable=True)
    point         = Column(Float,   nullable=True)
    price         = Column(Integer, nullable=True)
    fetched_at    = Column(DateTime, nullable=True)


class ProjectionSnapshot(NBABase):
    """
    Frozen snapshot of player projections at a specific time.
    """
    __tablename__ = "projection_snapshot"

    id              = Column(Integer,   primary_key=True)
    snapshot_date   = Column(DateTime,  nullable=False, index=True)
    player_name     = Column(String,    nullable=False, index=True)
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
    ownership       = Column(Float,     nullable=True, default=0.0)  # Projected ownership %

class ClassicContestMetadata(NBABase):
    """
    Contest-level metadata and distribution parameters for Classic contests.
    Stores normalized histogram data for projection and ownership distributions.
    Used to train models that predict field distributions based on contest parameters.
    """
    __tablename__ = "classic_contest_metadata"

    contest_id = Column(Integer, ForeignKey("historical_contest_information.contest_id"),
                    primary_key=True)
    
    max_projection  = Column(Float, nullable=False)
    max_ownership   = Column(Float, nullable=False) 

    # Histogram bins (BINS=20 fixed-width bins [0-1])
    # Each entry is a count, normalized to sum to 1.0
    hist_proj_counts = Column(JSON, nullable=True)  # 20-bin normalized projection distribution
    hist_own_counts = Column(JSON, nullable=True)   # 20-bin normalized ownership distribution

class StartingLineups(NBABase):
    """
    projected starting lineups via scrape of rotowire nba lineups url
    """
    __tablename__ = "starting_lineups"

    player_name = Column(String, nullable=False, primary_key=True)
    team        = Column(String, nullable=False)
    position    = Column(String, nullable=False)
    confirmed   = Column(Boolean, nullable=False, default=False)  # True if lineup is confirmed, False if expected
    fetched_at  = Column(DateTime, nullable=True)                 # when the Rotowire page was read

