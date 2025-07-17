# src/ivan/sports/mlb/models.py
from datetime import datetime
from typing import Optional, List, Any, Tuple
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
    UniqueConstraint
)
from dataclasses import dataclass
from ivan.database.session import Base
from ivan.sports.mlb.player_calculations import get_adjusted_nb_params
from ivan.core.player import BasePlayer
from ivan.core.lineup import BaseLineup


class MLBStat(Base):
    __tablename__ = "mlb_stats"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    date           = Column(DateTime,    nullable=False, index=True)
    player_id      = Column(String,  nullable=False, index=True)
    name           = Column(String,  nullable=False)
    team           = Column(String,  nullable=False)
    position       = Column(String,  nullable=False)

    batting_order  = Column(Integer, default=0)

    pa             = Column(Integer, default=0)
    ab             = Column(Integer, default=0)
    runs           = Column(Integer, default=0)
    hits           = Column(Integer, default=0)
    doubles        = Column(Integer, default=0)
    triples        = Column(Integer, default=0)
    hr             = Column(Integer, default=0)
    rbi            = Column(Integer, default=0)
    bb             = Column(Integer, default=0)
    sb             = Column(Integer, default=0)
    hbp            = Column(Integer, default=0)

    dk_fpts        = Column(Float, nullable=False, default=0.0)
    fd_fpts        = Column(Float, nullable=False, default=0.0)

    obp            = Column(Float, nullable=True)
    slg            = Column(Float, nullable=True)



class BatterProjection(Base):
    __tablename__ = 'mlb_batter_projection'
    id                = Column(Integer, primary_key=True, autoincrement=True)
    player_id         = Column(Integer, nullable=False, index=True)
    name              = Column(String)
    hand              = Column(String)
    team              = Column(String)
    opp_tm            = Column(String)
    salary            = Column(Integer)
    pos               = Column(String)   # e.g. "1B/OF"
    lp                = Column(Integer)
    ol                = Column(Boolean)
    od                = Column(Boolean)
    ipl               = Column(Boolean)
    error             = Column(String)
    doubles_2h        = Column(String)
    pitcher           = Column(String)
    catcher           = Column(String)
    umpire            = Column(String)
    park              = Column(String)
    roof              = Column(String)
    gvf               = Column(String)
    platoon           = Column(Boolean)
    hfa               = Column(Boolean)
    ph_pct            = Column(String)
    cold              = Column(Boolean)
    tilt_bias         = Column(String)
    fpts              = Column(Float)
    fpts_per_dollar   = Column(Float)
    r                 = Column(Float)
    sb                = Column(Float)
    cs                = Column(Float)
    sh                = Column(Float)
    sf                = Column(Float)
    gidp              = Column(Float)
    hr_dependence     = Column(String)
    ph_fpts_pct       = Column(String)
    slate             = Column(String)
    floor             = Column(Float)
    ceiling           = Column(Float)
    pown              = Column(Float)
    pa                = Column(Float)
    pavssp            = Column(Float)
    pavssp_pct        = Column(String)
    ba                = Column(Float)
    obp               = Column(Float)
    slg               = Column(Float)
    iso               = Column(Float)
    woba              = Column(Float)
    cbwoba            = Column(Float)
    xwoba_cs          = Column(Float)
    xwoba_ls          = Column(Float)
    k                 = Column(Float)
    bb                = Column(Float)
    ibb               = Column(Float)
    hbp               = Column(Float)
    h                 = Column(Float)
    single            = Column(Float)
    double            = Column(Float)
    triple            = Column(Float)
    hr                = Column(Float)
    tb                = Column(Float)
    rbi               = Column(Float)
    partnerid         = Column(Float)
    # ...other columns as needed...
    created_at        = Column(DateTime, nullable=True)

class PitcherProjection(Base):
    __tablename__ = 'mlb_pitcher_projection'
    id                = Column(Integer, primary_key=True, autoincrement=True)
    player_id         = Column(Integer, nullable=False, index=True)
    player            = Column(String)
    salary            = Column(Integer)
    pos               = Column(String)
    team              = Column(String)
    opp               = Column(String)
    schedule_id       = Column(Integer)
    slate             = Column(String)
    tm                = Column(String)
    opp_tm            = Column(String)
    hand              = Column(String)
    ol                = Column(Boolean)
    od                = Column(Boolean)
    pcc               = Column(Boolean)
    error             = Column(String)
    doubles_2h        = Column(String)
    bpc               = Column(Float)
    ppc               = Column(Float)
    mpc               = Column(Float)
    opener            = Column(Boolean)
    catcher           = Column(String)
    umpire            = Column(String)
    park              = Column(String)
    roof              = Column(String)
    platoon           = Column(String)
    split             = Column(String)
    gvf               = Column(String)
    hfa               = Column(Boolean)
    dh                = Column(Boolean)
    familiarity       = Column(String)
    tilt_bias         = Column(String)
    spin_ls           = Column(Integer)
    spin_cs           = Column(Integer)
    spin_rec          = Column(Integer)
    spin_change       = Column(Integer)
    fpts              = Column(Float)
    fpts_per_dollar   = Column(Float)
    ip                = Column(Float)
    outs              = Column(Float)
    era               = Column(Float)
    cnera             = Column(Float)
    w                 = Column(Float)
    l                 = Column(Float)
    qs                = Column(Boolean)
    cg                = Column(Boolean)
    cgsh              = Column(Boolean)
    tbf               = Column(Float)
    k                 = Column(Float)
    bb                = Column(Float)
    ibb               = Column(Float)
    hbp               = Column(Float)
    h                 = Column(Float)
    hr                = Column(Float)
    tb                = Column(Float)
    sh                = Column(Float)
    sf                = Column(Float)
    gidp              = Column(Float)
    sb                = Column(Float)
    cs                = Column(Float)
    er                = Column(Float)
    floor             = Column(Float)
    ceiling           = Column(Float)
    pown              = Column(Float)
    rgid              = Column(Float)
    partnerid         = Column(Float)
    ownership         = Column(Float)
    created_at        = Column(DateTime, nullable=True)

class HeroProjection(Base):
    __tablename__ = 'mlb_hero_projection'
    id                = Column(Integer, primary_key=True, autoincrement=True)
    source_id         = Column(Integer)  # original hero file ID
    name              = Column(String)
    pos               = Column(String)
    team              = Column(String)
    opp               = Column(String)
    salary            = Column(Integer)
    stdv              = Column(Float)
    value             = Column(Float)
    hand              = Column(String)
    order_rank        = Column(Integer)
    runs              = Column(Float)
    era               = Column(Float)
    k9                = Column(Float)
    whip              = Column(Float)
    woba              = Column(Float)
    iso               = Column(Float)
    floor             = Column(Float)
    ceiling           = Column(Float)
    rnkgpp            = Column(Integer)
    rnkcash           = Column(Integer)
    top1              = Column(Float)
    top5              = Column(Float)
    top10             = Column(Float)
    two_x             = Column(Float)
    two_5_x           = Column(Float)
    three_x           = Column(Float)
    three_5_x         = Column(Float)
    four_x            = Column(Float)
    four_5_x          = Column(Float)
    five_x            = Column(Float)
    five_5_x          = Column(Float)
    optimal           = Column(Float)
    lev               = Column(Float)
    hero_own          = Column(Float)
    hero_proj         = Column(Float)
    user_own          = Column(Float)
    user_proj         = Column(Float)
    agg_proj          = Column(Float)
    rdm_pct           = Column(Float)
    min_exp           = Column(Float)
    max_exp           = Column(Float)
    build_exp         = Column(Float)
    created_at        = Column(DateTime, nullable=True)

class MLBPlayer(Base, BasePlayer):
    __tablename__ = 'mlb_player'
    #from dk salary file
    id              = Column(Integer,   primary_key=True) 
    name            = Column(String,    nullable=False)
    team            = Column(String,    nullable=False)
    gametime        = Column(DateTime,  nullable=True)
    salary          = Column(Integer,   nullable=False)
    position        = Column(String,    nullable=False)

    #from RotoGrinders 'THE BAT X' projection files
    opponent        = Column(String,    nullable=False)
    fpts_projection = Column(Float,     nullable=False)
    ownership       = Column(Float,     nullable=True)
    lp              = Column(Integer, nullable=True)
    ipl             = Column(Boolean, nullable=True)
    doubles_2h      = Column(String,  nullable=True)

    #from enrichments
    stdv            = Column(Float,     nullable=True)
    ops             = Column(Float,     nullable=True)
    adj_r           = Column(Float,     nullable=False)
    adj_p           = Column(Float,     nullable=False)
    
    created_at      = Column(DateTime, nullable=False)

    def calculate_ops(self, session, limit: int = 10) -> float:
        """
        Query the last `limit` games from mlb_stats,
        and return average OPS by reading obp and slg columns directly.
        """

        recent = (
            session.query(MLBStat)
                   .filter(MLBStat.name == self.name.strip().lower())
                   .order_by(MLBStat.date.desc())
                   .limit(limit)
                   .all()
        )

        if not recent:
            return 0.0

        # Read obp/slg from the columns (they were pre‐computed on ingest)
        ops_vals = []
        for s in recent:
            obp = s.obp if getattr(s, 'obp', None) is not None else 0.0
            slg = s.slg if getattr(s, 'slg', None) is not None else 0.0
            ops_vals.append(obp + slg)

        return sum(ops_vals) / len(ops_vals)
    
    def enrich(self, session, historical_model) -> tuple[float,float]:
        """
        -builds self.ops via calculate_ops()
        -uses get_adjusted_nb_params to shift this NB mean to fpts_projection
        """

        self.ops = self.calculate_ops(session)
        r, p_new = get_adjusted_nb_params(self.ops, self.fpts_projection, historical_model)
        self.adj_r = r
        self.adj_p = p_new
        return r, p_new
    
@dataclass
class FieldPlayer:
    id: int
    name: str
    salary: int
    position: str
    team: str
    opponent: str
    fpts_projection: float
    stdv: float
    adj_r: float
    adj_p: float

    def __eq__(self, other):
        return isinstance(other, FieldPlayer) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

class Lineup(Base, BaseLineup):
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
            session.query(MLBPlayer)
                   .filter(MLBPlayer.id.in_(self.player_ids_list))
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

    def get_sim_arrays(self) -> Tuple[List[float],List[float],List[float],List[int]]:
        """
        Return the (fpts, payout, roi, rank) lists from the first sim result.
        """
        if not self.simulation_results:
            return [], [], [], []
        res = self.simulation_results[0]
        return res.fpts, res.payout, res.roi, res.rank

class LineupSimulationResult(Base):
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
    


class HistoricalContest(Base):
    __tablename__ = "historical_contest_information"

    contest_id          = Column(Integer, primary_key=True)
    draft_group_id      = Column(Integer, nullable=False)
    name                = Column(String, nullable=True)
    date                = Column(DateTime, nullable=True)
    entry_fee           = Column(Float, nullable=False)
    contest_type        = Column(String, nullable=True)
    total_entries       = Column(Integer, nullable=False)
    multi_entry         = Column(Integer, nullable=False)

class ContestPayout(Base):
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

    # (Optional) backref if you want to navigate from HistoricalContest → payouts
    # contest = relationship("HistoricalContest", back_populates="payouts")

class GeneralizedContestMetrics(Base):
    __tablename__ = "generalized_contest_metrics"

    # One row per HistoricalContest.contest_id
    contest_id   = Column(Integer,
                          ForeignKey("historical_contest_information.contest_id"),
                          primary_key=True,
                          index=True,
                          nullable=False)
    # JSON blob storing { "P": [...], "O": [...], "S": [...], "E": [...] }
    metrics_json = Column(JSON, nullable=False)

    created_at   = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<GeneralizedContestMetrics(contest_id={self.contest_id})>"
    
class ActualContestEntry(Base):
    __tablename__ = "actual_contest_entry"
    contest_id      = Column(
        Integer,
        ForeignKey("historical_contest_information.contest_id"),
        nullable=False,
        index=True
    )
    entry_id        = Column(Integer, primary_key=True)
    lineup_id       = Column(Integer, nullable=False)
    username        = Column(String, nullable=True)
    rank            = Column(Integer, nullable=False)
    fantasy_points  = Column(Float, nullable=True)
    payout          = Column(Float, nullable=False)

class SlateKDEModel(Base):
    __tablename__ = 'slate_kde_model'

    # Composite PK for one model per slate-feature combination
    contest_date    = Column(DateTime, primary_key=True) #date of the contest
    num_games       = Column(Integer, primary_key=True) #total games on the slate
    total_entries   = Column(Integer, primary_key=True) #total entries in contest
    max_entries     = Column(Integer, primary_key=True) #total entries one user can enter (i.e. 1, 20, 150)
    entry_fee       = Column(Float, primary_key=True) #entry fee of the contest
    top_owned       = Column(Float, primary_key=True) #top owned player on the slate, expressed as a %
    top_5_owned     = Column(Float, primary_key=True) #average ownership of top 5 highest owned players on the slate, expressed as a %
    top_10_owned    = Column(Float, primary_key=True) #average ownership of top 10 highest owned players on the slate, expressed as a %
    first_pct       = Column(Float, primary_key=True) #percent of the total prize pool paid out to first place
    top5_pct        = Column(Float, primary_key=True) #percent of the total prize pool paid out to the first 5 places. 

    # KDE parameters
    bw_own          = Column(Float,   nullable=False)
    bw_proj         = Column(Float,   nullable=False)
    max_proj        = Column(Float,   nullable=False)

    # Optional: coarse histograms (20-bin)
    hist_own_bins    = Column(JSON,    nullable=True)
    hist_own_counts  = Column(JSON,    nullable=True)
    hist_proj_bins   = Column(JSON,    nullable=True)
    hist_proj_counts = Column(JSON,    nullable=True)
    hist_stack_types = Column(JSON,    nullable=False)
    hist_stack_probs = Column(JSON,    nullable=False)
    

