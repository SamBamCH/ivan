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
from ivan.core.player import BasePlayer
from ivan.core.lineup import BaseLineup


class LOLStat(Base):
    __tablename__ = "lol_stats"

    id                        = Column(Integer, primary_key=True, autoincrement=True)

    game_id                   = Column(String,   nullable=False, index=True)
    data_completeness         = Column(String,   nullable=True)
    url                       = Column(String,   nullable=True)
    league                    = Column(String,   nullable=False)
    year                      = Column(Integer,  nullable=False)
    split                     = Column(String,   nullable=True)
    playoffs                  = Column(Integer,  default=0)

    date                      = Column(String, nullable=False, index=True)
    game                      = Column(Integer,  default=0)
    patch                     = Column(Float,    default=0.0)

    participant_id            = Column(String,   nullable=False, index=True)
    side                      = Column(String,   nullable=True)
    position                  = Column(String,   nullable=True)

    player_name               = Column(String,   nullable=False)
    player_id                 = Column(String,   nullable=False)
    team_name                 = Column(String,   nullable=False)
    team_id                   = Column(String,   nullable=False)

    champion                  = Column(String,   nullable=False)

    ban1                      = Column(String,   nullable=True)
    ban2                      = Column(String,   nullable=True)
    ban3                      = Column(String,   nullable=True)
    ban4                      = Column(String,   nullable=True)
    ban5                      = Column(String,   nullable=True)

    pick1                     = Column(String,   nullable=True)
    pick2                     = Column(String,   nullable=True)
    pick3                     = Column(String,   nullable=True)
    pick4                     = Column(String,   nullable=True)
    pick5                     = Column(String,   nullable=True)

    game_length               = Column(Integer,  default=0)
    result                    = Column(Integer,  default=0)

    kills                     = Column(Integer,  default=0)
    deaths                    = Column(Integer,  default=0)
    assists                   = Column(Integer,  default=0)
    team_kills                = Column(Integer,  default=0)
    team_deaths               = Column(Integer,  default=0)

    double_kills              = Column(Integer,  default=0)
    triple_kills              = Column(Integer,  default=0)
    quadrakills               = Column(Integer,  default=0)
    pentakills                = Column(Integer,  default=0)

    first_blood               = Column(Integer,  default=0)
    first_blood_kill          = Column(Integer,  default=0)
    first_blood_assist        = Column(Integer,  default=0)
    first_blood_victim        = Column(String,   nullable=True)

    team_kpm                  = Column(Float,    nullable=True)
    ckpm                      = Column(Float,    nullable=True)

    first_dragon              = Column(Integer,  default=0)
    dragons                   = Column(Integer,  default=0)
    opp_dragons               = Column(Integer,  default=0)
    elemental_drakes          = Column(Integer,  default=0)
    opp_elemental_drakes      = Column(Integer,  default=0)

    infernals                 = Column(Integer,  default=0)
    mountains                 = Column(Integer,  default=0)
    clouds                    = Column(Integer,  default=0)
    oceans                    = Column(Integer,  default=0)
    chemtechs                 = Column(Integer,  default=0)
    hextechs                  = Column(Integer,  default=0)
    dragons_type_unknown      = Column(Integer,  default=0)

    elders                    = Column(Integer,  default=0)
    opp_elders                = Column(Integer,  default=0)

    first_herald              = Column(Integer,  default=0)
    heralds                   = Column(Integer,  default=0)
    opp_heralds               = Column(Integer,  default=0)

    void_grubs                = Column(Integer,  default=0)
    opp_void_grubs            = Column(Integer,  default=0)

    first_baron               = Column(Integer,  default=0)
    barons                    = Column(Integer,  default=0)
    opp_barons                = Column(Integer,  default=0)

    first_tower               = Column(Integer,  default=0)
    towers                    = Column(Integer,  default=0)
    opp_towers                = Column(Integer,  default=0)

    first_mid_tower           = Column(Integer,  default=0)
    first_to_three_towers     = Column(Integer,  default=0)

    turret_plates             = Column(Integer,  default=0)
    opp_turret_plates         = Column(Integer,  default=0)

    inhibitors                = Column(Integer,  default=0)
    opp_inhibitors            = Column(Integer,  default=0)

    damage_to_champions       = Column(Integer,  default=0)
    dpm                       = Column(Float,    nullable=True)
    damage_share              = Column(Float,    nullable=True)
    damage_taken_per_minute   = Column(Float,    nullable=True)
    damage_mitigated_per_minute = Column(Float,  nullable=True)

    wards_placed              = Column(Integer,  default=0)
    wpm                       = Column(Float,    nullable=True)
    wards_killed              = Column(Integer,  default=0)
    wcpm                      = Column(Float,    nullable=True)
    control_wards_bought      = Column(Integer,  default=0)

    vision_score              = Column(Integer,  default=0)
    vspm                      = Column(Float,    nullable=True)

    total_gold                = Column(Integer,  default=0)
    earned_gold               = Column(Integer,  default=0)
    earned_gpm                = Column(Float,    nullable=True)
    earned_gold_share         = Column(Float,    nullable=True)
    gold_spent                = Column(Integer,  default=0)

    gspd                      = Column(Float,    nullable=True)
    gpr                       = Column(Float,    nullable=True)

    total_cs                  = Column(Integer,  default=0)
    minion_kills              = Column(Integer,  default=0)
    monster_kills             = Column(Integer,  default=0)
    monster_kills_own_jungle  = Column(Integer,  default=0)
    monster_kills_enemy_jungle= Column(Integer,  default=0)
    cspm                      = Column(Float,    nullable=True)

    gold_at_10                = Column(Integer,  default=0)
    xp_at_10                  = Column(Integer,  default=0)
    cs_at_10                  = Column(Integer,  default=0)
    opp_gold_at_10            = Column(Integer,  default=0)
    opp_xp_at_10              = Column(Integer,  default=0)
    opp_cs_at_10              = Column(Integer,  default=0)
    gold_diff_at_10           = Column(Integer,  default=0)
    xp_diff_at_10             = Column(Integer,  default=0)
    cs_diff_at_10             = Column(Integer,  default=0)
    kills_at_10               = Column(Integer,  default=0)
    assists_at_10             = Column(Integer,  default=0)
    deaths_at_10              = Column(Integer,  default=0)
    opp_kills_at_10           = Column(Integer,  default=0)
    opp_assists_at_10         = Column(Integer,  default=0)
    opp_deaths_at_10          = Column(Integer,  default=0)

    gold_at_15                = Column(Integer,  default=0)
    xp_at_15                  = Column(Integer,  default=0)
    cs_at_15                  = Column(Integer,  default=0)
    opp_gold_at_15            = Column(Integer,  default=0)
    opp_xp_at_15              = Column(Integer,  default=0)
    opp_cs_at_15              = Column(Integer,  default=0)
    gold_diff_at_15           = Column(Integer,  default=0)
    xp_diff_at_15             = Column(Integer,  default=0)
    cs_diff_at_15             = Column(Integer,  default=0)
    kills_at_15               = Column(Integer,  default=0)
    assists_at_15             = Column(Integer,  default=0)
    deaths_at_15              = Column(Integer,  default=0)
    opp_kills_at_15           = Column(Integer,  default=0)
    opp_assists_at_15         = Column(Integer,  default=0)
    opp_deaths_at_15          = Column(Integer,  default=0)

    gold_at_20                = Column(Integer,  default=0)
    xp_at_20                  = Column(Integer,  default=0)
    cs_at_20                  = Column(Integer,  default=0)
    opp_gold_at_20            = Column(Integer,  default=0)
    opp_xp_at_20              = Column(Integer,  default=0)
    opp_cs_at_20              = Column(Integer,  default=0)
    gold_diff_at_20           = Column(Integer,  default=0)
    xp_diff_at_20             = Column(Integer,  default=0)
    cs_diff_at_20             = Column(Integer,  default=0)
    kills_at_20               = Column(Integer,  default=0)
    assists_at_20             = Column(Integer,  default=0)
    deaths_at_20              = Column(Integer,  default=0)
    opp_kills_at_20           = Column(Integer,  default=0)
    opp_assists_at_20         = Column(Integer,  default=0)
    opp_deaths_at_20          = Column(Integer,  default=0)

    gold_at_25                = Column(Integer,  default=0)
    xp_at_25                  = Column(Integer,  default=0)
    cs_at_25                  = Column(Integer,  default=0)
    opp_gold_at_25            = Column(Integer,  default=0)
    opp_xp_at_25              = Column(Integer,  default=0)
    opp_cs_at_25              = Column(Integer,  default=0)
    gold_diff_at_25           = Column(Integer,  default=0)
    xp_diff_at_25             = Column(Integer,  default=0)
    cs_diff_at_25             = Column(Integer,  default=0)
    kills_at_25               = Column(Integer,  default=0)
    assists_at_25             = Column(Integer,  default=0)
    deaths_at_25              = Column(Integer,  default=0)
    opp_kills_at_25           = Column(Integer,  default=0)
    opp_assists_at_25         = Column(Integer,  default=0)
    opp_deaths_at_25          = Column(Integer,  default=0)

