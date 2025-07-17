

import os
import json
import tempfile
import gdown
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# import your model
from ivan.sports.lol.models import LOLStat
from ivan.database.session import get_session, init_db
from ivan.core.utils import PROJECT_ROOT

# 1. CONFIGURATION
DATABASE_URL = PROJECT_ROOT / 'data' / 'lol' / 'lol_history_sql'

# 2. DOWNLOAD CSV
def fetch_csv(destination: str):
    FILE_ID = "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc"
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    gdown.download(url, destination, quiet=False)

# 3. LOAD INTO PANDAS
def load_dataframe(csv_path: str) -> pd.DataFrame:
    # 1) read raw CSV
    df = pd.read_csv(csv_path)

    # 2) drop rows that are entirely blank
    df = df.dropna(how='all')

    # 3) normalize column names to snake_case
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r'[^0-9a-z]+', '_', regex=True)
          .str.strip('_')
    )

    # 4) infer better dtypes
    df = df.infer_objects()

    # 5) convert any object-dtype series that look numeric
    for col in df.select_dtypes(include=['object']):
        coerced = pd.to_numeric(df[col], errors='ignore')
        if coerced.dtype != object:
            df[col] = coerced

    # 6) fill missing values
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)

    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].fillna('')

    return df

# 4. UPSERT TO DATABASE
def upsert_stats(df: pd.DataFrame, session):
    for _, row in df.iterrows():
        # upsert key: game_id + participant_id
        existing = session.query(LOLStat).filter_by(
            game_id=row["gameid"],
            participant_id=row["participantid"]
        ).one_or_none()

        stat = existing or LOLStat(
            game_id=row["gameid"],
            participant_id=row["participantid"]
        )

        # --- 1. top‐level metadata
        stat.data_completeness         = row.get("data_completeness")
        stat.url                       = row.get("url")
        stat.league                    = row["league"]
        stat.year                      = int(row["year"])
        stat.split                     = row.get("split")
        stat.playoffs                  = int(row.get("playoffs", 0))
        stat.date                      = row["date"]            # already parsed to datetime
        stat.game                      = int(row.get("game", 0))
        stat.patch                     = float(row.get("patch", 0.0))

        # --- 2. participant & team info
        stat.side                      = row.get("side")
        stat.position                  = row.get("position")
        stat.player_name               = row["playername"]
        stat.player_id                 = row["playerid"]
        stat.team_name                 = row["teamname"]
        stat.team_id                   = row["teamid"]
        stat.champion                  = row.get("champion")

        # --- 3. bans & picks
        stat.ban1                      = row.get("ban1")
        stat.ban2                      = row.get("ban2")
        stat.ban3                      = row.get("ban3")
        stat.ban4                      = row.get("ban4")
        stat.ban5                      = row.get("ban5")
        stat.pick1                     = row.get("pick1")
        stat.pick2                     = row.get("pick2")
        stat.pick3                     = row.get("pick3")
        stat.pick4                     = row.get("pick4")
        stat.pick5                     = row.get("pick5")

        # --- 4. game outcome & core stats
        stat.game_length               = int(row.get("game_length", 0))
        stat.result                    = int(row.get("result", 0))
        stat.kills                     = int(row.get("kills", 0))
        stat.deaths                    = int(row.get("deaths", 0))
        stat.assists                   = int(row.get("assists", 0))
        stat.team_kills                = int(row.get("team_kills", 0))
        stat.team_deaths               = int(row.get("team_deaths", 0))

        # --- 5. multikills
        stat.double_kills              = int(row.get("double_kills", 0))
        stat.triple_kills              = int(row.get("triple_kills", 0))
        stat.quadrakills               = int(row.get("quadrakills", 0))
        stat.pentakills                = int(row.get("pentakills", 0))

        # --- 6. first blood
        stat.first_blood               = int(row.get("first_blood", 0))
        stat.first_blood_kill          = int(row.get("first_blood_kill", 0))
        stat.first_blood_assist        = int(row.get("first_blood_assist", 0))
        stat.first_blood_victim        = row.get("first_blood_victim")

        # --- 7. pacing metrics
        stat.team_kpm                  = float(row.get("team_kpm", 0.0))
        stat.ckpm                      = float(row.get("ckpm", 0.0))

        # --- 8. dragon stats
        stat.first_dragon              = int(row.get("first_dragon", 0))
        stat.dragons                   = int(row.get("dragons", 0))
        stat.opp_dragons               = int(row.get("opp_dragons", 0))
        stat.elemental_drakes          = int(row.get("elemental_drakes", 0))
        stat.opp_elemental_drakes      = int(row.get("opp_elemental_drakes", 0))
        stat.infernals                 = int(row.get("infernals", 0))
        stat.mountains                 = int(row.get("mountains", 0))
        stat.clouds                    = int(row.get("clouds", 0))
        stat.oceans                    = int(row.get("oceans", 0))
        stat.chemtechs                 = int(row.get("chemtechs", 0))
        stat.hextechs                  = int(row.get("hextechs", 0))
        stat.dragons_type_unknown      = int(row.get("dragons_type_unknown", 0))

        # --- 9. elder dragon
        stat.elders                    = int(row.get("elders", 0))
        stat.opp_elders                = int(row.get("opp_elders", 0))

        # --- 10. herald stats
        stat.first_herald              = int(row.get("first_herald", 0))
        stat.heralds                   = int(row.get("heralds", 0))
        stat.opp_heralds               = int(row.get("opp_heralds", 0))

        # --- 11. void grubs
        stat.void_grubs                = int(row.get("void_grubs", 0))
        stat.opp_void_grubs            = int(row.get("opp_void_grubs", 0))

        # --- 12. baron stats
        stat.first_baron               = int(row.get("first_baron", 0))
        stat.barons                    = int(row.get("barons", 0))
        stat.opp_barons                = int(row.get("opp_barons", 0))

        # --- 13. tower stats
        stat.first_tower               = int(row.get("first_tower", 0))
        stat.towers                    = int(row.get("towers", 0))
        stat.opp_towers                = int(row.get("opp_towers", 0))
        stat.first_mid_tower           = int(row.get("first_mid_tower", 0))
        stat.first_to_three_towers     = int(row.get("first_to_three_towers", 0))
        stat.turret_plates             = int(row.get("turret_plates", 0))
        stat.opp_turret_plates         = int(row.get("opp_turret_plates", 0))
        stat.inhibitors                = int(row.get("inhibitors", 0))
        stat.opp_inhibitors            = int(row.get("opp_inhibitors", 0))

        # --- 14. damage metrics
        stat.damage_to_champions       = int(row.get("damage_to_champions", 0))
        stat.dpm                       = float(row.get("dpm", 0.0))
        stat.damage_share              = float(row.get("damage_share", 0.0))
        stat.damage_taken_per_minute   = float(row.get("damage_taken_per_minute", 0.0))
        stat.damage_mitigated_per_minute = float(row.get("damage_mitigated_per_minute", 0.0))

        # --- 15. vision & wards
        stat.wards_placed              = int(row.get("wards_placed", 0))
        stat.wpm                       = float(row.get("wpm", 0.0))
        stat.wards_killed              = int(row.get("wards_killed", 0))
        stat.wcpm                      = float(row.get("wcpm", 0.0))
        stat.control_wards_bought      = int(row.get("control_wards_bought", 0))
        stat.vision_score              = int(row.get("vision_score", 0))
        stat.vspm                      = float(row.get("vspm", 0.0))

        # --- 16. gold metrics
        stat.total_gold                = int(row.get("total_gold", 0))
        stat.earned_gold               = int(row.get("earned_gold", 0))
        stat.earned_gpm                = float(row.get("earned_gpm", 0.0))
        stat.earned_gold_share         = float(row.get("earned_gold_share", 0.0))
        stat.gold_spent                = int(row.get("gold_spent", 0))
        stat.gspd                      = float(row.get("gspd", 0.0))
        stat.gpr                       = float(row.get("gpr", 0.0))

        # --- 17. creep score & jungle
        stat.total_cs                  = int(row.get("total_cs", 0))
        stat.minion_kills              = int(row.get("minion_kills", 0))
        stat.monster_kills             = int(row.get("monster_kills", 0))
        stat.monster_kills_own_jungle  = int(row.get("monster_kills_own_jungle", 0))
        stat.monster_kills_enemy_jungle= int(row.get("monster_kills_enemy_jungle", 0))
        stat.cspm                      = float(row.get("cspm", 0.0))

        # --- 18. 10-minute snapshot
        stat.gold_at_10                = int(row.get("gold_at_10", 0))
        stat.xp_at_10                  = int(row.get("xp_at_10", 0))
        stat.cs_at_10                  = int(row.get("cs_at_10", 0))
        stat.opp_gold_at_10            = int(row.get("opp_gold_at_10", 0))
        stat.opp_xp_at_10              = int(row.get("opp_xp_at_10", 0))
        stat.opp_cs_at_10              = int(row.get("opp_cs_at_10", 0))
        stat.gold_diff_at_10           = int(row.get("gold_diff_at_10", 0))
        stat.xp_diff_at_10             = int(row.get("xp_diff_at_10", 0))
        stat.cs_diff_at_10             = int(row.get("cs_diff_at_10", 0))
        stat.kills_at_10               = int(row.get("kills_at_10", 0))
        stat.assists_at_10             = int(row.get("assists_at_10", 0))
        stat.deaths_at_10              = int(row.get("deaths_at_10", 0))
        stat.opp_kills_at_10           = int(row.get("opp_kills_at_10", 0))
        stat.opp_assists_at_10         = int(row.get("opp_assists_at_10", 0))
        stat.opp_deaths_at_10          = int(row.get("opp_deaths_at_10", 0))

        # --- 19. 15-minute snapshot
        stat.gold_at_15                = int(row.get("gold_at_15", 0))
        stat.xp_at_15                  = int(row.get("xp_at_15", 0))
        stat.cs_at_15                  = int(row.get("cs_at_15", 0))
        stat.opp_gold_at_15            = int(row.get("opp_gold_at_15", 0))
        stat.opp_xp_at_15              = int(row.get("opp_xp_at_15", 0))
        stat.opp_cs_at_15              = int(row.get("opp_cs_at_15", 0))
        stat.gold_diff_at_15           = int(row.get("gold_diff_at_15", 0))
        stat.xp_diff_at_15             = int(row.get("xp_diff_at_15", 0))
        stat.cs_diff_at_15             = int(row.get("cs_diff_at_15", 0))
        stat.kills_at_15               = int(row.get("kills_at_15", 0))
        stat.assists_at_15             = int(row.get("assists_at_15", 0))
        stat.deaths_at_15              = int(row.get("deaths_at_15", 0))
        stat.opp_kills_at_15           = int(row.get("opp_kills_at_15", 0))
        stat.opp_assists_at_15         = int(row.get("opp_assists_at_15", 0))
        stat.opp_deaths_at_15          = int(row.get("opp_deaths_at_15", 0))

        # --- 20. 20-minute snapshot
        stat.gold_at_20                = int(row.get("gold_at_20", 0))
        stat.xp_at_20                  = int(row.get("xp_at_20", 0))
        stat.cs_at_20                  = int(row.get("cs_at_20", 0))
        stat.opp_gold_at_20            = int(row.get("opp_gold_at_20", 0))
        stat.opp_xp_at_20              = int(row.get("opp_xp_at_20", 0))
        stat.opp_cs_at_20              = int(row.get("opp_cs_at_20", 0))
        stat.gold_diff_at_20           = int(row.get("gold_diff_at_20", 0))
        stat.xp_diff_at_20             = int(row.get("xp_diff_at_20", 0))
        stat.cs_diff_at_20             = int(row.get("cs_diff_at_20", 0))
        stat.kills_at_20               = int(row.get("kills_at_20", 0))
        stat.assists_at_20             = int(row.get("assists_at_20", 0))
        stat.deaths_at_20              = int(row.get("deaths_at_20", 0))
        stat.opp_kills_at_20           = int(row.get("opp_kills_at_20", 0))
        stat.opp_assists_at_20         = int(row.get("opp_assists_at_20", 0))
        stat.opp_deaths_at_20          = int(row.get("opp_deaths_at_20", 0))

        # --- 21. 25-minute snapshot
        stat.gold_at_25                = int(row.get("gold_at_25", 0))
        stat.xp_at_25                  = int(row.get("xp_at_25", 0))
        stat.cs_at_25                  = int(row.get("cs_at_25", 0))
        stat.opp_gold_at_25            = int(row.get("opp_gold_at_25", 0))
        stat.opp_xp_at_25              = int(row.get("opp_xp_at_25", 0))
        stat.opp_cs_at_25              = int(row.get("opp_cs_at_25", 0))
        stat.gold_diff_at_25           = int(row.get("gold_diff_at_25", 0))
        stat.xp_diff_at_25             = int(row.get("xp_diff_at_25", 0))
        stat.cs_diff_at_25             = int(row.get("cs_diff_at_25", 0))
        stat.kills_at_25               = int(row.get("kills_at_25", 0))
        stat.assists_at_25             = int(row.get("assists_at_25", 0))
        stat.deaths_at_25              = int(row.get("deaths_at_25", 0))
        stat.opp_kills_at_25           = int(row.get("opp_kills_at_25", 0))
        stat.opp_assists_at_25         = int(row.get("opp_assists_at_25", 0))
        stat.opp_deaths_at_25          = int(row.get("opp_deaths_at_25", 0))

        session.add(stat)

    session.commit()

def main():
# 1) init DB/session
    cfg_path = PROJECT_ROOT / 'config' / 'lol.json'
    cfg      = json.load(open(cfg_path, encoding='utf-8'))
    engine   = init_db('lol', cfg.get('database_url'))
    session     = get_session(engine)
    # temp file
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "lol_stats.csv")
        fetch_csv(csv_path)
        df = load_dataframe(csv_path)
        upsert_stats(df, session)
    session.close()

if __name__ == "__main__":
    main()
