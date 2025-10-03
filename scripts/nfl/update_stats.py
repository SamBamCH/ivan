#!/usr/bin/env python3
import json
import nfl_data_py as nfl

from ivan.database.session import init_db
from sqlalchemy import MetaData, Table, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ivan.core.utils import PROJECT_ROOT

# ── CONFIG ────────────────────────────────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "config" / "nfl.json"
YEARS       = [2023, 2024, 2025]
TABLE_NAME  = "nfl_stats"
# ────────────────────────────────────────────────────────────────────────────

def main():
    # 1) load config & init DB
    cfg    = json.load(open(CONFIG_PATH, encoding="utf-8"))
    engine = init_db("nfl", cfg["database_url"])
    Session = sessionmaker(bind=engine)
    session = Session()

    # 2) fetch & write table
    df = nfl.import_weekly_data(years=YEARS, downcast=True)
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
    print(f"✅ Table `{TABLE_NAME}` created/replaced in DB")

    # 3) reflect the table
    metadata = MetaData()
    nfl_stats = Table(
        TABLE_NAME,
        metadata,
        autoload_with=engine
    )

    # 4) inject a composite PK on (game_id, player_display_name)
    #    adjust column names if yours differ!
    nfl_stats.append_constraint(
        PrimaryKeyConstraint("week", "season", "season_type", "player_display_name", name="pk_nfl_stats")
    )

    # 5) map it into an ORM class
    OrmBase = declarative_base(metadata=metadata)
    class NFLStat(OrmBase):
        __table__ = nfl_stats

    session.close()

if __name__ == "__main__":
    main()
