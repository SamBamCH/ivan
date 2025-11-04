#!/usr/bin/env python
import os
import json
from sqlalchemy import text
from ivan.database.session import init_db, get_session, Base
from ivan.core.utils import PROJECT_ROOT

def main():
    # assume this script lives in <project_root>/mlb/
    CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'nba.json')

    # load your DB connection info
    with open(CONFIG_PATH, encoding='utf-8') as f:
        cfg = json.load(f)

    # initialize engine + session factory
    cfg    = json.load(open(CONFIG_PATH, encoding='utf-8'))
    engine = init_db('nba', cfg.get('database_url'))
    sess   = get_session(engine)

    try:
        print("Clearing lineup_simulation_results table...")
        sess.execute(text("DELETE FROM lineup_simulation_results"))
        sess.commit()
        print("✅ lineup_simulation_results has been emptied.")
    except Exception as e:
        sess.rollback()
        print(f"❌ Error clearing table: {e}")
    finally:
        sess.close()

if __name__ == "__main__":
    main()