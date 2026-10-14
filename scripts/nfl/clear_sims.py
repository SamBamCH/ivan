#!/usr/bin/env python
from sqlalchemy import text
from ivan.database.session import init_db, get_session

def main():

    # initialize engine + session factory
    engine = init_db('nfl')
    sess   = get_session(engine)

    try:
        print("Clearing lineup_simulation_results table...")
        sess.execute(text("DELETE FROM lineup_simulation_results"))
        sess.commit()
        print("✅ lineup_simulation_results has been emptied.")
        sess.execute(text("DELETE FROM contest_sim_report"))
        sess.commit()
        print("✅ contest_sim_report has been emptied.")
    except Exception as e:
        sess.rollback()
        print(f"❌ Error clearing table: {e}")
    finally:
        sess.close()

if __name__ == "__main__":
    main()
