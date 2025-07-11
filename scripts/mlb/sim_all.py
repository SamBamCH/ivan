import os
import json
import argparse

import pandas as pd

from ivan.database.session import init_db, get_session, Base
from ivan.sports.mlb.simulation import simulate_one_contest
from sqlalchemy import text


def parse_args():
    p = argparse.ArgumentParser(
        description="Run simulations for each contest listed in entries_csv"
    )
    p.add_argument(
        "--num_sims", "-n",
        type=int,
        required=True,
        help="Number of Monte Carlo simulations per lineup (e.g. 100, 500, 1000)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    here   = os.path.dirname(__file__)
    cfg_fp = os.path.abspath(os.path.join(here, "../../config/mlb.json"))
    cfg    = json.load(open(cfg_fp, encoding="utf-8"))

    # 1) Initialize DB + session
    engine  = init_db("mlb", cfg.get("database_url"))
    Base.metadata.create_all(engine)
    session = get_session(engine)

    # 1a) clear existing sims in the DB
    try:
        print("Clearing lineup_simulation_results table...")
        session.execute(text("DELETE FROM lineup_simulation_results"))
        session.commit()
        print("✅ lineup_simulation_results has been emptied.")
    except Exception as e:
        session.rollback()
        print(f"❌ Error clearing table: {e}")
    finally:
        session.close()


    # 2) Read entries_csv with both "Contest ID" and "Contest Name"
    df = pd.read_csv(
        cfg["paths"]["entries_csv"],
        engine="python",
        usecols=["Contest ID", "Contest Name"],
        dtype={"Contest ID": str, "Contest Name": str}
    )

    # Trim whitespace
    df["Contest ID"]   = df["Contest ID"].str.strip()
    df["Contest Name"] = df["Contest Name"].str.strip()

    # Drop rows missing either column
    df = df.dropna(subset=["Contest ID", "Contest Name"]).copy()

    # Convert Contest ID → int
    df["Contest ID"] = df["Contest ID"].astype(int)

    # 3) Only keep unique (Contest ID, Contest Name) pairs
    df = df.sort_values("Contest ID")
    unique_df = df.drop_duplicates(subset=["Contest Name"], keep="last")

    # 4) Loop over each unique contest
    for _, row in unique_df.iterrows():
        cid  = int(row["Contest ID"])
        name = row["Contest Name"]

        print(f"→ Starting contest {cid} ('{name}')")
        try:
            simulate_one_contest(
                session       = session,
                contest_id    = cid,
                contest_name  = name,
                num_sims      = args.num_sims,
                config_path   = cfg_fp
            )
        except Exception as e:
            print(f"⚠️  Skipped contest {cid} ('{name}') due to error: {e}\n")


if __name__ == "__main__":
    main()
