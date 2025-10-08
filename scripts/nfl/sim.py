#!/usr/bin/env python3
# scripts/nfl/sim_all.py
import os, json, argparse
import pandas as pd

from ivan.database.session import init_db, get_session, Base
from ivan.sports.nfl.simulate_contest import simulate_one_contest  

def parse_args():
    p = argparse.ArgumentParser(description="Run NFL simulations for each contest in entries_csv")
    p.add_argument("--config", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/nfl.json")))
    p.add_argument("-n", type=int, default=1000, help="Number of simulation samples per contest (default: 1000)")
    p.add_argument("-c", type=int, default=10000, help="Field cap (default: 100)")
    p.add_argument("-d", type=bool, default=False, help="Allow duplicates in classic slates (default: False)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg   = json.load(open(args.config, encoding="utf-8"))

    engine  = init_db("nfl", cfg.get("database_url"))
    Base.metadata.create_all(engine)
    session = get_session(engine)

    # *** NEW: Clean old diagnostics once at start of entire sim run ***
    from ivan.sports.nfl.simulate_contest import clean_old_diagnostics
    print("🧹 Cleaning old diagnostic folders...")
    clean_old_diagnostics(days_to_keep=1)
    print()

    df = pd.read_csv(
        cfg["paths"]["entries_csv"],
        engine="python",
        usecols=["Contest ID", "Contest Name"],
        dtype={"Contest ID": str, "Contest Name": str},
    )
    df["Contest ID"]   = df["Contest ID"].str.strip()
    df["Contest Name"] = df["Contest Name"].str.strip()
    df = df.dropna(subset=["Contest ID", "Contest Name"]).copy()
    df["Contest ID"] = df["Contest ID"].astype(int)
    df = df.sort_values("Contest ID")
    unique_df = df.drop_duplicates(subset=["Contest Name"], keep="last")

    for _, row in unique_df.iterrows():
        cid  = int(row["Contest ID"])
        name = row["Contest Name"]
        print(f"→ Starting contest {cid} ('{name}')")
        try:
            simulate_one_contest(
                session      = session,
                contest_id   = cid,
                contest_name = name,
                config_path  = args.config,
                field_cap    = args.c,
                num_sims     = args.n,
                use_classic_dupes = args.d,
            )
        except Exception as e:
            print(f"⚠️  Skipped contest {cid} ('{name}') due to error: {e}\n")

if __name__ == "__main__":
    main()
