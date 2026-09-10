#!/usr/bin/env python3
# scripts/nfl/sim.py
"""
Simulate every contest in the entries CSV against the generated field.

    scripts/nfl/sim.py [--entries file.csv] [-n 1000] [-c 10000] [-d]

Each contest's per-lineup results replace any earlier ones, and a report of
the run (field sample, payout scaling, invariants, correlation check) is
stored for the web UI's Contests page and written to data/nfl/diagnostics/.
"""
import argparse
import traceback

import pandas as pd

from ivan.core.config import config_path, load_config
from ivan.database.session import get_session, init_db
from ivan.sports.nfl import sim_report
from ivan.sports.nfl.models import NFLBase
from ivan.sports.nfl.simulate_contest import clean_old_diagnostics, simulate_one_contest


def parse_args():
    p = argparse.ArgumentParser(description="Run NFL simulations for each contest in entries_csv")
    p.add_argument("--config", default=str(config_path("nfl")), help="path to nfl json config file")
    p.add_argument("--entries", default=None, help="entries CSV (default: paths.entries_csv from config, or $IVAN_ENTRIES_CSV)")
    p.add_argument("-n", type=int, default=1000, help="Number of simulation samples per contest (default: 1000)")
    p.add_argument("-c", type=int, default=10000, help="Field cap: largest field size to simulate (default: 10000)")
    p.add_argument("-d", action="store_true", help="Allow duplicates in classic slates (default: False)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config("nfl", args.config)
    entries_csv = args.entries or cfg["paths"]["entries_csv"]

    engine = init_db("nfl", cfg.get("database_url"))
    NFLBase.metadata.create_all(engine)          # contest_sim_report on an older database
    session = get_session(engine)

    removed = clean_old_diagnostics(days_to_keep=1)
    if removed:
        print(f"Removed {removed} old diagnostics folder(s).")

    df = pd.read_csv(entries_csv, engine="python", usecols=["Contest ID", "Contest Name"],
                     dtype={"Contest ID": str, "Contest Name": str})
    df["Contest ID"] = df["Contest ID"].str.strip()
    df["Contest Name"] = df["Contest Name"].str.strip()
    df = df.dropna(subset=["Contest ID", "Contest Name"]).copy()
    df["Contest ID"] = df["Contest ID"].astype(int)
    contests = df.sort_values("Contest ID").drop_duplicates(subset=["Contest Name"], keep="last")

    print(f"Simulating {len(contests)} contest(s): {args.n} sims each, field cap {args.c:,}\n")
    failed = 0
    for i, (cid, name) in enumerate(zip(contests["Contest ID"], contests["Contest Name"]), 1):
        print(f"--- {i}/{len(contests)} ---")
        try:
            report = simulate_one_contest(session=session, contest_id=int(cid), contest_name=name,
                                          config_path=args.config, field_cap=args.c, num_sims=args.n,
                                          use_classic_dupes=args.d)
            print(sim_report.format_summary(report), flush=True)
        except Exception as e:
            failed += 1
            session.rollback()
            print(f"FAILED contest {cid} ('{name}'): {e}")
            traceback.print_exc()
        print(flush=True)

    done = len(contests) - failed
    print(f"Done: {done} contest(s) simulated" + (f", {failed} failed" if failed else "") + ". Details: web UI -> Contests.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
