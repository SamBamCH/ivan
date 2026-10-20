#!/usr/bin/env python3
"""
scripts/nfl/sim.py

Evaluate every contest in the entries CSV for every candidate lineup, world
by world (ivan.sports.nfl.simulate_contest.simulate_entries): the slate's
worlds are drawn once, and each contest gets every lineup's expected payout,
ROI, rank and cash chance per world against its real field and payouts.

    scripts/nfl/sim.py [--entries file.csv] [-n 1000] [-d]

Each contest's per-lineup results replace any earlier ones, and a report of
the run (field, payouts, conservation checks, correlation check, the worlds)
is stored for the web UI's Contests page and written to data/nfl/diagnostics/.
"""
import argparse

from ivan.core import artifacts
from ivan.core.config import config_path, load_config
from ivan.database.session import get_session, init_db
from ivan.sports.nfl.models import NFLBase
from ivan.sports.nfl.paths import OUTPUT_DIR
from ivan.sports.nfl.simulate_contest import clean_old_diagnostics, simulate_entries


def main() -> int:
    p = argparse.ArgumentParser(description="Run NFL simulations for each contest in entries_csv")
    p.add_argument("--config", default=str(config_path("nfl")), help="path to nfl json config file")
    p.add_argument("--entries", default=None, help="entries CSV (default: paths.entries_csv from config, or $IVAN_ENTRIES_CSV)")
    p.add_argument("-n", type=int, default=1000, help="Simulations per contest (worlds, under config sim.method = conditioned) (default: 1000)")
    p.add_argument("-c", type=int, default=10000, help="Field cap: largest field size to simulate (default: 10000; unused by the conditioned method)")
    p.add_argument("-d", action="store_true", help="Allow duplicates in classic slates (default: False)")
    a = p.parse_args()
    cfg = load_config("nfl", a.config)
    engine = init_db("nfl", cfg.get("database_url"))
    NFLBase.metadata.create_all(engine)          # contest_sim_report on an older database
    session = get_session(engine)
    removed = clean_old_diagnostics(days_to_keep=1)
    if removed:
        print(f"Removed {removed} old diagnostics folder(s).")
    r = simulate_entries(session, a.entries or cfg["paths"]["entries_csv"], a.config, n_sims=a.n, field_cap=a.c,
                         use_classic_dupes=a.d, artifact_dir=artifacts.sim_dir(OUTPUT_DIR))
    print("Details: web UI -> Contests.")
    return 1 if r["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
