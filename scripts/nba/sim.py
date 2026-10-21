#!/usr/bin/env python3
"""
scripts/nba/sim.py

Simulate every contest in the entries CSV for every candidate lineup
(ivan.sports.nba.simulate_contest.simulate_entries): the field is sampled
from the generated pool, scored with correlated player draws, ranked and paid.

    scripts/nba/sim.py [--entries file.csv] [-n 1000] [-c 10000]

Each contest's per-lineup results replace any earlier ones; a report of the
run (field, payouts, score distribution, conservation checks) is stored in
contest_sim_report and written to data/nba/diagnostics/, and the run's
artifacts go to data/nba/output/sim/<stamp>/.
"""
import argparse

from ivan.core import artifacts
from ivan.core.config import config_path, load_config
from ivan.database.session import get_session, init_db
from ivan.sports.nba.ingest import ensure_schema
from ivan.sports.nba.paths import OUTPUT_DIR
from ivan.sports.nba.simulate_contest import clean_old_diagnostics, simulate_entries
from ivan.sports.nba.utils import _cfg


def main() -> int:
    p = argparse.ArgumentParser(description="Run NBA contest simulations for each contest in the entries CSV")
    p.add_argument("--config", default=str(config_path("nba")), help="path to the nba json config file")
    p.add_argument("--entries", default=None, help="entries CSV (default: paths.entries_csv from config, or $IVAN_ENTRIES_CSV)")
    p.add_argument("-n", type=int, default=None, help="simulations per contest (default: config sim.n_sims, 1000)")
    p.add_argument("-c", type=int, default=None, help="field cap: largest field size to simulate (default: config field.cap, 10000)")
    a = p.parse_args()
    cfg = load_config("nba", a.config)
    engine = init_db("nba", cfg.get("database_url")); ensure_schema(engine); session = get_session(engine)
    removed = clean_old_diagnostics(days_to_keep=7)
    if removed:
        print(f"Removed {removed} old diagnostics folder(s).")
    r = simulate_entries(session, a.entries or cfg["paths"]["entries_csv"], a.config,
                         n_sims=a.n or int(_cfg(cfg, "sim.n_sims", 1000)), field_cap=a.c or int(_cfg(cfg, "field.cap", 10000)),
                         artifact_dir=artifacts.sim_dir(OUTPUT_DIR))
    return 1 if r["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
