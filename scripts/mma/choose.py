#!/usr/bin/env python3
# scripts/mma/choose.py
"""
Choose the lineups to enter from the simulated contests.

    scripts/mma/choose.py                                   fill every entry in data/mma/input/entries.csv
    scripts/mma/choose.py --entries my_entries.csv
    scripts/mma/choose.py --name <contest> [--lineups 150]  one contest, N lineups (default: the latest simulated)

Options for both: --roi-weight 0.8, --diversity-weight 0.4 (defaults from config/mma.json -> choose).

The entries form chooses, per contest, as many lineups as the file holds
entries for it and writes chosen_entries.<stamp>.csv in DraftKings' upload
format (the entry rows with the six F columns filled), plus
chosen_entries.{lineups,exposures}.<stamp>.csv, to data/mma/output/. The
one-contest form writes chosen_lineups.{lineups,upload,exposures}.<stamp>.csv.
"""
import argparse
from pathlib import Path

from sqlalchemy import func

from ivan.core.config import config_path, load_config
from ivan.database.session import get_session, init_db
from ivan.sports.mma.choose import (choose_entries, choose_lineups, export, export_entries, exposures, format_contests,
                                    format_exposures, lineup_rows)
from ivan.sports.mma.entries import load_entries
from ivan.sports.mma.models import LineupSimulationResult


def run_entries(session, cfg, entries_csv: str, args) -> int:
    entries = load_entries(entries_csv)
    print(f"{len(entries)} entries from {entries_csv}\n")
    chosen = choose_entries(session, entries, cfg, roi_weight=args.roi_weight, diversity_weight=args.diversity_weight)
    print(format_contests(chosen["contests"]))
    if not chosen["picks"]:
        print("\nnothing chosen: no contest in the entries file has been simulated; run scripts/mma/sim.py first")
        return 1
    print(f"\n[exposures over {len(chosen['picks'])} entries]")
    print(format_exposures(exposures(session, chosen["picks"])))
    paths = export_entries(session, chosen)
    print()
    for k, v in paths.items():
        print(f"{k}: {v}")
    skipped = [c for c in chosen["contests"] if c["skipped"]]
    if skipped:
        print(f"\n{len(skipped)} contest(s) not simulated, their entries left out; run scripts/mma/sim.py")
    return 1 if skipped else 0


def run_one(session, cfg, args) -> int:
    name = args.name
    if name is None:
        row = (session.query(LineupSimulationResult.contest_name, func.max(LineupSimulationResult.created_at))
               .group_by(LineupSimulationResult.contest_name).order_by(func.max(LineupSimulationResult.created_at).desc()).first())
        if row is None:
            raise SystemExit("no simulation results; run scripts/mma/sim.py first")
        name = row[0]
    picks = choose_lineups(session, name, cfg, num_lineups=args.lineups, roi_weight=args.roi_weight,
                           diversity_weight=args.diversity_weight)
    df = lineup_rows(session, picks)
    print(f"contest {name!r}: chose {len(picks)} lineups; mean cash {df['Cash %'].mean():.1f}%, "
          f"mean ROI {100 * df['Avg ROI'].mean():+.1f}%, mean payout ${df['Avg Payout'].mean():.2f}\n")
    print(format_exposures(exposures(session, picks)))
    paths = export(session, picks)
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Choose MMA lineups from simulation results")
    p.add_argument("--config", default=str(config_path("mma")))
    p.add_argument("--entries", default=None, help="entries CSV (default: paths.entries_csv, or $IVAN_ENTRIES_CSV)")
    p.add_argument("--name", default=None, help="one contest, by the name sim.py stored it under")
    p.add_argument("--lineups", type=int, default=None, help="one contest: how many to pick (default: choose.num_lineups)")
    p.add_argument("--roi-weight", type=float, default=None, help="weight on average ROI vs cash rate (default: choose.roi_weight)")
    p.add_argument("--diversity-weight", type=float, default=None, help="penalty per overlapping fighter (default: choose.diversity_weight)")
    args = p.parse_args()

    cfg = load_config("mma", args.config)
    session = get_session(init_db("mma", cfg.get("database_url")))

    if args.name is None and args.lineups is None:
        entries_csv = args.entries or cfg.get("paths", {}).get("entries_csv")
        if args.entries or (entries_csv and Path(entries_csv).exists()):
            return run_entries(session, cfg, entries_csv, args)
        print(f"no entries file at {entries_csv}; choosing from the latest simulated contest")
    return run_one(session, cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
