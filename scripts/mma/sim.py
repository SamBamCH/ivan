#!/usr/bin/env python3
# scripts/mma/sim.py
"""
Sample the field and simulate contests against it.

    scripts/mma/sim.py                                      every contest in data/mma/input/entries.csv
    scripts/mma/sim.py --entries my_entries.csv             (the DraftKings entries export)
    scripts/mma/sim.py --contest "mini-MAX"                 the upcoming DK contest whose name contains the text
    scripts/mma/sim.py --contest-id 123456789
    scripts/mma/sim.py --payouts payouts.csv --total-entries 47500 --fee 0.5 --name "UFC Sat"

Options for every form: --field <simulated field size>, -n <sims>, --no-field-csv.

With no contest named, every contest in the entries file is simulated, as
the NFL sim does; contests the ingest listing missed are fetched from
DraftKings by id. The DK forms use the contest and payout tables ingest
stored; the last form describes the contest by hand, as the original
mma_dfs did (payouts CSV, real entry count and fee; defaults from
config/mma.json -> contest and paths). Results for a contest name replace
any earlier ones; each sampled field is written to
data/mma/output/field.<name>.<stamp>.csv.
"""
import argparse
import re
import traceback
from datetime import datetime
from pathlib import Path

from ivan.core.config import config_path, load_config
from ivan.database.session import get_session, init_db
from ivan.sports.mma.contests import ensure_contests
from ivan.sports.mma.entries import contests_in, load_entries
from ivan.sports.mma.models import MMABase
from ivan.sports.mma.paths import OUTPUT_DIR
from ivan.sports.mma.sample_field import format_field_diagnostics
from ivan.sports.mma.simulate_contest import field_frame, find_contest_id, resolve_contest, simulate_contest


def summary_line(report: dict) -> str:
    return (f"{report['n_sims']:,} sims over {report['candidates']:,} candidates; scaled prize pool "
            f"${report['scaled_prize_pool']:,.2f} across {report['paying_ranks']:,} paying ranks; "
            f"pool error per sim {report['pool_max_abs_error']:.2e}; lineup fpts mean {report['fpts_mean']:.1f}, "
            f"p90 {report['fpts_p90']:.1f}; {report['seconds']:.0f}s")


def write_field(session, contest_name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", contest_name).strip("_")[:40]
    path = OUTPUT_DIR / f"field.{slug}.{datetime.now():%Y%m%d-%H%M%S}.csv"
    field_frame(session, contest_name).to_csv(path, index=False)
    return path


def run_one(session, cfg, contest: dict, args, show_field: bool = True) -> dict:
    print(f"contest {contest['name']!r}: {contest['total_entries']:,} entries @ ${contest['entry_fee']:.2f}, "
          f"{len(contest['payouts'])} payout tiers")
    report = simulate_contest(session, cfg, contest, field_size=args.field, n_sims=args.n)
    if show_field:
        print(format_field_diagnostics(report["field"], report["exposures"]))
        print()
    print(summary_line(report))
    if not args.no_field_csv:
        print(f"field written to {write_field(session, contest['name'])}")
    return report


def run_entries(session, cfg, entries_csv: str, args) -> int:
    entries = load_entries(entries_csv)
    contests = contests_in(entries)
    print(f"{len(entries)} entries in {len(contests)} contest(s) from {entries_csv}")
    try:
        missing = ensure_contests(session, [c["contest_id"] for c in contests])["missing"]
    except Exception as e:                                   # network: carry on with what is stored
        print(f"could not fetch contest details from DraftKings: {e}")
        missing = []
    failed = 0
    for i, c in enumerate(contests, 1):
        print(f"\n--- {i}/{len(contests)}: {c['entries']} entries ---")
        try:
            if c["contest_id"] in missing:
                raise RuntimeError("no payout table could be fetched for this contest")
            contest = resolve_contest(session, cfg, contest_id=c["contest_id"], contest_name=c["name"])
            run_one(session, cfg, contest, args, show_field=(i == 1))
        except (Exception, SystemExit) as e:
            failed += 1
            session.rollback()
            print(f"FAILED contest {c['contest_id']} ({c['name']!r}): {e}")
            if not isinstance(e, (SystemExit, RuntimeError)):
                traceback.print_exc()
    print(f"\nDone: {len(contests) - failed} contest(s) simulated" + (f", {failed} failed" if failed else "") + ".")
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description="Simulate MMA contests against a sampled field")
    p.add_argument("--config", default=str(config_path("mma")))
    p.add_argument("--entries", default=None, help="entries CSV (default: paths.entries_csv, or $IVAN_ENTRIES_CSV)")
    p.add_argument("--contest-id", type=int, default=None, help="one DK contest id in the database")
    p.add_argument("--contest", default=None, help="the one DK contest whose name contains this text (case-insensitive)")
    p.add_argument("--payouts", default=None, help="payouts CSV for a hand-described contest (default: paths.payouts_csv)")
    p.add_argument("--total-entries", type=int, default=None, help="hand-described: real entry count (default: contest.total_entries)")
    p.add_argument("--fee", type=float, default=None, help="hand-described: entry fee (default: contest.entry_fee)")
    p.add_argument("--name", default=None, help="hand-described: contest name the results are stored under")
    p.add_argument("--field", type=int, default=None, help="simulated field size (default: field.size)")
    p.add_argument("-n", type=int, default=None, help="simulations (default: simulation.n_sims)")
    p.add_argument("--no-field-csv", action="store_true", help="do not write the sampled field to data/mma/output")
    args = p.parse_args()

    cfg = load_config("mma", args.config)
    engine = init_db("mma", cfg.get("database_url"))
    MMABase.metadata.create_all(engine)
    session = get_session(engine)

    one_contest = any(v is not None for v in (args.contest_id, args.contest, args.payouts, args.total_entries, args.fee, args.name))
    if not one_contest:
        entries_csv = args.entries or cfg.get("paths", {}).get("entries_csv")
        if args.entries or (entries_csv and Path(entries_csv).exists()):
            return run_entries(session, cfg, entries_csv, args)
        print(f"no entries file at {entries_csv}; simulating the hand-described contest from config")

    contest_id = args.contest_id
    if contest_id is None and args.contest:
        contest_id = find_contest_id(session, args.contest)
    contest = resolve_contest(session, cfg, contest_id=contest_id, payouts_csv=args.payouts,
                              total_entries=args.total_entries, entry_fee=args.fee, contest_name=args.name)
    run_one(session, cfg, contest, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
