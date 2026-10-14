#!/usr/bin/env python3
"""
After-tax backtest of the lineups the pipeline chose.

    python scripts/nfl/backtest_tax.py [--dk-history export.csv] [--runs-dir DIR ...] [--tax-year 2026] [--bankroll 2000]
        [--cap 0.9] [--rate 0.22] [--brackets] [--filing-status single] [--other-income 90000]
        [--other-itemized 0] [--state-rate 0.05] [--mode gross_165d|net_dfs] [--itemize auto|always|never]
        [--reserve-pct 0.3] [--played-only] [--in-field auto|yes|no] [--volumes 0.5,1,2,4,8] [--no-plot]

For each contest, the last choose run before lock says what we would have
submitted; those entries are settled from the best evidence available (a
DraftKings contest-history export, our scraped entries, or the lineup's
actual points placed in the scraped field) and run through the tax engine
(ivan.sports.nfl.tax) as a running ledger. By default the runs are the live
ones under data/nfl/output (and output/web); --runs-dir points at other
directories, e.g. replayed slates. Prints the report and writes the tables
under data/nfl/output/backtest/<stamp>/ (see ivan.sports.nfl.backtest_report).

Not tax advice. Rules and 2026 figures are in config/tax.json; verify them.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ivan.database.session import add_missing_columns, get_session, init_db
from ivan.sports.nfl import backtest_report, backtest_tax as bt
from ivan.sports.nfl.models import NFLBase
from ivan.sports.nfl.paths import OUTPUT_DIR
from ivan.sports.nfl.tax import GROSS, NET, load_tax_config

BACKTEST_DIR = OUTPUT_DIR / "backtest"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dk-history", default=None, help="DraftKings contest-history CSV export (the real results)")
    ap.add_argument("--runs-dir", action="append", default=None, help="directory of choose runs (repeatable; default the live output dirs)")
    ap.add_argument("--tax-year", type=int, default=None, help="calendar year to report (default: the year with the most entries)")
    ap.add_argument("--since", default=None, help="only slates locking on/after this local date")
    ap.add_argument("--until", default=None, help="only slates locking on/before this local date")
    ap.add_argument("--bankroll", type=float, default=0.0, help="bankroll at the start of the year (default 0)")
    ap.add_argument("--played-only", action="store_true", help="with --dk-history: only entries that were really played")
    ap.add_argument("--in-field", choices=["auto", "yes", "no"], default="auto",
                    help="does a scraped field already contain our entries? auto removes one field entry per matching score")
    t = ap.add_argument_group("tax rules (defaults from config/tax.json)")
    t.add_argument("--tax-config", default=None, help="alternative tax json")
    t.add_argument("--cap", type=float, default=None, dest="loss_cap_pct", help="deductible share of losses (0.9; 1.0 = repeal)")
    t.add_argument("--rate", type=float, default=None, dest="marginal_rate", help="flat marginal rate")
    t.add_argument("--brackets", action="store_true", help="bracket mode: layer winnings on --other-income")
    t.add_argument("--filing-status", default=None, choices=["single", "married_joint", "married_separate", "head_of_household"])
    t.add_argument("--std-deduction", type=float, default=None, dest="standard_deduction")
    t.add_argument("--other-itemized", type=float, default=None)
    t.add_argument("--other-income", type=float, default=None, dest="other_taxable_income")
    t.add_argument("--state-rate", type=float, default=None)
    t.add_argument("--no-state-loss-deduction", action="store_true")
    t.add_argument("--mode", choices=[GROSS, NET], default=None, dest="reporting_mode")
    t.add_argument("--itemize", choices=["auto", "always", "never"], default=None)
    t.add_argument("--reserve-pct", type=float, default=None, dest="tax_reserve_pct", help="share of each win swept to a tax reserve")
    v = ap.add_argument_group("volume sweep")
    v.add_argument("--volumes", default="0.25,0.5,1,2,4,8", help="multiples of the observed slate count")
    v.add_argument("--n-boot", type=int, default=4000)
    v.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output directory (default data/nfl/output/backtest/<stamp>)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--db-url", default=None)
    return ap.parse_args()


def _local_date(s):
    return bt.local_to_utc(datetime.fromisoformat(s).astimezone()) if s else None


def main() -> int:
    a = parse_args()
    overrides = dict(loss_cap_pct=a.loss_cap_pct, marginal_rate=a.marginal_rate, filing_status=a.filing_status,
                     standard_deduction=a.standard_deduction, other_itemized=a.other_itemized,
                     other_taxable_income=a.other_taxable_income, state_rate=a.state_rate,
                     reporting_mode=a.reporting_mode, itemize=a.itemize, tax_reserve_pct=a.tax_reserve_pct,
                     use_brackets=True if a.brackets else None,
                     state_allows_loss_deduction=False if a.no_state_loss_deduction else None)
    cfg = load_tax_config(a.tax_config, **overrides)

    engine = init_db("nfl", a.db_url)
    NFLBase.metadata.create_all(engine)
    add_missing_columns(engine, NFLBase.metadata)
    session = get_session(engine)

    history = bt.load_dk_history(a.dk_history) if a.dk_history else None
    runs = bt.find_choose_runs([Path(d) for d in a.runs_dir]) if a.runs_dir else bt.find_choose_runs()
    out = Path(a.out) if a.out else BACKTEST_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = backtest_report.run_report(
        session, runs, cfg, bankroll=a.bankroll, out=out, dk_history=history, in_field=a.in_field,
        since=_local_date(a.since), until=_local_date(a.until), played_only=a.played_only, tax_year=a.tax_year,
        volumes=[float(x) for x in a.volumes.split(",") if x.strip()], n_boot=a.n_boot, seed=a.seed, plot=not a.no_plot)
    return 0 if summary.get("entries") else 1


if __name__ == "__main__":
    raise SystemExit(main())
