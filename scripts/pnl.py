#!/usr/bin/env python3
"""
scripts/pnl.py

The money over every sport and the tax on it (ivan.core.pnl): what the Ivan home page's strip shows, in the terminal.
Each sport's results ledger (scripts/nba/results.py, scripts/nfl/results.py) summed over one calendar year, then the
US federal tax on the total under the filing in config/tax.json and the scenarios beside it (ivan.core.tax: from 2026
only 90% of losses are deductible, and only by itemizing). An estimate, not tax advice.

    scripts/pnl.py [--year 2026] [--sports nba,nfl] [--cap 0.9] [--rate 0.22] [--basis entry|fees]
"""
import argparse
from datetime import datetime

from ivan.core import bankroll, dk_history, pnl
from ivan.core.tax import load_tax_config

SPORTS = ("nba", "nfl")


ledger = bankroll.sport_ledger       # nba and nfl from their results code, any other sport from DraftKings' entry history alone


def money(x) -> str:
    return "—" if x is None else f"{'-' if x < 0 else '+'}${abs(x):,.2f}"


def usd(x) -> str:
    return "—" if x is None else f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:+.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=None, help="the tax year (default: this year)")
    ap.add_argument("--sports", default=",".join(list(SPORTS) + [s for s in dk_history.sports() if s not in SPORTS]),
                    help="default: nba, nfl and every other sport the imported DraftKings entry history holds")
    ap.add_argument("--cap", type=float, default=None, help="share of losses deductible in the configured row (config/tax.json: 0.90)")
    ap.add_argument("--rate", type=float, default=None, help="flat marginal rate (config/tax.json)")
    ap.add_argument("--basis", choices=("fees", "entry"), default=None,
                    help="what counts as winnings and losses (config/tax.json wager_basis: fees = every payout and every entry fee)")
    a = ap.parse_args()
    now = datetime.now()
    ledgers = {}
    for s in [x.strip().lower() for x in a.sports.split(",") if x.strip()]:
        try:
            ledgers[s] = ledger(s, now)
        except Exception as e:   # noqa: BLE001 - one sport's trouble must not hide the others
            print(f"{s}: results unavailable: {type(e).__name__}: {e}")
    m = pnl.combined(ledgers, a.year or now.year, load_tax_config(loss_cap_pct=a.cap, marginal_rate=a.rate, wager_basis=a.basis))
    tot = m["total"]
    for s in m["sports"]:
        print(f"{s['label']:4} {m['year']}: {money(s['net']):>11} ({pct(s['roi']):>5}) on {usd(s['fees']):>10} in fees; {s['cashed']}/{s['entries']} cashed; "
              f"{s['contests']} contests; {s['days']} days" + (f"; {s['incomplete']} slates not scored ({usd(s['fees_unscored'])} out)" if s["incomplete"] else ""))
    print(f"ALL  {m['year']}: {money(tot['net']):>11} ({pct(tot['roi']):>5}) on {usd(tot['fees']):>10} in fees; {tot['cashed']}/{tot['entries']} cashed; {tot['contests']} contests")
    if not tot["contests"]:
        return 0
    c = m["config"]
    own = m["tax"][0]
    counted = (f"paid out {usd(own['W'])} (the winnings), entry fees {usd(own['L'])} (the losses)" if c["wager_basis"] == "fees"
               else f"won {usd(own['W'])} on the entries that beat their fee, lost {usd(own['L'])} on the rest")
    print(f"\ntax on {m['year']}: {counted} "
          f"({c['filing_status']}, standard deduction {usd(c['standard_deduction'])}, {c['describe']})")
    print(f"  {'':72} {'deducted':>10} {'taxable':>10} {'phantom':>10} {'SE tax':>9} {'est. tax':>10} {'after tax':>11} {'ROI':>5}")
    for t in m["tax"]:
        ded = t["deductible"] if (t["itemize"] or t["reporting_mode"] != "gross_165d") else 0.0
        print(f"  {t['label'][:72]:72} {usd(ded):>10} {usd(t['taxable_gambling']):>10} {usd(t['phantom']):>10} {usd(t['se_tax']):>9} {usd(t['est_tax']):>10} {money(t['after_tax_profit']):>11} {pct(t['after_tax_roi']):>5}")
    if m["all_time"]["contests"] != tot["contests"]:
        print(f"\nall years on file: {money(m['all_time']['net'])} on {usd(m['all_time']['fees'])} in fees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
