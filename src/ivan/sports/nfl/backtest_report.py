# src/ivan/sports/nfl/backtest_report.py
"""
The settlement + tax report over a set of choose runs (live picks or
replayed slates): prints it and writes entries.csv, ledger.csv,
scenarios.csv, formats.csv, sim_vs_actual.csv, volume_sweep.csv,
volume_sweep.png and summary.json under `out`. Used by
scripts/nfl/backtest_tax.py and scripts/replay.py.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ivan.sports.nfl import backtest_tax as bt
from ivan.sports.nfl.tax import GROSS, TaxConfig, breakeven_profit_ratio, compute_tax, scenario_configs


def money(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.1%}"


def run_report(session, runs: Sequence[bt.ChooseRun], cfg: TaxConfig, *, bankroll: float = 0.0, out: Path,
               dk_history: Optional[pd.DataFrame] = None, in_field: str = "auto", since: Optional[datetime] = None,
               until: Optional[datetime] = None, played_only: bool = False, tax_year: Optional[int] = None,
               volumes: Sequence[float] = (0.25, 0.5, 1, 2, 4, 8), n_boot: int = 4000, seed: int = 0,
               plot: bool = True, title: str = "", echo=print) -> dict:
    entries, cov = bt.assemble(session, runs, dk_history=dk_history, in_field=in_field, since=since, until=until,
                               played_only=played_only)
    if not entries:
        echo("No entries found: no chosen_entries.lineups.*.csv in the run directories and no DK history.")
        return {"entries": 0}

    years = pd.Series([e.tax_year for e in entries if e.tax_year]).value_counts()
    year = tax_year or (int(years.idxmax()) if len(years) else datetime.now().year)
    entries = [e for e in entries if e.tax_year == year or e.tax_year is None]
    settled = [e for e in entries if e.settled]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if title:
        echo(f"== {title}")
    echo(f"Tax year {year}: {cfg.describe()}" + (f"; reserve {cfg.tax_reserve_pct:.0%} of wins" if cfg.tax_reserve_pct else ""))
    echo(f"Choose runs: {len(runs)}" + (f" ({runs[0].stamp} .. {runs[-1].stamp})" if runs else ""))
    echo(f"Contests with a pre-lock pick: {cov.contests_chosen}; settled: {cov.contests_settled}"
         + (f"; history-only entries: {cov.history_only_entries}" if cov.history_only_entries else ""))
    echo("Entries by source: " + ", ".join(f"{k} {v}" for k, v in sorted(cov.entries.items(), key=lambda kv: bt.SOURCE_RANK.get(kv[0], 9))))
    for n in cov.notes:
        echo("  note: " + n)
    if cov.unsettled_contests:
        echo(f"Unsettled contests ({len(cov.unsettled_contests)}):")
        for u in cov.unsettled_contests[:25]:
            when = bt.utc_to_local(u["lock"]).strftime("%Y-%m-%d %H:%M") if u["lock"] else "?"
            echo(f"  {when}  {u['contest_id']}  {u['name'][:60]:60s} {u['entries']:>4} entries  {u['why']}")
        if len(cov.unsettled_contests) > 25:
            echo(f"  ... {len(cov.unsettled_contests) - 25} more")

    bt.entries_frame(entries).to_csv(out / "entries.csv", index=False)
    summary = {"title": title, "tax_year": year, "config": cfg.describe(), "start_bankroll": bankroll, "entries": len(entries),
               "settled": len(settled),
               "coverage": {"contests_chosen": cov.contests_chosen, "contests_settled": cov.contests_settled,
                            "by_source": cov.entries, "history_only_entries": cov.history_only_entries},
               "unsettled_contests": [{**u, "lock": u["lock"].isoformat() if u["lock"] else None} for u in cov.unsettled_contests]}
    if not settled:
        echo("\nNothing settled, so no tax ledger. To settle entries:")
        echo("  - export your DraftKings contest history and pass it with --dk-history, or")
        echo("  - scrape the contests: scripts/nfl/get_contest_data.py --contest-ids ... --username <dk user>, and")
        echo("  - make sure the week's box scores are loaded (scripts/nfl/overnight.py).")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    scen = scenario_configs(cfg)
    led = bt.ledger(entries, cfg, bankroll, scenarios=scen[1:])
    led.to_csv(out / "ledger.csv", index=False)
    sc = bt.scenario_table(entries, cfg, bankroll)
    sc.to_csv(out / "scenarios.csv", index=False)
    fmt = bt.format_table(entries, cfg, by="format_class+contest_type")
    fmt.to_csv(out / "formats.csv", index=False)
    sva = bt.sim_vs_actual(entries)
    sva.to_csv(out / "sim_vs_actual.csv", index=False)
    sweep = pd.concat([bt.volume_sweep(entries, cfg, bankroll, volumes, n_boot, seed, basis="actual"),
                       bt.volume_sweep(entries, cfg, bankroll, volumes, n_boot, seed, basis="sim")], ignore_index=True)
    sweep.to_csv(out / "volume_sweep.csv", index=False)

    W, L, fees = sum(e.winnings for e in settled), sum(e.losses for e in settled), sum(e.entry_fee for e in settled)
    t = compute_tax(W, L, cfg, fees)
    played = [e for e in settled if e.played]
    echo(f"\n== Year {year}: {len(settled)} settled entries in {len({e.contest_id for e in settled})} contests over "
         f"{len({e.slate for e in settled})} slates" + (f" ({len(played)} known to be played)" if dk_history is not None else ""))
    echo(f"fees {money(fees)}   payout {money(fees + t.P)}   W {money(W)}   L {money(L)}   P {money(t.P)}   ROI {pct(t.P / fees if fees else None)}")
    echo(f"deductible {money(t.deductible)}   {'itemizing' if t.itemize else 'standard deduction'}   taxable {money(t.taxable_gambling)}"
         f"   phantom {money(t.phantom)}   est tax {money(t.est_tax)}   after tax {money(t.after_tax_profit)} ({pct(t.after_tax_roi)})")
    if cfg.reporting_mode == GROSS and L > 0:
        echo(f"break-even P/L at this rate: {breakeven_profit_ratio(cfg.marginal_rate if cfg.mode == 'flat' else 0.22, cfg.loss_cap_pct):.1%}"
             f"   actual P/L: {t.P / L:+.1%}")
    bank_end = bankroll + t.P
    echo(f"bankroll now {money(bank_end)} vs tax owed {money(t.est_tax)}: {'TAX OWED EXCEEDS BANKROLL' if t.est_tax > bank_end else 'ok'}")
    if led.tax_shortfall.max() > 0:
        worst = led.loc[led.tax_shortfall.idxmax()]
        echo(f"worst point: {worst.slate} tax {money(worst.est_tax)} vs bankroll {money(worst.bankroll)} (short {money(worst.tax_shortfall)})")

    echo("\n== Scenarios (same results, different rules)")
    cols = ["scenario", "deductible", "taxable_gambling", "phantom", "est_tax", "after_tax_profit", "after_tax_roi"]
    show = sc[cols].copy()
    for c in cols[1:-1]:
        show[c] = show[c].map(money)
    show["after_tax_roi"] = show["after_tax_roi"].map(pct)
    echo(show.to_string(index=False))

    if not fmt.empty:
        echo("\n== By format (tax_added = what the year's tax bill would drop by without this format)")
        show = fmt[["format_class", "contest_type", "entries", "fees", "P", "roi", "cash_rate", "losses_per_fee", "losses_per_profit", "tax_added"]].copy()
        for c in ("fees", "P", "tax_added"):
            show[c] = show[c].map(money)
        show["roi"] = show["roi"].map(pct)
        show["cash_rate"] = show["cash_rate"].map(lambda x: f"{x:.0%}")
        show["losses_per_fee"] = show["losses_per_fee"].map(lambda x: f"{x:.2f}" if x is not None else "-")
        show["losses_per_profit"] = show["losses_per_profit"].map(lambda x: f"{x:.1f}" if x is not None and np.isfinite(x) else "-")
        echo(show.to_string(index=False))

    if not sva.empty:
        both = sva.dropna(subset=["sim_expected_profit", "actual_profit"])
        echo(f"\n== Sim vs actual ({len(both)} contests with both)")
        if len(both):
            echo(f"sim expected profit {money(both.sim_expected_profit.sum())}   actual {money(both.actual_profit.sum())}   "
                 f"sim cash % {both.sim_cash_pct.mean():.1%}   actual cash rate {both.actual_cash_rate.mean():.1%}")
            show = both[["slate", "contest_name", "entries", "fees", "sim_expected_profit", "actual_profit", "sim_cash_pct", "actual_cash_rate", "source"]].copy()
            show["contest_name"] = show.contest_name.str.slice(0, 48)
            for c in ("fees", "sim_expected_profit", "actual_profit"):
                show[c] = show[c].map(money)
            show["sim_cash_pct"] = show.sim_cash_pct.map(lambda x: f"{x:.0%}" if x is not None and np.isfinite(x) else "-")
            show["actual_cash_rate"] = show.actual_cash_rate.map(lambda x: f"{x:.0%}")
            echo(show.to_string(index=False))

    echo("\n== Running ledger (per slate)")
    show = led[["slate", "entries", "slate_fees", "slate_P", "W", "L", "P", "taxable_gambling", "phantom", "est_tax", "after_tax_profit", "bankroll", "tax_exceeds_bankroll"]].copy()
    for c in ("slate_fees", "slate_P", "W", "L", "P", "taxable_gambling", "phantom", "est_tax", "after_tax_profit", "bankroll"):
        show[c] = show[c].map(money)
    echo(show.to_string(index=False))

    if not sweep.empty:
        echo("\n== Volume sweep (bootstrapped years of the observed slates; multiplier 1 = this window's slate count)")
        show = sweep[["basis", "multiplier", "slates", "fees", "pre_tax_profit", "est_tax", "phantom", "after_tax_profit",
                      "after_tax_p10", "after_tax_p90", "p_aftertax_loss", "p_tax_exceeds_bankroll"]].copy()
        for c in ("fees", "pre_tax_profit", "est_tax", "phantom", "after_tax_profit", "after_tax_p10", "after_tax_p90"):
            show[c] = show[c].map(money)
        for c in ("p_aftertax_loss", "p_tax_exceeds_bankroll"):
            show[c] = show[c].map(lambda x: f"{x:.0%}")
        echo(show.to_string(index=False))
        if plot:
            try:
                plot_sweep(sweep, out / "volume_sweep.png", bankroll)
            except Exception as e:                     # a plot must never sink the report
                echo(f"(plot skipped: {e})")

    summary.update({"year_end": t.as_dict(), "fees": fees, "scenarios": sc.to_dict("records"),
                    "formats": fmt.to_dict("records") if not fmt.empty else []})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    echo(f"\nWrote {out}/ (entries, ledger, scenarios, formats, sim_vs_actual, volume_sweep .csv, summary.json"
         + ("" if not plot or sweep.empty else ", volume_sweep.png") + ")")
    return summary


def plot_sweep(sweep: pd.DataFrame, path: Path, start_bankroll: float) -> None:
    """Two panels over volume (total entry fees): expected after-tax profit with its p10-p90 band, and P(tax owed > bankroll)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"actual": "#2a78d6", "sim": "#eb6834"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for basis, g in sweep.groupby("basis"):
        g = g.sort_values("fees")
        c = colors.get(basis, "#555555")
        ax1.plot(g.fees, g.after_tax_profit, color=c, lw=2, marker="o", ms=5, label=f"{basis}: expected after-tax profit")
        ax1.fill_between(g.fees, g.after_tax_p10, g.after_tax_p90, color=c, alpha=0.12, lw=0)
        ax1.plot(g.fees, g.pre_tax_profit, color=c, lw=1, ls="--", label=f"{basis}: pre-tax")
        ax2.plot(g.fees, g.p_tax_exceeds_bankroll, color=c, lw=2, marker="o", ms=5, label=basis)
    ax1.axhline(0, color="#999999", lw=1)
    ax1.set_ylabel("profit for the year ($)")
    ax1.set_title("Same edge at more volume (band: p10-p90)")
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel(f"P(tax owed > bankroll of ${start_bankroll:,.0f} + profit)")
    ax2.set_xlabel("entry fees for the year ($)")
    ax2.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
