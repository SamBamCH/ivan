# src/ivan/core/pnl.py
"""
The money across sports: what every sport's results ledger rolls up to, and what it costs in tax.

A sport's ledger (ivan.sports.nba.backtest.results.ledger_days, ivan.sports.nfl.results.ledger_days) is built from
contest rows, one per contest we played: {final, fees, won, found, cashed, gross_win, gross_loss, ...}. `gross_win` is
the sum over our entries of payout - fee where positive, `gross_loss` the sum of fee - payout where positive (the W
and L of ivan.core.tax), so gross_win - gross_loss = won - fees.

    agg(rows)                    fees, won, net, roi, entries, cashed, contests, gross_win, gross_loss over the final rows
    roll_up(slates)              slates -> {"overall", "days", "slates"}: the shape the picker and the tiles read
    year_totals(ledger, year)    one sport's totals over the days of one calendar year (the tax year)
    combined(ledgers, year)      every sport's year, the total, and the tax on the total under each scenario

The tax is on the year's total over every sport (one return): ivan.core.tax.compute_tax(W, L) with config/tax.json,
where only loss_cap_pct (90% from 2026) of the losses is deductible, and only by itemizing. What W and L are is the
config's wager_basis (ivan.core.tax.wagering_totals): "entry" (the default: each entry's fee netted against its own
payout, gross_win and gross_loss) or "fees" (every payout is winnings, every entry fee a loss: the conservative reading).

    scripts/pnl.py               the same numbers in the terminal (the home page's strip)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ivan.core.tax import BUSINESS, ENTRY, FEES, GROSS, NET, TaxConfig, compute_tax, load_tax_config, wagering_totals

SUMS = ("fees", "won", "gross_win", "gross_loss", "fees_unscored")
COUNTS = ("entries", "cashed", "contests", "unscored")


def agg(rows: Iterable[dict]) -> dict:
    """The totals over contest rows: the money of the final ones, and what the unscored ones cost."""
    rows = list(rows)
    fin = [r for r in rows if r.get("final")]
    fees = float(sum(r["fees"] or 0 for r in fin)); won = float(sum(r["won"] or 0 for r in fin))
    return dict(fees=fees, won=won, net=won - fees, roi=(won - fees) / fees if fees else None,
                entries=int(sum(r.get("found") or 0 for r in fin)), cashed=int(sum(r.get("cashed") or 0 for r in fin)),
                contests=len(fin), unscored=len(rows) - len(fin),
                fees_unscored=float(sum(r.get("fees") or 0 for r in rows if not r.get("final"))),
                gross_win=float(sum(r.get("gross_win") or 0 for r in fin)), gross_loss=float(sum(r.get("gross_loss") or 0 for r in fin)))


def total(aggs: Iterable[dict]) -> dict:
    """Several totals as one (sports into the site's, days into a year's)."""
    aggs = list(aggs)
    out = {k: float(sum(a.get(k) or 0 for a in aggs)) for k in SUMS}
    out.update({k: int(sum(a.get(k) or 0 for a in aggs)) for k in COUNTS})
    out["net"] = out["won"] - out["fees"]
    out["roi"] = out["net"] / out["fees"] if out["fees"] else None
    return out


def roll_up(slates: List[dict]) -> dict:
    """
    [{"key", "day" (iso), "complete", "contests": [rows]}] -> {"overall": agg + days, slates, incomplete,
    "days": {iso day: agg + slates (keys), incomplete}, newest first, "slates": {key: agg + day, complete}}.
    A slate with no contest rows is left out.
    """
    slates = [s for s in slates if s.get("contests")]
    by_day: Dict[str, List[dict]] = {}
    for s in slates:
        by_day.setdefault(str(s["day"]), []).append(s)
    days = {}
    for day, group in by_day.items():
        a = agg(c for s in group for c in s["contests"])
        a["slates"] = [s["key"] for s in group]
        a["incomplete"] = sum(1 for s in group if not s["complete"])
        days[day] = a
    overall = agg(c for s in slates for c in s["contests"])
    overall.update(days=len(days), slates=len(slates), incomplete=sum(1 for s in slates if not s["complete"]))
    return dict(overall=overall, days=dict(sorted(days.items(), reverse=True)),
                slates={s["key"]: dict(agg(s["contests"]), day=str(s["day"]), complete=bool(s["complete"])) for s in slates})


def year_totals(ledger: Optional[dict], year: int) -> dict:
    """One sport's totals over the days of `year` (a day belongs to the year its slate locked in)."""
    days = (ledger or {}).get("days") or {}
    mine = [a for day, a in days.items() if str(day)[:4] == str(year)]
    out = total(mine)
    out["days"] = len(mine)
    out["incomplete"] = int(sum(a.get("incomplete") or 0 for a in mine))
    return out


def tax_scenarios(base: TaxConfig) -> List[TaxConfig]:
    """The rows of the tax table: the configured filing, then what the cap and the choice to itemize each change."""
    other = (base.with_(wager_basis=ENTRY, label="as configured, but each entry's fee netted against its own payout first") if base.wager_basis == FEES else
             base.with_(wager_basis=FEES, label="as configured, but every payout as winnings and every entry fee as a loss"))
    if base.reporting_mode == BUSINESS:
        return [base.with_(label="as configured"),
                base.with_(loss_cap_pct=1.00, label="a business, 100% of losses deductible (pre-2026, or a repeal)"),
                base.with_(se_tax=False, label="a business, without self-employment tax"),
                base.with_(reporting_mode=GROSS, itemize="always", label="a hobby, itemizing (90% of losses deductible)"),
                base.with_(reporting_mode=GROSS, itemize="never", label="a hobby, the standard deduction (no losses deducted)"),
                other]
    return [base.with_(label="as configured"),
            base.with_(reporting_mode=GROSS, loss_cap_pct=0.90, itemize="always", label="itemizing, 90% of losses deductible (the 2026 rule)"),
            base.with_(reporting_mode=GROSS, loss_cap_pct=1.00, itemize="always", label="itemizing, 100% of losses deductible (pre-2026, or a repeal)"),
            base.with_(reporting_mode=GROSS, itemize="never", label="standard deduction (no losses deducted)"),
            base.with_(reporting_mode=NET, label="net DFS (taxed on the profit, as the 1099 reports it)"),
            other]


def combined(ledgers: Dict[str, Optional[dict]], year: int, cfg: Optional[TaxConfig] = None, labels: Optional[Dict[str, str]] = None) -> dict:
    """
    Every sport's `year`, the total, and the tax on the total: {"year", "sports": [{key, label, **totals}], "total",
    "all_time" (the total over every year on file), "tax": [{label, describe, **TaxResult}], "config"}. A sport whose
    ledger is None (no results on this machine, or a ledger that could not be read) is left out; with none left, "sports" is empty.
    """
    cfg = cfg or load_tax_config()
    labels = labels or {}
    sports = [dict(year_totals(led, year), key=k, label=labels.get(k, k.upper())) for k, led in ledgers.items() if led and led.get("overall")]
    tot = total(sports)
    tot["days"] = int(sum(s["days"] for s in sports))
    tot["incomplete"] = int(sum(s["incomplete"] for s in sports))
    all_time = total(led["overall"] for led in ledgers.values() if led and led.get("overall"))
    tax = []
    for c in tax_scenarios(cfg):
        W, L = wagering_totals(c, fees=tot["fees"], won=tot["won"], gross_win=tot["gross_win"], gross_loss=tot["gross_loss"])
        r = compute_tax(W, L, c, total_fees=tot["fees"])
        tax.append(dict(r.as_dict(), label=c.label, describe=c.describe(), loss_cap_pct=c.loss_cap_pct, wager_basis=c.wager_basis,
                        rate=c.marginal_rate if c.mode == "flat" else None))
    return dict(year=int(year), sports=sports, total=tot, all_time=all_time, tax=tax,
                config=dict(describe=cfg.describe(), reporting_mode=cfg.reporting_mode, business_expenses=cfg.business_expenses, se_tax=cfg.se_tax,
                            wager_basis=cfg.wager_basis, loss_cap_pct=cfg.loss_cap_pct, marginal_rate=cfg.marginal_rate, mode=cfg.mode,
                            filing_status=cfg.filing_status, standard_deduction=cfg.std_deduction, state_rate=cfg.state_rate))
