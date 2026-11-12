# tests/core/test_pnl.py
"""The money across sports (ivan.core.pnl): the roll-up every sport's ledger shares, a year's totals, and the tax on the total."""
from __future__ import annotations

import pytest

from ivan.core import pnl
from ivan.core.tax import TaxConfig


def row(fees, won, gross_win, found=10, cashed=3, final=True):
    if not final:
        return dict(final=False, fees=fees, won=None, found=0, cashed=0, gross_win=None, gross_loss=None)
    return dict(final=True, fees=fees, won=won, found=found, cashed=cashed, gross_win=gross_win, gross_loss=gross_win - (won - fees))


def ledger(*slates):
    return pnl.roll_up([dict(key=k, day=day, complete=all(r["final"] for r in rows), contests=rows) for k, day, rows in slates])


def test_agg_counts_final_rows_and_prices_the_unscored():
    a = pnl.agg([row(100.0, 60.0, 30.0), row(50.0, 90.0, 70.0), row(20.0, None, None, final=False)])
    assert (a["fees"], a["won"], a["net"], a["contests"], a["unscored"], a["fees_unscored"]) == (150.0, 150.0, 0.0, 2, 1, 20.0)
    assert (a["gross_win"], a["gross_loss"]) == (100.0, 100.0) and a["roi"] == 0.0
    assert pnl.agg([])["roi"] is None


def test_roll_up_by_day_and_year_totals():
    led = ledger(("a", "2025-12-28", [row(10.0, 0.0, 0.0)]), ("b", "2026-01-03", [row(100.0, 150.0, 90.0)]),
                 ("c", "2026-01-03", [row(40.0, None, None, final=False)]), ("empty", "2026-01-04", []))
    assert list(led["days"]) == ["2026-01-03", "2025-12-28"] and led["overall"]["slates"] == 3 and led["overall"]["incomplete"] == 1
    assert led["days"]["2026-01-03"]["slates"] == ["b", "c"] and led["slates"]["c"]["complete"] is False
    y = pnl.year_totals(led, 2026)
    assert (y["net"], y["fees"], y["gross_win"], y["gross_loss"], y["days"], y["incomplete"], y["fees_unscored"]) == (50.0, 100.0, 90.0, 40.0, 1, 1, 40.0)
    assert pnl.year_totals(led, 2025)["net"] == -10.0 and pnl.year_totals(None, 2026)["contests"] == 0


def test_combined_taxes_the_total_over_every_sport():
    """A losing sport offsets a winning one before tax, but only 90% of the losses count, and only by itemizing. wager_basis "entry"
    (the default) nets each entry's fee against its own payout; "fees" counts every payout as winnings and every entry fee as a loss."""
    nba = ledger(("n", "2026-11-01", [row(1000.0, 400.0, 300.0)]))                 # per entry: W 300, L 900
    nfl = ledger(("f", "2026-11-02", [row(2000.0, 3000.0, 2500.0)]))               # per entry: W 2500, L 1500
    assert TaxConfig().wager_basis == "entry"
    cfg = TaxConfig(marginal_rate=0.25, itemize="auto", standard_deduction=16100.0, wager_basis="fees")
    m = pnl.combined({"nba": nba, "nfl": nfl, "mma": None}, 2026, cfg, labels={"nba": "NBA", "nfl": "NFL"})
    assert [s["label"] for s in m["sports"]] == ["NBA", "NFL"] and m["config"]["wager_basis"] == "fees"
    tot = m["total"]
    assert (tot["net"], tot["fees"], tot["won"], tot["gross_win"], tot["gross_loss"], tot["contests"]) == (400.0, 3000.0, 3400.0, 2800.0, 2400.0, 2)
    own = m["tax"][0]                                                              # W 3400 (payouts), L 3000 (fees); 0.9 * 3000 < the standard deduction
    assert (own["W"], own["L"]) == (3400.0, 3000.0) and not own["itemize"]
    assert own["taxable_gambling"] == 3400.0 and own["est_tax"] == pytest.approx(850.0) and own["after_tax_profit"] == pytest.approx(-450.0)
    capped = m["tax"][1]                                                           # itemizing under the 90% cap: 10% of all the fees stays taxable
    assert capped["deductible"] == pytest.approx(2700.0) and capped["taxable_gambling"] == pytest.approx(700.0)
    assert capped["phantom"] == pytest.approx(300.0) and capped["est_tax"] == pytest.approx(175.0) and capped["after_tax_profit"] == pytest.approx(225.0)
    assert capped["after_tax_roi"] == pytest.approx(0.075)
    full = m["tax"][2]                                                             # 100%: taxed on the profit
    assert full["taxable_gambling"] == pytest.approx(400.0) and full["phantom"] == 0.0
    assert m["tax"][4]["taxable_gambling"] == pytest.approx(400.0)                 # net DFS
    other = m["tax"][5]                                                            # the other count: per entry
    assert other["wager_basis"] == "entry" and (other["W"], other["L"]) == (2800.0, 2400.0)
    assert m["all_time"]["net"] == 400.0 and m["year"] == 2026
    e = pnl.combined({"nba": nba, "nfl": nfl}, 2026, cfg.with_(wager_basis="entry"))
    assert e["tax"][1]["deductible"] == pytest.approx(2160.0) and e["tax"][1]["taxable_gambling"] == pytest.approx(640.0)
    assert e["tax"][1]["after_tax_profit"] == pytest.approx(240.0) and e["tax"][5]["wager_basis"] == "fees"
    with pytest.raises(ValueError):
        TaxConfig(wager_basis="sessions")


def test_combined_with_no_ledger_is_empty():
    m = pnl.combined({"nba": None, "nfl": {"error": "x", "overall": None, "days": {}}}, 2026, TaxConfig())
    assert m["sports"] == [] and m["total"]["contests"] == 0 and m["total"]["roi"] is None
