"""Spec section 7 test cases for the tax engine (dfs_tax_backtest_spec.md)."""
import math

import pytest

from ivan.sports.nfl.tax import (GROSS, NET, TaxConfig, after_tax_closed_form, bracket_tax, breakeven_profit_ratio,
                                 compute_tax, load_tax_config, scenario_configs)

ITEMIZE = TaxConfig(loss_cap_pct=0.90, marginal_rate=0.22, itemize="always")
STANDARD = ITEMIZE.with_(itemize="never")


def test_single_slate_itemizing():
    r = compute_tax(100, 150, ITEMIZE)
    assert r.deductible == 100
    assert r.taxable_gambling == 0
    assert r.est_tax == 0
    assert r.phantom == 0


def test_single_slate_standard_deduction():
    r = compute_tax(100, 150, STANDARD)
    assert r.taxable_gambling == 100
    assert r.est_tax == pytest.approx(22.0)
    assert r.phantom == 100          # lost money, taxed on 100


def test_big_win():
    r = compute_tax(4000, 100, ITEMIZE)
    assert r.deductible == pytest.approx(90)
    assert r.taxable_gambling == pytest.approx(3910)
    assert r.phantom == pytest.approx(10)


@pytest.mark.parametrize("n,W,L,ded,taxable,P,phantom", [
    (0, 4000, 100, 90, 3910, 3900, 10),
    (20, 6000, 3100, 2790, 3210, 2900, 310),
    (40, 8000, 6100, 5490, 2510, 1900, 610),
    (60, 10000, 9100, 8190, 1810, 900, 910),
    (78, 11800, 11800, 10620, 1180, 0, 1180),
    # the spec's table prints 1210 here, but its own definition (3d: taxable - max(P, 0)) gives 1110
    (80, 12000, 12100, 10890, 1110, -100, 1110),
])
def test_big_win_plus_losing_slates(n, W, L, ded, taxable, P, phantom):
    assert (4000 + 100 * n, 100 + 150 * n) == (W, L)
    r = compute_tax(W, L, ITEMIZE)
    assert r.deductible == pytest.approx(ded)
    assert r.taxable_gambling == pytest.approx(taxable)
    assert r.taxable_gambling == pytest.approx(3910 - 35 * n)
    assert r.P == pytest.approx(P)
    assert r.phantom == pytest.approx(phantom)


def test_itemize_vs_standard_decision():
    cfg = TaxConfig(other_itemized=5000, standard_deduction=16100, itemize="auto")
    r = compute_tax(12000, 12100, cfg)
    assert r.itemized_total == pytest.approx(15890)
    assert r.itemize is False
    assert r.taxable_gambling == 12000
    assert r.incremental_taxable == pytest.approx(12000)
    # once other deductions push itemizing over the line, the gambling losses count
    r2 = compute_tax(12000, 12100, cfg.with_(other_itemized=6000))
    assert r2.itemize is True
    assert r2.taxable_gambling == pytest.approx(12000 - 10890)
    # the switch itself is priced: deduction goes from 16100 (std) to 16890 (itemized)
    assert r2.incremental_taxable == pytest.approx(12000 - 16890 + 16100)


def test_repeal_scenario():
    r = compute_tax(12000, 12100, ITEMIZE.with_(loss_cap_pct=1.0))
    assert r.deductible == 12000
    assert r.taxable_gambling == 0


def test_net_dfs_mode():
    cfg = TaxConfig(reporting_mode=NET, marginal_rate=0.22)
    assert compute_tax(12000, 12100, cfg).taxable_gambling == 0
    r = compute_tax(4000, 100, cfg)
    assert r.taxable_gambling == 3900
    assert r.phantom == 0
    assert r.est_tax == pytest.approx(0.22 * 3900)


def test_breakeven():
    t, L = 0.22, 10000.0
    P = 282.0
    r = compute_tax(L + P, L, ITEMIZE.with_(marginal_rate=t))
    assert abs(r.after_tax_profit) < 1.0
    assert r.after_tax_profit == pytest.approx(after_tax_closed_form(P, L, t), abs=1e-6)
    assert breakeven_profit_ratio(0.22) == pytest.approx(0.0282, abs=5e-4)
    assert breakeven_profit_ratio(0.24) == pytest.approx(0.0316, abs=5e-4)


def test_closed_form_matches_engine_when_cap_binds():
    t = 0.24
    for W, L in [(5000, 1000), (20000, 15000), (900, 1000)]:
        if 0.9 * L < W:
            r = compute_tax(W, L, ITEMIZE.with_(marginal_rate=t))
            assert r.after_tax_profit == pytest.approx(after_tax_closed_form(W - L, L, t))


def test_bracket_tax():
    br = ((0, 0.10), (10000, 0.20), (50000, 0.30))
    assert bracket_tax(5000, br) == pytest.approx(500)
    assert bracket_tax(20000, br) == pytest.approx(1000 + 2000)
    assert bracket_tax(60000, br) == pytest.approx(1000 + 8000 + 3000)
    assert bracket_tax(-5, br) == 0


def test_bracket_mode_is_incremental_over_wages():
    br = ((0, 0.10), (10000, 0.20), (50000, 0.30))
    cfg = TaxConfig(brackets=br, other_taxable_income=60000, standard_deduction=10000, itemize="never")
    r = compute_tax(4000, 0, cfg)
    # wages alone: taxable 50000 -> tax 1000 + 8000 = 9000; with 4000 of winnings -> 54000 -> +1200 at 30%
    assert r.federal_tax == pytest.approx(1200)
    assert r.mode == "bracket"


def test_state_without_loss_deduction_taxes_gross():
    cfg = ITEMIZE.with_(state_rate=0.05, state_allows_loss_deduction=False)
    r = compute_tax(100, 150, cfg)
    assert r.federal_tax == 0
    assert r.state_tax == pytest.approx(5.0)


def test_after_tax_roi_and_effective_rate():
    r = compute_tax(4000, 100, ITEMIZE, total_fees=1000)
    assert r.after_tax_roi == pytest.approx((3900 - 0.22 * 3910) / 1000)
    assert r.effective_rate == pytest.approx(0.22)


def test_scenario_table_has_both_caps_and_modes():
    rows = scenario_configs(TaxConfig())
    labels = [c.label for c in rows]
    assert labels[0] == "as configured"
    assert any("100%" in l for l in labels) and any("90%" in l for l in labels)
    assert rows[-1].reporting_mode == NET


def test_load_tax_config_defaults_and_overrides(tmp_path):
    cfg = load_tax_config()                      # the checked-in config/tax.json
    assert cfg.loss_cap_pct == 0.90 and cfg.reporting_mode == GROSS and cfg.mode == "flat"
    assert cfg.std_deduction == 16100
    mfj = load_tax_config(filing_status="married_joint", use_brackets=True)
    assert mfj.std_deduction == 32200 and mfj.mode == "bracket"
    missing = load_tax_config(tmp_path / "nope.json", marginal_rate=0.35)
    assert missing.marginal_rate == 0.35 and missing.std_deduction == 16100


def test_validation():
    with pytest.raises(ValueError):
        TaxConfig(reporting_mode="weird")
    with pytest.raises(ValueError):
        TaxConfig(loss_cap_pct=1.5)
