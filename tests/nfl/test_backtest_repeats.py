"""The window as a distribution over repeats of the pipeline."""
import numpy as np
import pandas as pd

from ivan.sports.nfl.backtest_repeats import slate_matrix, summarize
from ivan.sports.nfl.tax import TaxConfig


def frames(n_rep=40, seed=1):
    rng = np.random.default_rng(seed)
    slates = ["2025-11-03 19:15", "2025-11-10 19:15", "2025-11-27 19:20"]
    means = [-20.0, -10.0, 80.0]          # one slate carries the window
    rows, srows = [], []
    for k in range(n_rep):
        P = 0.0; sim = 0.0
        for s, m in zip(slates, means):
            p = m + rng.normal(0, 15)
            P += p; sim += 100 + rng.normal(0, 5)
            srows.append({"repeat": k, "slate": s, "contests": 2, "entries": 40, "fees": 40.0, "W": max(p, 0), "L": max(-p, 0), "P": p,
                          "sim_expected_profit": 100.0})
        W = max(P, 0) + 30; L = W - P
        rows.append({"repeat": k, "seed": k, "entries": 120, "contests": 6, "slates": 3, "unsettled_contests": 0, "fees": 120.0,
                     "W": W, "L": L, "P": P, "roi": P / 120, "cash_rate": 0.25, "sim_expected_profit": sim, "sim_entries": 120,
                     "deductible": 0, "itemize": False, "taxable_gambling": W, "phantom": 0, "est_tax": 0.22 * W,
                     "after_tax_profit": P - 0.22 * W, "after_tax_roi": (P - 0.22 * W) / 120})
    per = pd.DataFrame(rows); per.attrs["tax_year"] = 2025
    return per, pd.DataFrame(srows)


def test_summarize_reports_spread_bootstrap_and_leave_one_out():
    per, by = frames()
    summ = summarize(per, by, TaxConfig(), n_boot=500, seed=0)
    assert summ["repeats"] == 40 and summ["tax_year"] == 2025
    assert 20 < summ["P"]["mean"] < 80 and summ["P"]["p5"] < summ["P"]["p50"] < summ["P"]["p95"]
    assert summ["P"]["p_positive"] > 0.9                       # +50 on average with sd ~26
    assert len(summ["per_slate"]) == 3
    best = summ["without_best_slate"]
    assert best["slate"] == "2025-11-27 19:20" and best["mean"] < 0 and best["p_positive"] < 0.2
    assert summ["slate_bootstrap"]["n"] == 500 and 0 < summ["slate_bootstrap"]["p_positive"] < 1
    assert len(summ["leave_one_out"]) == 3
    assert summ["variance"]["pipeline_sd"] > 0 and summ["variance"]["between_slate_sd_of_means"] > 40
    assert summ["corr_sim_expected_vs_actual"] is not None and abs(summ["corr_sim_expected_vs_actual"]) < 0.5


def test_slate_matrix_fills_missing_repeats_with_zero():
    _, by = frames(n_rep=3)
    by = by[~((by.repeat == 1) & (by.slate == "2025-11-03 19:15"))]
    m = slate_matrix(by)
    assert m.shape == (3, 3) and m.loc["2025-11-03 19:15", 1] == 0.0


def test_summarize_empty():
    per = pd.DataFrame(columns=["repeat", "P"]); per.attrs["tax_year"] = None
    assert summarize(per, pd.DataFrame(), TaxConfig())["repeats"] == 0
