"""The public (Stokastic-like) projection replica: priced from the props, unpriced from history and salary."""
import pandas as pd

from ivan.sports.nba import public_projection as PP


def test_prop_sum_needs_the_core_markets_and_scales():
    assert PP.prop_sum({"points": 20.0, "rebounds": 5.0}, 2.0) is None
    s = PP.prop_sum({"points": 20.0, "rebounds": 8.0, "assists": 4.0, "fg3m": 2.0}, 2.0)
    assert s == 20 + 10 + 6 + 1 - 1.0
    assert abs((PP.PRICED_SCALE * s + PP.PRICED_SHIFT) - 35.0) < 1.0      # 36 raw -> ~35.5: the replica shrinks the top a little


def test_unpriced_rookie_falls_back_on_salary():
    h = PP.history_features(None, {})
    x = dict(h, salary_k=6.0)
    v = max(0.0, PP.UNPRICED_INTERCEPT + sum(PP.UNPRICED[k] * x[k] for k in PP.UNPRICED))
    assert 5.0 < v < 8.0


def test_missing_minor_market_counts_as_zero_not_nan():
    s = PP.prop_sum({"points": 20.0, "rebounds": 8.0, "assists": 4.0, "fg3m": float("nan"), "steals": None}, 0.0)
    assert s == 20 + 10 + 6
