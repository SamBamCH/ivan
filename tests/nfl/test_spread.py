"""Predictive-sd calibration (ivan.sports.nfl.refactor.spread): the flat factor and its power-law generalisation."""
import numpy as np
import pandas as pd

from ivan.sports.nfl import backtest_projections as BP
from ivan.sports.nfl.refactor import spread as SP


def test_power_law_reduces_to_the_flat_factor_when_b_is_one():
    cal = {"factors": {"WR": 1.2}, "power": {"WR": {"a": 1.2, "b": 1.0}}, "n": {}}
    for sd in (2.0, 6.0, 12.0):
        assert abs(SP.spread_factor(cal, "WR", sd) - 1.2) < 1e-12
    assert SP.spread_factor(cal, "WR") == 1.2                       # no model sd: the flat factor
    assert SP.spread_factor(cal, "RB", 5.0) == 1.0                  # no entry: model spread as is


def test_power_law_widens_low_sd_players_and_narrows_high_sd_ones():
    cal = {"factors": {"TE": 1.27}, "power": {"TE": {"a": 2.17, "b": 0.58}}, "n": {}}
    low, high = SP.spread_factor(cal, "TE", 1.5), SP.spread_factor(cal, "TE", 7.2)
    assert low > 1.5 and 0.9 < high < 1.05
    rng = np.random.default_rng(0)
    samples = rng.normal(10.0, 7.2, 20000)
    out = SP.calibrate_spread(samples, "TE", cal)
    assert abs(out.mean() - samples.mean()) < 1e-9
    assert abs(out.std() / samples.std() - SP.spread_factor(cal, "TE", float(samples.std()))) < 1e-6


def test_write_and_load_round_trip_with_powers(tmp_path):
    cal = SP.write_sd_calibration({"WR": 1.19, "RB": 1.21}, {"WR": 700, "RB": 440}, ("2026-08-01", "2026-10-18"),
                                  model_dir=tmp_path, powers={"WR": {"a": 1.56, "b": 0.83}})
    assert cal["power"]["WR"] == {"a": 1.56, "b": 0.83} and "RB" not in cal["power"]
    again = SP.load_sd_calibration(tmp_path)
    assert again["factors"]["RB"] == 1.21 and again["power"]["WR"]["b"] == 0.83
    # a later write without a power law for WR drops it, so the file describes what ingest applies
    SP.write_sd_calibration({"WR": 1.1}, {"WR": 50}, ("x", "y"), model_dir=tmp_path)
    assert "WR" not in SP.load_sd_calibration(tmp_path)["power"]
    assert "sd^" in SP.describe_sd_calibration(again)


def test_fit_recovers_a_known_power_law():
    rng = np.random.default_rng(1)
    n = 2000
    model_sd = rng.uniform(1.5, 12.0, n)
    true_sd = 2.0 * model_sd ** 0.7
    scored = pd.DataFrame({"position": "WR", "fpts_mean": 10.0, "fpts_std_model": model_sd, "fpts_std": model_sd,
                           "actual": 10.0 + rng.normal(0, 1, n) * true_sd})
    pw = BP.sd_calibration_powers(scored, ["WR"])["WR"]
    assert abs(pw["b"] - 0.7) < 0.1 and abs(pw["a"] - 2.0) < 0.5
    assert BP.sd_calibration_powers(scored.head(100), ["WR"]) == {}
