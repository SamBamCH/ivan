import numpy as np
import pytest

from ivan.sports.mma.fight import Fight, Fighter
from ivan.sports.mma.hazard import DEFAULTS, HazardFight, build_fight
from ivan.sports.mma.odds import american_to_prob


def fighters(**kw):
    f1 = Fighter(1, "fav", 9000, win_odds=-300, itd_odds=-150, sig_strikes=80, takedowns=2)
    f2 = Fighter(2, "dog", 7000, win_odds=250, itd_odds=400, sig_strikes=40, takedowns=0.5)
    return f1, f2


def test_hazard_reproduces_the_itd_price_and_the_moneyline():
    f1, f2 = fighters()
    h = HazardFight("x", f1, f2, rounds=3, itd_odds=-120)
    p = american_to_prob(-120)
    assert h.finish_prob() == pytest.approx(p)
    assert 1 - h.decision_prob() == pytest.approx(p)
    # the segment probabilities and the decision mass partition the fight
    assert sum(prob for _, _, _, prob in h.segments()) + h.decision_prob() == pytest.approx(1.0)
    # the fighters' ITD prices split the stoppages; the overall win rate is still the moneyline split
    q1, q2 = american_to_prob(-150), american_to_prob(400)
    assert h.finish_split() == pytest.approx(q1 / (q1 + q2))
    p_win = american_to_prob(-300) / (american_to_prob(-300) + american_to_prob(250))
    assert h.finish_split() * p + h.decision_split() * (1 - p) == pytest.approx(p_win)
    a, b = h.simulate(100000, np.random.default_rng(0))
    assert abs((a > b).mean() - p_win) < 0.01


def test_flat_hazard_round_shares_follow_the_survival_curve():
    f1, f2 = fighters()
    h = HazardFight("x", f1, f2, rounds=3, itd_odds=100)        # 50% finish
    lam = -np.log(0.5) / 900
    seg = {k: p for k, _, _, p in h.segments()}
    assert seg["u60"] == pytest.approx(1 - np.exp(-lam * 60))
    assert seg["u60"] + seg["r1"] == pytest.approx(1 - np.exp(-lam * 300))
    assert h.expected_time()[0] == pytest.approx((1 - np.exp(-lam * 900)) / lam)
    # a five-round fight spreads the same finish probability over 25 minutes
    h5 = HazardFight("y", f1, f2, rounds=5, itd_odds=100)
    assert h5.hazard() < h.hazard()
    assert [k for k, *_ in h5.segments()] == ["u60", "r1", "r2", "r3", "r4", "r5"]


def test_first_minute_multiplier_front_loads_without_changing_the_finish_probability():
    f1, f2 = fighters()
    flat = HazardFight("x", f1, f2, rounds=3, itd_odds=-120)
    bump = HazardFight("x", f1, f2, rounds=3, itd_odds=-120, params={**DEFAULTS, "first_minute_mult": 2.0})
    assert bump.finish_prob() == pytest.approx(flat.finish_prob())
    assert bump.decision_prob() == pytest.approx(flat.decision_prob())
    assert dict((k, p) for k, _, _, p in bump.segments())["u60"] > dict((k, p) for k, _, _, p in flat.segments())["u60"]
    # the inverse of the cumulative hazard is its inverse on both sides of the first minute
    for t in (10.0, 59.0, 61.0, 400.0):
        assert bump.inverse_cumulative_hazard(bump.cumulative_hazard(np.array([t])))[0] == pytest.approx(t)


def test_lines_are_held_as_means_and_expected_points_match_the_simulation():
    f1, f2 = fighters()
    h = HazardFight("x", f1, f2, rounds=3, itd_odds=-120)
    rng = np.random.default_rng(1)
    n = 200000
    T = h.total_time
    tau = h.inverse_cumulative_hazard(-np.log(rng.random(n)))
    t = np.minimum(tau, T)
    u = rng.random(n)
    f1_wins = np.where(tau < T, u < h.finish_split(), u < h.decision_split())
    r_sig, r_td = h.rates(True)
    weight = np.where(f1_wins, 1.0, h.scoring["loser_production_share"]) * t
    assert (r_sig * weight).mean() == pytest.approx(80, rel=0.02)
    assert (r_td * weight).mean() == pytest.approx(2, rel=0.02)
    a, b = h.simulate(n, rng)
    assert a.mean() == pytest.approx(h.expected_points(True), rel=0.02)
    assert b.mean() == pytest.approx(h.expected_points(False), rel=0.02)
    # every outcome awards exactly one bonus and takedowns are whole: scores are on the 5-point lattice plus strikes
    assert np.maximum(a, b).min() >= 30 * 0.99


def test_missing_markets_fall_back_sensibly():
    # no ITD anywhere: the default finish probability and the moneyline split for both finish and decision
    f1 = Fighter(1, "a", 9000, win_odds=-300, sig_strikes=60, takedowns=1)
    f2 = Fighter(2, "b", 7000, win_odds=250, sig_strikes=40, takedowns=1)
    h = HazardFight("x", f1, f2, rounds=3)
    assert h.finish_prob() == pytest.approx(DEFAULTS["default_itd_prob"])
    p_win = american_to_prob(-300) / (american_to_prob(-300) + american_to_prob(250))
    assert h.finish_split() == pytest.approx(p_win)
    assert h.decision_split() == pytest.approx(p_win)
    # fighters' ITD prices without a fight ITD price: their sum is the finish probability (capped)
    f1.itd_odds, f2.itd_odds = -150, 400
    assert h.finish_prob() == pytest.approx(american_to_prob(-150) + american_to_prob(400))
    f1.itd_odds, f2.itd_odds = -2000, -2000
    assert h.finish_prob() == pytest.approx(DEFAULTS["max_itd_prob"])
    # a fighter without lines produces nothing but can still win the bonus
    g = HazardFight("y", Fighter(1, "a", 9000, win_odds=-100), Fighter(2, "b", 9000, win_odds=-100), rounds=3, itd_odds=-900)
    a, b = g.simulate(2000, np.random.default_rng(2))
    assert set(np.round(np.maximum(a, b), 6)).issubset({115.0, 90.0, 70.0, 45.0, 30.0})
    assert (np.minimum(a, b) == 0).all()


def test_build_fight_picks_the_model_by_config():
    f1, f2 = fighters()
    assert isinstance(build_fight("x", f1, f2, 3, itd_odds=-120, scoring=Fight.__dataclass_fields__["scoring"].default_factory(),
                                  fight_cfg={"model": "tree"}), Fight)
    h = build_fight("x", f1, f2, 3, itd_odds=-120, scoring=Fight.__dataclass_fields__["scoring"].default_factory(), fight_cfg=None)
    assert isinstance(h, HazardFight)
    h2 = build_fight("x", f1, f2, 3, itd_odds=-120, scoring=h.scoring, fight_cfg={"rate_cv": 0.0})
    assert h2.params["rate_cv"] == 0.0 and h2.params["first_minute_mult"] == 1.0
