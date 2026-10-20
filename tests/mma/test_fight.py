import numpy as np
import pytest

from ivan.sports.mma.fight import Fight, Fighter, project_fighter


def fight(**kw):
    f1 = Fighter(id=1, name="A", salary=9000, win_odds=-1000, itd_odds=-200, r1_win_odds=200, sig_strikes=80, takedowns=2)
    f2 = Fighter(id=2, name="B", salary=6500, win_odds=650, itd_odds=400, r1_win_odds=900, sig_strikes=40, takedowns=0.5)
    return Fight(id="A-B", f1=f1, f2=f2, **kw)


def test_nodes_sum_to_one_and_follow_rounds():
    f3 = fight(rounds=3, u60_odds=1800, r1_odds=400, itd_odds=130)
    n3 = f3.nodes()
    assert sum(n.prob for n in n3) == pytest.approx(1.0)
    assert [n.key for n in n3] == ["u60", "r1", "r2", "r3", "decision"]
    f5 = fight(rounds=5, u60_odds=1800, r1_odds=400, itd_odds=130)
    assert [n.key for n in f5.nodes()] == ["u60", "r1", "r2", "r3", "r4", "r5", "decision"]
    assert f5.total_time == 1500


def test_missing_fight_odds_is_a_coin_flip_between_quick_finish_and_decision():
    n = fight(rounds=3).nodes()
    by = {x.key: x.prob for x in n}
    assert by["u60"] == pytest.approx(0.5)
    assert by["decision"] == pytest.approx(0.5)


def test_heavy_favourite_wins_almost_always_and_winner_gets_bonus():
    f = fight(rounds=3, u60_odds=1800, r1_odds=400, itd_odds=130)
    rng = np.random.default_rng(0)
    a, b = f.simulate(4000, rng)
    assert a.shape == (4000,)
    assert (a > b).mean() > 0.75
    # every outcome awards exactly one bonus; the smallest bonus is the decision (30)
    assert np.maximum(a, b).min() >= 30 * 0.99


def test_no_production_means_only_bonuses():
    f1 = Fighter(id=1, name="A", salary=9000, win_odds=-100, sig_strikes=0, takedowns=0)
    f2 = Fighter(id=2, name="B", salary=9000, win_odds=-100, sig_strikes=0, takedowns=0)
    f = Fight(id="A-B", f1=f1, f2=f2, rounds=3, u60_odds=100000, r1_odds=100000, itd_odds=100000)   # decision nearly certain
    a, b = f.simulate(500, np.random.default_rng(1))
    bonuses = {30.0}
    assert set(np.round(np.maximum(a, b), 6)).issubset(bonuses | {115.0, 90.0, 70.0, 45.0})
    assert (np.minimum(a, b) == 0).all()


def test_projection_matches_original_formula():
    f = Fighter(id=1, name="A", salary=9000, win_odds=-340, sig_strikes=75, takedowns=2)
    win_prob, proj = project_fighter(f, fight_itd_odds=200)
    assert win_prob == pytest.approx(340 / 440)
    itd = 100 / 300
    assert proj == pytest.approx(75 * 0.4 + 2 * 5 + win_prob * (itd * 70 + (1 - itd) * 30))


def test_winner_split_isolates_each_finish_and_matches_the_moneyline():
    # favourite priced to finish: moneyline -300 (75%), ITD -150 (60%), R1 +200 (33%)
    # underdog: +250 (29%), ITD +400 (20%), R1 +900 (10%)
    f1 = Fighter(1, "fav", 9000, win_odds=-300, itd_odds=-150, r1_win_odds=200)
    f2 = Fighter(2, "dog", 7000, win_odds=250, itd_odds=400, r1_win_odds=900)
    fight = Fight("x", f1, f2, u60_odds=1500, r1_odds=250, itd_odds=-120)
    assert fight.winner_share("r1") == pytest.approx(0.3333 / (0.3333 + 0.1), abs=1e-3)
    assert fight.winner_share("itd") == pytest.approx((0.6 - 0.3333) / ((0.6 - 0.3333) + (0.2 - 0.1)), abs=1e-3)
    # by decision the favourite's edge is much smaller than the moneyline says
    dec = fight.winner_share("decision")
    assert dec == pytest.approx((0.75 - 0.6) / ((0.75 - 0.6) + (0.2857 - 0.2)), abs=1e-3) and dec < 0.75
    # without the fighters' finish markets every node falls back to the moneyline split
    plain = Fight("y", Fighter(1, "a", 9000, win_odds=-300), Fighter(2, "b", 7000, win_odds=250), itd_odds=-120)
    for kind in ("r1", "itd", "decision"):
        assert plain.winner_share(kind) == pytest.approx(0.75 / (0.75 + 0.2857), abs=1e-3)
    # and the simulated win rate stays near the moneyline either way
    rng = np.random.default_rng(0)
    for fx in (fight, plain):
        a, b = fx.simulate(20000, rng)
        assert abs((a > b).mean() - 0.75 / (0.75 + 0.2857)) < 0.04
