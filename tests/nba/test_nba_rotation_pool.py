"""The rotation drawn per team, per world (gamesim.stints.sit_draw)."""
import numpy as np

from ivan.sports.nba.gamesim.stints import sit_draw


def test_each_man_sits_his_share_and_the_count_holds():
    q = [0.5, 0.5, 0.1, 0.0, 0.3]
    s = sit_draw(q, 100000, np.random.default_rng(0))
    assert np.allclose(s.mean(axis=0), q, atol=0.01) and not s[:, 3].any()
    assert set(np.unique(s.sum(axis=1))) == {1, 2}                       # 1.4 expected: one or two sit, never none, never three
    assert (s[:, 0] & s[:, 1]).mean() < 0.5 * 0.5 - 0.05                 # the two toss-ups sit together less often than two coin flips would


def test_nobody_sits_without_a_chance():
    assert not sit_draw([0.0, 0.0], 50, np.random.default_rng(1)).any() and sit_draw([], 5, np.random.default_rng(1)).shape == (5, 0)
    assert sit_draw([1.0, 0.0], 50, np.random.default_rng(1))[:, 0].all()
