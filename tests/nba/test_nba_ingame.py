"""In-game update (ivan.sports.nba.post_lock.ingame): banking, pace, blowouts."""
import numpy as np

from ivan.sports.nba.post_lock import ingame as G


def _draws(**kw):
    rng = np.random.default_rng(0)
    S = 45 + 12 * rng.standard_normal(4000)
    args = dict(samples=S, mu=45.0, prior_minutes=34.0, fpts_so_far=0.0, minutes_so_far=0.0, left=1.0,
                final_margin=np.zeros(4000), starter=True, rng=rng)
    args.update(kw)
    return G.update(**args)


def test_final_game_is_the_box_score():
    assert (_draws(fpts_so_far=51.25, left=0.0, final=True) == 51.25).all()


def test_hot_first_quarter_ends_above_the_prior_but_far_below_pace():
    d = _draws(fpts_so_far=30.0, minutes_so_far=10.0, left=0.75)
    assert 45 + 10 < d.mean() < 120                       # banked 30 + ~3/4 of a game at about his usual rate
    assert d.min() >= 30.0                                # what is banked is banked


def test_blowout_cuts_a_star_and_never_goes_below_banked():
    calm = _draws(fpts_so_far=40.0, minutes_so_far=27.0, left=0.25, final_margin=np.zeros(4000))
    blow = _draws(fpts_so_far=40.0, minutes_so_far=27.0, left=0.25, final_margin=np.full(4000, 30.0))
    assert blow.mean() < calm.mean() and blow.min() >= 40.0


def test_margin_draws_shrink_with_time_left():
    rng = np.random.default_rng(1)
    early, late = G.margin_draws(10, 0.75, 20000, rng), G.margin_draws(10, 0.25, 20000, rng)
    assert abs(early.mean() - 9.2) < 0.5 and early.std() > late.std() > 7
