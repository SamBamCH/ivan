"""slate_backtest: the view a slate opens in, ties sharing payouts, DK names against box names."""
from datetime import date, datetime

import numpy as np

from ivan.sports.nba import slate_backtest as SB


def test_mode_for():
    now = datetime(2026, 10, 29, 18, 30)
    assert SB.mode_for(date(2026, 10, 28), datetime(2026, 10, 28, 18, 0), now) == "backtest"
    assert SB.mode_for(date(2026, 10, 29), datetime(2026, 10, 29, 18, 0), now) == "lateswap"
    assert SB.mode_for(date(2026, 10, 29), datetime(2026, 10, 29, 19, 0), now) == "prelock"
    assert SB.mode_for(date(2026, 10, 30), datetime(2026, 10, 30, 19, 0), now) == "prelock"


def test_ties_share_the_places_they_cover():
    pay = np.array([10.0, 6.0, 2.0, 0.0])
    rank, won = SB.ranked_payouts(np.array([300.0, 280.0, 280.0, 250.0]), pay)
    assert list(rank) == [1, 2, 2, 4] and list(won) == [10.0, 4.0, 4.0, 0.0]


def test_box_points_match_dk_spellings():
    box = SB.BoxPoints({"robert williams iii": 20.0, "gg jackson ii": 15.5, "jalen williams": 40.0, "jaylin williams": 9.0})
    assert box.points("robert williams") == 20.0          # the suffix
    assert box.points("gregory jackson") == 15.5          # the one Jackson that day
    assert box.points("cody williams") == 0.0              # three Williamses: no guess, he did not play
