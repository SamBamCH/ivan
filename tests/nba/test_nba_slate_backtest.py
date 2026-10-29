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


def test_roi_timeline(tmp_path):
    import json
    (tmp_path / "lateswap.20261028-185931.json").write_text(json.dumps({"when": "2026-10-28T18:57:13", "games_started": 2, "games_total": 3, "contests": [
        {"contest_id": "7", "entries": [{"entry_id": "a", "original": {"exp_payout": 1.0}, "best": {"exp_payout": 2.0}},
                                         {"entry_id": "zz", "original": {"exp_payout": 9.0}, "best": {"exp_payout": 9.0}}]}]}))
    results = [dict(contest_id=7, entry_fee=1.0, fees=1.0, won=0.5, entries=[dict(entry_id="a")],
                    claim=dict(roi=3.0, at=datetime(2026, 10, 28, 17, 50)))]
    tl = SB.roi_timeline(results, datetime(2026, 10, 28, 18, 0), date(2026, 10, 28), out_dir=tmp_path)
    assert [x["what"].split(":")[0] for x in tl] == ["the chooser's claim before lock", "late swap", "final"]
    assert tl[1]["roi"] == 0.0 and tl[1]["roi_swapped"] == 1.0 and tl[1]["entries"] == 1     # another player's entry is not ours
    assert tl[-1]["roi"] == -0.5
