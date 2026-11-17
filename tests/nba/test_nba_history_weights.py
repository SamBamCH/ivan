"""The trailing history after a change of team or season (projection.history.build_history) and its backtest."""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ivan.sports.nba.backtest import history_weights as HW
from ivan.sports.nba.projection import history as H
from ivan.sports.nba.scoring import STATS


def _box(spans):
    """spans: (team, first day, games, minutes, points per minute) for one player, a game every other day."""
    rows = []
    for team, start, n, mins, ppm in spans:
        for i in range(n):
            d = start + timedelta(days=2 * i)
            r = dict(game_id=f"002{d:%Y%m%d}", game_date=d, player_name="p", team=team, position="BENCH", minutes=float(mins),
                     dk_fpts=mins * ppm, starter=0.0, played=mins > 0)
            r.update({s: 0.0 for s in STATS}); r["points"] = mins * ppm
            rows.append(r)
    return pd.DataFrame(rows)


def test_a_new_team_discounts_what_came_before():
    box = _box([("AAA", date(2025, 11, 1), 30, 30, 1.2), ("BBB", date(2026, 1, 10), 6, 10, 0.6)])
    new = H.build_history(box).rows.iloc[-1]
    old = H.build_history(box, **HW.OLD).rows.iloc[-1]
    assert new.team == "BBB" and new.games == 36
    assert new.min_ewm < 11 < 13 < old.min_ewm                                  # six games in: the bench player, not the starter
    assert new.rate_points < 0.95 < 1.05 < old.rate_points
    same = _box([("AAA", date(2025, 11, 1), 36, 30, 1.2)])
    a, b = H.build_history(same).rows.iloc[-1], H.build_history(same, min_team_keep=1.0, rate_team_keep=1.0).rows.iloc[-1]
    assert np.isclose(a.eff_min, b.eff_min) and np.isclose(a.min_ewm, b.min_ewm)   # nobody moved: nothing discounted


def test_a_new_season_discounts_minutes_only():
    box = _box([("AAA", date(2025, 12, 1), 20, 30, 1.0), ("AAA", date(2026, 10, 21), 2, 12, 1.0)])
    new = H.build_history(box).rows.iloc[-1]
    kept = H.build_history(box, min_season_keep=1.0).rows.iloc[-1]
    assert new.min_ewm < kept.min_ewm - 4
    assert np.isclose(new.eff_min, kept.eff_min)                                   # the rates carry over the summer


def test_compare_scores_the_new_team_stretch():
    box = _box([("AAA", date(2025, 11, 1), 30, 30, 1.2), ("BBB", date(2026, 1, 10), 10, 10, 0.6)])
    t = HW.compare(box, start="2025-11-01", settings=[("old", HW.OLD), ("live", {})])
    assert t.loc["old", "n_new_team"] == 10
    assert t.loc["live", "min_mae_new_team"] < t.loc["old", "min_mae_new_team"]
    assert t.loc["live", "fpts_mae_new_team"] < t.loc["old", "fpts_mae_new_team"]
