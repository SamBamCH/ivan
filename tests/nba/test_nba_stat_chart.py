"""The stat hover's chart (ivan.sports.nba.desk.rows.stat_chart) and the World's record behind it (projection.worlds.stat_shares)."""
import numpy as np
import pytest

from ivan.sports.nba.desk.rows import CHART_BINS, stat_chart
from ivan.sports.nba.projection.worlds import stat_shares


def test_stat_shares_is_the_share_of_worlds_at_each_whole_value():
    s = stat_shares(np.array([0, 2, 2, 3.0]))
    assert s == [0.25, 0.0, 0.5, 0.25] and stat_shares([]) == []


def test_stat_chart_puts_history_and_ours_on_the_same_bins():
    c = stat_chart([1, 2, 2, 5], [0.1, 0.2, 0.4, 0.2, 0.1], line=2.5)
    assert c["step"] == 1 and [b["label"] for b in c["bins"]] == ["0", "1", "2", "3", "4", "5"]
    assert [b["hist"] for b in c["bins"]] == [0, 0.25, 0.5, 0, 0, 0.25] and c["bins"][2]["ours"] == pytest.approx(0.4)
    assert c["n_hist"] == 4 and c["hist_mean"] == 2.5 and c["ours_mean"] == pytest.approx(2.0) and c["has_ours"]
    assert c["line_at"] == 3.0                                       # 2.5 sits on the edge between the bars of 2 and 3
    assert c["ymax"] == stat_chart([0] * 9 + [1], [0.9, 0.1])["ymax"] == stat_chart([20, 30], None)["ymax"] == 0.5     # one axis for every chart


def test_stat_chart_bins_a_wide_stat_and_stands_without_a_draw():
    pts = stat_chart([18, 25, 31, 40], stat_shares(np.arange(8, 44)), line=24.5)
    assert pts["step"] == 3 and len(pts["bins"]) <= CHART_BINS and pts["bins"][0]["label"] == "0-2"
    assert sum(b["hist"] for b in pts["bins"]) == pytest.approx(1.0) and sum(b["ours"] for b in pts["bins"]) == pytest.approx(1.0, abs=1e-3)
    none = stat_chart([0, 1], None)
    assert not none["has_ours"] and none["ours_mean"] is None and none["line_at"] is None and len(none["bins"]) == 2
