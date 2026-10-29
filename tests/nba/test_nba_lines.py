"""GameLines: one game per team, the one on the slate; never another night's line."""
from datetime import datetime

import pandas as pd

from ivan.sports.nba.projection.lines import GameLines


def _cons():
    return pd.DataFrame([
        dict(game_id="a", commence_time=datetime(2026, 10, 23, 19, 10), home_abbr="MIA", away_abbr="DET", spread_home=9.25, total=232.0, p_home=0.22, books=5),
        dict(game_id="b", commence_time=datetime(2026, 10, 29, 19, 10), home_abbr="DET", away_abbr="SAC", spread_home=-7.5, total=228.0, p_home=0.75, books=5),
        dict(game_id="c", commence_time=datetime(2026, 11, 25, 21, 10), home_abbr="POR", away_abbr="SAS", spread_home=4.5, total=238.5, p_home=0.41, books=5),
    ])


def test_picks_the_slate_game_by_tip():
    L = GameLines(_cons(), tips={"DET": datetime(2026, 10, 29, 19, 0), "SAC": datetime(2026, 10, 29, 19, 0), "POR": datetime(2026, 10, 29, 22, 0)})
    assert L.team_spread("DET") == -7.5 and L.by_team["DET"]["opponent"] == "SAC"
    assert L.team_spread("SAC") == 7.5
    assert L.team_spread("POR") is None                     # its only stored game is weeks away: no line, not that one
    assert L.favourite("SAC") == ("DET", 7.5) and L.favourite("DET") == ("DET", 7.5)
    assert L.favourite("POR") is None


def test_without_tips_the_next_game():
    L = GameLines(_cons(), now=datetime(2026, 10, 29, 12, 0))
    assert L.by_team["DET"]["opponent"] == "SAC"            # not the 10-23 game at Miami
    assert "MIA" not in L.by_team                           # only a past game
    assert L.by_team["POR"]["opponent"] == "SAS"
