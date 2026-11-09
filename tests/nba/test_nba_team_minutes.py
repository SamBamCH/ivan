"""The team minutes draw (ivan.sports.nba.projection.team_minutes) and the game-lines consensus (projection.lines)."""
import pandas as pd

from ivan.sports.nba.projection import team_minutes as TM
from ivan.sports.nba.projection.lines import GameLines, consensus


def test_scratch_probability_and_roles():
    assert TM.scratch_probability("Q", 1.0, 30, TM.DEFAULTS) == 0.35 and TM.scratch_probability(None, 0.4, 8, TM.DEFAULTS) == 0.0   # play rate is in the allocation
    assert TM.scratch_probability("gtd", params=TM.DEFAULTS) == 0.35 and TM.scratch_probability(None, 0.4, 30, TM.DEFAULTS) == 0.0
    assert [TM.role_of(m, s) for m, s in ((34, True), (22, True), (22, False), (12, False), (4, False))] == ["star", "starter", "rotation", "rotation", "deep"]


def test_lines_consensus_and_team_lookup():
    rows = pd.DataFrame([
        dict(game_id="g", commence_time=pd.Timestamp("2026-10-21 18:00"), home_abbr="WAS", away_abbr="MIL", bookmaker="a", market="spreads", side="home", point=-4.5, price=-110),
        dict(game_id="g", commence_time=pd.Timestamp("2026-10-21 18:00"), home_abbr="WAS", away_abbr="MIL", bookmaker="b", market="spreads", side="home", point=-5.5, price=-110),
        dict(game_id="g", commence_time=pd.Timestamp("2026-10-21 18:00"), home_abbr="WAS", away_abbr="MIL", bookmaker="a", market="totals", side="over", point=239.0, price=-110),
        dict(game_id="g", commence_time=pd.Timestamp("2026-10-21 18:00"), home_abbr="WAS", away_abbr="MIL", bookmaker="a", market="h2h", side="home", point=None, price=-180),
        dict(game_id="g", commence_time=pd.Timestamp("2026-10-21 18:00"), home_abbr="WAS", away_abbr="MIL", bookmaker="a", market="h2h", side="away", point=None, price=150),
    ])
    c = consensus(rows)
    assert len(c) == 1 and c.iloc[0].spread_home == -5.0 and c.iloc[0].total == 239.0 and 0.6 < c.iloc[0].p_home < 0.65
    gl = GameLines(c, tips={"WAS": pd.Timestamp("2026-10-21 18:00").to_pydatetime(), "MIL": pd.Timestamp("2026-10-21 18:00").to_pydatetime()})
    assert gl.team_spread("WAS") == -5.0 and gl.team_spread("MIL") == 5.0 and gl.team_total("MIL") == 239.0 and gl.team_spread("BOS") is None

