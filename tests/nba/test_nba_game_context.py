"""Which played games are a picture of a team's usual rotation (ivan.sports.nba.projection.rotation.absent_regulars / game_context)."""
from datetime import date, timedelta

import pandas as pd

from ivan.sports.nba.projection import rotation as ROT


def _box(games):
    """games: [(game_id, day offset, {team: {player: (minutes, points)}})] -> the box score frame the desk holds."""
    rows = []
    for gid, k, teams in games:
        for team, players in teams.items():
            for name, (m, pts) in players.items():
                rows.append(dict(game_id=gid, game_date=date(2026, 11, 1) + timedelta(days=k), team=team, player_name=name, minutes=float(m), points=pts))
    df = pd.DataFrame(rows)
    df["played"] = df.minutes > 0
    return df


def _team(star=34, pts=110):
    return {"a star": (star, pts - 60), "b wing": (30, 30), "c big": (28, 20), "d guard": (26, 10), "e bench": (12, 0)}


def test_absent_regulars_reads_the_five_games_before():
    games = [(f"g{i}", i, {"AAA": _team(), "BBB": {"x": (30, 100)}}) for i in range(4)]
    games.append(("g4", 4, {"AAA": {k: v for k, v in _team().items() if k != "a star"}, "BBB": {"x": (30, 100)}}))     # the star sits
    games.append(("g5", 5, {"AAA": dict(_team(), **{"a star": (0, 0)}), "BBB": {"x": (30, 100)}}))                     # in the box score, did not play
    out = ROT.absent_regulars(_box(games), "AAA")
    assert out["g0"] == [] and out["g2"] == []                       # the first games of a season say nothing; then nobody is missing
    assert out["g4"] == ["a star"] and out["g5"] == ["a star"]       # a 20+ minute regular of the games before, absent
    assert "e bench" not in out["g4"]                                # a 12-minute player is not a regular


def test_game_context_shades_a_blowout_and_an_absence():
    class NoStints:                                                   # no stint table: the box score alone still judges the game
        def get_bind(self):
            raise RuntimeError("no database")
    games = [(f"g{i}", i, {"AAA": _team(), "BBB": {"x": (30, 104)}}) for i in range(4)]                                 # 110-104: close
    games.append(("g4", 4, {"AAA": _team(pts=130), "BBB": {"x": (30, 100)}}))                                           # 130-100: not close
    games.append(("g5", 5, {"AAA": {k: v for k, v in _team().items() if k != "a star"}, "BBB": {"x": (30, 58)}}))       # the star sits; 60-58
    ctx = ROT.game_context(NoStints(), _box(games), "AAA")
    assert ctx["g1"]["shade"] == "" and ctx["g1"]["margin"] == 6 and ctx["g1"]["why"] == ""
    assert ctx["g4"]["shade"] == "g-blow" and ctx["g4"]["not_close"] and not ctx["g4"]["injury"] and "+30" in ctx["g4"]["why"]
    assert ctx["g5"]["shade"] == "g-inj" and ctx["g5"]["absent"] == ["a star"] and "Star" in ctx["g5"]["tag"]
    assert ROT.game_context(NoStints(), _box(games), "BBB")["g4"]["shade"] == "g-blow"                                  # the loser's side: not close either
