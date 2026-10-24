"""Live box score parsing (ivan.sports.nba.post_lock.live) on a hand-written payload in NBA's live-data shape."""
from ivan.sports.nba.post_lock import live as L


def test_clock_elapsed_and_dk_points():
    assert L.minutes_of("PT24M30.00S") == 24.5 and L.clock_seconds("PT05M00.00S") == 300
    assert L.elapsed_seconds(1, "PT12M00.00S") == 0 and L.elapsed_seconds(2, "PT06M00.00S") == 1080 and L.elapsed_seconds(5, "PT05M00.00S") == 2880
    st = dict(points=20, reboundsTotal=10, assists=5, steals=1, blocks=0, turnovers=2, threePointersMade=2)
    assert L.dk_points(st) == 20 + 1 + 12.5 + 7.5 + 2 - 1 + 1.5          # double-double bonus


def test_parse_boxscore_margins_and_rows():
    game = {"gameStatus": 2, "period": 2, "gameClock": "PT06M00.00S",
            "homeTeam": {"teamTricode": "AAA", "score": 60, "players": [
                {"name": "A One", "starter": "1", "oncourt": "1", "played": "1",
                 "statistics": {"minutes": "PT14M00.00S", "points": 18, "reboundsTotal": 3, "assists": 2, "foulsPersonal": 2}}]},
            "awayTeam": {"teamTricode": "BBB", "score": 48, "players": [
                {"name": "B One", "starter": "0", "oncourt": "0", "played": "0", "statistics": {"minutes": ""}}]}}
    df = L.parse_boxscore(game).set_index("player_name")
    assert df.loc["a one", "margin"] == 12 and df.loc["b one", "margin"] == -12 and df.loc["a one", "elapsed"] == 1080
    assert df.loc["a one", "minutes"] == 14 and df.loc["a one", "fpts"] == 18 + 3.75 + 3 and bool(df.loc["a one", "starter"])
    assert df.loc["b one", "minutes"] == 0 and not df.loc["b one", "played"]
