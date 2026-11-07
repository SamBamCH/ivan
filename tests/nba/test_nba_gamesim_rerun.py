"""The as-of rerun's pieces (ivan.sports.nba.gamesim.rerun): the histogram, the scorecard, the lines from a tip record, the player view."""
import numpy as np
import pandas as pd

from ivan.sports.nba.gamesim import rerun as RR
from ivan.sports.nba.gamesim import world as W


def test_histogram_marks_the_actual_and_the_band():
    rng = np.random.default_rng(0)
    x = rng.normal(30, 5, 2000)
    h = RR.histogram(x, 30.0)
    assert len(h["bars"]) == RR.HIST_BINS and max(h["bars"]) == 1.0 and h["band"][0] < h["at"] < h["band"][1] and 0.4 < h["pit"] < 0.6
    far = RR.histogram(x, 60.0)
    assert far["hi"] >= 60.0 and far["at"] == RR.HIST_BINS - 1 and far["pit"] > 0.99
    none = RR.histogram(x, None)
    assert none["at"] is None and none["pit"] is None
    flat = RR.histogram(np.full(100, 12.0), 12.0)
    assert flat["hi"] > flat["lo"]


def _world():
    n = 500; rng = np.random.default_rng(1)
    names = ["a", "b", "c"]; pids = [1, 2, 3]; team_of = ["AAA", "AAA", "BBB"]
    stats = {s: np.zeros((n, 3)) for s in ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov", "minutes", "fouls", "fga", "fta")}
    stats["minutes"][:] = rng.normal([30, 20, 25], 3, (n, 3)); stats["pts"][:] = rng.poisson([20, 10, 15], (n, 3)); stats["reb"][:] = rng.poisson([5, 3, 8], (n, 3))
    stats["fga"][:] = rng.poisson([15, 8, 12], (n, 3)); stats["fta"][:] = rng.poisson([4, 2, 3], (n, 3))
    fpts = W.dk_points(stats)
    games = [dict(home="AAA", away="BBB", P=100.0, spread_home=-4.0, total=226.0, margin=rng.normal(4, 13, n), total_pts=rng.normal(226, 17, n), poss=rng.normal(100, 4, n),
                  overtime=np.zeros(n, dtype=int), garbage_at=np.full(n, -1), factor={"AAA": 1.0, "BBB": 1.0}, exp_points={"AAA": 113, "BBB": 113}, target_points={"AAA": 115, "BBB": 111})]
    from datetime import datetime
    return W.World(n=n, seed=1, names=names, pids=pids, team_of=team_of, stats=stats, fpts=fpts, games=games, drawn_at=datetime.now())


def test_scorecard_rows_per_layer():
    w = _world()
    act = pd.DataFrame([dict(game_id="g", player_name="A", team="AAA", player="a", minutes=31, points=22, fg3m=1, rebounds=4, assists=3, steals=1, blocks=0, turnovers=2, fga=16, fta=4, dk_fpts=38.5),
                        dict(game_id="g", player_name="B", team="AAA", player="b", minutes=0, points=0, fg3m=0, rebounds=0, assists=0, steals=0, blocks=0, turnovers=0, fga=0, fta=0, dk_fpts=0),
                        dict(game_id="g", player_name="C", team="BBB", player="c", minutes=26, points=14, fg3m=0, rebounds=9, assists=2, steals=1, blocks=1, turnovers=1, fga=11, fta=4, dk_fpts=31.25)])
    games_act = {("AAA", "BBB"): dict(poss=101.0, pts_home=118.0, pts_away=110.0, overtime=0)}
    published = {1: dict(minutes=29.0, fpts=36.0), 3: dict(minutes=24.0, fpts=30.0)}
    card = pd.DataFrame(RR.scorecard(w, act, games_act, published)).set_index(["layer", "metric"])
    assert card.loc[("possessions", "poss"), "n"] == 1 and abs(card.loc[("possessions", "poss"), "mae"] - abs(w.games[0]["poss"].mean() - 101.0)) < 1e-6
    assert card.loc[("points and margin", "total"), "mae_published"] == 2.0 and card.loc[("points and margin", "margin"), "mae_published"] == 4.0
    assert card.loc[("minutes", "minutes"), "n"] == 2                                          # the DNP is left out
    assert card.loc[("fantasy points", "fpts"), "mae_published"] is not None and 0.0 <= card.loc[("fantasy points", "fpts"), "in_band"] <= 1.0
    assert card.loc[("ownership and the field", "ownership"), "n"] == 0
    assert set(card.index.get_level_values(0)) == {l for l, _, _ in RR.LAYERS} | {"ownership and the field"}


def test_lines_from_record_and_player_view():
    rows = [dict(game_id="x", commence_time="2026-11-06 23:00:00", home_abbr="AAA", away_abbr="BBB", bookmaker="dk", market="spreads", side="home", point=-4.0, price=-110),
            dict(game_id="x", commence_time="2026-11-06 23:00:00", home_abbr="AAA", away_abbr="BBB", bookmaker="dk", market="spreads", side="away", point=4.0, price=-110),
            dict(game_id="x", commence_time="2026-11-06 23:00:00", home_abbr="AAA", away_abbr="BBB", bookmaker="dk", market="totals", side="over", point=226.0, price=-110),
            dict(game_id="x", commence_time="2026-11-06 23:00:00", home_abbr="AAA", away_abbr="BBB", bookmaker="dk", market="h2h", side="home", point=None, price=-180),
            dict(game_id="x", commence_time="2026-11-06 23:00:00", home_abbr="AAA", away_abbr="BBB", bookmaker="dk", market="h2h", side="away", point=None, price=150)]
    players = [dict(team="AAA", gametime="2026-11-06T18:00:00"), dict(team="BBB", gametime="2026-11-06T18:00:00")]
    by = RR.lines_from_record(rows, players)
    assert by["AAA"]["spread"] == -4.0 and by["AAA"]["total"] == 226.0 and by["AAA"]["home"] and by["BBB"]["spread"] == 4.0 and not by["BBB"]["home"]
    assert RR.lines_from_record([], players) == {}
    rec = dict(arrays={"fpts": np.tile([[30.0, 10.0]], (50, 1)), "minutes": np.tile([[32.0, 12.0]], (50, 1)), "pts": np.tile([[20.0, 5.0]], (50, 1))},
               players=[dict(name="a", pid=1, team="AAA", published=dict(minutes=30, fpts=28), actual=dict(minutes=31, fpts=35, pts=22)),
                        dict(name="b", pid=2, team="AAA", published=None, actual=None)])
    v = RR.player_view(rec, "pts")
    assert list(v) == ["AAA"] and v["AAA"][0]["name"] == "a" and v["AAA"][0]["played"] and v["AAA"][0]["stat_act"] == 22 and v["AAA"][0]["hist"]["at"] is not None
    assert not v["AAA"][1]["played"] and v["AAA"][1]["hist"]["at"] is None
