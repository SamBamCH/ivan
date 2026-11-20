"""One game of a past slate (ivan.sports.nba.backtest.game): the games from the tip records, the edits, the World's inputs, the charts."""
import json
from datetime import date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ivan.sports.nba.backtest import game as G
from ivan.sports.nba.backtest import rerun as RR
from ivan.sports.nba.gamesim import world as W

RATES = dict(points=0.5, rebounds=0.2, assists=0.1, steals=0.03, blocks=0.02, turnovers=0.05, fg3m=0.05)
GAME = dict(key="AAA-BBB", home="BBB", away="AAA", tip=datetime(2026, 11, 18, 19, 0), record=datetime(2026, 11, 18, 19, 0), sided=True, spread_home=-4.0, total=224.0)


def _pool():
    rows, pid = [], 0
    for team, opp in (("AAA", "BBB"), ("BBB", "AAA")):
        for i, mins in enumerate([36, 34, 32, 30, 28, 24, 20, 18, 18]):
            pid += 1
            rows.append(dict(id=pid, name=f"{team.lower()} p{i}", team=team, opponent=opp, position="G", salary=9000 - 500 * i, status="None", gametime="2026-11-18T19:00:00",
                             starter="expected" if i < 5 else "bench", minutes_proj=float(mins), fpts_mean=mins * 1.0, fpts_std=8.0, p_out=0.1 if i == 1 else None,
                             p_cut=0.2 if i >= 5 else 0.0, fringe_p=0.0, fringe_minutes=0.0, minutes_sd=4.0, rotation_cells=[1 if i < 5 else 0] + [0] * 47))
        pid += 1                                                      # in the pool, not projected, no fringe chance: outside the World until minutes are typed
        rows.append(dict(id=pid, name=f"{team.lower()} deep", team=team, opponent=opp, position="C", salary=3000, status="None", gametime="2026-11-18T19:00:00",
                         starter="bench", minutes_proj=0.0, fpts_mean=None, fpts_std=None, p_out=None, p_cut=None, fringe_p=None, fringe_minutes=None, minutes_sd=None, rotation_cells=None))
    return rows


def _game(edits=None, **kw):
    pool = _pool()
    return G.build_game(GAME, pool, {p["name"]: RATES for p in pool}, {"AAA": 99.0}, 98.0, edits, **kw)


def test_clean_edits_keeps_only_what_was_typed():
    e = G.clean_edits(dict(minutes={"AAA P0": "30", "x": "", "y": "abc", "z": -3, "w": 400}, starters={"AAA P6": True, "q": None}, spread_home="", total="231.5"))
    assert e == dict(minutes={"aaa p0": 30.0, "w": 73.0}, starters={"aaa p6": True}, total=231.5)
    assert G.clean_edits(None) == {} and G.clean_edits(dict(minutes={}, starters={}, spread_home=None)) == {}


def test_build_game_without_edits_is_the_tip_record():
    g = _game()
    assert (g.home, g.away, g.spread_home, g.total, g.pace_away, g.pace_home) == ("BBB", "AAA", -4.0, 224.0, 99.0, 98.0)
    a = {p.name: p for p in g.players["AAA"]}
    assert len(a) == 9 and "aaa deep" not in a                       # the unprojected player is not drawn
    assert sum(p.minutes for p in a.values()) == 240 and a["aaa p1"].p_out == 0.1 and a["aaa p6"].p_cut == 0.2 and a["aaa p0"].cells[0] == 1


def test_typed_minutes_are_certain_and_the_rest_share_what_is_left():
    g = _game(dict(minutes={"aaa p1": 0, "aaa p0": 40, "aaa deep": 6}))
    a = {p.name: p for p in g.players["AAA"]}
    assert "aaa p1" not in a                                          # typed out
    assert a["aaa p0"].minutes == 40 and a["aaa p0"].p_cut == 0 and a["aaa deep"].minutes == 6 and a["aaa deep"].rates == RATES
    free = [p for n, p in a.items() if n not in ("aaa p0", "aaa deep")]
    assert sum(p.minutes for p in a.values()) == pytest.approx(240.0) and all(p.minutes > q for p, q in zip(free, [32, 30, 28, 24, 20, 18, 18]))
    assert a["aaa p6"].p_cut == 0.2                                   # an untyped player keeps his cut chance
    assert sum(p.minutes for p in g.players["BBB"]) == 240            # the other team is untouched


def test_every_player_typed_is_not_rescaled():
    mins = {f"aaa p{i}": m for i, m in enumerate([40, 38, 36, 34, 30, 26, 22, 20, 19])}       # an overtime game: 265
    g = _game(dict(minutes=mins))
    assert sum(p.minutes for p in g.players["AAA"]) == 265 and all(p.p_out == 0 and p.p_cut == 0 for p in g.players["AAA"])


def test_starter_calls_and_the_line():
    g = _game(dict(starters={"aaa p0": False, "aaa p6": True}, spread_home=3, total=240))
    a = {p.name: p for p in g.players["AAA"]}
    assert a["aaa p0"].cells[0] == 0 and a["aaa p6"].cells[0] == 1 and (g.spread_home, g.total) == (3.0, 240.0)
    g = _game(dict(starters={"aaa p0": False}), starter_calls=False)  # the plan rotation draws the stored rows: the call is not applied
    assert {p.name: p for p in g.players["AAA"]}["aaa p0"].cells[0] == 1


def test_chart_bins_the_draw_and_marks_the_actual():
    shares = [0.0] * 10 + [0.1] * 10                                  # 10..19, flat
    c = G.chart(shares, actual=14, mark=15.0)
    assert len(c["bins"]) == 10 and c["step"] == 2 and [b["hit"] for b in c["bins"]].index(True) == 7 and c["bins"][7]["label"] == "14-15"
    assert c["mean"] == pytest.approx(14.5) and (c["p10"], c["p90"]) == (10, 18) and c["pct"] == pytest.approx(0.45) and c["actual_at"] == pytest.approx(7.25)
    two = G.chart(shares, other=[0.0] * 5 + [0.2] * 5, actual=30)
    assert two["bins"][-1]["hit"] and two["other_mean"] == pytest.approx(7.0) and two["pct"] == 1.0 and all(b["b"] is not None for b in two["bins"])
    assert G.chart(None) is None and G.chart([]) is None


def test_percentiles():
    assert G._pct_shares([0.25, 0.5, 0.25], 1) == 0.5 and G._pct_shares([0.25, 0.5, 0.25], 9) == 1.0 and G._pct_shares(None, 1) is None
    q = [float(x) for x in range(101)]
    assert 0.29 < G._pct_quantiles(q, 30.0) < 0.31 and G._pct_quantiles(q, 500.0) == 1.0 and G._pct_quantiles(q, None) is None


def test_errors_count_everyone_projected_or_played():
    rows = [dict(min_proj=30, min_act=20, fpts_proj=30.0, fpts_act=18.0), dict(min_proj=None, min_act=10, fpts_proj=None, fpts_act=9.0),
            dict(min_proj=12, min_act=0.0, fpts_proj=10.0, fpts_act=0.0), dict(min_proj=None, min_act=None, fpts_proj=None, fpts_act=None)]
    e = G._errors(rows)
    assert e["n"] == 3 and e["min_mae"] == pytest.approx((10 + 10 + 12) / 3) and e["fpts_mae"] == pytest.approx((12 + 9 + 10) / 3) and e["fpts_bias"] == pytest.approx((12 - 9 + 10) / 3)
    assert G._errors([dict(min_proj=30, fpts_proj=30.0, min_act=None, fpts_act=None)])["fpts_mae"] is None     # no box score yet


def test_soft_pool_fills_a_record_whose_fields_are_empty():
    state = pd.DataFrame(index=pd.Index([], name="player"))
    with_values = [dict(name="a", minutes_proj=20.0, p_cut=0.3, fpts_mean=20.0)]
    assert RR.soft_pool(with_values, state) == 0 and with_values[0]["p_cut"] == 0.3
    empty = [dict(name="a", minutes_proj=20.0, p_cut=None, fringe_p=None, fpts_mean=20.0, starter="bench", status="None")]   # the key is there, the value is not
    try:
        n = RR.soft_pool(empty, state)
    except FileNotFoundError:
        pytest.skip("no cuts table on this machine")
    assert n == 1 and empty[0]["p_cut"] is not None


@pytest.fixture
def slate(tmp_path):
    """A slate directory with one tip record and a history database with the day's box scores."""
    from ivan.sports.nba.backtest import tips as T
    dg = 999
    d = T.tip_dir(dg, datetime(2026, 11, 18, 19, 0), tmp_path)
    d.mkdir(parents=True)
    pool = _pool()
    (d / "players.json").write_text(json.dumps(pool))
    (d / "meta.json").write_text(json.dumps(dict(draft_group=dg, tip="2026-11-18T19:00")))
    eng = create_engine("sqlite://")
    cols = "game_id text, game_date text, player_name text, team_abbreviation text, minutes real, points real, rebounds real, assists real, steals real, blocks real, turnovers real, fg3m real, fga real, fta real, dk_fpts real"
    with eng.begin() as c:
        c.execute(text(f"create table nba_stats ({cols})"))
        for p in pool:
            if p["fpts_mean"] is None or p["name"] == "aaa p1":                                     # the deep men and one projected player sat
                continue
            m = p["minutes_proj"] - 4
            c.execute(text("insert into nba_stats values ('g1', '2026-11-18', :n, :t, :m, :pts, 4, 3, 1, 0, 2, 1, 10, 3, :f)"),
                      dict(n=p["name"].upper(), t=p["team"], m=m, pts=50 if p["team"] == "AAA" else 55, f=m * 1.0 + (20 if p["name"] == "bbb p0" else 0)))
        c.execute(text("insert into nba_stats values ('g1', '2026-11-18', 'New Guy', 'AAA', 12, 6, 4, 3, 1, 0, 2, 1, 10, 3, 14)"))    # played, not in the pool
    G._GAMES.clear()
    return dg, tmp_path, sessionmaker(bind=eng)()


def test_slate_games_and_overview_from_the_tip_record(slate):
    dg, root, session = slate
    games = G.slate_games(dg, root)
    assert len(games) == 1 and games[0]["tip"] == datetime(2026, 11, 18, 19, 0) and games[0]["record"] == datetime(2026, 11, 18, 19, 0) and games[0]["total"] is None
    assert G.find_game(dg, "bbb", root)["key"] == games[0]["key"] and G.find_game(dg, "ZZZ", root) is None
    (row,) = G.overview(session, dg, root=root)
    assert row["can_open"] and not row["drawn"] and row["final"]["total"] == 50 * 9 + 55 * 9 - 50 + 6 and row["n"] == 18      # the 18 projected; the one who played from outside the pool is not a row
    aaa = next(t for t in row["teams"] if t["team"] == "AAA")
    assert aaa["fpts_proj"] == 240.0 and aaa["n"] == 9 and aaa["min_mae"] == pytest.approx((8 * 4 + 34) / 9)
    assert row["misses"][0]["name"] == "aaa p1" and row["misses"][0]["delta"] == -34.0 and row["misses"][1]["name"] == "bbb p0"


def test_published_rows_and_actual_edits(slate):
    dg, root, session = slate
    g = G.slate_games(dg, root)[0]
    rows = G.published_rows(dg, g, G.box(session, date(2026, 11, 18), ("AAA", "BBB")), root)
    assert {r["name"]: r["min_act"] for r in rows["AAA"]}["aaa p1"] == 0.0 and "aaa deep" not in {r["name"] for r in rows["AAA"]}
    with session.get_bind().begin() as c:
        c.execute(text("create table nba_stint (game_id text, game_date text, team text, player_name text, stint_no int, start_sec real, end_sec real, minutes real, starter int)"))
    e = G.actual_edits(session, dg, g["key"], root)
    assert e["minutes"]["aaa p0"] == 32.0 and e["minutes"]["aaa p1"] == 0.0 and "aaa deep" not in e["minutes"] and "starters" not in e     # no stints: the starters are left alone
    game = G.build_game(g, _pool(), {}, {}, 98.0, e)
    assert sum(p.minutes for p in game.players["AAA"]) == sum(m - 4 for m in [36, 32, 30, 28, 24, 20, 18, 18])


def test_edits_are_stored_per_game_and_reset(slate):
    dg, root, _ = slate
    assert G.load_edits(dg, "AAA-BBB", root) == {}
    G.save_edits(dg, "AAA-BBB", dict(minutes={"AAA P0": 30}, total=230), root)
    assert G.load_edits(dg, "AAA-BBB", root) == dict(minutes={"aaa p0": 30.0}, total=230.0)
    (G.game_dir(dg, "AAA-BBB", root) / "what_if.json").write_text("{}"); (G.game_dir(dg, "AAA-BBB", root) / "base.json").write_text(json.dumps(dict(meta={}, players={})))
    G.reset(dg, "AAA-BBB", root)
    assert G.load_edits(dg, "AAA-BBB", root) == {} and G.load_draw(dg, "AAA-BBB", "what_if", root) is None and G.load_draw(dg, "AAA-BBB", "base", root) is not None


def test_the_backtest_templates_compile():
    from pathlib import Path
    from jinja2 import Environment, FileSystemLoader
    import ivan.web
    env = Environment(loader=FileSystemLoader(str(Path(ivan.web.__file__).parent / "templates")))
    for t in ("_nba_backtest.html", "_nba_bt_fold.html", "_nba_bt_game.html", "_nba_bt_player.html"):
        env.get_template(t)
