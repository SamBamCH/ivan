"""newsfeed.group_news: the desk's news table splits the slate into questions, new outs and standing outs."""
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from ivan.sports.nba import newsfeed as NF


def _player(name, team, status, salary=5000, gametime=None, opp="OPP"):
    return SimpleNamespace(name=name, team=team, status=status, salary=salary, position="SF", opponent=opp,
                           gametime=gametime or datetime(2026, 10, 29, 19, 0))


def _box():
    rows = []
    for d, gid in ((date(2026, 10, 25), "g1"), (date(2026, 10, 27), "g2")):
        rows += [dict(game_id=gid, game_date=d, player_name="a star", team="AAA", minutes=34.0, starter=1.0),
                 dict(game_id=gid, game_date=d, player_name="b bench", team="AAA", minutes=0.0 if gid == "g2" else 12.0, starter=0.0),
                 dict(game_id=gid, game_date=d, player_name="c sixth", team="AAA", minutes=22.0, starter=0.0)]
    box = pd.DataFrame(rows)
    box["played"] = box.minutes > 0
    return box


def _state():
    return pd.DataFrame({"min_ewm": [33.0, 11.0, 21.0], "play_ewm": [1.0, 0.8, 1.0]}, index=["a star", "b bench", "c sixth"])


def test_group_news_splits_by_last_game():
    now = datetime(2026, 10, 29, 12, 0)
    players = [_player("A Star", "AAA", "OUT"), _player("B Bench", "AAA", "OUT"), _player("C Sixth", "AAA", "GTD"),
               _player("D Fine", "AAA", None), _player("E New", "AAA", "O")]
    news = {"c sixth": [SimpleNamespace(headline="Sixth (ankle) questionable Wednesday", text="t", source_url="https://x", url="https://rw",
                                        injury="Ankle", first_seen=now - timedelta(minutes=30))]}
    t = NF.group_news(players, _box(), _state(), news, now=now, calls={"c sixth": "pinned 20 min"}, minutes={"c sixth": 20.0},
                      starters={"AAA": {"a star"}})
    assert [r["name"] for r in t["questionable"]] == ["c sixth"]
    q = t["questionable"][0]
    assert q["call"] == "pinned 20 min" and q["minutes"] == 20.0 and q["hist"] == 21.0
    assert q["news"]["headline"].startswith("Sixth") and q["news"]["new"] is True and q["news"]["age"] == "30m ago"
    # A played the last game (10-27, 34 min): the rotation moves tonight. B sat it out and E has no game on file: standing outs.
    assert [r["name"] for r in t["new_out"]] == ["a star"]
    a = t["new_out"][0]
    assert a["played_last"] and a["last_min"] == 34.0 and a["last_date"] == "2026-10-27" and a["starter"] is True
    assert [r["name"] for r in t["out"]] == ["b bench", "e new"]
    b = t["out"][0]
    assert b["played_last"] is False and b["last_min"] == 0.0 and b["hist"] == 11.0 * 0.8
    e = t["out"][1]                       # never in a box score: the team's last game is still the date shown, with "did not play"
    assert e["last_date"] == "2026-10-27" and e["played_last"] is False and e["last_min"] is None and e["news"] is None and e["hist"] is None


def test_group_news_orders_by_tip_then_minutes():
    now = datetime(2026, 10, 29, 12, 0)
    late, early = datetime(2026, 10, 29, 22, 0), datetime(2026, 10, 29, 19, 0)
    players = [_player("A Star", "AAA", "Q", gametime=late), _player("C Sixth", "AAA", "Q", gametime=early), _player("B Bench", "AAA", "D", gametime=early)]
    t = NF.group_news(players, _box(), _state(), {}, now=now)
    assert [r["name"] for r in t["questionable"]] == ["c sixth", "b bench", "a star"]


def test_group_news_new_flag_uses_mapped_time():
    now = datetime(2026, 10, 29, 12, 0)
    seen = now - timedelta(hours=5)
    players = [_player("C Sixth", "AAA", "Q")]
    news = {"c sixth": [SimpleNamespace(headline="h", text="", source_url=None, url="u", injury=None, first_seen=seen)]}
    assert NF.group_news(players, _box(), _state(), news, now=now)["questionable"][0]["news"]["new"] is False
    assert NF.group_news(players, _box(), _state(), news, now=now, mapped_at={"AAA": seen - timedelta(hours=1)})["questionable"][0]["news"]["new"] is True
