"""considerations: rest, starters out, a new five, a player back, a big spread."""
from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd

from ivan.sports.nba import considerations as CN


def _p(name, status=None, team="AAA", opp="BBB"):
    return SimpleNamespace(name=name, status=status, team=team, opponent=opp, gametime=datetime(2026, 10, 29, 19, 0))


def _box():
    rows = []
    for d, gid in ((date(2026, 10, 26), "g1"), (date(2026, 10, 28), "g2")):
        for n, st, m in (("a one", 1, 34), ("b two", 1, 30), ("c three", 1, 28), ("d four", 1, 26), ("e five", 1, 24), ("f six", 0, 22 if gid == "g1" else 0)):
            rows.append(dict(game_id=gid, game_date=d, player_name=n, team="AAA", minutes=float(m), starter=float(st)))
    rows.append(dict(game_id="x", game_date=date(2026, 3, 1), player_name="g seven", team="CCC", minutes=20.0, starter=0.0))
    b = pd.DataFrame(rows); b["played"] = b.minutes > 0
    return b


def _state():
    names = ["a one", "b two", "c three", "d four", "e five", "f six", "g seven"]
    return pd.DataFrame({"min_ewm": [34, 30, 28, 26, 24, 22, 20], "play_ewm": [1.0] * 7, "start_ewm": [1, 1, 1, 1, 0.9, 0.1, 0.0]}, index=names)


def test_considerations():
    ps = [_p("A One", "OUT"), _p("B Two"), _p("C Three"), _p("D Four"), _p("E Five", "Q"), _p("F Six"), _p("G Seven")]
    xs = CN.team_considerations("AAA", ps, _box(), _state(), starters={"b two", "c three", "d four", "e five", "f six"},
                                confirmed=True, spread=-11.5, opponent="BBB")
    text = " | ".join(x["text"] for x in xs)
    assert "AAA on a back to back" in text
    assert "regular starter A One OUT" in text and "regular starter E Five Q" in text
    assert "new starting five" in text and "in: F Six" in text and "out of it: A One (OUT)" in text
    assert "F Six sat out the last game" in text
    assert "G Seven new to AAA: his last game was for CCC" in text
    assert "AAA favoured by 11.5" in text
    assert [x["level"] for x in xs] == sorted((x["level"] for x in xs), key=lambda l: 0 if l == "warn" else 1)


def test_no_line_and_rest():
    xs = CN.team_considerations("AAA", [_p("B Two")], _box()[_box().game_date < date(2026, 10, 27)], _state(), starters=None,
                                spread=None, tip=datetime(2026, 10, 30, 19, 0))
    kinds = {x["kind"]: x["text"] for x in xs}
    assert "rested: 3 days off" in kinds["rest"] and "no line" in kinds["spread"] and "no starting five" in kinds["lineup"]
