"""changes.diff: what a refresh moved, per team."""
from datetime import datetime
from types import SimpleNamespace

from ivan.sports.nba import changes as CH


def _snap(players, starters, confirmed=None, dg=1):
    return dict(at=datetime.now(), draft_group=dg, starters=starters, confirmed=confirmed or {},
                players={n: dict(raw=n.title(), team=t, status=st, fpts=f) for n, (t, st, f) in players.items()})


def test_diff():
    b = _snap({"a one": ("AAA", "Q", 30.0), "b two": ("AAA", "", 20.0), "c three": ("BBB", "", 10.0)}, {"AAA": {"a one", "b two"}})
    a = _snap({"a one": ("AAA", "OUT", 0.0), "b two": ("AAA", "", 26.5), "d four": ("BBB", "", 5.0)}, {"AAA": {"b two", "e five"}}, {"AAA": True})
    news = [SimpleNamespace(player_name="b two", player_raw="B Two", headline="Will start Thursday")]
    d = CH.diff(b, a, news)
    texts = [x["text"] for x in d["AAA"]]
    assert "A One: Q → OUT" in texts and "B Two: projection 20.0 → 26.5" in texts
    assert "a one out of the starting five" in " ".join(texts).lower() and "lineup confirmed" in texts
    assert "B Two: Will start Thursday" in texts
    assert {x["text"] for x in d["BBB"]} == {"D Four added to the pool", "C Three left the pool"}


def test_another_slate_is_not_a_change():
    assert CH.diff(_snap({}, {}, dg=1), _snap({"x": ("AAA", "", 1.0)}, {}, dg=2)) == {}
