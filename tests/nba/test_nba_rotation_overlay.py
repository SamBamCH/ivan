"""The team page's "vs our grid" on a past game: our grid for that day (slate_backtest.grid_for_day: drawn that day, else the
archive's grid rows at that day's lock) over the played grid (Desk.overlay_ours)."""
import json
from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ivan.core import archive
from ivan.sports.nba import slate_backtest as SB
from ivan.sports.nba import slates as S
from ivan.sports.nba.desk.rotation import RotationMixin


@pytest.fixture
def sess(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.sqlite'}", future=True)
    archive.ensure_tables(engine)
    return sessionmaker(bind=engine)()


def test_grid_for_day_prefers_the_drawn_grid_then_the_archive(tmp_path, monkeypatch, sess):
    monkeypatch.setattr(S, "SLATES_DIR", tmp_path / "slates")
    day = date(2026, 11, 4)
    # the archive: a capture of a slate of that day with grid rows for BOS (another team's rows are not ours)
    lock = datetime(2026, 11, 4, 18, 0)
    archive.archive_slate(sess, sport="nba", slate_key="155", lock=lock, mode="classic", source="tip", captured_at=datetime(2026, 11, 4, 17, 58),
                          players=[dict(name="A B", team="BOS", rotation_cells=[1] * 30 + [0] * 18),
                                   dict(name="C D", team="BOS", rotation_cells=[0] * 24 + [1] * 24),
                                   dict(name="Z Z", team="NYK", rotation_cells=[1] * 48)])
    g, src = SB.grid_for_day(sess, "BOS", day)
    assert list(g.index) == ["a b", "c d"] and int(g.loc["a b"].sum()) == 30 and "17:58" in src and "155" in src
    assert SB.grid_for_day(sess, "BOS", date(2026, 11, 5))[0] is None                      # no slate of that day
    # a grid drawn on the desk that day wins
    p = S.days_dir() / day.isoformat() / "edits" / "rotations" / "BOS.json"; p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(players=["a b"], grid=[[1] * 20 + [0] * 28], saved_at="2026-11-04T17:30:00")))
    g, src = SB.grid_for_day(sess, "BOS", day)
    assert list(g.index) == ["a b"] and int(g.loc["a b"].sum()) == 20 and src.startswith("drawn on the desk")


def test_overlay_marks_both_act_ours_and_adds_dnp_rows(monkeypatch):
    ours = pd.DataFrame([[True] * 30 + [False] * 18, [True] * 10 + [False] * 38], index=["a b", "e f"], columns=list(range(48)))
    monkeypatch.setattr(SB, "grid_for_day", lambda session, team, day, root=None: (ours, "the projection at 17:58"))
    desk = SimpleNamespace(session=None, players=lambda team: [SimpleNamespace(name="E F", position="C", salary=4000, status=None)])
    rot = dict(kind="last", last_game="2026-11-04", rows=[
        dict(name="a b", key="a b", cells=[True] * 24 + [False] * 24, minutes=24),
        dict(name="c d", key="c d", cells=[False] * 24 + [True] * 24, minutes=24)])
    RotationMixin.overlay_ours(desk, "BOS", rot)
    a, c, e = rot["rows"]
    assert a["cmp"][0] == "both" and a["cmp"][26] == "ours" and a["min_ours"] == 30
    assert c["cmp"][30] == "act" and c["min_ours"] == 0
    assert e["dnp"] and e["key"] == "e f" and e["min_ours"] == 10 and e["cmp"][0] == "ours" and e["pos"] == "C"
    assert rot["overlay"]["found"] and rot["overlay"]["agreement"] == pytest.approx(24 / 240)
    # no grid of ours for that day: the toggle says so, the rows stay as played
    monkeypatch.setattr(SB, "grid_for_day", lambda session, team, day, root=None: (None, "no grid of ours on file"))
    rot = dict(kind="last", last_game="2026-11-04", rows=[dict(name="a b", key="a b", cells=[True] * 48, minutes=48)])
    RotationMixin.overlay_ours(desk, "BOS", rot)
    assert rot["overlay"]["found"] is False and "cmp" not in rot["rows"][0]
