"""A slate is a directory (ivan.sports.nba.slates): its own database with the history attached, its paths, its record, the index of
held slates, the current slate in context."""
import sqlite3
from datetime import datetime, timedelta

import pytest

from ivan.sports.nba import slates as S


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SLATES_DIR", tmp_path / "slates")
    hist = tmp_path / "nba_history.sqlite"
    con = sqlite3.connect(hist)
    con.execute("create table nba_stats (id integer primary key, player_name text, minutes real)")
    con.execute("insert into nba_stats values (1, 'a b', 30.0)")
    con.commit(); con.close()
    monkeypatch.setattr(S, "history_db_path", lambda: hist)
    return tmp_path


def test_paths_meta_and_the_current_slate(root):
    with pytest.raises(S.NoSlate):
        S.current()
    with S.using(155):
        assert S.current() == 155
        sp = S.paths()
        assert sp.root == root / "slates" / "155" and sp.overrides_csv == sp.root / "edits" / "minutes_overrides.csv"
        assert S.store_dir() == sp.root / "projection" and S.entries_csv() == sp.root / "input" / "entries.csv"
        assert S.output_dir() == sp.root / "output" and S.published_rotations_dir() == sp.root / "published" / "rotations"
    with pytest.raises(S.NoSlate):
        S.current()
    lock = datetime(2026, 11, 4, 18, 0)
    S.create(155, lock=lock, label="main", games=11)
    m = S.read_meta(155)
    assert m["draft_group"] == 155 and m["lock"] == "2026-11-04T18:00:00" and m["day"] == "2026-11-04" and m["games"] == 11 and m["created_at"]
    S.touch(155, "refreshed", by="autopilot")
    m = S.read_meta(155)
    assert m["refreshed_by"] == "autopilot" and S.meta_time(m, "refreshed_at") is not None and S.meta_time(m, "lock") == lock


def test_the_slate_database_sees_the_history_through_the_attached_file(root):
    S.create(155, lock=datetime(2026, 11, 4, 18, 0))
    sess = S.session(155)
    from sqlalchemy import text
    from ivan.sports.nba.models import NBAPlayer
    assert sess.execute(text("select count(*) from nba_stats")).scalar() == 1                 # the history, through `hist`
    assert sess.query(NBAPlayer).count() == 0                                                  # the slate's own table, in its file
    con = sqlite3.connect(root / "slates" / "155" / "slate.sqlite")
    names = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert "nba_player" in names and "lineup" in names and "nba_stats" not in names           # no history table was created in the slate file
    con.close()
    sess.close()


def test_held_open_and_main(root):
    now = datetime(2026, 11, 4, 12, 0)
    assert S.held() == [] and S.main_slate_today(now) is None
    with pytest.raises(S.NoSlate):
        S.resolve(None)
    S.create(100, lock=now - timedelta(days=1), games=8)                                       # yesterday: held, locked
    S.create(101, lock=now + timedelta(hours=6), games=3)                                      # today, the small slate
    S.create(102, lock=now + timedelta(hours=7), games=10)                                     # today, the main slate
    S.create(103, lock=now + timedelta(days=1), games=12)                                      # tomorrow
    assert S.held() == [103, 102, 101, 100] and S.holds(101) and not S.holds(99)
    assert S.open_slates(now) == [101, 102, 103]
    assert S.main_slate_today(now) == 102 and S.resolve(None) in S.held()
    assert S.resolve("101") == 101
    with pytest.raises(S.NoSlate):
        S.open(99)
