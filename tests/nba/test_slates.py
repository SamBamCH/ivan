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
    assert S.main_slate_today(now) == 102 and S.resolve(None, now=now) == 102                 # the test's clock, not the wall's
    assert S.resolve("101") == 101
    with pytest.raises(S.NoSlate):
        S.open(99)


def test_open_any_falls_back_to_the_newest_held_slate(root, monkeypatch):
    """A script that spans slates (standings, the field backtest) runs the morning after: with no open slate it opens the newest
    held one; --slate still wins; nothing held is still an error."""
    now = datetime(2026, 11, 6, 8, 0)
    monkeypatch.setattr(S, "main_slate_today", lambda now_=None, root=None: None)
    with pytest.raises(S.NoSlate):
        S.open_any(None)
    S.create(100, lock=now - timedelta(days=2), games=8)
    S.create(101, lock=now - timedelta(days=1), games=3)
    dg, sess = S.open_any(None)
    assert dg == 101
    sess.close()
    dg, sess = S.open_any(100)
    assert dg == 100
    sess.close()


def test_edits_are_the_days_shared_by_its_slates(root):
    """A slate's pins, grids and scenarios live under data/nba/days/<day>/edits/: the two slates of one evening share them, the next
    day has its own, and a slate whose day is not on record yet keeps them under itself."""
    S.create(200, lock=datetime(2026, 11, 4, 18, 0), games=8)
    S.create(201, lock=datetime(2026, 11, 4, 18, 30), games=6)
    S.create(202, lock=datetime(2026, 11, 5, 18, 0), games=9)
    a, b, c = S.paths(200), S.paths(201), S.paths(202)
    assert a.day == "2026-11-04" and a.day_root == root / "days" / "2026-11-04"
    assert a.edits == b.edits == root / "days" / "2026-11-04" / "edits" and a.published == b.published
    assert c.edits == root / "days" / "2026-11-05" / "edits" and a.scenarios != c.scenarios
    assert a.rotations.is_dir() and not (a.root / "edits").exists()                             # created under the day, not the slate
    a.overrides_csv.write_text("player,minutes\nx,30\n")
    with S.using(201):
        assert S.overrides_csv().read_text().startswith("player") and S.scenarios_dir() == a.scenarios
    with S.using(202):
        assert not S.overrides_csv().exists()
    assert S.paths(300).edits == root / "slates" / "300" / "edits"                               # no record: under the slate




def test_a_published_layer_counts_only_on_a_consumer(root, monkeypatch):
    """A producer ignores a published pins file or grid it may find under the day (only a machine that pulls a feed has one)."""
    S.create(400, lock=datetime(2026, 11, 4, 18, 0))
    sp = S.paths(400)
    sp.published.mkdir(parents=True); sp.published_overrides_csv.write_text("player,minutes\na,30\n")
    sp.published_rotations.mkdir(); (sp.published_rotations / "ATL.json").write_text("{}")
    monkeypatch.delenv("IVAN_FEED_SSH", raising=False); monkeypatch.delenv("IVAN_FEED_URL", raising=False)
    with S.using(400):
        assert not S.is_consumer() and S.published_pins() is None and S.published_grid("ATL") is None
        monkeypatch.setenv("IVAN_FEED_SSH", "me@laptop")
        assert S.is_consumer() and S.published_pins() == sp.published_overrides_csv and S.published_grid("ATL") == sp.published_rotations / "ATL.json"


def test_overrides_file_carries_stat_totals_and_reads_older_files(root):
    """A stat pin (points, rebounds, ...) is a column of minutes_overrides.csv like the minutes pin; a file written before the
    stat columns reads with them blank; the roster hands the engine the totals as stat_overrides."""
    from ivan.sports.nba.projection.live import OVERRIDE_KEYS, STAT_KEYS, read_overrides_full, write_overrides
    assert set(STAT_KEYS) <= set(OVERRIDE_KEYS) and STAT_KEYS == ("points", "rebounds", "assists", "fg3m", "steals", "blocks", "turnovers")
    S.create(400, lock=datetime(2026, 11, 7, 19, 0), games=4)
    with S.using(400):
        write_overrides({"a": {"minutes": 34.0, "rebounds": 9.0}, "b": {"points": 30.0}})
        got = read_overrides_full()
        assert got["a"]["minutes"] == 34.0 and got["a"]["rebounds"] == 9.0 and got["a"]["points"] is None and got["b"]["points"] == 30.0
        header = S.overrides_csv().read_text().splitlines()[0]
        assert header == "player," + ",".join(OVERRIDE_KEYS)
        S.overrides_csv().write_text("player,minutes,fppm\nc,28,1.1\n")                # the file as it was before the stat columns
        got = read_overrides_full()
        assert got["c"]["minutes"] == 28.0 and got["c"]["fppm"] == 1.1 and all(got["c"][k] is None for k in STAT_KEYS)


def test_status_marks_carry_across_the_pool_replace_and_move_on_a_downgrade(root):
    """changes.mark_status_changes: after the pool is replaced, a player downgraded (to an out status, out of the five) is stamped
    now, one upgraded (into the five, back from out) loses his stamp, questionable either way is nothing, the rest carry their
    earlier stamp, a newcomer carries none; touch_status stamps or clears one player; status_cutoffs lists the stamped."""
    from ivan.sports.nba import changes as CH
    from ivan.sports.nba.models import NBAPlayer, StartingLineups
    S.create(500, lock=datetime(2026, 11, 7, 19, 0), games=2)
    sess = S.session(500)
    tip = datetime(2026, 11, 7, 19, 0)
    def pool(status_a, names=("a one", "b two", "c three")):
        sess.query(NBAPlayer).delete(synchronize_session=False)
        for i, n in enumerate(names):
            sess.add(NBAPlayer(name=n, team="AAA", opponent="BBB", position="PG", roster_slot_id=1, gametime=tip, salary=5000, draft_group_id=500,
                               status=status_a if n == "a one" else None))
        sess.flush()
    def five(names):
        sess.query(StartingLineups).delete(synchronize_session=False)
        for n in names:
            sess.add(StartingLineups(player_name=n, team="AAA", position="PG", confirmed=False))
        sess.flush()
    with S.using(500):
        pool(None); five(["a one", "b two"])
        t0 = datetime(2026, 11, 7, 15, 0)
        assert CH.mark_status_changes(sess, CH.status_state(sess), now=t0) == []             # nothing moved: nothing stamped
        before = CH.status_state(sess)
        assert before["a one"] == dict(status="", starter=True, at=None)
        pool("Q"); five(["b two", "c three"])                                                # a one: Q (nothing) and out of the five (a downgrade); c three: into the five (an upgrade)
        t1 = datetime(2026, 11, 7, 17, 30)
        assert CH.mark_status_changes(sess, before, now=t1) == ["a one"]
        assert CH.status_cutoffs(sess) == {"a one": t1}
        before = CH.status_state(sess)
        pool(None, names=("a one", "b two", "c three", "d four")); five(["b two", "c three"])   # a one Q -> active: nothing; d four is new
        t2 = datetime(2026, 11, 7, 18, 0)
        assert CH.mark_status_changes(sess, before, now=t2) == [] and CH.status_cutoffs(sess) == {"a one": t1}
        before = CH.status_state(sess)
        pool("OUT", names=("a one", "b two", "c three", "d four")); five(["a one", "b two"])    # a one ruled out (a downgrade, stamped again) ... and c three out of the five
        t3 = datetime(2026, 11, 7, 18, 10)
        assert CH.mark_status_changes(sess, before, now=t3) == ["a one", "c three"] and CH.status_cutoffs(sess) == {"a one": t3, "c three": t3}
        before = CH.status_state(sess)
        pool(None, names=("a one", "b two", "c three", "d four")); five(["a one", "b two", "c three"])   # a one ruled in, c three back in the five: both stamps lifted
        assert CH.mark_status_changes(sess, before, now=datetime(2026, 11, 7, 18, 20)) == [] and CH.status_cutoffs(sess) == {}
        assert CH.touch_status(sess, "b two", now=t2) and CH.status_cutoffs(sess)["b two"] == t2
        assert CH.touch_status(sess, "b two", now=t3, clear=True) and "b two" not in CH.status_cutoffs(sess)
        assert not CH.touch_status(sess, "nobody", now=t2)
    sess.close()


def test_market_rates_are_current_only_for_the_odds_table_they_were_built_from(root, monkeypatch):
    """ingest.market_rates_current: the rates carry the odds table's (rows, max id) at their build; a props fetch replaces rows, so a
    different stamp means the books moved. No odds at all: current. Rates without a stamp (from before it): moved, once."""
    import pandas as pd
    from ivan.sports.nba import ingest as I
    from ivan.sports.nba.projection.live import store_market_rates
    S.create(600, lock=datetime(2026, 11, 7, 19, 0), games=2)
    sess = S.session(600)
    with S.using(600):
        monkeypatch.setattr(I, "odds_stamp", lambda session: (0, None))
        assert I.market_rates_current(sess)                                                     # no odds: nothing to follow
        monkeypatch.setattr(I, "odds_stamp", lambda session: (500, 9000))
        assert not I.market_rates_current(sess)                                                 # odds, no rates yet: build them
        means = pd.DataFrame([dict(player="a one", stat="points", mean=20.0, pairs=4, books=3, line=19.5)])
        store_market_rates(sess, means, {"a one": 30.0}, odds_stamp=(500, 9000)); sess.commit()
        assert I.market_rates_current(sess)
        monkeypatch.setattr(I, "odds_stamp", lambda session: (520, 9400))                      # a fetch replaced rows
        assert not I.market_rates_current(sess)
        store_market_rates(sess, means, {"a one": 30.0}); sess.commit()                       # rates from before the stamp
        assert not I.market_rates_current(sess)
    sess.close()

