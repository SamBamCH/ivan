"""The data feed (ivan.core.feed): publish from one database, pull from a directory into another, only the feed's tables replaced."""
import sqlite3

from ivan.core import feed


def _db(path, rows):
    con = sqlite3.connect(path)
    con.execute("create table nba_player (id integer primary key, name text)")
    con.executemany("insert into nba_player values (?, ?)", rows)
    con.execute("create index ix_player_name on nba_player(name)")
    con.execute("create table nba_stats (id integer primary key, pts real)"); con.execute("insert into nba_stats values (1, 30.0)")
    con.commit(); con.close()


def test_publish_pull_from_directory_replaces_only_feed_tables(tmp_path):
    src = tmp_path / "producer.sqlite"; _db(src, [(1, "a"), (2, "b")])
    m = feed.publish("nba", out_dir=tmp_path / "feed", which="both", log=lambda s: None, db_path=src)
    assert m["live"]["bytes"] > 0 and m["reference"]["bytes"] > 0 and "nba_player" in m["live"]["tables"]
    dst = tmp_path / "consumer.sqlite"
    con = sqlite3.connect(dst)
    con.execute("create table lineup (id integer primary key)"); con.execute("insert into lineup values (7)")
    con.execute("create table nba_player (id integer primary key, name text)"); con.execute("insert into nba_player values (9, 'stale')"); con.commit(); con.close()
    changed = feed.pull(f"file://{tmp_path / 'feed'}", None, "nba", tmp_path / "state", db_path=dst, log=lambda s: None)
    assert "nba_player" in changed["live"] and "nba_stats" in changed["reference"]
    con = sqlite3.connect(dst)
    assert con.execute("select name from nba_player order by id").fetchall() == [("a",), ("b",)]
    assert con.execute("select id from lineup").fetchall() == [(7,)]                      # the consumer's own table survives
    assert con.execute("select count(*) from sqlite_master where name='ix_player_name'").fetchone()[0] == 1
    assert feed.pull(f"file://{tmp_path / 'feed'}", None, "nba", tmp_path / "state", db_path=dst, log=lambda s: None) == {}   # nothing new


def test_live_bundle_mirrors_the_projection_store_whole(tmp_path):
    """The producer's projection store travels with the live bundle and replaces the consumer's copy whole: a team the producer
    dropped is dropped on the consumer too."""
    prod = tmp_path / "producer"; cons = tmp_path / "consumer"
    (prod / "data" / "nba").mkdir(parents=True); (cons / "data" / "nba").mkdir(parents=True)
    _db(prod / "data" / "nba" / "nba_history.sqlite", [(1, "a")]); _db(cons / "data" / "nba" / "nba_history.sqlite", [(9, "stale")])
    store = prod / "data" / "nba" / "derived" / "projection"; store.mkdir(parents=True)
    (store / "BOS.pkl").write_bytes(b"bos-1"); (store / "NYK.pkl").write_bytes(b"nyk-1")
    cstore = cons / "data" / "nba" / "derived" / "projection"; cstore.mkdir(parents=True); (cstore / "LAL.pkl").write_bytes(b"old")
    m = feed.publish("nba", out_dir=tmp_path / "feed", which="live", log=lambda s: None, root=prod)
    assert m["live_mirror"]["count"] == 2 and m["live_mirror"]["dirs"] == ["data/nba/derived/projection"]
    changed = feed.pull(f"file://{tmp_path / 'feed'}", None, "nba", tmp_path / "state", which="live", log=lambda s: None, root=cons)
    assert changed["live_mirror"] == 2
    assert sorted(p.name for p in cstore.iterdir()) == ["BOS.pkl", "NYK.pkl"] and (cstore / "BOS.pkl").read_bytes() == b"bos-1"
    # the next publish without NYK removes it from the consumer on the next pull
    (store / "NYK.pkl").unlink()
    feed.publish("nba", out_dir=tmp_path / "feed", which="live", log=lambda s: None, root=prod)
    feed.pull(f"file://{tmp_path / 'feed'}", None, "nba", tmp_path / "state", which="live", log=lambda s: None, root=cons)
    assert sorted(p.name for p in cstore.iterdir()) == ["BOS.pkl"]


def test_push_is_a_no_op_without_a_destination(tmp_path, monkeypatch):
    monkeypatch.delenv(feed.PUSH_ENV, raising=False)
    assert feed.push("nba", out_dir=tmp_path, log=lambda s: None) is False
