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


def test_push_is_a_no_op_without_a_destination(tmp_path, monkeypatch):
    monkeypatch.delenv(feed.PUSH_ENV, raising=False)
    assert feed.push("nba", out_dir=tmp_path, log=lambda s: None) is False
