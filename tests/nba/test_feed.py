"""The data feed (ivan.core.feed): publish from one database, pull from a directory into another, only the feed's tables replaced."""
import shutil
import sqlite3
import subprocess

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


def test_pull_ssh_asks_the_producer_to_publish_then_rsyncs_then_applies(tmp_path, monkeypatch):
    """The consumer's Refresh: one ssh 'publish', one read-only rsync of the producer's feed directory into ours, then the
    ordinary pull from that directory. ssh and rsync are stood in for: the producer's feed is published into the directory
    the rsync would have filled."""
    prod = tmp_path / "producer"; cons = tmp_path / "consumer"
    (prod / "data" / "nba").mkdir(parents=True); (cons / "data" / "nba").mkdir(parents=True)
    _db(prod / "data" / "nba" / "nba_history.sqlite", [(1, "a"), (2, "b")]); _db(cons / "data" / "nba" / "nba_history.sqlite", [(9, "stale")])
    calls = []
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] not in ("ssh", "rsync"):
            return real_run(cmd, **kw)                                                                            # feed.publish dumps with sqlite3
        calls.append(cmd)
        if cmd[0] == "ssh":
            assert cmd[-2:] == ["publish", "nba"] and "BatchMode=yes" in " ".join(cmd)
            feed.publish("nba", out_dir=prod / "data" / "feed", which="live", log=lambda s: None, root=prod)       # what feed_serve.sh does on the laptop
            return subprocess.CompletedProcess(cmd, 0, stdout="live 20261103-120000.000001 reference None\n", stderr="")
        assert cmd[0] == "rsync" and cmd[1:4] == ["-az", "--delete", "-e"] and cmd[-2] == "laptop:nba/"
        shutil.copytree(prod / "data" / "feed" / "nba", cmd[-1], dirs_exist_ok=True)                              # what rrsync would have served
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(feed.subprocess, "run", fake_run)
    changed = feed.pull_ssh("laptop", "nba", tmp_path / "state", log=lambda s: None, root=cons)
    assert [c[0] for c in calls] == ["ssh", "rsync"]
    assert "nba_player" in changed["live"] and changed["published"].startswith("live 20261103")
    con = sqlite3.connect(cons / "data" / "nba" / "nba_history.sqlite")
    assert con.execute("select name from nba_player order by id").fetchall() == [("a",), ("b",)]
    # nothing new on the producer: the pull says so without touching the tables
    changed2 = feed.pull_ssh("laptop", "nba", tmp_path / "state", log=lambda s: None, root=cons)
    assert {k: v for k, v in changed2.items() if k != "published"} == {}


def test_every_sport_has_a_spec_and_the_nfl_reference_is_every_table_but_the_working_set(tmp_path):
    """The same pull for every sport: the NBA has a live part, the NFL and MMA a reference part only; the NFL's "*" resolves to
    every table except its per-slate working tables and MLB's, the MMA's to everything."""
    assert feed.parts_of("nba") == ["live", "reference"] and feed.parts_of("nfl") == ["reference"] and feed.parts_of("mma") == ["reference"]
    assert feed.parts_of("nfl", "live") == [] and feed.parts_of("nba", "live") == ["live"]
    db = tmp_path / "nfl.sqlite"; con = sqlite3.connect(db)
    for t in ("nfl_stats", "nfl_player", "odds", "lineup", "mlb_stats", "historical_contest_information", "contest_entry_result"):
        con.execute(f"create table {t} (id integer primary key)")
    con.commit(); con.close()
    assert feed.resolve_tables(db, "nfl", "reference") == ["contest_entry_result", "historical_contest_information", "nfl_stats"]
    assert feed.resolve_tables(db, "mma", "reference") == sorted(["nfl_stats", "nfl_player", "odds", "lineup", "mlb_stats", "historical_contest_information", "contest_entry_result"])
    assert feed.resolve_tables(db, "nba", "live") == feed.TABLES["nba"]["live"]
    # publish for the NFL: the reference bundle and the models directory, minus its archive; a pull applies both
    prod = tmp_path / "producer"; cons = tmp_path / "consumer"
    (prod / "data" / "nfl" / "models" / "archive").mkdir(parents=True); (cons / "data" / "nfl").mkdir(parents=True)
    shutil.copy(db, prod / "data" / "nfl" / "nfl_history.sqlite")
    (prod / "data" / "nfl" / "models" / "dst.json").write_text("{}"); (prod / "data" / "nfl" / "models" / "archive" / "old.json").write_text("{}")
    m = feed.publish("nfl", out_dir=tmp_path / "feed", log=lambda s: None, root=prod)
    assert "live" not in m and m["reference"]["tables"] == ["contest_entry_result", "historical_contest_information", "nfl_stats"]
    assert m["files"]["count"] == 1                                                              # the archive stayed home
    cdb = cons / "data" / "nfl" / "nfl_history.sqlite"; c2 = sqlite3.connect(cdb); c2.execute("create table nfl_player (id integer primary key)"); c2.execute("insert into nfl_player values (5)"); c2.commit(); c2.close()
    changed = feed.pull(f"file://{tmp_path / 'feed'}", None, "nfl", tmp_path / "state", log=lambda s: None, root=cons)
    assert set(changed["reference"]) == {"contest_entry_result", "historical_contest_information", "nfl_stats"} and changed["files"] == 1
    c2 = sqlite3.connect(cdb)
    assert c2.execute("select id from nfl_player").fetchall() == [(5,)]                           # the consumer's working table survives
    assert (cons / "data" / "nfl" / "models" / "dst.json").exists() and not (cons / "data" / "nfl" / "models" / "archive").exists()


def test_producer_source_reads_the_environment(monkeypatch):
    monkeypatch.delenv(feed.SSH_ENV, raising=False); monkeypatch.delenv(feed.URL_ENV, raising=False)
    assert feed.producer_source() is None
    monkeypatch.setenv(feed.URL_ENV, "file:///x/feed")
    assert feed.producer_source() == ("url", "file:///x/feed")
    monkeypatch.setenv(feed.SSH_ENV, "sam@laptop")                       # ssh wins when both are set
    assert feed.producer_source() == ("ssh", "sam@laptop")
