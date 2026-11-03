"""The data feed (ivan.core.feed): publish from one database, pull from a directory into another, only the feed's tables replaced."""
import shutil
import sqlite3
import subprocess

from ivan.core import feed


def _db(path, rows):
    """A history database: the shared live tables (odds carries `rows`) and a reference table."""
    con = sqlite3.connect(path)
    con.execute("create table odds (id integer primary key, player text)")
    con.executemany("insert into odds values (?, ?)", rows)
    con.execute("create index ix_odds_player on odds(player)")
    for t in ("game_odds", "nba_news", "nba_slate_news"):
        con.execute(f"create table {t} (id integer primary key, x text)")
    con.execute("create table nba_stats (id integer primary key, pts real)"); con.execute("insert into nba_stats values (1, 30.0)")
    con.commit(); con.close()


def _slate(root, dg, rows, history):
    """A held slate under root/data/nba/slates/<dg>/: its database (the pool rows), a pin and a store record."""
    from ivan.sports.nba import slates as S
    sdir = root / "data" / "nba" / "slates"
    sp = S.create(dg, lock=__import__("datetime").datetime(2026, 11, 4, 18, 0), games=4, root=sdir, history=history)
    con = sqlite3.connect(sp.db)
    con.executemany("insert into nba_player (id, name, team, position, roster_slot_id, gametime, salary, draft_group_id) values (?, ?, 'BOS', 'PG', 1, '2026-11-04 18:00:00', 5000, ?)", [(i, n, dg) for i, n in rows])
    con.commit(); con.close()
    sp.overrides_csv.write_text("player,minutes\na,36\n"); (sp.store / "BOS.pkl").write_bytes(b"bos-1")
    return sp


def test_publish_pull_from_directory_replaces_only_feed_tables(tmp_path):
    src = tmp_path / "producer.sqlite"; _db(src, [(1, "a"), (2, "b")])
    m = feed.publish("nba", out_dir=tmp_path / "feed", which="both", log=lambda s: None, db_path=src)
    assert m["live"]["bytes"] > 0 and m["reference"]["bytes"] > 0 and "odds" in m["live"]["tables"]
    dst = tmp_path / "consumer.sqlite"
    con = sqlite3.connect(dst)
    con.execute("create table my_own (id integer primary key)"); con.execute("insert into my_own values (7)")
    con.execute("create table odds (id integer primary key, player text)"); con.execute("insert into odds values (9, 'stale')"); con.commit(); con.close()
    changed = feed.pull(f"file://{tmp_path / 'feed'}", None, "nba", tmp_path / "state", db_path=dst, log=lambda s: None)
    assert "odds" in changed["live"] and "nba_stats" in changed["reference"]
    con = sqlite3.connect(dst)
    assert con.execute("select player from odds order by id").fetchall() == [("a",), ("b",)]
    assert con.execute("select id from my_own").fetchall() == [(7,)]                      # the consumer's own table survives
    assert con.execute("select count(*) from sqlite_master where name='ix_odds_player'").fetchone()[0] == 1
    assert feed.pull(f"file://{tmp_path / 'feed'}", None, "nba", tmp_path / "state", db_path=dst, log=lambda s: None) == {}   # nothing new


def test_a_bundle_per_held_slate_lands_in_the_consumers_slate_directory(tmp_path):
    """The producer publishes one bundle per held slate (its feed tables, its edits, its store, its record); the consumer makes the
    slate's directory, replaces the feed tables in its own slate database (its lineups survive), takes the edits as the published
    layer and mirrors the store, and notes pulled_at. An unchanged publish is nothing new; an edit alone travels alone."""
    from ivan.sports.nba import slates as S
    prod = tmp_path / "producer"; cons = tmp_path / "consumer"
    (prod / "data" / "nba").mkdir(parents=True); (cons / "data" / "nba").mkdir(parents=True)
    _db(prod / "data" / "nba" / "nba_history.sqlite", [(1, "a")]); _db(cons / "data" / "nba" / "nba_history.sqlite", [(9, "stale")])
    _slate(prod, 155, [(1, "a"), (2, "b")], prod / "data" / "nba" / "nba_history.sqlite")
    m = feed.publish("nba", out_dir=tmp_path / "feed", which="live", log=lambda s: None, root=prod)
    e = m["slates"]["155"]
    assert "nba_player" in e["tables"] and e["edits"]["count"] == 1 and e["mirror"]["count"] == 1 and e["meta"]["games"] == 4 and e["meta"]["published_at"]
    assert (tmp_path / "feed" / "nba" / "slates" / "155" / "slate.sql.gz").exists()
    changed = feed.pull(f"file://{tmp_path / 'feed'}", None, "nba", tmp_path / "state", which="live", log=lambda s: None, root=cons)
    assert changed["slates"][155]["tables"] and changed["slates"][155]["edits"] == 1 and changed["slates"][155]["mirror"] == 1
    csd = cons / "data" / "nba" / "slates"
    sp = S.paths(155, csd)
    con = sqlite3.connect(sp.db)
    assert con.execute("select name from nba_player order by id").fetchall() == [("a",), ("b",)]
    con.execute("insert into lineup (id, draft_group_id, player_ids_list) values (1, 155, '[1, 2]')"); con.commit(); con.close()   # the consumer's own work
    assert sp.published_overrides_csv.read_text().startswith("player") and (sp.store / "BOS.pkl").read_bytes() == b"bos-1"
    meta = S.read_meta(155, csd)
    assert meta["games"] == 4 and meta["lock"] == "2026-11-04T18:00:00" and meta["pulled_at"] and not sp.overrides_csv.exists()
    # nothing changed on the producer: nothing new
    feed.publish("nba", out_dir=tmp_path / "feed", which="live", log=lambda s: None, root=prod)
    assert feed.pull(f"file://{tmp_path / 'feed'}", None, "nba", tmp_path / "state", which="live", log=lambda s: None, root=cons) == {}
    # a pin lifted on the producer: only the edits travel, the consumer's lineup is untouched, the pin is gone from the published layer
    S.paths(155, prod / "data" / "nba" / "slates").overrides_csv.unlink()
    feed.publish("nba", out_dir=tmp_path / "feed", which="live", log=lambda s: None, root=prod)
    changed = feed.pull(f"file://{tmp_path / 'feed'}", None, "nba", tmp_path / "state", which="live", log=lambda s: None, root=cons)
    assert changed["slates"][155] == {"edits": 0}
    con = sqlite3.connect(sp.db)
    assert con.execute("select count(*) from lineup").fetchone()[0] == 1 and not sp.published_overrides_csv.exists()
    con.close()


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
    assert "odds" in changed["live"] and changed["published"].startswith("live 20261103")
    con = sqlite3.connect(cons / "data" / "nba" / "nba_history.sqlite")
    assert con.execute("select player from odds order by id").fetchall() == [("a",), ("b",)]
    # nothing new on the producer: the pull says so without touching the tables
    changed2 = feed.pull_ssh("laptop", "nba", tmp_path / "state", log=lambda s: None, root=cons)
    assert {k: v for k, v in changed2.items() if k != "published"} == {}


def test_pull_ssh_asks_for_a_slate_and_waits_longer_for_it(tmp_path, monkeypatch):
    """A consumer that picks a slate the producer does not hold asks 'publish nba <dg>' (the forced command loads it first);
    a URL source cannot ask."""
    cons = tmp_path / "consumer"; (cons / "data" / "nba").mkdir(parents=True)
    _db(cons / "data" / "nba" / "nba_history.sqlite", [(9, "stale")])
    seen = []

    def fake_run(cmd, **kw):
        if cmd[0] == "ssh":
            seen.append((cmd[-3:], kw.get("timeout")))
            raise subprocess.CalledProcessError(1, cmd, stderr="feed_serve: refused")
        raise AssertionError(cmd)
    monkeypatch.setattr(feed.subprocess, "run", fake_run)
    try:
        feed.pull_ssh("laptop", "nba", tmp_path / "state", log=lambda s: None, root=cons, slate=12345)
    except RuntimeError as e:
        assert "did not publish" in str(e)
    assert seen == [(["publish", "nba", "12345"], 600)]
    monkeypatch.setenv(feed.URL_ENV, "file:///x"); monkeypatch.delenv(feed.SSH_ENV, raising=False)
    try:
        feed.pull_from_producer("nba", tmp_path / "state", root=cons, slate=12345)
        raise AssertionError("a URL source must refuse a slate ask")
    except RuntimeError as e:
        assert "only an ssh producer" in str(e)


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
