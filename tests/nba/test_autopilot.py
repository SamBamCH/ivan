"""The autopilot switch and guard (ivan.sports.nba.autopilot): the file switch, the desk's footprint, the running-job check,
what travels in the feed manifest."""
import json
import time
from datetime import datetime, timedelta

from ivan.core import feed
from ivan.sports.nba import autopilot as ap


def test_switch_defaults_off_and_round_trips(tmp_path):
    p = tmp_path / "autopilot.json"
    assert ap.is_on(p) is False and ap.state(p)["on"] is False
    st = ap.set_on(True, by="cli", path=p)
    assert st["on"] is True and ap.is_on(p) is True and ap.state(p)["by"] == "cli"
    ap.set_on(False, path=p)
    assert ap.is_on(p) is False
    ap.record_tick("did a thing", ok=True, detail=dict(pushed=True), path=p)
    lt = ap.state(p)["last_tick"]
    assert lt["result"] == "did a thing" and lt["ok"] is True and lt["pushed"] is True and lt["at"]


def test_guard_holds_after_an_unpublished_edit_and_releases_after_publish(tmp_path):
    act, jobs = tmp_path / "desk_activity.json", tmp_path / "jobs"
    now = datetime(2026, 10, 27, 12, 0, 0)
    assert ap.desk_hold(now, act, jobs) is None                                       # no footprint: free to run
    act.write_text(json.dumps({"edit_at": (now - timedelta(minutes=5)).isoformat()}))
    assert "edited 5 min ago" in ap.desk_hold(now, act, jobs)                         # a fresh edit, not published
    assert ap.desk_hold(now + timedelta(minutes=40), act, jobs) is None               # the hold expires
    act.write_text(json.dumps({"edit_at": (now - timedelta(minutes=5)).isoformat(), "publish_at": (now - timedelta(minutes=1)).isoformat()}))
    assert ap.desk_hold(now, act, jobs) is None                                       # published since: free
    act.write_text(json.dumps({"edit_at": now.isoformat(), "publish_at": (now - timedelta(minutes=1)).isoformat()}))
    assert ap.desk_hold(now, act, jobs) is not None                                   # edited again after the publish


def test_touch_writes_the_footprint(tmp_path):
    act = tmp_path / "a.json"
    ap.touch("edit", act); ap.touch("publish", act)
    d = json.loads(act.read_text())
    assert d["edit_at"] and d["publish_at"]


def test_guard_holds_while_a_desk_job_runs_but_not_for_a_dead_one(tmp_path):
    jobs = tmp_path / "jobs"; jobs.mkdir()
    now = time.time()
    (jobs / "j1.json").write_text(json.dumps({"id": "j1", "status": "running", "started": now - 60}))
    assert ap.running_job(jobs, now=now) == "j1"
    assert "job is running" in ap.desk_hold(datetime.fromtimestamp(now), tmp_path / "none.json", jobs)
    (jobs / "j1.json").write_text(json.dumps({"id": "j1", "status": "running", "started": now - 5 * 3600}))
    assert ap.running_job(jobs, now=now) is None                                      # a desk that died mid-job
    (jobs / "j1.json").write_text(json.dumps({"id": "j1", "status": "done", "started": now - 60}))
    assert ap.running_job(jobs, now=now) is None


def test_producer_note_travels_in_the_manifest_and_the_pull_state(tmp_path, monkeypatch):
    import sqlite3
    src = tmp_path / "p.sqlite"
    con = sqlite3.connect(src); con.execute("create table nba_player (id integer primary key, name text)"); con.commit(); con.close()
    monkeypatch.setattr(ap, "AUTOPILOT_JSON", tmp_path / "autopilot.json")
    ap.set_on(True, by="test", path=tmp_path / "autopilot.json")
    note = ap.producer_info("autopilot")
    assert note["source"] == "autopilot" and note["autopilot"] is True and note["host"]
    m = feed.publish("nba", out_dir=tmp_path / "feed", which="live", log=lambda s: None, db_path=src, producer=note)
    assert m["producer"]["source"] == "autopilot"
    dst = tmp_path / "c.sqlite"
    feed.pull(f"file://{tmp_path / 'feed'}", None, "nba", tmp_path / "state", db_path=dst, log=lambda s: None)
    st = feed.status("nba", state_dir=tmp_path / "state")
    assert st["producer"]["autopilot"] is True and st["published_at"] and st["pulled_at"] and st["checked_at"] and st["new"] is False
    # a second pull with nothing new still records the check
    before = st["checked_at"]; time.sleep(1.1)
    feed.pull(f"file://{tmp_path / 'feed'}", None, "nba", tmp_path / "state", db_path=dst, log=lambda s: None)
    assert feed.status("nba", state_dir=tmp_path / "state")["checked_at"] > before
