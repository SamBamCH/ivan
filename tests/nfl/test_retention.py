"""What the retention sweep lets go of, and what it never touches."""
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from ivan.sports.nfl import retention


def touch(p: Path, when: datetime, size: int = 10) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    os.utime(p, (when.timestamp(), when.timestamp()))
    return p


def make_run(root: Path, when: datetime, status="ok", repeats=True, pinned=False) -> Path:
    touch(root / "run.json", when).write_text(json.dumps({"name": root.name, "pinned": pinned}))
    os.utime(root / "run.json", (when.timestamp(), when.timestamp()))
    touch(root / "replay.sqlite", when, 1000)
    touch(root / "replay.w0.sqlite", when, 5000)
    touch(root / "replay.w0.sqlite-wal", when, 50)
    s = root / "nfl" / "20261011-1920_showdown_154487"
    m = {"slate_key": "154487", "lock": "2026-10-11T19:20:00", "mode": "showdown", "status": status, "repeats": []}
    touch(s / "replay.json", when).write_text(json.dumps(m))
    os.utime(s / "replay.json", (when.timestamp(), when.timestamp()))
    touch(s / "projections.csv", when)
    touch(s / "entries.csv", when)
    if repeats:
        touch(s / "rep_000" / "chosen_entries.choose.x.json", when, 300)
        touch(s / "rep_000" / "sim" / "contest_1.npz", when, 300)
    else:
        touch(s / "chosen_entries.simroi.x.npz", when, 300)
        touch(s / "sim" / "contest_1.npz", when, 300)
    touch(root / "nfl" / "worker_0.log", when)
    touch(root / "nfl" / "report_20261011_20261011" / "summary.json", when).write_text("{}")
    touch(root / "nfl" / "report_20261011_20261011" / "sim_vs_actual.csv", when)
    touch(root / "nfl" / "validation" / "rep_000" / "summary.json", when).write_text("{}")
    touch(root / "nfl" / "validation" / "rep_000" / "pool.csv", when, 900)
    for p in root.rglob("*"):
        os.utime(p, (when.timestamp(), when.timestamp()))
    return root


def test_fresh_run_keeps_heavy_files_but_loses_day_old_clones(tmp_path):
    now = datetime(2026, 10, 28, 12)
    out = tmp_path / "output"
    r = make_run(out / "backtests" / "fresh", now - timedelta(days=3))
    s = retention.plan(now, output_dir=out, replay_dir=tmp_path / "replay")
    st = {x.name: x for x in s.runs}["fresh"]
    assert [p.path.name for p in st.clones] == ["replay.w0.sqlite", "replay.w0.sqlite-wal"]
    assert st.heavy == []
    retention.apply(s, now, log=lambda *_: None)
    assert not (r / "replay.w0.sqlite").exists() and (r / "replay.sqlite").exists()
    assert (r / "nfl" / "20261011-1920_showdown_154487" / "rep_000").is_dir()
    assert "purged_at" not in json.loads((r / "run.json").read_text())


def test_old_run_loses_heavy_files_and_keeps_its_record(tmp_path):
    now = datetime(2026, 10, 28, 12)
    out = tmp_path / "output"
    r = make_run(out / "backtests" / "old", now - timedelta(days=31))
    flat = make_run(out / "models" / "projections" / "20260901-1", now - timedelta(days=40), repeats=False)
    s = retention.plan(now, output_dir=out, replay_dir=tmp_path / "replay")
    assert {x.kind for x in s.runs} == {"backtest", "models/projections"}
    retention.apply(s, now, log=lambda *_: None)
    slate = r / "nfl" / "20261011-1920_showdown_154487"
    # gone: repeats, clones, the db copy, logs, validation csvs
    assert not (slate / "rep_000").exists()
    assert not (r / "replay.sqlite").exists() and not (r / "replay.w0.sqlite").exists()
    assert not (r / "nfl" / "worker_0.log").exists()
    assert not (r / "nfl" / "validation" / "rep_000" / "pool.csv").exists()
    # kept: the record a slate's history is read from
    assert (slate / "replay.json").is_file() and (slate / "projections.csv").is_file() and (slate / "entries.csv").is_file()
    assert (r / "nfl" / "report_20261011_20261011" / "sim_vs_actual.csv").is_file()
    assert (r / "nfl" / "validation" / "rep_000" / "summary.json").is_file()
    assert json.loads((r / "run.json").read_text())["purged_at"] == now.isoformat(timespec="seconds")
    fslate = flat / "nfl" / "20261011-1920_showdown_154487"
    assert not (fslate / "sim").exists() and not list(fslate.glob("chosen_entries.*")) and (fslate / "replay.json").is_file()


def test_pinned_and_active_runs_are_left_alone(tmp_path):
    now = datetime(2026, 10, 28, 12)
    out = tmp_path / "output"
    pinned = make_run(out / "backtests" / "pinned", now - timedelta(days=90), pinned=True)
    active = make_run(out / "backtests" / "active", now - timedelta(hours=2), status="running")
    s = retention.plan(now, output_dir=out, replay_dir=tmp_path / "replay")
    by = {x.name: x for x in s.runs}
    assert by["pinned"].heavy == [] and [p.path.name for p in by["pinned"].clones] == ["replay.w0.sqlite", "replay.w0.sqlite-wal"]
    assert by["active"].active and by["active"].heavy == [] and by["active"].clones == []
    # an unpin makes the sweep take the heavy files next time
    retention.set_pinned(pinned, False)
    s = retention.plan(now, output_dir=out, replay_dir=tmp_path / "replay")
    assert {x.name: x for x in s.runs}["pinned"].heavy


def test_stray_clones_of_the_default_replay_db(tmp_path):
    now = datetime(2026, 10, 28, 12)
    rd = tmp_path / "replay"
    touch(rd / "nfl_replay.sqlite", now - timedelta(days=5), 100)
    touch(rd / "nfl_replay.w0.sqlite", now - timedelta(days=5), 100)
    touch(rd / "nfl_replay.w1.sqlite", now - timedelta(minutes=5), 100)
    touch(rd / "nfl_replay.usage.sqlite", now - timedelta(days=5), 100)
    s = retention.plan(now, output_dir=tmp_path / "output", replay_dir=rd)
    assert [i.path.name for i in s.stray] == ["nfl_replay.w0.sqlite"]
    retention.apply(s, now, log=lambda *_: None)
    assert (rd / "nfl_replay.sqlite").exists() and (rd / "nfl_replay.w1.sqlite").exists() and (rd / "nfl_replay.usage.sqlite").exists()


def test_find_run_by_name_forms(tmp_path):
    now = datetime(2026, 10, 28, 12)
    out = tmp_path / "output"
    make_run(out / "backtests" / "abc", now)
    make_run(out / "models" / "field" / "20261015-1", now)
    assert retention.find_run("abc", out) == out / "backtests" / "abc"
    assert retention.find_run("field/20261015-1", out) == out / "models" / "field" / "20261015-1"
    assert retention.find_run(str(out / "backtests" / "abc"), out) == out / "backtests" / "abc"
    assert retention.find_run("nope", out) is None
