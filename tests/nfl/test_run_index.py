"""The per-slate run index: every run that replayed a slate, from the files retention keeps."""
import csv
import json
from pathlib import Path

from ivan.sports.nfl import run_index


def write_run(root: Path, slate_dirs: dict, report_rows=None, repeat_rows=None, run_json=None):
    if run_json is not None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "run.json").write_text(json.dumps(run_json))
    for name, m in slate_dirs.items():
        d = root / "nfl" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "replay.json").write_text(json.dumps(m))
    if report_rows is not None:
        rep = root / "nfl" / "report_20261011_20261019"
        rep.mkdir(parents=True, exist_ok=True)
        (rep / "summary.json").write_text("{}")
        with open(rep / "sim_vs_actual.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["slate", "contest_id", "fees", "sim_expected_profit", "actual_profit"])
            w.writeheader(); w.writerows(report_rows)
        if repeat_rows is not None:
            with open(rep / "repeats_by_contest.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["repeat", "contest_id", "actual_profit"])
                w.writeheader(); w.writerows(repeat_rows)


def manifest(key, lock, status="ok", contests=(), repeats=None, through="choose", git=None, method="portfolio"):
    reps = [{"k": k, "status": "ok", "stages": {"choose": {"weights": {"method": method}}}} for k in range(repeats)] if repeats else []
    return {"slate_key": key, "lock": lock, "mode": "showdown", "status": status, "started_at": f"{lock[:10]}T20:00:00",
            "finished_at": f"{lock[:10]}T21:00:00", "through": through, "git": git or {},
            "options": {"n_lineups": 3000, "n_sims": 500, "field_cap": 5000, "seed": 0, "repeats": repeats or 1},
            "models": {"cutoff": "2026-10-11"}, "contests": [{"contest_id": c, "entries": 20} for c in contests], "repeats": reps,
            "stages": {} if repeats else {"choose": {"weights": {"method": method}}}}


def test_slate_runs_span_hub_models_and_lab_roots(tmp_path):
    out = tmp_path / "output"
    write_run(out / "backtests" / "r1", {"20261011-1920_showdown_154487": manifest("154487", "2026-10-11T19:20:00", contests=(1, 2), repeats=2,
                                                                              git={"commit": "abc1234", "dirty": False, "subject": "fix"}),
                                       "20261012-1915_showdown_154488": manifest("154488", "2026-10-12T19:15:00", contests=(3,))},
              report_rows=[{"slate": "2026-10-11 19:20", "contest_id": 1, "fees": 60, "sim_expected_profit": 10, "actual_profit": -5},
                           {"slate": "2026-10-11 19:20", "contest_id": 2, "fees": 20, "sim_expected_profit": 5, "actual_profit": 30},
                           {"slate": "2026-10-12 19:15", "contest_id": 3, "fees": 10, "sim_expected_profit": 1, "actual_profit": 1}],
              repeat_rows=[{"repeat": 0, "contest_id": 1, "actual_profit": -5}, {"repeat": 0, "contest_id": 2, "actual_profit": 30},
                           {"repeat": 1, "contest_id": 1, "actual_profit": -20}, {"repeat": 1, "contest_id": 2, "actual_profit": 10},
                           {"repeat": 0, "contest_id": 3, "actual_profit": 1}],
              run_json={"name": "r1", "params": {"method": "portfolio"}, "purged_at": "2026-11-20T06:30:00"})
    write_run(out / "models" / "projections" / "20261015-1", {"20261011-1920_showdown_154487": {**manifest("154487", "2026-10-11T19:20:00", through="project", contests=(1,)), "started_at": "2026-10-15T15:42:00"}},
              run_json={"name": "20261015-1", "git": {"commit": "def5678", "dirty": True, "subject": "try"}, "label": "shade 0.5"})
    write_run(out / "replay_lab", {"20261011-1920_showdown_154487": manifest("154487", "2026-10-11T19:20:00", status="error", contests=(1,))})
    (out / "replay_lab" / "nfl" / "20261011-1920_showdown_154487" / "replay.json").write_text(
        json.dumps({**manifest("154487", "2026-10-11T19:20:00", status="error", contests=(1,)), "started_at": "2026-10-09T10:00:00", "error": "boom"}))

    rows = run_index.slate_runs("154487", out)
    assert [r.kind for r in rows] == ["models/projections", "backtest", "replay"]       # newest first
    hub = rows[1]
    assert hub.commit == "abc1234" and hub.through == "choose" and hub.method == "portfolio" and hub.models == "2026-10-11"
    assert hub.repeats_ok == 2 and hub.repeats_wanted == 2 and hub.contests == 2 and hub.entries == 40
    assert hub.fees == 80 and hub.sim == 15 and hub.actual == 25                        # this slate's contests only
    assert hub.repeat_profits == [25.0, -10.0] and hub.repeat_mean == 7.5 and hub.p_profit == 0.5
    assert hub.purged_at == "2026-11-20T06:30:00" and hub.report_id == "backtests/r1/nfl/report_20261011_20261019"
    mod = rows[0]
    assert mod.commit == "def5678" and mod.dirty and mod.label == "shade 0.5" and mod.sim is None and mod.through == "project"
    lab = rows[2]
    assert lab.status == "error" and lab.error == "boom" and lab.commit == ""

    assert [r.kind for r in run_index.slate_runs("154488", out)] == ["backtest"]
    assert run_index.slate_runs("nope", out) == []
    idx = {s["slate_key"]: s for s in run_index.slates(out)}
    assert idx["154487"]["runs"] == 3 and idx["154487"]["ok"] == 2 and idx["154488"]["runs"] == 1
    assert [s["slate_key"] for s in run_index.slates(out)] == ["154488", "154487"]      # newest lock first
