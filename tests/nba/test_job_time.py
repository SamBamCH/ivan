"""A job's time before it starts (ivan.sports.nba.job_time): the work units per step, the fit from past job records, the estimate and
the lock guard."""
import json
from datetime import datetime, timedelta

from ivan.sports.nba import job_time as JT


def _job(d, name, secs, cmd, meta=None, i=0):
    j = dict(id=f"j{i}", name=name, cmd=cmd, status="done", created=1e9 + i, started=1e9 + i, finished=1e9 + i + secs, meta=meta or {"step": name})
    (d / f"{name}-{i}.json").write_text(json.dumps(j))


def test_units_and_knobs_from_the_command_line():
    assert JT.units("generate", {"lineups": "20000", "jobs": "8"}) == 2500.0
    assert JT.units("sim", {"sims": 1000, "field_cap": 10000}, n_contests=6) == 60.0
    assert JT.units("sim", {"sims": 1000, "field_cap": 10000}) == 10.0                   # contests unknown: one
    assert JT.units("lateswap", {"ls_candidates": 10, "ls_sims": 1000, "ls_field_cap": 5000}, 2) == 100.0
    assert JT.units("choose", {}) == 1.0
    assert JT.knobs_from_cmd("generate", ["python", "generate_lineups.py", "500", "8", "-c", "cfg.json", "--no-progress"]) == {"lineups": "500", "jobs": "8"}
    assert JT.knobs_from_cmd("sim", ["python", "sim.py", "--config", "c", "-n", "1000", "-c", "10000"]) == {"sims": "1000", "field_cap": "10000"}


def test_fit_estimate_and_the_guard(tmp_path):
    d = tmp_path / "jobs"; d.mkdir()
    # generate: 2 s per lineup-per-worker plus 20 s; sim: 4 s per unit (contests x sims x cap / 1e6) plus 30 s
    for i, (lu, jobs) in enumerate([(500, 8), (1000, 8), (2000, 4), (4000, 8), (8000, 8)]):
        _job(d, "generate", 20 + 2 * lu / jobs, ["py", "generate_lineups.py", str(lu), str(jobs), "-c", "cfg"], i=i)
    for i, (n, cap, c) in enumerate([(1000, 10000, 3), (500, 5000, 3), (2000, 10000, 2), (1000, 20000, 4)]):
        _job(d, "sim", 30 + 4 * c * n * cap / 1e6, ["py", "sim.py", "-n", str(n), "-c", str(cap)], meta={"step": "sim", "n_contests": c}, i=10 + i)
    _job(d, "choose", 18, ["py", "choose.py"], i=20)                                    # too few runs: the default stands in
    model = JT.fit(d)
    assert model["generate"]["basis"] == "fitted" and model["generate"]["n"] == 5 and abs(model["generate"]["per_unit"] - 2.0) < 0.3
    assert model["sim"]["basis"] == "fitted" and abs(model["sim"]["per_unit"] - 4.0) < 0.6
    assert model["choose"]["basis"] == "default" and model["choose"]["n"] == 1
    now = datetime(2026, 11, 4, 17, 0)
    est = JT.estimate("sim", {"sims": 1000, "field_cap": 10000}, n_contests=6, model=model, now=now)
    assert 200 < est["seconds"] < 320 and est["ends_at"] > now and est["upto"] >= est["seconds"]
    ok = JT.deadline_check(est, now + timedelta(hours=1))
    assert ok[0] == "ok" and "lock 18:00" in ok[1]
    late = JT.deadline_check(est, now + timedelta(minutes=6))
    assert late[0] == "late" and "inside" in late[1]
    past = JT.deadline_check(est, now - timedelta(minutes=1))
    assert past[0] == "late" and "after lock" in past[1]
    tight = JT.deadline_check(JT.estimate("sim", {"sims": 1000, "field_cap": 10000}, 6, dict(model, sim=dict(model["sim"], p90_ratio=3.0)), now),
                              now + timedelta(minutes=12))
    assert tight[0] == "tight"
    assert JT.deadline_check(est, None)[0] == "ok"
    whole = JT.estimate("all", {"lineups": 2000, "jobs": 8, "sims": 1000, "field_cap": 10000}, 6, model, now)
    assert whole["seconds"] > est["seconds"] and len(whole["parts"]) == 3
    assert JT.fmt(45) == "45 s" and JT.fmt(150) == "2 min 30 s" and JT.fmt(3700) == "1 h 01 min"
