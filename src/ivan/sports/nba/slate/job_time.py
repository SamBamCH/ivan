"""
How long a desk job will take, before it starts, and whether it ends before lock.

Every job the desk runs leaves a record under data/nba/jobs/<id>.json (ivan.web.jobs): its start, finish, command line and
meta. The work of each step scales with a few knobs, so one number per step, seconds per unit of work, fitted from this
machine's completed jobs, gives the estimate (`estimate`); with fewer than MIN_RUNS runs a conservative default stands in.
The guard (`deadline_check`) compares the estimate's end with the slate's lock (the late swap: the next tip) less a margin.

    units(step, knobs, n_contests)       the step's work in its unit (generate: lineups per worker; sim and late swap: contests
                                         x sims x field cap / 1e6; the rest: one)
    fit(jobs_dir)                        {step: {"per_unit", "fixed", "n", "p90_ratio"}} from the completed jobs
    estimate(step, knobs, ...)           {"seconds", "ends_at", "basis", "n"}
    deadline_check(est, deadline, ...)   ("ok" | "tight" | "late", message)

The command line: scripts/nba/job_time.py fit | estimate --step sim --sims 1000 --field-cap 10000 --contests 6
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ivan.sports.nba.paths import NBA_DATA_DIR

JOBS_DIR = NBA_DATA_DIR / "jobs"
MIN_RUNS = 4
MARGIN_MIN = 5.0                 # the estimate must end this many minutes before the deadline to be "ok"
# conservative defaults (seconds) when a step has too few runs: seconds per unit and a fixed part
DEFAULTS = {"generate": (3.0, 20.0), "sim": (4.0, 30.0), "lateswap": (6.0, 60.0), "choose": (0.0, 40.0), "field": (0.0, 30.0),
            "fetch_standings": (0.0, 40.0), "solve": (0.0, 15.0), "all": (0.0, 60.0)}
STEPS = tuple(DEFAULTS)


def _f(v, default=0.0) -> float:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return default


def units(step: str, knobs: dict, n_contests: Optional[int] = None) -> float:
    """The step's work in its unit. generate: lineups / workers. sim: contests x sims x field cap / 1e6 (one contest when the
    entries are unknown). lateswap: contests x candidates x sims x field cap / 1e6. Everything else: 1."""
    k = {str(a): b for a, b in (knobs or {}).items()}
    c = max(1, int(n_contests or 1))
    if step == "generate":
        return max(1.0, _f(k.get("lineups"), 1000.0)) / max(1.0, _f(k.get("jobs"), 1.0))
    if step == "sim":
        return c * max(1.0, _f(k.get("sims"), 1000.0)) * max(1.0, _f(k.get("field_cap"), 10000.0)) / 1e6
    if step == "lateswap":
        return c * max(1.0, _f(k.get("ls_candidates", k.get("candidates")), 10.0)) * max(1.0, _f(k.get("ls_sims", k.get("sims")), 1000.0)) \
            * max(1.0, _f(k.get("ls_field_cap", k.get("field_cap")), 5000.0)) / 1e6
    return 1.0


def knobs_from_cmd(step: str, cmd: list) -> dict:
    """The knobs a past job ran with, from its command line (jobs older than the meta's `units` field)."""
    cmd = [str(c) for c in cmd]
    out: dict = {}

    def opt(name, default=None):
        return cmd[cmd.index(name) + 1] if name in cmd and cmd.index(name) + 1 < len(cmd) else default
    if step == "generate":
        pos = [c for c in cmd[2:] if c.isdigit()]
        if len(pos) >= 2:
            out["lineups"], out["jobs"] = pos[0], pos[1]
    elif step == "sim":
        out["sims"] = opt("-n", 1000); out["field_cap"] = opt("-c", 10000)
    elif step == "lateswap":
        out["candidates"] = opt("--candidates", 10); out["sims"] = opt("--sims", 1000); out["field_cap"] = opt("--field-cap", 5000)
    return out


def records(jobs_dir: Path = JOBS_DIR) -> list[dict]:
    """Every completed job on this machine: {step, seconds, units}."""
    out = []
    for f in sorted(Path(jobs_dir).glob("*.json")):
        try:
            j = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if j.get("status") != "done" or not j.get("started") or not j.get("finished"):
            continue
        meta = j.get("meta") or {}
        step = meta.get("step") or j.get("name")
        if step not in DEFAULTS:
            continue
        secs = float(j["finished"]) - float(j["started"])
        if secs <= 0:
            continue
        u = meta.get("units")
        if u is None:
            u = units(step, knobs_from_cmd(step, j.get("cmd") or []), meta.get("n_contests"))
        out.append(dict(step=step, seconds=secs, units=float(u), at=float(j["finished"])))
    return out


def fit(jobs_dir: Path = JOBS_DIR, recent: int = 40) -> dict:
    """Per step: seconds per unit as the median ratio over the last `recent` completed runs (robust to one slow night), a fixed
    part (the smallest run, bounded), and the p90 of actual over fitted, which the estimate shows as its "up to"."""
    rows = records(jobs_dir)
    model = {}
    for step in DEFAULTS:
        mine = sorted([r for r in rows if r["step"] == step], key=lambda r: r["at"])[-recent:]
        if len(mine) < MIN_RUNS:
            per, fixed = DEFAULTS[step]
            model[step] = dict(per_unit=per, fixed=fixed, n=len(mine), p90_ratio=1.5, basis="default")
            continue
        fixed = min(r["seconds"] for r in mine) * 0.5
        ratios = [max(0.0, r["seconds"] - fixed) / max(r["units"], 1e-9) for r in mine]
        per = statistics.median(ratios)
        fitted = [fixed + per * r["units"] for r in mine]
        over = sorted(r["seconds"] / f for r, f in zip(mine, fitted) if f > 0)
        p90 = over[min(len(over) - 1, int(0.9 * (len(over) - 1)))] if over else 1.5
        model[step] = dict(per_unit=per, fixed=fixed, n=len(mine), p90_ratio=max(1.0, p90), basis="fitted")
    return model


def estimate(step: str, knobs: dict, n_contests: Optional[int] = None, model: Optional[dict] = None, now: Optional[datetime] = None,
             jobs_dir: Path = JOBS_DIR) -> dict:
    """{"seconds", "upto", "ends_at", "upto_at", "units", "basis", "n"} for a run of `step` with these knobs starting now. "all" is
    generate + sim + choose."""
    now = now or datetime.now()
    model = model or fit(jobs_dir)
    if step == "all":
        parts = [estimate(s, knobs, n_contests, model, now, jobs_dir) for s in ("generate", "sim", "choose")]
        secs = sum(p["seconds"] for p in parts); upto = sum(p["upto"] for p in parts)
        return dict(seconds=secs, upto=upto, ends_at=now + timedelta(seconds=secs), upto_at=now + timedelta(seconds=upto), units=None,
                    basis="+".join(p["basis"] for p in parts), n=min(p["n"] for p in parts), parts=parts)
    m = model.get(step) or dict(per_unit=DEFAULTS.get(step, (0.0, 30.0))[0], fixed=DEFAULTS.get(step, (0.0, 30.0))[1], n=0, p90_ratio=1.5, basis="default")
    u = units(step, knobs, n_contests)
    secs = m["fixed"] + m["per_unit"] * u
    upto = secs * m["p90_ratio"]
    return dict(seconds=secs, upto=upto, ends_at=now + timedelta(seconds=secs), upto_at=now + timedelta(seconds=upto), units=u,
                basis=m["basis"], n=m["n"])


def deadline_check(est: dict, deadline: Optional[datetime], margin_min: float = MARGIN_MIN, label: str = "lock") -> tuple[str, str]:
    """("ok" | "tight" | "late", a sentence). ok: even the "up to" end leaves the margin before the deadline. tight: the expected
    end does, the "up to" does not. late: the expected end is inside the margin or past the deadline. No deadline: ok."""
    if deadline is None:
        return "ok", f"about {fmt(est['seconds'])} (up to {fmt(est['upto'])}); no {label} to beat"
    left = (deadline - est["ends_at"]).total_seconds() / 60
    left_upto = (deadline - est["upto_at"]).total_seconds() / 60
    day = "%a %H:%M" if deadline.date() != est["ends_at"].date() else "%H:%M"          # a lock on another day says which
    when = f"done {est['ends_at']:%H:%M}, up to {est['upto_at']:%H:%M}; {label} {deadline:{day}}"
    if left < margin_min:
        return "late", f"about {fmt(est['seconds'])}: {when}. That ends {'after' if left < 0 else 'inside ' + fmt(margin_min * 60) + ' of'} {label}."
    if left_upto < margin_min:
        return "tight", f"about {fmt(est['seconds'])}: {when}. A slow run would cut into the last {fmt(margin_min * 60)} before {label}."
    return "ok", f"about {fmt(est['seconds'])}: {when}"


def fmt(seconds: float) -> str:
    s = int(round(seconds))
    if s < 90:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min" + (f" {s % 60:02d} s" if s % 60 and s < 600 else "")
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"
