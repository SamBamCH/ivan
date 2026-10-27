# src/ivan/web/backtest_hub.py
"""
The Backtest page, organised around a *run*:

    data/nfl/output/backtests/<name>/
        run.json            name, window, status, picks, params, the jobs it started
        selection.json      the contest selection scripts/replay.py reads (--menu)
        nfl/<slate dirs>    the replay's outputs, its report_*/ and validation/rep_000/

A run goes through steps: data (what the window lacks, fetched only after
the user confirms) -> contests (tick what to enter, how many entries) ->
params (repeats, workers, ...) -> running -> results (run -> slate ->
contest, with the distribution across repeats). Only runs made here are
listed; the replay roots the CLI writes elsewhere are not.

The readiness module does the looking (ivan.sports.nfl.readiness); this
module keeps the run state and builds the job commands.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ivan.core import model_store
from ivan.core.replay import Menu
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nfl.paths import MODELS_DIR, OUTPUT_DIR
from ivan.web import backtest_view as bv
from ivan.web import validate_view as vv

MENU_PATH = PROJECT_ROOT / "config" / "replay" / "nfl.json"
SCRIPTS = PROJECT_ROOT / "scripts" / "nfl"
BACKTESTS_DIR = OUTPUT_DIR / "backtests"
STEPS = ("data", "contests", "params", "results")
STATUS_STEP = {"new": "data", "data": "data", "contests": "contests", "params": "params", "running": "results", "done": "results"}


# --------------------------------------------------------------------------- run state
def run_root(name: str) -> Path:
    return BACKTESTS_DIR / name


def clean_name(name: str, since: str, until: str) -> str:
    base = name.strip().replace(" ", "_") if name.strip() else f"{since}_{until}"
    return "".join(ch for ch in base if ch.isalnum() or ch in "-_.") or f"{since}_{until}"


def load_run(name: str) -> Optional[dict]:
    p = run_root(name) / "run.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def save_run(run: dict) -> dict:
    root = run_root(run["name"])
    root.mkdir(parents=True, exist_ok=True)
    run["updated_at"] = datetime.now().isoformat(timespec="seconds")
    (root / "run.json").write_text(json.dumps(run, indent=1, default=str))
    return run


def new_run(name: str, since: str, until: str) -> dict:
    datetime.fromisoformat(since); datetime.fromisoformat(until)
    name = clean_name(name, since, until)
    if (run_root(name) / "run.json").exists():
        raise ValueError(f"a run named {name!r} exists; open it from the menu or pick another name")
    return save_run({"name": name, "since": since, "until": until, "status": "new", "created_at": datetime.now().isoformat(timespec="seconds"),
                     "picks": {}, "params": {}, "jobs": []})


def runs() -> list[dict]:
    """Every run made on this page, newest first."""
    out = []
    if BACKTESTS_DIR.is_dir():
        for d in sorted(BACKTESTS_DIR.iterdir()):
            r = load_run(d.name)
            if r:
                conf = None
                vs = d / "nfl" / "validation" / "rep_000" / "summary.json"
                if vs.is_file():
                    try:
                        conf = (json.loads(vs.read_text()).get("confidence") or {})
                    except (OSError, ValueError):
                        conf = None
                out.append({**r, "step": STATUS_STEP.get(r.get("status", "new"), "data"), "progress": bv.run_progress(d), "confidence": conf})
    return sorted(out, key=lambda r: r.get("created_at", ""), reverse=True)


def step_for(run: dict, requested: Optional[str]) -> str:
    if requested in STEPS:
        return requested
    return STATUS_STEP.get(run.get("status", "new"), "data")


# --------------------------------------------------------------------------- the window's data
def window(since: str, until: str) -> tuple[datetime, datetime]:
    a = datetime.fromisoformat(since)
    b = datetime.fromisoformat(until) + timedelta(days=1) - timedelta(seconds=1)
    return a, b


def plan(session, since: str, until: str) -> dict:
    from ivan.sports.nfl.readiness import backtest_plan
    a, b = window(since, until)
    menu = Menu.load(MENU_PATH) if MENU_PATH.exists() else Menu(items=[])
    return backtest_plan(session, a, b, menu)


def data_check(session, run: dict) -> dict:
    """
    What the window lacks, grouped by the fetch that fills it, each with its
    cost and caveat, plus what is already there. Nothing is fetched here.
    """
    p = plan(session, run["since"], run["until"])
    slates = p["slates"]
    weeks = sorted({f"{s.season_week[0]} wk {s.season_week[1]}" for s in slates if s.season_week and not s.box_scores})
    no_capture = [s for s in slates if s.captures == 0]
    captured = [s for s in slates if s.captures > 0]
    unscraped = sorted({c.contest_id for s in captured for c in s.picks if not c.scraped})
    no_ours = sum(1 for s in captured if s.picks and s.our_entries == 0)
    models = model_store.resolve(MODELS_DIR, datetime.fromisoformat(run["since"])).describe()
    items = []
    if weeks:
        items.append({"key": "fetch_actuals", "label": f"Box scores for {len(weeks)} week(s): {', '.join(weeks)}",
                      "detail": "nflverse schedule, player box scores and the DST table (free; a day or two after the games).", "n": len(weeks)})
    if no_capture:
        items.append({"key": "backfill", "label": f"Pre-lock captures for {len(no_capture)} slate(s) that only have scraped contests",
                      "detail": "Rebuilt from the projection snapshot (players) and stored game lines; the player props come from the-odds-api history "
                                "at about 230 credits per game. Without props a slate cannot be projected, so this buys them.", "n": len(no_capture),
                      "slates": [f"{s.lock:%Y-%m-%d %H:%M} {s.mode} {s.slate_key}" for s in no_capture]})
    if unscraped:
        items.append({"key": "scrape", "label": f"{len(unscraped)} contest(s) of the default pool not scraped yet",
                      "detail": "Standings and lineups from DraftKings through the saved Chrome profile (needs a login there; a 403 in the job log means log in once "
                                "and run again). Your username records your own entries' finishes.", "n": len(unscraped), "ids": unscraped})
    if models.get("fallback"):
        items.append({"key": "build_models", "label": f"Model version as of {run['since']}",
                      "detail": "DST parameters, classic field models, the dupe model and the sd calibration trained only on data before the window; "
                                "without it the re-run uses the live models, which have seen these slates.", "n": 1})
    return {"slates": slates, "summary": p["summary"], "needs": items, "models": models,
            "have": {"slates": len(slates), "captured": len(captured), "weeks_ok": sorted({f"{s.season_week[0]} wk {s.season_week[1]}" for s in slates if s.season_week and s.box_scores}),
                     "scraped": sum(1 for s in captured for c in s.picks if c.scraped), "pool": sum(len(s.picks) for s in captured), "no_ours": no_ours},
            "ready": not items}


TOP_CONTESTS = 10            # contests shown per slate: the largest by entries, plus any default pick outside them


def contests_view(session, run: dict, top: int = TOP_CONTESTS) -> list[dict]:
    """The replayable slates of the window with their largest contests (and the default picks), for the contests step."""
    p = plan(session, run["since"], run["until"])
    picks = {int(k): int(v) for k, v in (run.get("picks") or {}).items()}
    out = []
    for s in p["slates"]:
        if not s.replayable:
            continue
        default = {c.contest_id: c.entries for c in s.picks}
        rows = []
        ranked = sorted(s.contests, key=lambda c: -c.total_entries)
        keep = {c.contest_id for c in ranked[:top]} | set(default) | set(picks)
        for c in [c for c in ranked if c.contest_id in keep]:
            cap = max(1, int(c.max_entries or 1))                     # the contest's own entry limit is the default
            rows.append({"contest_id": c.contest_id, "name": c.name, "fee": c.entry_fee, "size": c.total_entries, "max_entries": cap,
                         "entries": min(cap, max(1, int(picks.get(c.contest_id, cap)))),
                         "picked": (c.contest_id in picks) if picks else (c.contest_id in default),
                         "payouts": c.has_payouts, "scraped": c.scraped, "points": c.players_with_points, "own": c.players_with_ownership, "ours": c.our_entries})
        out.append({"slate_key": s.slate_key, "mode": s.mode, "lock": s.lock.strftime("%Y-%m-%d %H:%M"), "status": s.status, "contests": rows,
                    "n_total": len(s.contests)})
    return out


# --------------------------------------------------------------------------- job commands
def write_selection(run: dict) -> Path:
    base = json.loads(MENU_PATH.read_text()) if MENU_PATH.exists() else {}
    weights = dict(base.get("weights") or {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1})
    p = run.get("params") or {}
    if p.get("method"):                                         # the chooser and its weights picked on the parameters step
        weights["method"] = str(p["method"])
        weights["bankroll"] = float(p.get("bankroll") or 3.0)
        weights["bankroll_dollars"] = float(p.get("bankroll_dollars") or 1000.0)
        for k in ("cash_pct", "avg_roi_win", "diversity_lambda"):
            if p.get(k) is not None:
                weights[k] = float(p[k])
    sel = {"_note": "written by the Backtest page: the run's contest selection for scripts/replay.py",
           "weights": weights,
           "picks": [{"contest_id": int(c), "entries": int(n)} for c, n in sorted(run["picks"].items(), key=lambda kv: int(kv[0]))]}
    path = run_root(run["name"]) / "selection.json"
    path.write_text(json.dumps(sel, indent=1))
    return path


def replay_command(run: dict, redo: bool = True) -> list[str]:
    """The replay for this run; redo=False resumes (finished slates and repeats are kept, only the missing ones run)."""
    p = run["params"]
    root = run_root(run["name"])
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "replay.py"), "--since", run["since"], "--until", run["until"],
           "--menu", str(root / "selection.json"), "--repeats", str(p["repeats"]), "--seed", str(p["seed"]), "--workers", str(p["workers"]),
           "--lineups", str(p["lineups"]), "--sims", str(p["sims"]), "--field-cap", str(p["field_cap"]), "--out", str(root), "--no-plot",
           "--replay-db", str(root / "replay.sqlite")]
    if not (root / "replay.sqlite").exists():                 # a fresh copy of the live database, so new captures are in it
        cmd.append("--rebuild-db")
    if redo and (root / "nfl").is_dir():
        cmd.append("--redo")
    return cmd


def resettle_command(run: dict) -> list[str]:
    """Settle the replayed slates again against the live database (after a scrape, a repair or a settlement fix), no re-run."""
    root = run_root(run["name"])
    return [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "replay.py"), "--since", run["since"], "--until", run["until"],
            "--report-only", "--out", str(root), "--replay-db", str(root / "replay.sqlite"), "--no-plot"]


def validate_command(run: dict) -> list[str]:
    return [sys.executable, "-u", str(SCRIPTS / "validate.py"), "--replay", str(run_root(run["name"])), "--repeat", "0", "--title", f"backtest {run['name']}"]


def settle_command(run: dict, username: Optional[str], scrape: bool = True) -> list[str]:
    cmd = [sys.executable, "-u", str(SCRIPTS / "settle.py"), "--since", run["since"], "--until", run["until"]]
    user = username or os.environ.get("IVAN_DK_USERNAME")
    if user:
        cmd += ["--username", user]
    if not scrape:
        cmd.append("--no-scrape")
    return cmd


def fetch_commands(run: dict, keys: list[str], ids: list[int], username: Optional[str], buy_props: bool) -> list[tuple[str, list[str]]]:
    """(job name, command) per ticked data item, in the order they should run."""
    out = []
    if "fetch_actuals" in keys:
        out.append(("Fetch actuals (schedule, box scores, DST)", [sys.executable, "-u", str(SCRIPTS / "fetch_actuals.py")]))
    if "backfill" in keys:
        cmd = [sys.executable, "-u", str(SCRIPTS / "backfill_archive.py"), "--since", run["since"], "--until", run["until"]]
        if buy_props:
            cmd.append("--odds")
        out.append((f"Backfill captures {run['since']}..{run['until']}" + (" with props" if buy_props else ""), cmd))
    if "scrape" in keys and ids:
        cmd = [sys.executable, "-u", str(SCRIPTS / "get_contest_data.py"), "--contest-ids", ",".join(map(str, ids))]
        user = username or os.environ.get("IVAN_DK_USERNAME")
        if user:
            cmd += ["--username", user]
        out.append((f"Scrape {len(ids)} contests", cmd))
    if "build_models" in keys:
        out.append((f"Build models as of {run['since']}", [sys.executable, "-u", str(SCRIPTS / "build_models.py"), "--cutoff", run["since"]]))
    return out


# --------------------------------------------------------------------------- progress
STAGES = ("restore", "project", "generate", "simulate", "choose")


def progress_view(run: dict, jobs: list[dict]) -> dict:
    """What the run is doing right now: its jobs, the replay's per-slate stages and repeats, the latest log lines."""
    root = run_root(run["name"])
    rep = bv.run_progress(root)
    slates = []
    for d in sorted((root / "nfl").iterdir()) if (root / "nfl").is_dir() else []:
        if not (d / "replay.json").is_file():
            continue
        try:
            m = json.loads((d / "replay.json").read_text())
        except (OSError, ValueError):
            continue
        done_stages = [s for s in STAGES if s in (m.get("stages") or {})]
        reps = m.get("repeats") or []
        for r in reps:
            for s in STAGES:
                if s in (r.get("stages") or {}) and s not in done_stages:
                    done_stages.append(s)
        wanted = int((m.get("options") or {}).get("repeats") or 1)
        ok = sum(1 for r in reps if r.get("status") == "ok")
        log = d / "replay.log"
        last = ""
        if log.is_file():
            lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
            last = lines[-1][:160] if lines else ""
        slates.append({"lock": (m.get("lock") or "")[:16].replace("T", " "), "key": m.get("slate_key"), "mode": m.get("mode"), "status": m.get("status"),
                       "stages": done_stages, "next": next((s for s in STAGES if s not in done_stages), None) if m.get("status") == "running" else None,
                       "repeats_ok": ok, "repeats_wanted": wanted, "seconds": m.get("seconds"), "error": m.get("error"), "last_line": last,
                       "contests": len(m.get("contests") or [])})
    active = any(j["status"] in ("queued", "running") for j in jobs)
    wanted = sum(s["repeats_wanted"] for s in slates) or 1
    done = sum(s["repeats_ok"] for s in slates)
    share = min(1.0, done / wanted) if wanted else 0.0
    phase = "idle"
    if active:
        j = next(j for j in jobs if j["status"] in ("queued", "running"))
        phase = f"{j['stage']} ({j['status']})"
    elif slates:
        phase = "finished" if all(s["status"] in ("ok", "error") for s in slates) else "stopped"
    return {"active": active, "phase": phase, "slates": slates, "done": done, "wanted": wanted, "share": share, "percent": int(round(share * 100)),
            "n_ok": sum(1 for s in slates if s["status"] == "ok"), "n_err": sum(1 for s in slates if s["status"] == "error"),
            "no_slates": bool(not slates and any(j["stage"] == "replay" and j["status"] == "done" for j in jobs)),
            "stale": bool(rep and rep.get("stale"))}


# --------------------------------------------------------------------------- results
def _csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as fh:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) else f
    except (TypeError, ValueError):
        return v


def _pct(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_view(run: dict) -> dict:
    """The run's results: the report (sim vs actual, repeats), the checks, and run -> slate -> contest with the spread across repeats."""
    root = run_root(run["name"])
    nfl = root / "nfl"
    reports = sorted(nfl.glob("report_*")) if nfl.is_dir() else []
    report_dir = reports[-1] if reports and (reports[-1] / "summary.json").is_file() else None
    validation = nfl / "validation" / "rep_000"
    manifests = []
    for d in sorted(nfl.iterdir()) if nfl.is_dir() else []:
        if (d / "replay.json").is_file():
            try:
                manifests.append(json.loads((d / "replay.json").read_text()))
            except (OSError, ValueError):
                pass
    view = {**run, "root": root, "progress": bv.run_progress(root),
            "slates_ok": sum(1 for m in manifests if m.get("status") == "ok"), "slates_total": len(manifests),
            "repeats": max((len(m.get("repeats") or []) for m in manifests), default=0),
            "models": sorted({(m.get("models") or {}).get("cutoff") or "live" for m in manifests if m.get("models")}),
            "report": bv.report_view(report_dir) if report_dir else None,
            "validation": vv.report_view(validation) if (validation / "summary.json").is_file() else None,
            "failed": [{"slate": m.get("slate_key"), "lock": m.get("lock"), "error": m.get("error")} for m in manifests if m.get("status") == "error"]}
    sva = view["report"]["sim_vs_actual"] if view["report"] else []
    field_rows = {int(r["contest_id"]): r for r in ((view["validation"] or {}).get("field") or {}).get("rows", []) if r.get("contest_id") is not None}
    # our entries placed in the real field (the report's entries.csv) and the field's exposures vs actual ownership (validation players.csv)
    entries: dict[int, list] = {}
    for row in (_csv(report_dir / "entries.csv") if report_dir else []):
        try:
            entries.setdefault(int(row["contest_id"]), []).append(row)
        except (TypeError, ValueError):
            continue
    exposures: dict[int, list] = {}
    for row in (_csv(validation / "players.csv") if (validation / "players.csv").is_file() else []):
        try:
            cid_ = int(row["contest_id"])
        except (TypeError, ValueError):
            continue
        if row.get("actual_own") not in (None, ""):
            exposures.setdefault(cid_, []).append(row)
    dist: dict[int, list] = {}
    for row in (_csv(report_dir / "repeats_by_contest.csv") if report_dir else []):
        dist.setdefault(int(row["contest_id"]), []).append(row)
    slates: dict[str, dict] = {}
    for row in sva:
        cid = int(row["contest_id"])
        profits = [float(x["actual_profit"]) for x in dist.get(cid, []) if x.get("actual_profit") not in (None, "")]
        sims = [float(x["sim_expected_profit"]) for x in dist.get(cid, []) if x.get("sim_expected_profit") not in (None, "")]
        sl = slates.setdefault(str(row["slate"]), {"slate": str(row["slate"]), "contests": [], "fees": 0.0, "sim": 0.0, "actual": 0.0})
        fr = field_rows.get(cid, {})
        ours = sorted(entries.get(cid, []), key=lambda e: (float(e.get("rank") or 1e9)))
        exp = sorted(exposures.get(cid, []), key=lambda r: -float(r.get("actual_own") or 0))[:12]
        sl["contests"].append({**row, "repeats": len(profits), "actual_mean": sum(profits) / len(profits) if profits else None,
                               "field": {k: fr.get(k) for k in ("sim_cash_p10", "sim_cash_p50", "sim_cash_p90", "actual_cash_line", "sim_win_p50", "actual_win",
                                                                   "sim_top_share", "actual_top_share", "simulated_size", "field_size")},
                               "entries_ranked": [{k: e.get(k) for k in ("entry_id", "points", "rank", "payout", "sim_cash_pct", "sim_avg_roi", "lineup", "field_size")} for e in ours],
                               "exposures": [{k: r.get(k) for k in ("name", "position", "slot", "salary", "proj_mean", "actual_fpts", "own_proj", "own_field", "actual_own")} for r in exp],
                               "actual_p10": _pct(profits, 10), "actual_p90": _pct(profits, 90),
                               "sim_mean": sum(sims) / len(sims) if sims else None,
                               "p_profit": (sum(1 for x in profits if x > 0) / len(profits)) if profits else None})
        sl["fees"] += float(row.get("fees") or 0); sl["sim"] += float(row.get("sim_expected_profit") or 0); sl["actual"] += float(row.get("actual_profit") or 0)
    view["slates"] = sorted(slates.values(), key=lambda s: s["slate"])
    view["dist"] = bv.dist_rows(view["report"]["repeats"]) if view["report"] and view["report"].get("repeats") else []
    return view
