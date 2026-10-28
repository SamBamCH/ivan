# src/ivan/web/models_view.py
"""
The Models page: test one piece of the pipeline against what happened,
without running the pieces after it.

Every component maps to the last replay stage it needs (scripts/replay.py
--through) and to the validation checks that score it. A test run replays
a window through that stage into

    data/nfl/output/models/<component>/<stamp>/
        run.json              window, stage, parameters, the git commit, the jobs
        nfl/<slate dirs>      the replay's outputs for the stages that ran
        nfl/validation/rep_000/summary.json   the checks (scripts/nfl/validate.py)

and the page keeps a scoreboard per component: one row per run, one column
per check, with the commit the code was at, so a model change can be read
off as a before / after. Nothing here computes; the replay and the
validation do.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ivan.core.gitinfo import git_state  # noqa: F401 - the page's runs are labelled with the commit
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nfl.paths import OUTPUT_DIR
from ivan.web import validate_view as vv

MODELS_OUT = OUTPUT_DIR / "models"
MENU_PATH = PROJECT_ROOT / "config" / "replay" / "nfl.json"
SCRIPTS = PROJECT_ROOT / "scripts"

# the pipeline's components, in pipeline order: what each predicts, what it is scored against, how far the replay must run
COMPONENTS = [
    {"key": "projections", "label": "Projections", "through": "project", "checks": ("projections",), "stage": "projections",
     "predicts": "every player's points distribution (mean, spread, quantiles) from the props, the lines and the usage model",
     "against": "the box scores: bias, error against a salary-only line, where the actual lands in the projected distribution",
     "cost": "seconds to a couple of minutes per slate (fit only)"},
    {"key": "environment", "label": "Game environment", "through": "project", "checks": ("environment", "usage"), "stage": None,
     "predicts": "team points, plays and pass rate from the lines (layer 1), and how the priced player lines cover each team's volume (layer 2)",
     "against": "the schedule and team box scores; the usage coverage needs no actuals",
     "cost": "seconds to a couple of minutes per slate (fit only)"},
    {"key": "ownership", "label": "Ownership", "through": "project", "checks": ("ownership",), "stage": "ownership",
     "predicts": "each player's projected ownership (the learned model, or the field-sampling heuristic)",
     "against": "the ownership DraftKings reported for scraped contests: MAE, rank correlation, chalk bias",
     "cost": "seconds to a couple of minutes per slate (fit only)"},
    {"key": "field", "label": "Field", "through": "field", "checks": ("field",), "stage": "field",
     "predicts": "the opposing field: the generated pool, the sampling weights raked to ownership, the copies the dupe model hands out",
     "against": "the real field, with the sampled field scored on the box scores: exposure vs actual ownership, cash line, winning score",
     "cost": "minutes per slate (generate the pool and sample the fields; no simulation)"},
    {"key": "dupes", "label": "Dupes", "through": "field", "checks": ("dupes",), "stage": "dupes",
     "predicts": "how many copies the field holds of each lineup (the product rule with its captain, salary-left and projection-gap terms)",
     "against": "the copies the scraped fields actually held: the most-duplicated lineup's share, sim vs real",
     "cost": "minutes per slate; the lineup-level analysis on the Dupes page needs no replay at all"},
    {"key": "simulator", "label": "Simulator", "through": "simulate", "checks": ("simulator",), "stage": "simulator",
     "predicts": "each candidate lineup's cash % and ROI against the sampled field over correlated outcome draws",
     "against": "the same lineups placed in the real field with the real scores: does sim ROI rank them?",
     "cost": "tens of minutes per slate (the full simulation, no chooser)"},
    {"key": "chooser", "label": "Chooser", "through": "choose", "checks": ("chooser",), "stage": None,
     "predicts": "which lineups to enter, and the ROI the sim promised for them",
     "against": "what those entries returned, against random picks from the same pool and the top-K by sim ROI",
     "cost": "tens of minutes per slate (the whole pipeline); this is the Backtest page's run"},
]
BY_KEY = {c["key"]: c for c in COMPONENTS}
STAGE_PARAMS = {        # which parameters a stage takes; the form shows only those
    "project": ("workers",),
    "field": ("workers", "lineups", "field_cap", "seed"),
    "simulate": ("workers", "lineups", "field_cap", "sims", "seed"),
    "choose": ("workers", "lineups", "field_cap", "sims", "repeats", "seed"),
}
DEFAULTS = {"workers": 3, "lineups": 15000, "field_cap": 10000, "sims": 300, "repeats": 1, "seed": 0}


def component(key: Optional[str]) -> dict:
    return BY_KEY.get(key or "", COMPONENTS[0])


# --------------------------------------------------------------------------- runs
def run_root(key: str, stamp: str) -> Path:
    return MODELS_OUT / key / stamp


def new_run(key: str, since: str, until: str, params: dict, label: str = "") -> dict:
    datetime.fromisoformat(since); datetime.fromisoformat(until)
    c = BY_KEY[key]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    clean = "".join(ch for ch in label.strip().replace(" ", "_") if ch.isalnum() or ch in "-_.")
    name = f"{stamp}-{clean}" if clean else stamp
    root = run_root(key, name)
    root.mkdir(parents=True, exist_ok=False)
    allowed = STAGE_PARAMS[c["through"]]
    run = {"name": name, "component": key, "through": c["through"], "since": since, "until": until, "label": label.strip(),
           "params": {k: int(params.get(k, DEFAULTS[k])) for k in allowed}, "git": git_state(),
           "created_at": datetime.now().isoformat(timespec="seconds"), "jobs": []}
    save_run(run)
    return run


def save_run(run: dict) -> dict:
    (run_root(run["component"], run["name"]) / "run.json").write_text(json.dumps(run, indent=1, default=str))
    return run


def load_run(key: str, name: str) -> Optional[dict]:
    p = run_root(key, name) / "run.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def replay_command(run: dict) -> list[str]:
    root = run_root(run["component"], run["name"])
    p = {**DEFAULTS, **run["params"]}
    return [sys.executable, "-u", str(SCRIPTS / "replay.py"), "--since", run["since"], "--until", run["until"], "--through", run["through"],
            "--menu", str(MENU_PATH), "--repeats", str(p["repeats"]), "--seed", str(p["seed"]), "--workers", str(p["workers"]),
            "--lineups", str(p["lineups"]), "--sims", str(p["sims"]), "--field-cap", str(p["field_cap"]),
            "--out", str(root), "--replay-db", str(root / "replay.sqlite"), "--rebuild-db", "--no-plot"]


def validate_command(run: dict) -> list[str]:
    root = run_root(run["component"], run["name"])
    return [sys.executable, "-u", str(SCRIPTS / "nfl" / "validate.py"), "--replay", str(root), "--repeat", "0",
            "--title", f"models {run['component']} {run['name']}"]


def validation_dir(run: dict) -> Path:
    return run_root(run["component"], run["name"]) / "nfl" / "validation" / "rep_000"


def _slate_status(root: Path) -> dict:
    """How many slates the replay finished, and the first error, from the manifests."""
    nfl = root / "nfl"
    ok = err = total = 0
    error = None
    for d in sorted(nfl.iterdir()) if nfl.is_dir() else []:
        mp = d / "replay.json"
        if not mp.is_file():
            continue
        try:
            m = json.loads(mp.read_text())
        except (OSError, ValueError):
            continue
        total += 1
        if m.get("status") == "ok":
            ok += 1
        elif m.get("status") == "error":
            err += 1
            error = error or f"{m.get('slate_key')}: {m.get('error')}"
    return {"ok": ok, "error": err, "total": total, "first_error": error}


def _report_id(vdir: Path) -> Optional[str]:
    try:
        return vv.report_id(vdir)
    except ValueError:                      # outside the output tree: no link to the Validation page
        return None


def runs(key: str, job_status: Optional[Callable[[str], Optional[str]]] = None) -> list[dict]:
    """Every test run of a component, newest first, with its checks (this component's only), models and job state."""
    c = BY_KEY[key]
    out = []
    root = MODELS_OUT / key
    for d in sorted(root.iterdir(), reverse=True) if root.is_dir() else []:
        run = load_run(key, d.name)
        if not run:
            continue
        vdir = validation_dir(run)
        summary = None
        if (vdir / "summary.json").is_file():
            try:
                summary = json.loads((vdir / "summary.json").read_text())
            except (OSError, ValueError):
                summary = None
        checks = [ch for ch in (summary or {}).get("checks", []) if ch.get("component") in c["checks"]]
        conf = ((summary or {}).get("confidence") or {}).get("stages", {}).get(c["stage"]) if c["stage"] else None
        jobs = [{**j, "status": job_status(j["id"]) if job_status else None} for j in run.get("jobs", [])]
        active = any(j["status"] in ("queued", "running") for j in jobs)
        failed = any(j["status"] == "failed" or (j["status"] == "cancelled") for j in jobs)
        status = "running" if active else ("done" if summary else ("failed" if failed else "no report"))
        out.append({**run, "root": d, "summary": summary, "checks": checks, "stage_score": conf, "jobs": jobs, "status": status,
                    "slates": _slate_status(d), "contests": (summary or {}).get("contests", 0),
                    "models": (summary or {}).get("models") or [], "validation_id": _report_id(vdir) if summary else None})
    return out


def scoreboard(run_list: list[dict]) -> dict:
    """Runs x checks: the columns are the component's check names in the order they first appear, newest run first."""
    cols: list[str] = []
    for r in run_list:
        for ch in r["checks"]:
            if ch["name"] not in cols:
                cols.append(ch["name"])
    rows = []
    for r in run_list:
        by_name = {ch["name"]: ch for ch in r["checks"]}
        rows.append({"run": r, "cells": [by_name.get(n) for n in cols]})
    return {"columns": cols, "rows": rows}


def run_view(run: dict) -> Optional[dict]:
    """The full validation report of one run (validate_view.report_view), for the component-specific tables."""
    vdir = validation_dir(run)
    return vv.report_view(vdir) if (vdir / "summary.json").is_file() else None
