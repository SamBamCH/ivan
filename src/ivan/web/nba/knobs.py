"""
The pipeline's knobs: the defaults from config/nba.json, the consumer limits, the form the desk keeps between runs and the desk_config.json the job scripts read.
"""
from __future__ import annotations

import json
from pathlib import Path
from ivan.core.config import config_path
from ivan.sports.nba.paths import NBA_DATA_DIR
from .core import LIMITED, LIMITS, MAX_JOBS


def _config() -> dict:
    with open(config_path("nba"), encoding="utf-8") as f:
        return json.load(f)


def _pipeline_defaults() -> dict:
    """The panel's fields, seeded from config/nba.json: the generator's knobs, the sim's, the chooser's weights."""
    c = _config(); g = c.get("generate", {}); ch = c.get("choose", {})
    fmt = lambda v: "" if v is None else ("%g" % v if isinstance(v, (int, float)) else str(v))
    return {"lineups": 25000, "jobs": MAX_JOBS, "sims": fmt(c.get("sim", {}).get("n_sims", 1000)), "field_cap": fmt(c.get("field", {}).get("cap", 10000)),
            "method": ch.get("method", "greedy"), "min_fpts_frac": fmt(g.get("min_fpts_frac", 0.9)),
            "randomness": fmt(g.get("randomness", 0.25)), "n_samples": fmt(g.get("n_samples", 5000)), "jitter": fmt(g.get("jitter", 0.05)),
            "player_min_fpts": fmt(g.get("player_min_fpts", c.get("player_min_fpts", 15))),
            "max_salary": fmt(c.get("max_salary", 50000)), "min_salary": fmt(c.get("min_salary", 49000)), "max_salary_per_team": fmt(c.get("max_salary_per_team", 30000)),
            "cash_pct": fmt(ch.get("cash_pct", 0.5)), "avg_roi_win": fmt(ch.get("avg_roi_win", 0.5)), "diversity_lambda": fmt(ch.get("diversity_lambda", 0.1)),
            "bankroll": fmt(ch.get("bankroll", 3.0)), "jam": "",
            "ls_candidates": "10", "ls_field_cap": "5000", "ls_sims": "1000", "ls_source": "auto", "ls_field": "shared", "ls_live": ""}


def clamp_form(form: dict) -> list:
    """Clamp the pipeline settings to LIMITS on a limited desk; returns the notes of what was lowered."""
    notes = []
    if not LIMITED:
        return notes
    for k, cap in LIMITS.items():
        raw = str(form.get(k) or "").strip()
        try:
            v = float(raw)
        except ValueError:
            continue
        if v > cap:
            form[k] = str(int(cap)); notes.append(f"{k} {raw} -> {int(cap)}")
    return notes


PIPELINE_DEFAULTS = _pipeline_defaults()


if LIMITED:
    for _k, _cap in LIMITS.items():
        if _k in PIPELINE_DEFAULTS and float(PIPELINE_DEFAULTS[_k] or 0) > _cap:
            PIPELINE_DEFAULTS[_k] = str(_cap)


PIPELINE_DEFAULTS.setdefault("fresh", "on")          # generate starts the pool fresh unless the box is unticked (add to the pool: a second scenario's lineups)


pipeline_form: dict = dict(PIPELINE_DEFAULTS)


DESK_CONFIG = NBA_DATA_DIR / "jobs" / "desk_config.json"           # config/nba.json with the panel's knobs applied; the jobs read it


MAX_UPLOAD = 5 * 1024 * 1024   # the DK entries export is tens of KB


def _field(raw, label: str, kind=float):
    raw = str(raw if raw is not None else "").strip()
    if not raw:
        return None
    try:
        return kind(float(raw))
    except ValueError:
        raise ValueError(f"{label}: {raw!r} is not a number")


def write_desk_config(form: dict) -> Path:
    """config/nba.json with the panel's knobs applied, written for the jobs (-c). Blank fields keep the file's values."""
    c = _config(); g = c.setdefault("generate", {}); ch = c.setdefault("choose", {})
    for key in ("max_salary", "min_salary", "max_salary_per_team"):
        v = _field(form.get(key), key, int)
        if v is not None:
            c[key] = v
    for key, kind in (("n_samples", int), ("jitter", float), ("min_fpts_frac", float), ("player_min_fpts", float), ("randomness", float)):
        v = _field(form.get(key), key, kind)
        if v is not None:
            g[key] = v
    if str(form.get("player_min_fpts") or "").strip():
        c["player_min_fpts"] = g["player_min_fpts"]
    for key in ("cash_pct", "avg_roi_win", "diversity_lambda", "bankroll"):
        v = _field(form.get(key), key)
        if v is not None:
            ch[key] = v
    if form.get("jam"):
        ch["diversity_lambda"] = 0.0                              # jam: the top lineups by quality, no spread between them
    ch["method"] = str(form.get("method") or ch.get("method", "greedy"))
    if int(c["min_salary"]) > int(c["max_salary"]):
        raise ValueError("min salary is above max salary")
    DESK_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    DESK_CONFIG.write_text(json.dumps(c, indent=2))
    return DESK_CONFIG
