# src/ivan/web/validate_view.py
"""
The Validation page: projected vs actual per component, read back from the
summary.json that scripts/nfl/validate.py writes (ivan.sports.nfl.validate
/ ivan.core.validation). Nothing here computes.

A validation report is a directory with summary.json under
    data/nfl/output/validation/<stamp>/                  (live runs)
    data/nfl/output/replay*/nfl/validation/rep_<k>/      (replays)
    data/nfl/output/backtests/<run>/nfl/validation/...   (the Backtest page's runs)
    data/nfl/output/models/<component>/<run>/nfl/validation/...   (the Models page's tests)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nfl.paths import OUTPUT_DIR

FILES = ("summary.json", "environment.csv", "usage.csv", "players.csv", "field.csv", "lineups.csv", "pool.csv", "pool_deciles.csv", "environment_backtest.csv")
ENV_KEYS = ("n", "actual_mean", "proj_mean", "bias", "mae", "mae_league", "mae_lines_only", "corr", "corr_lines_only", "sd_ratio", "coverage_10_90", "pit_mean")
STATUS_ORDER = {"fail": 0, "warn": 1, "pass": 2, "na": 3}
COMPONENTS = ("environment", "usage", "projections", "ownership", "field", "dupes", "simulator", "chooser")


def _candidates() -> list[Path]:
    out = []
    root = OUTPUT_DIR / "validation"
    if root.exists():
        out += [d for d in root.iterdir() if d.is_dir()]
    for rr in list(OUTPUT_DIR.glob("replay*")) + list((OUTPUT_DIR / "backtests").glob("*")) + list((OUTPUT_DIR / "models").glob("*/*")):
        v = rr / "nfl" / "validation"
        if v.exists():
            out += [d for d in v.iterdir() if d.is_dir()]
    return [d for d in out if (d / "summary.json").is_file()]


def report_id(d: Path) -> str:
    try:
        return str(d.relative_to(OUTPUT_DIR))
    except ValueError:                  # a report outside the output tree (tests, ad hoc runs): the path itself, which report_path refuses
        return str(d)


def report_path(rid: str) -> Optional[Path]:
    d = (OUTPUT_DIR / rid).resolve()
    if OUTPUT_DIR.resolve() not in d.parents or not (d / "summary.json").is_file():
        return None
    return d


def reports() -> list[dict]:
    rows = []
    for d in _candidates():
        try:
            s = json.loads((d / "summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({"id": report_id(d), "path": d, "title": s.get("title") or d.name, "created_at": s.get("created_at"),
                     "since": (s.get("since") or "")[:10], "until": (s.get("until") or "")[:10], "contests": s.get("contests", 0),
                     "counts": s.get("check_counts", {}), "mtime": (d / "summary.json").stat().st_mtime})
    return sorted(rows, key=lambda r: -r["mtime"])


ERR_KEYS = ("n", "proj_mean", "actual_mean", "bias", "mae", "rmse", "corr", "baseline_mae", "sd_ratio", "z_abs_mean", "coverage_10_90",
            "coverage_05_95", "pit_mean", "pit_shape", "n_pit")
OWN_KEYS = ("n", "bias", "mae", "corr", "spearman", "proj_sum", "actual_sum", "max_abs_err", "chalk_bias")
POOL_ROW_KEYS = ("picked_actual_roi", "picked_sim_roi", "topk_by_sim_roi_actual_roi", "topk_by_sim_cash_actual_roi", "random_k_actual_roi",
                 "random_k_p10", "random_k_p90", "p_random_beats_picked", "spearman_sim_cash_vs_actual", "spearman_sim_roi_vs_actual", "pool_actual_roi",
                 "pool_sim_roi", "picked_lineups", "picked_entries", "pool_lineups", "contest_name", "slate")
POOL_KEYS = ("spearman_median", "spearman_positive_share", "picked_actual_roi", "picked_sim_roi", "random_k_actual_roi", "topk_by_sim_roi_actual_roi",
             "pool_actual_roi", "contests_picked_beats_random", "contests")
WORST_KEYS = ("slate", "contest_id", "slot", "name", "team", "position", "salary", "proj_mean", "proj_sd", "actual_fpts", "err", "pit", "own_proj",
              "own_field", "actual_own", "source")


def _fill(row: Optional[dict], keys) -> dict:
    """Every key present (None when missing), so the template never meets an undefined attribute."""
    row = dict(row or {})
    for k in keys:
        row.setdefault(k, None)
    return row


def _fill_players(p: dict) -> dict:
    p = dict(p or {})
    p["overall"] = _fill(p.get("overall"), ERR_KEYS)
    for k in ("by_position", "by_slot", "by_salary_tier", "by_own_decile", "by_slate", "by_volume_source", "by_scored"):
        p[k] = [_fill(r, ERR_KEYS) for r in (p.get(k) or [])]
    p["ownership"] = _fill(p.get("ownership"), OWN_KEYS)
    p["ownership_field"] = _fill(p.get("ownership_field"), OWN_KEYS)
    p["ownership_by_decile"] = [_fill(r, OWN_KEYS) for r in (p.get("ownership_by_decile") or [])]
    p["worst"] = [_fill(r, WORST_KEYS) for r in (p.get("worst") or [])]
    p.setdefault("pit_method", "")
    p.setdefault("baseline", "")
    return p


def report_view(d: Path) -> dict:
    s = json.loads((d / "summary.json").read_text())
    checks = sorted(s.get("checks", []), key=lambda c: (STATUS_ORDER.get(c["status"], 9), COMPONENTS.index(c["component"]) if c["component"] in COMPONENTS else 9))
    rep = s.get("reports", {})
    players = _fill_players(rep.get("players") or {})
    pit = players.get("pit_hist") or []
    pit_max = max([h["share"] for h in pit] + [0.1]) or 0.1
    for h in pit:
        h["width"] = 100.0 * h["share"] / pit_max
        h["ideal_width"] = 100.0 * h["ideal"] / pit_max
    pool = _fill(rep.get("pool") or {}, POOL_KEYS)
    pool["rows"] = [_fill(r, POOL_ROW_KEYS) for r in (pool.get("rows") or [])]
    dec = pool.get("deciles") or []
    lineups = dict(rep.get("lineups") or {})
    for k in ("fees", "actual_profit", "actual_roi", "sim_expected_profit", "sim_roi", "sim_cash_pct", "actual_cash_rate", "settled", "dupes"):
        lineups.setdefault(k, None)
    lineups["by_contest"] = lineups.get("by_contest") or []
    field = dict(rep.get("field") or {})
    for k in ("cash_line", "winning_score", "entry_score", "dupes"):
        field[k] = _fill(field.get(k), ("n", "in_band_share", "bias", "mae", "p50_bias", "p90_bias", "top_share_ratio_median", "dupe_max_ratio_median"))
    field["rows"] = field.get("rows") or []
    env = dict(rep.get("environment") or {})
    env["quantities"] = {q: _fill(v, ENV_KEYS) for q, v in (env.get("quantities") or {}).items()}
    env["rows"] = env.get("rows") or []
    btk = env.get("backtest") or {}
    env["backtest_quantities"] = {q: {"overall": _fill(v.get("overall"), ENV_KEYS),
                                      "by_season": [_fill(r, ENV_KEYS + ("season",)) for r in v.get("by_season", [])],
                                      "by_spread": [_fill(r, ENV_KEYS + ("spread",)) for r in v.get("by_spread", [])]}
                                  for q, v in (btk.get("quantities") or {}).items()}
    env["residual_corr"] = btk.get("residual_corr_within_team") or {}
    use = dict(rep.get("usage") or {})
    use["stats"] = {k: _fill(v, ("n", "coverage_mean", "coverage_median", "coverage_min", "coverage_max", "over_share", "floored_share",
                                 "total_mean", "fixed_mean", "residual_mean", "unpriced_mean")) for k, v in (use.get("stats") or {}).items()}
    use["rows"] = use.get("rows") or []
    files = [f for f in FILES if (d / f).is_file()]
    return {"id": report_id(d), "environment": env, "usage": use, "summary": s, "checks": checks, "counts": s.get("check_counts", {}), "coverage": s.get("coverage", []),
            "players": players, "players_artifacts": _fill_players(rep.get("players_artifacts") or {}), "pit": pit,
            "field": field, "lineups": lineups, "pool": pool, "deciles": dec, "files": files}


def validate_command(mode: str, since: str, until: str, replay_root: Optional[Path] = None, repeat: int = 0) -> list[str]:
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "nfl" / "validate.py")]
    if since:
        cmd += ["--since", since]
    if until:
        cmd += ["--until", until]
    if mode == "replay" and replay_root is not None:
        cmd += ["--replay", str(replay_root), "--repeat", str(int(repeat))]
    return cmd


def replay_roots() -> list[str]:
    return sorted(d.name for d in OUTPUT_DIR.glob("replay*") if (d / "nfl").is_dir())
