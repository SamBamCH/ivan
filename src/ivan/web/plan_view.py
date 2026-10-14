# src/ivan/web/plan_view.py
"""
The backtest plan page: a window of slates and a contest pool, what each
slate has and lacks for an honest replay, and the commands that fill the
gaps (ivan.sports.nfl.readiness does the looking; this module shapes it
for the template and builds the job commands).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ivan.core.replay import Menu
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nfl.paths import OUTPUT_DIR

MENU_PATH = PROJECT_ROOT / "config" / "replay" / "nfl.json"
SCRIPTS = PROJECT_ROOT / "scripts" / "nfl"


def default_menu_text() -> str:
    return MENU_PATH.read_text() if MENU_PATH.exists() else "{}"


def menu_from_text(raw: str) -> Menu:
    """A Menu from the form's json (the replay menu format); falls back to the file."""
    raw = (raw or "").strip()
    if not raw:
        return Menu.load(MENU_PATH)
    tmp = OUTPUT_DIR / "plan_menu.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    json.loads(raw)                                   # validate before writing
    tmp.write_text(raw)
    return Menu.load(tmp)


def window(since: str, until: str) -> tuple[datetime, datetime]:
    a = datetime.fromisoformat(since)
    b = datetime.fromisoformat(until) + timedelta(days=1) - timedelta(seconds=1)
    return a, b


def plan_view(session, since: str, until: str, menu_text: str) -> dict:
    from ivan.sports.nfl.readiness import backtest_plan
    a, b = window(since, until)
    menu = menu_from_text(menu_text)
    plan = backtest_plan(session, a, b, menu)
    rows = []
    for s in plan["slates"]:
        rows.append({"slate_key": s.slate_key, "mode": s.mode, "lock": s.lock.strftime("%Y-%m-%d %H:%M"), "captures": s.captures,
                     "capture": (f"{s.capture_source}, {s.capture_minutes_before_lock:.0f} min before lock, {s.capture_players} players, {s.capture_odds} props"
                                 if s.captures else "none"),
                     "contests_at_lock": s.contests_at_lock, "picks": len(s.picks), "scraped": s.scraped_picks, "our_entries": s.our_entries,
                     "box_scores": f"{s.box_scores_teams}/{s.teams} teams" if s.teams else (f"{s.box_scores_week_rows} rows" if s.box_scores_week_rows else "none"),
                     "box_ok": s.box_scores, "week": f"{s.season_week[0]} wk {s.season_week[1]}" if s.season_week else "",
                     "models": s.models_label, "models_fallback": bool(s.models.get("fallback")), "replayed": ", ".join(s.replayed),
                     "status": s.status, "missing": s.missing,
                     "contests": [{"contest_id": p.contest_id, "name": p.name, "fee": p.entry_fee, "size": p.total_entries, "entries": p.entries,
                                   "payouts": p.has_payouts, "field": p.field_scores, "points": p.players_with_points, "own": p.players_with_ownership,
                                   "ours": p.our_entries, "scraped": p.scraped} for p in s.picks]})
    summary = plan["summary"]
    ids = summary["unscraped_contest_ids"]
    cmds = {
        "fetch_actuals": "venv/bin/python scripts/nfl/fetch_actuals.py",
        "scrape": ("venv/bin/python scripts/nfl/get_contest_data.py --username YOUR_DK_USERNAME --contest-ids " + ",".join(map(str, ids))) if ids else "",
        "backfill": f"venv/bin/python scripts/nfl/backfill_archive.py --since {since} --until {until}",
        "build_models": f"venv/bin/python scripts/nfl/build_models.py --cutoff {since}",
        "replay": f"venv/bin/python scripts/replay.py --since {since} --until {until} --workers 3 --no-report --out data/nfl/output/replay_{since}_{until}",
    }
    return {"since": since, "until": until, "rows": rows, "summary": summary, "model_versions": plan["model_versions"], "commands": cmds,
            "unscraped_ids": ids}


# --------------------------------------------------------------------------- job commands
def replay_root(since: str, until: str, name: str = "") -> Path:
    return OUTPUT_DIR / (f"replay_{name}" if name else f"replay_{since}_{until}")


def replay_command(since: str, until: str, root: Path, workers: int = 3, repeats: int = 1, redo: bool = False) -> list[str]:
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "replay.py"), "--since", since, "--until", until, "--workers", str(workers),
           "--repeats", str(repeats), "--no-report", "--out", str(root)]
    if redo:
        cmd.append("--redo")
    return cmd


def validate_replay_command(root: Path, title: str = "") -> list[str]:
    cmd = [sys.executable, "-u", str(SCRIPTS / "validate.py"), "--replay", str(root), "--repeat", "0"]
    if title:
        cmd += ["--title", title]
    return cmd


def fetch_actuals_command() -> list[str]:
    return [sys.executable, "-u", str(SCRIPTS / "fetch_actuals.py")]


def scrape_command(ids: list[int], username: Optional[str]) -> list[str]:
    cmd = [sys.executable, "-u", str(SCRIPTS / "get_contest_data.py"), "--contest-ids", ",".join(map(str, ids))]
    user = username or os.environ.get("IVAN_DK_USERNAME")
    if user:
        cmd += ["--username", user]
    return cmd


def backfill_command(since: str, until: str, odds: bool = False) -> list[str]:
    cmd = [sys.executable, "-u", str(SCRIPTS / "backfill_archive.py"), "--since", since, "--until", until]
    if odds:
        cmd.append("--odds")
    return cmd


def build_models_command(cutoff: str) -> list[str]:
    return [sys.executable, "-u", str(SCRIPTS / "build_models.py"), "--cutoff", cutoff]
