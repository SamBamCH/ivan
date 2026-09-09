# src/ivan/web/pipeline_nfl.py
"""
The NFL pre-lock pipeline as the web UI sees it: ordered steps, each a thin
wrapper around one of the CLI scripts in scripts/nfl/. Nothing here knows how
the scripts work internally; it only builds their command lines and reports
coarse progress from the database while they run.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import text

from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nfl.paths import INPUT_DIR, OUTPUT_DIR

PYTHON = sys.executable
SCRIPTS = PROJECT_ROOT / "scripts" / "nfl"
WEB_INPUT_DIR = INPUT_DIR / "web"
WEB_OUTPUT_DIR = OUTPUT_DIR / "web"
CURRENT_ENTRIES = WEB_INPUT_DIR / "entries.csv"


@dataclass
class Field:
    name: str
    label: str
    kind: str = "number"            # number | checkbox
    default: object = None
    help: str = ""
    step: str = "1"
    min: object = None


@dataclass
class Step:
    key: str
    title: str
    blurb: str
    build: Callable[[dict], list[list[str]]]
    fields: list[Field] = field(default_factory=list)
    needs_entries: bool = False
    progress: Optional[Callable[[object], str]] = None   # (session) -> one-line status


def _num(form: dict, name: str, default, cast=int):
    v = form.get(name)
    return default if v in (None, "") else cast(v)


def _flag(form: dict, name: str) -> bool:
    return str(form.get(name, "")).lower() in ("on", "true", "1")


def _script(name: str) -> list[str]:
    return [PYTHON, str(SCRIPTS / name)]


_slate_cache: dict = {"at": 0.0, "items": []}


def slates(max_age: float = 120) -> list[dict]:
    """Upcoming DK slates, cached briefly so page loads don't hammer DraftKings."""
    if time.time() - _slate_cache["at"] > max_age:
        from ivan.sports.common.draftkings import SUPPORTED_GAME_TYPES, list_slates
        items = list_slates("nfl")
        for s in items:
            s["supported"] = s["game_type"] in SUPPORTED_GAME_TYPES
        _slate_cache.update(at=time.time(), items=items)
    return _slate_cache["items"]


def _require_slate(form: dict) -> str:
    dg = str(form.get("draft_group", "")).strip()
    if not dg.isdigit():
        raise ValueError("Pick a slate first. The list under step 1 loads from DraftKings.")
    return dg


def _players_progress(s) -> str:
    n, teams = s.execute(text("SELECT count(*), count(DISTINCT team) FROM nfl_player")).one()
    return f"{n} player rows across {teams} teams"


def _lineups_progress(s) -> str:
    return f"{s.execute(text('SELECT count(*) FROM lineup')).scalar():,} lineups in DB"


def _sims_progress(s) -> str:
    done = s.execute(text("SELECT count(DISTINCT contest_name) FROM lineup_simulation_results")).scalar()
    return f"{done} contest(s) simulated"


def _contests_progress(s) -> str:
    n = s.execute(text("SELECT count(*) FROM payouts")).scalar()
    return f"{n:,} payout rows in DB"


STEPS: list[Step] = [
    Step(
        key="ingest", title="Ingest the slate",
        blurb="Pulls players and salaries from DraftKings and player odds from the-odds-api, fits each player's fantasy-point distribution, then projects ownership by sampling a field of lineups. Run this first, and again close to lock for fresh odds and ownership.",
        fields=[Field("prelock", "Re-fetch players from DraftKings", "checkbox", True,
                      help="Leave on. Turn off only to refresh odds without touching the player pool."),
                Field("update_odds", "Update odds", "checkbox", True)],
        build=lambda f: [_script("ingest.py")
                         + (["-p", "--draft-group", _require_slate(f)] if _flag(f, "prelock") else [])
                         + ["--update-odds", "true" if _flag(f, "update_odds") else "false"]],
        progress=_players_progress,
    ),
    Step(
        key="contests", title="Fetch contests & payouts",
        blurb="Looks up the upcoming DraftKings contests and their payout tables. The simulator needs these for every contest in your entries file.",
        build=lambda f: [_script("get_contest_ids.py"), _script("get_payouts.py")],
        progress=_contests_progress,
    ),
    Step(
        key="generate", title="Generate the field",
        blurb="Builds the pool of candidate lineups that the simulator picks from. More lineups = more choice, longer run.",
        fields=[Field("num_lineups", "Lineups to generate", default=10000, min=100, step="100"),
                Field("num_jobs", "Parallel workers", default=max(1, min(8, os.cpu_count() or 1)), min=1,
                      help="Use the number of CPU cores, or fewer if the machine should stay responsive.")],
        build=lambda f: [_script("generate_lineups.py") + [str(_num(f, "num_lineups", 10000)), str(_num(f, "num_jobs", 4))]],
        progress=_lineups_progress,
    ),
    Step(
        key="sim", title="Simulate your contests",
        blurb="Simulates every contest in your entries file against the generated field and scores each candidate lineup (cash %, ROI, rank).",
        fields=[Field("n", "Simulations per contest", default=1000, min=100, step="100"),
                Field("c", "Field cap", default=10000, min=100, step="100",
                      help="Largest field size to simulate; bigger contests are scaled down to this."),
                Field("d", "Allow duplicate lineups in classic slates", "checkbox", False)],
        needs_entries=True,
        build=lambda f: [_script("sim.py") + ["--entries", str(CURRENT_ENTRIES), "-n", str(_num(f, "n", 1000)),
                                              "-c", str(_num(f, "c", 10000))] + (["-d"] if _flag(f, "d") else [])],
        progress=_sims_progress,
    ),
    Step(
        key="choose", title="Choose lineups",
        blurb="Picks the best lineups for each contest from the simulation results and writes a DraftKings upload file. Weights say what 'best' means.",
        fields=[Field("cash_pct", "Weight on cash %", default=0.5, min=0, step="0.05"),
                Field("avg_roi_win", "Weight on ROI", default=0.5, min=0, step="0.05"),
                Field("diversity_lambda", "Diversity", default=0.1, min=0, step="0.05",
                      help="Higher = more different lineups across your entries.")],
        needs_entries=True,
        build=lambda f: [_script("choose.py") + ["--entries", str(CURRENT_ENTRIES), "--no-prompt",
                                                 "--cash-pct", str(_num(f, "cash_pct", 0.5, float)),
                                                 "--avg-roi-win", str(_num(f, "avg_roi_win", 0.5, float)),
                                                 "--diversity-lambda", str(_num(f, "diversity_lambda", 0.1, float)),
                                                 "-o", str(WEB_OUTPUT_DIR / "chosen.csv")]],
    ),
]
STEP_BY_KEY = {s.key: s for s in STEPS}


def entries_summary() -> Optional[dict]:
    """What's in the currently uploaded entries file, or None if there isn't one."""
    if not CURRENT_ENTRIES.exists():
        return None
    try:
        df = pd.read_csv(CURRENT_ENTRIES, engine="python", usecols=["Entry ID", "Contest ID", "Contest Name"], dtype=str)
        df = df.dropna(subset=["Entry ID"])
        df = df[df["Entry ID"].str.strip() != ""]
        contests = (df.groupby(["Contest ID", "Contest Name"], sort=False).size()
                      .reset_index(name="entries").to_dict("records"))
        error = None
    except Exception as e:
        contests, error = [], f"Could not read this file as a DraftKings entries export: {e}"
    return {"path": CURRENT_ENTRIES, "mtime": CURRENT_ENTRIES.stat().st_mtime,
            "contests": contests, "total": sum(c["entries"] for c in contests), "error": error}


def slate_status(session) -> dict:
    players, teams, first, last = session.execute(
        text("SELECT count(*), count(DISTINCT team), min(gametime), max(gametime) FROM nfl_player")).one()
    teams_list = [r[0] for r in session.execute(text("SELECT DISTINCT team FROM nfl_player ORDER BY team"))]
    lineups = session.execute(text("SELECT count(*) FROM lineup")).scalar()
    sims = session.execute(text(
        "SELECT contest_name, count(*) FROM lineup_simulation_results GROUP BY contest_name ORDER BY contest_name")).all()
    owned = session.execute(text("SELECT count(*) FROM nfl_player WHERE ownership_proj IS NOT NULL AND ownership_proj > 0")).scalar()
    return {"players": players, "teams": teams_list, "first_game": first, "last_game": last,
            "lineups": lineups, "sims": [{"contest": c, "rows": n} for c, n in sims], "players_with_ownership": owned}
