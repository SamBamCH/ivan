"""
ivan.sports.nba.scenarios

A scenario is a team's edits saved under a name with the assumption they were built on: "Giannis out" holds a scratch
pin on him and the grid painted around his absence; "Giannis in" holds his row at 32 minutes. Each is a copy of the two
things a team's edits live in (its rows of data/nba/input/minutes_overrides.csv and data/nba/input/rotations/<TEAM>.json)
plus `assumes`: {player: out | in | starts | bench}, read off the calls in the edits when it is saved. Activating one puts
its pins and grid back live for that team only; the desk re-projects the team.

The assumptions are what makes a scenario pick itself: when a refresh or a late-news call changes the facts (DK statuses,
Rotowire starters, the call you typed), the scenario whose assumptions all hold is activated for the team (one match,
not already active). Two matches or none leave the desk as it is and say so.

Files: data/nba/input/scenarios/<TEAM>/<slug>.json, and _active.json beside them naming the active slug.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ivan.sports.nba.paths import INPUT_DIR
from ivan.sports.nba.projection.rotation import ROTATIONS_DIR, drawn_path

SCENARIOS_DIR = INPUT_DIR / "scenarios"
CALLS = ("out", "in", "starts", "bench")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:40] or "scenario"


def team_dir(team: str) -> Path:
    return SCENARIOS_DIR / team


def _active_path(team: str) -> Path:
    return team_dir(team) / "_active.json"


def active_slug(team: str) -> Optional[str]:
    p = _active_path(team)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("slug")
    except (OSError, ValueError):
        return None


def set_active(team: str, slug: Optional[str]) -> None:
    p = _active_path(team); p.parent.mkdir(parents=True, exist_ok=True)
    if slug is None:
        p.unlink(missing_ok=True)
    else:
        p.write_text(json.dumps({"slug": slug, "since": datetime.now().isoformat(timespec="seconds")}))


def list_scenarios(team: str) -> List[dict]:
    d = team_dir(team)
    if not d.is_dir():
        return []
    out = []
    act = active_slug(team)
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            s = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        s["slug"] = f.stem; s["active"] = f.stem == act
        out.append(s)
    out.sort(key=lambda s: s.get("saved_at") or "")
    return out


def load_scenario(team: str, slug: str) -> Optional[dict]:
    p = team_dir(team) / f"{slug}.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text()); s["slug"] = slug; s["active"] = slug == active_slug(team)
    return s


def delete_scenario(team: str, slug: str) -> bool:
    p = team_dir(team) / f"{slug}.json"
    if not p.exists():
        return False
    p.unlink()
    if active_slug(team) == slug:
        set_active(team, None)
    return True


def assumptions(team_overrides: Dict[str, dict], statuses: Dict[str, str], questionable: Iterable[str], grid: Optional[dict]) -> Dict[str, str]:
    """
    What the edits assume, read off the calls: a scratch pin on a player DK does not list as OUT = out; a starter call =
    starts / bench; a questionable player with minutes pinned above zero or a locked row with minutes = in. Players DK
    already rules out carry no assumption (that is a fact, not a guess).
    """
    out: Dict[str, str] = {}
    q = set(questionable)
    locked_minutes = {}
    if grid:
        for n, row in zip(grid.get("players", []), grid.get("grid", [])):
            if n in (grid.get("locked") or []):
                locked_minutes[n] = int(sum(row))
    for n, ov in team_overrides.items():
        st = (statuses.get(n) or "").upper()
        if ov.get("minutes") is not None and float(ov["minutes"]) <= 0 and st != "OUT":
            out[n] = "out"
        elif ov.get("starter") is not None:
            out[n] = "starts" if float(ov["starter"]) > 0 else "bench"
        elif n in q and ov.get("minutes") is not None and float(ov["minutes"]) > 0:
            out[n] = "in"
    for n, m in locked_minutes.items():
        if n in q and n not in out and m > 0:
            out[n] = "in"
        elif n in q and n not in out and m == 0:
            out[n] = "out"
    return out


def holds(assumes: Dict[str, str], facts: dict) -> bool:
    """Every assumption against the facts: {out: set, available: set, starters: set} (DK statuses, Rotowire, a typed call)."""
    for n, call in assumes.items():
        if call == "out" and n not in facts.get("out", set()):
            return False
        if call == "in" and n not in facts.get("available", set()):
            return False
        if call == "starts" and n not in facts.get("starters", set()):
            return False
        if call == "bench" and (n not in facts.get("available", set()) or n in facts.get("starters", set())):
            return False
    return True


def matching(team: str, facts: dict) -> List[dict]:
    """The team's scenarios whose assumptions all hold (a scenario without assumptions never matches by itself)."""
    return [s for s in list_scenarios(team) if s.get("assumes") and holds(s["assumes"], facts)]


def save_scenario(team: str, name: str, team_overrides: Dict[str, dict], assumes: Dict[str, str], note: str = "") -> dict:
    """Copy the team's live edits (its override rows and its drawn grid file) under the name; the copy becomes the active one."""
    d = team_dir(team); d.mkdir(parents=True, exist_ok=True)
    slug = slugify(name)
    grid = None
    gp = drawn_path(team)
    if gp.exists():
        try:
            grid = json.loads(gp.read_text())
        except (OSError, ValueError):
            grid = None
    s = dict(team=team, name=name, saved_at=datetime.now().isoformat(timespec="seconds"), note=note, assumes=dict(assumes),
             overrides={n: {k: v for k, v in ov.items() if v is not None} for n, ov in team_overrides.items()}, grid=grid)
    (d / f"{slug}.json").write_text(json.dumps(s, indent=0))
    set_active(team, slug)
    s["slug"] = slug; s["active"] = True
    return s


def activate(team: str, slug: str, overrides_all: Dict[str, dict], team_players: Iterable[str]) -> Optional[Dict[str, dict]]:
    """
    Put the scenario live: the team's players' override rows become the scenario's (other teams untouched) and the drawn
    grid file is replaced (removed when the scenario has none). Returns the new overrides dict for the caller to write,
    or None when the scenario does not exist.
    """
    s = load_scenario(team, slug)
    if s is None:
        return None
    players = set(team_players)
    new = {n: ov for n, ov in overrides_all.items() if n not in players}
    for n, ov in (s.get("overrides") or {}).items():
        new[n] = dict(ov)
    gp = drawn_path(team)
    if s.get("grid"):
        ROTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        gp.write_text(json.dumps(s["grid"], indent=0))
    else:
        gp.unlink(missing_ok=True)
    set_active(team, slug)
    return new
