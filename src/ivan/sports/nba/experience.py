"""
ivan.sports.nba.experience

Each player's first NBA season, from nba_api's all-players list (CommonAllPlayers, current season: FROM_YEAR per
player). One request; the nightly fetches it and the desk reads the file, so a rookie is marked the same on both
desks (data/nba/derived travels with the nightly feed).

    fetch(season=None)       request the list, keep it raw (nba_api/allplayers), write EXPERIENCE_JSON
    load()                   the file's first seasons and teams by loose name (empty when it has never been fetched)
    is_rookie(name, year, exp, played_before, team)
                             True when the list's first season is `year` (matched by the loose name, then by last name on
                             the same team); a name the list lacks falls back to `played_before` (no NBA game on file
                             before this season = rookie)

The box scores alone cannot tell: they reach back to October 2025, so a veteran who missed last season reads as new.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, Optional

from ivan.core.utils import normalize_name
from ivan.sports.nba.paths import NBA_DATA_DIR

EXPERIENCE_JSON = NBA_DATA_DIR / "derived" / "players" / "experience.json"


def season_start_year(today=None) -> int:
    """The year the current season started: October onwards is the new season."""
    today = today or datetime.now().date()
    return today.year if today.month >= 10 else today.year - 1


def fetch(season: Optional[str] = None, log=print) -> int:
    """Fetch the current season's players with their first season; returns how many were written."""
    from nba_api.stats.endpoints import commonallplayers
    from ivan.core import raw
    y = season_start_year()
    season = season or f"{y}-{str(y + 1)[2:]}"
    df = commonallplayers.CommonAllPlayers(is_only_current_season=1, season=season, timeout=30).get_data_frames()[0]
    rows = df[["PERSON_ID", "DISPLAY_FIRST_LAST", "FROM_YEAR", "TO_YEAR", "TEAM_ABBREVIATION"]].to_dict("records")
    try:
        raw.store("nba", "nba_api/allplayers", season, rows)
    except Exception as e:   # noqa: BLE001 - the raw copy is a convenience
        log(f"raw store failed: {type(e).__name__}: {e}")
    first, teams = {}, {}
    for r in rows:
        try:
            k = loose(str(r["DISPLAY_FIRST_LAST"]))
            first[k] = int(r["FROM_YEAR"]); teams[k] = str(r.get("TEAM_ABBREVIATION") or "")
        except (TypeError, ValueError):
            continue
    EXPERIENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    EXPERIENCE_JSON.write_text(json.dumps(dict(season=season, fetched_at=datetime.now().isoformat(timespec="seconds"), first_year=first, team=teams),
                                          indent=0, sort_keys=True))
    log(f"experience: {len(first)} players, {sum(1 for v in first.values() if v == y)} rookies ({season})")
    return len(first)


SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def loose(name: str) -> str:
    """The key both spellings meet on: 'Robert Williams III' = 'robert williams', 'David Jones Garcia' = 'david jonesgarcia'."""
    n = SUFFIX.sub(" ", normalize_name(name).replace(".", " "))
    return re.sub(r"[^a-z]", "", n)


def last_name(name: str) -> str:
    parts = SUFFIX.sub(" ", normalize_name(name).replace(".", " ")).split()
    return re.sub(r"[^a-z]", "", parts[-1]) if parts else ""


def load() -> dict:
    """{first_year: {loose name: year}, team: {loose name: team}} from the file (empty when it was never fetched)."""
    try:
        d = json.loads(EXPERIENCE_JSON.read_text())
        return dict(first_year={str(k): int(v) for k, v in d.get("first_year", {}).items()}, team={str(k): str(v) for k, v in d.get("team", {}).items()})
    except (OSError, ValueError):
        return dict(first_year={}, team={})


def first_year(name: str, team: Optional[str], exp: dict) -> Optional[int]:
    """His first season by the loose name, else the one player on his team with his last name ('EJ' = 'Elijah' Harkless)."""
    fy = exp.get("first_year", {})
    k = loose(name)
    if k in fy:
        return fy[k]
    if team:
        ln = last_name(name)
        hits = [v for kk, v in fy.items() if exp.get("team", {}).get(kk) == team and kk.endswith(ln)]
        if len(hits) == 1:
            return hits[0]
    return None


def is_rookie(name: str, year: int, exp: dict, played_before: bool, team: Optional[str] = None) -> bool:
    fy = first_year(name, team, exp)
    if fy is not None:
        return fy >= year
    return not played_before
