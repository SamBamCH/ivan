"""
The projection store: the loaded slate's projection, one record per team, on disk.

Whatever projects a team (ingest.project_team: a refresh, a publish, an edit on the desk, scripts/nba/project.py, a
consumer re-projecting its own edits) writes the team's rows to nba_player AND its record here. The desk only reads:
its frames, the grid behind them and the history-only comparison come from these files, so a page render never
re-projects and a desk picking up a pool another process wrote (the capture tick, a feed pull) loads it in a few
milliseconds. The records travel with the live feed bundle (ivan.core.feed, live_mirror), so a consumer shows the
producer's projection exactly, re-projecting only the teams it edited.

A record (pickle, data/nba/derived/projection/<TEAM>.pkl):
    team, draft_group, written_at, n_sims
    frame   the team's projection (projection.engine.project_team's frame after the fpts overrides)
    hist    the same roster projected from history alone (n_sims=0, market_weight=0): the "vs history" column and flags
    rot     the rotation inputs and the grid behind the frame (LiveSlate.rotation_for + project_team's grid, targets, zero_minutes)
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ivan.sports.nba.paths import NBA_DATA_DIR

STORE_DIR = NBA_DATA_DIR / "derived" / "projection"


def path(team: str) -> Path:
    return STORE_DIR / f"{team}.pkl"


def record(team: str, frame: pd.DataFrame, hist: pd.DataFrame, rot: dict, draft_group: Optional[int], n_sims: int) -> dict:
    return dict(team=str(team), draft_group=None if draft_group is None else int(draft_group), written_at=datetime.now(), n_sims=int(n_sims),
                frame=frame, hist=hist, rot=dict(rot or {}))


def save(rec: dict) -> Path:
    """Write the record atomically (a reader never sees a half-written file)."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    p = path(rec["team"]); tmp = p.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)
    return p


def load(team: str) -> Optional[dict]:
    p = path(team)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            rec = pickle.load(f)
    except Exception:   # noqa: BLE001 - a record from another version of the code: project the team again
        return None
    return rec if isinstance(rec, dict) and "frame" in rec else None


def load_all(draft_group: Optional[int] = None) -> dict[str, dict]:
    """{team: record} of every record on disk, only the slate's when draft_group is given (another slate's are ignored, not deleted)."""
    out = {}
    if not STORE_DIR.is_dir():
        return out
    for p in sorted(STORE_DIR.glob("*.pkl")):
        rec = load(p.stem)
        if rec is None:
            continue
        if draft_group is not None and rec.get("draft_group") is not None and int(rec["draft_group"]) != int(draft_group):
            continue
        out[rec["team"]] = rec
    return out


def prune(draft_group: Optional[int], teams: Iterable[str]) -> int:
    """Remove the records that are not the slate's (another draft group, or a team no longer on it). Returns how many went."""
    keep = {str(t) for t in teams}
    n = 0
    if not STORE_DIR.is_dir():
        return 0
    for p in STORE_DIR.glob("*.pkl"):
        rec = load(p.stem)
        stale = rec is None or p.stem not in keep or (draft_group is not None and rec.get("draft_group") is not None
                                                      and int(rec["draft_group"]) != int(draft_group))
        if stale:
            p.unlink(missing_ok=True); n += 1
    return n


def clear() -> int:
    n = 0
    if STORE_DIR.is_dir():
        for p in STORE_DIR.glob("*.pkl"):
            p.unlink(missing_ok=True); n += 1
    return n


def stamp() -> tuple:
    """(newest record's mtime, record count): changes whenever any process writes a team."""
    if not STORE_DIR.is_dir():
        return (None, 0)
    ps = list(STORE_DIR.glob("*.pkl"))
    return (max((p.stat().st_mtime for p in ps), default=None), len(ps))
