"""
The projection store: the loaded slate's projection, one record per team, on disk.

Whatever projects a team (ingest.project_team: a refresh, a publish, an edit on the desk, scripts/nba/project.py, a
consumer re-projecting its own edits) writes the team's rows to nba_player AND its record here. The desk only reads:
its frames, the grid behind them and the history-only comparison come from these files, so a page render never
re-projects and a desk picking up a pool another process wrote (the capture tick, a feed pull) loads it in a few
milliseconds. The records travel with the live feed bundle (ivan.core.feed, live_mirror), so a consumer shows the
producer's projection exactly, re-projecting only the teams it edited.

A record (pickle, slates/<dg>/projection/<TEAM>.pkl):
    team, draft_group, written_at, n_worlds
    frame     layer 1, the mapping (projection.engine.project_team): minutes, rates, proj_<stat>, p_out, role, the plan's fpts_mean
    hist      the same roster mapped from history alone (market_weight=0): the "vs history" column and flags
    rot       the rotation inputs and the grid behind the frame (LiveSlate.rotation_for + project_team's grid, targets, zero_minutes)
    sim       layer 2, the World's draw per player (projection.worlds.simulate_game): fpts_mean, fpts_std, quantiles, minutes_sim,
              minutes_p10/p90, sim_<stat>, stat_dist ({stat: the share of the worlds at each whole value}: the stat hover's
              chart); fpts_model beside fpts_mean where a hand fpts pin moved it. None until the game is drawn
    samples   {names, pids, fpts (n, k), minutes (n, k)} float32: the per-world draws the optimal-lineup share is solved on
    game      the game's draw: the attempt factor per side, the lines, the drawn total and margin, overtime and garbage shares
`merged(rec)` is the frame with the sim's columns laid over it: what the capture rows, the CSV and the desk's rows read.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ivan.sports.nba import slates as S



def path(team: str) -> Path:
    return S.store_dir() / f"{team}.pkl"


SIM_COLS = ("fpts_mean", "fpts_std", "quantiles", "fpts_model", "minutes_sim", "minutes_p10", "minutes_p90", "p_zero",
            "sim_points", "sim_rebounds", "sim_assists", "sim_fg3m", "sim_steals", "sim_blocks", "sim_turnovers", "stat_dist")


def record(team: str, frame: pd.DataFrame, hist: pd.DataFrame, rot: dict, draft_group: Optional[int], n_worlds: int = 0,
           sim: Optional[pd.DataFrame] = None, samples: Optional[dict] = None, game: Optional[dict] = None) -> dict:
    return dict(team=str(team), draft_group=None if draft_group is None else int(draft_group), written_at=datetime.now(), n_worlds=int(n_worlds),
                frame=frame, hist=hist, rot=dict(rot or {}), sim=sim, samples=samples, game=dict(game or {}))


def set_sim(rec: dict, sim: pd.DataFrame, samples: dict, game: dict, n_worlds: int) -> dict:
    """The record with a fresh layer 2 (the game was drawn again); written_at moves."""
    rec.update(sim=sim, samples=samples, game=dict(game or {}), n_worlds=int(n_worlds), written_at=datetime.now())
    return rec


def merged(rec: dict) -> pd.DataFrame:
    """The frame with the World's columns laid over it (fpts_mean, fpts_std, the quantiles, the minutes band, the stat means);
    a player the draw did not hold keeps the plan's fpts_mean and gets no sd. The frame alone when the game is not drawn yet."""
    frame = rec["frame"].copy()
    sim = rec.get("sim")
    if sim is None or not len(sim):
        for c in ("fpts_std", "minutes_p10", "minutes_p90"):
            if c not in frame.columns:
                frame[c] = float("nan")
        if "quantiles" not in frame.columns:
            frame["quantiles"] = None
        return frame
    for c in SIM_COLS:
        if c in sim.columns:
            col = sim[c].reindex(frame.index)
            if c == "fpts_mean":
                frame[c] = col.where(col.notna(), frame[c].astype(float))
            else:
                frame[c] = col if c != "quantiles" else col.where(col.notna(), None)
    return frame


def save(rec: dict) -> Path:
    """Write the record atomically (a reader never sees a half-written file)."""
    S.store_dir().mkdir(parents=True, exist_ok=True)
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
    if not S.store_dir().is_dir():
        return out
    for p in sorted(S.store_dir().glob("*.pkl")):
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
    if not S.store_dir().is_dir():
        return 0
    for p in S.store_dir().glob("*.pkl"):
        rec = load(p.stem)
        stale = rec is None or p.stem not in keep or (draft_group is not None and rec.get("draft_group") is not None
                                                      and int(rec["draft_group"]) != int(draft_group))
        if stale:
            p.unlink(missing_ok=True); n += 1
    return n


def clear() -> int:
    n = 0
    if S.store_dir().is_dir():
        for p in S.store_dir().glob("*.pkl"):
            p.unlink(missing_ok=True); n += 1
    return n


def stamp() -> tuple:
    """(newest record's mtime, record count): changes whenever any process writes a team."""
    if not S.store_dir().is_dir():
        return (None, 0)
    ps = list(S.store_dir().glob("*.pkl"))
    return (max((p.stat().st_mtime for p in ps), default=None), len(ps))
