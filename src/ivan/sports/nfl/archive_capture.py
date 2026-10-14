# src/ivan/sports/nfl/archive_capture.py
"""
What the ingest (ivan.sports.nfl.ingest.store_slate) archives after each run: the raw pool, props and
lines of the capture, keyed by the slate's draft group (see ivan.core.archive).
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

from ivan.core import clock
from ivan.core.archive import ArchivedSlate, archive_slate, captures, rows_from_objects
from ivan.sports.nfl.models import NFLPlayer

POOL_FIELDS = ("id", "roster_slot_id", "name", "team", "opponent", "position", "gametime", "salary", "status")


def _pool_rows(players: Optional[Iterable], session) -> list[dict]:
    if players is not None:
        return [{"id": p.dk_id, "roster_slot_id": p.roster_slot_id, "name": p.name, "team": p.team, "opponent": p.opponent,
                 "position": p.position, "gametime": p.gametime, "salary": p.salary, "status": p.status} for p in players]
    return rows_from_objects(session.query(NFLPlayer).all(), POOL_FIELDS)


def _naive(dt) -> Optional[datetime]:
    if dt is None or dt == "":
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt


def archive_ingest_capture(session, players: Optional[Iterable] = None, odds_records: Optional[list] = None,
                           game_lines: Optional[pd.DataFrame] = None) -> Optional[ArchivedSlate]:
    """
    Archive one ingest capture. `players` are the Player objects the DK fetch
    returned (None on an odds-only refresh, when the pool is read back from
    nfl_player); `odds_records` the parsed prop rows; `game_lines` the parsed
    lines frame. Returns the capture, or None when there is no pool.
    """
    pool = _pool_rows(players, session)
    pool = [{**r, "gametime": _naive(r.get("gametime"))} for r in pool]
    if not pool:
        return None
    lock = min(r["gametime"] for r in pool if r["gametime"])
    slate_key = ""
    if players is not None:
        slate_key = str(getattr(next(iter(players)), "draft_group_id", "") or "")
    if not slate_key:   # odds-only refresh: the same slate as the latest capture of a slate locking at this time
        prev = [c for c in captures(session, "nfl", since=lock.replace(hour=0, minute=0, second=0))
                if abs((c.lock - lock).total_seconds()) < 3600]
        slate_key = prev[-1].slate_key if prev else f"lock-{lock:%Y%m%d-%H%M}"
    mode = "showdown" if any(int(r["roster_slot_id"] or 0) == 511 for r in pool) else "classic"
    lines = game_lines.to_dict(orient="records") if game_lines is not None and len(game_lines) else []
    return archive_slate(session, sport="nfl", slate_key=slate_key, lock=lock, mode=mode, players=pool,
                         odds=list(odds_records or []), game_lines=lines, source="ingest",
                         note="players refetched" if players is not None else "odds refresh",
                         captured_at=clock.now())
