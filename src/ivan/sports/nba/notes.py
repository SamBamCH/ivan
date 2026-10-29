"""
ivan.sports.nba.notes

Notes typed next to players on the desk. A note belongs to the slate it was written on (draft group + the slate's
day): it shows while that slate is loaded, including a past day restored from its archive, and on no other slate. It
stands until cleared; clearing keeps the row with cleared_at, so the slate's scorecard lists the notes that stood at
lock (validation.build_scorecard) next to the players' misses.

    set_note(session, dg, day, player, text, team=None, raw=None)   write or replace (a blank text clears)
    clear(session, dg, player)
    active(session, dg)                     {player: row} not cleared
    at(session, dg, when)                   the notes that stood at `when` (written before it, not cleared by then)

Each machine keeps its own (the table is not in the feed): the laptop's notes are the laptop's, the server's desk's
the server's. Terminal: scripts/nba/notes.py.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from ivan.core.utils import normalize_name
from ivan.sports.nba.models import NBAPlayerNote


def _open(session: Session, dg: int, player: str):
    return (session.query(NBAPlayerNote).filter(NBAPlayerNote.draft_group_id == int(dg), NBAPlayerNote.player_name == player,
                                                NBAPlayerNote.cleared_at.is_(None)).order_by(NBAPlayerNote.id.desc()).first())


def set_note(session: Session, dg: int, day: Optional[str], player: str, text: str, team: Optional[str] = None,
             raw: Optional[str] = None, now: Optional[datetime] = None) -> Optional[NBAPlayerNote]:
    """Write the player's note on this slate (replacing the open one's text); a blank text clears it. Returns the row or None."""
    now = now or datetime.now()
    player = normalize_name(player)
    text = (text or "").strip()
    if not text:
        clear(session, dg, player, now=now)
        return None
    row = _open(session, dg, player)
    if row is None:
        row = NBAPlayerNote(draft_group_id=int(dg), slate_day=day, player_name=player, player_raw=raw or player, team=team,
                            text=text, created_at=now, updated_at=now)
        session.add(row)
    else:
        row.text = text; row.updated_at = now
        if team:
            row.team = team
    session.commit()
    return row


def clear(session: Session, dg: int, player: str, now: Optional[datetime] = None) -> int:
    now = now or datetime.now()
    rows = (session.query(NBAPlayerNote).filter(NBAPlayerNote.draft_group_id == int(dg), NBAPlayerNote.player_name == normalize_name(player),
                                                NBAPlayerNote.cleared_at.is_(None)).all())
    for r in rows:
        r.cleared_at = now
    session.commit()
    return len(rows)


def active(session: Session, dg: Optional[int]) -> Dict[str, NBAPlayerNote]:
    if dg is None:
        return {}
    rows = (session.query(NBAPlayerNote).filter(NBAPlayerNote.draft_group_id == int(dg), NBAPlayerNote.cleared_at.is_(None))
            .order_by(NBAPlayerNote.id).all())
    return {r.player_name: r for r in rows}


def at(session: Session, dg: int, when: Optional[datetime]) -> List[NBAPlayerNote]:
    """The notes that stood at `when` (None: every note the slate ever had, cleared or not)."""
    q = session.query(NBAPlayerNote).filter(NBAPlayerNote.draft_group_id == int(dg))
    rows = q.order_by(NBAPlayerNote.id).all()
    if when is None:
        return rows
    return [r for r in rows if r.created_at <= when and (r.cleared_at is None or r.cleared_at > when)]


def as_dict(r: NBAPlayerNote) -> dict:
    return dict(player=r.player_name, player_raw=r.player_raw, team=r.team, text=r.text, day=r.slate_day,
                created_at=r.created_at.isoformat(timespec="minutes"), updated_at=r.updated_at.isoformat(timespec="minutes"),
                cleared_at=r.cleared_at.isoformat(timespec="minutes") if r.cleared_at else None)
