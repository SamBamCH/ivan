# src/ivan/core/archive.py
"""
The slate archive: every raw pre-lock input a slate was ingested with, frozen
as it was fetched, so the slate can be replayed later with whatever code is
current. Sport-agnostic; lives in each sport's database.

    archived_slate   one row per capture: sport, slate key (the DraftKings
                     draft group), lock time, mode, when it was captured
    archived_blob    the rows of one kind for a capture, as JSON:
                     players (the DK pool), odds (player props), game_lines

Ingest adds a capture every time it runs, so "the last capture before lock"
is what the slate was really played with; a backfill adds captures marked
source="backfill". Nothing here is ever modelled: projections, ownership and
lineups are recomputed by the replay.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Session, declarative_base

ArchiveBase = declarative_base()

KINDS = ("players", "odds", "game_lines")


class ArchivedSlate(ArchiveBase):
    __tablename__ = "archived_slate"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    sport       = Column(String, nullable=False, index=True)
    slate_key   = Column(String, nullable=False, index=True)      # DK draft group id, as text
    lock        = Column(DateTime, nullable=False, index=True)    # local naive, like player gametimes
    mode        = Column(String, nullable=False)                  # showdown | classic | ...
    source      = Column(String, nullable=False, default="ingest")   # ingest | backfill
    captured_at = Column(DateTime, nullable=False, default=datetime.now)
    n_players   = Column(Integer, nullable=False, default=0)
    n_odds      = Column(Integer, nullable=False, default=0)
    n_lines     = Column(Integer, nullable=False, default=0)
    note        = Column(Text, nullable=True)

    def __repr__(self):
        return (f"<ArchivedSlate {self.sport} {self.slate_key} lock={self.lock:%Y-%m-%d %H:%M} {self.mode} "
                f"{self.source} players={self.n_players} odds={self.n_odds}>")


class ArchivedBlob(ArchiveBase):
    __tablename__ = "archived_blob"

    id       = Column(Integer, primary_key=True, autoincrement=True)
    slate_id = Column(Integer, ForeignKey("archived_slate.id"), nullable=False, index=True)
    kind     = Column(String, nullable=False)
    rows     = Column(JSON, nullable=False)


def ensure_tables(engine) -> None:
    ArchiveBase.metadata.create_all(engine)


def _jsonable(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "item"):                      # numpy scalars
        return _jsonable(v.item())
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def rows_from_objects(objs: Iterable, fields: Iterable[str]) -> List[dict]:
    """[{field: value}] from ORM rows or plain objects, JSON-safe."""
    fields = list(fields)
    return [{f: _jsonable(getattr(o, f, None)) for f in fields} for o in objs]


def archive_slate(session: Session, *, sport: str, slate_key: str, lock: datetime, mode: str,
                  players: List[dict], odds: Optional[List[dict]] = None, game_lines: Optional[List[dict]] = None,
                  source: str = "ingest", note: str = "", captured_at: Optional[datetime] = None) -> ArchivedSlate:
    """Store one capture of a slate's raw inputs. Returns the new archived_slate row."""
    odds = odds or []
    game_lines = game_lines or []
    row = ArchivedSlate(sport=sport, slate_key=str(slate_key), lock=lock, mode=mode, source=source,
                        captured_at=captured_at or datetime.now(), n_players=len(players), n_odds=len(odds),
                        n_lines=len(game_lines), note=note or None)
    session.add(row)
    session.flush()
    for kind, rows in (("players", players), ("odds", odds), ("game_lines", game_lines)):
        session.add(ArchivedBlob(slate_id=row.id, kind=kind, rows=[_jsonable(r) for r in rows]))
    session.commit()
    return row


def captures(session: Session, sport: str, since: Optional[datetime] = None, until: Optional[datetime] = None,
             source: Optional[str] = None) -> List[ArchivedSlate]:
    q = session.query(ArchivedSlate).filter(ArchivedSlate.sport == sport)
    if since is not None:
        q = q.filter(ArchivedSlate.lock >= since)
    if until is not None:
        q = q.filter(ArchivedSlate.lock <= until)
    if source:
        q = q.filter(ArchivedSlate.source == source)
    return q.order_by(ArchivedSlate.lock.asc(), ArchivedSlate.captured_at.asc()).all()


def slates(session: Session, sport: str, since: Optional[datetime] = None, until: Optional[datetime] = None,
           prefer: str = "ingest") -> List[ArchivedSlate]:
    """
    One capture per slate key in the window: the last one taken before lock
    (else the last one at all). Captures from `prefer` win over other sources
    when both exist for a slate.
    """
    by_key: Dict[str, List[ArchivedSlate]] = {}
    for c in captures(session, sport, since, until):
        by_key.setdefault(c.slate_key, []).append(c)
    out = []
    for key, cs in by_key.items():
        pool = [c for c in cs if c.source == prefer] or cs
        pre = [c for c in pool if c.captured_at <= c.lock]
        out.append(max(pre or pool, key=lambda c: c.captured_at))
    return sorted(out, key=lambda c: c.lock)


def load_blobs(session: Session, slate_id: int) -> Dict[str, List[dict]]:
    out = {k: [] for k in KINDS}
    for b in session.query(ArchivedBlob).filter(ArchivedBlob.slate_id == slate_id).all():
        out[b.kind] = list(b.rows or [])
    return out


def delete_captures(session: Session, sport: str, slate_key: str, source: Optional[str] = None) -> int:
    q = session.query(ArchivedSlate).filter(ArchivedSlate.sport == sport, ArchivedSlate.slate_key == str(slate_key))
    if source:
        q = q.filter(ArchivedSlate.source == source)
    ids = [c.id for c in q.all()]
    if ids:
        session.query(ArchivedBlob).filter(ArchivedBlob.slate_id.in_(ids)).delete(synchronize_session=False)
        session.query(ArchivedSlate).filter(ArchivedSlate.id.in_(ids)).delete(synchronize_session=False)
        session.commit()
    return len(ids)
