# src/ivan/core/slates.py
"""
One status row per slate: what the settle job found and what is still
missing, so the Backtest page and the nightly loop read a table instead of
re-deriving it.

    slate_status(sport, slate_key)  mode, lock, captured_at, played (contest ids we entered),
                                     public (the draft group's largest contests), scraped (ids with
                                     standings), box_scores (bool), our_entries (rows), status, missing,
                                     settled_at, checked_at

status: "open" (lock not passed), "waiting" (box scores or standings pending), "login_needed"
(the scrape could not get in), "settled" (everything there), "no_contests".
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable, List, Optional

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

SlateBase = declarative_base()
STATUSES = ("open", "waiting", "login_needed", "settled", "no_contests")


class SlateStatusRow(SlateBase):
    __tablename__ = "slate_status"
    __table_args__ = (UniqueConstraint("sport", "slate_key", name="uq_slate_status"),)

    id          = Column(Integer, primary_key=True, autoincrement=True)
    sport       = Column(String, nullable=False, index=True)
    slate_key   = Column(String, nullable=False, index=True)
    mode        = Column(String, nullable=True)
    lock        = Column(DateTime, nullable=True, index=True)
    captured_at = Column(DateTime, nullable=True)
    played      = Column(Text, nullable=True)        # json list of contest ids we entered
    public      = Column(Text, nullable=True)        # json list of the draft group's largest contests
    scraped     = Column(Text, nullable=True)        # json list of contest ids with standings
    box_scores  = Column(Boolean, nullable=False, default=False)
    our_entries = Column(Integer, nullable=False, default=0)
    status      = Column(String, nullable=False, default="open")
    missing     = Column(Text, nullable=True)        # json list of strings
    settled_at  = Column(DateTime, nullable=True)
    checked_at  = Column(DateTime, nullable=False, default=datetime.now)

    def as_dict(self) -> dict:
        j = lambda v: json.loads(v) if v else []
        return {"sport": self.sport, "slate_key": self.slate_key, "mode": self.mode, "lock": self.lock, "captured_at": self.captured_at,
                "played": j(self.played), "public": j(self.public), "scraped": j(self.scraped), "box_scores": bool(self.box_scores),
                "our_entries": int(self.our_entries or 0), "status": self.status, "missing": j(self.missing), "settled_at": self.settled_at,
                "checked_at": self.checked_at}


def ensure_tables(engine) -> None:
    SlateBase.metadata.create_all(engine)


def upsert(session, sport: str, slate_key: str, **fields) -> SlateStatusRow:
    row = session.query(SlateStatusRow).filter_by(sport=sport, slate_key=str(slate_key)).first()
    if row is None:
        row = SlateStatusRow(sport=sport, slate_key=str(slate_key))
        session.add(row)
    for k, v in fields.items():
        if k in ("played", "public", "scraped", "missing"):
            v = json.dumps(list(v) if v is not None else [], default=str)
        setattr(row, k, v)
    row.checked_at = datetime.now()
    if row.status == "settled" and row.settled_at is None:
        row.settled_at = datetime.now()
    if row.status != "settled":
        row.settled_at = None
    session.commit()
    return row


def rows(session, sport: str, since: Optional[datetime] = None, until: Optional[datetime] = None) -> List[SlateStatusRow]:
    q = session.query(SlateStatusRow).filter(SlateStatusRow.sport == sport)
    if since is not None:
        q = q.filter(SlateStatusRow.lock >= since)
    if until is not None:
        q = q.filter(SlateStatusRow.lock <= until)
    return q.order_by(SlateStatusRow.lock.asc()).all()


def by_key(session, sport: str, keys: Iterable[str]) -> dict:
    keys = [str(k) for k in keys]
    if not keys:
        return {}
    return {r.slate_key: r for r in session.query(SlateStatusRow).filter(SlateStatusRow.sport == sport, SlateStatusRow.slate_key.in_(keys)).all()}
