# src/ivan/core/raw.py
"""
Raw artifacts: every payload a fetcher receives, kept exactly as received.

    data/<sport>/raw/<kind>/<key>/<stamp>.<ext>.gz      the payload, gzip
    data/<sport>/raw/index.jsonl                         one line per artifact (append-only)
    raw_artifact (table)                                 the same index in the sport's database, for queries

Kinds (nfl): dk/draftables (key: draft group), dk/lobby (key: date),
dk/payouts (key: contest id), dk/standings (key: contest id),
odds/events, odds/props (key: event id), odds/lines, nflverse/stats
(key: season). A fetcher calls store() before anything parses the payload,
so a parser change later is a re-run over the files, not a re-fetch.

    art = raw.store("nfl", "dk/draftables", draft_group_id, payload)        # dict/list -> json, str/bytes -> as is
    raw.load(art.path)                                                       # the payload back (json decoded when .json)
    raw.find("nfl", "dk/standings", key=184253394)                           # artifacts, newest last
    raw.sync_index(session, "nfl")                                           # index.jsonl -> raw_artifact table

Nothing here deletes or rewrites a file: an artifact with the same kind,
key and stamp is left alone.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

from ivan.core.utils import PROJECT_ROOT

RawBase = declarative_base()
INDEX = "index.jsonl"


class RawArtifactRow(RawBase):
    __tablename__ = "raw_artifact"
    __table_args__ = (UniqueConstraint("sport", "path", name="uq_raw_artifact_path"),)

    id         = Column(Integer, primary_key=True, autoincrement=True)
    sport      = Column(String, nullable=False, index=True)
    kind       = Column(String, nullable=False, index=True)
    key        = Column(String, nullable=False, index=True)
    stamp      = Column(String, nullable=False)
    path       = Column(String, nullable=False)          # relative to the sport's raw dir
    bytes      = Column(Integer, nullable=False)         # compressed size
    sha256     = Column(String, nullable=False)          # of the payload, uncompressed
    meta       = Column(Text, nullable=True)             # json
    created_at = Column(DateTime, nullable=False, default=datetime.now)


def ensure_tables(engine) -> None:
    RawBase.metadata.create_all(engine)


@dataclass
class RawArtifact:
    sport: str
    kind: str
    key: str
    stamp: str
    path: str                 # relative to raw_dir(sport)
    bytes: int
    sha256: str
    meta: dict
    created_at: str

    @property
    def abspath(self) -> Path:
        return raw_dir(self.sport) / self.path


def raw_dir(sport: str) -> Path:
    return PROJECT_ROOT / "data" / sport / "raw"


def _stamp(when: Optional[datetime] = None) -> str:
    return (when or datetime.now()).strftime("%Y%m%d-%H%M%S")


def _encode(payload: Any, ext: Optional[str]) -> tuple[bytes, str]:
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload), ext or "bin"
    if isinstance(payload, str):
        return payload.encode("utf-8"), ext or "txt"
    return json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8"), ext or "json"


def store(sport: str, kind: str, key: Any, payload: Any, *, stamp: Optional[str] = None, meta: Optional[dict] = None,
          ext: Optional[str] = None, when: Optional[datetime] = None) -> RawArtifact:
    """Write one payload (dict/list as json; str/bytes as is) under kind/key/stamp and append it to the index."""
    data, ext = _encode(payload, ext)
    stamp = stamp or _stamp(when)
    rel = Path(kind) / str(key) / f"{stamp}.{ext}.gz"
    path = raw_dir(sport) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():                                   # never rewrite an artifact
        with gzip.open(path, "wb", compresslevel=6) as fh:
            fh.write(data)
    art = RawArtifact(sport=sport, kind=kind, key=str(key), stamp=stamp, path=str(rel).replace("\\", "/"), bytes=path.stat().st_size,
                      sha256=hashlib.sha256(data).hexdigest(), meta=dict(meta or {}), created_at=datetime.now().isoformat(timespec="seconds"))
    with open(raw_dir(sport) / INDEX, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(art), default=str) + "\n")
    return art


def load(path: Path | str, sport: Optional[str] = None) -> Any:
    """The payload of an artifact: json decoded for .json files, text for .csv/.txt, bytes otherwise."""
    p = Path(path)
    if not p.is_absolute():
        if sport is None:
            raise ValueError("a relative artifact path needs the sport")
        p = raw_dir(sport) / p
    with gzip.open(p, "rb") as fh:
        data = fh.read()
    name = p.name[:-3] if p.name.endswith(".gz") else p.name
    if name.endswith(".json"):
        return json.loads(data.decode("utf-8"))
    if name.endswith((".csv", ".txt")):
        return data.decode("utf-8")
    return data


def index(sport: str) -> List[RawArtifact]:
    """Every artifact in the sport's index file, in the order written."""
    p = raw_dir(sport) / INDEX
    out = []
    if p.is_file():
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(RawArtifact(**json.loads(line)))
                    except (TypeError, ValueError):
                        continue
    return out


def find(sport: str, kind: str, key: Any = None, since: Optional[str] = None, until: Optional[str] = None) -> List[RawArtifact]:
    """Artifacts of a kind (and key), newest last; since/until bound the stamp (YYYYMMDD-HHMMSS, prefix compare)."""
    out = [a for a in index(sport) if a.kind == kind and (key is None or a.key == str(key))]
    if since:
        out = [a for a in out if a.stamp >= since]
    if until:
        out = [a for a in out if a.stamp <= until]
    return sorted(out, key=lambda a: a.stamp)


def latest(sport: str, kind: str, key: Any = None) -> Optional[RawArtifact]:
    arts = find(sport, kind, key)
    return arts[-1] if arts else None


def sync_index(session, sport: str) -> int:
    """Copy index lines the database does not have yet into raw_artifact; returns rows added."""
    ensure_tables(session.get_bind())
    have = {p for (p,) in session.query(RawArtifactRow.path).filter(RawArtifactRow.sport == sport).all()}
    added = 0
    for a in index(sport):
        if a.path in have:
            continue
        session.add(RawArtifactRow(sport=a.sport, kind=a.kind, key=a.key, stamp=a.stamp, path=a.path, bytes=int(a.bytes), sha256=a.sha256,
                                   meta=json.dumps(a.meta, default=str) if a.meta else None,
                                   created_at=datetime.fromisoformat(a.created_at) if a.created_at else datetime.now()))
        have.add(a.path)
        added += 1
    if added:
        session.commit()
    return added


def summary(sport: str) -> dict:
    """Counts and bytes per kind, for a status page."""
    out: dict = {}
    for a in index(sport):
        d = out.setdefault(a.kind, {"n": 0, "bytes": 0, "keys": set(), "first": a.stamp, "last": a.stamp})
        d["n"] += 1; d["bytes"] += int(a.bytes); d["keys"].add(a.key)
        d["first"] = min(d["first"], a.stamp); d["last"] = max(d["last"], a.stamp)
    return {k: {**v, "keys": len(v["keys"])} for k, v in out.items()}
