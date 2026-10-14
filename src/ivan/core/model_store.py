# src/ivan/core/model_store.py
"""
Versioned model directories, resolved by the replay clock.

A sport's model directory (data/<sport>/models) holds the live models: the
ones the pre-lock pipeline uses today, trained on everything in the
database. A replay must not use them, because they have seen the slates it
replays. So trainers can also write *as-of* versions:

    data/<sport>/models/asof/<cutoff>/            cutoff = YYYY-MM-DD
        manifest.json                             what was built, from data before the cutoff
        <the same files the live directory has>

and the loaders resolve the directory through `resolve(base, as_of)`:
live (no clock) -> base; replay (clock frozen at lock) -> the newest as-of
directory whose cutoff is on or before the slate's lock date, or base with
`fallback=True` when none exists (the validation report says so).

    from ivan.core import model_store
    d = model_store.resolve(MODELS_DIR, clock.as_of())     # a ModelDir: .path, .cutoff, .fallback
    model_store.write_manifest(d.path, cutoff, built={"dupe_model": {...}})
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

ASOF = "asof"
MANIFEST = "manifest.json"


@dataclass
class ModelDir:
    path: Path
    cutoff: Optional[date] = None        # None: the live directory
    fallback: bool = False               # True: an as-of directory was wanted but none precedes the date
    manifest: Dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.cutoff is None:
            return "live models" + (" (no as-of version precedes the slate)" if self.fallback else "")
        return f"models as of {self.cutoff:%Y-%m-%d}"

    def describe(self) -> dict:
        return {"path": str(self.path), "cutoff": self.cutoff.isoformat() if self.cutoff else None, "fallback": self.fallback,
                "built_at": self.manifest.get("built_at"), "built": sorted((self.manifest.get("built") or {}).keys())}


def _parse_cutoff(name: str) -> Optional[date]:
    try:
        return date.fromisoformat(name)
    except ValueError:
        return None


def versions(base: Path) -> List[ModelDir]:
    """Every as-of directory under `base`, oldest cutoff first."""
    root = Path(base) / ASOF
    out = []
    if root.is_dir():
        for d in root.iterdir():
            c = _parse_cutoff(d.name)
            if d.is_dir() and c is not None:
                out.append(ModelDir(d, c, False, read_manifest(d)))
    return sorted(out, key=lambda m: m.cutoff)


def resolve(base: Path, as_of: Optional[datetime | date]) -> ModelDir:
    """The model directory to use at `as_of` (None = live)."""
    base = Path(base)
    if as_of is None:
        return ModelDir(base, None, False, read_manifest(base))
    day = as_of.date() if isinstance(as_of, datetime) else as_of
    before = [m for m in versions(base) if m.cutoff <= day]
    if before:
        return before[-1]
    return ModelDir(base, None, True, read_manifest(base))


def asof_dir(base: Path, cutoff: date) -> Path:
    return Path(base) / ASOF / cutoff.isoformat()


def read_manifest(path: Path) -> Dict:
    p = Path(path) / MANIFEST
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_manifest(path: Path, cutoff: Optional[date], built: Dict[str, dict], notes: Optional[List[str]] = None) -> Dict:
    """Merge `built` ({builder name: details}) into the directory's manifest."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    m = read_manifest(path)
    m["cutoff"] = cutoff.isoformat() if cutoff else None
    m["built_at"] = datetime.now().isoformat(timespec="seconds")
    m.setdefault("built", {}).update(built)
    if notes:
        m.setdefault("notes", []).extend(notes)
    (path / MANIFEST).write_text(json.dumps(m, indent=1, default=str))
    return m
