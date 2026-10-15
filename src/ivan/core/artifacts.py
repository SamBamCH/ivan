# src/ivan/core/artifacts.py
"""
Where a pipeline run leaves the things the validation layer reads back.

Every stage that makes a prediction writes it down at the time it is made:
the projection step writes the player projections, the simulator writes
what it assumed about each contest's field and what it expected of every
candidate lineup, the chooser writes its picks (chosen_entries.* files).
After the games are played, ivan.core.validation lines those files up with
what actually happened. Nothing is recomputed from the database, because
the working tables are cleared every week and the models drift.

Layout. Live runs (IVAN_RUN_DIR unset) write stamped files under the
sport's output directory:

    <output>/projections/projections.<stamp>.csv
    <output>/sim/<stamp>/contest_<id>.npz + contest_<id>.json
    <output>/chosen_entries.*.<stamp>.*

A replay sets IVAN_RUN_DIR to the slate (projection) or repeat (sim,
choose) directory, and everything lands there unstamped:

    <run dir>/projections.csv
    <run dir>/sim/contest_<id>.npz + .json

A file's "when" is its stamp, or, inside a replayed slate, one minute before
the slate's lock (the replay stands in for a run just before lock), so the
same "last run before lock" rule pairs live and replayed files alike.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

ENV = "IVAN_RUN_DIR"
STAMP_FMT = "%Y%m%d-%H%M%S"
REPLAY_MANIFEST = "replay.json"


def run_dir() -> Optional[Path]:
    """The replay's run directory, or None when running live."""
    raw = os.environ.get(ENV)
    return Path(raw) if raw else None


def stamp(when: Optional[datetime] = None) -> str:
    return (when or datetime.now()).strftime(STAMP_FMT)


def projections_path(output_dir: Path, when: Optional[datetime] = None) -> Path:
    rd = run_dir()
    if rd is not None:
        return rd / "projections.csv"
    return Path(output_dir) / "projections" / f"projections.{stamp(when)}.csv"


def sim_dir(output_dir: Path, when: Optional[datetime] = None) -> Path:
    rd = run_dir()
    if rd is not None:
        return rd / "sim"
    return Path(output_dir) / "sim" / stamp(when)


def field_dir(output_dir: Path, when: Optional[datetime] = None) -> Path:
    """Where the field stage writes its sampled fields (the sim artifact's layout, minus the outcomes)."""
    rd = run_dir()
    if rd is not None:
        return rd / "field"
    return Path(output_dir) / "field" / stamp(when)


def contest_artifact_paths(sim_directory: Path, contest_id: int) -> tuple[Path, Path]:
    base = Path(sim_directory) / f"contest_{int(contest_id)}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


# --------------------------------------------------------------------------- discovery
def replay_lock(path: Path) -> Optional[datetime]:
    """The lock of the replayed slate a path sits in (its directory, or the parent for a repeat), else None."""
    for d in (Path(path), Path(path).parent, Path(path).parent.parent):
        mp = d / REPLAY_MANIFEST
        if mp.exists():
            try:
                return datetime.fromisoformat(json.loads(mp.read_text())["lock"])
            except (KeyError, ValueError, json.JSONDecodeError):
                return None
    return None


def when_of(path: Path, stamp_text: Optional[str]) -> Optional[datetime]:
    """When a file was made: one minute before lock inside a replayed slate, else its stamp."""
    lock = replay_lock(Path(path).parent)
    if lock is not None:
        return lock - timedelta(minutes=1)
    if stamp_text:
        try:
            return datetime.strptime(stamp_text, STAMP_FMT)
        except ValueError:
            return None
    return None


@dataclass
class ProjectionFile:
    when: datetime
    path: Path


@dataclass
class SimRun:
    when: datetime
    dir: Path

    def contests(self) -> List[int]:
        return sorted(int(p.stem.split("_")[1]) for p in self.dir.glob("contest_*.npz"))


def find_projection_files(dirs: Iterable[Path]) -> List[ProjectionFile]:
    """projections*.csv under the given directories (and their projections/ subfolder), oldest first."""
    out: List[ProjectionFile] = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in list(d.glob("projections.csv")) + list(d.glob("projections.*.csv")) + list((d / "projections").glob("projections.*.csv")):
            parts = p.name.split(".")
            when = when_of(p, parts[1] if len(parts) == 3 else None)
            if when is not None:
                out.append(ProjectionFile(when, p))
    return sorted(out, key=lambda f: f.when)


def find_sim_runs(dirs: Iterable[Path]) -> List[SimRun]:
    """Simulator artifact directories under the given directories (<dir>/sim or <dir>/sim/<stamp>), oldest first."""
    return _find_artifact_runs(dirs, "sim")


def find_field_runs(dirs: Iterable[Path]) -> List[SimRun]:
    """Field-stage artifact directories (<dir>/field or <dir>/field/<stamp>), oldest first; same layout as a sim run."""
    return _find_artifact_runs(dirs, "field")


def _find_artifact_runs(dirs: Iterable[Path], sub: str) -> List[SimRun]:
    out: List[SimRun] = []
    for d in dirs:
        root = Path(d) / sub
        if not root.exists():
            continue
        if any(root.glob("contest_*.npz")):                      # a replay's run directory
            when = when_of(root, None)
            if when is not None:
                out.append(SimRun(when, root))
            continue
        for stamped in root.iterdir():
            if stamped.is_dir() and any(stamped.glob("contest_*.npz")):
                when = when_of(stamped, stamped.name)
                if when is not None:
                    out.append(SimRun(when, stamped))
    return sorted(out, key=lambda r: r.when)


def latest_before(items: Sequence, when: Optional[datetime]):
    """The last item (with a .when) at or before `when`; the last of all when `when` is None."""
    cands = [i for i in items if when is None or i.when <= when]
    return cands[-1] if cands else None
