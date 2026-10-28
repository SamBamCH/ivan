# src/ivan/sports/nfl/retention.py
"""
What the NFL data directory keeps, and what it lets go of.

Kept for good, because a model build or a backtest needs it and nothing can
recompute it: the live database with its slate archive (captures, raw
artifacts, contests, standings, box scores, our finishes), data/nfl/raw,
data/nfl/derived, data/nfl/models (live, asof, archive) and data/nfl/input.
Also kept for good, because it is small and it is the record of what a run
found: every run's run.json and selection.json, each replayed slate's
replay.json manifest with its projections / entries / environment / usage
csv, the report_* directories (sim vs actual, the repeats distribution) and
the validation summary.json. The per-slate run index (run_index.py) reads
only these, so a slate's history of builds survives every sweep.

Let go of, because running the replay again rebuilds it from the above:

  * worker clones of the replay database (replay.w*.sqlite and the sidecar
    -wal/-shm/-journal files) once a day has passed since their last write
    (a live replay writes its clone constantly); scripts/replay.py deletes
    them itself when its workers finish, so these are the ones a crash or a
    cancel left behind;
  * HEAVY_DAYS after a run's last write, its heavy files: the per-repeat
    outputs (rep_*/, or the chooser files, simroi npz and sim/ of a flat
    single-repeat slate), the validation csvs (summary.json stays), the run's
    own copy of the replay database and the logs. The sweep writes purged_at
    into run.json so the pages can say so.

A run with "pinned": true in its run.json keeps its heavy files
(scripts/nfl/retention.py --pin NAME, or the Pin button on its page). A run
with a slate still "running" and written to in the last day is not touched.

Roots swept: data/nfl/output/backtests/<name> (the Backtest hub's runs),
data/nfl/output/models/<component>/<stamp> (the Models page's runs), the
lab replay roots data/nfl/output/replay* and the clones in data/nfl/replay.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from ivan.sports.nfl.paths import NFL_DATA_DIR, OUTPUT_DIR

HEAVY_DAYS = 30            # a run's heavy files live this long after its last write
CLONE_HOURS = 24           # a worker clone not written to for this long is a leftover
SPORT = "nfl"
KEEP_IN_VALIDATION = {"summary.json"}


@dataclass
class Item:
    path: Path
    bytes: int
    reason: str             # "clone" | "heavy"
    run: str                # the run root's name, for the log


@dataclass
class RunState:
    root: Path
    name: str
    kind: str                           # backtest | models | replay | replay-dir
    last_write: datetime
    pinned: bool = False
    active: bool = False                # a slate still running and written to within CLONE_HOURS
    purged_at: Optional[str] = None
    heavy: List[Item] = field(default_factory=list)
    clones: List[Item] = field(default_factory=list)

    @property
    def heavy_bytes(self) -> int:
        return sum(i.bytes for i in self.heavy)

    @property
    def clone_bytes(self) -> int:
        return sum(i.bytes for i in self.clones)


# --------------------------------------------------------------------------- the roots
def run_roots(output_dir: Path = OUTPUT_DIR) -> List[tuple[str, Path]]:
    """(kind, root) for every run directory on disk: hub runs, Models page runs, lab replay roots."""
    out: List[tuple[str, Path]] = []
    bt = output_dir / "backtests"
    for d in sorted(bt.iterdir()) if bt.is_dir() else []:
        if d.is_dir():
            out.append(("backtest", d))
    mo = output_dir / "models"
    for comp in sorted(mo.iterdir()) if mo.is_dir() else []:
        for d in sorted(comp.iterdir()) if comp.is_dir() else []:
            if d.is_dir():
                out.append((f"models/{comp.name}", d))
    for d in sorted(output_dir.glob("replay*")):
        if d.is_dir():
            out.append(("replay", d))
    return out


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    total = 0
    for dirpath, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _mtime(p: Path) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime)
    except OSError:
        return None


def _manifest(d: Path) -> Optional[dict]:
    try:
        return json.loads((d / "replay.json").read_text())
    except (OSError, ValueError):
        return None


def clone_files(where: Path, stem: str = "replay") -> List[Path]:
    """The worker clones under a directory: <stem>.w<i>.sqlite plus their sidecar files."""
    out: List[Path] = []
    for p in sorted(where.glob(f"{stem}.w*.sqlite*")):
        if p.is_file():
            out.append(p)
    return out


def inspect_run(kind: str, root: Path, now: datetime, heavy_days: int = HEAVY_DAYS, clone_hours: int = CLONE_HOURS) -> RunState:
    """What a run holds and what the rules say about it; nothing is deleted here."""
    run_json = root / "run.json"
    run: dict = {}
    if run_json.is_file():
        try:
            run = json.loads(run_json.read_text())
        except (OSError, ValueError):
            run = {}
    sport = root / SPORT
    writes: List[datetime] = []         # the replay's own writes; run.json is rewritten by pins and status changes and does not count
    active = False
    slate_dirs: List[Path] = []
    for d in sorted(sport.iterdir()) if sport.is_dir() else []:
        if not d.is_dir():
            continue
        m = _manifest(d)
        if m is None:
            continue
        slate_dirs.append(d)
        t = _mtime(d / "replay.json")
        if t:
            writes.append(t)
            if m.get("status") == "running" and now - t < timedelta(hours=clone_hours):
                active = True
    for rep in sorted(sport.glob("report_*")) if sport.is_dir() else []:
        t = _mtime(rep / "summary.json")
        if t:
            writes.append(t)
    for sj in sorted(sport.glob("validation/*/summary.json")) if sport.is_dir() else []:
        t = _mtime(sj)
        if t:
            writes.append(t)
    last_write = max(writes) if writes else (_mtime(run_json) or _mtime(root) or now)
    st = RunState(root=root, name=root.name if kind != "replay" else root.name, kind=kind, last_write=last_write,
                  pinned=bool(run.get("pinned")), active=active, purged_at=run.get("purged_at"))

    # worker clones: leftovers once nothing has written to them for a day
    for p in clone_files(root) + (clone_files(sport) if sport.is_dir() else []):
        t = _mtime(p)
        if t and now - t > timedelta(hours=clone_hours) and not active:
            st.clones.append(Item(p, _size(p), "clone", st.name))

    # heavy files: after heavy_days, unless pinned or active
    if not st.pinned and not active and now - last_write > timedelta(days=heavy_days):
        heavy: List[Path] = []
        for name in ("replay.sqlite", "replay.sqlite-wal", "replay.sqlite-shm", "replay.sqlite-journal"):
            if (root / name).is_file():
                heavy.append(root / name)
        if sport.is_dir():
            heavy += [p for p in sport.glob("*.log") if p.is_file()]
            vdir = sport / "validation"
            if vdir.is_dir():
                for dirpath, _dirs, files in os.walk(vdir):
                    for f in files:
                        if f not in KEEP_IN_VALIDATION:
                            heavy.append(Path(dirpath) / f)
        for d in slate_dirs:
            heavy += sorted(p for p in d.glob("rep_*") if p.is_dir())
            if (d / "sim").is_dir():
                heavy.append(d / "sim")
            heavy += sorted(p for p in d.glob("chosen_entries.*") if p.is_file())
            heavy += sorted(p for p in d.glob("*.npz") if p.is_file())
            if (d / "replay.log").is_file():
                heavy.append(d / "replay.log")
        for p in heavy:
            st.heavy.append(Item(p, _size(p), "heavy", st.name))
    return st


def stray_clones(replay_dir: Path = NFL_DATA_DIR / "replay", now: Optional[datetime] = None, clone_hours: int = CLONE_HOURS) -> List[Item]:
    """Clones of the default replay database (scripts/replay.py without --replay-db) a day after their last write."""
    now = now or datetime.now()
    out = []
    for p in clone_files(replay_dir, stem=f"{SPORT}_replay"):
        t = _mtime(p)
        if t and now - t > timedelta(hours=clone_hours):
            out.append(Item(p, _size(p), "clone", replay_dir.name))
    return out


# --------------------------------------------------------------------------- the sweep
@dataclass
class Sweep:
    runs: List[RunState]
    stray: List[Item]
    applied: bool = False
    freed: int = 0

    @property
    def items(self) -> List[Item]:
        out: List[Item] = []
        for r in self.runs:
            out += r.clones + r.heavy
        return out + self.stray

    @property
    def bytes(self) -> int:
        return sum(i.bytes for i in self.items)


def plan(now: Optional[datetime] = None, output_dir: Path = OUTPUT_DIR, replay_dir: Optional[Path] = None,
         heavy_days: int = HEAVY_DAYS, clone_hours: int = CLONE_HOURS) -> Sweep:
    now = now or datetime.now()
    runs = [inspect_run(kind, root, now, heavy_days, clone_hours) for kind, root in run_roots(output_dir)]
    return Sweep(runs=runs, stray=stray_clones(replay_dir or NFL_DATA_DIR / "replay", now, clone_hours))


def _remove(p: Path) -> None:
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    elif p.exists() or p.is_symlink():
        p.unlink()


def apply(sweep: Sweep, now: Optional[datetime] = None, log: Callable[[str], None] = print) -> Sweep:
    """Delete what the plan lists; mark each run whose heavy files went with purged_at in its run.json."""
    now = now or datetime.now()
    freed = 0
    for r in sweep.runs:
        for it in r.clones + r.heavy:
            try:
                _remove(it.path)
                freed += it.bytes
            except OSError as e:
                log(f"  could not remove {it.path}: {e}")
        if r.heavy:
            rj = r.root / "run.json"
            run = {}
            if rj.is_file():
                try:
                    run = json.loads(rj.read_text())
                except (OSError, ValueError):
                    run = {}
            run["purged_at"] = now.isoformat(timespec="seconds")
            run.setdefault("name", r.name)
            rj.write_text(json.dumps(run, indent=1, default=str))
            r.purged_at = run["purged_at"]
            log(f"  {r.kind} {r.name}: heavy files removed ({r.heavy_bytes / 1e6:.0f} MB; last write {r.last_write:%Y-%m-%d})")
        if r.clones:
            log(f"  {r.kind} {r.name}: {len(r.clones)} clone(s) removed ({r.clone_bytes / 1e9:.1f} GB)")
    for it in sweep.stray:
        try:
            _remove(it.path)
            freed += it.bytes
            log(f"  {it.path}: stray clone removed ({it.bytes / 1e9:.1f} GB)")
        except OSError as e:
            log(f"  could not remove {it.path}: {e}")
    sweep.applied = True
    sweep.freed = freed
    return sweep


def sweep(dry_run: bool = False, now: Optional[datetime] = None, heavy_days: int = HEAVY_DAYS, log: Callable[[str], None] = print) -> Sweep:
    """The nightly entry point: plan, print, and (unless dry_run) apply."""
    now = now or datetime.now()
    s = plan(now, heavy_days=heavy_days)
    n_clone = sum(len(r.clones) for r in s.runs) + len(s.stray)
    n_heavy = sum(1 for r in s.runs if r.heavy)
    log(f"retention: {len(s.runs)} run(s); {n_clone} leftover clone(s), heavy files of {n_heavy} run(s) older than {heavy_days} days; "
        f"{s.bytes / 1e9:.1f} GB" + (" (dry run)" if dry_run else ""))
    if dry_run:
        for r in s.runs:
            if r.clones or r.heavy:
                log(f"  {r.kind} {r.name}: last write {r.last_write:%Y-%m-%d}" + (f", {len(r.clones)} clone(s) {r.clone_bytes / 1e9:.1f} GB" if r.clones else "")
                    + (f", heavy {r.heavy_bytes / 1e6:.0f} MB" if r.heavy else "") + (" (pinned)" if r.pinned else ""))
        for it in s.stray:
            log(f"  stray clone {it.path} {it.bytes / 1e9:.1f} GB")
        return s
    return apply(s, now, log)


# --------------------------------------------------------------------------- pins
def find_run(name: str, output_dir: Path = OUTPUT_DIR) -> Optional[Path]:
    """A run root by name (a hub run), by component/stamp (a Models run), by a replay root's name, or by path."""
    p = Path(name)
    if p.is_dir() and (p / SPORT).is_dir():
        return p
    for kind, root in run_roots(output_dir):
        if root.name == name or f"{root.parent.name}/{root.name}" == name or str(root) == name:
            return root
    return None


def set_pinned(root: Path, pinned: bool) -> dict:
    rj = root / "run.json"
    run: dict = {}
    if rj.is_file():
        try:
            run = json.loads(rj.read_text())
        except (OSError, ValueError):
            run = {}
    run.setdefault("name", root.name)
    run["pinned"] = bool(pinned)
    rj.write_text(json.dumps(run, indent=1, default=str))
    return run


def describe(items: Iterable[Item]) -> str:
    items = list(items)
    return f"{len(items)} item(s), {sum(i.bytes for i in items) / 1e9:.1f} GB"
