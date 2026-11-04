#!/usr/bin/env python3
"""
scripts/nba/migrate_edits_by_day.py — one-time (idempotent): a slate's edits become its day's.

Before, each slate directory held its own edits/ (pins, painted grids, scenarios) and published/ layer. After, the day holds
them (data/nba/days/<YYYY-MM-DD>/, ivan.sports.nba.slates: a game is mapped once, the slates of one evening share it, the next
day starts clean). This moves every slate's files there:

  - pins (minutes_overrides.csv) and grids (rotations/<TEAM>.json): into the day's; when two slates of the day both have one,
    the newer file stands and the other is kept beside it as <name>.from-<dg>.<ext>
  - scenarios (scenarios/<TEAM>/*.json, _active.json): by the day they were saved (saved_at / since inside the file, else the
    file's time), so a scenario from another day is filed under that day and never picks itself for this one
  - published/ (a consumer): into the day's published/
  - a slate whose recorded lock is between midnight and 6 AM was recorded in UTC (no NBA slate locks then): lock and day are
    rewritten in local time first

    venv/bin/python scripts/nba/migrate_edits_by_day.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ivan.sports.nba import slates as S


def _empty(f: Path) -> bool:
    """A pins file with only its header (what an untouched desk writes) says nothing: it never outranks a file with rows."""
    try:
        return f.suffix == ".csv" and len(f.read_text().strip().splitlines()) <= 1
    except OSError:
        return False


def _place(src: Path, dst: Path, dg: int, log, dry: bool) -> None:
    """Move src to dst; when dst exists and differs, the newer stands and the older is filed beside it (a header-only pins file
    counts as absent)."""
    if dst.exists() and _empty(dst) and not _empty(src):
        log(f"  {dst} has no rows: replaced"); dry or dst.unlink()
    elif dst.exists():
        if dst.read_bytes() == src.read_bytes():
            log(f"  same as {dst}: {src.name} dropped"); dry or src.unlink(); return
        if dst.stat().st_mtime >= src.stat().st_mtime:
            other = dst.with_name(f"{dst.stem}.from-{dg}{dst.suffix}")
            log(f"  {dst} is newer: {src} filed as {other.name}"); dry or shutil.move(str(src), str(other)); return
        other = dst.with_name(f"{dst.stem}.from-{dst.parent.name}{dst.suffix}")
        log(f"  {src} is newer: the day's {dst.name} filed as {other.name}"); dry or shutil.move(str(dst), str(other))
    log(f"  {src} -> {dst}")
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(src), str(dst))


def _saved_day(f: Path) -> str:
    try:
        j = json.loads(f.read_text())
        for k in ("saved_at", "since"):
            if j.get(k):
                return str(j[k])[:10]
    except (OSError, ValueError, AttributeError):
        pass
    return datetime.fromtimestamp(f.stat().st_mtime).date().isoformat()


def _rmdir_empty(d: Path) -> None:
    for sub in sorted(d.rglob("*"), reverse=True):
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()
    if d.is_dir() and not any(d.iterdir()):
        d.rmdir()


def fix_utc_lock(dg: int, meta: dict, log, dry: bool) -> dict:
    lock = S.meta_time(meta, "lock")
    if lock is None or not (0 <= lock.hour < 6):
        return meta
    local = lock.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    log(f"slate {dg}: lock {lock:%Y-%m-%d %H:%M} was UTC -> {local:%Y-%m-%d %H:%M}, day {local.date()}")
    if dry:
        return dict(meta, lock=local.isoformat(timespec="seconds"), day=local.date().isoformat())
    return S.write_meta(dg, lock=local, day=local.date())


def migrate(log=print, dry: bool = False) -> int:
    moved = 0
    for dg, meta in S.each():
        meta = fix_utc_lock(dg, meta, log, dry)
        if not meta.get("day"):
            log(f"slate {dg}: no day on record (no lock): its edits stay under the slate"); continue
        sp = S.paths(dg) if not dry else S.SlatePaths(dg=dg, root=S.paths(dg).root, day=str(meta["day"])[:10], days=S.days_dir())
        if sp.day_root is None:
            continue
        old_edits, old_pub = sp.root / "edits", sp.root / "published"
        if not old_edits.is_dir() and not old_pub.is_dir():
            continue
        log(f"slate {dg} ({meta['day']}): edits -> {sp.day_root}")
        if old_edits.is_dir():
            f = old_edits / "minutes_overrides.csv"
            if f.exists():
                _place(f, sp.overrides_csv, dg, log, dry); moved += 1
            for g in sorted((old_edits / "rotations").glob("*.json")) if (old_edits / "rotations").is_dir() else []:
                _place(g, sp.rotations / g.name, dg, log, dry); moved += 1
            if (old_edits / "scenarios").is_dir():
                for team_dir in sorted(d for d in (old_edits / "scenarios").iterdir() if d.is_dir()):
                    for sc in sorted(team_dir.glob("*.json")):
                        day = _saved_day(sc)
                        target_root = sp.day_root if day >= str(meta["day"])[:10] else sp.days / day
                        if target_root != sp.day_root:
                            log(f"  {sc} is from {day}: filed under that day")
                        _place(sc, target_root / "edits" / "scenarios" / team_dir.name / sc.name, dg, log, dry); moved += 1
            dry or _rmdir_empty(old_edits)
        if old_pub.is_dir():
            for f in sorted(p for p in old_pub.rglob("*") if p.is_file()):
                _place(f, sp.published / f.relative_to(old_pub), dg, log, dry); moved += 1
            dry or _rmdir_empty(old_pub)
    log(f"{'would move' if dry else 'moved'} {moved} file(s)")
    return moved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    migrate(dry=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
