# src/ivan/sports/nba/autopilot.py
"""
Autopilot: the laptop refreshes and publishes on its own when nobody is at the desk.

The capture tick (scripts/nba/capture.py, a user timer every 15 minutes) does nothing unless the switch is on.
With it on, each tick refreshes the slate the desk left loaded (or the one that locks next), fetches the props at
the usual offsets before lock, projects with the pins and grids on file, and publishes + pushes the feed, so the
consumer's desk (the brother's) keeps getting fresh projections on a day the producer is not playing.

The switch is a file (data/nba/autopilot.json) so the desk, the tick and the command line agree without a shared
process. Off by default. The desk toggles it (POST /autopilot) and shows the state; scripts/nba/autopilot.py on|off|status
does the same from a terminal.

The guard: the tick stands down while someone is working the desk, even with the switch on. The desk records its
activity (data/nba/desk_activity.json: the last edit, the last publish) and its jobs leave their status under
data/nba/jobs/; a tick within HOLD_MINUTES of an edit that has not been published since, or while a job is running,
skips and says why. `--force` on the tick overrides the switch and the guard for a test.

The state of the switch and of the last tick travel with the feed manifest (feed.publish(..., producer=...)), so the
consumer's desk can say "autopilot on, last tick 14:30" or "autopilot off: the laptop publishes by hand".
"""
from __future__ import annotations

import json
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ivan.sports.nba.paths import NBA_DATA_DIR

AUTOPILOT_JSON = NBA_DATA_DIR / "autopilot.json"
DESK_ACTIVITY_JSON = NBA_DATA_DIR / "desk_activity.json"
JOBS_DIR = NBA_DATA_DIR / "jobs"
HOLD_MINUTES = 30            # a tick this soon after an unpublished desk edit stands down
JOB_STALE_MINUTES = 90       # a job still "running" after this long is a dead desk process, not a job


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=1, default=str))
    tmp.replace(path)


def _parse(ts) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(ts)) if ts else None
    except ValueError:
        return None


# ---------------------------------------------------------------- the switch
def is_on(path: Optional[Path] = None) -> bool:
    return bool(_read(path or AUTOPILOT_JSON).get("on"))


def set_on(on: bool, by: str = "desk", path: Optional[Path] = None) -> dict:
    path = path or AUTOPILOT_JSON
    st = _read(path)
    st.update(on=bool(on), since=datetime.now().isoformat(timespec="seconds"), by=by)
    _write(path, st)
    return st


def record_tick(result: str, *, ok: bool, detail: Optional[dict] = None, path: Optional[Path] = None) -> dict:
    """What the last tick did (or why it stood down), for the headers of both desks."""
    path = path or AUTOPILOT_JSON
    st = _read(path)
    st["last_tick"] = dict(at=datetime.now().isoformat(timespec="seconds"), result=result, ok=bool(ok), **(detail or {}))
    _write(path, st)
    return st


def state(path: Optional[Path] = None) -> dict:
    """{on, since, by, last_tick: {at, result, ok, ...} | None} as stored (empty values when never set)."""
    st = _read(path or AUTOPILOT_JSON)
    return dict(on=bool(st.get("on")), since=st.get("since"), by=st.get("by"), last_tick=st.get("last_tick"))


# ---------------------------------------------------------------- the desk's footprint
def touch(kind: str, path: Optional[Path] = None) -> None:
    """The desk calls this on every edit (kind='edit') and every publish (kind='publish')."""
    path = path or DESK_ACTIVITY_JSON
    st = _read(path)
    st[f"{kind}_at"] = datetime.now().isoformat(timespec="seconds")
    _write(path, st)


def running_job(jobs_dir: Optional[Path] = None, now: Optional[float] = None) -> Optional[str]:
    """The id of a desk job whose status file says running (and started recently enough to believe it), else None."""
    jobs_dir = jobs_dir or JOBS_DIR
    now = now or time.time()
    if not jobs_dir.is_dir():
        return None
    for p in jobs_dir.glob("*.json"):
        d = _read(p)
        if d.get("status") == "running" and (now - float(d.get("started") or 0)) < JOB_STALE_MINUTES * 60:
            return str(d.get("id") or p.stem)
    return None


def desk_hold(now: Optional[datetime] = None, activity: Optional[Path] = None, jobs_dir: Optional[Path] = None,
              hold_minutes: int = HOLD_MINUTES) -> Optional[str]:
    """Why the tick should stand down right now, or None: someone edited within hold_minutes and has not published since,
    or a desk job is running. The switch is checked separately (is_on)."""
    now = now or datetime.now()
    job = running_job(jobs_dir, now=now.timestamp())
    if job:
        return f"a desk job is running ({job})"
    st = _read(activity or DESK_ACTIVITY_JSON)
    edit, pub = _parse(st.get("edit_at")), _parse(st.get("publish_at"))
    if edit is not None and (pub is None or edit > pub):
        ago = (now - edit).total_seconds() / 60
        if ago < hold_minutes:
            return f"the desk was edited {ago:.0f} min ago and has not published since (holding {hold_minutes} min)"
    return None


# ---------------------------------------------------------------- what the feed carries
def producer_info(source: str) -> dict:
    """The producer's note for the feed manifest: who published (desk | autopilot | nightly | capture), the switch, the last tick."""
    st = state()
    return dict(source=source, host=socket.gethostname(), autopilot=st["on"], autopilot_since=st["since"],
                last_tick=st["last_tick"], at=datetime.now().isoformat(timespec="seconds"))
