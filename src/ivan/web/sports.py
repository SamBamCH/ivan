# src/ivan/web/sports.py
"""
The sports the web UI serves and the slate index each one gives the frame.

One Ivan UI: pick the sport, then the date, then the slate (ivan.web.main). Every sport describes
itself with a `Sport`: its key and label, the views a slate offers (the dropdown after the slate),
the tools that are not about one slate (jobs, the backtest runs, models), a `days` function that
lists its slates by day (the DraftKings lobby, the archive, what the workspace holds now) and a
`summary` for the home page. The frame (ivan.web.frame) renders the same bar and the same picker
page for every sport from this description; the sport's own module renders the slate's screens.

    REGISTRY        key -> Sport, filled by main.build_app; the frame's sport dropdown reads it
    SlateInfo       one slate as the picker and the breadcrumb know it
    Day             one day of slates
    group_days      slates -> days, newest first, each day's slates by lock
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Optional


@dataclass
class SlateInfo:
    key: str                              # the DraftKings draft group id, as text
    day: date                             # the day the slate locks (local)
    lock: Optional[datetime]              # first lock, when known
    label: str                            # "19:00 · 8 games", what the dropdown shows
    source: str                           # lobby | archive | working
    loaded: bool = False                  # the sport's workspace holds this slate now
    mode: str = ""                        # classic | showdown | ...
    detail: str = ""                      # one line for the picker card: contests, lineups, settle status
    status: str = ""                      # open | waiting | settled | ... (the settle job's word), "" when unknown
    load: Optional[dict] = None           # for a slate the picker can load into the workspace: {"url": POST target, "fields": {...}}

    def path(self, root: str) -> str:
        return f"{root}/{self.day.isoformat()}/{self.key}/"

    def as_dict(self) -> dict:
        return {"key": self.key, "day": self.day.isoformat(), "lock": self.lock.isoformat(timespec="minutes") if self.lock else None,
                "label": self.label, "source": self.source, "loaded": self.loaded, "mode": self.mode, "detail": self.detail,
                "status": self.status, "load": self.load}


@dataclass
class Day:
    day: date
    label: str                            # "Thu 10-30 · today"
    tag: str                              # today | tomorrow | yesterday | ""
    slates: list[SlateInfo] = field(default_factory=list)

    @property
    def iso(self) -> str:
        return self.day.isoformat()

    def as_dict(self) -> dict:
        return {"day": self.iso, "label": self.label, "tag": self.tag, "slates": [s.as_dict() for s in self.slates]}


@dataclass
class Sport:
    key: str                              # nba | nfl | mma
    label: str                            # NBA
    views: list[tuple[str, str]]          # (view key, label) in dropdown order; every sport ends with ("backtest", "backtest")
    days: Callable[[datetime], list[Day]]           # the slate index
    summary: Callable[[datetime], dict]             # the home tile: {"loaded": SlateInfo|None, "job": str|None, "note": str}
    tools: list[tuple[str, str]] = field(default_factory=list)   # (label, path under the sport root) for pages that are not one slate's
    default_view: Callable[[SlateInfo, datetime], str] = lambda s, now: "backtest"
    app: object = None                    # the mounted sub-application (FastAPI), None while it is not built
    error: str = ""                       # why the workspace is missing (an import that failed); the home page says so


REGISTRY: dict[str, Sport] = {}


def day_tag(d: date, today: date) -> str:
    return {0: "today", 1: "tomorrow", -1: "yesterday"}.get((d - today).days, "")


def day_label(d: date, today: date) -> str:
    tag = day_tag(d, today)
    return f"{d:%a %m-%d}" + (f" · {tag}" if tag else "")


def group_days(slates: list[SlateInfo], now: datetime) -> list[Day]:
    """Slates into days, newest day first; within a day by lock, the loaded slate first on a tie."""
    by_day: dict[date, list[SlateInfo]] = {}
    for s in slates:
        by_day.setdefault(s.day, []).append(s)
    out = []
    for d in sorted(by_day, reverse=True):
        rows = sorted(by_day[d], key=lambda s: (s.lock or datetime.combine(d, datetime.min.time()), not s.loaded, s.label))
        out.append(Day(day=d, label=day_label(d, now.date()), tag=day_tag(d, now.date()), slates=rows))
    return out


def pick_day(days: list[Day], wanted: Optional[date], now: datetime) -> Optional[Day]:
    """The day the picker opens on: the one asked for, else the loaded slate's, else today, else the next day ahead, else the newest."""
    if not days:
        return None
    if wanted is not None:
        return next((d for d in days if d.day == wanted), None)
    loaded = next((d for d in days if any(s.loaded for s in d.slates)), None)
    if loaded is not None:
        return loaded
    today = now.date()
    exact = next((d for d in days if d.day == today), None)
    if exact is not None:
        return exact
    ahead = [d for d in days if d.day > today]
    return min(ahead, key=lambda d: d.day) if ahead else days[0]


def find_slate(days: list[Day], key: str) -> Optional[SlateInfo]:
    for d in days:
        for s in d.slates:
            if s.key == str(key):
                return s
    return None


def lobby_lock(start_utc) -> Optional[datetime]:
    """DraftKings' StartDate ("2026-10-25T23:00:00.0000000Z") as a local naive datetime, like player gametimes."""
    if not start_utc:
        return None
    try:
        from datetime import timezone
        return datetime.fromisoformat(str(start_utc)[:19]).replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    except ValueError:
        return None


def lobby_day(start: str, start_utc, now: datetime) -> date:
    """The day of a lobby slate: its StartDate when the lobby gave one, else DK's display string ("Sun 12:00PM", "Tomorrow 7:00PM")."""
    lock = lobby_lock(start_utc)
    if lock is not None:
        return lock.date()
    head = (start or "").split()[0].lower()[:3] if start else ""
    wd = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    if head in wd:
        return now.date() + timedelta(days=(wd[head] - now.weekday()) % 7)
    if head == "tom":
        return now.date() + timedelta(days=1)
    return now.date()
