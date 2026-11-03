"""
The NBA desk's application: the FastAPI app mounted at /nba by ivan.web.main, the one Desk object every route reads under LOCK, the job runner,
consumer / limits switches, the render helpers and the sport description (ivan.web.sports.Sport) the frame uses.
"""
from __future__ import annotations

import os
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ivan.core import feed
from ivan.sports.nba import autopilot
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nba.paths import NBA_DATA_DIR
from ivan.web.jobs import JobRunner
from ivan.web import auth, frame
from ivan.web.sports import Day, SlateInfo, Sport, group_days
from ivan.sports.nba.desk import CONSUMER, Desk, engine   # noqa: F401  (engine is re-exported to the route modules)


HERE = Path(__file__).parent.parent                               # ivan/web: the shared templates and static files










app = FastAPI(title="Ivan · NBA desk", docs_url=None, redoc_url=None)


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


templates = Jinja2Templates(directory=HERE / "templates")


KEY, LABEL = "nba", "NBA"


VIEWS = [("games", "pre-lock"), ("players", "players"), ("lateswap", "post-lock"), ("backtest", "backtest")]


SPORT: Optional[Sport] = None                                 # set by sport(); the frame reads it


NOT_AN_EDIT = ("/refresh", "/reload", "/news", "/autopilot", "/login", "/publish")


@app.middleware("http")
async def _desk_activity(request: Request, call_next):
    """The producer's footprint for the autopilot tick (ivan.sports.nba.autopilot): every POST that works the slate is an
    edit; the tick stands down for a while after one that was not published since. A consumer leaves no footprint."""
    if request.method == "POST" and not CONSUMER:
        path = request.url.path[len(auth.root_of(request)):] or "/"
        if not path.startswith(NOT_AN_EDIT) and not path.endswith("/note"):      # a note is not a projection edit
            try:
                autopilot.touch("edit")
            except OSError:
                pass
    return await call_next(request)


LOCK = threading.Lock()


runner = JobRunner(log_dir=NBA_DATA_DIR / "jobs", cwd=PROJECT_ROOT, max_seconds=None)      # the limit is set below once LIMITED is known


SCRIPTS = PROJECT_ROOT / "scripts" / "nba"


MAX_JOBS = 8


LIMITED = CONSUMER or os.environ.get("IVAN_DESK_LIMITS", "").strip() in ("1", "true", "yes")


LIMITS = {"lineups": 30000, "jobs": max(1, min(MAX_JOBS, (os.cpu_count() or 2) - 1)), "sims": 2000, "field_cap": 20000,
          "n_samples": 10000, "ls_candidates": 20, "ls_field_cap": 10000, "ls_sims": 2000}


JOB_MAX_MINUTES = 45


if LIMITED:
    runner.max_seconds = JOB_MAX_MINUTES * 60.0


NEWS_AUTO_SECONDS = 300    # the desk's auto-check interval while it is on


desk = Desk()


def _pool_vanished_handler():
    from sqlalchemy.orm.exc import ObjectDeletedError

    async def handler(request: Request, exc: ObjectDeletedError):
        with LOCK:
            desk.session.rollback(); desk.rebuild()
            desk.flash.append("the player pool was replaced on disk while the page was built; reloaded")
        return RedirectResponse(_here(request), status_code=303)
    app.add_exception_handler(ObjectDeletedError, handler)


_pool_vanished_handler()


def _busy(request: Request):
    """A refresh is replacing the pool: no edits or jobs until it is done (the board shows the state)."""
    if desk.busy:
        desk.flash.append(f"wait: {desk.busy}")
        return RedirectResponse(_here(request), status_code=303)
    return None


SHOWS_FLASH = {"_nba_board.html", "_nba_side.html", "_nba_rotation.html"}   # the templates that display the flash: only they consume it


def _render(request: Request, template: str, **ctx):
    # the page shell and the partials that do not show messages leave them for the next render that does (the board after
    # a redirect, the card after an edit); before this every render cleared them and most were never seen
    flash = desk.flash[:]
    if template in SHOWS_FLASH:
        desk.flash.clear()
    else:
        flash = []
    return templates.TemplateResponse(request, template, {"now": datetime.now(), "flash": flash, "root": auth.root_of(request), "here": _here(request), "consumer": CONSUMER,
                                                          "limited": LIMITED, "limits": LIMITS, "job_max_minutes": JOB_MAX_MINUTES, **ctx})


def _loaded_slate() -> tuple[Optional[int], Optional[datetime]]:
    """(draft group, first lock) of the pool the desk holds; (None, None) when it holds nothing. No lock taken: a column read."""
    try:
        dg = desk.pool_stamp[2] if desk.pool_stamp else None
        lock = min((p.gametime for p in desk.slate.players if p.gametime), default=None) if desk.slate is not None else None
        return (int(dg) if dg is not None else None), lock
    except Exception:   # noqa: BLE001
        return None, None


def _here(request: Request) -> str:
    """The loaded slate's page (/nba/<day>/<dg>/): where every action returns. The picker when nothing is loaded."""
    root = auth.root_of(request)
    dg, lock = _loaded_slate()
    if dg is None:
        return root + "/"
    return f"{root}/{(lock or datetime.now()).date().isoformat()}/{dg}/"


def _days_of(rows: list, now: datetime) -> list[Day]:
    """Desk.slate_days rows -> the frame's days (ivan.web.sports). A lobby slate that is not loaded carries a load form (its page
    starts that load when opened; the form is the picker's button and the fallback when a job holds the pool): the producer
    fetches the slate from DraftKings, the consumer asks the laptop for it. An archived one opens on its backtest, where
    Restore loads it."""
    out = []
    for d_iso, _label, slates in rows:
        d = date.fromisoformat(d_iso)
        for sl in slates:
            lock = datetime.fromisoformat(sl["lock"]) if sl.get("lock") else None
            src = {"loaded": "working"}.get(sl["source"], sl["source"])
            load = None
            if sl["source"] == "lobby" and not sl["loaded"]:
                load = {"url": "/slate", "fields": {"draft_group": sl["id"], "source": "lobby", "day": d_iso},
                        "label": "fetch from the laptop" if CONSUMER else "load into the desk",
                        "title": ("ask the laptop to load this slate and copy it here (both desks move to it)" if CONSUMER else
                                  "fetch this slate from DraftKings, project it and work it (the loaded slate keeps its lineups, sims and edits)")}
            detail = (f"{sl['contests']} contests in the lobby" if sl.get("contests") else "")
            out.append(SlateInfo(key=str(sl["id"]), day=d, lock=lock, label=sl["label"].replace(" · archived", ""), source=src, loaded=bool(sl["loaded"]),
                                 mode="classic", detail=detail, load=load, games=sl.get("games")))
    return group_days(out, now)


def nba_days(now: datetime) -> list[Day]:
    """The slate index for the frame and the home page (takes the desk lock: the picker is its own request)."""
    with LOCK:
        if not desk.busy:
            desk.maybe_reload()
        dg, _lock = _loaded_slate()
        return _days_of(desk.slate_days(now, dg), now)


def nba_summary(now: datetime) -> dict:
    days = nba_days(now)
    loaded = next((s for d in days for s in d.slates if s.loaded), None)
    cur = runner.current()
    note = ""
    if CONSUMER:
        fd = feed.status("nba") or {}
        note = f"consumer · laptop published {str(fd.get('published_at') or '')[11:16]}" if fd.get("published_at") else "consumer · nothing pulled yet"
    else:
        ap = autopilot.state()
        note = f"autopilot {'on' if ap.get('on') else 'off'}"
    ahead = [d for d in days if d.day >= now.date()]
    nxt = min(ahead, key=lambda d: d.day) if ahead else None
    return {"loaded": loaded, "job": cur.name if cur else None, "note": note, "n_days": len(days), "next": nxt.label if nxt else ""}


def _team_url(request: Request, team: str) -> str:
    """The team's page under the loaded slate: /nba/<day>/<draft group>/<TEAM>/ (the desk page when the slate has no lock to name a day)."""
    dg, lock = _loaded_slate()
    if dg is None:
        return f"{auth.root_of(request)}/desk/"
    day = (lock or datetime.now()).date().isoformat()
    return f"{auth.root_of(request)}/{day}/{dg}/{team}/"


def sport() -> Sport:
    """The description ivan.web.main mounts: the desk with the frame's picker at its root."""
    global SPORT
    from ivan.sports.nba.slate_backtest import mode_for
    SPORT = Sport(key=KEY, label=LABEL, views=VIEWS, days=nba_days, summary=nba_summary, tools=[], app=app,
                  default_view=lambda s, now: {"lateswap": "lateswap", "backtest": "backtest"}.get(mode_for(s.day, s.lock, now), "games"))
    frame.install_picker(app, SPORT)
    return SPORT
