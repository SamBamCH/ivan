"""
The NBA desk's application: the FastAPI app mounted at /nba by ivan.web.main, one Desk per slate (ivan.sports.nba.desk) made
on first use and bound to a request by its address (/nba/<day>/<draft group>/...), the job runner, consumer / limits switches,
the render helpers and the sport description (ivan.web.sports.Sport) the frame uses.

Every route under a slate (`slate` router, prefix /{day}/{key}) works that slate's desk: `desk` in the route modules is a
proxy to the desk bound to the current request, `LOCK` to its lock, so a refresh of one slate never touches another and
two slates' pages serve at once. A slate this machine does not hold yet has no desk: while it is being fetched LOADING
holds the message its page shows.
"""
from __future__ import annotations

import contextvars
import os
import re
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ivan.core import feed
from ivan.sports.nba.slate import autopilot
from ivan.sports.nba.backtest import results
from ivan.sports.nba import slates as S
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nba.paths import NBA_DATA_DIR
from ivan.web.jobs import JobRunner
from ivan.web import auth, frame
from ivan.web.sports import Day, SlateInfo, Sport, group_days
from ivan.sports.nba.desk import CONSUMER, Desk, LOBBY, engine   # noqa: F401  (engine, the history database, is re-exported to the route modules)


HERE = Path(__file__).parent.parent                               # ivan/web: the shared templates and static files

app = FastAPI(title="Ivan · NBA desk", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

slate = APIRouter(prefix="/{day}/{key}")                         # every action of one slate; ivan.web.nba includes it once the route modules are loaded

KEY, LABEL = "nba", "NBA"
VIEWS = [("games", "pre-lock"), ("players", "players"), ("lateswap", "post-lock"), ("backtest", "backtest")]
SPORT: Optional[Sport] = None                                 # set by sport(); the frame reads it
NOT_AN_EDIT = ("/refresh", "/reload", "/news", "/autopilot", "/login", "/publish", "/slate", "/board", "/pipeline", "/export", "/download")


# ------------------------------------------------------------------------------ the desks
DESKS: dict[int, Desk] = {}
_DESKS_LOCK = threading.Lock()
LOADING: dict[int, str] = {}                                  # a slate being fetched that has no desk yet: draft group -> what is happening
PROGRESS: dict[int, dict] = {}                                # every slate with a refresh, load, restore or pull running: {what, started, lines}


def progress_of(dg: Optional[int]) -> Optional[dict]:
    """The running background work on a slate, with elapsed seconds, for the banner; None when nothing runs."""
    if dg is None or int(dg) not in PROGRESS:
        return None
    pr = PROGRESS[int(dg)]
    return dict(pr, elapsed=int((datetime.now() - pr["started"]).total_seconds()))
_bound: contextvars.ContextVar[Optional[Desk]] = contextvars.ContextVar("ivan_nba_desk", default=None)


class NoDesk(RuntimeError):
    pass


def desk_for(dg: int, create: bool = True) -> Optional[Desk]:
    """The slate's desk, made on first use when this machine holds the slate (a directory with its database); None otherwise."""
    dg = int(dg)
    with _DESKS_LOCK:
        d = DESKS.get(dg)
        if d is None and create and S.holds(dg):
            with S.using(dg):
                d = Desk(dg)
            DESKS[dg] = d
        return d


def bind(d: Optional[Desk], dg: Optional[int] = None):
    """Make `d` (and its slate) current for this context; returns the tokens for `unbind`. Threads that work a slate call this."""
    t1 = S.use(dg if dg is not None else d.dg) if (d is not None or dg is not None) else None
    t2 = _bound.set(d)
    return (t1, t2)


def unbind(tokens) -> None:
    t1, t2 = tokens
    _bound.reset(t2)
    if t1 is not None:
        S.release(t1)


def bound() -> Optional[Desk]:
    """The desk bound to the current request (None on a page outside a slate, or of a slate not held)."""
    return _bound.get()


class _DeskProxy:
    """`desk` in the route modules: the desk bound to the current request."""
    def _d(self) -> Desk:
        d = _bound.get()
        if d is None:
            raise NoDesk("no slate's desk is bound to this request (the address names no held slate)")
        return d

    def __getattr__(self, name):
        return getattr(self._d(), name)

    def __setattr__(self, name, value):
        setattr(self._d(), name, value)

    def __bool__(self):
        return _bound.get() is not None


class _LockProxy:
    """`with LOCK:` in the route modules: the bound desk's lock."""
    def __enter__(self):
        return _bound.get().lock.__enter__()

    def __exit__(self, *exc):
        return _bound.get().lock.__exit__(*exc)


desk = _DeskProxy()
LOCK = _LockProxy()

SLATE_PATH = re.compile(r"^/(\d{4}-\d{2}-\d{2})/(\d+)(?:/|$)")


@app.middleware("http")
async def _bind_slate(request: Request, call_next):
    """A request under /nba/<day>/<draft group>/ works that slate: its desk (made on first use, off the event loop) and its paths
    are bound for the request. Also the producer's footprint for the autopilot tick (ivan.sports.nba.slate.autopilot): every POST
    that works a slate is an edit; the tick stands down for a while after one. A consumer leaves no footprint."""
    path = request.url.path[len(auth.root_of(request)):] or "/"
    m = SLATE_PATH.match(path)
    tokens = None
    if m:
        dg = int(m.group(2))
        request.state.slate_day, request.state.dg = m.group(1), dg
        d = await run_in_threadpool(desk_for, dg)
        tokens = bind(d, dg)
    try:
        if request.method == "POST" and not CONSUMER:
            rest = path[m.end() - 1:] if m else path
            if not rest.startswith(NOT_AN_EDIT) and not rest.endswith("/note"):      # a note is not a projection edit
                try:
                    autopilot.touch("edit")
                except OSError:
                    pass
        return await call_next(request)
    finally:
        if tokens is not None:
            unbind(tokens)


# ------------------------------------------------------------------------------ jobs
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


def slate_jobs(dg: Optional[int] = None) -> list:
    """The jobs of one slate (the bound one by default), newest first."""
    dg = int(dg) if dg is not None else (bound().dg if bound() else None)
    return [j for j in runner.all() if (j.meta or {}).get("slate") == dg]


def _active(dg: Optional[int] = None) -> list:
    """The slate's running or queued jobs: what refuses a refresh of that slate (another slate's jobs do not)."""
    return [j for j in slate_jobs(dg) if j.is_active]


def _submit(name: str, cmd: list, step: str, extra: Optional[dict] = None) -> object:
    """Submit a job for the bound slate: `--slate <dg>` on the script, the slate in the job's record (with the work units and the
    time estimate the desk made, ivan.sports.nba.slate.job_time, so the status line counts down and later fits learn from the run)."""
    d = bound()
    return runner.submit(name, [str(c) for c in cmd] + ["--slate", str(d.dg)], meta=dict(step=step, slate=d.dg, **(extra or {})))


# ------------------------------------------------------------------------------ render helpers
def _busy(request: Request):
    """A refresh is replacing this slate's pool: no edits or jobs on it until it is done (reads keep working)."""
    if bound() is not None and desk.busy:
        desk.flash.append(f"wait: {desk.busy}")
        return RedirectResponse(_here(request), status_code=303)
    return None


SHOWS_FLASH = {"_nba_board.html", "_nba_side.html", "_nba_rotation.html", "_nba_pipeline.html"}   # the templates that display the flash: only they consume it


def sport_root(request: Request) -> str:
    return auth.root_of(request)


def slate_root(request: Request) -> str:
    """The bound slate's root, /nba/<day>/<dg>: what the templates' action addresses hang off."""
    day = getattr(request.state, "slate_day", None); dg = getattr(request.state, "dg", None)
    if dg is None:
        return auth.root_of(request)
    return f"{auth.root_of(request)}/{day}/{dg}"


def _render(request: Request, template: str, **ctx):
    # the page shell and the partials that do not show messages leave them for the next render that does (the board after
    # a redirect, the card after an edit); before this every render cleared them and most were never seen
    d = bound()
    flash = d.flash[:] if d is not None else []
    if d is not None and template in SHOWS_FLASH:
        d.flash.clear()
    elif template not in SHOWS_FLASH:
        flash = []
    return templates.TemplateResponse(request, template, {"now": datetime.now(), "flash": flash, "root": slate_root(request), "sport_root": sport_root(request),
                                                          "here": _here(request), "consumer": CONSUMER, "limited": LIMITED, "limits": LIMITS,
                                                          "job_max_minutes": JOB_MAX_MINUTES, "refreshing": (d.busy if d is not None else None),
                                                          "progress": progress_of(getattr(request.state, "dg", None)), **ctx})


def _loaded_slate() -> tuple[Optional[int], Optional[datetime]]:
    """(draft group, first lock) of the bound slate; (None, None) outside a slate. No lock taken: a column read."""
    d = bound()
    if d is None:
        return None, None
    try:
        lock = min((p.gametime for p in d.slate.players if p.gametime), default=None) if d.slate is not None else None
        return d.dg, lock
    except Exception:   # noqa: BLE001
        return d.dg, None


def slate_url(request: Request, dg: int, lock: Optional[datetime] = None) -> str:
    """/nba/<day>/<dg>/ for any slate: the day from its lock, its record, else today."""
    day = (lock or S.meta_time(S.read_meta(dg), "lock") or datetime.now()).date().isoformat()
    return f"{auth.root_of(request)}/{day}/{int(dg)}/"


def _here(request: Request) -> str:
    """The bound slate's page (/nba/<day>/<dg>/): where every action returns. The picker outside a slate."""
    day = getattr(request.state, "slate_day", None); dg = getattr(request.state, "dg", None)
    if dg is None:
        return auth.root_of(request) + "/"
    return f"{auth.root_of(request)}/{day}/{dg}/"


def _team_url(request: Request, team: str) -> str:
    """The team's page under the bound slate: /nba/<day>/<draft group>/<TEAM>/."""
    return _here(request) + (f"{team}/" if getattr(request.state, "dg", None) is not None else "")


def _days_of(rows: list, now: datetime) -> list[Day]:
    """Lobby.slate_days rows -> the frame's days (ivan.web.sports). A lobby slate this machine does not hold carries a load form (its
    page starts that load when opened; the form is the picker's button and the fallback when a job holds the slate): the producer
    fetches the slate from DraftKings, the consumer asks the laptop for it. An archived one opens on its backtest, where Restore
    loads it."""
    out = []
    for d_iso, _label, slates in rows:
        d = date.fromisoformat(d_iso)
        for sl in slates:
            lock = datetime.fromisoformat(sl["lock"]) if sl.get("lock") else None
            src = {"loaded": "working", "held": "working"}.get(sl["source"], sl["source"])
            load = None
            if sl["source"] == "lobby" and not sl["loaded"]:
                load = {"url": "/slate", "fields": {"draft_group": sl["id"], "source": "lobby", "day": d_iso},
                        "label": "fetch from the laptop" if CONSUMER else "load into the desk",
                        "title": ("ask the laptop for this slate and copy it here" if CONSUMER else
                                  "fetch this slate from DraftKings, project it and work it (every other slate stays as it is)")}
            detail = (f"{sl['contests']} contests in the lobby" if sl.get("contests") else "")
            out.append(SlateInfo(key=str(sl["id"]), day=d, lock=lock, label=sl["label"].replace(" · archived", ""), source=src, loaded=bool(sl["loaded"]),
                                 mode="classic", detail=detail, load=load, games=sl.get("games")))
    return group_days(out, now)


def nba_days(now: datetime) -> list[Day]:
    """The slate index for the frame and the home page: the lobby, the archive and every held slate (Lobby.slate_days)."""
    return _days_of(LOBBY.slate_days(now), now)


def nba_summary(now: datetime) -> dict:
    days = nba_days(now)
    main = S.main_slate_today(now)
    loaded = next((s for d in days for s in d.slates if main is not None and s.key == str(main)), None)
    cur = runner.current()
    note = ""
    if CONSUMER:
        fd = feed.status("nba") or {}
        note = f"consumer · laptop published {str(fd.get('published_at') or '')[11:16]}" if fd.get("published_at") else "consumer · nothing pulled yet"
    else:
        ap = autopilot.state()
        note = f"autopilot {'on' if ap.get('on') else 'off'}"
    held = S.held()
    if held:
        note += f" · {len(held)} slate{'s' if len(held) != 1 else ''} held"
    ahead = [d for d in days if d.day >= now.date()]
    nxt = min(ahead, key=lambda d: d.day) if ahead else None
    ledger = None
    if not CONSUMER:                                              # the money, this machine only (ivan.sports.nba.backtest.results)
        try:
            ledger = results.ledger_days(now)["overall"]
        except Exception:   # noqa: BLE001 - the home page opens without it
            ledger = None
    return {"loaded": loaded, "job": cur.name if cur else None, "note": note, "n_days": len(days), "next": nxt.label if nxt else "", "ledger": ledger}


def sport_job() -> Optional[dict]:
    """The sport-level job on the picker (fetch upcoming): the newest job with no slate, its status and the tail of its log."""
    jobs = [j for j in runner.all() if (j.meta or {}).get("slate") is None]
    if not jobs:
        return None
    j = jobs[0]
    if not j.is_active and j.finished and (time.time() - j.finished) > 1800:
        return None                                                   # an old run: nothing to say
    return dict(id=j.id, name=j.name, status=j.status, duration=j.duration, log=runner.read_log(j, tail_lines=8))


ACTIONS = [dict(label="Fetch upcoming slates", path="/fetch-upcoming",
                title="load every classic slate the DraftKings lobby lists (today and tomorrow) that this desk does not hold yet, then store "
                      "every contest of those slates and the payout table of each (scripts/nba/fetch_upcoming.py; free; a few minutes)")]


def sport() -> Sport:
    """The description ivan.web.main mounts: the desk with the frame's picker at its root."""
    global SPORT
    from ivan.sports.nba.backtest.slate import mode_for
    from ivan.sports.nba.backtest import readiness
    SPORT = Sport(key=KEY, label=LABEL, views=VIEWS, days=nba_days, summary=nba_summary, tools=[], app=app,
                  ledger=None if CONSUMER else results.ledger_days,       # the picker's money per day; never on the consumer (the server)
                  day_status=None if CONSUMER else readiness.day_status,  # a past day's readiness for the backtest: the calendar's colours
                  actions=[] if CONSUMER else ACTIONS, job=None if CONSUMER else sport_job,
                  default_view=lambda s, now: {"lateswap": "lateswap", "backtest": "backtest"}.get(mode_for(s.day, s.lock, now), "games"))
    frame.install_picker(app, SPORT)
    return SPORT
