# src/ivan/web/nba_desk.py
"""
NBA news desk: one page to manage a slate in the last minutes before lock.

Teams are cards sorted by how far they are from their norm (scratches, new
starters, unconfirmed lineups near tip, minutes far from history, the market
disagreeing with history, unpriced rotation players). Open a card to see the
roster with the model's minutes; type minutes to pin them and the team
re-projects in place; scratch pins 0. Publish writes projections and the
overrides file the capture loop and the sim read.

Separate process from the NFL app: python scripts/nba/serve.py (port 8001);
also mounted at /nba inside the NFL app (ivan.web.app) so one server process
serves both. Every link carries the mount prefix (`root`), and
IVAN_DESK_PASSWORD puts a login page in front (ivan.web.auth).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta
from ivan.sports.nba.ingest import locally_edited_teams, restore_from_archive
from ivan.sports.nba import newsfeed as NF
from ivan.sports.nba import scenarios as SC
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv

from ivan.core import feed
from ivan.core.config import config_path
from ivan.sports.nba import autopilot
from ivan.core.utils import PROJECT_ROOT, normalize_name
from ivan.database.session import get_session, init_db
from ivan.sports.nba.ingest import ensure_schema, fetch_slate, project_slate, refresh_pool, store_slate, IngestError
from ivan.sports.nba.models import HistoricalContest, Lineup, LineupPool, LineupSimulationResult, NBAMarketRate, NBAPlayer
from ivan.sports.nba.paths import NBA_DATA_DIR, MINUTES_OVERRIDES_CSV, OUTPUT_DIR, INPUT_DIR
from ivan.web.jobs import JobRunner
from ivan.web import auth
from ivan.sports.nba.projection.engine import ProjectionParams
from ivan.sports.nba.projection.live import LiveSlate, OVERRIDE_KEYS, PUBLISHED_OVERRIDES_CSV, override_layers, override_of, read_overrides_full, write_overrides
from ivan.sports.nba.projection.history import STATS
from ivan.sports.nba.validation import market_points, recent_minutes, review_team
from ivan.sports.nba.projection import rotation as ROT
from ivan.sports.nba.projection.groups import group_of as SK_group
from ivan.sports.nba.projection.book_minutes import book_minutes
TEAM_BUDGET = 241.0
FOUL_SHORT_PF = 4          # a game with this many fouls...
FOUL_SHORT_SHARE = 0.8     # ...and minutes under this share of the player's recent average is marked foul-shortened
ADD_BACK_MINUTES = 6.0     # a player the rotation left off, added back by hand, starts at his history or this many minutes

HERE = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")                                  # IVAN_FEED_URL (consumer) / IVAN_FEED_PUSH (producer)
CONSUMER = bool(os.environ.get("IVAN_FEED_URL", "").strip())        # the server: takes what the laptop pushes, never fetches
engine = init_db("nba")
ensure_schema(engine)
app = FastAPI(title="Ivan · NBA desk", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")
auth.install(app, templates)                                  # a password when IVAN_DESK_PASSWORD is set; open otherwise
NOT_AN_EDIT = ("/refresh", "/reload", "/news", "/autopilot", "/login", "/publish")


@app.middleware("http")
async def _desk_activity(request: Request, call_next):
    """The producer's footprint for the autopilot tick (ivan.sports.nba.autopilot): every POST that works the slate is an
    edit; the tick stands down for a while after one that was not published since. A consumer leaves no footprint."""
    if request.method == "POST" and not CONSUMER:
        path = request.url.path[len(auth.root_of(request)):] or "/"
        if not path.startswith(NOT_AN_EDIT):
            try:
                autopilot.touch("edit")
            except OSError:
                pass
    return await call_next(request)


STATE_FILE = NBA_DATA_DIR / "capture_state.json"
LOCK = threading.Lock()
runner = JobRunner(log_dir=NBA_DATA_DIR / "jobs", cwd=PROJECT_ROOT, max_seconds=None)      # the limit is set below once LIMITED is known
SCRIPTS = PROJECT_ROOT / "scripts" / "nba"
MAX_JOBS = 8
# training wheels: on a shared or public desk (the server: IVAN_FEED_URL set, or IVAN_DESK_LIMITS=1) every pipeline setting is
# clamped to these, a job is stopped after JOB_MAX_MINUTES, and the late swap (it needs the laptop's DraftKings login) is off
LIMITED = CONSUMER or os.environ.get("IVAN_DESK_LIMITS", "").strip() in ("1", "true", "yes")
LIMITS = {"lineups": 30000, "jobs": max(1, min(MAX_JOBS, (os.cpu_count() or 2) - 1)), "sims": 2000, "field_cap": 20000,
          "n_samples": 10000, "ls_candidates": 20, "ls_field_cap": 10000, "ls_sims": 2000}
JOB_MAX_MINUTES = 45


def _config() -> dict:
    with open(config_path("nba"), encoding="utf-8") as f:
        return json.load(f)


def _pipeline_defaults() -> dict:
    """The panel's fields, seeded from config/nba.json: the generator's knobs, the sim's, the chooser's weights."""
    c = _config(); g = c.get("generate", {}); ch = c.get("choose", {})
    fmt = lambda v: "" if v is None else ("%g" % v if isinstance(v, (int, float)) else str(v))
    return {"lineups": 25000, "jobs": MAX_JOBS, "sims": fmt(c.get("sim", {}).get("n_sims", 1000)), "field_cap": fmt(c.get("field", {}).get("cap", 10000)),
            "method": ch.get("method", "greedy"), "min_fpts_frac": fmt(g.get("min_fpts_frac", 0.9)),
            "randomness": fmt(g.get("randomness", 0.25)), "n_samples": fmt(g.get("n_samples", 5000)), "jitter": fmt(g.get("jitter", 0.05)),
            "player_min_fpts": fmt(g.get("player_min_fpts", c.get("player_min_fpts", 15))),
            "max_salary": fmt(c.get("max_salary", 50000)), "min_salary": fmt(c.get("min_salary", 49000)), "max_salary_per_team": fmt(c.get("max_salary_per_team", 30000)),
            "cash_pct": fmt(ch.get("cash_pct", 0.5)), "avg_roi_win": fmt(ch.get("avg_roi_win", 0.5)), "diversity_lambda": fmt(ch.get("diversity_lambda", 0.1)),
            "bankroll": fmt(ch.get("bankroll", 3.0)), "jam": "",
            "ls_candidates": "10", "ls_field_cap": "5000", "ls_sims": "1000", "ls_source": "draftkings", "ls_live": ""}


if LIMITED:
    runner.max_seconds = JOB_MAX_MINUTES * 60.0


def clamp_form(form: dict) -> list:
    """Clamp the pipeline settings to LIMITS on a limited desk; returns the notes of what was lowered."""
    notes = []
    if not LIMITED:
        return notes
    for k, cap in LIMITS.items():
        raw = str(form.get(k) or "").strip()
        try:
            v = float(raw)
        except ValueError:
            continue
        if v > cap:
            form[k] = str(int(cap)); notes.append(f"{k} {raw} -> {int(cap)}")
    return notes


PIPELINE_DEFAULTS = _pipeline_defaults()
if LIMITED:
    for _k, _cap in LIMITS.items():
        if _k in PIPELINE_DEFAULTS and float(PIPELINE_DEFAULTS[_k] or 0) > _cap:
            PIPELINE_DEFAULTS[_k] = str(_cap)
PIPELINE_DEFAULTS.setdefault("fresh", "on")          # generate starts the pool fresh unless the box is unticked (add to the pool: a second scenario's lineups)
pipeline_form: dict = dict(PIPELINE_DEFAULTS)
DESK_CONFIG = NBA_DATA_DIR / "jobs" / "desk_config.json"           # config/nba.json with the panel's knobs applied; the jobs read it

MAX_UPLOAD = 5 * 1024 * 1024   # the DK entries export is tens of KB
ROTATION_MIN = 15.0        # a scratch/questionable above this baseline is news
QUESTION_STATUSES = {"Q", "GTD", "D", "P"}   # DK statuses that leave the player's night open
NEWS_AUTO_SECONDS = 300    # the desk's auto-check interval while it is on
NEAR_TIP_MIN = 75          # minutes before tip when an unconfirmed lineup becomes a warning
MINUTES_DEV = 8.0          # model minutes this far from history is a role change
MARKET_GAP = 0.20          # market mean vs history-only projection, relative


def utc_to_local(stamp) -> datetime:
    """An ISO time from the odds API ('2026-10-23T19:45:19Z') as naive local time, like every other desk time."""
    t = pd.Timestamp(stamp)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert(datetime.now().astimezone().tzinfo).tz_localize(None).to_pydatetime()


def short_name(name: str) -> str:
    """'jaren jackson jr' -> 'jackson': the surname, for a lane segment's label."""
    toks = [t for t in str(name or "").split() if t not in ("jr", "sr", "ii", "iii", "iv")]
    return toks[-1] if toks else str(name or "")


class Desk:
    """Session + LiveSlate + per-team projections, rebuilt when the pool changes under us."""
    def __init__(self):
        self.session = get_session(engine)
        self.slate: Optional[LiveSlate] = None
        self.pool_stamp = None
        self.overrides: dict[str, float] = {}
        self.proj: dict[str, pd.DataFrame] = {}
        self.rot: dict[str, dict] = {}
        self.hist_only: dict[str, pd.DataFrame] = {}
        self.flash: list[str] = []
        self.busy: Optional[str] = None                     # "loading slate 155581" while a background refresh runs; the board polls
        self._lobby: tuple = (0.0, {})
        self.rebuild()
        try:
            backfill_lineup_slates(self.session, log=self.flash.append)
        except Exception:   # noqa: BLE001 - the export does it again on demand
            self.session.rollback()

    def lobby_slates(self) -> dict:
        """Today's classic slates in the DraftKings lobby (cached two minutes), labelled with DK's start string; the loaded
        slate is always present. Falls back to the slates the capture state knows when the lobby is unreachable."""
        now = time.time()
        if now - self._lobby[0] < 120:
            return self._lobby[1]
        out = {}
        try:
            from ivan.sports.nba.ingest import classic_slates
            for sl in classic_slates():                       # every day the lobby lists; the day picker places each on its day
                start = str(sl.get("start") or "")
                out[str(sl["id"])] = dict(label=f"{start} · {str(sl.get('name') or '').replace('NBA ', '')[:30]}", lock_dt=None, start=start, name=sl.get("name"),
                                          games=sl.get("games"), lock=sl.get("lock"))
        except Exception as e:   # noqa: BLE001 - the lobby is a convenience
            self.flash.append(f"could not list the lobby's slates ({type(e).__name__}); showing the known ones")
        self._lobby = (now, out)
        return out

    def slate_games(self, key: str) -> Optional[int]:
        """How many games an archived slate had: the distinct team pairs of its capture's players (cached per slate key)."""
        cache = self.__dict__.setdefault("_slate_games", {})
        if key in cache:
            return cache[key]
        from ivan.core import archive
        try:
            caps = [c for c in archive.captures(self.session, "nba") if str(c.slate_key) == str(key)]
            cap = max(caps, key=lambda c: c.captured_at) if caps else None
            players = archive.load_blobs(self.session, cap.id).get("players", []) if cap else []
            pairs = {frozenset((str(r.get("team")), str(r.get("opponent")))) for r in players if r.get("team") and r.get("opponent")}
            cache[key] = len(pairs) or None
        except Exception:   # noqa: BLE001
            cache[key] = None
        return cache[key]

    def slate_days(self, now: datetime, dg) -> list:
        """
        [(date iso, date label, [slate, ...])] newest day first, each slate {id, label, source, loaded}: the DraftKings lobby's
        draft groups placed on their day (DK gives a weekday + time), then every slate with an archive capture (loadable from
        the capture: scripts/nba archive at ingest and at tip), the loaded slate always present. The picker: a day, then its slates.
        """
        days: dict = {}
        seen = set()
        wd = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
        for key, v in self.lobby_slates().items():
            start = str(v.get("start") or v.get("label") or "")
            head = start.split()[0].lower()[:3] if start else ""
            if head in wd:
                d = now.date() + timedelta(days=(wd[head] - now.weekday()) % 7)
            elif head == "tom":
                d = now.date() + timedelta(days=1)
            else:
                d = now.date()
            lk = v.get("lock"); ng = v.get("games")
            if lk is not None:
                d = lk.date()
            when = f"{lk:%H:%M}" if lk is not None else (start.split(" ", 1)[1] if " " in start else start)
            days.setdefault(d.isoformat(), []).append(dict(id=str(key), label=when + (f" · {ng} game{'s' if ng != 1 else ''}" if ng else ""),
                                                           source="lobby", sort=(lk.strftime("%H:%M") if lk is not None else start), loaded=str(key) == str(dg)))
            seen.add(str(key))
        from sqlalchemy import text as _t
        for key, lock, src, n in self.session.execute(_t("select slate_key, min(lock), max(source), count(*) from archived_slate where sport = 'nba' group by slate_key")).all():
            if str(key) in seen:
                continue
            try:
                t = datetime.fromisoformat(str(lock))
            except (TypeError, ValueError):
                continue
            ng = self.slate_games(str(key))
            days.setdefault(t.date().isoformat(), []).append(dict(id=str(key), label=f"{t:%H:%M}" + (f" · {ng} game{'s' if ng != 1 else ''}" if ng else "") + " · archived",
                                                                  source="archive", sort=t.strftime("%H:%M"), loaded=str(key) == str(dg)))
            seen.add(str(key))
        if dg is not None and str(dg) not in seen:
            lock = min((p.gametime for p in self.slate.players if p.gametime), default=None)
            d = (lock or now).date().isoformat()
            ng = len({frozenset((p.team, p.opponent)) for p in self.slate.players if p.team and p.opponent})
            days.setdefault(d, []).append(dict(id=str(dg), label=(f"{lock:%H:%M}" if lock else "loaded") + (f" · {ng} game{'s' if ng != 1 else ''}" if ng else ""), source="loaded", sort="", loaded=True))
        out = []
        for d in sorted(days, reverse=True):
            dd = date.fromisoformat(d)
            tag = "today" if dd == now.date() else ("tomorrow" if dd == now.date() + timedelta(days=1) else ("yesterday" if dd == now.date() - timedelta(days=1) else ""))
            out.append((d, f"{dd:%a %m-%d}" + (f" · {tag}" if tag else ""), sorted(days[d], key=lambda x: x["sort"])))
        return out

    def _stamp(self):
        """(latest projection time, player count, draft group, hash of the player ids) of the pool on disk. A column query:
        it does not touch the player objects the page holds, so a stale page still renders from them."""
        rows = self.session.query(NBAPlayer.id, NBAPlayer.fpts_updated_at, NBAPlayer.draft_group_id).all()
        return (max((r[1] for r in rows if r[1]), default=None), len(rows), rows[0][2] if rows else None, hash(frozenset(int(r[0]) for r in rows)))

    def history(self):
        """(box, HistoryState), rebuilt only when nba_stats grew (the nightly box scores)."""
        n = self.session.execute(__import__("sqlalchemy").text("select count(*), max(game_date) from nba_stats")).one()
        key = (int(n[0]), str(n[1]))
        if getattr(self, "_hist_key", None) != key:
            from ivan.sports.nba.projection.history import build_history, load_box_scores
            box = load_box_scores(self.session)
            self._hist = (box, build_history(box)); self._hist_key = key
        return self._hist

    def rebuild(self):
        self.session.expire_all()
        self.slate = LiveSlate(self.session, history=self.history())
        self.pool_stamp = self._stamp(); self.stale = False
        self.overrides = read_overrides_full()            # what the projection reads: the published layer (a consumer) under this machine's own
        self.local_overrides = read_overrides_full(layered=False) if MINUTES_OVERRIDES_CSV.exists() else {}   # what this machine writes
        self.__dict__.pop("_layers_cache", None); self.__dict__.pop("_published_cache", None)
        self.proj, self.hist_only, self.rot = {}, {}, {}
        self._auto_pending = True                         # scenarios are matched once the rest of the rebuild is done (see end of rebuild)
        if getattr(self, "_calls_dg", None) != self.pool_stamp[2]:
            self._calls, self._calls_dg = {}, self.pool_stamp[2]   # typed calls belong to their slate
        self.season = self.season_averages(self.slate.box)
        from ivan.sports.nba import pbp as PBP
        try:
            self.fouls = PBP.fouls_per_game(self.session)
        except Exception:   # noqa: BLE001 - no foul table yet
            self.fouls = {}
        self.recent = self.recent_games(self.slate.box, fouls=self.fouls)
        self.market_rows = {}
        for r in self.session.query(NBAMarketRate).all():
            self.market_rows.setdefault(r.player_name, {})[r.stat] = dict(mean=float(r.market_mean), line=r.line, books=r.books, rate=float(r.rate), basis=float(r.basis_minutes),
                                                                           odds=None)
        for (name, stat), txt in self.consensus_odds().items():
            if name in self.market_rows and stat in self.market_rows[name]:
                self.market_rows[name][stat]["odds"] = txt
        for t in self.slate.teams():
            self.reproject(t)

        if getattr(self, "_auto_pending", False):
            self._auto_pending = False
            try:
                for m in self.auto_scenarios():
                    self.flash.append(m)
            except Exception as e:   # noqa: BLE001 - the pick is a convenience
                self.flash.append(f"scenario pick failed: {type(e).__name__}: {e}")

    @staticmethod

    def season_averages(box: pd.DataFrame) -> dict:
        """player -> (games, minutes per game, fantasy points per minute) this season, and the same over last season as a fallback."""
        today = datetime.now().date()
        start = date(today.year if today.month >= 10 else today.year - 1, 10, 1)
        out = {}
        for label, sub in (("season", box[(box.game_date >= start) & box.played]), ("last", box[(box.game_date < start) & box.played])):
            g = sub.groupby("player_name").agg(games=("minutes", "size"), minutes=("minutes", "mean"), fpts=("dk_fpts", "sum"), mins=("minutes", "sum"))
            for name, r in g.iterrows():
                out.setdefault(name, {})[label] = dict(games=int(r.games), minutes=float(r.minutes), fpm=float(r.fpts / r.mins) if r.mins > 0 else None)
        return out

    def consensus_odds(self) -> dict:
        """(player, stat) -> 'o-112 / u-108': the books' prices at the consensus line averaged in probability, the vig left in."""
        from ivan.sports.nba.projection.market import MARKET_TO_STAT, american_to_prob
        odds = pd.read_sql("select player, market, sportsbook, side, point, price from odds", self.session.get_bind())
        if odds.empty:
            return {}
        odds = odds[odds.market.isin(MARKET_TO_STAT) & ~odds.market.str.endswith("alternate")].copy()
        odds["stat"] = odds.market.map(MARKET_TO_STAT)
        out = {}
        for (name, stat), g in odds.groupby(["player", "stat"]):
            line = self.market_rows.get(name, {}).get(stat, {}).get("line")
            if line is None or pd.isna(line):
                continue
            at = g[np.isclose(g.point.astype(float), float(line))]
            parts = []
            for side, tag in (("over", "o"), ("under", "u")):
                prices = at[at.side.str.lower() == side].price.astype(float)
                if len(prices):
                    p_avg = float(np.mean([american_to_prob(x) for x in prices]))
                    amer = -100 * p_avg / (1 - p_avg) if p_avg >= 0.5 else 100 * (1 - p_avg) / p_avg
                    parts.append(f"{tag}{amer:+.0f}")
            if parts:
                out[(name, stat)] = " / ".join(parts)
        return out

    @staticmethod
    def recent_games(box: pd.DataFrame, n: int = 10, fouls: Optional[dict] = None) -> dict:
        """player -> his last n games played, newest first: (date, team, minutes, fppm, pf, foul_short). foul_short marks a
        game with FOUL_SHORT_PF fouls or more and minutes under FOUL_SHORT_SHARE of his average over these games: the
        minutes were probably cut by fouls, not by the coach's plan."""
        out = {}; fouls = fouls or {}
        sub = box[box.played].sort_values(["game_date", "game_id"])
        for name, g in sub.groupby("player_name"):
            tail = g.tail(n)
            avg = float(tail.minutes.mean()) if len(tail) else 0.0
            rows = []
            for r in tail.itertuples():
                pf = fouls.get((str(r.game_id), name))
                rows.append(dict(date=str(r.game_date), team=r.team, minutes=float(r.minutes), fpts=float(r.dk_fpts),
                                 fpm=float(r.dk_fpts) / float(r.minutes) if r.minutes > 0 else None, pf=pf,
                                 foul_short=bool(pf is not None and pf >= FOUL_SHORT_PF and avg > 0 and float(r.minutes) < FOUL_SHORT_SHARE * avg)))
            out[name] = rows[::-1]
        return out

    def maybe_reload(self):
        """
        The pool changed on disk (capture tick, feed pull). New projections for the same players: say so and keep the page
        as it is until Refresh. Different players (another slate, a player added or removed): the rows the page was built
        from are gone, so it has to reload, and says why.
        """
        now = self._stamp()
        if now == self.pool_stamp:
            self.stale = False
        elif now[1:] != self.pool_stamp[1:]:
            self.rebuild()
            self.flash.append("the player pool changed on disk (a capture tick moved to another slate or the roster changed); reloaded")
        else:
            self.stale = True

    def reproject(self, team: str, n_sims: int = 600):
        rng = np.random.default_rng(0)
        self.proj[team] = self.slate.project(team, self.overrides, set(), ProjectionParams.from_config(n_sims=n_sims), rng=rng)
        self.rot[team] = dict(getattr(self.slate, "_last_rotation", {}).get(team) or {})      # the grid behind this projection
        self.hist_only[team] = self.slate.project(team, self.overrides, set(),
                                                  ProjectionParams.from_config(n_sims=0, market_weight=0.0), rng=rng)

    # ---------------------------------------------------------------- views
    def players(self, team: str) -> list[NBAPlayer]:
        return sorted([p for p in self.slate.players if p.team == team], key=lambda p: -(p.salary or 0))

    def tip(self, team: str) -> Optional[datetime]:
        return min((p.gametime for p in self.players(team) if p.gametime), default=None)

    def market_means(self) -> dict[str, float]:
        """market fpts per priced player (sum of priced stats at DK weights; turnovers from history are small)"""
        out = {}
        rows = self.session.query(NBAMarketRate).all()
        from ivan.sports.nba.projection.rates import DK_WEIGHTS
        for r in rows:
            out[r.player_name] = out.get(r.player_name, 0.0) + DK_WEIGHTS.get(r.stat, 0.0) * float(r.market_mean)
        return out

    def rotation(self, team: str, source: str = "auto", game_id: Optional[str] = None) -> dict:
        """
        The team's rotation grid for the card.
          source "auto"    the projection's grid: the coach's slots filled at the solve's minutes, the desk's locked rows held
                           (ivan.sports.nba.projection.skeleton). Out players show their usual stretches faintly (the vacated
                           slots), players the schedule left off the floor are listed under the grid with an add button.
          source "propose" the same grid with every lock lifted (a preview, nothing saved)
          source "last"    one of this season's games (`game_id`, else the latest) as played, fouls marked on the minute
        """
        from ivan.sports.nba import pbp as PBP
        from ivan.sports.nba.projection.live import OUT_STATUSES
        from ivan.sports.nba.projection import skeleton as SK
        players = self.players(team)
        names = {normalize_name(p.name): p for p in players}
        is_out = {n: (p.status or "").upper() in OUT_STATUSES or (override_of(self.overrides, n).get("minutes") is not None and override_of(self.overrides, n).get("minutes") <= 0)
                  for n, p in names.items()}
        dropped = {n for n, p in names.items() if (p.status or "").upper() not in OUT_STATUSES and override_of(self.overrides, n).get("minutes") is not None
                   and override_of(self.overrides, n).get("minutes") <= 0}
        starters = self.starters_of(team) or set()
        today = datetime.now().date(); season_start = date(today.year if today.month >= 10 else today.year - 1, 10, 1)
        games = [dict(g, season="this") for g in ROT.team_games(self.session, team, since=season_start)]
        last_start = date(season_start.year - 1, 10, 1)              # last season's games stay selectable (a returning player's habits, a roster that
        games += [dict(g, season="last") for g in ROT.team_games(self.session, team, before=season_start, since=last_start, limit=100)]   # has barely played)
        rot = self.rot.get(team) or {}
        curve = rot.get("curve")
        # which played game is the last time tonight's group played together: tonight's rotation (available, with history
        # minutes, or a named starter) against each game's players with 5+ minutes; the dropdown says how close each one is
        matches: list = []
        if games:
            try:
                usual = {} if curve is None or curve.empty else {str(n): float(curve.loc[n, "minutes"]) for n in curve.index}
                grid_now = rot.get("grid")                      # tonight's grid when it exists: its players with minutes are the group
                if grid_now is not None and len(grid_now):
                    tonight = {str(n): max(usual.get(str(n), 0.0), float(grid_now.loc[n].sum())) for n in grid_now.index if float(grid_now.loc[n].sum()) > 0 and not is_out.get(str(n), False)}
                else:
                    tonight = {n: usual.get(n, 0.0) for n, p in names.items() if not is_out[n] and (n in starters or usual.get(n, 0.0) >= ROTATION_MIN)}
                matches = ROT.lineup_matches(games, ROT.played_sets(self.session, team, [g["game_id"] for g in games]), tonight)
                oddities = ROT.game_oddities(self.session, team, [g["game_id"] for g in games], usual)
                for m in matches:
                    m.update(oddities.get(str(m["game_id"]), {}))
                def label(m):
                    if m["exact"]:
                        return "same group"
                    bits = [f"{n.split()[-1]} out" for n in sorted(m["missing"], key=lambda k: -m["missing"][k])[:2]] + \
                           [f"{n.split()[-1]} played" for n in sorted(m["extra"], key=lambda k: -m["extra"][k])[:2]]
                    more = len(m["missing"]) + len(m["extra"]) - len(bits)
                    return ", ".join(bits) + (f" +{more}" if more > 0 else "")
                for m in matches:
                    m["match"] = label(m)
                games = matches
            except Exception:   # noqa: BLE001 - the match is a convenience; the latest game is the fallback
                matches = []
        proj = self.proj.get(team)
        targets = rot.get("targets") or {}
        base = dict(team=team, games=games, kind="none", drawn=False, locked=[], rows=[], counts=[], benched=[], game_id=None, last_game=None,
                    has_history=curve is not None and len(curve) > 0, curve_games=int(curve.games.max()) if curve is not None and len(curve) else 0,
                    fouls=False, owners=[])

        if source == "last":
            # the default comes from this season when it has any game; last season's are there to pick, not to pick from
            this_season = [m for m in matches if m.get("season") == "this"]
            best = ROT.best_match(this_season or matches) if (game_id is None and matches) else None
            if best is not None:
                game_id = best["game_id"]
            last = ROT.last_game_grid(self.session, team, since=season_start, game_id=game_id)
            if last is None:
                return base
            grid = last["grid"]
            fouls: dict = {}
            try:
                f = PBP.game_fouls(self.session, last["game_id"], team)
                for r in f[f.personal == 1].itertuples():
                    fouls.setdefault(str(r.player_name), []).append((int(r.minute), None if r.foul_no is None or pd.isna(r.foul_no) else int(r.foul_no)))
            except Exception:   # noqa: BLE001
                pass
            rows = []
            try:
                pairs = PBP.game_subs(self.session, last["game_id"], team).to_dict("records")
            except Exception:   # noqa: BLE001
                pairs = []
            # hover text on the cells where a player came on or went off: the recorded substitution, the game clock, how many changed at once
            def clock(t):
                per = min(4, int(t // 720) + 1) if t < 2880 else 4 + int((t - 2880) // 300) + 1
                left = (per * 720 - t) if t < 2880 else (2880 + (per - 4) * 300 - t)
                return f"{'Q' + str(per) if per <= 4 else 'OT' + str(per - 4)} {int(left // 60)}:{int(left % 60):02d}"
            at_once: dict = {}
            for pr in pairs:
                at_once[round(float(pr["elapsed"]), 1)] = at_once.get(round(float(pr["elapsed"]), 1), 0) + 1
            tips: dict = {}
            for pr in pairs:
                t = float(pr["elapsed"]); m = int(np.floor(t / 60.0 + 0.5)); k = at_once[round(t, 1)]
                note = "" if k == 1 else f" · {k} changed at once, the feed's pairing is its typing order"
                if 1 <= m <= 47:
                    tips.setdefault(pr["player_in"], {})[m] = f"in at {clock(t)} for {pr['player_out']}{note}"
                    tips.setdefault(pr["player_out"], {})[m - 1] = f"out at {clock(t)}, {pr['player_in']} in{note}"
            spots, owners = ROT.spots_of(grid, starters=None, groups={n: SK_group(names[n].position) for n in names}, positions={n: names[n].position for n in names},
                                         pairs=pairs)
            for n in grid.index:
                cells = [bool(x) for x in grid.loc[n].values]
                marks: list = [None] * 48
                for m, no in fouls.get(n, []):
                    marks[m] = no if no is not None else "T"
                hist = None if curve is None or n not in curve.index else float(curve.loc[n, "minutes"])
                mins = int(grid.loc[n].sum()); pf = len(fouls.get(n, []))
                cell_tips = [None] * 48
                for m, txt in tips.get(n, {}).items():
                    cell_tips[m] = txt
                for m, no in fouls.get(n, []):
                    cell_tips[m] = (f"foul {no}" if no is not None else "technical") + (f" · {cell_tips[m]}" if cell_tips[m] else "")
                rows.append(dict(name=n, key=n, pos=names[n].position if n in names else "", salary=names[n].salary if n in names else None, status=((names[n].status or '').upper() if n in names else ''),
                                 starter=bool(cells[0]), out=bool(is_out.get(n, False)), gone=n not in names, vacated=False, locked=False, minutes=mins, target=None,
                                 hist=hist, cells=cells, marks=marks, pf=pf, spots=[int(x) for x in spots.loc[n].values], tips=cell_tips,
                                 group=SK_group(names[n].position) if n in names else "wing", rank=ROT._spot_rank(names[n].position if n in names else None, None),
                                 foul_short=bool(pf >= FOUL_SHORT_PF and hist and mins < FOUL_SHORT_SHARE * hist), heir_of=""))
            match = next((m for m in matches if str(m["game_id"]) == str(last["game_id"])), None)
            return {**base, "kind": "last", "rows": rows, "counts": ROT.column_counts(grid), "game_id": last["game_id"], "last_game": last["game_date"], "fouls": True,
                    "owners": owners, "match": match, "games": games, "garbage_from": (match or {}).get("garbage_from")}

        grid = rot.get("grid")
        if grid is None or proj is None:
            return base
        locked = list(grid.attrs.get("locked", []))
        if source == "propose":
            avail = [n for n in grid.index]
            groups = {n: SK_group(names[n].position) for n in names}
            grid = SK.build_grid(curve, targets, avail, starters=[n for n in avail if n in starters], groups=groups, locks=None, vacated=rot.get("vacated"),
                                 shares=rot.get("shares"))
            locked = []
        inherited: dict = {}
        for owner, heirs in (grid.attrs.get("inherited") or {}).items():
            for h, share in heirs.items():
                if share >= 0.2:
                    inherited.setdefault(h, []).append(f"{owner.replace(' (old slot)', '')} {share:.0%}")
        rows = []
        shares = rot.get("shares") or {}
        spots, owners = ROT.spots_of(grid, starters=starters, groups={n: SK_group(names[n].position) for n in names}, positions={n: names[n].position for n in names},
                                     shares=shares)
        for n in grid.index:
            hist = None if curve is None or n not in curve.index else float(curve.loc[n, "minutes"])
            mins = int(grid.loc[n].sum())
            if mins <= 0 and n not in locked:
                continue
            rows.append(dict(name=n, key=n, pos=names[n].position if n in names else "", salary=names[n].salary if n in names else None, status=((names[n].status or '').upper() if n in names else ''),
                             starter=n in starters, out=False, gone=False, vacated=False, locked=n in locked, minutes=mins,
                             target=None if n not in targets else round(float(targets[n]), 1),
                             hist=hist, cells=[bool(x) for x in grid.loc[n].values], marks=None, pf=None, foul_short=False,
                             spots=[int(x) for x in spots.loc[n].values], group=SK_group(names[n].position) if n in names else "wing",
                             rank=ROT._spot_rank(names[n].position if n in names else None, None),
                             heir_of=", ".join(inherited.get(n, []))))
        # the vacated slots: out players with a curve, their usual stretches shown faintly; a player dropped by hand is marked so he can be restored
        for n in names:
            if is_out.get(n) and curve is not None and n in curve.index and float(curve.loc[n, "minutes"]) >= 3:
                cells = [bool(float(curve.loc[n, m]) >= 0.5) for m in range(48)]
                rows.append(dict(name=n, key=n, pos=names[n].position, salary=names[n].salary, status=(names[n].status or '').upper(), starter=n in starters, out=True, gone=False, vacated=True,
                                 locked=False, minutes=int(round(float(curve.loc[n, "minutes"]))), target=None, hist=float(curve.loc[n, "minutes"]),
                                 cells=cells, marks=None, pf=None, foul_short=False, heir_of="", spots=[-1] * 48, dropped=n in dropped,
                                 group=SK_group(names[n].position), rank=ROT._spot_rank(names[n].position, None)))
        # one order for every panel (tonight, the unlocked preview, the game below): the saved projection's minutes,
        # starters not promoted, vacated slots last; so the rows line up whatever is shown
        rows = self.order_rows(rows, owners)
        self.mark_starter_rule(rows)
        lanes = self.lanes_of(rows, owners)
        # off the floor tonight: available players the schedule gave no minutes (added back by hand)
        benched = []
        for n, p in names.items():
            if any(r["key"] == n for r in rows) or (is_out.get(n) and n not in dropped):
                continue
            hist = None if curve is None or n not in curve.index else float(curve.loc[n, "minutes"])
            benched.append(dict(name=n, key=n, pos=p.position, salary=p.salary, hist=hist, status=(p.status or "").upper(), dropped=n in dropped,
                                add_minutes=round(max(ADD_BACK_MINUTES, hist or 0.0))))
        benched.sort(key=lambda r: -(r["salary"] or 0))
        # they sit in the grid as blank rows (paint one to give him minutes; a dropped player shows DROPPED with his + to restore)
        out_rows = [r for r in rows if r.get("vacated") or r.get("out")]
        rows = [r for r in rows if not (r.get("vacated") or r.get("out"))]
        for b in benched:
            rows.append(dict(name=b["name"], key=b["key"], pos=b["pos"] or "", salary=b["salary"], status=b["status"], starter=False, out=False, gone=False,
                             vacated=False, locked=False, minutes=0, target=None, hist=b["hist"], cells=[False] * 48, marks=[None] * 48, pf=0,
                             spots=[-1] * 48, tips=[None] * 48, group=SK_group(b["pos"] or ""), rank=ROT._spot_rank(b["pos"], None), foul_short=False,
                             heir_of="", dropped=b["dropped"], bench=True))
        rows += out_rows
        avail_keys = {r["key"] for r in rows}
        shares_json = json.dumps({o: {i: round(v, 3) for i, v in hv.items() if i in avail_keys} for o, hv in shares.items() if o in avail_keys})
        n_pins = sum(1 for n in names if override_of(self.overrides, n).get("minutes") is not None)
        return {**base, "kind": "model" if source != "propose" else "propose", "rows": rows, "counts": ROT.column_counts(grid), "drawn": bool(locked), "pins": n_pins,
                "locked": locked, "benched": benched, "owners": owners, "lanes": lanes, "shares_json": shares_json}

    @staticmethod
    def mark_starter_rule(rows: list[dict]) -> list[dict]:
        """sep=True on the last row of the leading block of starters: the dark line under the starters. A starter listed lower
        (a game's starter who is out tonight, or one the model plays less than a bench player) stays bold below the line."""
        last = None
        for i, r in enumerate(rows):
            if r.get("starter") and not r.get("vacated"):
                last = i
            else:
                break
        if last is None:
            last = max((i for i, r in enumerate(rows) if r.get("starter") and not r.get("vacated")), default=None)
        for i, r in enumerate(rows):
            r["sep"] = (i == last)
        return rows

    def row_order(self, team: str) -> dict[str, int]:
        """player -> rank from the saved projection's grid: minutes descending, salary as the tie-break; starters are not promoted."""
        rot = self.rot.get(team) or {}
        grid = rot.get("grid")
        if grid is None:
            return {}
        sal = {normalize_name(p.name): (p.salary or 0) for p in self.players(team)}
        mins = {str(n): int(grid.loc[n].sum()) for n in grid.index}
        names = sorted(mins, key=lambda n: (-mins[n], -sal.get(n, 0), n))
        return {n: i for i, n in enumerate(names)}

    def game_grid(self, team: str, game_id: Optional[str] = None) -> dict:
        """A played game's grid for the comparison under tonight's: rows in the same order as tonight's panel, then the rest by minutes."""
        rot = self.rotation(team, source="last", game_id=game_id)
        # the game's starters first (struck through when out tonight: those are the minutes to give away), then the bench
        rot["rows"] = self.mark_starter_rule(self.order_rows(rot.get("rows") or [], rot.get("owners") or []))
        rot["lanes"] = self.lanes_of(rot["rows"], rot.get("owners") or [])
        if rot.get("game_id"):
            try:
                rot["proj"], rot["proj_meta"], rot["missed"] = self.game_projection(team, rot["game_id"], rot["last_game"], {r["key"] for r in rot["rows"]})
            except Exception as e:   # noqa: BLE001 - the comparison is a convenience; the grid stands without it
                rot["proj"], rot["proj_meta"], rot["missed"] = {}, dict(error=f"{type(e).__name__}: {e}"), []
        return rot

    def game_projection(self, team: str, game_id: str, game_date, grid_keys: set) -> tuple[dict, dict, list]:
        """
        What we projected for this team's players at this game, next to what they did: the at-tip archive of the game
        (scripts/nba/archive_at_lock.py) when one exists, else the last projection published before the tip, else the last
        one captured for the game. Returns ({player: pmin, pfpts, psd, pstarter, min_act, fpts_act, status}, meta, missed)
        where `missed` lists players we projected minutes for who are not on the grid (did not play).
        """
        from sqlalchemy import text as _t
        day = str(game_date)[:10]
        caps = pd.read_sql(_t("select captured_at, source, draft_group_id, gametime, player_name, status, starter, minutes_proj, fpts_mean, fpts_std, pinned "
                              "from nba_projection_capture where team = :t and date(gametime) = :d and source in ('at_tip', 'ours')").bindparams(t=team, d=day),
                           self.session.get_bind())
        if caps.empty:
            return {}, dict(none=True), []
        caps["captured_at"] = pd.to_datetime(caps["captured_at"]); caps["gametime"] = pd.to_datetime(caps["gametime"])
        tip = caps["gametime"].max()
        at_tip = caps[caps.source == "at_tip"]
        if len(at_tip):
            pick, how = at_tip[at_tip.captured_at == at_tip.captured_at.max()], "at tip"
        else:
            before = caps[(caps.source == "ours") & (caps.captured_at <= tip)]
            src = before if len(before) else caps[caps.source == "ours"]
            pick, how = src[src.captured_at == src.captured_at.max()], ("published" if len(before) else "captured after the tip")
        pick = pick.drop_duplicates("player_name", keep="last")
        act = pd.read_sql(_t("select player_name, minutes, dk_fpts, position from nba_stats where game_id = :g and team_abbreviation = :t").bindparams(g=str(game_id), t=team),
                          self.session.get_bind())
        act_by = {str(r.player_name): r for r in act.itertuples()}
        by, missed = {}, []
        for r in pick.itertuples():
            n = str(r.player_name); a = act_by.get(n)
            d = dict(pmin=None if pd.isna(r.minutes_proj) else float(r.minutes_proj), pfpts=None if pd.isna(r.fpts_mean) else float(r.fpts_mean),
                     psd=None if pd.isna(r.fpts_std) else float(r.fpts_std), pstarter=str(r.starter or ""), status=str(r.status or ""), pinned=bool(r.pinned),
                     min_act=None if a is None else float(a.minutes), fpts_act=None if a is None else float(a.dk_fpts),
                     started_act=bool(a is not None and a.position and a.position != "BENCH" and float(a.minutes) > 0))
            by[n] = d
            if n not in grid_keys and (d["pmin"] or 0) >= 1:
                missed.append(dict(name=n, pmin=d["pmin"], pfpts=d["pfpts"], status=d["status"]))
        missed.sort(key=lambda m: -(m["pmin"] or 0))
        stamp = pick.captured_at.max().to_pydatetime()
        meta = dict(how=how, captured_at=stamp, draft_group=int(pick.draft_group_id.iloc[0]) if pick.draft_group_id.notna().any() else None,
                    before_tip=None if tip is None or pd.isna(tip) else (tip.to_pydatetime() - stamp).total_seconds() / 60.0,
                    n=len(by), pmin_sum=float(pick.minutes_proj.fillna(0).sum()), pfpts_sum=float(pick.fpts_mean.fillna(0).sum()),
                    min_sum=float(act.minutes.sum()) if len(act) else 0.0, fpts_sum=float(act.dk_fpts.sum()) if len(act) else 0.0)
        return by, meta, missed

    @staticmethod
    def order_rows(rows: list[dict], owners: list[str]) -> list[dict]:
        """
        Starters in spot order 1-5 (guards to bigs, the lanes' order). The bench in spot order too when every bench player
        fills one spot all game; when any of them fills several, the bench is by minutes instead. Vacated slots last.
        """
        spot_of_owner = {o: k for k, o in enumerate(owners)}
        def spots_filled(r):
            return sorted({sp for sp, on in zip(r.get("spots") or [], r.get("cells") or []) if on and sp >= 0})
        starters = [r for r in rows if r.get("starter") and not r.get("vacated")]
        bench = [r for r in rows if not r.get("starter") and not r.get("vacated")]
        vacated = [r for r in rows if r.get("vacated")]
        starters.sort(key=lambda r: (spot_of_owner.get(r["key"], (spots_filled(r) or [9])[0]), -r["minutes"]))
        single = all(len(spots_filled(r)) <= 1 for r in bench)
        if single:
            bench.sort(key=lambda r: ((spots_filled(r) or [9])[0], -r["minutes"], -(r.get("salary") or 0)))
        else:
            bench.sort(key=lambda r: (-r["minutes"], -(r.get("salary") or 0)))
        vacated.sort(key=lambda r: -r["minutes"])
        return starters + bench + vacated

    @staticmethod
    def lanes_of(rows: list[dict], owners: list[str]) -> list[dict]:
        """
        The five spots traced minute by minute: for each spot, the stretches (start, length, player) of who held it, so a line
        (the centre's, say) reads as one band with the holder's name on each stretch; a minute nobody holds is a gap.
        """
        lanes = []
        for k, owner in enumerate(owners):
            holder = [None] * 48
            for r in rows:
                if r.get("vacated"):
                    continue
                for m, sp in enumerate(r.get("spots") or []):
                    if sp == k and r["cells"][m]:
                        holder[m] = r["key"]
            segs = []; start = 0
            for m in range(1, 49):
                if m == 48 or holder[m] != holder[start]:
                    name = holder[start]
                    segs.append(dict(start=start, length=m - start, key=name, short=short_name(name) if name else ""))
                    start = m
            lanes.append(dict(spot=k, owner=owner, segments=segs))
        return lanes

    def apply_rotation(self, team: str, players: list[str], grid: list[list[int]], locked: Optional[list[str]] = None, note: str = "") -> int:
        """
        Save the painted grid: the locked rows are held exactly by the skeleton and their row totals become the players'
        minutes (a row painted empty scratches him); a row that was locked and is now unlocked loses its minutes pin; the
        unlocked rows are re-filled around the locks by the next projection. Out players' rows are dropped.
        """
        from ivan.sports.nba.projection.live import OUT_STATUSES
        status = {normalize_name(p.name): (p.status or "").upper() for p in self.players(team)}
        keep = [i for i, n in enumerate(players) if n and len(grid[i]) == ROT.MINUTES]       # rows without a key (a popup's table) are not the grid's
        players = [players[i] for i in keep]; grid = [grid[i] for i in keep]
        g = pd.DataFrame(np.array(grid, dtype=bool).reshape(len(players), ROT.MINUTES), index=[normalize_name(n) for n in players], columns=list(range(ROT.MINUTES)))
        g = g.loc[[n for n in g.index if status.get(n, "") not in OUT_STATUSES]]
        lock_names = [normalize_name(n) for n in (locked if locked is not None else list(g.index))]
        lock_names = [n for n in lock_names if n in g.index]
        before = ROT.load_drawn(team)
        was_locked = set(before.attrs.get("locked", list(before.index))) if before is not None else set()
        for n in was_locked - set(lock_names):
            self.set_override(n, minutes=None)
        for n in lock_names:
            self.set_override(n, minutes=float(g.loc[n].sum()))
        # the fingerprint is taken after the pins: a row locked empty scratches its player, and that is part of the mapping
        ROT.save_drawn(team, g, note, locked=lock_names, fingerprint=self.fingerprint(team))
        self.reproject(team)
        return len(lock_names)

    def fingerprint(self, team: str) -> dict:
        """What a mapping is made against: who is available, who is out, each player's status, the five starters (Rotowire or hand calls)."""
        from ivan.sports.nba.projection.live import OUT_STATUSES
        avail, out, status = [], [], {}
        for p in self.players(team):
            n = normalize_name(p.name); st = (p.status or "").upper()
            ov = override_of(self.overrides, n)
            (out if (st in OUT_STATUSES or (ov.get("minutes") is not None and ov.get("minutes") <= 0)) else avail).append(n)
            if st and st != "NONE":
                status[n] = st
        return dict(available=sorted(avail), out=sorted(out), status=status, starters=sorted(self.starters_of(team) or []))

    def mapped_state(self, team: str) -> dict:
        """
        The team's mapping: unmapped (no held grid: nothing locked, or rows with minutes left to the model), mapped (every row
        with minutes locked and the inputs as they were when it was saved) or changed (held, but the availability, a status
        or the starters moved since: the differences listed). A refresh re-renders the card, so a change shows at once.
        """
        g = ROT.load_drawn(team)
        if g is None:
            return dict(state="unmapped", diff=[], saved_at=None)
        locked = set(g.attrs.get("locked", [])); rows = [str(n) for n in g.index if g.loc[n].sum() > 0]
        free = [n for n in rows if n not in locked]
        if not rows or free:
            return dict(state="unmapped", diff=([f"{len(free)} row(s) left to the model"] if rows else []), saved_at=g.attrs.get("saved_at"))
        fp = g.attrs.get("fingerprint")
        if not fp:
            return dict(state="changed", diff=["saved before fingerprints: map it again to track changes"], saved_at=g.attrs.get("saved_at"))
        now = self.fingerprint(team); diff = []
        short = lambda n: n.split()[-1]   # noqa: E731
        for n in sorted(set(fp["available"]) - set(now["available"])):
            diff.append(f"{short(n)} now OUT" if n in now["out"] else f"{short(n)} gone")
        for n in sorted(set(now["available"]) - set(fp["available"])):
            diff.append(f"{short(n)} back" if n in fp["out"] else f"{short(n)} new")
        if set(fp["starters"]) != set(now["starters"]):
            diff.append("starters: " + ", ".join([f"+{short(n)}" for n in sorted(set(now["starters"]) - set(fp["starters"]))] +
                                                 [f"-{short(n)}" for n in sorted(set(fp["starters"]) - set(now["starters"]))]))
        for n, st in now["status"].items():
            if n in now["available"] and n in fp["available"] and (fp.get("status") or {}).get(n) != st:
                diff.append(f"{short(n)} {st}")
        return dict(state="changed" if diff else "mapped", diff=diff, saved_at=g.attrs.get("saved_at"))

    def questions(self, team: str, now: Optional[datetime] = None) -> list:
        """
        The team's questionable players (DK status Q / GTD / D / P, not ruled out): who they are, what they usually play,
        whether you have made a call (a minutes pin or a starter call answers the question), and the latest news on file
        for each with its age and whether it arrived after the team was mapped. No questions = good to go at lock, barring
        a surprise lineup.
        """
        now = now or datetime.now()
        from ivan.sports.nba.projection.live import OUT_STATUSES
        hist = {r["key"]: r for r in self.team_rows(team)} if team in self.proj else {}
        ms = self.mapped_state(team)
        mapped_at = None
        if ms.get("state") in ("mapped", "changed") and ms.get("saved_at"):
            try:
                mapped_at = datetime.fromisoformat(str(ms["saved_at"]))
            except ValueError:
                mapped_at = None
        qs = []
        for p in self.players(team):
            st = (p.status or "").upper()
            if st not in QUESTION_STATUSES or st in OUT_STATUSES:
                continue
            n = normalize_name(p.name); ov = override_of(self.overrides, n)
            call = None
            if ov.get("minutes") is not None:
                call = "scratched" if ov["minutes"] <= 0 else f"pinned {ov['minutes']:g} min"
            elif ov.get("starter") is not None:
                call = "starts" if ov["starter"] > 0 else "benched"
            r = hist.get(n, {})
            qs.append(dict(name=n, raw=p.name, status=st, hist=r.get("hist_min"), minutes=r.get("minutes"), starter=(r.get("starter") == "S"),
                           answered=call is not None, call=call, salary=p.salary, news=None))
        if qs:
            news = NF.latest(self.session, [q["name"] for q in qs], hours=72, now=now)
            for q in qs:
                items = news.get(q["name"]) or []
                if items:
                    x = items[0]
                    q["news"] = dict(headline=x.headline, text=x.text, url=x.source_url or x.url, rw=x.url, injury=x.injury, age=NF.age(x.first_seen, now),
                                     first_seen=x.first_seen, new=(x.first_seen > mapped_at) if mapped_at else ((now - x.first_seen) <= timedelta(hours=3)),
                                     n=len(items))
        qs.sort(key=lambda q: (q["answered"], -(q["hist"] or 0)))
        return qs

    def questions_summary(self, now: Optional[datetime] = None) -> dict:
        now = now or datetime.now()
        teams = sorted({p.team for p in self.slate.players if p.team})
        by = {t: self.questions(t, now) for t in teams}
        by = {t: q for t, q in by.items() if q}
        return dict(n=sum(len(q) for q in by.values()), open=sum(1 for q in by.values() for x in q if not x["answered"]),
                    with_news=sum(1 for q in by.values() for x in q if x["news"]), new=sum(1 for q in by.values() for x in q if x["news"] and x["news"]["new"]),
                    teams={t: [f"{x['raw'].split()[-1]} {x['status']}" + (" ✓" if x["answered"] else "") for x in q] for t, q in by.items()})

    def news_status(self, now: Optional[datetime] = None) -> dict:
        now = now or datetime.now()
        pool = [p.name for p in self.slate.players]
        recent = NF.latest(self.session, pool, hours=24, now=now)
        n = sum(len(v) for v in recent.values())
        last = getattr(self, "news_checked_at", None)
        return dict(checked_at=last, checked=NF.age(last, now) if last else None, auto=bool(getattr(self, "news_auto", False)), n_slate=n,
                    newest=max((v[0].first_seen for v in recent.values()), default=None))

    def team_states(self, now: Optional[datetime] = None) -> list:
        """
        One chip per team on the slate: green = mapped and no open question; yellow = an open question (a questionable
        player who could shake the rotation), mapped or not; red = mapped but the inputs moved since; grey = unmapped
        with nothing open. The header's one-line read of the slate.
        """
        now = now or datetime.now()
        out = []
        for t in sorted({p.team for p in self.slate.players if p.team}):
            ms = self.mapped_state(t); qs = self.questions(t, now)
            open_q = [q for q in qs if not q["answered"]]
            if ms["state"] == "changed":
                colour, why = "changed", "mapped, but changed since: " + ", ".join(ms["diff"])
            elif open_q:
                colour, why = "open", f"{len(open_q)} open question(s): " + ", ".join(f"{q['raw']} {q['status']}" for q in open_q) + (" · mapped" if ms["state"] == "mapped" else " · unmapped")
            elif ms["state"] == "mapped":
                colour, why = "ready", "mapped, no open questions"
            else:
                colour, why = "unmapped", "unmapped" + (f" · {', '.join(ms['diff'])}" if ms.get("diff") else "")
            out.append(dict(team=t, state=ms["state"], colour=colour, why=why, open=len(open_q), new=sum(1 for q in open_q if q["news"] and q["news"]["new"])))
        return out

    # ---------------------------------------------------------------- scenarios: a team's edits saved under an assumption
    def team_overrides(self, team: str) -> dict:
        names = {normalize_name(p.name) for p in self.players(team)}
        return {n: dict(ov) for n, ov in self.local_overrides.items() if n in names}

    def raw_facts(self, team: str, calls: Optional[dict] = None) -> dict:
        """The facts a scenario is matched against: DK statuses and Rotowire's starters, plus the calls typed into the late-news form
        (not the pins: an active scenario's own scratch pin must not keep it matching after DK says otherwise)."""
        from ivan.sports.nba.projection.live import OUT_STATUSES
        out, avail, status_now = set(), set(), {}
        for p in self.players(team):
            n = normalize_name(p.name); status_now[n] = (p.status or "").upper()
            (out if status_now[n] in OUT_STATUSES else avail).add(n)
        starters = set(self.slate.starters.get(team) or set())
        # the calls typed into the late-news form stay facts until DK's own status for the player moves (then DK wins)
        kept = self.__dict__.setdefault("_calls", {}).get(team, {})
        merged = {n: c for n, (c, st0) in kept.items() if status_now.get(n) == st0}
        merged.update(calls or {})
        for n, call in merged.items():
            if call == "out":
                out.add(n); avail.discard(n); starters.discard(n)
            elif call in ("in", "back"):
                avail.add(n); out.discard(n)
            elif call == "starts":
                avail.add(n); out.discard(n); starters.add(n)
            elif call == "bench":
                starters.discard(n)
        return dict(out=out, available=avail, starters=starters)

    def note_call(self, team: str, player: str, call: str) -> None:
        """Remember a typed late-news call as a fact for the scenario pick, with DK's status at the time (see raw_facts)."""
        st = next(((p.status or "").upper() for p in self.players(team) if normalize_name(p.name) == player), "")
        self.__dict__.setdefault("_calls", {}).setdefault(team, {})[player] = (call, st)

    def scenario_list(self, team: str) -> list:
        out = []
        for sc in SC.list_scenarios(team):
            a = sc.get("assumes") or {}
            sc["assumes_text"] = ", ".join(f"{n.split()[-1]} {c}" for n, c in a.items()) or "no assumption (never picked by itself)"
            sc["holds"] = bool(a) and SC.holds(a, self.raw_facts(team))
            out.append(sc)
        return out

    def save_scenario(self, team: str, name: str) -> dict:
        statuses = {normalize_name(p.name): (p.status or "").upper() for p in self.players(team)}
        q = [n for n, st in statuses.items() if st in QUESTION_STATUSES]
        grid = None
        gp = ROT.drawn_path(team)
        if gp.exists():
            try:
                grid = json.loads(gp.read_text())
            except (OSError, ValueError):
                grid = None
        assumes = SC.assumptions(self.team_overrides(team), statuses, q, grid)
        return SC.save_scenario(team, name, self.team_overrides(team), assumes)

    def activate_scenario(self, team: str, slug: str) -> Optional[dict]:
        new = SC.activate(team, slug, self.local_overrides, [normalize_name(p.name) for p in self.players(team)])
        if new is None:
            return None
        self.local_overrides = new
        self.write_local()
        self.reproject(team)
        return SC.load_scenario(team, slug)

    def auto_scenarios(self, calls: Optional[dict] = None) -> list:
        """
        When the facts of a team with scenarios change (a refresh moved a status or the lineup, a late-news call), the one
        scenario whose assumptions all hold is activated. Returns the messages to flash. Two matches or none: said, not acted on.
        """
        msgs = []
        seen = self.__dict__.setdefault("_facts_seen", {})
        if not SC.SCENARIOS_DIR.is_dir():
            return msgs
        teams = {p.team for p in self.slate.players if p.team}
        for d in SC.SCENARIOS_DIR.iterdir():
            team = d.name
            if not d.is_dir() or team not in teams:
                continue
            scs = SC.list_scenarios(team)
            if not scs:
                continue
            facts = self.raw_facts(team, (calls or {}).get(team))
            key = (frozenset(facts["out"]), frozenset(facts["available"]), frozenset(facts["starters"]))
            if seen.get(team) == key:
                continue
            seen[team] = key
            matches = [sc for sc in scs if sc.get("assumes") and SC.holds(sc["assumes"], facts)]
            act = SC.active_slug(team)
            if len(matches) == 1:
                if matches[0]["slug"] != act:
                    self.activate_scenario(team, matches[0]["slug"])
                    msgs.append(f"{team}: the news matches scenario '{matches[0]['name']}' ({', '.join(f'{n.split()[-1]} {c}' for n, c in matches[0]['assumes'].items())}): activated")
            elif len(matches) > 1:
                msgs.append(f"{team}: {len(matches)} scenarios match the news ({', '.join(m['name'] for m in matches)}): pick one")
            elif act:
                cur = SC.load_scenario(team, act)
                if cur and cur.get("assumes") and not SC.holds(cur["assumes"], facts):
                    msgs.append(f"{team}: the active scenario '{cur['name']}' no longer matches the news; no saved scenario does")
        return msgs

    def mapped_summary(self) -> dict:
        teams = sorted({p.team for p in self.slate.players if p.team})
        states = {t: self.mapped_state(t) for t in teams}
        return dict(n=len(teams), n_mapped=sum(1 for v in states.values() if v["state"] == "mapped"),
                    changed=[t for t, v in states.items() if v["state"] == "changed"], unmapped=[t for t, v in states.items() if v["state"] == "unmapped"])

    def clear_rotation(self, team: str) -> int:
        """Clear locks: the drawn grid goes, and so does every typed minutes pin on the team (locked rows' pins and hand-typed ones alike),
        so the grid is the model's again. Rates, bands, starter calls and the sim / public overrides stay. Returns the pins lifted."""
        ROT.clear_drawn(team)
        n = 0
        for p in self.players(team):
            key = normalize_name(p.name)
            if override_of(self.overrides, key).get("minutes") is not None:
                self.set_override(key, minutes=None); n += 1
        self.reproject(team)
        return n

    def add_back(self, team: str, player: str) -> float:
        """A player the rotation left off the floor, back in by hand: a minutes pin at his history (at least ADD_BACK_MINUTES)."""
        n = normalize_name(player)
        curve = (self.rot.get(team) or {}).get("curve")
        hist = float(curve.loc[n, "minutes"]) if curve is not None and n in curve.index else 0.0
        mins = float(max(ADD_BACK_MINUTES, round(hist)))
        self.set_override(n, minutes=mins)
        self.reproject(team)
        return mins

    def starters_of(self, team: str) -> Optional[set]:
        """Rotowire's lineup with the hand starter overrides applied; None when there is no lineup and no override."""
        base = self.slate.starters.get(team)
        ov = {normalize_name(p.name): override_of(self.overrides, normalize_name(p.name)).get("starter") for p in self.players(team)}
        ov = {n: v for n, v in ov.items() if v is not None}
        if base is None and not ov:
            return None
        out = set(base or ())
        for n, v in ov.items():
            (out.add if float(v) > 0 else out.discard)(n)
        return out

    def priced_total(self, team: str, name: str) -> Optional[float]:
        """A priced player's projected total as it stands (his book total): kept when only his minutes change."""
        ov = override_of(self.overrides, name)
        if ov.get("minutes") is not None and ov.get("fppm") is not None:
            return float(ov["minutes"]) * float(ov["fppm"])
        r = self.proj.get(team)
        if r is None or name not in r.index or not bool(r.loc[name].get("priced")):
            return None
        return float(r.loc[name, "fpts_mean"])

    def pins(self) -> dict[str, float]:
        """player -> pinned minutes (the overrides' minutes column only)."""
        return {k: v["minutes"] for k, v in self.overrides.items() if v.get("minutes") is not None}

    def review(self, team: str) -> dict:
        """The rotation review (ivan.sports.nba.validation.review_team): provenance of every player's minutes and the flags."""
        if getattr(self, "_recent", None) is None or self._recent_stamp != self.pool_stamp:
            self._recent = recent_minutes(self.slate.box); self._recent_stamp = self.pool_stamp
        return review_team(self.slate, team, self.proj[team], self._recent, market_points(self.session), self.pins())

    def set_override(self, name: str, **fields) -> None:
        """Set or clear a player's pins in this machine's own layer (a value None clears that field here; on a consumer the
        published value then shows through); the merged view is refreshed. Persisted at once."""
        rec = dict(self.local_overrides.get(name, {}))
        for k, v in fields.items():
            rec[k] = v
        rec = {k: v for k, v in rec.items() if v is not None}
        if rec:
            self.local_overrides[name] = {k: rec.get(k) for k in OVERRIDE_KEYS}
        else:
            self.local_overrides.pop(name, None)
        self.write_local()

    def write_local(self) -> None:
        """Write this machine's own layer and refresh the merged view the projection reads."""
        write_overrides(self.local_overrides)
        self.overrides = read_overrides_full()
        self.__dict__.pop("_layers_cache", None); self.__dict__.pop("_published_cache", None)

    def _layer_of(self, name: str) -> dict:
        """{field: 'local' | 'published'} for the player's pins: what this machine typed vs what the producer published (a consumer)."""
        if not PUBLISHED_OVERRIDES_CSV.exists():
            return {}
        cache = self.__dict__.setdefault("_layers_cache", {})
        if name not in cache:
            pub, loc = override_layers(name)
            cache[name] = {**{k: "published" for k in pub}, **{k: "local" for k in loc}}
        return cache[name]

    def _published_of(self, name: str) -> dict:
        """The producer's published pins for the player (a consumer), {} elsewhere."""
        if not PUBLISHED_OVERRIDES_CSV.exists():
            return {}
        cache = self.__dict__.setdefault("_published_cache", {})
        if name not in cache:
            cache[name] = override_layers(name)[0]
        return cache[name]

    def team_rows(self, team: str) -> list[dict]:
        proj, hist = self.proj[team], self.hist_only[team]
        review_flags = {r["name"]: r["flags"] for r in self.review(team)["rows"]}
        state = self.slate.state
        starters = self.starters_of(team)
        confirmed = self.slate.starters_confirmed.get(team, False)
        mk = self.market_means()
        rows = []
        for p in self.players(team):
            name = normalize_name(p.name)
            s = state.loc[name] if name in state.index else None
            r = proj.loc[name] if name in proj.index else None
            h = hist.loc[name] if name in hist.index else None
            status = (p.status or "").upper()
            out = r is None
            nomin = out and name in ((self.rot.get(team) or {}).get("zero_minutes") or [])
            starter = None if starters is None else (name in starters)
            flags = []
            hist_min = float(s["min_ewm"]) * float(s.get("play_ewm", 1.0)) if s is not None else None
            if nomin:
                flags.append("no minutes in the rotation")
            elif out and status != "OUT":
                flags.append("scratched")
            if status == "OUT" and hist_min and hist_min >= ROTATION_MIN:
                flags.append(f"out, frees {hist_min:.0f} min")
            if status in ("Q", "GTD", "D") and hist_min and hist_min >= ROTATION_MIN:
                flags.append("questionable")
            if starter and s is not None and float(s["start_ewm"]) < 0.5:
                flags.append("new starter")
            if starter is False and s is not None and float(s["start_ewm"]) > 0.6 and not out:
                flags.append("benched")
            if r is not None and hist_min is not None and abs(float(r["minutes"]) - hist_min) >= MINUTES_DEV:
                flags.append(f"{float(r['minutes']) - hist_min:+.0f} min vs history")
            if r is not None and s is None:
                flags.append("no history")
            market = mk.get(name)
            if market and h is not None and s is not None and h["fpts_mean"] > 0:
                gap = market / float(h["fpts_mean"]) - 1
                if abs(gap) >= MARKET_GAP:
                    flags.append(f"market {gap:+.0%} vs history")
            if r is not None and not r.get("priced") and float(r["minutes"]) >= 20:
                flags.append("unpriced")
            if r is not None and r.get("minutes_p90") is not None and not pd.isna(r.get("minutes_p90")) and float(r["minutes_p90"]) > 48:
                flags.append(f"impossible band: p90 {float(r['minutes_p90']):.0f} min")
            if r is not None and float(r["fpts_mean"]) >= 8 and float(r["fpts_std"]) > 1.2 * float(r["fpts_mean"]):
                flags.append(f"distribution suspect: sd {float(r['fpts_std']):.0f} on a mean of {float(r['fpts_mean']):.0f}")
            flags += [f for f in review_flags.get(name, []) if not f.startswith(("questionable", "pinned"))]
            # the flags as cell colours: which column each one belongs to, and how loud
            cells = {"name": [], "min": [], "hist": [], "fpm": []}
            for f in flags:
                if f.startswith(("out", "scratched", "questionable")):
                    cells["name"].append(f)
                elif f.startswith(("new team", "no history")):
                    cells["hist"].append(f)
                elif f.startswith("market"):
                    cells["fpm"].append(f)
                else:
                    cells["min"].append(f)
            rotation = r is not None and float(r["minutes"]) >= ROTATION_MIN      # the deep bench's quirks are not worth a colour
            loud = {k: ("bad" if any(x.startswith(("dnp-prone", "deep bench", "out", "scratched", "impossible band", "distribution suspect")) for x in v)
                        else "warn" if v and (rotation or k == "name") else "") for k, v in cells.items()}
            sa = self.season.get(name, {})
            cur, last_season = sa.get("season"), sa.get("last")
            fpts = None if r is None else float(r["fpts_mean"])
            ov = override_of(self.overrides, name)
            mk = self.market_rows.get(name, {})
            stats = [] if r is None else [dict(stat=st, proj=float(r[f"proj_{st}"]), line=mk.get(st, {}).get("line"), mean=mk.get(st, {}).get("mean"),
                                              books=mk.get(st, {}).get("books"), odds=mk.get(st, {}).get("odds"), priced=st in mk) for st in STATS]
            bm = book_minutes({st: v.get("mean") for st, v in self.market_rows.get(name, {}).items()}, s, self.slate.hist.league_rates) if not out else None
            rows.append(dict(id=p.id, name=p.name, key=name, salary=p.salary, pos=p.position, status=status if status != "NONE" else "", nomin=nomin,
                             book_min=None if bm is None else round(bm["minutes"], 1), book_src=None if bm is None else bm["source"],
                             book_by=None if bm is None else bm["by_stat"],
                             starter=("S" if starter else ("" if starter is not None else "?")),
                             confirmed=confirmed, out=out,
                             hist_min=hist_min, minutes=None if r is None else float(r["minutes"]),
                             min_p10=None if r is None or pd.isna(r.get("minutes_p10", np.nan)) else float(r["minutes_p10"]),
                             min_p90=None if r is None or pd.isna(r.get("minutes_p90", np.nan)) else float(r["minutes_p90"]),
                             pinned=ov.get("minutes") is not None, pin=ov.get("minutes"), starter_pin=ov.get("starter"),
                             fppm_pin=ov.get("fppm"), low_pin=ov.get("min_low"), high_pin=ov.get("min_high"), stats=stats,
                             fpts_pin=ov.get("fpts"), pub_pin=ov.get("pub"), fpts_model=None if r is None else float(r.get("fpts_model", r["fpts_mean"])),
                             layer=self._layer_of(name), local=dict(self.local_overrides.get(name, {})), published_pins=self._published_of(name),
                             fpts=fpts, sd=None if r is None else float(r["fpts_std"]),
                             fpm=None if r is None else float(r["fpm"]),
                             season_games=cur["games"] if cur else 0, season_min=cur["minutes"] if cur else None, season_fpm=cur["fpm"] if cur else None,
                             last_games=last_season["games"] if last_season else 0, last_min=last_season["minutes"] if last_season else None,
                             last_fpm=last_season["fpm"] if last_season else None,
                             value=None if fpts is None or not p.salary else fpts / (p.salary / 1000.0),
                             own=None if p.ownership in (None, 0) else float(p.ownership), pub=getattr(p, "pub_fpts", None),
                             hist_fpts=None if h is None else float(h["fpts_mean"]), market=market,
                             dk_avg=p.dk_avg_fpts, priced=bool(r is not None and r.get("priced")), recent=self.recent.get(name, []),
                             team=p.team, opp=p.opponent,
                             usage=None if r is None else float(r["usage_mult"]), flags=flags,
                             cells={k: " · ".join(v) for k, v in cells.items()}, loud=loud))
        # starters first, then the players who are out, then the rest of the bench; minutes descending within each group
        rows.sort(key=lambda x: (0 if (x["starter"] == "S" and not x["out"]) else (1 if x["out"] else 2), -(x["minutes"] or 0), -(x["salary"] or 0)))
        return rows

    def team_card(self, team: str, now: datetime) -> dict:
        rows = self.team_rows(team)
        tip = self.tip(team)
        mins_to_tip = (tip - now).total_seconds() / 60 if tip else None
        starters = self.starters_of(team)
        confirmed = self.slate.starters_confirmed.get(team, False)
        alerts = []
        for r in rows:
            for f in r["flags"]:
                if f.startswith("out, frees") or f == "scratched":
                    alerts.append(("bad", f"{r['name']}: {f}"))
                elif f == "no minutes in the rotation":
                    if (r["hist_min"] or 0) >= ROTATION_MIN:
                        alerts.append(("warn", f"{r['name']}: {f} ({r['hist_min']:.0f} min history)"))
                elif f in ("new starter", "benched") or f.startswith("market"):
                    alerts.append(("warn", f"{r['name']}: {f}"))
                elif f == "questionable":
                    alerts.append(("warn", f"{r['name']}: questionable ({r['hist_min']:.0f} min)"))
                elif f.endswith("vs history") and r["minutes"] and r["minutes"] >= 12:
                    alerts.append(("info", f"{r['name']}: {f}"))
                elif f.startswith(("dnp-prone", "deep bench", "impossible band", "distribution suspect")):
                    alerts.append(("bad", f"{r['name']}: {f}"))
                elif f.startswith(("new team", "above recent", "market implies", "no history")) and r["minutes"] and r["minutes"] >= 10:
                    alerts.append(("warn", f"{r['name']}: {f}"))
        if starters is None:
            alerts.append(("warn" if (mins_to_tip is not None and mins_to_tip <= NEAR_TIP_MIN) else "info", "no lineup from Rotowire"))
        elif not confirmed and mins_to_tip is not None and mins_to_tip <= NEAR_TIP_MIN:
            alerts.append(("warn", "lineup still expected, not confirmed"))
        avail = [r for r in rows if not r["out"]]
        base_sum = sum((r["hist_min"] or 0) for r in avail)
        if abs(base_sum - 241) >= 25:
            alerts.append(("info", f"history minutes of the available roster sum to {base_sum:.0f} (team gets 241)"))
        rv = self.review(team)
        alerts += [(k, t) for k, t in rv["alerts"] if t.startswith("rotation depth")]
        unpriced = [r for r in avail if not r["priced"] and (r["minutes"] or 0) >= 20]
        if unpriced:
            alerts.append(("info", f"{len(unpriced)} rotation players unpriced"))
        score = sum({"bad": 10, "warn": 3, "info": 1}[k] for k, _ in alerts)
        team_fpts = float(self.proj[team]["fpts_mean"].sum()) if len(self.proj[team]) else 0.0
        assigned = sum((r["minutes"] or 0) for r in avail)
        to_place = (241.0 - assigned) if any(r["pinned"] for r in rows) else 0.0
        if abs(to_place) < 1.5:
            to_place = 0.0
        return dict(team=team, opp=next((p.opponent for p in self.players(team)), ""), tip=tip, mins_to_tip=mins_to_tip,
                    assigned=assigned, to_place=to_place, rotation_drawn=ROT.drawn_path(team).exists(), rotation=self.rotation(team), mapped=self.mapped_state(team),
                    questions=self.questions(team, now), scenarios=self.scenario_list(team),
                    published=dict(grid=ROT.drawn_source(team), pins=sum(1 for r in rows if "published" in r.get("layer", {}).values()),
                                   own=sum(1 for r in rows if "local" in r.get("layer", {}).values()), consumer=PUBLISHED_OVERRIDES_CSV.exists() or ROT.PUBLISHED_ROTATIONS_DIR.is_dir()),
                    last_game=self.game_grid(team),
                    locked=bool(mins_to_tip is not None and mins_to_tip < 0), alerts=alerts, score=score,
                    starters="confirmed" if confirmed else ("expected" if starters else "none"),
                    n_out=sum(1 for r in rows if r["out"]), n_pinned=sum(1 for r in rows if r["pinned"]),
                    team_fpts=team_fpts, trailing=self.slate.team_fpts.get(team), rows=rows, info={r["key"]: r for r in rows},
                    n_book=sum(1 for r in rows if r.get("book_min") is not None))

    def freshness(self, now: datetime) -> list:
        """When each input was last updated, read from disk on every page (so it is current even while the page is stale):
        [(label, time or None, minutes ago or None, warn after minutes, what it is)]."""
        from sqlalchemy import text as _t
        q = lambda sql: self.session.execute(_t(sql)).scalar()          # noqa: E731
        props = None
        pdir = NBA_DATA_DIR / "raw" / "odds" / "props"
        if pdir.is_dir():                                               # the producer: the newest stored props fetch
            mt = [f.stat().st_mtime for d in pdir.iterdir() if d.is_dir() for f in d.iterdir()]
            props = datetime.fromtimestamp(max(mt)) if mt else None
        if props is None:                                               # a consumer has no raw store: the books' newest update
            last = q("select max(last_update) from odds")
            if last:
                props = utc_to_local(last)
        dg = next((p.draft_group_id for p in self.slate.players if p.draft_group_id), None)
        dk = q(f"select max(captured_at) from archived_slate where slate_key = '{int(dg)}'") if dg else None
        rows = [("props", props, 90, "the books' player props, last fetched"),
                ("DraftKings", dk, 30, "the DK pool: salaries, statuses, FPPG"),
                ("starters", q("select max(fetched_at) from starting_lineups"), 30, "Rotowire lineups, last read"),
                ("projected", q("select max(fpts_updated_at) from nba_player"), 30, "the projection, last calculated")]
        if CONSUMER:                                                    # what the laptop last published, as pulled (the Refresh button pulls)
            fs = feed.status("nba")
            prod = fs.get("producer") or {}
            rows.append(("published", fs.get("published_at"), 60,
                         "the laptop's last publish of the feed" + (f" ({prod.get('source')}, autopilot {'on' if prod.get('autopilot') else 'off'})" if prod else "")
                         + (f"; last pulled here {str(fs.get('pulled_at'))[11:16]}" if fs.get("pulled_at") else "")))
        out = []
        for label, t, warn, what in rows:
            t = pd.Timestamp(t).to_pydatetime() if t is not None else None
            age = (now - t).total_seconds() / 60 if t is not None else None
            out.append(dict(label=label, at=t, age=age, warn=bool(age is not None and age > warn), what=what))
        return out

    def header(self, now: datetime) -> dict:
        ps = self.slate.players
        dg = next((p.draft_group_id for p in ps if p.draft_group_id), None)
        lock = min((p.gametime for p in ps if p.gametime), default=None)
        st = {}
        if STATE_FILE.exists():
            try:
                st = json.loads(STATE_FILE.read_text())
            except Exception:
                st = {}
        odds_t = self.session.execute(__import__("sqlalchemy").text("select max(last_update) from odds")).scalar()
        slates = dict(self.lobby_slates())
        for k, v in (st.get("slates") or {}).items():
            if k in slates:
                continue
            try:
                t = datetime.fromisoformat(str(v.get("lock")))
            except (TypeError, ValueError):
                continue
            slates[k] = dict(v, lock_dt=t, label=f"{t:%a %m-%d %H:%M} · " + str(v.get("name", "")).replace("NBA ", "")[:28])
        if dg is not None and str(dg) not in slates:          # the lobby drops a group near lock and the capture state may never have seen it
            slates[str(dg)] = dict(lock_dt=lock, label=(f"{lock:%a %m-%d %H:%M} · " if lock else "") + f"loaded slate {dg}")
        days = sorted({v["lock_dt"].date() for v in slates.values() if v.get("lock_dt") is not None})
        if days:                                                  # dated slates: today's only (the next day's when today has none); the lobby's
            ahead = [d for d in days if d >= now.date()]          # (already today's, undated) and the loaded slate always stay
            day = now.date() if now.date() in days else (ahead[0] if ahead else days[-1])
            slates = {k: v for k, v in slates.items() if v.get("lock_dt") is None or v["lock_dt"].date() == day or k == str(dg)}
        slates = dict(sorted(slates.items(), key=lambda kv: (kv[1].get("lock_dt") is None, kv[1].get("lock_dt") or now, kv[1]["label"])))
        return dict(draft_group=dg, lock=lock, mins_to_lock=(lock - now).total_seconds() / 60 if lock else None,
                    pool_at=self.pool_stamp[0], n_players=len(ps), n_priced=len(self.slate.market),
                    starters_at=self.slate.starters_fetched_at, odds_at=odds_t, stale=getattr(self, "stale", False), fresh=self.freshness(now),
                    slates=slates, slate_days=self.slate_days(now, dg), props_fetches=st.get("props_fetches", 0), mapped=self.mapped_summary(),
                    questions=self.questions_summary(now), news=self.news_status(now), team_states=self.team_states(now),
                    overrides=len(self.overrides), overrides_path=str(MINUTES_OVERRIDES_CSV.relative_to(PROJECT_ROOT)),
                    autopilot=None if CONSUMER else autopilot.state(), feed=feed.status("nba") if CONSUMER else None)


desk = Desk()


def _pool_vanished_handler():
    from sqlalchemy.orm.exc import ObjectDeletedError

    async def handler(request: Request, exc: ObjectDeletedError):
        with LOCK:
            desk.session.rollback(); desk.rebuild()
            desk.flash.append("the player pool was replaced on disk while the page was built; reloaded")
        return RedirectResponse(auth.root_of(request) + "/", status_code=303)
    app.add_exception_handler(ObjectDeletedError, handler)


_pool_vanished_handler()


def _busy(request: Request):
    """A refresh is replacing the pool: no edits or jobs until it is done (the board shows the state)."""
    if desk.busy:
        desk.flash.append(f"wait: {desk.busy}")
        return RedirectResponse(auth.root_of(request) + "/", status_code=303)
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
    return templates.TemplateResponse(request, template, {"now": datetime.now(), "flash": flash, "root": auth.root_of(request), "consumer": CONSUMER,
                                                          "limited": LIMITED, "limits": LIMITS, "job_max_minutes": JOB_MAX_MINUTES, **ctx})


def games_menu(now: datetime) -> list[dict]:
    """
    The slate's games for the menu without building every team's card: per team its tip, lineup state, mapped / question
    state (team_states, with the open questions as the chip's hover), paired into games with the line, grouped by tip.
    """
    states = {t["team"]: t for t in desk.team_states(now)}
    teams = {}
    for t in desk.slate.teams():
        tip = desk.tip(t); mtt = (tip - now).total_seconds() / 60 if tip else None
        st = desk.starters_of(t); confirmed = desk.slate.starters_confirmed.get(t, False)
        opp = next((p.opponent for p in desk.players(t)), "")
        qs = [q for q in desk.questions(t, now) if not q["answered"]]
        teams[t] = dict(team=t, opp=opp, tip=tip, mins_to_tip=mtt, locked=bool(mtt is not None and mtt < 0),
                        starters="confirmed" if confirmed else ("expected" if st else "none"),
                        qtext="; ".join(f"{q['raw']} {q['status']}" + (f" · {q['news']['headline']}" if q.get("news") else "") for q in qs),
                        **{k: states.get(t, {}).get(k) for k in ("colour", "why", "open", "new", "state")})
    seen, games = set(), []
    for t, c in sorted(teams.items(), key=lambda kv: (kv[1]["tip"] or now, kv[0])):
        if t in seen:
            continue
        opp = teams.get(c["opp"])
        pair = [c] + ([opp] if opp is not None else [])
        seen.update(x["team"] for x in pair)
        info = (desk.slate._lines.by_team.get(t) if getattr(desk.slate, "_lines", None) else None) or {}
        if len(pair) == 2 and info.get("home") is True:
            pair = [pair[1], pair[0]]
        home = pair[-1]["team"] if len(pair) == 2 else None
        spread = desk.slate.spread(home) if home else None
        total = desk.slate._lines.team_total(home) if home and getattr(desk.slate, "_lines", None) else None
        games.append(dict(teams=pair, tip=c["tip"], mins_to_tip=c["mins_to_tip"], locked=all(x["locked"] for x in pair), home=home, spread=spread, total=total))
    games.sort(key=lambda g: (g["locked"], g["tip"] or now))
    menu = []
    for g in games:
        if not menu or menu[-1]["tip"] != g["tip"]:
            menu.append(dict(tip=g["tip"], games=[]))
        menu[-1]["games"].append(g)
    return menu


def games_of(cards: list[dict], now: datetime) -> list[dict]:
    """Pair the team cards into games (away on the left, home on the right when the lines say which is which), earliest tip first."""
    by_team = {c["team"]: c for c in cards}
    seen, games = set(), []
    for c in cards:
        if c["team"] in seen:
            continue
        opp = by_team.get(c["opp"])
        pair = [c] + ([opp] if opp is not None else [])
        seen.update(x["team"] for x in pair)
        info = (desk.slate._lines.by_team.get(c["team"]) if getattr(desk.slate, "_lines", None) else None) or {}
        if len(pair) == 2 and info.get("home") is True:
            pair = [pair[1], pair[0]]                      # away first
        home = pair[-1]["team"] if len(pair) == 2 else None
        spread = desk.slate.spread(home) if home else None
        total = desk.slate._lines.team_total(home) if home and getattr(desk.slate, "_lines", None) else None
        games.append(dict(teams=pair, tip=c["tip"], mins_to_tip=c["mins_to_tip"], locked=all(x["locked"] for x in pair),
                          score=max(x["score"] for x in pair), home=home, spread=spread, total=total))
    return sorted(games, key=lambda g: (g["locked"], g["tip"] or now))


@app.get("/", response_class=HTMLResponse)
def index(request: Request, open: Optional[str] = None, view: str = "games"):
    """view=games: the team cards paired into games; view=players: every player in one table, highest ownership first."""
    with LOCK:
        if not desk.busy:
            desk.maybe_reload()
        now = datetime.now()
        return _render(request, "nba_desk.html", header=desk.header(now), open_team=open, view=view, busy=desk.busy)


@app.get("/board", response_class=HTMLResponse)
def board(request: Request, view: str = "games", set: str = "pool", metric: str = "proj", bins: str = "", team: str = ""):
    """The slate's board (the heavy part): the games menu, the pipeline panel and the cards / players / late swap view."""
    with LOCK:
        now = datetime.now()
        if desk.busy:
            return _render(request, "_nba_board.html", busy=desk.busy, view=view, games=[], menu=[], players=[], lateswap=None, exposures=None)
        for t in desk.slate.teams():
            desk.slate.spread(t)                            # loads the lines once
        menu = games_menu(now)                               # the games strip: tip, line, one chip per team (its state, the questions on hover)
        teams = desk.slate.teams()
        if team and team in teams:
            desk.open_team = team
        open_team = getattr(desk, "open_team", None)
        if open_team not in teams:
            open_team = next((c["team"] for grp in menu for g in grp["games"] for c in g["teams"]), teams[0] if teams else None)
            desk.open_team = open_team
        card = desk.team_card(open_team, now) if (view == "games" and open_team) else None
        game = next((g for grp in menu for g in grp["games"] if any(c["team"] == open_team for c in g["teams"])), None)
        cards = [desk.team_card(t, now) for t in teams] if view == "players" else []
        games = []
        players = []; lateswap = None; exposures = None
        if view == "lateswap":
            lateswap = _lateswap_ctx(now)
        if view == "exposures":
            exposures = _exposures_view(set, metric, int(bins) if str(bins).strip().isdigit() else None)
        if view == "players":
            players = [r for c in cards for r in c["rows"]]
            players.sort(key=lambda r: (-(r["own"] or 0), -(r["fpts"] or 0)))
        return _render(request, "_nba_board.html", busy=None, games=games, menu=menu, view=view, players=players, lateswap=lateswap, exposures=exposures,
                       card=card, game=game, open_team=open_team, **_pipeline_ctx())


def _exposures_full(mine: Optional[set] = None) -> Optional[dict]:
    """
    The exposures view: every player in the newest choose run that covered this slate's contests, joined to the pool as it
    stands now. Per player: entries holding him, exposure, the field's ownership and the leverage against it, the
    projection he was chosen on and ours now (the delta is what the news did to him since), minutes now, the public
    number, status (a chosen player now OUT is the first thing to see), value. Per team: the entries' exposure summed.
    """
    mine = _slate_contest_ids() if mine is None else mine
    for classic in sorted(OUTPUT_DIR.glob("chosen_entries.classic.*.csv"), reverse=True):
        stamp = classic.name.split(".")[2]
        lu = OUTPUT_DIR / f"chosen_entries.lineups.{stamp}.csv"
        if not lu.exists():
            continue
        try:
            L = pd.read_csv(lu)
        except Exception:   # noqa: BLE001
            continue
        L = L[L["contest_id"].astype(int).isin(mine)]
        if not len(L):
            continue
        n_entries = int(L["entry_id"].nunique())
        e = (L.groupby(["player_id"]).agg(name=("name", "first"), team=("team", "first"), position=("position", "first"), salary=("salary", "first"),
                                           n=("entry_id", "nunique"), proj_then=("projection", "first")).reset_index())
        now = {int(p.id): p for p in desk.session.query(NBAPlayer).all()}
        rows = []
        for r in e.itertuples():
            p = now.get(int(r.player_id))
            proj_now = None if p is None or p.fpts_mean is None else float(p.fpts_mean)
            own = None if p is None or p.ownership in (None, 0) else float(p.ownership)
            exp = float(r.n) / max(1, n_entries)
            rows.append(dict(player_id=int(r.player_id), name=r.name, team=r.team, position=r.position, salary=int(r.salary or 0), n=int(r.n), exposure=exp,
                             own=own, leverage=None if own is None else exp * 100 - own, status=((p.status or "").upper() if p and p.status and p.status != "None" else ""),
                             in_pool=p is not None, proj_then=None if pd.isna(r.proj_then) else float(r.proj_then), proj_now=proj_now,
                             delta=None if proj_now is None or pd.isna(r.proj_then) else proj_now - float(r.proj_then),
                             minutes=None if p is None or p.minutes_proj is None else float(p.minutes_proj),
                             pub=None if p is None or getattr(p, "pub_fpts", None) is None else float(p.pub_fpts),
                             value=None if proj_now is None or not r.salary else proj_now / (float(r.salary) / 1000.0)))
        rows.sort(key=lambda x: (-x["exposure"], -(x["proj_now"] or 0)))
        teams = {}
        for x in rows:
            t = teams.setdefault(x["team"], dict(team=x["team"], exposure=0.0, players=0, out=0))
            t["exposure"] += x["exposure"]; t["players"] += 1; t["out"] += 1 if x["status"] in ("OUT", "O", "IR", "SUSP") else 0
        return dict(stamp=stamp, when=datetime.strptime(stamp, "%Y%m%d-%H%M%S"), entries=n_entries, contests=int(L["contest_id"].nunique()), rows=rows,
                    teams=sorted(teams.values(), key=lambda t: -t["exposure"]), out=[x for x in rows if x["status"] in ("OUT", "O", "IR", "SUSP")],
                    questionable=[x for x in rows if x["status"] in QUESTION_STATUSES])
    return None


def _pool_exposures(top: int = 24) -> Optional[dict]:
    """The candidate pool's exposures: how many of the generated lineups hold each player, joined to the pool as it stands."""
    from ivan.core.utils import coerce_player_ids
    dg = next((p.draft_group_id for p in desk.slate.players if p.draft_group_id), None)
    q = desk.session.query(Lineup.player_ids_list)
    if dg is not None:
        q = q.filter((Lineup.draft_group_id == int(dg)) | (Lineup.draft_group_id.is_(None)))
    counts: dict = {}; n = 0
    for (ids,) in q.all():
        n += 1
        for pid in coerce_player_ids(ids):
            counts[int(pid)] = counts.get(int(pid), 0) + 1
    if not n:
        return None
    now = {int(p.id): p for p in desk.session.query(NBAPlayer).all()}
    rows = []
    for pid, k in counts.items():
        p = now.get(pid)
        if p is None:
            continue
        rows.append(dict(name=p.name, team=p.team, position=p.position, n=k, exposure=k / n, own=None if p.ownership in (None, 0) else float(p.ownership),
                         proj=None if p.fpts_mean is None else float(p.fpts_mean), status=((p.status or "").upper() if p.status and p.status != "None" else "")))
    rows.sort(key=lambda x: (-x["exposure"], -(x["proj"] or 0)))
    return dict(lineups=n, players=len(rows), rows=rows[:top])


def _side_ctx() -> dict:
    """The dashboard's side panel: the exposures of the candidate pool or of the chosen entries (desk.exp_mode), compact."""
    mode = getattr(desk, "exp_mode", "entries")
    if mode == "pool":
        return dict(mode=mode, pool=_pool_exposures(), entries=None)
    ex = _exposures_full()
    if ex:
        ex = dict(ex, rows=ex["rows"][:24])
    return dict(mode=mode, pool=None, entries=ex)


@app.post("/pipeline/exposures", response_class=HTMLResponse)
def pipeline_exposures(request: Request, mode: str = Form("entries")):
    """Flip the side panel between the candidate pool's exposures and the chosen entries'."""
    with LOCK:
        desk.exp_mode = "pool" if mode == "pool" else "entries"
        return _render(request, "_nba_pipeline_side.html", side=_side_ctx(), oob=False)


# ---------------------------------------------------------------- the exposures view: pool | field | entries, histograms, every player column
HIST_COLORS = {"field": "#2a78d6", "pool": "#eb6834", "entries": "#1baf7a"}   # the reference palette's first three slots (validated together)


def _field_sample() -> Optional[list]:
    """The sampled public field for the loaded pool, from its cache (refresh_ownership writes it with every pool refresh); None when absent."""
    from ivan.sports.nba import public_field as PF
    try:
        path = PF.FIELD_CACHE_DIR / f"{PF.field_key(desk.session)}.json"
        if not path.exists():
            return None
        return [list(x) for x in json.loads(path.read_text())]
    except Exception:   # noqa: BLE001
        return None


def _pool_lineups() -> list:
    from ivan.core.utils import coerce_player_ids
    dg = next((p.draft_group_id for p in desk.slate.players if p.draft_group_id), None)
    q = desk.session.query(Lineup.player_ids_list)
    if dg is not None:
        q = q.filter((Lineup.draft_group_id == int(dg)) | (Lineup.draft_group_id.is_(None)))
    return [coerce_player_ids(ids) for (ids,) in q.all()]


def _entries_lineups(contest: Optional[int] = None) -> Optional[dict]:
    """The chosen entries of the newest run covering this slate: {stamp, when, contests: [(id, name, n)], lineups: [[ids]], frame}."""
    mine = _slate_contest_ids()
    for classic in sorted(OUTPUT_DIR.glob("chosen_entries.classic.*.csv"), reverse=True):
        stamp = classic.name.split(".")[2]
        lu = OUTPUT_DIR / f"chosen_entries.lineups.{stamp}.csv"
        if not lu.exists():
            continue
        try:
            L = pd.read_csv(lu)
        except Exception:   # noqa: BLE001
            continue
        L = L[L["contest_id"].astype(int).isin(mine)]
        if not len(L):
            continue
        contests = [(int(cid), str(g["contest_name"].iloc[0]), int(g["entry_id"].nunique())) for cid, g in L.groupby("contest_id")]
        if contest is not None:
            L = L[L["contest_id"].astype(int) == int(contest)]
        lineups = [[int(x) for x in g["player_id"].tolist()] for _, g in L.groupby("entry_id")]
        return dict(stamp=stamp, when=datetime.strptime(stamp, "%Y%m%d-%H%M%S"), contests=contests, lineups=lineups, frame=L)
    return None


def _hist(series: dict, step: float, lo: Optional[float] = None, hi: Optional[float] = None) -> dict:
    """Shared bins over every series: {edges, series: {name: {shares, counts, n, mean}}}; shares are per-series so sizes compare."""
    vals = {k: np.asarray(v, dtype=float) for k, v in series.items() if v is not None and len(v)}
    if not vals:
        return dict(edges=[], series={})
    allv = np.concatenate(list(vals.values()))
    lo = float(np.floor((allv.min() if lo is None else lo) / step) * step); hi = float(np.ceil((allv.max() if hi is None else hi) / step) * step)
    if hi <= lo:
        hi = lo + step
    edges = np.arange(lo, hi + step / 2, step)
    out = {}
    for k, v in vals.items():
        c, _ = np.histogram(v, bins=edges)
        out[k] = dict(counts=[int(x) for x in c], shares=[float(x) / max(1, len(v)) for x in c], n=int(len(v)), mean=float(v.mean()), p90=float(np.percentile(v, 90)))
    return dict(edges=[float(e) for e in edges], series=out, ymax=max((max(d["shares"]) for d in out.values()), default=0.0))


def _sim_fields() -> dict:
    """The sampled field of every simmed contest of this slate, from the newest sim artifacts (unique lineups x copies): {contest_id: {name, lineups, dupes}}."""
    mine = _slate_contest_ids()
    if not mine:
        return {}
    names = {int(c.contest_id): str(c.name or "") for c in desk.session.query(HistoricalContest).filter(HistoricalContest.contest_id.in_(list(mine))).all()}
    base = OUTPUT_DIR / "sim"
    if not base.is_dir():
        return {}
    for d in sorted(base.iterdir(), reverse=True):
        found = {}
        for f in d.glob("contest_*.npz"):
            try:
                cid = int(f.stem.split("_")[1])
            except ValueError:
                continue
            if cid not in mine:
                continue
            try:
                z = np.load(f, allow_pickle=True)
                found[cid] = dict(name=names.get(cid, str(cid)), lineups=z["lineup_players"].astype(int), dupes=z["dupes"].astype(float), stamp=d.name)
            except Exception:   # noqa: BLE001
                continue
        if found:
            return found
    return {}


def _hist1(values: np.ndarray, weights: Optional[np.ndarray], step: float, bins: Optional[int] = None) -> dict:
    """One series: {edges, counts (weighted), shares, n, mean, p10, p50, p90}. `bins` fixes the number of equal bins over the range; else `step` sets the width."""
    if values is None or not len(values):
        return dict(edges=[], counts=[], shares=[], n=0)
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if bins:
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            hi = lo + step
        edges = np.linspace(lo, hi, int(bins) + 1)
    else:
        lo = float(np.floor(values.min() / step) * step); hi = float(np.ceil(values.max() / step) * step)
        if hi <= lo:
            hi = lo + step
        edges = np.arange(lo, hi + step / 2, step)
    c, _ = np.histogram(values, bins=edges, weights=w)
    tot = float(w.sum()) or 1.0
    order = np.argsort(values); cw = np.cumsum(w[order]) / tot
    q = lambda f: float(values[order][min(len(values) - 1, int(np.searchsorted(cw, f)))])   # noqa: E731
    return dict(edges=[float(e) for e in edges], counts=[float(x) for x in c], shares=[float(x) / tot for x in c], n=int(round(tot)),
                mean=float(np.average(values, weights=w)), p10=q(0.10), p50=q(0.50), p90=q(0.90), ymax=float(c.max() / tot) if len(c) else 0.0)


_FIELD_TARGET: dict = {}


def _field_target(metric: str) -> Optional[dict]:
    """
    The real fields the two-group sampler was fitted to (data/nba/derived/contests/<id>: DraftKings standings parsed at
    lock, one row per unique lineup with copies): per-lineup values for the histogram's target line. proj = the lineup's
    projected total as a share of that contest's best projected lineup (slates differ in scale; tonight's best rescales it),
    own = the sum of the eight players' real ownership, sal = salary used. Cached per process. None without derived contests.
    """
    if metric in _FIELD_TARGET:
        return _FIELD_TARGET[metric]
    from ivan.sports.nba.standings import DERIVED_DIR
    vals, wts, n_contests, dates = [], [], 0, []
    for d in sorted(DERIVED_DIR.glob("*")) if DERIVED_DIR.is_dir() else []:
        lp, pc = d / "lineups.parquet", d / "players.csv"
        if not lp.exists():
            continue
        try:
            lu = pd.read_parquet(lp)
            if metric == "proj":
                if "proj_sum" not in lu or lu.proj_sum.max() <= 0:
                    continue
                v = lu.proj_sum.values / float(lu.proj_sum.max())
            elif metric == "own":
                if not pc.exists():
                    continue
                pl = pd.read_csv(pc); own = dict(zip(pl["id"].astype(int), pd.to_numeric(pl["drafted_pct"], errors="coerce").fillna(0.0)))
                v = np.array([sum(own.get(int(getattr(r, f"p{j}")), 0.0) for j in range(1, 9)) for r in lu.itertuples()])
            else:
                if "salary" not in lu:
                    continue
                v = lu.salary.values.astype(float)
            w = lu.copies.values.astype(float); v = np.asarray(v, dtype=float)
            ok = np.isfinite(v)                                   # a lineup with an unprojected player has no total
            vals.append(v[ok]); wts.append(w[ok]); n_contests += 1
            try:
                dates.append(json.loads((d / "field.json").read_text()).get("capture_at", "")[:10])
            except Exception:   # noqa: BLE001
                pass
        except Exception:   # noqa: BLE001
            continue
    out = None if not vals else dict(values=np.concatenate(vals), weights=np.concatenate(wts), contests=n_contests,
                                     span=(min(x for x in dates if x), max(x for x in dates if x)) if any(dates) else None)
    _FIELD_TARGET[metric] = out
    return out


def _target_line(target: dict, edges: list, scale: float = 1.0) -> Optional[dict]:
    """The target's share per bin of the histogram's own edges (mass outside the edges is kept in the denominator)."""
    if not target or not edges:
        return None
    v = target["values"] * scale; w = target["weights"]
    c, _ = np.histogram(v, bins=np.asarray(edges), weights=w)
    tot = float(w.sum()) or 1.0
    return dict(shares=[float(x) / tot for x in c], contests=target["contests"], span=target["span"], inside=float(c.sum() / tot))


def _exposures_view(set_key: str = "pool", metric: str = "proj", bins: Optional[int] = None) -> dict:
    """
    The exposures screen: one set of lineups in view (the candidate pool, a simmed contest's sampled field, or the chosen
    entries of one contest or all), one histogram of a per-lineup metric over it (projected total under our projection,
    ownership sum under the field's ownership, or salary used), and every player in the set with the team card's full row
    plus his exposure in the set, ownership and leverage (and for entries the projection at pick with the delta).
    """
    players = {int(p.id): p for p in desk.session.query(NBAPlayer).all()}
    means = {i: float(p.fpts_mean or 0.0) for i, p in players.items()}
    owns = {i: float(p.ownership or 0.0) for i, p in players.items()}
    sals = {i: float(p.salary or 0.0) for i, p in players.items()}
    fields = _sim_fields()
    ent = _entries_lineups(None)
    options = [dict(key="pool", label=f"candidate pool · {len(_pool_lineups())} lineups", group="pool")]
    if fields:
        options.append(dict(key="field", label=f"sampled fields · all {len(fields)} contests · {int(round(sum(f['dupes'].sum() for f in fields.values())))} entries", group="field"))
    for cid, f in fields.items():
        options.append(dict(key=f"field:{cid}", label=f"sampled field · {f['name'][:46]} · {int(round(f['dupes'].sum()))} entries", group="field"))
    if ent:
        options.append(dict(key="entries", label=f"chosen entries · all contests · {len(ent['lineups'])} entries", group="entries"))
        for cid, name, k in ent["contests"]:
            options.append(dict(key=f"entries:{cid}", label=f"chosen entries · {name[:46]} · {k} entries", group="entries"))
    keys = {o["key"] for o in options}
    if set_key not in keys:
        set_key = "pool"
    weights = None; label = next(o["label"] for o in options if o["key"] == set_key); then = {}
    if set_key == "pool":
        lineups = [np.asarray(x, dtype=int) for x in _pool_lineups()]
    elif set_key == "field":
        lineups = [row for f in fields.values() for row in f["lineups"]]; weights = np.concatenate([f["dupes"] for f in fields.values()])
    elif set_key.startswith("field:"):
        f = fields[int(set_key.split(":")[1])]
        lineups = [row for row in f["lineups"]]; weights = f["dupes"]
    else:
        cid = int(set_key.split(":")[1]) if ":" in set_key else None
        e = _entries_lineups(cid) if cid is not None else ent
        lineups = [np.asarray(x, dtype=int) for x in (e["lineups"] if e else [])]
        if e is not None:
            then = e["frame"].groupby("player_id")["projection"].first().to_dict()
    n = len(lineups)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    metric = metric if metric in ("proj", "own", "sal") else "proj"
    per = {"proj": (means, 5.0, "projected total per lineup (our projection)", "fpts"),
           "own": (owns, 20.0, "ownership sum per lineup (the field's ownership)", "%"),
           "sal": (sals, 500.0, "salary used per lineup", "$")}
    table, step, title, unit = per[metric]
    vals = np.array([sum(table.get(int(i), 0.0) for i in lu) for lu in lineups], dtype=float)
    bins = int(bins) if bins and 4 <= int(bins) <= 80 else None
    hist = _hist1(vals, w, step, bins=bins) if n else dict(edges=[], counts=[], shares=[], n=0)
    # the dotted target over a sampled field: the real fields' shape, projected totals rescaled to tonight's best lineup
    target = None
    if set_key.startswith("field") and hist.get("edges"):
        tg = _field_target(metric)
        if tg:
            anchor = 1.0
            if metric == "proj":
                pool_sums = [sum(means.get(int(i), 0.0) for i in lu) for lu in _pool_lineups()]
                anchor = float(max(pool_sums)) if pool_sums else float(vals.max())
            target = _target_line(tg, hist["edges"], anchor)
            if target:
                hist["ymax"] = max(float(hist.get("ymax") or 0.0), max(target["shares"], default=0.0))
    counts: dict = {}
    for lu, wt in zip(lineups, w):
        for i in lu:
            counts[int(i)] = counts.get(int(i), 0.0) + float(wt)
    tot = float(w.sum()) or 1.0
    by_id = {}
    for t in sorted({p.team for p in players.values() if p.team}):
        if t not in desk.proj:
            desk.reproject(t)
        for r in desk.team_rows(t):
            by_id[int(r["id"])] = r
    rows = []
    for pid, k in counts.items():
        r = by_id.get(pid)
        if r is None:
            continue
        exp = k / tot
        x = dict(r); x.update(n_in=int(round(k)), exposure=exp, leverage=None if r.get("own") is None else exp * 100 - r["own"],
                             proj_then=None if pid not in then or pd.isna(then[pid]) else float(then[pid]))
        x["delta"] = None if x["proj_then"] is None or r.get("fpts") is None else r["fpts"] - x["proj_then"]
        rows.append(x)
    rows.sort(key=lambda x: (-x["exposure"], -(x["fpts"] or 0)))
    return dict(set=set_key, metric=metric, bins=bins, options=options, label=label, n=int(round(tot)) if n else 0, rows=rows, hist=hist, title=title, unit=unit, target=target,
                is_entries=set_key.startswith("entries"), teams=sorted({x["team"] for x in rows}),
                color=HIST_COLORS["entries" if set_key.startswith("entries") else ("field" if set_key.startswith("field") else "pool")])


def _start_restore(draft_group: int) -> None:
    """Load a past slate from its archive capture in a background thread (ingest.restore_from_archive): nothing fetched."""
    def run():
        log = []
        sess = get_session(engine)
        try:
            info = restore_from_archive(sess, int(draft_group), n_sims=1000, log=log.append)
            msg = f"slate {draft_group} restored from its {info['source']} capture of {info['captured_at']:%m-%d %H:%M} ({info['players']} players, {info['odds']} prop rows)"
        except IngestError as e:
            msg = f"restore failed: {e}"
        except Exception as e:   # noqa: BLE001
            msg = f"restore failed, the previous pool is unchanged: {type(e).__name__}: {e}"
        finally:
            sess.close()
        with LOCK:
            desk.rebuild(); desk.busy = None; desk.flash.append(msg)
    desk.busy = f"restoring slate {draft_group} from the archive"
    threading.Thread(target=run, name="ivan-restore", daemon=True).start()


def _start_refresh(what: str, draft_group: int, update_odds: bool) -> None:
    """Fetch, store, project and refresh the public side in a background thread with its own session (one transaction);
    the board shows `what` until it is done, then the desk rebuilds from the new pool."""
    def run():
        log = []
        sess = get_session(engine)
        try:
            refresh_pool(sess, int(draft_group), update_odds=update_odds, starters=True, n_sims=1000, log=log.append)
            msg = f"{what}: done"
        except IngestError as e:
            msg = f"{what} failed: {e}"
        except Exception as e:   # noqa: BLE001 - the transaction rolled back: the old pool stands
            msg = f"{what} failed, the previous pool is unchanged: {type(e).__name__}: {e}"
        finally:
            sess.close()
        with LOCK:
            desk.rebuild(); desk.busy = None; desk.flash.append(msg)
    desk.busy = what
    threading.Thread(target=run, name="ivan-refresh", daemon=True).start()


@app.get("/team/{team}", response_class=HTMLResponse)
def team(request: Request, team: str):
    with LOCK:
        desk.maybe_reload()
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


def _num(raw: str, label: str) -> Optional[float]:
    raw = (raw or "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        desk.flash.append(f"'{raw}' is not a number for {label}")
        return None


@app.post("/team/{team}/pin", response_class=HTMLResponse)
def pin(request: Request, team: str, player: str = Form(...), minutes: str = Form(""), fppm: str = Form(""), low: str = Form(""), high: str = Form(""),
        fpts: str = Form(""), pub: str = Form("")):
    """One row's hand values: blank clears that field. Minutes 0 scratches."""
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        key = normalize_name(player)
        lo, hi = _num(low, "low"), _num(high, "high")
        if (lo is None) != (hi is None):
            # one side typed: the other side keeps its edit, else the model's current band
            r = desk.proj[team].loc[key] if key in desk.proj.get(team, pd.DataFrame()).index else None
            ov = desk.overrides.get(key, {})
            if lo is None:
                lo = ov.get("min_low") if ov.get("min_low") is not None else (None if r is None or pd.isna(r.get("minutes_p10", np.nan)) else round(float(r["minutes_p10"])))
            if hi is None:
                hi = ov.get("min_high") if ov.get("min_high") is not None else (None if r is None or pd.isna(r.get("minutes_p90", np.nan)) else round(float(r["minutes_p90"])))
        if lo is not None and hi is not None and hi <= lo:
            desk.flash.append(f"band {lo:g}–{hi:g} ignored: low must be below high"); lo = hi = None
        new_min, new_fppm = _num(minutes, "minutes"), _num(fppm, "fppm")
        old = override_of(desk.overrides, key)
        if desk.priced_total(team, key) and new_fppm is not None and new_fppm != old.get("fppm"):
            desk.flash.append(f"{key}: fppm overridden on a priced player; the books' total no longer holds for him")
        new_fpts, new_pub = _num(fpts, "fpts"), _num(pub, "pub")
        if new_min is not None and new_min != old.get("minutes"):
            # typed minutes repaint the row: the lock (if any) is lifted so the skeleton re-fills him to the number, and the
            # other free rows move around him; a row you still want held exactly, paint it again
            g = ROT.load_drawn(team)
            if g is not None and key in set(g.attrs.get("locked", [])):
                ROT.save_drawn(team, g, locked=[n for n in g.attrs.get("locked", []) if n != key], fingerprint=g.attrs.get("fingerprint"))
        desk.set_override(key, minutes=new_min, fppm=new_fppm, min_low=lo, min_high=hi, fpts=new_fpts, pub=new_pub)
        desk.reproject(team)                                  # the solve re-balances the team around the edit; the grid is rebuilt to the pins
        if new_pub != old.get("pub"):
            # the public number moved: the field and ownership are built on it, so they are rebuilt now (a few seconds)
            try:
                t0 = time.time()
                refresh_ownership(desk.session, log=lambda m: None)
                desk.flash.append(f"{key}: public projection {'set to %g' % new_pub if new_pub is not None else 'back to the replica'}; field and ownership rebuilt in {time.time() - t0:.0f}s")
            except Exception as e:   # noqa: BLE001
                desk.flash.append(f"{key}: public projection saved, but the field did not rebuild: {type(e).__name__}: {e}")
            desk.rebuild()
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/refigure", response_class=HTMLResponse)
async def refigure(request: Request, team: str):
    """
    Body: JSON {rows: [{player, minutes, fppm, fpts, pub}, ...]}: every typed value of the team's grid, applied in one go (blank
    clears that pin), then the team re-projected once: the skeleton re-fills the grid to the typed minutes (a row whose minutes
    changed loses its lock so it can move), the other free rows move around them. The public numbers that changed rebuild the field.
    """
    if (b := _busy(request)) is not None:
        return b
    body = await request.json()
    with LOCK:
        changed_min, changed_pub, n_set = [], 0, 0
        for r in body.get("rows") or []:
            key = normalize_name(str(r.get("player") or ""))
            if not key:
                continue
            old = override_of(desk.overrides, key)
            vals = {}
            for k in ("minutes", "fppm", "fpts", "pub"):
                if k in r:
                    try:
                        vals[k] = _num(r.get(k), k)
                    except ValueError as e:
                        desk.flash.append(str(e)); continue
            if not vals:
                continue
            if "minutes" in vals and vals["minutes"] != old.get("minutes"):
                changed_min.append(key)
            if "pub" in vals and vals["pub"] != old.get("pub"):
                changed_pub += 1
            if any(vals.get(k) != old.get(k) for k in vals):
                n_set += 1
            desk.set_override(key, **vals)
        if changed_min:
            g = ROT.load_drawn(team)
            if g is not None:
                kept = [n for n in g.attrs.get("locked", []) if n not in changed_min]
                if len(kept) != len(g.attrs.get("locked", [])):
                    ROT.save_drawn(team, g, locked=kept, fingerprint=g.attrs.get("fingerprint"))
        desk.reproject(team)
        msg = f"{team}: {n_set} value(s) applied, the grid refigured" + (f" ({len(changed_min)} minutes changed)" if changed_min else "")
        if changed_pub:
            try:
                t0 = time.time(); refresh_ownership(desk.session, log=lambda m: None)
                msg += f"; {changed_pub} public number(s) changed: field and ownership rebuilt in {time.time() - t0:.0f}s"
            except Exception as e:   # noqa: BLE001
                msg += f"; the field did not rebuild: {type(e).__name__}: {e}"
            desk.rebuild()
        desk.flash.append(msg)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/starter", response_class=HTMLResponse)
def starter_toggle(request: Request, team: str, player: str = Form(...)):
    """Click the st cell: a bench player becomes a starter, a starter goes to the bench; a second click on a hand call clears it
    back to Rotowire's lineup. Saved with the other edits; the team re-projects (the start shift and the rotation proposal move)."""
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        key = normalize_name(player)
        base = desk.slate.starters.get(team)
        rw = None if base is None else (key in base)
        cur = override_of(desk.overrides, key).get("starter")
        if cur is not None:
            new = None                                           # a hand call already: clear it
        else:
            new = 0.0 if rw else 1.0
        desk.set_override(key, starter=new)
        desk.reproject(team)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.get("/team/{team}/rotation", response_class=HTMLResponse)
def rotation_view(request: Request, team: str):
    with LOCK:
        return _render(request, "_nba_rotation.html", rot=desk.rotation(team))


@app.post("/team/{team}/rotation", response_class=HTMLResponse)
async def rotation_apply(request: Request, team: str):
    """Body: JSON {players: [...], grid: [[0/1 x 48], ...], locked: [...]} from the desk's grid; the locked rows are held, the rest re-filled."""
    body = await request.json()
    with LOCK:
        try:
            n = desk.apply_rotation(team, body["players"], body["grid"], locked=body.get("locked"), note=str(body.get("note", "")))
            desk.flash.append(f"{team}: rotation applied, {n} row(s) locked; the rest filled around them")
        except Exception as e:   # noqa: BLE001
            desk.flash.append(f"{team}: rotation not applied: {type(e).__name__}: {e}")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/rotation/propose", response_class=HTMLResponse)
def rotation_propose(request: Request, team: str, source: str = Form("propose"), game_id: str = Form("")):
    """source=propose: tonight's grid with every lock lifted (the top panel, nothing saved); source=last: a played game for the panel under it."""
    with LOCK:
        if source == "last":
            return _render(request, "_nba_rotation.html", rot=desk.game_grid(team, game_id=game_id.strip() or None))
        return _render(request, "_nba_rotation.html", rot=desk.rotation(team, source="propose"))


@app.post("/team/{team}/rotation/add", response_class=HTMLResponse)
def rotation_add(request: Request, team: str, player: str = Form(...)):
    """A player the rotation left off the floor, back in: a minutes pin at his history; the grid schedules him."""
    with LOCK:
        mins = desk.add_back(team, player)
        desk.flash.append(f"{team}: {player} added back at {mins:.0f} minutes (type another number to change it)")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/rotation/drop", response_class=HTMLResponse)
def rotation_drop(request: Request, team: str, player: str = Form(...)):
    """No minutes tonight: the player leaves the grid and the pool (a minutes pin of 0; the + beside his name brings him back)."""
    with LOCK:
        desk.set_override(normalize_name(player), minutes=0.0)
        desk.reproject(team)
        desk.flash.append(f"{team}: {player} dropped from the grid (no minutes tonight)")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/rotation/restore", response_class=HTMLResponse)
def rotation_restore(request: Request, team: str, player: str = Form(...)):
    """Lift a hand drop: the minutes pin goes and the model fills him again."""
    with LOCK:
        desk.set_override(normalize_name(player), minutes=None)
        desk.reproject(team)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/rotation/clear", response_class=HTMLResponse)
def rotation_clear(request: Request, team: str):
    with LOCK:
        n = desk.clear_rotation(team)
        back = "the published grid and pins" if ROT.drawn_source(team) == "published" or PUBLISHED_OVERRIDES_CSV.exists() else "the model's"
        desk.flash.append(f"{team}: locks cleared" + (f", {n} typed minutes lifted" if n else "") + f"; the grid is {back}")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/clear", response_class=HTMLResponse)
def clear_team(request: Request, team: str):
    with LOCK:
        for p in desk.players(team):
            desk.overrides.pop(normalize_name(p.name), None)
        desk.write_local()
        desk.reproject(team)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/publish")
def publish(request: Request):
    """Write the pins, project every team at full sims, store projections + a capture row set."""
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        desk.write_local()
        try:
            info = project_slate(desk.session, n_sims=2000, log=desk.flash.append)
            desk.flash.append(f"published: {info.players} players projected with {len(desk.overrides)} edits; edits saved to {MINUTES_OVERRIDES_CSV.name}")
            autopilot.touch("publish")                       # the edits are published: the autopilot tick may run again
            _push_feed(desk.flash.append)
        except IngestError as e:
            desk.flash.append(f"publish failed: {e}")
        desk.rebuild()
    return RedirectResponse(auth.root_of(request) + "/", status_code=303)


def _push_feed(log) -> None:
    """The laptop's projections to the server: publish the live bundle and rsync it (IVAN_FEED_PUSH); nothing on a consumer or when unset."""
    if CONSUMER or not os.environ.get(feed.PUSH_ENV, "").strip():
        return
    try:
        feed.publish("nba", which="live", log=lambda m: None, producer=autopilot.producer_info("desk"))
        feed.push("nba", log=log)
    except Exception as e:   # noqa: BLE001 - the publish itself succeeded; the push is best effort
        log(f"feed push failed: {type(e).__name__}: {e}")


def _news_check(log) -> dict:
    sess = get_session(engine)
    try:
        r = NF.check(sess, log=log)
        pool = {p.name for p in desk.slate.players}
        on_slate = [x for x in r["new"] if x.player_name in pool]
        desk.news_checked_at = r["fetched_at"]
        return dict(new=len(r["new"]), on_slate=on_slate, total=r["total"])
    finally:
        sess.close()


@app.post("/news/check")
def news_check(request: Request):
    """Fetch Rotowire's news page once, store the new headlines, say which are about this slate's players."""
    with LOCK:
        try:
            r = _news_check(desk.flash.append)
            desk.flash.append(f"news: {r['new']} new headline(s) of {r['total']}" + (": " + "; ".join(f"{x.player_raw} · {x.headline}" for x in r["on_slate"][:6]) if r["on_slate"] else "; none on this slate"))
        except Exception as e:   # noqa: BLE001
            desk.flash.append(f"news check failed: {type(e).__name__}: {e}")
    return RedirectResponse(auth.root_of(request) + "/", status_code=303)


@app.post("/news/auto")
def news_auto(request: Request, on: str = Form("")):
    """Auto-check every NEWS_AUTO_SECONDS while the desk runs (off by default; nothing else fetches on its own)."""
    with LOCK:
        desk.news_auto = bool(on)
        desk.flash.append("news auto-check " + ("on: every 5 minutes while the desk runs" if desk.news_auto else "off"))
    return RedirectResponse(auth.root_of(request) + "/", status_code=303)


@app.post("/autopilot")
def autopilot_toggle(request: Request, on: str = Form("")):
    """The autopilot switch (ivan.sports.nba.autopilot; scripts/nba/autopilot.py on|off from a terminal): with it on, the
    capture tick refreshes the loaded slate every 15 minutes, fetches props at the offsets and publishes + pushes the feed,
    standing down while this desk is being worked. Nothing on a consumer."""
    if CONSUMER:
        return RedirectResponse(auth.root_of(request) + "/", status_code=303)
    with LOCK:
        st = autopilot.set_on(bool(on), by="desk")
        desk.flash.append("autopilot " + ("on: the tick refreshes, fetches props at the offsets and publishes every 15 minutes while nobody is editing here"
                                          if st["on"] else "off: only Publish sends projections to the other desk"))
    return RedirectResponse(auth.root_of(request) + "/", status_code=303)


def _news_loop():
    while True:
        time.sleep(NEWS_AUTO_SECONDS)
        if not getattr(desk, "news_auto", False):
            continue
        try:
            with LOCK:
                r = _news_check(lambda m: None)
                if r["on_slate"]:
                    desk.flash.append(f"news: {len(r['on_slate'])} new on this slate: " + "; ".join(f"{x.player_raw} · {x.headline}" for x in r["on_slate"][:6]))
        except Exception:   # noqa: BLE001 - the next tick tries again
            pass


threading.Thread(target=_news_loop, name="ivan-news", daemon=True).start()


@app.post("/team/{team}/scenario/save", response_class=HTMLResponse)
def scenario_save(request: Request, team: str, name: str = Form(...)):
    """Save the team's current edits (pins + grid) under a name; the assumptions are read off the calls in them."""
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        name = name.strip()[:60]
        if name:
            sc = desk.save_scenario(team, name)
            desk.flash.append(f"{team}: scenario '{sc['name']}' saved" + (f" (assumes {', '.join(f'{n.split()[-1]} {c}' for n, c in sc['assumes'].items())})" if sc["assumes"] else " (no assumption: it will not pick itself)"))
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/scenario/activate", response_class=HTMLResponse)
def scenario_activate(request: Request, team: str, slug: str = Form(...)):
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        sc = desk.activate_scenario(team, slug)
        desk.flash.append(f"{team}: scenario '{sc['name']}' active" if sc else f"{team}: no scenario {slug}")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/scenario/delete", response_class=HTMLResponse)
def scenario_delete(request: Request, team: str, slug: str = Form(...)):
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        SC.delete_scenario(team, slug)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/reload")
def reload(request: Request):
    """Load what is on disk without fetching anything: the pool another process wrote (scripts/nba/ingest.py or project.py
    in a terminal, a feed pull), the pins file, the lineups. The "new data" badge posts here."""
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        desk.rebuild()
        desk.flash.append("reloaded from disk")
    return RedirectResponse(auth.root_of(request) + "/", status_code=303)


@app.post("/refresh")
def refresh(request: Request, what: str = Form("dk")):
    if CONSUMER and what == "props":
        what = "dk"                                              # a consumer has no props button; the laptop's capture loop fetches them
    """dk: pool + statuses + Rotowire (free). props: also the player props (costs credits)."""
    with LOCK:
        feed_url = os.environ.get("IVAN_FEED_URL", "").strip()
        dg = next((p.draft_group_id for p in desk.slate.players if p.draft_group_id), None)
        if feed_url:                                             # a consumer never fetches: it takes what the producer pushed
            try:
                from ivan.core import feed
                changed = feed.pull(feed_url, os.environ.get("IVAN_FEED_TOKEN"), "nba", NBA_DATA_DIR.parent / "feed-state", log=desk.flash.append)
                if changed:
                    mine = locally_edited_teams(desk.session)
                    if mine:
                        project_slate(desk.session, n_sims=2000, log=desk.flash.append, teams=mine)
                        desk.flash.append(f"the published projection is in; your own edits re-projected on top for {', '.join(sorted(mine))}")
                    else:
                        desk.flash.append("the published projection is in, as published")
                else:
                    fs = feed.status("nba")
                    prod = fs.get("producer") or {}
                    when = str(fs.get("published_at") or "")[11:16]
                    desk.flash.append("nothing new: the laptop has not published since " + (when or "the last pull")
                                      + (f" ({prod.get('source')}; autopilot {'on' if prod.get('autopilot') else 'off'})" if prod else ""))
            except Exception as e:   # noqa: BLE001
                desk.flash.append(f"feed pull failed: {type(e).__name__}: {e}")
        elif dg is None:
            desk.flash.append("no draft group in the pool; pick a slate first")
        elif runner.active():
            desk.flash.append("a job is running; it reads the pool, so refresh after it finishes (or cancel it)")
        elif desk.busy:
            desk.flash.append(f"already {desk.busy}")
        else:
            _start_refresh("refreshing the pool" + (" and the props" if what == "props" else ""), dg, update_odds=(what == "props"))
            return RedirectResponse(auth.root_of(request) + "/", status_code=303)
        desk.rebuild()
    return RedirectResponse(auth.root_of(request) + "/", status_code=303)


# ---------------------------------------------------------------- pipeline: generate / sim / choose as jobs
def _entries_path() -> Path:
    try:
        from ivan.core.config import load_config
        return Path(load_config("nba")["paths"]["entries_csv"])
    except Exception:   # noqa: BLE001
        return INPUT_DIR / "entries.csv"


def _entries_info() -> dict:
    p = _entries_path()
    if not p.exists():
        return dict(path=str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p), exists=False)
    try:
        from ivan.sports.nba.lineup_export import read_entries
        e = read_entries(p)
        contests = e.groupby("Contest Name").size().to_dict()
        return dict(path=str(p.relative_to(PROJECT_ROOT)), exists=True, entries=int(len(e)), contests=contests,
                    modified=datetime.fromtimestamp(p.stat().st_mtime))
    except Exception as ex:   # noqa: BLE001
        return dict(path=str(p), exists=True, error=f"{type(ex).__name__}: {ex}")


def _slate_contest_ids() -> set[int]:
    """The contests of the draft group loaded in the desk (historical_contest_information)."""
    dg = next((p.draft_group_id for p in desk.slate.players if p.draft_group_id), None)
    if dg is None:
        return set()
    from ivan.sports.nba.models import HistoricalContest
    return {int(c) for (c,) in desk.session.query(HistoricalContest.contest_id).filter(HistoricalContest.draft_group_id == int(dg)).all()}


def _latest_outputs() -> Optional[dict]:
    """
    The newest choose run that covered a contest of the current slate: the upload CSV, the per-contest
    summary from its choose json, and the exposures recomputed over this slate's entries only.
    """
    mine = _slate_contest_ids()
    for classic in sorted(OUTPUT_DIR.glob("chosen_entries.classic.*.csv"), reverse=True):
        stamp = classic.name.split(".")[2]
        js = OUTPUT_DIR / f"chosen_entries.choose.{stamp}.json"
        if not js.exists():
            continue
        try:
            d = json.loads(js.read_text())
        except ValueError:
            continue
        hit = {cid: v for cid, v in d.items() if int(cid) in mine}
        if not hit:
            continue
        out = dict(stamp=stamp, classic=classic.name, when=datetime.strptime(stamp, "%Y%m%d-%H%M%S"), slate_only=len(hit) < len(d))
        try:
            out["contests"] = [dict(name=v.get("contest_name"), entries=v.get("entries"), chosen=v.get("chosen"), roi=v.get("chosen_roi"), cash=v.get("chosen_cash"),
                                    top_roi=v.get("top_quality_roi"), book=(v.get("book") or {}).get("chosen") or {}) for v in hit.values()]
            up = pd.read_csv(classic); up = up[up["Contest ID"].astype(int).isin(mine)]
            out["rows"] = int(len(up))
            lu = OUTPUT_DIR / f"chosen_entries.lineups.{stamp}.csv"
            if lu.exists() and len(up):
                L = pd.read_csv(lu); L = L[L["contest_id"].astype(int).isin(mine)]
                n_entries = L["entry_id"].nunique()
                e = (L.groupby(["player_id", "name", "team", "position", "salary"]).agg(n=("entry_id", "nunique"), projection=("projection", "first"))
                     .reset_index())
                e["exposure"] = e["n"] / max(1, n_entries)
                out["exposures"] = e.sort_values("exposure", ascending=False).head(14)[["name", "team", "position", "salary", "exposure", "projection"]].to_dict("records")
        except Exception as ex:   # noqa: BLE001
            out["error"] = f"{type(ex).__name__}: {ex}"
        return out
    return None


def _lateswap_ctx(now: datetime) -> dict:
    """The late swap tab: our entries with each slot's lock state and projection, the lock clock, the newest report."""
    from ivan.sports.nba.lineup_export import POSITION_ORDER, read_entries
    from ivan.sports.nba.post_lock import lateswap as LS
    byid = {int(p.id): p for p in desk.slate.players}
    proj = {}
    for t, df in desk.proj.items():
        for name, r in df.iterrows():
            proj[name] = float(r["fpts_mean"])
    tips = sorted({p.gametime for p in desk.slate.players if p.gametime})
    out = dict(games_started=sum(1 for t in tips if t <= now), games_total=len(tips), next_tip=next((t for t in tips if t > now), None),
               open_players=sum(1 for p in byid.values() if p.gametime and p.gametime > now), contests=[], report=LS.latest_report(),
               has_display=bool(os.environ.get("DISPLAY")), entries_path=_entries_info().get("path"))
    p = _entries_path()
    names_to_id = LS.name_map(desk.session)
    if p.exists():
        try:
            df = read_entries(p)
        except Exception as ex:   # noqa: BLE001
            out["error"] = f"{type(ex).__name__}: {ex}"; df = None
        if df is not None:
            for (cid, cname), g in df.groupby(["Contest ID", "Contest Name"], sort=False):
                rows = []
                for _, r in g.iterrows():
                    slots = []
                    for slot in POSITION_ORDER:
                        m = str(r[slot]); pid = LS._pid(m)
                        if pid is None and "(" in m:
                            pid = LS._resolve(names_to_id, LS._cell_name(m))
                        pl = byid.get(pid)
                        slots.append(dict(slot=slot, pid=pid, name=pl.name if pl else LS._cell_name(m), team=pl.team if pl else "",
                                          locked=bool(LS._is_locked_cell(m) or (pl and pl.gametime and pl.gametime <= now)),
                                          fpts=proj.get(normalize_name(pl.name)) if pl else None, tip=pl.gametime if pl else None, on_slate=pl is not None))
                    rows.append(dict(entry_id=r["Entry ID"], slots=slots, locked=sum(1 for x in slots if x["locked"]),
                                     fpts=sum((x["fpts"] or 0) for x in slots)))
                from ivan.core import raw as _raw
                art = _raw.latest("nba", "dk/standings", int(cid))
                out["contests"].append(dict(contest_id=int(cid), name=cname, entries=rows,
                                            standings=None if art is None else datetime.strptime(art.stamp[:15], "%Y%m%d-%H%M%S")))
    rep = out["report"]
    if rep:
        for c in rep.get("contests", []):
            for e in c["entries"]:
                e["delta"] = e["best"]["exp_payout"] - e["original"]["exp_payout"]
            c["gain"] = sum(e["delta"] for e in c["entries"]); c["improved"] = sum(1 for e in c["entries"] if e["improved"])
        if "exposure" not in rep:                                  # reports written before the exposure table
            rep["n_entries"] = sum(len(c["entries"]) for c in rep["contests"]); rep["exposure"] = LS.exposure(rep, byid)
        try:
            rep["when_dt"] = datetime.fromisoformat(rep["when"])
        except (KeyError, ValueError):
            rep["when_dt"] = None
    return out


def _pipeline_ctx() -> dict:
    jobs = runner.all()[:8]
    cur = runner.current() or next((j for j in jobs if j.status == "queued"), None)
    log = runner.read_log(cur, tail_lines=14) if cur else (runner.read_log(jobs[0], tail_lines=14) if jobs else "")
    try:
        slates = lineup_slates(desk.session)
    except Exception:   # noqa: BLE001 - a desk without the lineup tables yet
        slates = []
    current = next((p.draft_group_id for p in (desk.slate.players if desk.slate else []) if p.draft_group_id), None)
    n_lineups = desk.session.query(Lineup).count(); n_sims = desk.session.query(LineupSimulationResult).count()
    projected = desk.session.query(NBAPlayer).filter(NBAPlayer.fpts_mean.isnot(None)).count() > 0
    ent = _entries_info()
    # each step opens when the one before it has run: generate needs a projected pool, sim needs lineups (and entries), choose needs sims
    ready = dict(generate=projected, sim=bool(projected and n_lineups and ent.get("exists")), choose=bool(n_sims and ent.get("exists")),
                 all=bool(projected and ent.get("exists")))
    just_finished = bool(jobs and not runner.active() and jobs[0].finished and (time.time() - jobs[0].finished) < 8)
    return dict(form=pipeline_form, jobs=jobs, current=cur, log=log, active=bool(runner.active()), entries=ent,
                outputs=_latest_outputs(), lineups_in_db=n_lineups, sims_in_db=n_sims, lineup_slates=slates, current_draft_group=current,
                side=_side_ctx(), ready=ready, projected=projected, just_finished=just_finished)


def _field(raw, label: str, kind=float):
    raw = str(raw if raw is not None else "").strip()
    if not raw:
        return None
    try:
        return kind(float(raw))
    except ValueError:
        raise ValueError(f"{label}: {raw!r} is not a number")


def write_desk_config(form: dict) -> Path:
    """config/nba.json with the panel's knobs applied, written for the jobs (-c). Blank fields keep the file's values."""
    c = _config(); g = c.setdefault("generate", {}); ch = c.setdefault("choose", {})
    for key in ("max_salary", "min_salary", "max_salary_per_team"):
        v = _field(form.get(key), key, int)
        if v is not None:
            c[key] = v
    for key, kind in (("n_samples", int), ("jitter", float), ("min_fpts_frac", float), ("player_min_fpts", float), ("randomness", float)):
        v = _field(form.get(key), key, kind)
        if v is not None:
            g[key] = v
    if str(form.get("player_min_fpts") or "").strip():
        c["player_min_fpts"] = g["player_min_fpts"]
    for key in ("cash_pct", "avg_roi_win", "diversity_lambda", "bankroll"):
        v = _field(form.get(key), key)
        if v is not None:
            ch[key] = v
    if form.get("jam"):
        ch["diversity_lambda"] = 0.0                              # jam: the top lineups by quality, no spread between them
    ch["method"] = str(form.get("method") or ch.get("method", "greedy"))
    if int(c["min_salary"]) > int(c["max_salary"]):
        raise ValueError("min salary is above max salary")
    DESK_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    DESK_CONFIG.write_text(json.dumps(c, indent=2))
    return DESK_CONFIG


def _submit(step: str, form: dict) -> list:
    py = sys.executable
    jobs = []
    cfg = write_desk_config(form)
    n_jobs = max(1, min(MAX_JOBS, int(_field(form.get("jobs"), "jobs", int) or MAX_JOBS)))
    if step in ("generate", "all"):
        cmd = [py, str(SCRIPTS / "generate_lineups.py"), str(int(form["lineups"])), str(n_jobs), "-c", str(cfg), "--no-progress"] + (["--fresh"] if str(form.get("fresh", "on")) else [])
        jobs.append(runner.submit("generate", cmd, meta=dict(step="generate")))
    if step in ("sim", "all"):
        jobs.append(runner.submit("sim", [py, str(SCRIPTS / "sim.py"), "--config", str(cfg), "-n", str(int(form["sims"])), "-c", str(int(form["field_cap"]))], meta=dict(step="sim")))
    if step == "lateswap":
        source = "raw" if LIMITED else str(form.get("ls_source") or "draftkings")     # no DraftKings login here: the uploaded standings
        cmd = [py, str(SCRIPTS / "lateswap.py"), "--source", source,
               "--candidates", str(int(_field(form.get("ls_candidates"), "candidates", int) or 10)),
               "--field-cap", str(int(_field(form.get("ls_field_cap"), "field cap", int) or 5000)),
               "--sims", str(int(_field(form.get("ls_sims"), "sims", int) or 1000))] + (["--live"] if form.get("ls_live") else [])
        jobs.append(runner.submit("lateswap", cmd, meta=dict(step="lateswap")))
    if step in ("choose", "all"):
        ch = json.loads(cfg.read_text())["choose"]
        cmd = [py, str(SCRIPTS / "choose.py"), "--config", str(cfg), "--no-prompt", "--method", str(ch["method"]),
               "--cash-pct", "%g" % ch["cash_pct"], "--avg-roi-win", "%g" % ch["avg_roi_win"], "--diversity-lambda", "%g" % ch["diversity_lambda"], "--bankroll", "%g" % ch["bankroll"]]
        jobs.append(runner.submit("choose", cmd, meta=dict(step="choose")))
    return jobs


@app.get("/pipeline/status", response_class=HTMLResponse)
def pipeline_status(request: Request):
    """Only the job status, log and results: what polls while a job runs, so nobody's typing in the form is overwritten."""
    with LOCK:
        return _render(request, "_nba_pipeline_status.html", **_pipeline_ctx())


@app.get("/pipeline", response_class=HTMLResponse)
def pipeline(request: Request):
    with LOCK:
        return _render(request, "_nba_pipeline.html", **_pipeline_ctx())


@app.post("/pipeline/{step}", response_class=HTMLResponse)
async def pipeline_run(request: Request, step: str):
    """generate | sim | choose | all | lateswap. Every job but choose publishes the desk's projections first, so the pool is what
    you see on the grid (2026-10-25: a sim run on a stale pool after a repaint was the late-swap mess)."""
    form = await request.form()
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        pipeline_form.update({k: str(form.get(k, "")) for k in PIPELINE_DEFAULTS})
        notes = clamp_form(pipeline_form)
        if notes:
            desk.flash.append("lowered to this desk's limits: " + ", ".join(notes))
        pipeline_form["jam"] = "on" if form.get("jam") else ""
        if step == "lateswap":
            pipeline_form["ls_live"] = "on" if form.get("ls_live") else ""
        if step not in ("generate", "sim", "choose", "all", "lateswap"):
            desk.flash.append(f"unknown step {step}")
        elif runner.active():
            desk.flash.append("a job is already running; wait for it or cancel it")
        else:
            ms = desk.mapped_summary()
            if ms["n"] and ms["n_mapped"] < ms["n"] and step != "choose":
                desk.flash.append(f"heads up: {ms['n_mapped']} of {ms['n']} teams mapped and unchanged"
                                  + (f"; changed: {', '.join(ms['changed'])}" if ms["changed"] else "") + (f"; unmapped: {', '.join(ms['unmapped'])}" if ms["unmapped"] else ""))
            if step in ("generate", "sim", "lateswap", "all"):
                desk.write_local()
                try:
                    info = project_slate(desk.session, n_sims=2000, log=lambda m: None)
                    desk.flash.append(f"published {info.players} players with {len(desk.overrides)} edits, then queued {step}")
                    autopilot.touch("publish")
                    _push_feed(desk.flash.append)
                except IngestError as e:
                    desk.flash.append(f"publish failed: {e}")
            try:
                _submit(step, pipeline_form)
            except (TypeError, ValueError) as e:
                desk.flash.append(f"bad setting: {e}")
        return _render(request, "_nba_pipeline.html", **_pipeline_ctx())


@app.post("/pipeline/cancel/{job_id}", response_class=HTMLResponse)
def pipeline_cancel(request: Request, job_id: str):
    with LOCK:
        runner.cancel(job_id)
        return _render(request, "_nba_pipeline.html", **_pipeline_ctx())


@app.post("/entries", response_class=HTMLResponse)
async def entries_upload(request: Request, file: UploadFile = File(...)):
    """The DraftKings entries export, saved as the configured entries file."""
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD or not (file.filename or "").lower().endswith(".csv"):
        with LOCK:
            desk.flash.append("entries upload refused: a .csv under 5 MB, please")
            return _render(request, "_nba_pipeline.html", **_pipeline_ctx())
    with LOCK:
        p = _entries_path(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        desk.flash.append(f"entries file saved: {file.filename} -> {p.name} ({len(data)} bytes)")
        return _render(request, "_nba_pipeline.html", **_pipeline_ctx())


MAX_STANDINGS = 60 * 1024 * 1024        # a 100k-entry contest's standings csv is ~15 MB; the zip DraftKings hands out is smaller


def _standings_text(name: str, data: bytes) -> str:
    """The standings CSV from what DraftKings' export button gives: a .zip holding one csv, or the csv itself."""
    import io, zipfile
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            members = [m for m in z.infolist() if m.filename.lower().endswith(".csv") and not m.is_dir()]
            if len(members) != 1:
                raise ValueError("the zip should hold exactly one .csv")
            if members[0].file_size > MAX_STANDINGS * 4:
                raise ValueError("the csv inside the zip is too large")
            return z.read(members[0]).decode("utf-8-sig", errors="replace")
    if name.lower().endswith(".csv"):
        return data.decode("utf-8-sig", errors="replace")
    raise ValueError("a .zip or .csv from DraftKings' standings export, please")


@app.post("/lateswap/standings", response_class=HTMLResponse)
async def standings_upload(request: Request, files: list[UploadFile] = File(...), contest_id: str = Form("")):
    """
    Contest standings exported from DraftKings (the contest page's export, once the contest has locked): the field the late
    swap plays against, on a desk without a DraftKings login. The contest id comes from the file name
    (contest-standings-<id>.zip) or the picker. Kept in the raw store as dk/standings, where `--source raw` finds it.
    """
    import re
    from ivan.core import raw
    from ivan.sports.nba.standings import parse_export
    saved, errors = [], []
    for f in files:
        data = await f.read(MAX_STANDINGS + 1)
        if len(data) > MAX_STANDINGS:
            errors.append(f"{f.filename}: over {MAX_STANDINGS // 2**20} MB"); continue
        try:
            text = _standings_text(f.filename or "", data)
            ent, _ = parse_export(text)
        except Exception as e:   # noqa: BLE001
            errors.append(f"{f.filename}: {type(e).__name__}: {e}"); continue
        m = re.search(r"(\d{6,})", f.filename or "")
        cid = int(m.group(1)) if m else (int(contest_id) if contest_id.strip().isdigit() else None)
        if cid is None:
            errors.append(f"{f.filename}: no contest id in the file name; pick the contest and upload again"); continue
        raw.store("nba", "dk/standings", cid, text, ext="csv", meta={"source": "upload", "file": f.filename})
        saved.append(f"{cid} ({len(ent):,} entries)")
    with LOCK:
        if saved:
            desk.flash.append("standings saved for " + ", ".join(saved))
        for e in errors:
            desk.flash.append("standings upload: " + e)
    return RedirectResponse(auth.root_of(request) + "/?view=lateswap", status_code=303)


# ---------------------------------------------------------------- exports (the brother's spreadsheet: plain CSVs of what the desk holds now)
EXPORT_SLOTS = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")


def _csv_line(values) -> str:
    import csv, io
    buf = io.StringIO(); csv.writer(buf, lineterminator="\n").writerow(values); return buf.getvalue()


def _csv_response(body, filename: str) -> StreamingResponse:
    """
    A CSV download that is never cached on the way: Cloudflare caches a .csv URL by default when the origin sends no
    cache headers, and the brother's desk arrives through the tunnel, so without these every download of the same URL
    came back as the first one for hours. no-store covers browsers and the edge (CDN-Cache-Control is Cloudflare's own).
    """
    return StreamingResponse(body, media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"',
                                      "Cache-Control": "no-store, private, max-age=0", "CDN-Cache-Control": "no-store",
                                      "Pragma": "no-cache", "Expires": "0"})


@app.post("/team/{team}/news", response_class=HTMLResponse)
def late_news(request: Request, team: str, player: str = Form(...), call: str = Form("out"), then: str = Form("sim")):
    """
    Late news in one move: the call (out / starts / bench / back), the team re-allocated around it, the pool published, and
    the sim + choose (before lock) or the late swap (after) queued. What the pin, Publish and the pipeline buttons did
    one after another, with nothing to forget in between.
    """
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        key = normalize_name(player)
        if call == "out":
            desk.set_override(key, minutes=0.0)
        elif call == "starts":
            desk.set_override(key, starter=1.0, minutes=None)          # a starter: any scratch pin lifted, the solve gives him his minutes
        elif call == "bench":
            desk.set_override(key, starter=0.0)
        elif call == "back":
            desk.set_override(key, minutes=None, starter=None)
        else:
            desk.flash.append(f"unknown call {call}")
            return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))
        desk.reproject(team)
        desk.write_local()
        desk.note_call(team, key, call)
        try:
            for m in desk.auto_scenarios(calls={team: {key: call}}):
                desk.flash.append(m)
        except Exception as e:   # noqa: BLE001
            desk.flash.append(f"scenario pick failed: {type(e).__name__}: {e}")
        try:
            info = project_slate(desk.session, n_sims=2000, log=lambda m: None)
            desk.flash.append(f"{team}: {player} {call}: published {info.players} players with {len(desk.overrides)} edits")
            _push_feed(desk.flash.append)
        except IngestError as e:
            desk.flash.append(f"{team}: {player} {call}: publish failed: {e}")
            then = "none"
        if then in ("sim", "lateswap"):
            if runner.active():
                desk.flash.append("a job is already running: nothing queued (cancel it or wait, then run Sim / Late swap)")
            else:
                try:
                    if then == "sim":
                        _submit("sim", pipeline_form); _submit("choose", pipeline_form)
                    else:
                        _submit("lateswap", pipeline_form)
                    desk.flash.append("queued " + ("sim + choose" if then == "sim" else "the late swap"))
                except (TypeError, ValueError) as e:
                    desk.flash.append(f"not queued: {e}")
        desk.rebuild()
        # the call moved the pool, the header (questions, chips), the scenarios and the pipeline state: the whole page reloads,
        # so nothing is rendered here (a render would consume the flash the reloaded board is about to show)
        return HTMLResponse("", headers={"HX-Refresh": "true"})


@app.get("/export/projections.csv")
def export_projections():
    """Every player in the current pool with our projection (after the desk's edits) and the public one, one row each."""
    with LOCK:
        desk.maybe_reload()
        if desk.stale:                       # a newer projection is on disk (a pull, a publish): a file has no page state to keep, so take it
            desk.rebuild()
        rows = [list(("name", "dk_id", "team", "opponent", "position", "salary", "status", "starter", "minutes", "solve_minutes", "minutes_p10", "minutes_p90",
                      "fpts", "fpts_sd", "fppm", "public_fpts", "ownership", "pinned", "locked", "in_pool"))]
        for team in desk.slate.teams():
            proj = desk.proj.get(team); rot = desk.rot.get(team) or {}
            starters = desk.starters_of(team) or set(); locked = set((rot.get("grid").attrs.get("locked", []) if rot.get("grid") is not None else []))
            for p in desk.players(team):
                n = normalize_name(p.name)
                r = proj.loc[n] if proj is not None and n in proj.index else None
                ov = override_of(desk.overrides, n)
                def f(x, nd=1):
                    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
                rows.append([p.name, p.id, p.team, p.opponent or "", p.position or "", p.salary or "", (p.status or ""), int(n in starters),
                             f(None if r is None else r["minutes"]), f((rot.get("targets") or {}).get(n)),
                             f(None if r is None else r.get("minutes_p10")), f(None if r is None else r.get("minutes_p90")),
                             f(None if r is None else r["fpts_mean"], 2), f(None if r is None else r["fpts_std"], 2), f(None if r is None else r["fpm"], 3),
                             f(getattr(p, "pub_fpts", None), 2), f(p.ownership, 3) if p.ownership not in (None, 0) else "",
                             int(ov.get("minutes") is not None), int(n in locked), int(r is not None and float(r["minutes"]) > 0)])
        proj_at = desk.pool_stamp[0] if desk.pool_stamp and desk.pool_stamp[0] else None      # when the pool's projection was published
        stamp = (proj_at.strftime("%Y%m%d-%H%M") if hasattr(proj_at, "strftime") else str(proj_at or "")[:16].replace(" ", "-").replace(":", "")) or datetime.now().strftime("%Y%m%d-%H%M")
    body = "".join(_csv_line(r) for r in rows)
    return _csv_response(iter([body]), f"nba_projections_{stamp}.csv")


def lineup_slates(session) -> list[dict]:
    """The slates with generated lineups, newest first: draft group, count, the pool snapshot's lock when known."""
    rows = session.execute(__import__("sqlalchemy").text("select draft_group_id, count(*) from lineup group by draft_group_id")).all()
    locks = {int(r[0]): r[1] for r in session.execute(__import__("sqlalchemy").text("select draft_group_id, min(lock) from lineup_pool group by draft_group_id")).all()}
    out = [dict(draft_group=None if dg is None else int(dg), n=int(n), lock=locks.get(int(dg)) if dg is not None else None) for dg, n in rows]
    return sorted(out, key=lambda r: (r["draft_group"] is None, -(r["draft_group"] or 0)))


def backfill_lineup_slates(session, log=lambda m: None) -> int:
    """
    Lineups generated before the slate was stamped: give each the newest slate whose pool held all its players, from the
    pool snapshots, the live pool, or the projection captures (the server has those for every slate it published).
    """
    from ivan.core.utils import coerce_player_ids
    from ivan.sports.nba.models import NBAProjectionCapture
    null_rows = session.query(Lineup.id, Lineup.player_ids_list).filter(Lineup.draft_group_id.is_(None)).all()
    if not null_rows:
        return 0
    pools: dict[int, set] = {}
    for dg, pid in session.query(LineupPool.draft_group_id, LineupPool.player_id).all():
        pools.setdefault(int(dg), set()).add(int(pid))
    for p in session.query(NBAPlayer.draft_group_id, NBAPlayer.id).all():
        if p[0] is not None:
            pools.setdefault(int(p[0]), set()).add(int(p[1]))
    for dg, pid in session.query(NBAProjectionCapture.draft_group_id, NBAProjectionCapture.player_id).filter(
            NBAProjectionCapture.draft_group_id.isnot(None), NBAProjectionCapture.player_id.isnot(None)).distinct().all():
        pools.setdefault(int(dg), set()).add(int(pid))
    order = sorted(pools, reverse=True)
    n = 0
    for lid, pids in null_rows:
        try:
            ids = {int(x) for x in coerce_player_ids(pids)}
        except (ValueError, TypeError):
            continue
        dg = next((d for d in order if ids <= pools[d]), None)
        if dg is not None:
            session.query(Lineup).filter(Lineup.id == lid).update({"draft_group_id": dg}, synchronize_session=False); n += 1
    session.commit()
    if n:
        log(f"{n} older lineups given their slate")
    return n


def lineup_names(session, draft_group: int) -> dict[int, dict]:
    """player id -> {name, position} for a slate: the pool snapshot, then the live pool, then the projection captures (names only)."""
    from ivan.sports.nba.models import NBAProjectionCapture
    out: dict[int, dict] = {}
    for r in session.query(LineupPool).filter(LineupPool.draft_group_id == draft_group).all():
        out[int(r.player_id)] = dict(name=r.name, position=r.position)
    for p in session.query(NBAPlayer).filter(NBAPlayer.draft_group_id == draft_group).all():
        out.setdefault(int(p.id), dict(name=p.name, position=p.position))
    for pid, name in session.query(NBAProjectionCapture.player_id, NBAProjectionCapture.player_name).filter(
            NBAProjectionCapture.draft_group_id == draft_group, NBAProjectionCapture.player_id.isnot(None)).distinct().all():
        out.setdefault(int(pid), dict(name=name, position=None))
    # positions for players without a snapshot: the archived pool of that slate (the laptop keeps the blobs), then the same
    # name's position in any pool we know (a player's DK positions do not change between slates)
    missing = [pid for pid, v in out.items() if not v.get("position")]
    if missing:
        try:
            from ivan.core import archive
            caps = [c for c in archive.captures(session, "nba") if str(c.slate_key) == str(draft_group)]
            if caps:
                for r in archive.load_blobs(session, max(caps, key=lambda c: c.captured_at).id).get("players", []):
                    if r.get("id") is not None and r.get("position"):
                        out.setdefault(int(r["id"]), dict(name=r.get("name"), position=None))
                        out[int(r["id"])]["position"] = out[int(r["id"])]["position"] or r["position"]
        except Exception:   # noqa: BLE001 - no archive here
            pass
        by_name: dict[str, str] = {}
        for name, pos in session.query(LineupPool.name, LineupPool.position).all():
            if pos:
                by_name.setdefault(normalize_name(name), pos)
        for name, pos in session.query(NBAPlayer.name, NBAPlayer.position).all():
            if pos:
                by_name.setdefault(normalize_name(name), pos)
        for pid, v in out.items():
            if not v.get("position") and v.get("name"):
                v["position"] = by_name.get(normalize_name(v["name"]))
    return out


@app.get("/export/lineups.csv")
def export_lineups(ids: int = 0, draft_group: Optional[int] = None):
    """
    The generated lineups of one slate, one row each, the eight players in DraftKings roster order (PG SG SF PF C G F
    UTIL) and nothing else: names, or DraftKings ids with ?ids=1. The slate is ?draft_group=, else the loaded slate when
    it has lineups, else the newest slate with lineups. Names and positions come from the pool snapshot written at
    generation, so the laptop moving the live pool to another slate cannot empty this.
    """
    from types import SimpleNamespace
    from ivan.sports.nba.lineup_export import assign_player_order_classic
    from ivan.core.utils import coerce_player_ids
    with LOCK:
        desk.maybe_reload()
        backfill_lineup_slates(desk.session)
        slates = lineup_slates(desk.session)
        current = next((p.draft_group_id for p in desk.slate.players if p.draft_group_id), None)
        have = [r["draft_group"] for r in slates if r["draft_group"] is not None]
        dg = draft_group if draft_group is not None else (current if current in have else (have[0] if have else None))
        if dg is None:
            body = _csv_line(["no generated lineups in the pool table: run generate first"])
            return _csv_response(iter([body]), "nba_lineups_none.csv")
        lookup = lineup_names(desk.session, int(dg))
        lineups = desk.session.query(Lineup.id, Lineup.player_ids_list).filter(Lineup.draft_group_id == int(dg)).all()
        stamp = datetime.now().strftime("%Y%m%d-%H%M")

    def gen():
        yield _csv_line(EXPORT_SLOTS)
        for lid, pids in lineups:
            try:
                pl = [int(x) for x in coerce_player_ids(pids)]
            except (ValueError, TypeError):
                continue
            objs = [SimpleNamespace(id=x, name=lookup.get(x, {}).get("name") or str(x), position=lookup.get(x, {}).get("position")) for x in pl]
            ordered = objs
            if len(objs) == len(EXPORT_SLOTS) and all(o.position for o in objs):
                try:
                    ordered = assign_player_order_classic(objs, prefer_late_in_flex=False)
                except ValueError:
                    ordered = objs                       # an odd eligibility: the stored order
            yield _csv_line([o.id if ids else o.name for o in ordered])
    return _csv_response(gen(), f'nba_lineups_{dg}_{stamp}{"_ids" if ids else ""}.csv')


@app.get("/download/{name}")
def download(name: str):
    p = OUTPUT_DIR / Path(name).name
    if not p.exists() or not p.name.startswith(("chosen_entries.", "lateswap.")):
        return HTMLResponse("not found", status_code=404)
    return FileResponse(p, filename=p.name, media_type="text/csv")


@app.post("/slate")
def switch_slate(request: Request, draft_group: int = Form(...), source: str = Form("lobby")):
    """Load another slate: a lobby slate is fetched from DraftKings, an archived one restored from its capture (nothing fetched).
    The pick is remembered (data/nba/active_slate.json)."""
    with LOCK:
        from ivan.sports.nba.paths import ACTIVE_SLATE_JSON
        try:
            ACTIVE_SLATE_JSON.write_text(json.dumps({"draft_group": int(draft_group), "chosen_at": datetime.now().isoformat(timespec="seconds")}))
        except OSError as e:
            desk.flash.append(f"could not save the slate choice ({e}); the capture tick may switch back")
        if runner.active():
            desk.flash.append("a job is running on the current slate; switch after it finishes (or cancel it)")
        elif desk.busy:
            desk.flash.append(f"already {desk.busy}")
        elif source == "archive":
            _start_restore(int(draft_group))
        else:
            _start_refresh(f"loading slate {draft_group}", int(draft_group), update_odds=False)
    return RedirectResponse(auth.root_of(request) + "/", status_code=303)
