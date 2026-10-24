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
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv

from ivan.core import feed
from ivan.core.config import config_path
from ivan.core.utils import PROJECT_ROOT, normalize_name
from ivan.database.session import get_session, init_db
from ivan.sports.nba.ingest import ensure_schema, fetch_slate, project_slate, store_slate, IngestError
from ivan.sports.nba.models import Lineup, LineupSimulationResult, NBAMarketRate, NBAPlayer
from ivan.sports.nba.paths import NBA_DATA_DIR, MINUTES_OVERRIDES_CSV, OUTPUT_DIR, INPUT_DIR
from ivan.web.jobs import JobRunner
from ivan.web import auth
from ivan.sports.nba.projection.engine import ProjectionParams
from ivan.sports.nba.projection.live import LiveSlate, OVERRIDE_KEYS, override_of, read_overrides_full, write_overrides
from ivan.sports.nba.projection.history import STATS
from ivan.sports.nba.validation import market_points, recent_minutes, review_team
from ivan.sports.nba.projection import rotation as ROT
from ivan.sports.nba.projection.minutes import salary_prior_minutes
from ivan.sports.nba.projection.book_minutes import book_minutes
TEAM_BUDGET = 241.0

HERE = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")                                  # IVAN_FEED_URL (consumer) / IVAN_FEED_PUSH (producer)
CONSUMER = bool(os.environ.get("IVAN_FEED_URL", "").strip())        # the server: takes what the laptop pushes, never fetches
engine = init_db("nba")
ensure_schema(engine)
app = FastAPI(title="Ivan · NBA desk", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")
auth.install(app, templates)                                  # a password when IVAN_DESK_PASSWORD is set; open otherwise
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
pipeline_form: dict = dict(PIPELINE_DEFAULTS)
DESK_CONFIG = NBA_DATA_DIR / "jobs" / "desk_config.json"           # config/nba.json with the panel's knobs applied; the jobs read it

MAX_UPLOAD = 5 * 1024 * 1024   # the DK entries export is tens of KB
ROTATION_MIN = 15.0        # a scratch/questionable above this baseline is news
NEAR_TIP_MIN = 75          # minutes before tip when an unconfirmed lineup becomes a warning
MINUTES_DEV = 8.0          # model minutes this far from history is a role change
MARKET_GAP = 0.20          # market mean vs history-only projection, relative


def utc_to_local(stamp) -> datetime:
    """An ISO time from the odds API ('2026-10-23T19:45:19Z') as naive local time, like every other desk time."""
    t = pd.Timestamp(stamp)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert(datetime.now().astimezone().tzinfo).tz_localize(None).to_pydatetime()


class Desk:
    """Session + LiveSlate + per-team projections, rebuilt when the pool changes under us."""
    def __init__(self):
        self.session = get_session(engine)
        self.slate: Optional[LiveSlate] = None
        self.pool_stamp = None
        self.overrides: dict[str, float] = {}
        self.proj: dict[str, pd.DataFrame] = {}
        self.hist_only: dict[str, pd.DataFrame] = {}
        self.flash: list[str] = []
        self.rebuild()

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
        self.balanced = set()                              # teams whose minutes were set from the books (the unpriced re-fill on every edit)
        self.fills: dict = {}                              # team -> the unpriced players whose minutes are a re-fill, not a hand pin
        self.overrides = read_overrides_full()            # the file is the source of truth: every edit is written to it
        self.proj, self.hist_only = {}, {}
        self.season = self.season_averages(self.slate.box)
        self.recent = self.recent_games(self.slate.box)
        self.market_rows = {}
        for r in self.session.query(NBAMarketRate).all():
            self.market_rows.setdefault(r.player_name, {})[r.stat] = dict(mean=float(r.market_mean), line=r.line, books=r.books, rate=float(r.rate), basis=float(r.basis_minutes),
                                                                           odds=None)
        for (name, stat), txt in self.consensus_odds().items():
            if name in self.market_rows and stat in self.market_rows[name]:
                self.market_rows[name][stat]["odds"] = txt
        for t in self.slate.teams():
            self.reproject(t)

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
    def recent_games(box: pd.DataFrame, n: int = 10) -> dict:
        """player -> his last n games played, newest first: (date, team, minutes, fppm)."""
        out = {}
        sub = box[box.played].sort_values(["game_date", "game_id"])
        for name, g in sub.groupby("player_name"):
            tail = g.tail(n)
            out[name] = [dict(date=str(r.game_date), team=r.team, minutes=float(r.minutes), fpts=float(r.dk_fpts),
                              fpm=float(r.dk_fpts) / float(r.minutes) if r.minutes > 0 else None) for r in tail.itertuples()][::-1]
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
        The team's rotation grid. source: "auto" = the drawn grid when one exists, else the team's last game;
        "last" = the last game, or the game `game_id` from the dropdown; "propose" = the model's proposal from the players' stint curves.
        Players who are out stay in a last-game grid with their minutes, marked so they can be reallocated.
        """
        from ivan.sports.nba.projection.live import OUT_STATUSES
        players = self.players(team)
        names = {normalize_name(p.name): p for p in players}
        is_out = {n: (p.status or "").upper() in OUT_STATUSES or (override_of(self.overrides, n).get("minutes") is not None and override_of(self.overrides, n).get("minutes") <= 0)
                  for n, p in names.items()}
        avail = [n for n in names if not is_out[n]]
        starters = self.starters_of(team) or set()
        curve = ROT.curves(self.session, players=list(names), n_games=10)
        drawn = ROT.load_drawn(team) if source == "auto" else None
        today = datetime.now().date(); season_start = date(today.year if today.month >= 10 else today.year - 1, 10, 1)
        games = ROT.team_games(self.session, team, since=season_start)
        last = ROT.last_game_grid(self.session, team, since=season_start, game_id=game_id if source == "last" else None) if source in ("auto", "last") and drawn is None else None
        kind = "drawn" if drawn is not None else ("last" if last is not None else "propose")
        if kind == "drawn":
            grid = drawn
        elif kind == "last":
            grid = last["grid"]
            if source == "auto" and len(last["grid"]) and ROT.column_counts(last["grid"]).count(5) < 30:
                kind = "propose"; last = None                    # a broken reconstruction: fall back to the proposal
        if kind == "propose":
            prior = {n: salary_prior_minutes(p.salary) for n, p in names.items()}
            grid = ROT.propose(curve, avail, [n for n in avail if n in starters], salary_prior=prior) if (avail and not curve.empty) else None
        if kind == "last" and last is not None:
            pass
        if grid is None:
            return dict(team=team, rows=[], counts=[], drawn=False, kind="none", has_history=len(curve) > 0, to_reallocate=0, out_names=[], games=games, game_id=None)
        grid = grid.copy()
        gone = [n for n in grid.index if n not in names and kind == "last"]     # played last game, not on tonight's roster
        for n in names:                                        # tonight's roster: everyone gets a row
            if n not in grid.index:
                grid.loc[n] = False
        grid = grid.loc[[n for n in grid.index if n in names or n in gone]]
        if kind != "last":                                     # only a last-game grid keeps out players' minutes on show
            for n in grid.index:
                if n in is_out and is_out[n]:
                    grid.loc[n] = False
        def unavailable(n):
            return n in gone or is_out.get(n, False)
        # starters on top (minutes descending), then the rest of the available roster, then anyone out or gone
        grid = grid.loc[sorted(grid.index, key=lambda n: (unavailable(n), n not in starters, -float(grid.loc[n].sum()), -((names[n].salary if n in names else 0) or 0)))]
        rows = [dict(name=n, key=n, pos=names[n].position if n in names else "", salary=names[n].salary if n in names else None, starter=n in starters,
                     out=unavailable(n), gone=n in gone, minutes=int(grid.loc[n].sum()),
                     hist=None if n not in curve.index else float(curve.loc[n, "minutes"]), cells=[bool(x) for x in grid.loc[n].values]) for n in grid.index]
        out_rows = [r for r in rows if r["out"] and r["minutes"] > 0]
        return dict(team=team, rows=rows, counts=ROT.column_counts(grid), drawn=kind == "drawn", kind=kind, has_history=len(curve) > 0,
                    curve_games=int(curve.games.max()) if len(curve) else 0, last_game=None if last is None else last["game_date"],
                    games=games, game_id=None if last is None else last["game_id"],
                    to_reallocate=int(sum(r["minutes"] for r in out_rows)), out_names=[r["name"] for r in out_rows])

    def apply_rotation(self, team: str, players: list[str], grid: list[list[int]], note: str = "") -> int:
        """Save a drawn grid and pin every available drawn player's minutes to his row (0 minutes = scratched); out players' rows are dropped."""
        import pandas as pd
        from ivan.sports.nba.projection.live import OUT_STATUSES
        status = {normalize_name(p.name): (p.status or "").upper() for p in self.players(team)}
        g = pd.DataFrame(np.array(grid, dtype=bool), index=[normalize_name(n) for n in players], columns=list(range(ROT.MINUTES)))
        g = g.loc[[n for n in g.index if status.get(n, "") not in OUT_STATUSES]]
        ROT.save_drawn(team, g, note)
        for n, m in ROT.minutes_of(g).items():
            self.set_override(n, minutes=float(m))
        self.reproject(team)
        return len(g)

    def clear_rotation(self, team: str) -> None:
        g = ROT.load_drawn(team)
        ROT.clear_drawn(team)
        if g is not None:
            for n in g.index:
                self.set_override(n, minutes=None)
        self.reproject(team)

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

    def balance_to_books(self, team: str) -> str:
        """
        Priced players at the books' minutes with their totals held; the unpriced available players fill what is left of the
        team's minutes in proportion to the model's minutes. Hand pins on unpriced players are kept as they are.
        """
        rows = [r for r in self.team_rows(team) if not r["out"]]
        priced = [r for r in rows if r.get("book_min") is not None]
        if not priced:
            return "no priced players with book minutes on this team"
        for r in priced:
            total = self.priced_total(team, r["key"])
            self.set_override(r["key"], minutes=float(r["book_min"]), fppm=None if not total else round(total / float(r["book_min"]), 4))
        self.balanced.add(team)
        n = self.refill(team, rows)
        return f"{len(priced)} priced players set to the books' minutes; {n} unpriced players fill the rest"

    def refill(self, team: str, rows: Optional[list] = None) -> int:
        """The unpriced available players (those not pinned by hand) take what the pinned players leave of the team's minutes."""
        self.reproject(team)
        rows = rows or [r for r in self.team_rows(team) if not r["out"]]
        fixed = [r for r in rows if r.get("book_min") is not None or (r["pinned"] and r["key"] not in self.fills.get(team, set()))]
        free = [r for r in rows if r not in fixed]
        left = TEAM_BUDGET - sum(float(override_of(self.overrides, r["key"]).get("minutes") or r["minutes"] or 0.0) for r in fixed)
        base = {r["key"]: float(self.hist_only[team].loc[r["key"], "minutes"]) if r["key"] in self.hist_only[team].index else float(r["minutes"] or 0.0) for r in free}
        tot = sum(base.values())
        self.fills[team] = set()
        for r in free:
            m = max(0.0, left) * base[r["key"]] / tot if tot > 0 else 0.0
            self.set_override(r["key"], minutes=round(m, 1)); self.fills[team].add(r["key"])
        self.reproject(team)
        return len(free)

    def pins(self) -> dict[str, float]:
        """player -> pinned minutes (the overrides' minutes column only)."""
        return {k: v["minutes"] for k, v in self.overrides.items() if v.get("minutes") is not None}

    def review(self, team: str) -> dict:
        """The rotation review (ivan.sports.nba.validation.review_team): provenance of every player's minutes and the flags."""
        if getattr(self, "_recent", None) is None or self._recent_stamp != self.pool_stamp:
            self._recent = recent_minutes(self.slate.box); self._recent_stamp = self.pool_stamp
        return review_team(self.slate, team, self.proj[team], self._recent, market_points(self.session), self.pins())

    def set_override(self, name: str, **fields) -> None:
        """Set or clear a player's minutes / fppm / min_low / min_high; a player with nothing left is dropped."""
        rec = dict(self.overrides.get(name, {}))
        for k, v in fields.items():
            rec[k] = v
        rec = {k: v for k, v in rec.items() if v is not None}
        if rec:
            self.overrides[name] = {k: rec.get(k) for k in OVERRIDE_KEYS}
        else:
            self.overrides.pop(name, None)
        write_overrides(self.overrides)                    # persisted at once; a reload or a capture tick cannot lose it

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
            starter = None if starters is None else (name in starters)
            flags = []
            hist_min = float(s["min_ewm"]) * float(s.get("play_ewm", 1.0)) if s is not None else None
            if out and status != "OUT":
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
            rows.append(dict(id=p.id, name=p.name, key=name, salary=p.salary, pos=p.position, status=status if status != "NONE" else "",
                             book_min=None if bm is None else round(bm["minutes"], 1), book_src=None if bm is None else bm["source"],
                             book_by=None if bm is None else bm["by_stat"],
                             starter=("S" if starter else ("" if starter is not None else "?")),
                             confirmed=confirmed, out=out,
                             hist_min=hist_min, minutes=None if r is None else float(r["minutes"]),
                             min_p10=None if r is None or pd.isna(r.get("minutes_p10", np.nan)) else float(r["minutes_p10"]),
                             min_p90=None if r is None or pd.isna(r.get("minutes_p90", np.nan)) else float(r["minutes_p90"]),
                             pinned=ov.get("minutes") is not None, pin=ov.get("minutes"), starter_pin=ov.get("starter"),
                             fppm_pin=ov.get("fppm"), low_pin=ov.get("min_low"), high_pin=ov.get("min_high"), stats=stats,
                             fpts=fpts, sd=None if r is None else float(r["fpts_std"]),
                             fpm=None if r is None else float(r["fpm"]),
                             season_games=cur["games"] if cur else 0, season_min=cur["minutes"] if cur else None, season_fpm=cur["fpm"] if cur else None,
                             last_games=last_season["games"] if last_season else 0, last_min=last_season["minutes"] if last_season else None,
                             last_fpm=last_season["fpm"] if last_season else None,
                             value=None if fpts is None or not p.salary else fpts / (p.salary / 1000.0),
                             own=None if p.ownership in (None, 0) else float(p.ownership),
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
                    assigned=assigned, to_place=to_place, rotation_drawn=ROT.drawn_path(team).exists(),
                    locked=bool(mins_to_tip is not None and mins_to_tip < 0), alerts=alerts, score=score,
                    starters="confirmed" if confirmed else ("expected" if starters else "none"),
                    n_out=sum(1 for r in rows if r["out"]), n_pinned=sum(1 for r in rows if r["pinned"]),
                    team_fpts=team_fpts, trailing=self.slate.team_fpts.get(team), rows=rows, balanced=team in self.balanced,
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
        slates = {}
        for k, v in (st.get("slates") or {}).items():
            try:
                t = datetime.fromisoformat(str(v.get("lock")))
            except (TypeError, ValueError):
                continue
            slates[k] = dict(v, lock_dt=t, label=f"{t:%a %m-%d %H:%M} · " + str(v.get("name", "")).replace("NBA ", "")[:28])
        days = sorted({v["lock_dt"].date() for v in slates.values()})
        if days:                                                  # today's slates only (the next day's when today has none), plus the loaded one
            ahead = [d for d in days if d >= now.date()]
            day = now.date() if now.date() in days else (ahead[0] if ahead else days[-1])
            slates = dict(sorted(((k, v) for k, v in slates.items() if v["lock_dt"].date() == day or k == str(dg)), key=lambda kv: kv[1]["lock_dt"]))
        return dict(draft_group=dg, lock=lock, mins_to_lock=(lock - now).total_seconds() / 60 if lock else None,
                    pool_at=self.pool_stamp[0], n_players=len(ps), n_priced=len(self.slate.market),
                    starters_at=self.slate.starters_fetched_at, odds_at=odds_t, stale=getattr(self, "stale", False), fresh=self.freshness(now),
                    slates=slates, props_fetches=st.get("props_fetches", 0),
                    overrides=len(self.overrides), overrides_path=str(MINUTES_OVERRIDES_CSV.relative_to(PROJECT_ROOT)))


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


def _render(request: Request, template: str, **ctx):
    flash = desk.flash[:]; desk.flash.clear()
    return templates.TemplateResponse(request, template, {"now": datetime.now(), "flash": flash, "root": auth.root_of(request), "consumer": CONSUMER,
                                                          "limited": LIMITED, "limits": LIMITS, "job_max_minutes": JOB_MAX_MINUTES, **ctx})


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
        desk.maybe_reload()
        now = datetime.now()
        for t in desk.slate.teams():
            desk.slate.spread(t)                            # loads the lines once
        cards = [desk.team_card(t, now) for t in desk.slate.teams()]
        games = games_of(cards, now)
        menu = []                                            # games grouped by tip time, in order
        for g in games:
            if not menu or menu[-1]["tip"] != g["tip"]:
                menu.append(dict(tip=g["tip"], games=[]))
            menu[-1]["games"].append(g)
        players = []; lateswap = None
        if view == "lateswap":
            lateswap = _lateswap_ctx(now)
        if view == "players":
            players = [r for c in cards for r in c["rows"]]
            players.sort(key=lambda r: (-(r["own"] or 0), -(r["fpts"] or 0)))
        return _render(request, "nba_desk.html", header=desk.header(now), games=games, menu=menu, open_team=open, view=view, players=players,
                       lateswap=lateswap, **_pipeline_ctx())


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
def pin(request: Request, team: str, player: str = Form(...), minutes: str = Form(""), fppm: str = Form(""), low: str = Form(""), high: str = Form("")):
    """One row's hand values: blank clears that field. Minutes 0 scratches."""
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
        total = desk.priced_total(team, key)
        if total and new_min and new_min > 0 and (new_fppm is None or new_fppm == old.get("fppm")):
            new_fppm = round(total / new_min, 4)                  # a priced player's total is the books': minutes move, the total holds
        elif total and new_fppm is not None and new_fppm != old.get("fppm"):
            desk.flash.append(f"{key}: fppm overridden on a priced player; the books' rate no longer applies to him")
        desk.set_override(key, minutes=new_min, fppm=new_fppm, min_low=lo, min_high=hi)
        desk.fills.get(team, set()).discard(key)
        if team in desk.balanced:
            desk.refill(team)
        else:
            desk.reproject(team)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/balance_all")
def balance_all(request: Request):
    """Every team with priced players: priced players to the books' minutes (totals held), the unpriced fill the rest."""
    with LOCK:
        done, short = [], []
        for t in desk.slate.teams():
            card = desk.team_card(t, datetime.now())
            if card["locked"] or not card["n_book"]:
                continue
            desk.balance_to_books(t)
            total = desk.team_card(t, datetime.now())["assigned"]
            if abs(total - TEAM_BUDGET) > 2.0:                     # verify: a team that did not land on its minutes gets one more fill
                desk.refill(t); total = desk.team_card(t, datetime.now())["assigned"]
            (short if abs(total - TEAM_BUDGET) > 2.0 else done).append(f"{t} {total:.0f}")
        desk.flash.append("balanced to the books: " + (", ".join(done) if done else "no team has priced players")
                          + (f" · SHORT, check these teams: {', '.join(short)}" if short else ""))
    return RedirectResponse(auth.root_of(request) + "/", status_code=303)


@app.post("/team/{team}/balance", response_class=HTMLResponse)
def balance(request: Request, team: str):
    """Priced players to the books' minutes (totals held), the unpriced fill the rest of the team's minutes."""
    with LOCK:
        desk.flash.append(desk.balance_to_books(team))
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/starter", response_class=HTMLResponse)
def starter_toggle(request: Request, team: str, player: str = Form(...)):
    """Click the st cell: a bench player becomes a starter, a starter goes to the bench; a second click on a hand call clears it
    back to Rotowire's lineup. Saved with the other edits; the team re-projects (the start shift and the rotation proposal move)."""
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
    """Body: JSON {players: [...], grid: [[0/1 x 48], ...]} from the desk's grid; saves it and pins the minutes."""
    body = await request.json()
    with LOCK:
        try:
            desk.apply_rotation(team, body["players"], body["grid"], note=str(body.get("note", "")))
            desk.flash.append(f"{team}: rotation applied, {len(body['players'])} players pinned to their drawn minutes")
        except Exception as e:   # noqa: BLE001
            desk.flash.append(f"{team}: rotation not applied: {type(e).__name__}: {e}")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/rotation/propose", response_class=HTMLResponse)
def rotation_propose(request: Request, team: str, source: str = Form("propose"), game_id: str = Form("")):
    """Show the model's proposal (source=propose) or the last game (source=last) without applying it."""
    with LOCK:
        return _render(request, "_nba_rotation.html", rot=desk.rotation(team, source=source if source in ("last", "propose") else "propose", game_id=game_id.strip() or None))


@app.post("/team/{team}/rotation/clear", response_class=HTMLResponse)
def rotation_clear(request: Request, team: str):
    with LOCK:
        desk.clear_rotation(team)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/clear", response_class=HTMLResponse)
def clear_team(request: Request, team: str):
    with LOCK:
        for p in desk.players(team):
            desk.overrides.pop(normalize_name(p.name), None)
        write_overrides(desk.overrides)
        desk.balanced.discard(team); desk.fills.pop(team, None)
        desk.reproject(team)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@app.post("/publish")
def publish(request: Request):
    """Write the pins, project every team at full sims, store projections + a capture row set."""
    with LOCK:
        write_overrides(desk.overrides)
        try:
            info = project_slate(desk.session, n_sims=2000, log=desk.flash.append)
            desk.flash.append(f"published: {info.players} players projected with {len(desk.overrides)} edits; edits saved to {MINUTES_OVERRIDES_CSV.name}")
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
        feed.publish("nba", which="live", log=lambda m: None)
        feed.push("nba", log=log)
    except Exception as e:   # noqa: BLE001 - the publish itself succeeded; the push is best effort
        log(f"feed push failed: {type(e).__name__}: {e}")


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
                    project_slate(desk.session, n_sims=1000, log=desk.flash.append)
                else:
                    desk.flash.append("feed unchanged since the last pull")
            except Exception as e:   # noqa: BLE001
                desk.flash.append(f"feed pull failed: {type(e).__name__}: {e}")
        elif dg is None:
            desk.flash.append("no draft group in the pool; run ingest first")
        else:
            try:
                f = fetch_slate(desk.session, dg, update_odds=(what == "props"), starters=True, log=desk.flash.append)
                store_slate(desk.session, f, log=desk.flash.append)
                project_slate(desk.session, n_sims=1000, log=desk.flash.append)
            except IngestError as e:
                desk.flash.append(f"refresh failed: {e}")
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
                        m = str(r[slot]); pid = int(m[m.rfind("(") + 1:m.rfind(")")]) if "(" in m else None
                        pl = byid.get(pid)
                        slots.append(dict(slot=slot, pid=pid, name=pl.name if pl else m.split(" (")[0], team=pl.team if pl else "",
                                          locked=bool(pl and pl.gametime and pl.gametime <= now), fpts=proj.get(normalize_name(pl.name)) if pl else None,
                                          tip=pl.gametime if pl else None, on_slate=pl is not None))
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
        try:
            rep["when_dt"] = datetime.fromisoformat(rep["when"])
        except (KeyError, ValueError):
            rep["when_dt"] = None
    return out


def _pipeline_ctx() -> dict:
    jobs = runner.all()[:8]
    cur = runner.current() or next((j for j in jobs if j.status == "queued"), None)
    log = runner.read_log(cur, tail_lines=14) if cur else (runner.read_log(jobs[0], tail_lines=14) if jobs else "")
    return dict(form=pipeline_form, jobs=jobs, current=cur, log=log, active=bool(runner.active()), entries=_entries_info(),
                outputs=_latest_outputs(), lineups_in_db=desk.session.query(Lineup).count(),
                sims_in_db=desk.session.query(LineupSimulationResult).count())


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
        cmd = [py, str(SCRIPTS / "generate_lineups.py"), str(int(form["lineups"])), str(n_jobs), "-c", str(cfg), "--fresh", "--no-progress"]
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
    """generate | sim | choose | all. Generate (and all) publish the desk's projections first, so the pool is what you see."""
    form = await request.form()
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
            if step in ("generate", "all"):
                write_overrides(desk.overrides)
                try:
                    info = project_slate(desk.session, n_sims=2000, log=lambda m: None)
                    desk.flash.append(f"published {info.players} players with {len(desk.overrides)} edits, then queued {step}")
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


@app.get("/download/{name}")
def download(name: str):
    p = OUTPUT_DIR / Path(name).name
    if not p.exists() or not p.name.startswith(("chosen_entries.", "lateswap.")):
        return HTMLResponse("not found", status_code=404)
    return FileResponse(p, filename=p.name, media_type="text/csv")


@app.post("/slate")
def switch_slate(request: Request, draft_group: int = Form(...)):
    """Load another slate and remember the pick, so the capture tick keeps it (scripts/nba/capture.py ends on it while open)."""
    with LOCK:
        from ivan.sports.nba.paths import ACTIVE_SLATE_JSON
        try:
            ACTIVE_SLATE_JSON.write_text(json.dumps({"draft_group": int(draft_group), "chosen_at": datetime.now().isoformat(timespec="seconds")}))
        except OSError as e:
            desk.flash.append(f"could not save the slate choice ({e}); the capture tick may switch back")
        try:
            f = fetch_slate(desk.session, draft_group, update_odds=False, starters=True, log=desk.flash.append)
            store_slate(desk.session, f, log=desk.flash.append)
            project_slate(desk.session, n_sims=1000, log=desk.flash.append)
        except IngestError as e:
            desk.flash.append(f"could not load slate {draft_group}: {e}")
        desk.rebuild()
    return RedirectResponse(auth.root_of(request) + "/", status_code=303)
