# src/ivan/web/nba_desk.py
"""
NBA news desk: one page to manage a slate in the last minutes before lock.

Teams are cards sorted by how far they are from their norm (scratches, new
starters, unconfirmed lineups near tip, minutes far from history, the market
disagreeing with history, unpriced rotation players). Open a card to see the
roster with the model's minutes; type minutes to pin them and the team
re-projects in place; scratch pins 0. Publish writes projections and the
overrides file the capture loop and the sim read.

Separate process from the NFL app: python scripts/nba/serve.py (port 8001).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ivan.core.utils import PROJECT_ROOT, normalize_name
from ivan.database.session import get_session, init_db
from ivan.sports.nba.ingest import ensure_schema, fetch_slate, project_slate, store_slate, IngestError
from ivan.sports.nba.models import NBAMarketRate, NBAPlayer
from ivan.sports.nba.paths import NBA_DATA_DIR, MINUTES_OVERRIDES_CSV
from ivan.sports.nba.projection.engine import ProjectionParams
from ivan.sports.nba.projection.live import LiveSlate, read_overrides, write_overrides
from ivan.sports.nba.validation import market_points, recent_minutes, review_team

HERE = Path(__file__).parent
engine = init_db("nba")
ensure_schema(engine)
app = FastAPI(title="Ivan · NBA desk", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")
STATE_FILE = NBA_DATA_DIR / "capture_state.json"
LOCK = threading.Lock()

ROTATION_MIN = 15.0        # a scratch/questionable above this baseline is news
NEAR_TIP_MIN = 75          # minutes before tip when an unconfirmed lineup becomes a warning
MINUTES_DEV = 8.0          # model minutes this far from history is a role change
MARKET_GAP = 0.20          # market mean vs history-only projection, relative


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
        rows = self.session.query(NBAPlayer.fpts_updated_at, NBAPlayer.draft_group_id).all()
        self.session.expire_all()
        return (max((r[0] for r in rows if r[0]), default=None), len(rows), rows[0][1] if rows else None)

    def rebuild(self):
        self.session.expire_all()
        self.slate = LiveSlate(self.session)
        self.pool_stamp = self._stamp()
        self.overrides = read_overrides()
        self.proj, self.hist_only = {}, {}
        for t in self.slate.teams():
            self.reproject(t)

    def maybe_reload(self):
        if self._stamp() != self.pool_stamp:
            self.rebuild()
            self.flash.append("pool changed on disk (capture tick); reloaded")

    def reproject(self, team: str, n_sims: int = 1000):
        rng = np.random.default_rng(0)
        self.proj[team] = self.slate.project(team, self.overrides, set(), ProjectionParams(n_sims=n_sims), rng=rng)
        self.hist_only[team] = self.slate.project(team, self.overrides, set(),
                                                  ProjectionParams(n_sims=0, market_weight=0.0), rng=rng)

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

    def review(self, team: str) -> dict:
        """The rotation review (ivan.sports.nba.validation.review_team): provenance of every player's minutes and the flags."""
        if getattr(self, "_recent", None) is None or self._recent_stamp != self.pool_stamp:
            self._recent = recent_minutes(self.slate.box); self._recent_stamp = self.pool_stamp
        return review_team(self.slate, team, self.proj[team], self._recent, market_points(self.session), self.overrides)

    def team_rows(self, team: str) -> list[dict]:
        proj, hist = self.proj[team], self.hist_only[team]
        review_flags = {r["name"]: r["flags"] for r in self.review(team)["rows"]}
        state = self.slate.state
        starters = self.slate.starters.get(team)
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
            flags += [f for f in review_flags.get(name, []) if not f.startswith(("questionable", "pinned"))]
            rows.append(dict(id=p.id, name=p.name, key=name, salary=p.salary, pos=p.position, status=status if status != "NONE" else "",
                             starter=("S" if starter else ("" if starter is not None else "?")),
                             confirmed=confirmed, out=out,
                             hist_min=hist_min, minutes=None if r is None else float(r["minutes"]),
                             pinned=name in self.overrides, pin=self.overrides.get(name),
                             fpts=None if r is None else float(r["fpts_mean"]), sd=None if r is None else float(r["fpts_std"]),
                             hist_fpts=None if h is None else float(h["fpts_mean"]), market=market,
                             dk_avg=p.dk_avg_fpts, priced=bool(r is not None and r.get("priced")),
                             usage=None if r is None else float(r["usage_mult"]), flags=flags))
        rows.sort(key=lambda x: (x["out"], -(x["minutes"] or 0)))
        return rows

    def team_card(self, team: str, now: datetime) -> dict:
        rows = self.team_rows(team)
        tip = self.tip(team)
        mins_to_tip = (tip - now).total_seconds() / 60 if tip else None
        starters = self.slate.starters.get(team)
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
                elif f.startswith(("dnp-prone", "deep bench")):
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
        return dict(team=team, opp=next((p.opponent for p in self.players(team)), ""), tip=tip, mins_to_tip=mins_to_tip,
                    locked=bool(mins_to_tip is not None and mins_to_tip < 0), alerts=alerts, score=score,
                    starters="confirmed" if confirmed else ("expected" if starters else "none"),
                    n_out=sum(1 for r in rows if r["out"]), n_pinned=sum(1 for r in rows if r["pinned"]),
                    team_fpts=team_fpts, trailing=self.slate.team_fpts.get(team), rows=rows)

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
        return dict(draft_group=dg, lock=lock, mins_to_lock=(lock - now).total_seconds() / 60 if lock else None,
                    pool_at=self.pool_stamp[0], n_players=len(ps), n_priced=len(self.slate.market),
                    starters_at=self.slate.starters_fetched_at, odds_at=odds_t,
                    slates=st.get("slates", {}), props_fetches=st.get("props_fetches", 0),
                    overrides=len(self.overrides), overrides_path=str(MINUTES_OVERRIDES_CSV.relative_to(PROJECT_ROOT)))


desk = Desk()


def _render(request: Request, template: str, **ctx):
    flash = desk.flash[:]; desk.flash.clear()
    return templates.TemplateResponse(request, template, {"now": datetime.now(), "flash": flash, **ctx})


@app.get("/", response_class=HTMLResponse)
def index(request: Request, open: Optional[str] = None):
    with LOCK:
        desk.maybe_reload()
        now = datetime.now()
        cards = sorted((desk.team_card(t, now) for t in desk.slate.teams()), key=lambda c: (c["locked"], -c["score"], c["tip"] or now))
        return _render(request, "nba_desk.html", header=desk.header(now), cards=cards, open_team=open)


@app.get("/team/{team}", response_class=HTMLResponse)
def team(request: Request, team: str):
    with LOCK:
        desk.maybe_reload()
        return _render(request, "_nba_team.html", card=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/pin", response_class=HTMLResponse)
def pin(request: Request, team: str, player: str = Form(...), minutes: str = Form("")):
    with LOCK:
        key = normalize_name(player)
        if minutes.strip() == "":
            desk.overrides.pop(key, None)
        else:
            try:
                desk.overrides[key] = float(minutes)
            except ValueError:
                desk.flash.append(f"'{minutes}' is not a number")
        desk.reproject(team)
        return _render(request, "_nba_team.html", card=desk.team_card(team, datetime.now()))


@app.post("/team/{team}/clear", response_class=HTMLResponse)
def clear_team(request: Request, team: str):
    with LOCK:
        for p in desk.players(team):
            desk.overrides.pop(normalize_name(p.name), None)
        desk.reproject(team)
        return _render(request, "_nba_team.html", card=desk.team_card(team, datetime.now()))


@app.post("/publish")
def publish():
    """Write the pins, project every team at full sims, store projections + a capture row set."""
    with LOCK:
        write_overrides(desk.overrides)
        try:
            info = project_slate(desk.session, n_sims=4000, log=desk.flash.append)
            desk.flash.append(f"published: {info.players} players projected, pins saved to {MINUTES_OVERRIDES_CSV.name}")
        except IngestError as e:
            desk.flash.append(f"publish failed: {e}")
        desk.rebuild()
    return RedirectResponse("/", status_code=303)


@app.post("/refresh")
def refresh(what: str = Form("dk")):
    """dk: pool + statuses + Rotowire (free). props: also the player props (costs credits)."""
    with LOCK:
        dg = next((p.draft_group_id for p in desk.slate.players if p.draft_group_id), None)
        if dg is None:
            desk.flash.append("no draft group in the pool; run ingest first")
        else:
            try:
                f = fetch_slate(desk.session, dg, update_odds=(what == "props"), starters=True, log=desk.flash.append)
                store_slate(desk.session, f, log=desk.flash.append)
                project_slate(desk.session, n_sims=1000, log=desk.flash.append)
            except IngestError as e:
                desk.flash.append(f"refresh failed: {e}")
            desk.rebuild()
    return RedirectResponse("/", status_code=303)


@app.post("/slate")
def switch_slate(draft_group: int = Form(...)):
    with LOCK:
        try:
            f = fetch_slate(desk.session, draft_group, update_odds=False, starters=True, log=desk.flash.append)
            store_slate(desk.session, f, log=desk.flash.append)
            project_slate(desk.session, n_sims=1000, log=desk.flash.append)
        except IngestError as e:
            desk.flash.append(f"could not load slate {draft_group}: {e}")
        desk.rebuild()
    return RedirectResponse("/", status_code=303)
