# src/ivan/web/main.py
"""
The one Ivan web UI: pick the sport, then the date, then the slate.

    scripts/serve.py [--host 127.0.0.1] [--port 8000] [--sports nba,nfl,mma]      -> ivan.web.main:app

One process, one password (ivan.web.auth: IVAN_PASSWORD_HASH in .env; open when none is set and the
site is not public), one frame (ivan.web.frame). Each sport is a sub-application mounted at
/<sport> with its own database, job runner and screens:

    /                     the sports, each tile with its money, and the money over every sport with its tax (ivan.web.main, ivan.core.pnl)
    /<sport>/             the sport's days                          (frame.install_picker)
    /<sport>/<day>/       that day's slates
    /<sport>/<day>/<key>/ the slate, ?view= one of the sport's views (the sport module)
    /<sport>/...          the sport's own actions, partials, exports and tools

    nba   ivan.web.nba        the desk: pre-lock, post-lock (late swap), backtest, the team page
    nfl   ivan.web.nfl        the desk, the pipeline and its report pages, the slate's builds (backtest)
    mma   ivan.web.mma        the slates only (dates, the picker, a backtest placeholder): no workspace yet

A sport whose module fails to import (a missing dependency, no database) is still listed on the
home page with the error; the rest of the site runs. IVAN_SPORTS (or --sports) limits what is
mounted, e.g. the laptop serving the NBA desk alone.
"""
from __future__ import annotations

import importlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ivan.core import pnl
from ivan.web import auth
from ivan.web.frame import ctx, templates
from ivan.web.sports import REGISTRY, Sport

HERE = Path(__file__).parent
ALL_SPORTS = ("nba", "nfl", "mma")
MODULES = {"nba": "ivan.web.nba", "nfl": "ivan.web.nfl", "mma": "ivan.web.mma"}
LABELS = {"nba": "NBA", "nfl": "NFL", "mma": "MMA"}


def _wanted(sports) -> list[str]:
    raw = sports or os.environ.get("IVAN_SPORTS", "")
    keys = [s.strip().lower() for s in (raw.split(",") if isinstance(raw, str) else raw) if s.strip()]
    return [k for k in (keys or ALL_SPORTS) if k in ALL_SPORTS]


def _load(key: str) -> Sport:
    """Import the sport's web module and take its `sport()` description; a failure leaves a listed sport with no workspace."""
    try:
        mod = importlib.import_module(MODULES[key])
        return mod.sport()
    except Exception as e:   # noqa: BLE001 - one sport's trouble must not take the site down
        print(f"{key}: workspace not mounted: {type(e).__name__}: {e}", flush=True)
        return Sport(key=key, label=LABELS[key], views=[("backtest", "backtest")], days=lambda now: [],
                     summary=lambda now: {"loaded": None, "job": None, "note": "", "n_days": 0, "next": ""}, error=f"{type(e).__name__}: {e}")


def sport_ledgers(now: datetime) -> dict:
    """{sport key: its results ledger} for the sports that give one (Sport.ledger: None on a machine that shows no money, and for a
    sport with no results yet); a ledger that cannot be read is {"error": ...}, and the tile says so."""
    out = {}
    for sport in REGISTRY.values():
        if sport.ledger is None:
            continue
        try:
            out[sport.key] = sport.ledger(now)
        except Exception as e:   # noqa: BLE001 - the home page opens without it
            out[sport.key] = {"error": f"{type(e).__name__}: {e}", "overall": None, "days": {}}
    return out


def money(ledgers: dict, now: datetime) -> Optional[dict]:
    """The home page's strip: this year's results over every sport and the tax on them (ivan.core.pnl.combined; scripts/pnl.py).
    None when no sport has a result to show."""
    try:
        m = pnl.combined(ledgers, now.year, labels={k: s.label for k, s in REGISTRY.items()})
    except Exception as e:   # noqa: BLE001 - a broken config/tax.json must not take the home page down
        return {"error": f"{type(e).__name__}: {e}"}
    return m if (m["total"]["contests"] or m["total"]["unscored"]) else None


def build_app(sports=None) -> FastAPI:
    app = FastAPI(title="Ivan", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    auth.install(app, templates)                      # the one password, for every path below

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return FileResponse(HERE / "static" / "favicon.svg", media_type="image/svg+xml")

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        now = datetime.now()
        ledgers = sport_ledgers(now)
        tiles = []
        for sport in REGISTRY.values():
            try:
                summary = sport.summary(now)
            except Exception as e:   # noqa: BLE001
                summary = {"loaded": None, "job": None, "note": f"could not read the state: {type(e).__name__}: {e}", "n_days": 0, "next": ""}
            tiles.append({"sport": sport, "summary": summary, "ledger": ledgers.get(sport.key)})
        return templates.TemplateResponse(request, "home.html", {"frame": ctx(request, None, now=now), "tiles": tiles, "pnl": money(ledgers, now)})

    REGISTRY.clear()
    for key in _wanted(sports):
        sport = _load(key)
        REGISTRY[key] = sport
        if sport.app is not None:
            app.mount(f"/{key}", sport.app, name=key)
    return app


app = build_app()
