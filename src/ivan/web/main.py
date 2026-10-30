# src/ivan/web/main.py
"""
The one Ivan web UI: pick the sport, then the date, then the slate.

    scripts/serve.py [--host 127.0.0.1] [--port 8000] [--sports nba,nfl,mma]      -> ivan.web.main:app

One process, one password (ivan.web.auth: IVAN_PASSWORD_HASH in .env; open when none is set and the
site is not public), one frame (ivan.web.frame). Each sport is a sub-application mounted at
/<sport> with its own database, job runner and screens:

    /                     the sports (ivan.web.main)
    /<sport>/             the sport's days                          (frame.install_picker)
    /<sport>/<day>/       that day's slates
    /<sport>/<day>/<key>/ the slate, ?view= one of the sport's views (the sport module)
    /<sport>/...          the sport's own actions, partials, exports and tools

    nba   ivan.web.nba_desk   the desk: pre-lock, post-lock (late swap), backtest
    nfl   ivan.web.nfl_app    the pipeline and its report pages, the slate's builds (backtest)
    mma   ivan.web.mma_slot   the slates only (dates, the picker, a backtest placeholder): no workspace yet

A sport whose module fails to import (a missing dependency, no database) is still listed on the
home page with the error; the rest of the site runs. IVAN_SPORTS (or --sports) limits what is
mounted, e.g. the laptop serving the NBA desk alone.
"""
from __future__ import annotations

import importlib
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ivan.web import auth
from ivan.web.frame import ctx, templates
from ivan.web.sports import REGISTRY, Sport

HERE = Path(__file__).parent
ALL_SPORTS = ("nba", "nfl", "mma")
MODULES = {"nba": "ivan.web.nba_desk", "nfl": "ivan.web.nfl_app", "mma": "ivan.web.mma_slot"}
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
        tiles = []
        for sport in REGISTRY.values():
            try:
                summary = sport.summary(now)
            except Exception as e:   # noqa: BLE001
                summary = {"loaded": None, "job": None, "note": f"could not read the state: {type(e).__name__}: {e}", "n_days": 0, "next": ""}
            tiles.append({"sport": sport, "summary": summary})
        return templates.TemplateResponse(request, "home.html", {"frame": ctx(request, None, now=now), "tiles": tiles})

    REGISTRY.clear()
    for key in _wanted(sports):
        sport = _load(key)
        REGISTRY[key] = sport
        if sport.app is not None:
            app.mount(f"/{key}", sport.app, name=key)
    return app


app = build_app()
