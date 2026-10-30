# src/ivan/web/frame.py
"""
The frame every Ivan page shares: the bar (Ivan › sport › date › slate › view, then the sport's
tools and sign out) and the picker page a sport opens on (its days on the left, the chosen day's
slates on the right). Rendered from the sport's description (ivan.web.sports.Sport) so every sport
looks the same; the slate's own screens are the sport module's.

    ctx(request, sport, days, day, slate, view)   the template context the bar needs ("frame")
    install_picker(app, sport)                     GET /  and  GET /{day}/  on a sport's sub-app
    templates                                      the shared Jinja environment (frame, home, picker)
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ivan.web import auth
from ivan.web.sports import REGISTRY, Day, SlateInfo, Sport, pick_day

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")


def site_root(request: Request) -> str:
    """The main app's root: the sport sub-apps are mounted one level below it."""
    root = auth.root_of(request)
    for sport in REGISTRY:
        if root.endswith(f"/{sport}"):
            return root[: -len(sport) - 1]
    return root


def ctx(request: Request, sport: Optional[Sport], days: Optional[list[Day]] = None, day: Optional[Day] = None,
        slate: Optional[SlateInfo] = None, view: Optional[str] = None, now: Optional[datetime] = None) -> dict:
    """What `_frame.html` renders: the sports, this sport's days, the day open, the slate open and its view."""
    now = now or datetime.now()
    site = site_root(request)
    root = f"{site}/{sport.key}" if sport else site
    if sport and days is None:
        days = sport.days(now)
    if sport and slate is not None and day is None:
        day = next((d for d in days or [] if d.day == slate.day), None)
    return {"site": site, "root": root, "sport": sport, "sports": list(REGISTRY.values()), "days": days or [], "day": day, "slate": slate,
            "view": view, "views": sport.views if sport else [], "tools": sport.tools if sport else [],
            "login": auth.configured(), "now": now}


def install_picker(app: FastAPI, sport: Sport) -> None:
    """The sport's landing pages: `/` (its days, the natural day open) and `/{day}/` (that day's slates)."""

    def page(request: Request, wanted: Optional[date]):
        now = datetime.now()
        days = sport.days(now)
        day = pick_day(days, wanted, now)
        if wanted is not None and day is None:
            day = Day(day=wanted, label=f"{wanted:%a %m-%d}", tag="", slates=[])
        return templates.TemplateResponse(request, "picker.html", {"frame": ctx(request, sport, days, day, None, None, now), "day": day, "days": days})

    @app.get("/", response_class=HTMLResponse)
    def picker_root(request: Request):
        return page(request, None)

    @app.get("/{day}/", response_class=HTMLResponse)
    def picker_day(request: Request, day: str):
        try:
            d = date.fromisoformat(day)
        except ValueError:
            raise HTTPException(404, "not a day")
        return page(request, d)


def parse_day(day: str) -> date:
    try:
        return date.fromisoformat(day)
    except ValueError:
        raise HTTPException(404, "not a day")
