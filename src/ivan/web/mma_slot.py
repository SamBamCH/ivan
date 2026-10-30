# src/ivan/web/mma_slot.py
"""
MMA in the one UI: the slot only. The picker lists the cards (the DraftKings lobby's MMA slates and
every draft group whose contests ingest.py stored), a slate page says what is on file for it, and
the backtest view says what settle has not built yet. The pipeline (scripts/mma/*) has no buttons
here yet; docs/mma/README.md is the flow.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func

from ivan.database.session import get_session, init_db
from ivan.sports.mma.models import HistoricalContest, MMABase, MMAPlayer
from ivan.web import frame
from ivan.web.sports import Day, SlateInfo, Sport, find_slate, group_days, lobby_day, lobby_lock

HERE = Path(__file__).parent
KEY, LABEL = "mma", "MMA"
VIEWS = [("slate", "slate"), ("backtest", "backtest")]
engine = init_db(KEY)
MMABase.metadata.create_all(engine)
_lobby: dict = {"at": 0.0, "items": []}


def lobby(max_age: float = 120) -> list[dict]:
    if time.time() - _lobby["at"] > max_age:
        try:
            from ivan.sports.common.draftkings import list_slates
            _lobby.update(at=time.time(), items=list_slates("mma"))
        except Exception:   # noqa: BLE001 - the lobby is a convenience
            _lobby.update(at=time.time(), items=[])
    return _lobby["items"]


def days(now: datetime) -> list[Day]:
    session = get_session(engine)
    try:
        stored = session.query(HistoricalContest.draft_group_id, func.min(HistoricalContest.date), func.count(HistoricalContest.contest_id),
                               func.max(HistoricalContest.total_entries)).group_by(HistoricalContest.draft_group_id).all()
        n_pool = session.query(func.count(MMAPlayer.id)).scalar() or 0
    finally:
        session.close()
    slates: dict[str, SlateInfo] = {}
    for s in lobby():
        key = str(s["id"]); lock = lobby_lock(s.get("start_utc"))
        slates[key] = SlateInfo(key=key, day=lobby_day(s.get("start"), s.get("start_utc"), now), lock=lock,
                                label=f"{s.get('game_type') or ''} · {str(s.get('name') or '')[:40]}".strip(" ·"), source="lobby",
                                mode=str(s.get("game_type") or "").lower(), detail=f"{s.get('contests') or 0} contests in the lobby")
    # the working pool: the newest stored card when fighters are on file (mma_player carries no draft group)
    working = max(stored, key=lambda r: (r[1] or datetime.min)) if (stored and n_pool) else None
    for dg, first, n, biggest in stored:
        key = str(dg)
        when = first or now
        if key in slates:
            slates[key].detail = f"{n} contests stored" + (f" · largest {biggest:,} entries" if biggest else "")
            slates[key].source = "working" if working is not None and working[0] == dg else slates[key].source
            slates[key].loaded = working is not None and working[0] == dg
            continue
        slates[key] = SlateInfo(key=key, day=when.date(), lock=first, label=f"card of {when:%a %m-%d}", mode="classic",
                                source="working" if working is not None and working[0] == dg else "archive",
                                loaded=working is not None and working[0] == dg,
                                detail=f"{n} contests stored" + (f" · largest {biggest:,} entries" if biggest else ""))
    return group_days(list(slates.values()), now)


def summary(now: datetime) -> dict:
    ds = days(now)
    loaded = next((s for d in ds for s in d.slates if s.loaded), None)
    ahead = [d for d in ds if d.day >= now.date()]
    nxt = min(ahead, key=lambda d: d.day) if ahead else None
    return {"loaded": loaded, "job": None, "note": "slates and contests only: the pipeline runs from scripts/mma/ (docs/mma/README.md)",
            "n_days": len(ds), "next": nxt.label if nxt else ""}


def sport() -> Sport:
    app = FastAPI(title="Ivan · MMA", docs_url=None, redoc_url=None)
    sp = Sport(key=KEY, label=LABEL, views=VIEWS, days=days, summary=summary, tools=[], app=app,
               default_view=lambda s, now: "backtest" if s.day < now.date() else "slate")
    frame.install_picker(app, sp)

    @app.get("/{day}/{key}/", response_class=HTMLResponse)
    def slate_page(request: Request, day: str, key: str, view: Optional[str] = None):
        now = datetime.now()
        d = frame.parse_day(day)
        ds = days(now)
        s = find_slate(ds, key)
        if s is None:
            s = SlateInfo(key=str(key), day=d, lock=None, label=f"slate {key}", source="archive", detail="nothing on file for this slate")
        view = view if view in dict(VIEWS) else sp.default_view(s, now)
        session = get_session(engine)
        try:
            contests = session.query(HistoricalContest).filter(HistoricalContest.draft_group_id == int(key)).order_by(HistoricalContest.entry_fee.desc()).all() if str(key).isdigit() else []
            fighters = session.query(func.count(MMAPlayer.id)).scalar() if s.loaded else 0
        finally:
            session.close()
        return frame.templates.TemplateResponse(request, "mma_slate.html", {"frame": frame.ctx(request, sp, ds, None, s, view, now), "slate": s, "view": view,
                                                                            "contests": contests, "fighters": fighters})

    return sp
