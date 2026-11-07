"""
The team page's routes: pins, starter calls, the rotation grid, scenarios, notes, the late-news call. Every edit re-projects the team and writes it into the pool (Desk.reproject).
"""
from __future__ import annotations

from datetime import datetime
from ivan.sports.nba import scenarios as SC
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import Form, Request
from fastapi.responses import HTMLResponse
from ivan.core.utils import normalize_name
from ivan.sports.nba.ingest import refresh_public_projection
from ivan.sports.nba.projection.live import override_of
from ivan.sports.nba.projection import rotation as ROT
from .core import _active, _busy, _render, desk, LOCK, slate
from .edits import _apply_pin_rows, _num
from .knobs import pipeline_form
from .pipeline import _submit
from ivan.sports.nba import slates as S


@slate.get("/team/{team}", response_class=HTMLResponse)
def team(request: Request, team: str):
    with LOCK:
        desk.maybe_reload()
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/pin", response_class=HTMLResponse)
def pin(request: Request, team: str, player: str = Form(...), minutes: str = Form(""), fppm: str = Form(""), low: str = Form(""), high: str = Form(""),
        fpts: str = Form(""), pub: str = Form(""), points: str = Form(""), rebounds: str = Form(""), assists: str = Form(""), fg3m: str = Form(""),
        steals: str = Form(""), blocks: str = Form(""), turnovers: str = Form("")):
    """One row's hand values: blank clears that field. Minutes 0 scratches. A stat total (points, rebounds, assists, fg3m, steals,
    blocks, turnovers) pins that stat for the game: its per-minute rate becomes total / minutes, the other stats keep the model's."""
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
        stat_vals = {k: _num(v, k) for k, v in dict(points=points, rebounds=rebounds, assists=assists, fg3m=fg3m, steals=steals, blocks=blocks, turnovers=turnovers).items()}
        if new_min is not None and new_min != old.get("minutes"):
            # typed minutes repaint the row: the lock (if any) is lifted so the skeleton re-fills him to the number, and the
            # other free rows move around him; a row you still want held exactly, paint it again
            g = ROT.load_drawn(team)
            if g is not None and key in set(g.attrs.get("locked", [])):
                ROT.save_drawn(team, g, locked=[n for n in g.attrs.get("locked", []) if n != key], fingerprint=g.attrs.get("fingerprint"))
        desk.set_override(key, minutes=new_min, fppm=new_fppm, min_low=lo, min_high=hi, fpts=new_fpts, pub=new_pub, **stat_vals)
        desk.reproject(team)                                  # the solve re-balances the team around the edit; the grid is rebuilt to the pins
        if new_pub != old.get("pub"):
            # the public number moved: the stored public projection follows at once; the field built on it waits for the next
            # sim or the Field + ownership button (2026-10-28: no field build on an edit)
            try:
                refresh_public_projection(desk.session, log=lambda m: None)
                desk.flash.append(f"{key}: public projection {'set to %g' % new_pub if new_pub is not None else 'back to the replica'}; the field follows on the next sim or Field + ownership")
            except Exception as e:   # noqa: BLE001
                desk.flash.append(f"{key}: public projection saved, but not written to the pool: {type(e).__name__}: {e}")
            desk.rebuild()
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/refigure", response_class=HTMLResponse)
async def refigure(request: Request, team: str):
    """
    Body: JSON {rows: [{player, minutes, fppm, fpts, pub, points, rebounds, ...}, ...]}: every typed value of the team's grid, applied in one go (blank
    clears that pin), then the team re-projected once: the skeleton re-fills the grid to the typed minutes (a row whose minutes
    changed loses its lock so it can move), the other free rows move around them. The public numbers that changed rebuild the field.
    """
    if (b := _busy(request)) is not None:
        return b
    body = await request.json()
    with LOCK:
        changed_min, changed_pub, n_set = _apply_pin_rows(team, body.get("rows") or [])
        desk.reproject(team)
        msg = f"{team}: {n_set} value(s) applied, the grid refigured" + (f" ({len(changed_min)} minutes changed)" if changed_min else "")
        if changed_pub:
            try:
                refresh_public_projection(desk.session, log=lambda m: None)
                msg += f"; {changed_pub} public number(s) changed: the field follows on the next sim or Field + ownership"
            except Exception as e:   # noqa: BLE001
                msg += f"; the public projection was not written to the pool: {type(e).__name__}: {e}"
            desk.rebuild()
        desk.flash.append(msg)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/note", response_class=HTMLResponse)
def player_note(request: Request, team: str, player: str = Form(...), text: Optional[str] = Form(None)):
    """A note next to a player on this slate (ivan.sports.nba.notes = scripts/nba/notes.py): the text comes from the form or
    the prompt (HX-Prompt); a blank text clears it. It shows while this slate is loaded and goes into its scorecard."""
    from ivan.sports.nba import notes as NOTES
    text = text if text is not None else request.headers.get("HX-Prompt", "")
    with LOCK:
        key = normalize_name(player)
        p = next((x for x in desk.players(team) if normalize_name(x.name) == key), None)
        dg = desk.draft_group()
        if p is None or dg is None:
            desk.flash.append(f"no {player} on {team} in the loaded slate")
        else:
            lock = min((x.gametime for x in desk.slate.players if x.gametime), default=None)
            NOTES.set_note(desk.session, dg, lock.date().isoformat() if lock else None, key, text, team=team, raw=p.name)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/starter", response_class=HTMLResponse)
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


@slate.get("/team/{team}/rotation", response_class=HTMLResponse)
def rotation_view(request: Request, team: str):
    with LOCK:
        return _render(request, "_nba_rotation.html", rot=desk.rotation(team))


@slate.post("/team/{team}/rotation", response_class=HTMLResponse)
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


@slate.post("/team/{team}/rotation/propose", response_class=HTMLResponse)
def rotation_propose(request: Request, team: str, source: str = Form("propose"), game_id: str = Form(""), overlay: str = Form("")):
    """source=propose: tonight's grid with every lock lifted (the top panel, nothing saved); source=last: a played game for the panel
    under it, with our grid of that day over it when `overlay` is set (the panel's "vs our grid" box)."""
    with LOCK:
        if source == "last":
            return _render(request, "_nba_rotation.html", rot=desk.game_grid(team, game_id=game_id.strip() or None, overlay=bool(overlay.strip())))
        return _render(request, "_nba_rotation.html", rot=desk.rotation(team, source="propose"))


@slate.post("/team/{team}/rotation/add", response_class=HTMLResponse)
def rotation_add(request: Request, team: str, player: str = Form(...)):
    """A player the rotation left off the floor, back in: a minutes pin at his history; the grid schedules him."""
    with LOCK:
        mins = desk.add_back(team, player)
        desk.flash.append(f"{team}: {player} added back at {mins:.0f} minutes (type another number to change it)")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/rotation/drop", response_class=HTMLResponse)
def rotation_drop(request: Request, team: str, player: str = Form(...)):
    """No minutes tonight: the player leaves the grid and the pool (a minutes pin of 0; the + beside his name brings him back)."""
    with LOCK:
        desk.set_override(normalize_name(player), minutes=0.0)
        desk.reproject(team)
        desk.flash.append(f"{team}: {player} dropped from the grid (no minutes tonight)")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/rotation/restore", response_class=HTMLResponse)
def rotation_restore(request: Request, team: str, player: str = Form(...)):
    """Lift a hand drop: the minutes pin goes and the model fills him again."""
    with LOCK:
        desk.set_override(normalize_name(player), minutes=None)
        desk.reproject(team)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/rotation/clear", response_class=HTMLResponse)
def rotation_clear(request: Request, team: str):
    with LOCK:
        n = desk.clear_rotation(team)
        back = "the published grid and pins" if ROT.drawn_source(team) == "published" or S.published_pins() is not None else "the model's"
        desk.flash.append(f"{team}: locks cleared" + (f", {n} typed minutes lifted" if n else "") + f"; the grid is {back}")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/clear", response_class=HTMLResponse)
def clear_team(request: Request, team: str):
    with LOCK:
        for p in desk.players(team):
            desk.overrides.pop(normalize_name(p.name), None)
        desk.write_local()
        desk.reproject(team)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/scenario/save", response_class=HTMLResponse)
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


@slate.post("/team/{team}/scenario/activate", response_class=HTMLResponse)
def scenario_activate(request: Request, team: str, slug: str = Form(...)):
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        sc = desk.activate_scenario(team, slug)
        desk.flash.append(f"{team}: scenario '{sc['name']}' active" if sc else f"{team}: no scenario {slug}")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/scenario/delete", response_class=HTMLResponse)
def scenario_delete(request: Request, team: str, slug: str = Form(...)):
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        SC.delete_scenario(team, slug)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/news", response_class=HTMLResponse)
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
        desk.flash.append(f"{team}: {player} {call}: the team is re-projected and written to the pool ({len(desk.overrides)} edits on the slate)")
        if then in ("sim", "lateswap"):
            if _active():
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
