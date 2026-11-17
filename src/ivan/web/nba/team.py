"""
The team page's routes: pins, starter calls, Simulate team, the mapping, the hovers, a played game's grid, scenarios, notes. Every edit re-projects the team and writes it into the pool (Desk.reproject).
"""
from __future__ import annotations

from datetime import datetime
from ivan.sports.nba.slate import scenarios as SC
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from ivan.core.utils import normalize_name
from ivan.sports.nba.slate import changes as CH
from ivan.sports.nba.ingest import refresh_market_rates, refresh_public_projection
from ivan.sports.nba.projection.live import override_of
from ivan.sports.nba.projection import rotation as ROT
from .core import _busy, _render, desk, LOCK, slate
from .edits import _apply_pin_rows, _num
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
        desk.set_override(key, minutes=new_min, fppm=new_fppm, min_low=lo, min_high=hi, fpts=new_fpts, pub=new_pub, **stat_vals)
        desk.reproject(team)                                  # the solve re-balances the team around the edit and the game is drawn again
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
    clears that pin), then the team re-projected once: the solve gives the players without a pin the minutes left and the game
    is drawn again (the heatmap, every drawn number and the hovers' distributions). The public numbers that changed rebuild the field.
    """
    if (b := _busy(request)) is not None:
        return b
    body = await request.json()
    with LOCK:
        changed_min, changed_pub, n_set = _apply_pin_rows(team, body.get("rows") or [])
        desk.reproject(team)
        msg = f"{team}: {n_set} value(s) applied, the game drawn again" + (f" ({len(changed_min)} minutes changed)" if changed_min else "")
        if changed_pub:
            try:
                refresh_public_projection(desk.session, log=lambda m: None)
                msg += f"; {changed_pub} public number(s) changed: the field follows on the next sim or Field + ownership"
            except Exception as e:   # noqa: BLE001
                msg += f"; the public projection was not written to the pool: {type(e).__name__}: {e}"
            desk.rebuild()
        desk.flash.append(msg)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


SIM_MIN, SIM_MAX = 250, 20000      # the Simulate team box's range of worlds


@slate.post("/team/{team}/simulate", response_class=HTMLResponse)
def simulate_team(request: Request, team: str, n: str = Form("")):
    """Simulate team: the team's game drawn again in `n` worlds (blank = config projection.n_worlds) from the pool as it stands, and
    written as the team's projection (Desk.reproject = ingest.project_team with n_worlds; terminal: scripts/nba/project.py --slate DG
    --team TEAM --worlds N). The heatmap, the drawn numbers and every hover's distribution are these worlds until the next draw
    (a Refigure or any edit draws at the config's number again)."""
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        try:
            worlds = max(SIM_MIN, min(SIM_MAX, int(float(n)))) if str(n).strip() else None
            desk.reproject(team, n_worlds=worlds)
            desk.flash.append(f"{team}: game simulated in {int((desk.world.get(team) or {}).get('n_worlds') or 0)} worlds")
        except Exception as e:   # noqa: BLE001
            desk.flash.append(f"{team}: not simulated: {type(e).__name__}: {e}")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/note", response_class=HTMLResponse)
def player_note(request: Request, team: str, player: str = Form(...), text: Optional[str] = Form(None)):
    """A note next to a player on this slate (ivan.sports.nba.slate.notes = scripts/nba/notes.py): the text comes from the form or
    the prompt (HX-Prompt); a blank text clears it. It shows while this slate is loaded and goes into its scorecard."""
    from ivan.sports.nba.slate import notes as NOTES
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
        # a call that benches him: his props priced the starter; the lines posted before this are set aside and the market rates
        # rebuilt without them (he projects from history until the books post again), then the team re-projects on the new rates.
        # A call that starts him (or clears back to a start) lifts the stamp; nothing else moves.
        was = (float(cur) > 0) if cur is not None else bool(rw)
        now_starting = (float(new) > 0) if new is not None else bool(rw)
        if was and not now_starting:
            CH.touch_status(desk.session, key)
        elif now_starting and not was:
            CH.touch_status(desk.session, key, clear=True)
        try:
            refresh_market_rates(desk.session, {normalize_name(p.name): p.gametime for p in desk.slate.players}, {normalize_name(p.name) for p in desk.slate.players},
                                 history=desk.history())
            desk.session.commit(); desk.slate.reload_market(); desk.load_market_rows()
        except Exception as e:   # noqa: BLE001 - the call stands; the rates follow on the next refresh
            desk.session.rollback(); desk.flash.append(f"{key}: starter call saved; market rates not rebuilt ({type(e).__name__}: {e})")
        desk.reproject(team)
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.get("/team/{team}/props/{player}/{stat}", response_class=HTMLResponse)
def props_hover(request: Request, team: str, player: str, stat: str):
    """A cell's hover, fetched the first time the cell is hovered (htmx, mouseenter once): the distribution the World drew for the
    player's stat tonight over his own games, and every book's line and price for it (Desk.stat_hover; `_nba_props.html`). `stat` is
    one of the seven stat columns, or `minutes` or `fpts` (the min and fpts cells: the same chart from the same draw, no books).
    Built on request so the card itself stays quick (2026-11-07: building every player's lines into the card made Refigure slow)."""
    from ivan.sports.nba.desk.rows import HOVER_STATS
    if stat not in HOVER_STATS:
        raise HTTPException(status_code=404, detail=f"no stat {stat}")
    with LOCK:
        desk.maybe_reload()
        return _render(request, "_nba_props.html", s=desk.stat_hover(team, normalize_name(player), stat))


@slate.post("/team/{team}/map", response_class=HTMLResponse)
def map_team(request: Request, team: str):
    """Map the team (Desk.map_team; scripts/nba/mapping.py --map TEAM): every player with minutes pinned at the minutes he shows and the
    pool it was looked at against remembered; the card says 'changed' when a status, the availability or the five moves."""
    if (b := _busy(request)) is not None:
        return b
    with LOCK:
        try:
            n = desk.map_team(team)
            desk.flash.append(f"{team}: mapped, {n} player(s) held at their minutes")
        except Exception as e:   # noqa: BLE001
            desk.flash.append(f"{team}: not mapped: {type(e).__name__}: {e}")
        return _render(request, "_nba_side.html", c=desk.team_card(team, datetime.now()))


@slate.post("/team/{team}/rotation/game", response_class=HTMLResponse)
def rotation_game(request: Request, team: str, game_id: str = Form(""), overlay: str = Form("")):
    """A played game for the panel under tonight's heatmap (the picker), with our grid of that day over it when `overlay` is set
    (the panel's "vs our grid" box)."""
    with LOCK:
        return _render(request, "_nba_rotation.html", rot=desk.game_grid(team, game_id=game_id.strip() or None, overlay=bool(overlay.strip())))


@slate.post("/team/{team}/rotation/add", response_class=HTMLResponse)
def rotation_add(request: Request, team: str, player: str = Form(...)):
    """A player the rotation left off the floor, back in: a minutes pin at his history."""
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
        back = "the published pins" if ROT.drawn_source(team) == "published" or S.published_pins() is not None else "the model's"
        desk.flash.append(f"{team}: mapping cleared" + (f", {n} minutes pin(s) lifted" if n else "") + f"; the minutes are {back}")
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
    """Save the team's current edits (its pins) under a name; the assumptions are read off the calls in them."""
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
