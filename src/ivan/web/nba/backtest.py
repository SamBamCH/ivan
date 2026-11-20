"""
The backtest view's own routes (a past slate, read only; the page is `_nba_backtest.html`, the builder
ivan.sports.nba.backtest.slate = scripts/nba/backtest_slate.py):

    GET  /<day>/<dg>/backtest/fold/<what>        a folded section built when opened: money, rerun, players (`_nba_bt_fold.html`)
    GET  /<day>/<dg>/backtest/team/<TEAM>        the team's panel: its game as played over our grid, its rows (`_nba_bt_team.html`)
    GET  /<day>/<dg>/backtest/contest/<id>       one contest drilled into: our entries, the field's ownership, the payouts (`_nba_bt_contest.html`)
    GET  /<day>/<dg>/backtest/fetch              the readiness strip: what is on file, the fetch buttons, the running fetch (`_nba_bt_fetch.html`)
    POST /<day>/<dg>/backtest/fetch/standings    scripts/nba/standings.py --slate <dg> scrape, as a job (the final exports of this slate's contests)
    POST /<day>/<dg>/backtest/fetch/boxscores    scripts/nba/nightly.py --max-games 40, as a job (the box scores, then the scorecards)
    GET  /<day>/<dg>/backtest/game/<AWAY-HOME>   one game drilled into (`_nba_bt_game.html`; ivan.sports.nba.backtest.game = scripts/nba/backtest_game.py):
                                                 both teams as published, as drawn from the tip record and as played, the inputs open to change
    POST /<day>/<dg>/backtest/game/<key>/simulate  draw the game with the form's edits (do = simulate), with every player at the minutes he
                                                 played (do = actual), or drop the edits (do = reset)
    GET  /<day>/<dg>/backtest/game/<key>/player/<name>   one player's drawn distributions against what he did (`_nba_bt_player.html`)

The slate's page shows its games (ivan.sports.nba.backtest.game.overview) and keeps the money, the whole-slate rerun and the
player table folded, each built when it is opened; a game opens with ?view=backtest&game=<AWAY-HOME> in the address (pages.board reads it off the page's address).

The slate's backtest is built once and kept for CACHE_SECONDS (about 4 s to build), so the panels open at once; a fetch job
that ends drops it and reloads the page.
"""
from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool

from ivan.database.session import get_session
from ivan.sports.nba import slates as S
from ivan.sports.nba.desk import CONSUMER
from .core import LIMITED, _render, bound, engine, runner, slate, slate_jobs

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts" / "nba"
CACHE_SECONDS = 120
FETCH_STEPS = {"standings": "fetch_standings", "boxscores": "fetch_boxscores"}
_CACHE: dict[int, tuple[float, dict, Optional[str]]] = {}
_CACHE_LOCK = threading.Lock()


def backtest_of(dg: int, fresh: bool = False) -> tuple[Optional[dict], Optional[str]]:
    """(the slate's backtest, the error): built by ivan.sports.nba.backtest.slate on the slate's own database when this machine holds
    it (its contests, output files, edits), else on the history (its captures); kept CACHE_SECONDS."""
    from ivan.sports.nba.backtest import slate as SB
    dg = int(dg)
    with _CACHE_LOCK:
        hit = _CACHE.get(dg)
        if hit is not None and not fresh and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1], hit[2]
    sess = S.session(dg) if S.holds(dg) else get_session(engine)      # its own session: the page never waits on the desk's lock
    try:
        with S.using(dg):
            bt, err = SB.slate_backtest(sess, dg), None
    except Exception as e:   # noqa: BLE001 - show what failed instead of a blank page
        bt, err = None, f"{type(e).__name__}: {e}"
    finally:
        sess.close()
    with _CACHE_LOCK:
        _CACHE[dg] = (time.time(), bt, err)
    return bt, err


def forget(dg: int) -> None:
    with _CACHE_LOCK:
        _CACHE.pop(int(dg), None)


def _dg(request: Request) -> int:
    dg = getattr(request.state, "dg", None)
    if dg is None:
        raise HTTPException(404, "a backtest panel belongs to a slate's page")
    return int(dg)


def _fetch_jobs(dg: int) -> list:
    return [j for j in slate_jobs(dg) if (j.meta or {}).get("step") in FETCH_STEPS.values()]


def fetch_ctx(dg: int) -> dict:
    """What the readiness strip shows: the slate's state (ivan.sports.nba.backtest.readiness.slate_status: the box scores on file, the
    money's exports), the fetch job running or just ended, whether this desk can fetch (the laptop: a DraftKings login for the
    standings, nba_api for the box scores)."""
    from ivan.sports.nba.backtest import readiness
    st = None
    if S.holds(dg):
        try:
            st = readiness.slate_status(dg)
        except Exception as e:   # noqa: BLE001
            st = dict(state="failed", why=f"readiness could not be read: {type(e).__name__}: {e}", box=None, money=None)
    jobs = _fetch_jobs(dg)
    active = next((j for j in jobs if j.is_active), None)
    last = jobs[0] if jobs else None
    just_finished = bool(last is not None and not last.is_active and last.finished and (time.time() - last.finished) < 8)
    box = (st or {}).get("box") or {}
    money = (st or {}).get("money") or {}
    return dict(fetch=dict(status=st, job=active, last=last, just_finished=just_finished,
                           log=runner.read_log(active or last, tail_lines=5) if (active is not None or (last is not None and last.status == "failed")) else "",
                           can=bool(S.holds(dg)) and not CONSUMER, standings_ok=not LIMITED,
                           box_missing=bool(box.get("missing")) or (box.get("teams") == 0),
                           money_open=bool(money.get("played")) and not money.get("complete")))


@slate.get("/backtest/fetch", response_class=HTMLResponse)
def fetch_strip(request: Request):
    """The readiness strip, polled every 3 s while a fetch job runs; the poll that finds it just done forgets the cached backtest and
    reloads the page (HX-Refresh), so the new exports or box scores are what the page shows."""
    dg = _dg(request)
    ctx = fetch_ctx(dg)
    if ctx["fetch"]["just_finished"] and ctx["fetch"]["last"].status == "done":
        forget(dg)
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    return _render(request, "_nba_bt_fetch.html", **ctx)


@slate.post("/backtest/fetch/{what}", response_class=HTMLResponse)
def fetch_run(request: Request, what: str):
    """Queue the fetch of what the backtest is missing for this slate: `standings` = scripts/nba/standings.py --slate <dg> scrape (every
    contest of ours and the largest public ones without a final export, through the logged-in Chrome), `boxscores` =
    scripts/nba/nightly.py --max-games 40 (the season's box scores not yet stored, this day's among them, then the scorecards)."""
    dg = _dg(request)
    if what not in FETCH_STEPS:
        raise HTTPException(404, "standings or boxscores")
    d = bound()
    ctx = fetch_ctx(dg)
    if d is None or not ctx["fetch"]["can"]:
        d and d.flash.append("this slate is not held here: nothing to fetch into")
    elif what == "standings" and LIMITED:
        d.flash.append("no DraftKings login on this desk: scrape the standings on the laptop")
    elif ctx["fetch"]["job"] is not None:
        d.flash.append("a fetch is already running for this slate")
    else:
        if what == "standings":
            lock = S.meta_time(S.read_meta(dg), "lock")
            days = max(1, (datetime.now() - lock).days + 2) if lock else 7
            cmd = [sys.executable, str(SCRIPTS / "standings.py"), "--slate", str(dg), "scrape", "--days", str(days)]
        else:
            cmd = [sys.executable, str(SCRIPTS / "nightly.py"), "--max-games", "40"]
        runner.submit(FETCH_STEPS[what], cmd, meta=dict(step=FETCH_STEPS[what], slate=dg))
    return _render(request, "_nba_bt_fetch.html", **fetch_ctx(dg))


@slate.get("/backtest/fold/{what}", response_class=HTMLResponse)
def fold(request: Request, what: str):
    """One folded section of the first screen, built when it is opened: money (the tiles, the contests), rerun (the whole slate by the
    current code), players (every player in one table). The slate's backtest is built here (about 5 s, then kept), not for the games."""
    dg = _dg(request)
    if what not in ("money", "rerun", "players"):
        raise HTTPException(404, "money, rerun or players")
    if what == "rerun":
        from .rerun import rerun_ctx
        try:
            rr = rerun_ctx(dg, request.query_params.get("stat")) if S.holds(dg) else {}
        except Exception:   # noqa: BLE001 - a record from another version of the code: the block says there is none
            rr = {}
        return _render(request, "_nba_bt_fold.html", what=what, bt=None, bt_error=None, **rr)
    if what == "money" and CONSUMER:
        return HTMLResponse('<p class="muted small">The entries\' results are shown on the laptop only; this desk shows the model against the games.</p>')
    bt, err = backtest_of(dg)
    return _render(request, "_nba_bt_fold.html", what=what, bt=bt, bt_error=err, actuals=not CONSUMER)


@slate.get("/backtest/team/{team}", response_class=HTMLResponse)
def team_panel(request: Request, team: str):
    """The team on the backtest: the game as played over our grid (the overlay) and its rows, from the cached backtest."""
    dg = _dg(request)
    bt, err = backtest_of(dg)
    team = team.upper()
    if bt is None or team not in bt["players"]:
        return _render(request, "_nba_bt_team.html", bt=bt, team=team, rows=[], ov=None, bt_error=err or f"{team} is not on this slate")
    rows = [p for p in bt["table"] if p["team"] == team]            # the flat table's rows: the team's, with position, miss and value
    return _render(request, "_nba_bt_team.html", bt=bt, team=team, rows=rows, ov=(bt.get("rotations") or {}).get(team), bt_error=None, actuals=not CONSUMER)


@slate.get("/backtest/contest/{cid}", response_class=HTMLResponse)
def contest_panel(request: Request, cid: int):
    """One contest drilled into (ivan.sports.nba.backtest.slate.contest_detail): our entries, the field's ownership against ours, the
    payout curve. The laptop only: a consumer desk shows no money."""
    dg = _dg(request)
    if CONSUMER:
        return HTMLResponse('<p class="muted small">The contests show on the laptop only.</p>')
    from ivan.sports.nba.backtest import slate as SB
    bt, err = backtest_of(dg)
    players = {p["name"]: p for p in (bt or {}).get("table") or []}
    sess = S.session(dg) if S.holds(dg) else get_session(engine)
    try:
        with S.using(dg):
            cd, cerr = SB.contest_detail(sess, dg, int(cid), lock=(bt or {}).get("lock"), players=players), None
    except Exception as e:   # noqa: BLE001
        cd, cerr = None, f"{type(e).__name__}: {e}"
    finally:
        sess.close()
    return _render(request, "_nba_bt_contest.html", cd=cd, bt_error=cerr, bt=bt)


# ----------------------------------------------------------------------------- one game
def games_ctx(dg: int, bt: Optional[dict]) -> dict:
    """The slate's games for the backtest's first screen (ivan.sports.nba.backtest.game.overview): from the tip records, or from the
    slate's backtest `bt` when it has none."""
    from ivan.sports.nba.backtest import game as G
    sess = get_session(engine)
    try:
        if bt:                                                       # no tip records: the games from the scorecard's rows
            return dict(bt_games=G.overview(sess, dg, bt["players"], bt.get("day"), bt.get("rotations")), bt_games_error=None)
        return dict(bt_games=G.overview(sess, dg), bt_games_error=None)
    except Exception as e:   # noqa: BLE001 - the rest of the page still shows
        return dict(bt_games=[], bt_games_error=f"{type(e).__name__}: {e}")
    finally:
        sess.close()


def _game_view(request: Request, dg: int, key: str, note: Optional[str] = None):
    from ivan.sports.nba.backtest import game as G
    sess = get_session(engine)
    try:
        v, err = G.view(sess, dg, key), None
    except Exception as e:   # noqa: BLE001
        v, err = None, f"{type(e).__name__}: {e}"
    finally:
        sess.close()
    return _render(request, "_nba_bt_game.html", gv=v, bt_error=err, gnote=note, sims_min=G.N_MIN, sims_max=G.N_MAX)


@slate.get("/backtest/game/{game}", response_class=HTMLResponse)
def game_panel(request: Request, game: str):
    """One game of the backtest: both teams, each pool player's minutes, fantasy points and stats as published at the tip, as drawn
    by the game simulation from the tip record and as played; the heat grid of the draw over the minutes he was on the floor."""
    return _game_view(request, _dg(request), game)


def _edits_of(form, view: dict) -> dict:
    """The form's edits: a number in a player's min box, a starter box that differs from the tip record's call, the line boxes."""
    mins, starters = {}, {}
    published = {r["name"]: r["starter_pub"] for t in view["teams"] for r in t["rows"]}
    for k, v in form.multi_items():
        if k.startswith("min:") and str(v).strip() != "":
            mins[k[4:]] = v
    for name in form.getlist("row"):
        on = f"st:{name}" in form
        if name in published and on != published[name]:
            starters[name] = on
    return dict(minutes=mins, starters=starters, spread_home=form.get("spread_home"), total=form.get("total"))


@slate.post("/backtest/game/{game}/simulate", response_class=HTMLResponse)
async def game_simulate(request: Request, game: str):
    """Draw the game (ivan.sports.nba.backtest.game.simulate; terminal: scripts/nba/backtest_game.py --slate DG --game KEY --simulate /
    --set / --actual / --reset). The form carries the sims count, `do` and every box of both teams; the draw runs off the event loop
    (about 15 s the first time a day is opened in this process, a second or two after that)."""
    from ivan.sports.nba.backtest import game as G
    dg, key = _dg(request), game
    form = await request.form()
    do = str(form.get("do") or "simulate")

    def work() -> Optional[str]:
        sess = get_session(engine)
        try:
            if do == "reset":
                G.reset(dg, key)
                if G.load_draw(dg, key) is not None:
                    return "edits dropped: the game is the tip record's again"
                edits = {}
            elif do == "actual":
                edits = dict(G.actual_edits(sess, dg, key), spread_home=form.get("spread_home"), total=form.get("total"))
            else:
                edits = _edits_of(form, G.view(sess, dg, key))
            try:
                n = int(float(form.get("n") or 0)) or None
            except ValueError:
                n = None
            out = G.simulate(sess, dg, key, n=n, edits=edits)
            m = (out["what_if"] or out["base"])["meta"]
            return f"drawn in {m['n']} worlds" + (" with the edits" if out["what_if"] else " from the tip record as it was")
        except Exception as e:   # noqa: BLE001
            return f"not drawn: {type(e).__name__}: {e}"
        finally:
            sess.close()

    note = await run_in_threadpool(work)
    return await run_in_threadpool(_game_view, request, dg, key, note)


@slate.get("/backtest/game/{game}/player/{name}", response_class=HTMLResponse)
def game_player(request: Request, game: str, name: str):
    """One player of the game: his drawn fantasy points, minutes and each stat as distributions with what he did marked."""
    from ivan.sports.nba.backtest import game as G
    dg = _dg(request)
    sess = get_session(engine)
    try:
        pv, err = G.player_view(sess, dg, game, name), None
    except Exception as e:   # noqa: BLE001
        pv, err = None, f"{type(e).__name__}: {e}"
    finally:
        sess.close()
    return _render(request, "_nba_bt_player.html", pv=pv, bt_error=err)
