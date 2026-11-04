"""
The pages: the shell, the slate page (read only: games strip, needs-you, actions, news, contests, exposures), the players view, the team page, the board partial, a past slate's backtest.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from ivan.database.session import get_session
from ivan.web import frame
from ivan.web.sports import SlateInfo, find_slate
from . import core as A
from .core import LOADING, LOCK, _days_of, _render, _team_url, app, bound, desk, engine, slate, slate_url
from ivan.sports.nba import slates as S
from ivan.sports.nba.desk import LOBBY, CONSUMER
from .lateswap import _lateswap_ctx
from .pipeline import _contests_ctx, _pipeline_ctx, begin_switch
from ivan.sports.nba.desk import NEAR_TIP_MIN


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
                        moved="; ".join(x["text"] for x in ((getattr(desk, "changes", None) or {}).get("teams") or {}).get(t, [])),
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
        fav = desk.slate._lines.favourite(t) if getattr(desk.slate, "_lines", None) else None
        games.append(dict(teams=pair, tip=c["tip"], mins_to_tip=c["mins_to_tip"], locked=all(x["locked"] for x in pair), home=home, spread=spread, total=total, fav=fav))
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
        fav = desk.slate._lines.favourite(c["team"]) if getattr(desk.slate, "_lines", None) else None
        games.append(dict(teams=pair, tip=c["tip"], mins_to_tip=c["mins_to_tip"], locked=all(x["locked"] for x in pair),
                          score=max(x["score"] for x in pair), home=home, spread=spread, total=total, fav=fav))
    return sorted(games, key=lambda g: (g["locked"], g["tip"] or now))


def _blank_header(now: datetime) -> dict:
    """The header of a page outside a held slate (a slate being fetched, a past slate's backtest): the shared index, no pool."""
    from ivan.sports.nba import autopilot
    from ivan.core import feed
    return dict(draft_group=None, lock=None, mins_to_lock=None, n_lineups=0, pool_at=None, n_players=0, n_priced=0, starters_at=None, odds_at=None,
                stale=False, fresh=[], slate_days=LOBBY.slate_days(now), meta={}, refreshed_at=None, refreshed_by=None, pulled_at=None, published_at=None,
                mapped={}, questions={}, news={}, team_states=[], overrides=0, overrides_path="",
                autopilot=None if CONSUMER else autopilot.state(), feed=feed.status("nba") if CONSUMER else None)


def _header(now: datetime) -> dict:
    d = bound()
    return d.header(now) if d is not None else _blank_header(now)


def _shell(request: Request, view: Optional[str], team: Optional[str], page: Optional[SlateInfo], now: datetime, header: dict,
           note: Optional[str] = None):
    """
    The page shell for a slate: held (its desk is bound: the board, or the team's card when `team` is set) or not (its backtest
    when its day has passed, else the loading state while it is fetched, else the load card). A held slate being refreshed
    renders as usual with a banner (its edits wait); only a slate with no desk yet shows the spinner alone.
    `note` is why a load did not start (the load card says so).
    """
    from ivan.sports.nba.slate_backtest import mode_for
    d = bound()
    days = _days_of(header["slate_days"], now)
    if page is None and d is not None:
        page = next((s for dd in days for s in dd.slates if s.key == str(d.dg)), None)
    dg = int(page.key) if page is not None and str(page.key).isdigit() else None
    loading = LOADING.get(dg) if dg is not None else None
    if d is None and loading:
        # the slate is being fetched for the first time: the bar keeps the slate asked for and the board shows the progress; its
        # poll reloads the page when the load is done, and the page is then the slate's desk
        fr = frame.ctx(request, A.SPORT, days, None, page, None, now)
        return _render(request, "nba_desk.html", header=header, team=None, game=None, view="games", busy=loading, dg=dg, frame=fr,
                       page_slate=fr["slate"], is_loaded=False)
    game = None
    if d is not None:                                                                  # a held slate: its desk
        if view is None:
            lock = header.get("lock")
            view = {"lateswap": "lateswap", "backtest": "backtest"}.get(mode_for(lock.date() if lock else None, lock, now), "games")
        view = VIEW_ALIAS.get(view, view)
        if team is not None:
            if team not in desk.slate.teams():
                raise HTTPException(404, f"{team} is not on this slate")
            for t in desk.slate.teams():
                desk.slate.spread(t)
            game = next((g for grp in games_menu(now) for g in grp["games"] if any(c["team"] == team for c in g["teams"])), None)
            view = "games"
        is_loaded = True
    else:                                                                              # not held: its backtest, or load it first
        team = None
        if view not in ("backtest", "load"):
            view = "backtest" if (page is None or page.source == "archive" or page.day < now.date()) else "load"
        is_loaded = False
    fr = frame.ctx(request, A.SPORT, days, None, page, view if view != "load" else None, now)
    return _render(request, "nba_desk.html", header=header, team=team, game=game, view=view, busy=None, dg=dg, frame=fr, page_slate=page,
                   is_loaded=is_loaded, load_note=note)


@app.get("/desk/", response_class=HTMLResponse)
def index(request: Request, view: Optional[str] = None, dg: Optional[int] = None, open: Optional[str] = None):
    """The flat address old links use: today's main held slate's page (the picker when none is held). view= and dg= go to that
    slate's page with the view; open=TEAM to its team page."""
    now = datetime.now()
    target = dg if dg is not None else S.main_slate_today(now)
    if target is None:
        return RedirectResponse(auth_root(request) + "/", status_code=303)
    url = slate_url(request, int(target))
    if open:
        return RedirectResponse(url + f"{open.upper()}/", status_code=303)
    return RedirectResponse(url + (f"?view={view}" if view else ""), status_code=303)


def auth_root(request: Request) -> str:
    from ivan.web import auth
    return auth.root_of(request)


@app.get("/{day}/{key}/", response_class=HTMLResponse)
def slate_page(request: Request, day: str, key: str, view: Optional[str] = None):
    """One slate: sport › day › slate › view. The page shows the slate in its address, so the desk follows the bar: a held slate
    is its desk (pre-lock, post-lock once it locked, backtest once its day passed); a slate of an earlier day not held is its
    backtest (read only; Restore loads it); a lobby slate of today or later that this machine does not hold starts loading when
    the page is opened (begin_switch: the producer fetches it into its own directory, the consumer asks the laptop for it), the
    page shows the progress and reloads on it when it is in. When that cannot start the page says why and offers a button."""
    d = frame.parse_day(day)
    now = datetime.now()
    desk_ = bound()
    if desk_ is not None:
        with LOCK:
            if not desk.busy:
                desk.maybe_reload()
            header = _header(now)
            page = find_slate(_days_of(header["slate_days"], now), key)
            return _shell(request, view, None, page, now, header)
    header = _header(now)
    page = find_slate(_days_of(header["slate_days"], now), key)
    if page is None:
        page = SlateInfo(key=str(key), day=d, lock=None, label=f"slate {key}", source="archive", detail="nothing on file for this slate")
    why = None
    if page.load is not None and not page.loaded and page.day >= now.date() and view in (None, "games", "load") and not LOADING.get(int(page.key)):
        why = begin_switch(int(page.key), "lobby")
    return _shell(request, view, None, page, now, header, note=why)


@app.get("/{day}/{key}/{team}/", response_class=HTMLResponse)
def team_page(request: Request, day: str, key: str, team: str):
    """One team of a held slate: sport › day › slate › TEAM. The rotation card (the grid, the rows, pins, scenarios, the played
    game) on a page of its own; every edit here writes the team's projection into the slate's pool at once, so leaving the page
    loses nothing. The slate page shows the result and links here from the games strip and the news table."""
    frame.parse_day(day)
    if bound() is None:
        raise HTTPException(404, "a team page belongs to a held slate; open the slate first")
    with LOCK:
        if not desk.busy:
            desk.maybe_reload()
        now = datetime.now()
        header = _header(now)
        page = find_slate(_days_of(header["slate_days"], now), key)
        return _shell(request, "games", team.upper(), page, now, header)


def _backtest_board(request: Request, dg: Optional[int]):
    """A slate's backtest, read only (ivan.sports.nba.slate_backtest = scripts/nba/backtest_slate.py): nothing is loaded or deleted.
    A held slate is read through its own database (its contests, lineups and sims), another through the history (its captures)."""
    from ivan.sports.nba import slate_backtest as SB
    if dg is None:
        return _render(request, "_nba_board.html", busy=None, view="backtest", bt=None, bt_error="no slate picked", menu=[], games=[], players=[],
                       lateswap=None, exposures=None, card=None, game=None, open_team=None)
    sess = S.session(dg) if S.holds(dg) else get_session(engine)      # its own session: the page never waits on the desk's lock
    try:
        with S.using(int(dg)):
            bt, err = SB.slate_backtest(sess, int(dg)), None
    except Exception as e:   # noqa: BLE001 - show what failed instead of a blank page
        bt, err = None, f"{type(e).__name__}: {e}"
    finally:
        sess.close()
    return _render(request, "_nba_board.html", busy=None, view="backtest", bt=bt, bt_error=err, menu=[], games=[], players=[],
                   lateswap=None, exposures=None, card=None, game=None, open_team=None)


VIEW_ALIAS = {"exposures": "games"}                           # the exposures are a section of pre-lock


@slate.get("/board", response_class=HTMLResponse)
def board(request: Request, view: str = "games", after: str = "", dg: Optional[int] = None):
    """The slate's board (the heavy part): the games menu, the pipeline panel and the cards / players / late swap view.
    after=busy is the loading poll's request: once the slate's first load is done the whole page reloads (HX-Refresh), because
    the header (the slate picker, the counts) was built before it. A held slate being refreshed renders its board as usual."""
    key = getattr(request.state, "dg", None)
    from fastapi.responses import Response
    if after == "busy" and bound() is not None:
        if desk.busy:
            return _render(request, "_nba_progress.html", view=view)      # still refreshing: the banner with the latest lines
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    if after == "busy" and bound() is None and key is not None and not LOADING.get(key) and S.holds(key):
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    if view == "backtest":
        return _backtest_board(request, dg if dg is not None else key)
    view = VIEW_ALIAS.get(view, view)
    if bound() is None:
        return _render(request, "_nba_board.html", busy=LOADING.get(key) or "this slate is not held here", view=view, games=[], menu=[], players=[], lateswap=None, exposures=None)
    with LOCK:
        now = datetime.now()
        for t in desk.slate.teams():
            desk.slate.spread(t)                            # loads the lines once
        menu = games_menu(now)                               # the games strip: tip, line, one chip per team (its state, the questions on hover)
        teams = desk.slate.teams()
        news_table = desk.news_table(now) if view == "games" else None
        players = []; lateswap = None; exposures = None
        if view == "lateswap":
            lateswap = _lateswap_ctx(now)
        if view == "players":                                # every team's rows, read only (the team page is where a row is edited)
            players = [r for t in teams for r in desk.team_rows(t)]
            players.sort(key=lambda r: (-(r["own"] or 0), -(r["fpts"] or 0)))
        return _render(request, "_nba_board.html", busy=None, menu=menu, view=view, players=players, lateswap=lateswap, exposures=exposures,
                       news_table=news_table, contests_ctx=_contests_ctx(), mapped=desk.mapped_summary(), needs=_needs_you(menu, now),
                       questions_sum=desk.questions_summary(now), team_url=lambda t: _team_url(request, t), **_pipeline_ctx())


NEAR_TIP_LOOK_MIN = 90      # a team with an open question or no lineup this close to its tip needs a look


def _needs_you(menu: list, now: datetime) -> list[dict]:
    """
    The teams that want a look, and why: mapped but the inputs moved since (a status, the lineup, an out), an open question
    (a questionable rotation player with no call) inside NEAR_TIP_LOOK_MIN of the tip, no Rotowire lineup inside NEAR_TIP_MIN,
    a headline newer than the mapping. Everything else is green: the slate page is for reading, the team page for the rest.
    Order: the nearest tip first. Empty = nothing to do until the next refresh.
    """
    out = []
    for grp in menu:
        for g in grp["games"]:
            for c in g["teams"]:
                if c.get("locked"):
                    continue
                why = []
                mtt = c.get("mins_to_tip")
                if c.get("colour") == "changed":
                    why.append(c.get("why") or "changed since mapped")
                if c.get("open") and (mtt is None or mtt <= NEAR_TIP_LOOK_MIN):
                    why.append(f"{c['open']} open question{'s' if c['open'] != 1 else ''}: {c.get('qtext') or ''}".rstrip(": "))
                if c.get("new"):
                    why.append("new headline on a questionable player")
                if c.get("starters") == "none" and mtt is not None and mtt <= NEAR_TIP_MIN:
                    why.append("no lineup from Rotowire this close to the tip")
                if c.get("state") == "unmapped" and mtt is not None and mtt <= NEAR_TIP_LOOK_MIN:
                    why.append("unmapped")
                if why:
                    out.append(dict(team=c["team"], tip=c.get("tip"), mins_to_tip=mtt, why=why, colour=c.get("colour")))
    out.sort(key=lambda x: (x["mins_to_tip"] if x["mins_to_tip"] is not None else 1e9, x["team"]))
    return out
