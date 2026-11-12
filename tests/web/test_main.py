"""The one Ivan UI (ivan.web.main): every sport mounted under one app, one password for all of it, the frame's pages up."""
import json

import pytest

pytest.importorskip("ivan.web.main")
from fastapi.testclient import TestClient

from ivan.web import auth
from ivan.web.main import app
from ivan.web.sports import REGISTRY

CF = {"cf-ray": "abc-ORD", "cf-connecting-ip": "203.0.113.7"}


@pytest.fixture
def open_site(monkeypatch):
    for k in list(auth.LEGACY) + list(auth.LEGACY.values()):
        monkeypatch.delenv(k, raising=False)
    return TestClient(app, client=("127.0.0.1", 50000))


@pytest.fixture
def locked_site(monkeypatch):
    for k in list(auth.LEGACY) + list(auth.LEGACY.values()):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(auth.ENV, "correct horse battery")
    return TestClient(app, client=("127.0.0.1", 50000))


def test_every_sport_is_listed_and_mounted(open_site):
    assert set(REGISTRY) == {"nba", "nfl", "mma"}
    r = open_site.get("/")
    assert r.status_code == 200
    for label in ("NBA", "NFL", "MMA"):
        assert label in r.text
    for sport in REGISTRY.values():
        assert sport.app is not None, f"{sport.key}: {sport.error}"


def test_pickers_and_slate_pages_render(open_site):
    for key in ("nba", "nfl", "mma"):
        r = open_site.get(f"/{key}/")
        assert r.status_code == 200, (key, r.text[:300])
        assert 'class="frame"' in r.text and "pick a date" in r.text
        assert open_site.get(f"/{key}/2026-10-30/").status_code == 200
        assert open_site.get(f"/{key}/not-a-day/").status_code == 404
    # a slate nobody has on file: the NFL pipeline (a day ahead), the MMA slate card, the NBA backtest view
    r = open_site.get("/nfl/2099-01-01/999999/")
    assert r.status_code == 200 and "not the working slate" in r.text
    assert open_site.get("/nfl/2020-01-01/999999/").status_code == 200           # a past slate: its builds page (none)
    assert open_site.get("/mma/2099-01-01/999999/").status_code == 200
    assert open_site.get("/nba/2020-01-01/999999/").status_code == 200
    assert open_site.get("/nfl/pipeline").status_code == 200


def test_one_password_guards_every_path(locked_site):
    for path in ("/", "/nfl/", "/nfl/jobs", "/nba/", "/nba/desk/", "/mma/"):
        r = locked_site.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login", path
    r = locked_site.get("/nfl/pipeline", headers={"HX-Request": "true"}, follow_redirects=False)
    assert r.status_code == 401 and r.headers["HX-Redirect"] == "/login"
    assert locked_site.get("/login").status_code == 200
    assert locked_site.get("/static/ivan.css").status_code == 200              # the stylesheet the login page needs is not gated
    r = locked_site.post("/login", data={"pw": "wrong"}, follow_redirects=False)
    assert r.status_code == 401
    r = locked_site.post("/login", data={"pw": "correct horse battery"}, follow_redirects=False)
    assert r.status_code == 303 and auth.COOKIE in r.cookies
    assert locked_site.get("/nfl/jobs").status_code == 200
    assert locked_site.get("/nba/").status_code == 200


def test_tunnel_requests_need_a_password(open_site):
    """Through the Cloudflare tunnel every path is forwarded; with no password configured the site refuses to serve."""
    for path in ("/", "/nba/", "/nfl/jobs"):
        assert open_site.get(path, headers=CF).status_code == 503, path


def test_a_day_opens_on_its_main_slate(open_site, monkeypatch):
    """The bar's day dropdown is a slate pick: /<sport>/<day>/ goes to the day's loaded slate, else its biggest, else its only one;
    ?pick (and a day whose main slate cannot be told) shows the day's slates."""
    from datetime import date, datetime
    from ivan.web.sports import Day, SlateInfo, main_slate
    d = date(2099, 1, 2)
    small = SlateInfo(key="1", day=d, lock=datetime(2099, 1, 2, 18), label="18:00 · 2 games", source="lobby", games=2)
    big = SlateInfo(key="2", day=d, lock=datetime(2099, 1, 2, 19), label="19:00 · 8 games", source="lobby", games=8)
    assert main_slate(Day(day=d, label="", tag="", slates=[small, big])) is big
    assert main_slate(Day(day=d, label="", tag="", slates=[small])) is small
    assert main_slate(Day(day=d, label="", tag="", slates=[])) is None
    loaded = SlateInfo(key="3", day=d, lock=None, label="loaded", source="working", loaded=True)
    assert main_slate(Day(day=d, label="", tag="", slates=[small, big, loaded])) is loaded
    assert main_slate(Day(day=d, label="", tag="", slates=[SlateInfo(key="4", day=d, lock=None, label="a", source="lobby"), SlateInfo(key="5", day=d, lock=None, label="b", source="lobby")])) is None
    monkeypatch.setattr(REGISTRY["nba"], "days", lambda now: [Day(day=d, label="Fri 01-02", tag="", slates=[small, big])])
    r = open_site.get("/nba/2099-01-02/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("/nba/2099-01-02/2/")
    r = open_site.get("/nba/2099-01-02/?pick")
    assert r.status_code == 200 and "pick a date" in r.text and "19:00 · 8 games" in r.text
    assert open_site.get("/nba/2099-01-03/").status_code == 200                     # a day with nothing: the picker says so


def test_nba_slate_page_loads_the_slate_it_names(open_site, monkeypatch, tmp_path):
    """Opening a lobby slate the desk does not hold starts its load into its own directory (the producer fetches it; a consumer
    asks the laptop) and the page shows the progress; when the slate's jobs run the page says why and offers the load as a button."""
    from datetime import date, datetime
    from ivan.web import nba as D
    from ivan.web.nba import pages as PG, pipeline as P
    from ivan.web.sports import Day, SlateInfo
    d = date(2099, 1, 2)
    lobby = SlateInfo(key="999002", day=d, lock=datetime(2099, 1, 2, 19), label="19:00 · 8 games", source="lobby", games=8,
                      load={"url": "/slate", "fields": {"draft_group": "999002", "source": "lobby", "day": d.isoformat()}, "label": "load into the desk"})
    monkeypatch.setattr(PG, "_days_of", lambda rows, now: [Day(day=d, label="Fri 01-02", tag="", slates=[lobby])])
    started = []

    def fake_refresh(what, dg, update_odds, lines=None):
        started.append((what, dg)); D.LOADING[int(dg)] = what
    monkeypatch.setattr(P, "_start_refresh", fake_refresh)
    monkeypatch.setattr(P, "_active", lambda dg=None: [])
    try:
        r = open_site.get("/nba/2099-01-02/999002/")
        assert r.status_code == 200 and started == [("loading slate 999002", 999002)]
        assert "loading slate 999002" in r.text and 'name="draft_group"' not in r.text
        # while it loads, opening the page again does not start another; the board poll keeps polling
        r = open_site.get("/nba/2099-01-02/999002/")
        assert len(started) == 1
        assert open_site.get("/nba/2099-01-02/999002/board?view=games&after=busy").status_code == 200
    finally:
        D.LOADING.pop(999002, None)
    # that slate's jobs run: the page says so and offers the button
    monkeypatch.setattr(P, "_active", lambda dg=None: [object()])
    r = open_site.get("/nba/2099-01-02/999002/")
    assert r.status_code == 200 and len(started) == 1
    assert "a job is running on this slate" in r.text and 'name="draft_group" value="999002"' in r.text


def test_nba_bar_keeps_the_slate_being_loaded(open_site, monkeypatch):
    """While a slate's first load runs the bar must show the slate asked for and the board the progress; no other slate is touched."""
    from ivan.web import nba as D
    D.LOADING[999001] = "loading slate 999001"
    try:
        r = open_site.get("/nba/2099-01-02/999001/")
        assert r.status_code == 200
        assert "loading slate 999001" in r.text
        assert 'value="/nba/2099-01-02/999001/"' not in r.text or 'value="/nba/2099-01-02/999001/" data-day="2099-01-02" selected' in r.text
    finally:
        D.LOADING.pop(999001, None)


def _held(open_site):
    """(draft group, day) of the main slate this database holds (today's main, else the newest held), or a skip."""
    from ivan.sports.nba import slates as S
    held = S.held()
    if not held:
        pytest.skip("no slate held in this database")
    dg = S.main_slate_today() or held[0]; lock = S.meta_time(S.read_meta(dg), "lock")
    return dg, (lock or __import__("datetime").datetime.now()).date().isoformat()


def test_nba_step_knobs_move_only_their_own(open_site, monkeypatch):
    """Generate and sim send their popover's knobs only; the others, and the fresh-pool box, keep what the desk last had."""
    from ivan.web import nba as D
    from ivan.web.nba import pipeline as P
    dg, day = _held(open_site)
    monkeypatch.setattr(P, "_active", lambda dg=None: [object()])          # "a job is running": the route stores the knobs, starts nothing
    monkeypatch.setitem(D.pipeline_form, "lineups", "12345")
    monkeypatch.setitem(D.pipeline_form, "fresh", "on")
    monkeypatch.setitem(D.pipeline_form, "sims", "1000")
    assert open_site.post(f"/nba/{day}/{dg}/pipeline/sim", data={"sims": "777", "field_cap": "8000"}).status_code == 200
    assert D.pipeline_form["sims"] == "777" and D.pipeline_form["lineups"] == "12345" and D.pipeline_form["fresh"] == "on"
    assert open_site.post(f"/nba/{day}/{dg}/pipeline/generate", data={"cfg_fresh": "1", "lineups": "2000"}).status_code == 200
    assert D.pipeline_form["lineups"] == "2000" and D.pipeline_form["fresh"] == "" and D.pipeline_form["sims"] == "777"


def test_nba_slate_page_reads_and_the_team_page_edits(open_site):
    """The slate page carries no projection edit control; every team chip and news name is a link to the team's own page, which
    holds the card. An old /desk/?open=TEAM link lands on the team page. Two held slates' pages serve side by side."""
    from ivan.web import nba as D
    from ivan.sports.nba import slates as S
    dg, day = _held(open_site)
    r = open_site.get(f"/nba/{day}/{dg}/")
    assert r.status_code == 200 and f'/nba/{day}/{dg}/board' in r.text
    d = D.desk_for(dg)
    assert d is not None and d.dg == dg
    teams = d.slate.teams()
    if not teams:
        pytest.skip("the held slate has no teams")
    team = teams[0]
    board = open_site.get(f"/nba/{day}/{dg}/board?view=games")
    assert board.status_code == 200
    assert 'name="minutes"' not in board.text and f'hx-post="/nba/{day}/{dg}/team/' not in board.text      # read only: no pin inputs, no edit posts
    assert f"/nba/{day}/{dg}/{team}/" in board.text                                                        # the chip links to the team page
    page = open_site.get(f"/nba/{day}/{dg}/{team}/")
    assert page.status_code == 200 and f'hx-get="/nba/{day}/{dg}/team/{team}"' in page.text
    others = [t for t in d.slate.teams() if t != team]
    assert 'class="tips"' in page.text and f" on\" href=\"/nba/{day}/{dg}/{team}/\"" in page.text                  # the games strip, this team's chip outlined
    assert any(f'href="/nba/{day}/{dg}/{t}/"' in page.text for t in others)                                   # the other teams one click away
    card = open_site.get(f"/nba/{day}/{dg}/team/{team}")
    assert card.status_code == 200 and 'name="minutes"' in card.text and f'/nba/{day}/{dg}/team/{team}/' in card.text   # the card is where the pins are
    r = open_site.get(f"/nba/desk/?open={team}", follow_redirects=False)
    assert r.status_code == 303 and "/nba/" in r.headers["location"]
    if S.main_slate_today():                                                                                      # the main held slate's team page; the picker on a day none is held
        assert r.headers["location"].endswith(f"/{team}/")
    # a second held slate (when there is one) serves at the same time, its own desk bound
    others = [x for x in S.held() if x != dg]
    if others:
        o = others[0]; oday = (S.meta_time(S.read_meta(o), "lock") or __import__("datetime").datetime.now()).date().isoformat()
        assert open_site.get(f"/nba/{oday}/{o}/").status_code == 200
        assert D.desk_for(o).dg == o and D.desk_for(dg).dg == dg


def test_nfl_desk_renders(open_site):
    """The NFL desk: the slate page's default view, its game card and pair partials, the old pipeline address redirecting to it."""
    r = open_site.get("/nfl/desk")
    assert r.status_code == 200 and "Actions" in r.text and 'id="game"' in r.text and 'id="players"' in r.text
    r = open_site.get("/nfl/pipeline", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("/nfl/desk")
    assert open_site.get("/nfl/desk/job").status_code == 200
    assert open_site.get("/nfl/desk/game?game=AAA-BBB").status_code == 200          # no such game: the card says so
    assert open_site.get("/nfl/desk/pair?team=GB&a=QB&b=WR1").status_code == 200
    assert open_site.get("/nfl/desk/pair?team=GB&a=QB&b=DST&opp=CAR").status_code == 200


def test_nfl_contest_weights_save_own_knobs(open_site, monkeypatch, tmp_path):
    """A contest's knobs save as typed: a value is its own, a blank drops back to the default, a bad value is refused."""
    from ivan.web.nfl import pipeline as P
    monkeypatch.setattr(P, "WEIGHTS_FILE", tmp_path / "weights.json")
    monkeypatch.setattr(P, "WEB_INPUT_DIR", tmp_path)
    r = open_site.post("/nfl/contest/123/weights", data={"method": "portfolio", "bankroll": "4", "cash_pct": ""})
    assert "own knobs" in r.text
    assert json.loads((tmp_path / "weights.json").read_text())["123"] == {"method": "portfolio", "bankroll": 4.0}
    assert "not a chooser method" in open_site.post("/nfl/contest/123/weights", data={"method": "yolo"}).text
    r = open_site.post("/nfl/contest/123/weights", data={"method": "", "bankroll": ""})
    assert "defaults" in r.text and "123" not in json.loads((tmp_path / "weights.json").read_text())


def test_spearman_se():
    from ivan.sports.nfl.correlation_check import spearman_se
    assert spearman_se(0.0, 3) is None
    assert abs(spearman_se(0.0, 103) - 0.1) < 1e-9
    assert spearman_se(0.5, 103) > spearman_se(0.0, 103)


def _nba_addresses(html: str) -> tuple[set, set]:
    """(GET addresses, POST addresses) under /nba/ that a rendered page carries: href and hx-get, form action and hx-post."""
    import re
    gets = set(re.findall(r'(?:href|hx-get)="(/nba/[^"#]*)"', html))
    posts = set(re.findall(r'(?:hx-post)="(/nba/[^"#]*)"', html))
    for m in re.finditer(r'<form[^>]*method="post"[^>]*action="(/nba/[^"#]*)"', html, flags=re.I):
        posts.add(m.group(1))
    for m in re.finditer(r'<form[^>]*action="(/nba/[^"#]*)"[^>]*method="post"', html, flags=re.I):
        posts.add(m.group(1))
    return gets, posts


def _route_exists(path: str, method: str) -> bool:
    """Does any route of the site (its mounted sports included) answer `method` at `path`? Nothing is sent."""
    from starlette.routing import Match
    from urllib.parse import urlsplit

    def walk(routes, scope):
        for r in routes:
            m, child = r.matches(scope)
            if m == Match.NONE:
                continue
            sub = getattr(r, "app", None)
            if hasattr(sub, "routes") and child.get("app") is not None and sub is not None and not hasattr(r, "endpoint"):
                if walk(sub.routes, {**scope, **child}):
                    return True
            elif m == Match.FULL:
                return True
        return False
    scope = {"type": "http", "method": method.upper(), "path": urlsplit(path).path, "root_path": "", "headers": [], "query_string": b""}
    return walk(app.routes, scope)


def test_every_address_the_nba_pages_carry_answers(open_site):
    """Every link and every form or htmx target on the NBA pages of a held slate resolves: the GET ones answer (no 404, no doubled
    slate in the path), the POST ones have a route. The team page's back link doubled the slate once (2026-11-03: `root` became
    the slate's root while the link still appended the slate's path to it)."""
    from ivan.web import nba as D
    dg, day = _held(open_site)
    d = D.desk_for(dg)
    team = d.slate.teams()[0] if d.slate.teams() else None
    pages = [f"/nba/{day}/{dg}/", f"/nba/{day}/{dg}/?view=players", f"/nba/{day}/{dg}/?view=lateswap", f"/nba/{day}/{dg}/?view=backtest",
             f"/nba/{day}/{dg}/board?view=games", f"/nba/{day}/{dg}/board?view=players", f"/nba/{day}/{dg}/board?view=lateswap",
             f"/nba/{day}/{dg}/board?view=backtest&dg={dg}", f"/nba/{day}/{dg}/pipeline", f"/nba/{day}/{dg}/pipeline/status", f"/nba/{day}/{dg}/exposures"]
    if team:
        pages += [f"/nba/{day}/{dg}/{team}/", f"/nba/{day}/{dg}/team/{team}", f"/nba/{day}/{dg}/team/{team}/rotation"]
    gets, posts = set(), set()
    for p in pages:
        r = open_site.get(p)
        assert r.status_code == 200, (p, r.status_code)
        g, po = _nba_addresses(r.text); gets |= g; posts |= po
    assert gets and posts
    bad = []
    for g in sorted(gets):
        assert g.count(f"/{dg}/") <= 1, f"the slate appears twice in {g}"
        r = open_site.get(g, follow_redirects=True)
        if r.status_code >= 400:
            bad.append((g, r.status_code))
    assert not bad, bad
    missing = [p for p in sorted(posts) if not _route_exists(p, "POST")]
    assert not missing, missing
    if team:
        page = open_site.get(f"/nba/{day}/{dg}/{team}/").text
        assert f'href="/nba/{day}/{dg}/"' in page                              # ‹ slate goes back to the slate page


def test_nba_refresh_shows_a_progress_banner_with_its_log(open_site):
    """While a held slate refreshes in the background its board carries a banner with what runs, the seconds so far and the last
    lines the refresh logged, polled every 2 s; the poll answers the banner while it runs and a reload when it is done."""
    from datetime import datetime
    from ivan.web import nba as D
    dg, day = _held(open_site)
    d = D.desk_for(dg)
    D.PROGRESS[dg] = dict(what="refreshing the pool", started=datetime.now(), lines=["fetching DK draftables for draft group %d" % dg, "stored 382 players"])
    d.busy = "refreshing the pool"
    try:
        r = open_site.get(f"/nba/{day}/{dg}/board?view=games")
        assert r.status_code == 200 and 'id="refresh-banner"' in r.text and "stored 382 players" in r.text and "refreshing the pool" in r.text
        assert "team-row" in r.text or "games" in r.text                                    # the board itself still renders beneath the banner
        r = open_site.get(f"/nba/{day}/{dg}/board?view=games&after=busy")
        assert r.status_code == 200 and 'id="refresh-banner"' in r.text and "HX-Refresh" not in r.headers
        page = open_site.get(f"/nba/{day}/{dg}/")
        assert "refreshing the pool…" in page.text                                           # the bar says so too
        edit = open_site.post(f"/nba/{day}/{dg}/refresh", data={"what": "dk"}, follow_redirects=False)
        assert edit.status_code == 303                                                       # a second refresh is refused while one runs
    finally:
        d.busy = None; D.PROGRESS.pop(dg, None)
    r = open_site.get(f"/nba/{day}/{dg}/board?view=games&after=busy")
    assert r.headers.get("HX-Refresh") == "true"


def test_nba_run_estimate_warns_but_never_refuses(open_site, monkeypatch):
    """The knobs popover's estimate answers for the step's knobs; a run the estimate puts inside the margin before lock starts
    anyway with a "heads up" flash (no run-anyway box, nothing refused); the job's record carries the estimate."""
    from ivan.web import nba as D
    from ivan.web.nba import pipeline as P
    from ivan.sports.nba.slate import job_time as JT
    dg, day = _held(open_site)
    slow = {s: dict(per_unit=0.0, fixed=360000.0, n=9, p90_ratio=1.2, basis="fitted") for s in JT.STEPS}     # every run "takes a hundred hours"
    monkeypatch.setattr(JT, "fit", lambda *a, **k: slow)
    monkeypatch.setattr(P, "_active", lambda dg=None: [])
    r = open_site.get(f"/nba/{day}/{dg}/pipeline/estimate?step=sim&sims=1000&field_cap=10000")
    assert r.status_code == 200 and "≈ 100 h" in r.text and 'class="est' in r.text
    d = D.desk_for(dg)
    has_lock = any(p.gametime for p in d.slate.players)
    submitted = []
    monkeypatch.setattr(P, "_submit_job", lambda name, cmd, step, extra=None: submitted.append((name, extra)) or object())
    monkeypatch.setattr(P, "capture_pool", lambda *a, **k: type("I", (), {"players": 1})())
    r = open_site.post(f"/nba/{day}/{dg}/pipeline/sim", data={"sims": "1000", "field_cap": "10000"})
    assert r.status_code == 200
    assert len(submitted) == 1 and submitted[0][0] == "sim" and submitted[0][1]["estimate"] == 360000
    assert "not started:" not in r.text and "run anyway" not in r.text                                 # the refusal flash is gone
    if has_lock and min(p.gametime for p in d.slate.players if p.gametime) > __import__("datetime").datetime.now():
        assert "heads up" in r.text                                                                     # inside the margin: warned, started


def test_nba_field_job_done_reloads_the_page_and_the_export_sees_the_ownership(open_site, monkeypatch):
    """There is no Snapshot button (Generate and Sim take the snapshot); the status poll that finds a job of the slate just done
    answers HX-Refresh (every job: the page shows what it wrote), and for a pool-writing one the desk's pool stamp sees an ownership
    written behind its back first, so the reloaded board and the projections CSV export carry the new numbers. A failed job leaves the
    page as it is, its log in the partial."""
    import time
    from ivan.web import nba as D
    from ivan.web.nba import pipeline as P
    from ivan.sports.nba.models import NBAPlayer
    dg, day = _held(open_site)
    r = open_site.get(f"/nba/{day}/{dg}/pipeline")
    assert r.status_code == 200 and "/publish" not in r.text and "Snapshot" not in r.text and "3–5 in a row" in r.text
    d = D.desk_for(dg)
    pid = d.slate.players[0].id
    before = d.session.query(NBAPlayer.ownership).filter(NBAPlayer.id == pid).scalar()        # the live pool: put back at the end
    with d.lock:
        d.session.query(NBAPlayer).filter(NBAPlayer.id == pid).update({"ownership": 0.4321 if before != 0.4321 else 0.4322}); d.session.commit()
    assert d._stamp() != d.pool_stamp                                                        # the stamp sees the ownership alone
    job = type("J", (), dict(name="field", status="done", finished=time.time() - 1, started=time.time() - 30, id="f1", meta={"slate": dg},
                             is_active=False, duration=29.0, cmd=[]))()
    monkeypatch.setattr(P, "slate_jobs", lambda dg=None: [job])
    monkeypatch.setattr(P, "_active", lambda dg=None: [])
    monkeypatch.setattr(P.runner, "read_log", lambda *a, **k: "")
    r = open_site.get(f"/nba/{day}/{dg}/pipeline/status")
    assert r.status_code == 200 and r.headers.get("HX-Refresh") == "true"
    assert d._stamp() == d.pool_stamp                                                        # the poll reloaded the desk
    r = open_site.get(f"/nba/{day}/{dg}/export/projections.csv")
    assert r.status_code == 200 and "0.432" in r.text
    job.name = "choose"
    r = open_site.get(f"/nba/{day}/{dg}/pipeline/status")
    assert r.status_code == 200 and r.headers.get("HX-Refresh") == "true"                     # a job that does not write the pool reloads too
    job.status = "failed"
    r = open_site.get(f"/nba/{day}/{dg}/pipeline/status")
    assert r.status_code == 200 and "HX-Refresh" not in r.headers                             # a failed job: the partial, no reload
    from ivan.sports.nba import slates as S
    with d.lock, S.using(dg):
        d.session.query(NBAPlayer).filter(NBAPlayer.id == pid).update({"ownership": before}); d.session.commit(); d.rebuild()


def test_nba_starter_toggle_posts_the_rows_player(open_site):
    """The st cell on the rotation grid toggles a starter call: the click sends the row's player (the grid rows carry no form field
    for him, so hx-include alone posted nothing and the route answered 422); a second click clears the call."""
    from ivan.web import nba as D
    from ivan.sports.nba.projection.live import override_of
    dg, day = _held(open_site)
    d = D.desk_for(dg)
    team = sorted({p.team for p in d.slate.players})[0]
    r = open_site.get(f"/nba/{day}/{dg}/team/{team}")
    assert r.status_code == 200 and "hx-vals='{&#34;player&#34;: " in r.text.replace('"', "&#34;") or "hx-vals='{\"player\": " in r.text
    assert f"/team/{team}/starter" in r.text
    assert open_site.post(f"/nba/{day}/{dg}/team/{team}/starter", data={}).status_code == 422                # what the old click sent
    key = next(k for k in (r.text.split('data-key="')[1:2] or ["x"]) ).split('"')[0]
    before = override_of(d.overrides, key).get("starter")
    from ivan.sports.nba.slate import changes as CH
    from ivan.sports.nba.names import normalize_name as _nn
    stamp_before = CH.status_cutoffs(d.session).get(key)
    rotowire_starter = key in (d.slate.starters.get(team) or set())
    try:
        r = open_site.post(f"/nba/{day}/{dg}/team/{team}/starter", data={"player": key})
        assert r.status_code == 200 and override_of(d.overrides, key).get("starter") is not None               # a hand call now
        stamp = CH.status_cutoffs(d.session).get(key)
        if rotowire_starter:
            assert stamp is not None and stamp != stamp_before                                                  # benched by hand: his props posted before are set aside
        else:
            assert stamp is None                                                                                 # made a starter: an upgrade sets nothing aside
        r = open_site.post(f"/nba/{day}/{dg}/team/{team}/starter", data={"player": key})
        assert r.status_code == 200 and override_of(d.overrides, key).get("starter") == before                  # cleared back
    finally:
        for p in d.slate.players:                                                                              # the live slate: leave his stamp as it was
            if _nn(p.name) == key:
                p.status_changed_at = stamp_before
        d.session.commit()


def test_nba_grid_stat_columns_pin_a_total(open_site):
    """The grid's right side is min, the seven stats, fpts, sim, val, pub, val, own: every stat cell a pin box with the books' lines
    in its hover. A total typed into a stat box (`/team/<TEAM>/pin` with points=...) is the player's for the game: his projected
    points become the total, his other stats keep the model's; blank clears it back."""
    from ivan.web import nba as D
    from ivan.sports.nba.projection.live import override_of
    from ivan.sports.nba.desk.rows import STAT_ORDER
    dg, day = _held(open_site)
    d = D.desk_for(dg)
    team = next((t for t in d.slate.teams() if len(d.proj.get(t, [])) >= 5), None)
    if team is None:
        pytest.skip("no projected team on the held slate")
    r = open_site.get(f"/nba/{day}/{dg}/team/{team}")
    assert r.status_code == 200 and "hist fppm" not in r.text and ">fpts</th>" in r.text
    for st in STAT_ORDER:
        assert f'name="{st}"' in r.text and f'data-pin="{st}"' in r.text
    assert r.text.count('class="px n hover statcell') >= 7 * 5                               # a cell per stat per row
    assert 'hx-trigger="mouseenter once"' in r.text and "loading the books' lines" in r.text  # the hover is fetched when hovered
    key = d.proj[team].index[0]
    h = open_site.get(f"/nba/{day}/{dg}/team/{team}/props/{key}/points")
    assert h.status_code == 200 and ("<th>book</th>" in h.text or "not priced for this game: history" in h.text)   # the books' lines, or why not
    assert open_site.get(f"/nba/{day}/{dg}/team/{team}/props/{key}/nope").status_code == 404
    old = override_of(d.overrides, key)
    before_pts, before_reb = float(d.proj[team].loc[key, "proj_points"]), float(d.proj[team].loc[key, "proj_rebounds"])
    try:
        r = open_site.post(f"/nba/{day}/{dg}/team/{team}/pin", data={"player": key, "points": "31"})
        assert r.status_code == 200 and override_of(d.overrides, key).get("points") == 31.0
        row = d.proj[team].loc[key]
        assert abs(float(row["proj_points"]) - 31.0) < 0.05 and list(row["stat_pinned"]) == ["points"]      # his points are the total
        assert abs(float(row["proj_rebounds"]) - before_reb) < 0.05                                          # his rebounds are the model's
        assert 'name="points"' in r.text and 'pinned' in r.text
        r = open_site.post(f"/nba/{day}/{dg}/team/{team}/pin", data={"player": key, "points": ""})
        assert r.status_code == 200 and override_of(d.overrides, key).get("points") is None
        assert abs(float(d.proj[team].loc[key, "proj_points"]) - before_pts) < 0.5
    finally:
        from ivan.sports.nba import slates as _S
        with _S.using(dg):                                                                         # the cleanup outside a request: the slate in context by hand
            d.set_override(key, points=old.get("points")); d.reproject(team)



def _ledger(fees, won, gross_win, day):
    """One sport's results ledger with one scored contest (ivan.core.pnl.roll_up)."""
    from ivan.core import pnl
    row = dict(final=True, fees=fees, won=won, found=10, cashed=2, gross_win=gross_win, gross_loss=gross_win - (won - fees))
    return dict(pnl.roll_up([dict(key="1", day=day, complete=True, contests=[row])]), source="a test ledger")


def test_home_shows_each_sports_money_and_the_total_with_its_tax(open_site, monkeypatch):
    """Each tile carries its sport's ledger (Sport.ledger); the strip above sums the year over every sport and prices the tax on it
    (ivan.core.pnl.combined: winnings less the deductible share of the losses). A machine with no ledger shows no money at all."""
    from datetime import datetime
    day = f"{datetime.now().year}-01-05"
    monkeypatch.setattr(REGISTRY["nba"], "ledger", lambda now: _ledger(1000.0, 400.0, 300.0, day))
    monkeypatch.setattr(REGISTRY["nfl"], "ledger", lambda now: _ledger(2000.0, 3000.0, 2500.0, day))
    monkeypatch.setattr(REGISTRY["mma"], "ledger", None)
    r = open_site.get("/")
    assert r.status_code == 200 and 'class="money"' in r.text
    for text in ("+$400.00", "-$600.00", "+$1,000.00", "Tax on", "90% of losses deductible", "won $2,800.00", "lost $2,400.00", "every entry fee as a loss"):
        assert text in r.text, text
    monkeypatch.setattr(REGISTRY["nfl"], "ledger", lambda now: 1 / 0)          # one sport's ledger failing: its tile says so, the rest stands
    r = open_site.get("/")
    assert r.status_code == 200 and "results unavailable: ZeroDivisionError" in r.text and "-$600.00" in r.text
    monkeypatch.setattr(REGISTRY["nba"], "ledger", None)
    monkeypatch.setattr(REGISTRY["nfl"], "ledger", None)
    r = open_site.get("/")
    assert r.status_code == 200 and 'class="money"' not in r.text and "net ·" not in r.text


def test_nfl_slate_page_shows_what_our_entries_made(open_site, monkeypatch):
    """A past NFL slate's backtest page leads with its results (ivan.sports.nfl.results.slate_results), a contest row opens the
    contest's panel; on a consumer (the server) neither is shown."""
    from ivan.core import pnl
    from ivan.web.nfl import core as N
    won = dict(contest_id=77, draft_group="999999", name="NFL Showdown $3 Play-Action", entry_fee=3.0, mode="showdown", field=1000, lock=None, day="2020-01-01",
               ours=3, found=3, final=True, detail=True, fees=9.0, won=12.0, net=3.0, roi=1 / 3, best_rank=1, best_points=120.5, cashed=2, min_cash=None,
               gross_win=7.0, gross_loss=4.0, export_at=None, scored="DraftKings' standings")
    open_ = dict(won, contest_id=78, name="NFL Showdown never scraped", final=False, found=0, ours=20, fees=60.0, won=None, net=None, roi=None,
                 best_rank=None, best_points=None, cashed=0, gross_win=None, gross_loss=None, scored="the chooser entered it; no standings scraped")
    monkeypatch.setattr(N.results, "slate_results", lambda session, key: dict(slate_key=str(key), contests=[won, open_], summary=pnl.agg([won, open_]), complete=False))
    monkeypatch.setattr(N.results, "contest_entries", lambda session, cid: dict(contest=won, entries=[
        dict(entry_id=1, rank=1, tie_size=1, points=120.5, payout=10.0, net=7.0, lineup="CPT Josh Allen FLEX James Cook", copies=1)]) if cid == 77 else None)
    r = open_site.get("/nfl/2020-01-01/999999/?view=backtest")
    assert r.status_code == 200
    for text in ("Results", "NFL Showdown $3 Play-Action", "+$3.00", "2 / 3", "/nfl/results/contest/77", "unscored: the chooser entered it", "$60.00 in fees out of the totals"):
        assert text in r.text, text
    r = open_site.get("/nfl/results/contest/77")
    assert r.status_code == 200 and "CPT Josh Allen FLEX James Cook" in r.text and "+$7.00" in r.text
    assert open_site.get("/nfl/results/contest/5").status_code == 404
    monkeypatch.setattr(N, "_consumer", lambda: True)
    r = open_site.get("/nfl/2020-01-01/999999/?view=backtest")
    assert r.status_code == 200 and "NFL Showdown $3 Play-Action" not in r.text
    assert open_site.get("/nfl/results/contest/77").status_code == 404
