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
    """Opening a lobby slate the desk does not hold starts its load (the producer fetches it; a consumer asks the laptop) and
    the page shows the progress; when a job holds the pool the page says why and offers the load as a button."""
    from datetime import date, datetime
    from ivan.web import nba as D
    from ivan.web.nba import pages as PG, pipeline as P
    from ivan.web.sports import Day, SlateInfo
    d = date(2099, 1, 2)
    lobby = SlateInfo(key="999002", day=d, lock=datetime(2099, 1, 2, 19), label="19:00 · 8 games", source="lobby", games=8,
                      load={"url": "/slate", "fields": {"draft_group": "999002", "source": "lobby", "day": d.isoformat()}, "label": "load into the desk"})
    monkeypatch.setattr(PG, "_days_of", lambda rows, now: [Day(day=d, label="Fri 01-02", tag="", slates=[lobby])])
    import ivan.sports.nba.paths as PATHS
    monkeypatch.setattr(PATHS, "ACTIVE_SLATE_JSON", tmp_path / "active.json")
    started = []

    def fake_refresh(what, dg, update_odds, lines=None):
        started.append((what, dg)); D.desk.busy = what
    monkeypatch.setattr(P, "_start_refresh", fake_refresh)
    monkeypatch.setattr(P.runner, "active", lambda: None)
    assert not D.desk.busy
    try:
        r = open_site.get("/nba/2099-01-02/999002/")
        assert r.status_code == 200 and started == [("loading slate 999002", 999002)]
        assert "loading slate 999002" in r.text and 'name="draft_group"' not in r.text
        assert json.loads((tmp_path / "active.json").read_text())["draft_group"] == 999002
        # while it loads, opening the page again does not start another
        r = open_site.get("/nba/2099-01-02/999002/")
        assert len(started) == 1
    finally:
        D.desk.busy = None
    # a job holds the pool: the page says so and offers the button
    monkeypatch.setattr(P.runner, "active", lambda: object())
    r = open_site.get("/nba/2099-01-02/999002/")
    assert r.status_code == 200 and len(started) == 1
    assert "a job is running on the current slate" in r.text and 'name="draft_group" value="999002"' in r.text


def test_nba_bar_keeps_the_slate_being_loaded(open_site, monkeypatch):
    """While a load runs the desk still holds the old pool; the bar must show the slate asked for, not the one being replaced."""
    from ivan.web import nba as D
    monkeypatch.setattr(D.desk, "busy", "loading slate 999001")
    r = open_site.get("/nba/2099-01-02/999001/")
    assert r.status_code == 200
    assert "loading slate 999001" in r.text
    assert 'value="/nba/2099-01-02/999001/"' not in r.text or 'value="/nba/2099-01-02/999001/" data-day="2099-01-02" selected' in r.text


def test_nba_step_knobs_move_only_their_own(open_site, monkeypatch):
    """Generate and sim send their popover's knobs only; the others, and the fresh-pool box, keep what the desk last had."""
    from ivan.web import nba as D
    monkeypatch.setattr(D.runner, "active", lambda: [object()])          # "a job is running": the route stores the knobs, starts nothing
    monkeypatch.setitem(D.pipeline_form, "lineups", "12345")
    monkeypatch.setitem(D.pipeline_form, "fresh", "on")
    monkeypatch.setitem(D.pipeline_form, "sims", "1000")
    assert open_site.post("/nba/pipeline/sim", data={"sims": "777", "field_cap": "8000"}).status_code == 200
    assert D.pipeline_form["sims"] == "777" and D.pipeline_form["lineups"] == "12345" and D.pipeline_form["fresh"] == "on"
    assert open_site.post("/nba/pipeline/generate", data={"cfg_fresh": "1", "lineups": "2000"}).status_code == 200
    assert D.pipeline_form["lineups"] == "2000" and D.pipeline_form["fresh"] == "" and D.pipeline_form["sims"] == "777"


def test_nba_slate_page_reads_and_the_team_page_edits(open_site):
    """The slate page carries no projection edit control; every team chip and news name is a link to the team's own page, which
    holds the card. An old /desk/?open=TEAM link lands on the team page."""
    from ivan.web import nba as D
    dg, lock = D._loaded_slate()
    if dg is None or lock is None or not D.desk.slate.teams():
        pytest.skip("no slate loaded in this database")
    day = lock.date().isoformat(); team = D.desk.slate.teams()[0]
    board = open_site.get("/nba/board?view=games")
    assert board.status_code == 200
    assert f"/nba/{day}/{dg}/{team}/" in board.text                          # a chip is a link
    assert 'name="minutes"' not in board.text and "hx-post=\"/nba/team/" not in board.text   # nothing edits here
    assert "needs you" in board.text and 'class="chain' in board.text and "Snapshot" in board.text
    page = open_site.get(f"/nba/{day}/{dg}/{team}/")
    assert page.status_code == 200 and f'id="card-{team}"' in page.text and "crumb-team" in page.text
    card = open_site.get(f"/nba/team/{team}")
    assert card.status_code == 200 and "rot-grid" in card.text and 'name="minutes"' in card.text
    assert open_site.get(f"/nba/{day}/{dg}/ZZZ/").status_code == 404
    r = open_site.get(f"/nba/desk/?open={team}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith(f"/{dg}/{team}/")
    players = open_site.get("/nba/board?view=players")
    assert players.status_code == 200 and 'class="players sortable"' in players.text and "Snapshot" not in players.text


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
    import json
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
