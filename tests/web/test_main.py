"""The one Ivan UI (ivan.web.main): every sport mounted under one app, one password for all of it, the frame's pages up."""
import os

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


def test_nba_bar_keeps_the_slate_being_loaded(open_site, monkeypatch):
    """While a load runs the desk still holds the old pool; the bar must show the slate asked for, not the one being replaced."""
    from ivan.web import nba_desk as D
    monkeypatch.setattr(D.desk, "busy", "loading slate 999001")
    r = open_site.get("/nba/2099-01-02/999001/")
    assert r.status_code == 200
    assert "loading slate 999001" in r.text
    assert 'value="/nba/2099-01-02/999001/"' not in r.text or 'value="/nba/2099-01-02/999001/" data-day="2099-01-02" selected' in r.text
