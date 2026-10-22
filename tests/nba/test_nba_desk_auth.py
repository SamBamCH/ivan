"""The desk's password gate (ivan.web.auth) on its own app and mounted under a prefix."""
import time

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from ivan.web import auth

TEMPLATES = Jinja2Templates(directory=str(__import__("ivan.web", fromlist=["x"]).__path__[0]) + "/templates")


def make(prefix: str = ""):
    inner = FastAPI(title="t")
    auth.install(inner, TEMPLATES)

    @inner.get("/")
    def home():
        return PlainTextResponse("home")

    if not prefix:
        return inner
    outer = FastAPI()
    outer.mount(prefix, inner)
    return outer


def test_open_when_no_password(monkeypatch):
    monkeypatch.delenv(auth.ENV, raising=False)
    assert TestClient(make()).get("/").text == "home"


def test_gate_login_and_cookie(monkeypatch):
    monkeypatch.setenv(auth.ENV, "s3cret")
    c = TestClient(make(), follow_redirects=False)
    r = c.get("/"); assert r.status_code == 303 and r.headers["location"] == "/login"
    r = c.get("/", headers={"HX-Request": "true"}); assert r.status_code == 401 and r.headers["HX-Redirect"] == "/login"
    assert "password" in c.get("/login").text
    t = time.time(); r = c.post("/login", data={"pw": "nope"}); assert r.status_code == 401 and time.time() - t >= 0.9
    r = c.post("/login", data={"pw": "s3cret"}); assert r.status_code == 303 and auth.COOKIE in r.cookies
    assert c.get("/").text == "home"
    r = c.get("/logout"); assert r.status_code == 303
    assert c.get("/").status_code == 303


def test_token_round_trip_and_expiry():
    assert auth.valid("pw", auth.token("pw"))
    assert not auth.valid("other", auth.token("pw"))
    assert not auth.valid("pw", auth.token("pw", exp=int(time.time()) - 1))
    assert not auth.valid("pw", "garbage") and not auth.valid("pw", None)


def test_mounted_prefix_in_redirects_and_forms(monkeypatch):
    monkeypatch.setenv(auth.ENV, "s3cret")
    c = TestClient(make("/nba"), follow_redirects=False)
    r = c.get("/nba/"); assert r.status_code == 303 and r.headers["location"] == "/nba/login"
    assert 'action="/nba/login"' in c.get("/nba/login").text
    r = c.post("/nba/login", data={"pw": "s3cret"}); assert r.headers["location"] == "/nba/"
    assert c.get("/nba/").text == "home"
