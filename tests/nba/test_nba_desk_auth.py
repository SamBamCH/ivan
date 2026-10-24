"""The desk's password gate (ivan.web.auth): hashed password, random cookie secret, throttle, mount prefix."""
import time

import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from ivan.web import auth

TEMPLATES = Jinja2Templates(directory=str(__import__("ivan.web", fromlist=["x"]).__path__[0]) + "/templates")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "SECRET_FILE", tmp_path / ".desk_secret"); monkeypatch.setattr(auth, "_secret", None)
    monkeypatch.setattr(auth, "throttle", auth.Throttle())
    monkeypatch.setattr(auth.time, "sleep", lambda s: None)
    monkeypatch.delenv(auth.ENV, raising=False); monkeypatch.delenv(auth.ENV_HASH, raising=False)


def make(prefix: str = ""):
    inner = FastAPI(title="t")
    auth.install(inner, TEMPLATES)

    @inner.get("/")
    def home():
        return PlainTextResponse("home")

    if not prefix:
        return inner
    outer = FastAPI(); outer.mount(prefix, inner)
    return outer


def test_open_when_no_password():
    assert TestClient(make()).get("/").text == "home"


def test_hashed_password_login_and_cookie(monkeypatch):
    monkeypatch.setenv(auth.ENV_HASH, auth.make_hash("s3cret-pw", iters=1000))
    c = TestClient(make(), follow_redirects=False)
    r = c.get("/"); assert r.status_code == 303 and r.headers["location"] == "/login" and r.headers["x-frame-options"] == "DENY"
    r = c.get("/", headers={"HX-Request": "true"}); assert r.status_code == 401 and r.headers["HX-Redirect"] == "/login"
    r = c.get("/login"); assert "password" in r.text and r.headers["cache-control"] == "no-store"
    assert c.post("/login", data={"pw": "nope"}).status_code == 401
    assert c.post("/login", data={"pw": "x" * 300}).status_code == 401                       # over the length cap, never hashed
    r = c.post("/login", data={"pw": "s3cret-pw"}); assert r.status_code == 303 and auth.COOKIE in r.cookies
    assert c.get("/").text == "home"
    assert auth.SECRET_FILE.exists() and (auth.SECRET_FILE.stat().st_mode & 0o777) == 0o600
    c.get("/logout"); assert c.get("/").status_code == 303


def test_plain_password_fallback(monkeypatch):
    monkeypatch.setenv(auth.ENV, "plain-pw")
    c = TestClient(make(), follow_redirects=False)
    assert c.post("/login", data={"pw": "plain-pw"}).status_code == 303


def test_throttle_locks_an_address_then_everyone(monkeypatch):
    monkeypatch.setenv(auth.ENV, "plain-pw")
    c = TestClient(make(), follow_redirects=False)
    for _ in range(auth.PER_IP_FREE):
        assert c.post("/login", data={"pw": "no"}).status_code == 401
    r = c.post("/login", data={"pw": "plain-pw"}); assert r.status_code == 429 and "wait" in r.text    # even the right password waits
    t = auth.Throttle()
    for i in range(auth.GLOBAL_MISSES):
        t.miss(f"10.0.0.{i}", now=1000.0)
    assert t.locked("10.9.9.9", now=1000.0) > 0                                                    # a fresh address is locked too
    assert t.locked("10.9.9.9", now=1000.0 + auth.GLOBAL_LOCK + 1) <= 0


def test_cookie_secret_not_derived_from_password(monkeypatch):
    monkeypatch.setenv(auth.ENV, "plain-pw")
    tok = auth.token()
    assert auth.valid(tok) and not auth.valid("1" + tok) and not auth.valid(auth.token(exp=int(time.time()) - 1))
    auth._secret = None; auth.SECRET_FILE.unlink()                                                # rotating the secret logs everyone out
    assert not auth.valid(tok)


def test_mounted_prefix_in_redirects_and_forms(monkeypatch):
    monkeypatch.setenv(auth.ENV, "s3cret")
    c = TestClient(make("/nba"), follow_redirects=False)
    r = c.get("/nba/"); assert r.status_code == 303 and r.headers["location"] == "/nba/login"
    assert 'action="/nba/login"' in c.get("/nba/login").text
    r = c.post("/nba/login", data={"pw": "s3cret"}); assert r.headers["location"] == "/nba/"
    assert c.get("/nba/").text == "home"


TUNNEL = {"cf-connecting-ip": "203.0.113.7", "cf-ray": "abc-ORD"}


def test_tunnel_desk_without_a_password_refuses(monkeypatch):
    c = TestClient(make(), client=("127.0.0.1", 50000))
    assert c.get("/").status_code == 200                                   # local and no password: the laptop's open desk
    assert c.get("/", headers=TUNNEL).status_code == 503                   # through the tunnel, no password: refuse
    monkeypatch.setenv(auth.PUBLIC_ENV, "1")
    assert c.get("/").status_code == 503                                   # marked public: refuse even locally


def test_tunnel_lockout_is_per_visitor_and_cookie_is_secure(monkeypatch):
    monkeypatch.setenv(auth.ENV, "right")
    c = TestClient(make(), client=("127.0.0.1", 50000))
    for _ in range(auth.PER_IP_FREE):
        c.post("/login", data={"pw": "wrong"}, headers={**TUNNEL, "cf-connecting-ip": "198.51.100.1"})
    assert c.post("/login", data={"pw": "right"}, headers={**TUNNEL, "cf-connecting-ip": "198.51.100.1"}).status_code == 429
    ok = c.post("/login", data={"pw": "right"}, headers=TUNNEL, follow_redirects=False)     # another visitor is not locked out
    assert ok.status_code == 303 and "secure" in ok.headers["set-cookie"].lower()


def test_forged_cloudflare_header_from_a_tailnet_peer_is_ignored(monkeypatch):
    monkeypatch.setenv(auth.ENV, "right")
    c = TestClient(make(), client=("100.64.0.9", 50000))                  # a tailnet member, not this machine
    for _ in range(auth.PER_IP_FREE):
        c.post("/login", data={"pw": "wrong"}, headers={"cf-connecting-ip": f"10.0.0.{_}"})
    assert c.post("/login", data={"pw": "right"}, headers={"cf-connecting-ip": "10.0.0.99"}).status_code == 429   # its own address counted
