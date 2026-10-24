"""The NFL web app gives requests from the Cloudflare tunnel the NBA desk only (ivan.web.app._public_only_the_desk)."""
import pytest

pytest.importorskip("ivan.web.app")
from fastapi.testclient import TestClient

from ivan.web.app import app

CF = {"cf-ray": "abc-ORD", "cf-connecting-ip": "203.0.113.7"}


def test_tunnel_requests_get_the_desk_only():
    c = TestClient(app, client=("127.0.0.1", 50000))
    assert c.get("/jobs", headers=CF).status_code == 404
    assert c.get("/static/htmx.min.js", headers=CF).status_code == 404
    r = c.get("/", headers=CF, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/nba/"
    assert c.get("/nba/login", headers=CF).status_code in (200, 503)      # the desk itself (503 when no password is set)
