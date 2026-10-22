# src/ivan/web/auth.py
"""
A password on a desk.

install(app, templates) adds a login page at <root>/login and a check on every
other request. With no password configured nothing changes (the laptop's
desk stays open on localhost).

The password comes from .env as IVAN_DESK_PASSWORD_HASH (pbkdf2, made by
scripts/desk_password.py; the plain password is never stored) or, as a
fallback, IVAN_DESK_PASSWORD in plain text. A browser that logged in holds a
cookie signed with a random secret kept in data/.desk_secret (mode 600, made
on first use); delete that file to log everyone out. The typed password is
only ever compared, never stored, logged, echoed or passed anywhere.

Guessing is throttled per client address (five misses, then a lockout that
doubles each miss, up to an hour) and globally (thirty misses in ten
minutes lock the page for ten minutes); every miss is logged with the
address. The app binds the tailnet address only, so a guesser has to be a
tailnet member to begin with.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ivan.core.utils import PROJECT_ROOT

log = logging.getLogger("ivan.web.auth")

COOKIE = "ivan_desk"
DAYS = 30
ENV = "IVAN_DESK_PASSWORD"
ENV_HASH = "IVAN_DESK_PASSWORD_HASH"
SECRET_FILE = PROJECT_ROOT / "data" / ".desk_secret"
MAX_LEN = 128                     # bytes of password accepted
PBKDF2_ITERS = 600_000
PER_IP_FREE = 5                   # misses before a lockout
PER_IP_MAX_LOCK = 3600.0
GLOBAL_WINDOW = 600.0
GLOBAL_MISSES = 30
GLOBAL_LOCK = 600.0


# --- the password ----------------------------------------------------------
def make_hash(pw: str, iters: int = PBKDF2_ITERS) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, iters)
    return f"pbkdf2_sha256${iters}${salt.hex()}${dk.hex()}"


def check_hash(pw: str, stored: str) -> bool:
    try:
        algo, iters, salt, dk = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        got = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(got, bytes.fromhex(dk))
    except (ValueError, TypeError):
        return False


def configured() -> bool:
    return bool(os.environ.get(ENV_HASH, "").strip() or os.environ.get(ENV, "").strip())


def check_password(pw: str) -> bool:
    """Constant-time against the hash when there is one, else the plain password."""
    if not pw or len(pw.encode()) > MAX_LEN:
        return False
    h = os.environ.get(ENV_HASH, "").strip()
    if h:
        return check_hash(pw, h)
    plain = os.environ.get(ENV, "").strip()
    return bool(plain) and hmac.compare_digest(pw.encode(), plain.encode())


# --- the cookie -------------------------------------------------------------
_secret: Optional[bytes] = None
_secret_lock = threading.Lock()


def secret() -> bytes:
    global _secret
    with _secret_lock:
        if _secret is None:
            try:
                _secret = bytes.fromhex(SECRET_FILE.read_text().strip())
                if len(_secret) < 32:
                    raise ValueError
            except (OSError, ValueError):
                _secret = secrets.token_bytes(32)
                SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
                SECRET_FILE.touch(mode=0o600, exist_ok=True)
                SECRET_FILE.write_text(_secret.hex())
        return _secret


def token(exp: Optional[int] = None) -> str:
    exp = exp or int(time.time()) + DAYS * 86400
    return f"{exp}.{hmac.new(secret(), str(exp).encode(), 'sha256').hexdigest()}"


def valid(cookie: Optional[str]) -> bool:
    if not cookie or "." not in cookie or len(cookie) > 100:
        return False
    exp_s, sig = cookie.split(".", 1)
    if not exp_s.isdigit() or int(exp_s) < time.time():
        return False
    return hmac.compare_digest(hmac.new(secret(), exp_s.encode(), "sha256").hexdigest(), sig)


# --- the throttle -----------------------------------------------------------
class Throttle:
    def __init__(self):
        self.lock = threading.Lock()
        self.misses: dict[str, int] = defaultdict(int)
        self.locked_until: dict[str, float] = {}
        self.recent: deque = deque()
        self.global_until = 0.0

    def locked(self, ip: str, now: Optional[float] = None) -> float:
        """Seconds of lockout left for this address (0 = free)."""
        now = now or time.time()
        with self.lock:
            return max(self.locked_until.get(ip, 0.0), self.global_until) - now

    def miss(self, ip: str, now: Optional[float] = None) -> None:
        now = now or time.time()
        with self.lock:
            self.misses[ip] += 1
            n = self.misses[ip]
            if n >= PER_IP_FREE:
                self.locked_until[ip] = now + min(PER_IP_MAX_LOCK, 60.0 * 2 ** (n - PER_IP_FREE))
            self.recent.append(now)
            while self.recent and self.recent[0] < now - GLOBAL_WINDOW:
                self.recent.popleft()
            if len(self.recent) >= GLOBAL_MISSES:
                self.global_until = now + GLOBAL_LOCK

    def hit(self, ip: str) -> None:
        with self.lock:
            self.misses.pop(ip, None); self.locked_until.pop(ip, None)


throttle = Throttle()


def root_of(request: Request) -> str:
    return request.scope.get("root_path", "") or ""


def _ip(request: Request) -> str:
    return request.client.host if request.client else "?"       # the app faces clients directly: no forwarded headers are trusted


class _Gate(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not configured():
            return await call_next(request)
        root = root_of(request)
        path = request.url.path
        if path == f"{root}/login" or path.startswith(f"{root}/static/") or valid(request.cookies.get(COOKIE)):
            resp = await call_next(request)
        elif request.headers.get("HX-Request"):
            resp = Response(status_code=401, headers={"HX-Redirect": f"{root}/login"})
        else:
            resp = RedirectResponse(f"{root}/login", status_code=303)
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        if path == f"{root}/login":
            resp.headers["Cache-Control"] = "no-store"
        return resp


def install(app: FastAPI, templates) -> None:
    app.add_middleware(_Gate)

    def page(request, error="", status=200):
        return templates.TemplateResponse(request, "_login.html", {"root": root_of(request), "error": error, "title": app.title}, status_code=status)

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        return page(request)

    @app.post("/login", response_class=HTMLResponse)
    def login(request: Request, pw: str = Form("")):
        ip = _ip(request)
        left = throttle.locked(ip)
        if left > 0:
            return page(request, f"too many tries; wait {int(left // 60) + 1} min", 429)
        if not configured() or not check_password(pw):
            throttle.miss(ip)
            log.warning(f"desk login failed from {ip}")
            time.sleep(1.0)
            return page(request, "wrong password", 401)
        throttle.hit(ip)
        log.info(f"desk login from {ip}")
        r = RedirectResponse(f"{root_of(request)}/", status_code=303)
        r.set_cookie(COOKIE, token(), max_age=DAYS * 86400, httponly=True, samesite="lax", path=f"{root_of(request)}/")
        return r

    @app.get("/logout")
    def logout(request: Request):
        r = RedirectResponse(f"{root_of(request)}/login", status_code=303)
        r.delete_cookie(COOKIE, path=f"{root_of(request)}/")
        return r
