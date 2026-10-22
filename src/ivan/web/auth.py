# src/ivan/web/auth.py
"""
A password on a desk.

install(app, templates) adds a login page at <root>/login and a check on every
other request. With IVAN_DESK_PASSWORD unset nothing changes (the laptop's
desk stays open on localhost); with it set, a browser needs to have logged in
once: the cookie is an HMAC of an expiry stamp under a key derived from the
password, so changing the password logs everyone out and nothing is stored
server side. Plain http is fine on the tailnet; the cookie is httponly + lax.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

COOKIE = "ivan_desk"
DAYS = 30
ENV = "IVAN_DESK_PASSWORD"


def password() -> Optional[str]:
    p = os.environ.get(ENV, "").strip()
    return p or None


def _key(pw: str) -> bytes:
    return hashlib.sha256(b"ivan-desk-cookie:" + pw.encode()).digest()


def token(pw: str, exp: Optional[int] = None) -> str:
    exp = exp or int(time.time()) + DAYS * 86400
    sig = hmac.new(_key(pw), str(exp).encode(), "sha256").hexdigest()
    return f"{exp}.{sig}"


def valid(pw: str, cookie: Optional[str]) -> bool:
    if not cookie or "." not in cookie:
        return False
    exp_s, sig = cookie.split(".", 1)
    if not exp_s.isdigit() or int(exp_s) < time.time():
        return False
    return hmac.compare_digest(hmac.new(_key(pw), exp_s.encode(), "sha256").hexdigest(), sig)


def root_of(request: Request) -> str:
    return request.scope.get("root_path", "") or ""


class _Gate(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        pw = password()
        if pw is None:
            return await call_next(request)
        root = root_of(request)
        path = request.url.path
        if path == f"{root}/login" or path.startswith(f"{root}/static/"):
            return await call_next(request)
        if valid(pw, request.cookies.get(COOKIE)):
            return await call_next(request)
        if request.headers.get("HX-Request"):
            return Response(status_code=401, headers={"HX-Redirect": f"{root}/login"})
        return RedirectResponse(f"{root}/login", status_code=303)


def install(app: FastAPI, templates) -> None:
    app.add_middleware(_Gate)

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        return templates.TemplateResponse(request, "_login.html", {"root": root_of(request), "error": "", "title": app.title})

    @app.post("/login", response_class=HTMLResponse)
    def login(request: Request, pw: str = Form("")):
        real = password()
        if real is None or not hmac.compare_digest(pw.encode(), real.encode()):
            time.sleep(1.0)                                    # a wrong guess costs a second
            return templates.TemplateResponse(request, "_login.html", {"root": root_of(request), "error": "wrong password", "title": app.title}, status_code=401)
        r = RedirectResponse(f"{root_of(request)}/", status_code=303)
        r.set_cookie(COOKIE, token(real), max_age=DAYS * 86400, httponly=True, samesite="lax", path=f"{root_of(request)}/")
        return r

    @app.get("/logout")
    def logout(request: Request):
        r = RedirectResponse(f"{root_of(request)}/login", status_code=303)
        r.delete_cookie(COOKIE, path=f"{root_of(request)}/")
        return r
