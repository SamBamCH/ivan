#!/usr/bin/env python3
"""
scripts/feed_server.py

Serve the data feed (ivan.core.feed bundles under data/feed) to other Ivans.

    scripts/feed_server.py --host <tailscale ip> --port 8790

Every request needs `Authorization: Bearer <token>` with the token from
$IVAN_FEED_TOKEN (.env). Paths: /<sport>/manifest.json, /<sport>/live.sql.gz,
/<sport>/reference.sql.gz, /<sport>/files.json, /<sport>/files/<path>.
Read-only, no listing, nothing outside the feed directory.
"""
from __future__ import annotations

import argparse
import hmac
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse

from ivan.core.feed import FEED_DIR

app = FastAPI(title="Ivan feed", docs_url=None, redoc_url=None)
ROOT = Path(os.environ.get("IVAN_FEED_DIR", str(FEED_DIR))).resolve()
TOKEN = os.environ.get("IVAN_FEED_TOKEN", "")


def _check(authorization: str | None) -> None:
    if not TOKEN:
        raise HTTPException(500, "IVAN_FEED_TOKEN is not set on the server")
    given = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(given, TOKEN):
        raise HTTPException(401, "bad token")


@app.get("/{sport}/{path:path}")
def get_file(sport: str, path: str, request: Request, authorization: str | None = Header(default=None)):
    _check(authorization)
    target = (ROOT / sport / path).resolve()
    if not str(target).startswith(str(ROOT / sport)) or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    if not TOKEN:
        print("set IVAN_FEED_TOKEN in .env first"); return 1
    print(f"feed: http://{a.host}:{a.port}/  serving {ROOT}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
