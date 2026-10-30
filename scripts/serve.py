#!/usr/bin/env python3
"""
Run the Ivan web UI: one site for every sport (ivan.web.main).

    scripts/serve.py [--host 127.0.0.1] [--port 8000] [--sports nba,nfl,mma]

The password is IVAN_PASSWORD_HASH in .env (scripts/password.py); without one the site is open, so
bind to localhost or the tailnet, never the internet. --sports (or IVAN_SPORTS) mounts a subset.
"""
import argparse
import os

import uvicorn


def main():
    p = argparse.ArgumentParser(description="Run the Ivan web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--sports", default=None, help="comma-separated subset to mount (default: every sport)")
    args = p.parse_args()
    if args.sports:
        os.environ["IVAN_SPORTS"] = args.sports
    print(f"Ivan: http://{args.host}:{args.port}/  (Ctrl-C to stop)", flush=True)
    uvicorn.run("ivan.web.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
