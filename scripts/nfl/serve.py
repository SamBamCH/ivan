#!/usr/bin/env python3
"""
Run the Ivan web UI.

    scripts/nfl/serve.py [--host 0.0.0.0] [--port 8000]

No auth: bind to a LAN/Tailscale address, not the public internet.
"""
import argparse

import uvicorn


def main():
    p = argparse.ArgumentParser(description="Run the Ivan web UI")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run("ivan.web.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
