#!/usr/bin/env python3
"""Serve the NBA news desk (ivan.web.nba_desk). Separate process from the NFL app."""
import argparse
import uvicorn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    print(f"NBA desk: http://{args.host}:{args.port}  (first request builds the slate; Ctrl-C to stop)", flush=True)
    uvicorn.run("ivan.web.nba_desk:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
