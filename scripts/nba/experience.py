#!/usr/bin/env python3
"""
scripts/nba/experience.py

Each player's first NBA season (ivan.sports.nba.experience): what the desk's rookie mark (R on the rotation grids) reads.
The nightly runs `fetch`; run it by hand when a new player shows up mid-day.

    scripts/nba/experience.py fetch      request nba_api's all-players list, write data/nba/derived/players/experience.json
    scripts/nba/experience.py rookies    the rookies on file
"""
import argparse

from ivan.sports.nba import experience as X


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "rookies"])
    a = ap.parse_args()
    if a.cmd == "fetch":
        X.fetch()
        return 0
    y = X.season_start_year()
    names = sorted(n for n, fy in X.load()["first_year"].items() if fy >= y)
    print(f"{len(names)} rookies ({y}): " + ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
