#!/usr/bin/env python
import argparse
from ivan.sports.mlb.stat_fetcher import fetch_and_store

if __name__ == "__main__":
    p = argparse.ArgumentParser("Backfill MLB boxscore data")
    p.add_argument(
        "--days-back", type=int, default=730,
        help="How many days to backfill (default: 730)"
    )
    args = p.parse_args()

    # one thin call into your library
    fetch_and_store(days_back=args.days_back)