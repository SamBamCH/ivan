#!/usr/bin/env python3
"""
scripts/nba/fetch_historical_lines.py

Closing-ish game lines for past slates from the-odds-api's historical endpoint:
one snapshot per game day (default 23:45 UTC, before the evening tips), every
game's moneyline, spread and total across books. Each snapshot is billed
(historical requests cost roughly ten times a live one per market), so the
script prints the cost of the first request and the projected total, and
--dry-run only lists the days it would fetch.

    scripts/nba/fetch_historical_lines.py --start 2025-10-21 --end 2026-01-31 --dry-run
    scripts/nba/fetch_historical_lines.py --start 2025-12-01 --end 2025-12-01          # one day: read the cost
    scripts/nba/fetch_historical_lines.py --start 2025-10-21 --end 2026-01-31 --max-credits 3500

Raw payloads -> data/nba/raw/odds/lines_hist/<day>; the consensus per game
(ivan.sports.nba.projection.lines.consensus) is appended to
data/nba/input/historical_lines.csv (date, home_abbr, away_abbr, spread_home,
total, p_home, books, snapshot). Days already in the CSV are skipped, so the
script can be rerun. Game days come from nba_stats.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import text

from ivan.core import raw
from ivan.core.config import require_env
from ivan.database.session import get_session, init_db
from ivan.sports.common.odds import parse_game_odds
from ivan.sports.nba.paths import INPUT_DIR
from ivan.sports.nba.projection.lines import consensus

OUT = INPUT_DIR / "historical_lines.csv"
URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"


def game_days(session, start: date, end: date) -> list[date]:
    rows = session.execute(text("select distinct game_date from nba_stats where substr(game_id,1,3) in ('002','004') and game_date between :a and :b order by game_date"),
                           {"a": start.isoformat(), "b": end.isoformat()}).all()
    return [pd.to_datetime(r[0]).date() for r in rows]


def fetch_day(key: str, day: date, hour_utc: int, minute_utc: int) -> tuple[dict | None, dict]:
    stamp = datetime(day.year, day.month, day.day, hour_utc, minute_utc, tzinfo=timezone.utc)
    resp = requests.get(URL, params={"apiKey": key, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american",
                                     "dateFormat": "iso", "date": stamp.strftime("%Y-%m-%dT%H:%M:%SZ")}, timeout=60)
    if resp.status_code != 200:
        print(f"  {day}: HTTP {resp.status_code} {resp.text[:200]}")
        return None, dict(resp.headers)
    return resp.json(), dict(resp.headers)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--snapshot-utc", default="23:45", help="time of day (UTC) of the snapshot, before the evening tips")
    ap.add_argument("--max-credits", type=int, default=None, help="stop once this many credits have been spent in this run")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    hh, mm = (int(x) for x in a.snapshot_utc.split(":"))
    session = get_session(init_db("nba"))
    days = game_days(session, date.fromisoformat(a.start), date.fromisoformat(a.end))
    have = set()
    if OUT.exists():
        have = set(pd.to_datetime(pd.read_csv(OUT)["date"]).dt.date)
    todo = [d for d in days if d not in have]
    print(f"{len(days)} game days in range, {len(have & set(days))} already fetched, {len(todo)} to fetch")
    if a.dry_run or not todo:
        return 0
    key = require_env("ODDS_API_KEY")
    spent = 0; cost_per = None
    for i, day in enumerate(todo, 1):
        payload, headers = fetch_day(key, day, hh, mm)
        used = headers.get("x-requests-last"); remaining = headers.get("x-requests-remaining")
        if used is not None:
            cost_per = int(float(used)); spent += cost_per
        if payload is None:
            continue
        data = payload.get("data") or []
        raw.store("nba", "odds/lines_hist", day.isoformat(), payload, meta={"snapshot": payload.get("timestamp"), "markets": "h2h,spreads,totals"})
        df = parse_game_odds(data)
        if df.empty:
            print(f"  {day}: no games in the snapshot ({payload.get('timestamp')})")
        else:
            df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
            c = consensus(df)
            c["date"] = c.commence_time.dt.date
            c["snapshot"] = payload.get("timestamp")
            c = c[c["date"] == day] if (c["date"] == day).any() else c
            c[["date", "game_id", "home_abbr", "away_abbr", "spread_home", "total", "p_home", "books", "snapshot"]].to_csv(
                OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"  {day}: {len(c)} games ({payload.get('timestamp')}); this request {cost_per} credits, {remaining} remaining")
        if i == 1 and cost_per:
            print(f"projected cost for the remaining {len(todo) - 1} days: ~{cost_per * (len(todo) - 1)} credits")
        if a.max_credits is not None and spent >= a.max_credits:
            print(f"stopping: {spent} credits spent in this run (--max-credits {a.max_credits})")
            break
    print(f"done: {spent} credits spent -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
