#!/usr/bin/env python3
"""
scripts/nba/fetch_historical_props.py

Player props for past NBA games from the-odds-api's historical endpoints, kept so
nothing is ever fetched twice.

Per game day: the event list at --events-utc (1 credit), then for every game that
day one snapshot of its player props --before minutes before its tip. Historical
odds are billed about 10 credits per market per region, so the default is the six
main markets (points, rebounds, assists, threes, steals, blocks: ~60 credits a
game); --alternates adds the alternate lines (~120 a game).

Storage (raw, the same store as the live props; the index makes it findable):
    data/nba/raw/odds/events_hist/<day>/<stamp>.json.gz       the day's event list
    data/nba/raw/odds/props_hist/<event id>/<stamp>.json.gz   one game's props snapshot
A game whose props are already stored is skipped, so reruns cost only the event
lists of days not yet listed. Derived (rebuildable from raw at any time):
    data/nba/derived/props_hist/<day>.parquet   player, market, sportsbook, side, point, price, last_update,
                                                event_id, commence_time, home, away, snapshot

Spending guards: --max-credits stops the run after that many credits; --reserve
stops before the account's remaining credits fall below it (default 3000, so the
live capture still has its props budget). --dry-run lists the days and the
estimated cost and spends nothing.

    scripts/nba/fetch_historical_props.py --stokastic-days --dry-run          # the days we hold Stokastic projections for
    scripts/nba/fetch_historical_props.py --stokastic-days --max-credits 1500
    scripts/nba/fetch_historical_props.py --field-days --max-credits 4000      # days with field histograms, richest first
    scripts/nba/fetch_historical_props.py --start 2025-12-01 --end 2025-12-07
    scripts/nba/fetch_historical_props.py --rebuild                           # re-derive the parquet files from raw, no requests
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import text

from ivan.core import raw
from ivan.core.config import require_env
from ivan.database.session import get_session, init_db
from ivan.sports.common.odds import credits_remaining, get_historical_event_odds, get_historical_events, parse_odds
from ivan.sports.nba.paths import NBA_DATA_DIR

SPORT_KEY = "basketball_nba"
MAIN = "player_points,player_rebounds,player_assists,player_threes,player_steals,player_blocks"
ALTS = ",".join(m + "_alternate" for m in MAIN.split(","))
OUT_DIR = NBA_DATA_DIR / "derived" / "props_hist"
CREDITS_PER_MARKET = 10


def stokastic_days(session) -> list[date]:
    rows = session.execute(text("select distinct substr(snapshot_date, 1, 10) from projection_snapshot order by 1")).all()
    return [date.fromisoformat(r[0]) for r in rows]


def field_days(session) -> list[date]:
    """Days with last season's field histograms (classic_contest_metadata), the days with the most contests first."""
    rows = session.execute(text("select h.date from classic_contest_metadata c join historical_contest_information h using(contest_id)")).all()
    days = pd.Series([pd.Timestamp(r[0]).tz_localize("UTC").tz_convert("America/New_York").date() if pd.Timestamp(r[0]).tzinfo is None
                      else pd.Timestamp(r[0]).tz_convert("America/New_York").date() for r in rows])
    return list(days.value_counts().index)


def game_days(session, start: date, end: date) -> list[date]:
    rows = session.execute(text("select distinct game_date from nba_stats where substr(game_id,1,3) in ('002','004') and game_date between :a and :b order by game_date"),
                           {"a": start.isoformat(), "b": end.isoformat()}).all()
    return [pd.to_datetime(r[0]).date() for r in rows]


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def day_events(key: str, day: date, events_utc: str, log=print) -> tuple[list, int]:
    """The day's games (from storage when listed before). Returns (events, credits spent)."""
    art = raw.latest("nba", "odds/events_hist", day.isoformat())
    if art is not None:
        body = raw.load(art.path, "nba")
        return list(body.get("data") or []), 0
    hh, mm = (int(x) for x in events_utc.split(":"))
    at = datetime(day.year, day.month, day.day, hh, mm, tzinfo=timezone.utc)
    data, ts, headers = get_historical_events(SPORT_KEY, key, _iso(at))
    cost = int(float(headers.get("x-requests-last") or 0))
    if data is None:
        return [], cost
    raw.store("nba", "odds/events_hist", day.isoformat(), {"data": data, "timestamp": ts}, meta={"at": _iso(at), "snapshot": ts})
    return list(data), cost


def on_day(events: list, day: date) -> list:
    """Events tipping on `day` in US Eastern time (the NBA's game date)."""
    out = []
    for e in events:
        t = pd.Timestamp(e["commence_time"]).tz_convert("America/New_York")
        if t.date() == day:
            out.append(e)
    return out


def derive_day(day: date, events: list) -> int:
    """Parse every stored props snapshot of the day's games into one parquet file. Returns rows."""
    frames = []
    for e in on_day(events, day):
        art = raw.latest("nba", "odds/props_hist", e["id"])
        if art is None:
            continue
        body = raw.load(art.path, "nba")
        df = parse_odds({e["id"]: body.get("data") or {}})
        if df.empty:
            continue
        frames.append(df.assign(event_id=e["id"], commence_time=e["commence_time"], home=e.get("home_team"), away=e.get("away_team"),
                                snapshot=body.get("timestamp")))
    if not frames:
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT_DIR / f"{day.isoformat()}.parquet", index=False)
    return len(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--stokastic-days", action="store_true", help="the days projection_snapshot holds Stokastic projections for")
    ap.add_argument("--field-days", action="store_true", help="the days with last season's field histograms, most contests first")
    ap.add_argument("--before", type=int, default=60, help="minutes before each tip for the props snapshot")
    ap.add_argument("--events-utc", default="16:00", help="time of day (UTC) of the event-list snapshot")
    ap.add_argument("--alternates", action="store_true", help="also the alternate lines (about doubles the cost)")
    ap.add_argument("--max-credits", type=int, default=None)
    ap.add_argument("--reserve", type=int, default=3000, help="never let the account's remaining credits fall below this")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="re-derive the parquet files from stored raw, no requests")
    a = ap.parse_args()
    session = get_session(init_db("nba"))
    days: list = []
    if a.field_days:
        days += field_days(session)                                # priority order: the cap covers the richest days first
    extra = set()
    if a.stokastic_days:
        extra |= set(stokastic_days(session))
    if a.start:
        extra |= set(game_days(session, date.fromisoformat(a.start), date.fromisoformat(a.end or a.start)))
    days += sorted(d for d in extra if d not in days)
    if not days:
        print("no days: give --stokastic-days and/or --start/--end"); return 1
    markets = MAIN + ("," + ALTS if a.alternates else "")
    per_game = CREDITS_PER_MARKET * len(markets.split(","))
    if a.rebuild:
        n = 0
        for d in days:
            art = raw.latest("nba", "odds/events_hist", d.isoformat())
            if art is not None:
                n += derive_day(d, list((raw.load(art.path, "nba") or {}).get("data") or []))
        print(f"re-derived {n:,} prop rows for {len(days)} days -> {OUT_DIR}"); return 0
    listed = [d for d in days if raw.latest("nba", "odds/events_hist", d.isoformat()) is not None]
    print(f"{len(days)} days ({min(days)} .. {max(days)}), {len(listed)} already listed; {per_game} credits per game for {len(markets.split(','))} markets")
    if a.dry_run:
        print(f"estimate: ~{len(days) - len(listed)} credits for event lists + ~{per_game} x games not yet stored "
              f"(~{per_game * 8 * len(days):,} if every day had 8 games and none were stored)")
        return 0
    key = require_env("ODDS_API_KEY")
    spent = 0; remaining = None; fetched = skipped = 0
    for d in days:
        events, c = day_events(key, d, a.events_utc); spent += c
        for e in on_day(events, d):
            if raw.latest("nba", "odds/props_hist", e["id"]) is not None:
                skipped += 1; continue
            if a.max_credits is not None and spent + per_game > a.max_credits:
                print(f"stopping: the next game would pass --max-credits {a.max_credits} ({spent} spent)"); derive_day(d, events); return _done(spent, fetched, skipped, remaining)
            if remaining is not None and remaining - per_game < a.reserve:
                print(f"stopping: {remaining} credits left, keeping the reserve of {a.reserve}"); derive_day(d, events); return _done(spent, fetched, skipped, remaining)
            at = pd.Timestamp(e["commence_time"]).to_pydatetime() - timedelta(minutes=a.before)
            data, ts, headers = get_historical_event_odds(SPORT_KEY, key, e["id"], _iso(at), markets=markets)
            spent += int(float(headers.get("x-requests-last") or 0)); remaining = credits_remaining(headers)[0] if credits_remaining(headers)[0] is not None else remaining
            if data is None:
                continue
            raw.store("nba", "odds/props_hist", e["id"], {"data": data, "timestamp": ts},
                      meta={"day": d.isoformat(), "at": _iso(at), "snapshot": ts, "markets": markets, "home": e.get("home_team"), "away": e.get("away_team")})
            fetched += 1
        rows = derive_day(d, events)
        print(f"{d}: {len(on_day(events, d))} games, {rows:,} prop rows stored | {spent} credits this run" + (f", {remaining} left" if remaining is not None else ""), flush=True)
    return _done(spent, fetched, skipped, remaining)


def _done(spent, fetched, skipped, remaining) -> int:
    print(f"done: {fetched} games fetched, {skipped} already stored, {spent} credits spent" + (f", {remaining} remaining" if remaining is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
