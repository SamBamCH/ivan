#!/usr/bin/env python3
"""
Rebuild the slate archive for past NFL slates, so they can be replayed.

    python scripts/nfl/backfill_archive.py --since 2025-11-01 --until 2025-11-30 [--odds] [--all-slates] [--dry-run]

For every projection snapshot in the window (by default only slates with at
least one scraped contest, since those are the ones that settle with real
payouts) the capture is built from:
  players    the snapshot's pool (ids, slots, salaries, teams); the snapshot
             was taken after OUT players were dropped, so status is unknown
  game_lines the rows game_odds already holds for the slate's games
  odds       with --odds, the player props as they stood 30 minutes before
             lock, from the-odds-api historical endpoint (billed per market:
             about 10 credits x 23 markets per game; the script prints the
             credits remaining after each slate and stops at --min-credits)

Captures are stored with source="backfill" and replace an earlier backfill
of the same slate. Ingest captures (source="ingest") are never touched.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import text

from ivan.core import archive
from ivan.core.config import require_env
from ivan.database.session import get_session, init_db, read_sql
from ivan.sports.common.odds import (DEFAULT_PLAYER_MARKETS, credits_remaining, get_historical_event_odds,
                                     get_historical_events, parse_odds)
from ivan.core.utils import map_team_name_to_abbreviation


def utc_naive_to_local(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)


def local_naive_to_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


GAME_RX = re.compile(r"\(([A-Z]{2,3})\s*(?:@|vs\.?|v)\s*([A-Z]{2,3})\)", re.I)


def game_teams(name: str) -> tuple[str, str] | None:
    m = GAME_RX.search(name or "")
    return (m.group(1).upper(), m.group(2).upper()) if m else None


def snapshot_slates(session, since: str, until: str, only_scraped: bool) -> list[dict]:
    """
    One row per showdown slate (DraftKings draft group) with a scraped field in
    the window: its lock, game, contests and the snapshot rows of that game's
    captain and flex pool. A snapshot date can hold several slates (a main
    slate and a same-kickoff showdown), so the pool is cut to the game's
    teams and must carry both CPT and FLEX rows; classic slates are skipped
    because no classic field is stored.
    """
    contests = read_sql(session, text("""SELECT h.contest_id, h.draft_group_id, h.date, h.contest_type, h.name,
                                                (m.contest_id IS NOT NULL) AS scraped
                                         FROM historical_contest_information h
                                         LEFT JOIN showdown_contest_metadata m ON m.contest_id = h.contest_id
                                         WHERE h.date BETWEEN :a AND :b AND h.contest_type = 'Showdown Captain Mode'""")
                        .bindparams(a=since, b=until + " 23:59:59"))
    if contests.empty:
        return []
    contests["date"] = pd.to_datetime(contests["date"])
    snaps = read_sql(session, text("""SELECT id, snapshot_date, player_name AS name, team, opponent, position, roster_slot_id, salary
                                      FROM projection_snapshot WHERE snapshot_date BETWEEN :a AND :b AND roster_slot_id IN (511, 512)""")
                     .bindparams(a=(pd.Timestamp(since) - timedelta(days=1)).strftime("%Y-%m-%d"), b=until + " 23:59:59"))
    snaps["snapshot_date"] = pd.to_datetime(snaps["snapshot_date"])
    out = []
    for dg, grp in contests.groupby("draft_group_id"):
        n_scraped = int(grp.scraped.sum())
        if only_scraped and n_scraped == 0:
            continue
        lock_utc = grp.date.mode().iloc[0].to_pydatetime()
        teams = next((t for t in (game_teams(n) for n in grp.name) if t), None)
        if teams is None:
            continue
        near_utc = (snaps.snapshot_date - lock_utc).abs() <= timedelta(minutes=5)
        near_local = (snaps.snapshot_date - utc_naive_to_local(lock_utc)).abs() <= timedelta(minutes=5)
        pool = snaps[snaps.team.isin(teams) & (near_utc | near_local)]
        pool = pool.drop_duplicates(["id"])
        if len(pool) < 30 or not (pool.roster_slot_id == 511).any() or not (pool.roster_slot_id == 512).any():
            out.append({"draft_group_id": int(dg), "lock_utc": lock_utc, "lock_local": utc_naive_to_local(lock_utc), "mode": "showdown",
                        "teams": list(teams), "contests": len(grp), "scraped": n_scraped, "pool": pool, "skip": "no showdown snapshot for this game"})
            continue
        out.append({"draft_group_id": int(dg), "snapshot_date": pool.snapshot_date.iloc[0].to_pydatetime(), "lock_utc": lock_utc,
                    "lock_local": utc_naive_to_local(lock_utc), "mode": "showdown", "teams": list(teams), "contests": len(grp),
                    "scraped": n_scraped, "pool": pool, "skip": None})
    return sorted(out, key=lambda s: s["lock_utc"])


def game_line_rows(session, lock_utc: datetime, teams: list[str]) -> list[dict]:
    rows = read_sql(session, text("""SELECT game_id, sport_key, commence_time, home_team, away_team, home_abbr, away_abbr, bookmaker,
                                            market, last_update, outcome_name, side, point, price
                                     FROM game_odds WHERE commence_time BETWEEN :a AND :b""")
                    .bindparams(a=(lock_utc - timedelta(hours=2)).isoformat(sep=" "), b=(lock_utc + timedelta(hours=6)).isoformat(sep=" ")))
    rows = rows[rows.home_abbr.isin(teams) | rows.away_abbr.isin(teams)]
    return rows.to_dict(orient="records")


def historical_player_odds(api_key: str, lock_utc: datetime, teams: list[str], minutes_before: int = 30) -> tuple[list[dict], dict]:
    at = (lock_utc - timedelta(minutes=minutes_before)).replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    events, ts, headers = get_historical_events("americanfootball_nfl", api_key, at)
    info = {"at": at, "events_snapshot": ts, "events": 0, "rows": 0}
    if not events:
        return [], info
    wanted = []
    for ev in events:
        ha, aa = map_team_name_to_abbreviation(ev.get("home_team")), map_team_name_to_abbreviation(ev.get("away_team"))
        start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
        if (ha in teams or aa in teams) and abs((start - lock_utc).total_seconds()) <= 6 * 3600:
            wanted.append(ev)
    by_team = {}
    for ev in wanted:
        odds, ts2, headers = get_historical_event_odds("americanfootball_nfl", api_key, ev["id"], at, markets=DEFAULT_PLAYER_MARKETS)
        if odds:
            by_team[ev["id"]] = odds
    df = parse_odds(by_team) if by_team else pd.DataFrame()
    info.update(events=len(wanted), rows=int(len(df)), credits_remaining=credits_remaining(headers)[0])
    return (df.to_dict(orient="records") if len(df) else []), info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--odds", action="store_true", help="fetch historical player props (costs API credits)")
    ap.add_argument("--all-slates", action="store_true", help="also slates with no scraped contest")
    ap.add_argument("--min-credits", type=int, default=500, help="stop fetching odds when fewer credits remain")
    ap.add_argument("--minutes-before", type=int, default=30, help="odds as of this many minutes before lock")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db-url", default=None)
    a = ap.parse_args()

    engine = init_db("nfl", a.db_url)
    archive.ensure_tables(engine)
    session = get_session(engine)
    slates = snapshot_slates(session, a.since, a.until, only_scraped=not a.all_slates)
    print(f"{len(slates)} showdown slate(s) between {a.since} and {a.until}"
          + ("" if a.all_slates else " with at least one scraped contest"))
    api_key = require_env("ODDS_API_KEY") if a.odds and not a.dry_run else None
    for s in slates:
        key = str(s["draft_group_id"])
        if s.get("skip"):
            print(f"{s['lock_local']:%Y-%m-%d %H:%M} {s['mode']:8s} key {key:>10s}  {'@'.join(s['teams'])}: skipped, {s['skip']}")
            continue
        pool = s["pool"]
        players = [{"id": int(r.id), "roster_slot_id": int(r.roster_slot_id), "name": r.name, "team": r.team, "opponent": r.opponent,
                    "position": r.position, "gametime": s["lock_local"].isoformat(), "salary": int(r.salary), "status": None}
                   for r in pool.itertuples()]
        lines = game_line_rows(session, s["lock_utc"], s["teams"])
        line = (f"{s['lock_local']:%Y-%m-%d %H:%M} {s['mode']:8s} key {key:>10s}  {'@'.join(s['teams']):7s} {len(players):3d} players  "
                f"{len(lines):3d} line rows  {s['contests']} contests ({s['scraped']} scraped)")
        if a.dry_run:
            print(line)
            continue
        odds, info = [], {}
        if a.odds:
            odds, info = historical_player_odds(api_key, s["lock_utc"], s["teams"], a.minutes_before)
            line += f"  odds {len(odds)} rows from {info.get('events')} event(s) as of {info.get('at')}  credits left {info.get('credits_remaining')}"
        archive.delete_captures(session, "nfl", key, source="backfill")
        cap = archive.archive_slate(session, sport="nfl", slate_key=key, lock=s["lock_local"], mode=s["mode"], players=players,
                                    odds=odds, game_lines=lines, source="backfill",
                                    note=f"from projection_snapshot {s['snapshot_date']:%Y-%m-%d %H:%M}; odds {info}" if info else
                                    f"from projection_snapshot {s['snapshot_date']:%Y-%m-%d %H:%M}; no player odds")
        print(line + f"  -> capture #{cap.id}")
        left = info.get("credits_remaining")
        if a.odds and left is not None and left < a.min_credits:
            print(f"stopping: {left} credits remaining (< --min-credits {a.min_credits})")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
