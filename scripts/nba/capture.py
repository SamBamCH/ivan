#!/usr/bin/env python3
"""
Game-day capture loop. Run every 15 minutes from a user timer (deploy/install_nba_timers.sh);
exits at once when nothing is upcoming.

Each tick, for every classic draft group in today's DK lobby that has not locked:
  * DraftKings draftables (free) -> raw dk/draftables, nba_player, archive capture
  * Rotowire lineups page (free) -> raw rotowire/lineups, starting_lineups
  * game lines (moneyline, spread, total; ~3 credits) at the first tick of the day and
    whenever props are fetched -> raw odds/lines, game_odds, archive capture
  * player props at fixed offsets before the slate's lock (costs odds-api credits:
    about 12 per game) -> raw odds/*, odds, nba_market_rate. One fetch per tick
    covers the teams of every slate that crossed an offset, and the rows are
    stored for every slate captured in that tick.
  * a projection -> nba_projection_capture rows (ours + DK's FPPG) and a CSV artifact
  * the slate's contests (from the lobby payload, free) -> historical_contest_information,
    and the payout table of any contest that lacks one -> payouts (ivan.sports.nba.contests)
Slates are processed latest lock first, so nba_player ends the tick holding the
slate that locks next, which is the one the news desk manages.

State (which props offsets are done per slate) lives in data/nba/capture_state.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from ivan.database.session import get_session, init_db
from ivan.sports.common.draftkings import fetch_all_contests, summarize_slates
from ivan.sports.common.odds import get_odds, parse_odds
from ivan.sports.nba import contests as C
from ivan.core.config import require_env
from ivan.sports.nba.ingest import (IngestError, ODDS_SPORT, ensure_schema, fetch_game_lines, fetch_slate, project_slate,
                                    store_slate)
from ivan.sports.nba.projection.market import NBA_MARKETS
from ivan.sports.nba.paths import NBA_DATA_DIR, LOGS_DIR

STATE = NBA_DATA_DIR / "capture_state.json"
DEFAULT_OFFSETS = (360, 180, 90, 45, 15)      # minutes before lock at which props are captured
log = logging.getLogger("capture")


def load_state() -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    st = {}
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text())
        except Exception:
            st = {}
    if st.get("date") != today:
        st = {"date": today, "slates": {}, "props_fetches": 0}
    return st


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, default=str))


def todays_classic_slates() -> tuple[list[dict], list[dict]]:
    """(classic slates in the lobby, the lobby's contest dicts)."""
    contests = fetch_all_contests("NBA", raw_sport="nba")
    out = []
    for dg, info in summarize_slates(contests).items():
        if str(info.get("gameType", "")).lower() == "classic":
            out.append({"id": int(dg), "name": info.get("name"), "contests": info.get("count")})
    return out, contests


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offsets", default=",".join(map(str, DEFAULT_OFFSETS)), help="props offsets in minutes before lock")
    ap.add_argument("--max-props-per-day", type=int, default=20, help="safety cap on props fetches per day (each costs ~12 credits per game)")
    ap.add_argument("--no-props", action="store_true")
    ap.add_argument("--no-contests", action="store_true", help="skip the contest + payout store")
    ap.add_argument("--no-feed", action="store_true", help="skip publishing the live feed bundle")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--grace", type=int, default=5, help="minutes after lock during which a slate is still captured")
    ap.add_argument("--only", type=int, default=None, help="draft group id: capture this slate only")
    args = ap.parse_args()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    offsets = sorted({int(x) for x in args.offsets.split(",") if x.strip()}, reverse=True)

    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    st = load_state()
    now = datetime.now()
    slates, lobby = todays_classic_slates()
    if args.only:
        slates = [s for s in slates if s["id"] == args.only]
    if not slates:
        log.info("no classic NBA slates in the lobby; nothing to do")
        return 0

    # pass 1: fetch every slate's pool (free) to learn its lock
    fetched = []
    for s in slates:
        try:
            f = fetch_slate(session, s["id"], update_odds=False, starters=False, log=log.info)
        except IngestError as e:
            log.warning(f"{s['id']}: {e}"); continue
        if f.lock is None or f.lock < now - timedelta(minutes=args.grace):
            log.info(f"{s['id']} {s['name']}: locked {f.lock}; skipped")
            continue
        fetched.append((s, f))
    if not fetched:
        log.info("every slate has locked; nothing to do")
        save_state(st); return 0
    # contests + payouts for the unlocked slates (free; the payout fetch is one request per new contest)
    if not args.no_contests:
        try:
            dgs = [s["id"] for s, _ in fetched]
            added = C.upsert_contests(session, lobby, draft_groups=dgs)
            ids = [c for (c,) in session.query(C.HistoricalContest.contest_id).filter(C.HistoricalContest.draft_group_id.in_(dgs)).all()]
            filled = C.fetch_payouts(session, contest_ids=ids, log=log.info)
            log.info(f"contests: {added} new, payouts: {filled} fetched")
        except Exception as e:   # noqa: BLE001 - never at the cost of the captures
            log.warning(f"contests/payouts failed: {type(e).__name__}: {e}")

    # props: one fetch per tick for the teams of every slate that crossed an undone offset
    props_teams: set = set()
    if not args.no_props and st["props_fetches"] < args.max_props_per_day:
        for s, f in fetched:
            mins = (f.lock - now).total_seconds() / 60
            done = set(st["slates"].setdefault(str(s["id"]), {}).setdefault("props_done", []))
            due = [o for o in offsets if mins <= o and o not in done]
            if due:
                props_teams |= set(f.teams)
                st["slates"][str(s["id"])]["props_done"] = sorted(done | set(due), reverse=True)
                log.info(f"{s['id']}: {mins:.0f} min to lock, props offsets due {due}")
    props_df = None
    if props_teams:
        try:
            odds = get_odds(ODDS_SPORT, require_env("ODDS_API_KEY"), sorted(props_teams), markets=NBA_MARKETS, raw_sport="nba")
            df = parse_odds(odds)
            if df is not None and len(df):
                props_df = df
                st["props_fetches"] += 1
                log.info(f"props: {len(df)} rows, {df.player.nunique()} players, {df.sportsbook.nunique()} books, "
                         f"{len(props_teams)} teams")
            else:
                log.warning("the-odds-api returned no NBA props; check ODDS_API_KEY and the plan")
        except Exception as e:   # noqa: BLE001 - a props failure must not cost the draftables capture
            log.warning(f"props fetch failed: {type(e).__name__}: {e}")
    # game lines: once a day at the first tick, then with every props fetch (the line moves with the news)
    lines_df = None
    if not args.no_props and (props_teams or not st.get("lines_fetched")):
        try:
            lines_df = fetch_game_lines(log=log.info)
            if lines_df is not None:
                st["lines_fetched"] = str(now)
        except Exception as e:   # noqa: BLE001
            log.warning(f"game lines fetch failed: {type(e).__name__}: {e}")
    # Rotowire once per tick: ride it on the first slate processed; props rows go to every slate of the tick
    order = sorted(fetched, key=lambda sf: sf[1].lock, reverse=True)
    for i, (s, f) in enumerate(order):
        try:
            full = fetch_slate(session, s["id"], update_odds=False, starters=(i == 0), log=log.info)
        except IngestError as e:
            log.warning(f"{s['id']}: {e}"); continue
        if props_df is not None:
            pool = {p["name"] for p in full.players}
            full.odds = props_df[props_df.player.isin(pool)].reset_index(drop=True)
        if lines_df is not None:
            full.game_lines = lines_df
        store_slate(session, full, log=log.info)
        project_slate(session, n_sims=args.sims, log=log.info)
        st["slates"].setdefault(str(s["id"]), {}).update(lock=str(full.lock), last_tick=str(now), name=s["name"])
        log.info(f"{s['id']} {s['name']}: lock {full.lock:%a %H:%M}, {len(full.players)} players captured")
    save_state(st)
    if not args.no_feed:
        try:
            from ivan.core import feed
            feed.publish("nba", which="live", log=log.info)
            feed.push("nba", log=log.info)                  # to the server when IVAN_FEED_PUSH is set
        except Exception as e:   # noqa: BLE001 - the feed is a convenience for the other machine
            log.warning(f"feed publish failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
