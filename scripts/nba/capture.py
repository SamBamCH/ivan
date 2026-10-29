#!/usr/bin/env python3
"""
The autopilot tick. Runs every 15 minutes from a user timer (deploy/install_nba_timers.sh, ivan-nba-capture.timer)
and does nothing unless the autopilot switch is on (ivan.sports.nba.autopilot: scripts/nba/autopilot.py on|off|status,
or the desk's header). With the switch on, each tick:

  * stands down while someone is at the desk: an edit in the last 30 minutes not published since, or a running desk job
  * finds today's classic slates in the DK lobby and picks ONE: the slate the desk left loaded
    (data/nba/active_slate.json) while it has not locked, else the one that locks next
  * refreshes it the way the desk's Refresh does (ingest.run_ingest: the fetch outside the database, then the pool
    replaced, projected with the pins and grids on file and its public side refreshed in one transaction; the slate's
    edits filed and restored on a switch): DK pool + statuses + Rotowire every tick; the player props (about 12 odds-api
    credits per game) when the slate crosses an offset before lock (360, 180, 90, 45, 15 minutes), the game lines
    (about 3 credits) with the props and once at the first tick of the day; the slate's contests and payouts (free)
  * publishes the live feed bundle and pushes it to the consumer (IVAN_FEED_PUSH) with a producer note, so the other
    desk's header can say who published and when
  * records what it did (or why it stood down) in data/nba/autopilot.json for both desks' headers

State (which props offsets are done per slate, the props fetches of the day) lives in data/nba/capture_state.json.
Until 2026-10-27 the tick captured every slate of the day in turn and wrote the pool table by table; the archive timer
(archive_at_lock.py) records the pool and projections at every tip now, and per-slate pins made a multi-slate rewrite
unsafe, so a tick is one slate through the desk's own path. --force runs with the switch off and the guard ignored.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta

from ivan.database.session import get_session, init_db
from ivan.sports.common.draftkings import fetch_all_contests, summarize_slates
from ivan.sports.nba import autopilot
from ivan.sports.nba import contests as C
from ivan.sports.nba.ingest import IngestError, ensure_schema, fetch_game_lines, fetch_slate, run_ingest
from ivan.sports.nba.paths import ACTIVE_SLATE_JSON, NBA_DATA_DIR, LOGS_DIR


def chosen_slate() -> int | None:
    """The draft group picked on the desk (ivan.web.nba_desk.switch_slate), or None."""
    try:
        return int(json.loads(ACTIVE_SLATE_JSON.read_text())["draft_group"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


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


def stand_down(reason: str, *, ok: bool = True, code: int = 0) -> int:
    log.info(reason)
    autopilot.record_tick(reason, ok=ok, detail=dict(skipped=True))
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offsets", default=",".join(map(str, DEFAULT_OFFSETS)), help="props offsets in minutes before lock")
    ap.add_argument("--max-props-per-day", type=int, default=20, help="safety cap on props fetches per day (each costs ~12 credits per game)")
    ap.add_argument("--no-props", action="store_true", help="never fetch props or game lines (free tick)")
    ap.add_argument("--no-contests", action="store_true", help="skip the contest + payout store")
    ap.add_argument("--no-feed", action="store_true", help="skip publishing and pushing the live feed bundle")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--grace", type=int, default=5, help="minutes after lock during which a slate is still refreshed")
    ap.add_argument("--only", type=int, default=None, help="draft group id: this slate, whatever the desk picked")
    ap.add_argument("--force", action="store_true", help="run with the autopilot switch off and the desk guard ignored (a test)")
    args = ap.parse_args()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    offsets = sorted({int(x) for x in args.offsets.split(",") if x.strip()}, reverse=True)

    if not args.force and not autopilot.is_on():
        log.info("autopilot off; nothing to do")                 # not recorded: the header says "off" on its own
        return 0
    hold = None if args.force else autopilot.desk_hold()
    if hold:
        return stand_down(f"standing down: {hold}")

    engine = init_db("nba"); ensure_schema(engine); session = get_session(engine)
    st = load_state()
    now = datetime.now()
    try:
        slates, lobby = todays_classic_slates()
    except Exception as e:   # noqa: BLE001
        return stand_down(f"DK lobby fetch failed: {type(e).__name__}: {e}", ok=False, code=1)
    if args.only:
        slates = [s for s in slates if s["id"] == args.only]
    if not slates:
        save_state(st)
        return stand_down("no classic NBA slates in the lobby; nothing to do")

    # pass 1: every slate's pool (free) to learn the locks; nothing stored
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
        save_state(st)
        return stand_down("every slate has locked; nothing to do")

    # the one slate of this tick: the desk's while it is open, else the one that locks next
    chosen = chosen_slate()
    pick = next(((s, f) for s, f in fetched if chosen is not None and int(s["id"]) == chosen), None)
    if pick is None:
        if chosen is not None:
            log.info(f"the desk's slate {chosen} has locked or left the lobby; the slate that locks next")
            ACTIVE_SLATE_JSON.unlink(missing_ok=True)
        pick = min(fetched, key=lambda sf: sf[1].lock)
    s, f = pick
    mins = (f.lock - now).total_seconds() / 60
    slate_st = st["slates"].setdefault(str(s["id"]), {})
    done = set(slate_st.setdefault("props_done", []))
    due = []
    if not args.no_props and st["props_fetches"] < args.max_props_per_day:
        due = [o for o in offsets if mins <= o and o not in done]
    log.info(f"{s['id']} {s['name']}: {mins:.0f} min to lock" + (f", props offsets due {due}" if due else ", props kept"))

    # the real fetch: statuses + Rotowire, the props when due (a props failure costs nothing but the props)
    try:
        full = fetch_slate(session, s["id"], update_odds=bool(due), starters=True, log=log.info, lines=False)   # lines below
    except IngestError as e:
        if not due:
            return stand_down(f"{s['id']}: {e}", ok=False, code=1)
        log.warning(f"props fetch failed ({e}); this tick refreshes the pool without them")
        due = []
        try:
            full = fetch_slate(session, s["id"], update_odds=False, starters=True, log=log.info, lines=False)
        except IngestError as e2:
            return stand_down(f"{s['id']}: {e2}", ok=False, code=1)
    if due:
        slate_st["props_done"] = sorted(done | set(due), reverse=True)
        st["props_fetches"] += 1
    # game lines: with every props fetch (the line moves with the news) and once at the first tick of the day
    if not args.no_props and (due or not st.get("lines_fetched")):
        try:
            lines = fetch_game_lines(log=log.info)
            if lines is not None:
                full.game_lines = lines
                st["lines_fetched"] = str(now)
        except Exception as e:   # noqa: BLE001
            log.warning(f"game lines fetch failed: {type(e).__name__}: {e}")

    # the desk's own path: replace, project and refresh the public side in one transaction
    try:
        out = run_ingest(session, s["id"], update_odds=False, starters=True, n_sims=args.sims, log=log.info, fetched=full)
    except Exception as e:   # noqa: BLE001
        save_state(st)
        return stand_down(f"{s['id']}: refresh failed: {type(e).__name__}: {e}", ok=False, code=1)
    proj = out.get("projected")
    log.info(f"{s['id']} {s['name']}: lock {full.lock:%a %H:%M}, {len(full.players)} players, {getattr(proj, 'players', '?')} projected")

    if not args.no_contests:   # the slate's contests and the payout table of any new one (free; one request per new contest)
        try:
            added = C.upsert_contests(session, lobby, draft_groups=[s["id"]])
            ids = [c for (c,) in session.query(C.HistoricalContest.contest_id).filter(C.HistoricalContest.draft_group_id == s["id"]).all()]
            filled = C.fetch_payouts(session, contest_ids=ids, log=log.info)
            log.info(f"contests: {added} new, payouts: {filled} fetched")
        except Exception as e:   # noqa: BLE001 - never at the cost of the refresh
            log.warning(f"contests/payouts failed: {type(e).__name__}: {e}")

    slate_st.update(lock=str(full.lock), last_tick=str(now), name=s["name"])
    save_state(st)
    result = f"{str(s['name']).replace('NBA ', '')[:40]}: {len(full.players)} players refreshed, props {'fetched' if due else 'kept'}"
    detail = dict(draft_group=s["id"], props=bool(due), lock=str(full.lock), pushed=None)
    autopilot.record_tick(result, ok=True, detail=detail)            # before the publish, so the manifest carries this tick
    if not args.no_feed:
        pushed = False
        try:
            from ivan.core import feed
            feed.publish("nba", which="live", log=log.info, producer=autopilot.producer_info("autopilot"))
            pushed = feed.push("nba", log=log.info)                  # to the consumer when IVAN_FEED_PUSH is set
        except Exception as e:   # noqa: BLE001 - the feed is a convenience for the other machine
            log.warning(f"feed publish failed: {type(e).__name__}: {e}")
        autopilot.record_tick(result + (", pushed" if pushed else ", not pushed"), ok=True, detail=dict(detail, pushed=pushed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
