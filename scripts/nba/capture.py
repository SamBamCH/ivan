#!/usr/bin/env python3
"""
The autopilot tick. Runs every 15 minutes from a user timer (deploy/install_nba_timers.sh, ivan-nba-capture.timer)
and does nothing unless the autopilot switch is on (ivan.sports.nba.slate.autopilot: scripts/nba/autopilot.py on|off|status,
or the desk's header). With the switch on, each tick:

  * stands down while a desk job is running (an edit counts as published the moment it is written, so edits never hold it)
  * refreshes EVERY slate this machine holds that has not locked (data/nba/slates/<dg>/, ivan.sports.nba.slates: the
    slates opened on the desk or by scripts/nba/ingest.py), each in its own directory and database, the way the desk's
    Refresh does (ingest.run_ingest: the fetch outside the database, then the slate's pool replaced, projected with its
    pins and grids and its public side refreshed in one transaction): DK pool + statuses + Rotowire every tick; the
    player props (about 12 odds-api credits per game) when a slate crosses an offset before lock (360, 180, 90, 45, 15
    minutes), the game lines (about 3 credits) with the props and once at the first tick of the day; the slate's
    contests and payouts (free). A slate nobody opened is not refreshed: opening it on the desk (or ingest.py) starts it.
  * records what it did (or why it stood down) in data/nba/autopilot.json for both desks' headers; the other desk pulls
    the pool itself (its "Refresh from laptop", feed.pull_ssh), nothing is pushed from here

State (which props offsets are done per slate, the props fetches of the day) lives in data/nba/capture_state.json.
The archive timer (archive_at_lock.py) records every held slate's pool and projections at each tip. --force runs with
the switch off and the guard ignored; --only DG refreshes one slate.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta

from ivan.database.session import init_db
from ivan.sports.common.draftkings import fetch_all_contests, summarize_slates
from ivan.sports.nba.slate import autopilot
from ivan.sports.nba.data_fetch import contests as C
from ivan.sports.nba.data_fetch.slate import IngestError, fetch_game_lines, fetch_slate
from ivan.sports.nba.ingest import ensure_schema, run_ingest
from ivan.sports.nba import slates as S
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
    ap.add_argument("--worlds", type=int, default=None, help="worlds per game (default config projection.n_worlds)")
    ap.add_argument("--grace", type=int, default=5, help="minutes after lock during which a slate is still refreshed")
    ap.add_argument("--only", type=int, default=None, help="draft group id: this held slate only")
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

    ensure_schema(init_db("nba"))
    st = load_state()
    now = datetime.now()
    try:
        slates, lobby = todays_classic_slates()
    except Exception as e:   # noqa: BLE001
        return stand_down(f"DK lobby fetch failed: {type(e).__name__}: {e}", ok=False, code=1)
    by_id = {int(s["id"]): s for s in slates}
    held = S.open_slates(now, grace_minutes=args.grace)
    if args.only:
        held = [d for d in held if d == args.only] or [args.only]
    if not held:
        save_state(st)
        return stand_down("no open slate held on this machine; open one on the desk (or scripts/nba/ingest.py --draft-group) to start it")

    results = []; failed = 0
    for dg in held:
        s = by_id.get(dg) or dict(id=dg, name=S.read_meta(dg).get("label") or f"slate {dg}")
        session = S.session(dg)
        try:
            r = tick_slate(session, s, st, now, args, offsets, lobby if dg in by_id else None)
            if r is not None:
                results.append(r)
        except Exception as e:   # noqa: BLE001
            failed += 1; log.warning(f"{dg}: refresh failed: {type(e).__name__}: {e}")
        finally:
            session.close()
    save_state(st)
    if not results and failed:
        return stand_down("every refresh failed (see the log)", ok=False, code=1)
    if not results:
        return stand_down("every held slate has locked; nothing to do")
    autopilot.record_tick("; ".join(r["result"] for r in results) + (f"; {failed} failed" if failed else ""), ok=not failed,
                          detail=dict(slates=[r["detail"] for r in results]))       # the other desk reads this note when it pulls (feed.publish, on its request)
    return 0 if not failed else 1


def tick_slate(session, s: dict, st: dict, now: datetime, args, offsets: list, lobby) -> dict | None:
    """One held slate through the desk's own path: the free fetch first (statuses + Rotowire; the props when an offset is due),
    then replace, project and refresh the public side in one transaction (run_ingest), then the contests. None when the slate
    has locked (its lock from the fetch); raises when the refresh failed."""
    dg = int(s["id"])
    with S.using(dg):
        probe = fetch_slate(session, dg, update_odds=False, starters=False, log=log.info)
        if probe.lock is None or probe.lock < now - timedelta(minutes=args.grace):
            log.info(f"{dg} {s['name']}: locked {probe.lock}; skipped")
            S.write_meta(dg, lock=probe.lock) if probe.lock else None
            return None
        mins = (probe.lock - now).total_seconds() / 60
        slate_st = st["slates"].setdefault(str(dg), {})
        done = set(slate_st.setdefault("props_done", []))
        due = []
        if not args.no_props and st["props_fetches"] < args.max_props_per_day:
            due = [o for o in offsets if mins <= o and o not in done]
        log.info(f"{dg} {s['name']}: {mins:.0f} min to lock" + (f", props offsets due {due}" if due else ", props kept"))
        # the real fetch: statuses + Rotowire, the props when due (a props failure costs nothing but the props)
        try:
            full = fetch_slate(session, dg, update_odds=bool(due), starters=True, log=log.info, lines=False)   # lines below
        except IngestError as e:
            if not due:
                raise
            log.warning(f"props fetch failed ({e}); this tick refreshes the pool without them")
            due = []
            full = fetch_slate(session, dg, update_odds=False, starters=True, log=log.info, lines=False)
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
        out = run_ingest(session, dg, update_odds=False, starters=True, n_worlds=args.worlds, log=log.info, fetched=full, by="autopilot")
        proj = out.get("projected")
        log.info(f"{dg} {s['name']}: lock {full.lock:%a %H:%M}, {len(full.players)} players, {getattr(proj, 'players', '?')} projected")
        if not args.no_contests and lobby is not None:   # the slate's contests and the payout table of any new one (free; one request per new contest)
            try:
                added = C.upsert_contests(session, lobby, draft_groups=[dg])
                ids = [c for (c,) in session.query(C.HistoricalContest.contest_id).filter(C.HistoricalContest.draft_group_id == dg).all()]
                filled = C.fetch_payouts(session, contest_ids=ids, log=log.info)
                log.info(f"contests: {added} new, payouts: {filled} fetched")
            except Exception as e:   # noqa: BLE001 - never at the cost of the refresh
                log.warning(f"contests/payouts failed: {type(e).__name__}: {e}")
        slate_st.update(lock=str(full.lock), last_tick=str(now), name=s["name"])
        return dict(result=f"{str(s['name']).replace('NBA ', '')[:40]}: {len(full.players)} players refreshed, props {'fetched' if due else 'kept'}",
                    detail=dict(draft_group=dg, props=bool(due), lock=str(full.lock)))


if __name__ == "__main__":
    raise SystemExit(main())
