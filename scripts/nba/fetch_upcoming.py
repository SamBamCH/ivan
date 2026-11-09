#!/usr/bin/env python3
"""
scripts/nba/fetch_upcoming.py

Everything upcoming in one go: every classic slate the DraftKings lobby lists (today and tomorrow) that this machine does
not hold is loaded into its own directory (the desk's own path, ingest.run_ingest: the pool, Rotowire's starters, the
projection and the public projection; no props), then every contest of those slates is stored
(historical_contest_information) and the payout table of each one fetched (payouts). Free: no odds credits. The picker's
**Fetch upcoming slates** button runs this as a job.

    scripts/nba/fetch_upcoming.py                 # load the slates not held, then contests + payouts for every lobby slate
    scripts/nba/fetch_upcoming.py --no-load       # contests + payouts only
    scripts/nba/fetch_upcoming.py --no-payouts    # the slates and their contests, no payout tables
    scripts/nba/fetch_upcoming.py --refresh       # also refresh the slates already held

A payout table is one request per contest; a night of slates has a few hundred contests, so this takes a few minutes the
first time and seconds afterwards (only the contests without a table are fetched).
"""
from __future__ import annotations

import argparse
from datetime import datetime

from ivan.database.session import init_db
from ivan.sports.common.draftkings import fetch_all_contests
from ivan.sports.nba.data_fetch import contests as C
from ivan.sports.nba import slates as S
from ivan.sports.nba.data_fetch.slate import IngestError, classic_slates
from ivan.sports.nba.ingest import ensure_schema, run_ingest


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-load", action="store_true", help="do not load slates; contests and payouts only")
    ap.add_argument("--no-payouts", action="store_true", help="skip the payout tables")
    ap.add_argument("--refresh", action="store_true", help="also refresh the slates already held")
    ap.add_argument("--sims", type=int, default=4000)
    a = ap.parse_args()
    ensure_schema(init_db("nba"))
    slates = classic_slates()
    if not slates:
        log("the lobby lists no classic NBA slate"); return 1
    held = set(S.held())
    log(f"{len(slates)} classic slate(s) in the lobby: " + ", ".join(f"{s['id']} {s['start']}" + (f" [{s['title']}]" if s.get("title") else "") for s in slates))
    failed = 0
    if not a.no_load:
        for s in slates:
            dg = int(s["id"])
            if dg in held and not a.refresh:
                log(f"{dg}: held already"); continue
            sess = S.session(dg)
            try:
                with S.using(dg):
                    out = run_ingest(sess, dg, update_odds=False, starters=True, n_sims=a.sims, log=log, by="script")
                if s.get("title"):
                    S.write_meta(dg, title=s["title"])
                st = out.get("stored"); pr = out.get("projected")
                log(f"{dg}: loaded, {getattr(st, 'players', '?')} players, {getattr(pr, 'players', '?')} projected")
            except IngestError as e:
                failed += 1; log(f"{dg}: load failed: {e}")
            except Exception as e:   # noqa: BLE001 - the next slate still loads
                failed += 1; log(f"{dg}: load failed: {type(e).__name__}: {e}")
            finally:
                sess.close()
    # the contests and the payout tables, through any held slate's session (the tables live in the history database)
    have = S.held()
    if not have:
        log("no slate held: the contests need a slate's session; nothing stored"); return 1
    dg0 = have[0]
    sess = S.session(dg0)
    try:
        with S.using(dg0):
            lobby = fetch_all_contests("NBA", raw_sport="nba")
            dgs = [int(s["id"]) for s in slates]
            added = C.upsert_contests(sess, lobby, draft_groups=dgs)
            ids = [int(c) for (c,) in sess.query(C.HistoricalContest.contest_id).filter(C.HistoricalContest.draft_group_id.in_(dgs)).all()]
            log(f"contests: {added} new, {len(ids)} on the {len(dgs)} slate(s)")
            if not a.no_payouts:
                missing = C.contests_without_payouts(sess, contest_ids=ids)
                log(f"payouts: {len(missing)} contest(s) without a table" + ("; fetching (one request each)" if missing else ""))
                filled = C.fetch_payouts(sess, contest_ids=ids, log=log)
                log(f"payouts: {filled} table(s) stored")
    except Exception as e:   # noqa: BLE001
        failed += 1; log(f"contests/payouts failed: {type(e).__name__}: {e}")
    finally:
        sess.close()
    log("done" + (f" with {failed} failure(s)" if failed else ""))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
