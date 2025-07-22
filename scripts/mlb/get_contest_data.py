#!/usr/bin/env python3
"""
scripts/mlb/get_contest_data.py

Orchestrates fetching all historical contests from the DB,
downloading each contest’s standings CSV, parsing it, and
upserting payouts, actual‐entries, and your slate‐KDE model.
"""
import logging
from datetime import datetime, date

from http.client import RemoteDisconnected
import zipfile
import time

from ivan.database.session    import init_db, get_session
from ivan.sports.common.draftkings import build_requests_session, get_contest_dataframe
from ivan.sports.mlb.historical_contests import (
    parse_df,
    upsert_payout_changes,
    upsert_actuals,
    fit_slate_kde,
)
from ivan.sports.mlb.models  import HistoricalContest, SlateKDEModel

profile_dir = r"C:\selenium\chrome-profile"

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    # —1) init DB & session
    engine  = init_db("mlb")
    session = get_session(engine)
    sess    = build_requests_session(profile_dir)

    # —2) load every contest we’ve ingested metadata for
    all_contests = session.query(HistoricalContest).all()
    log.info("Found %d historical contests in DB", len(all_contests))

    # —3) see which dates we’ve already modeled
    existing_dates = {
        dt.date()
        for (dt,) in session.query(SlateKDEModel.contest_date).distinct()
    }
    log.info("Already have models for %d slate dates", len(existing_dates))

    # —4) filter to only contests not yet modeled
    to_process = [
        hc for hc in all_contests
        if hc.date.date() not in existing_dates
    ]
    log.info("Need to process %d new contests", len(to_process))

    MAX_NONAUTH_RETRIES = 3

    # —5) loop, skipping future contests
    for hc in to_process:
        cid   = hc.contest_id
        sched = hc.date

        # ←— FUTURE CONTEST? SKIP & CONTINUE —→
        if sched > datetime.now():
            log.info("Skipping future contest %s scheduled for %s", cid, sched)
            continue

        log.info("→ Downloading contest %s on %s", cid, sched)
        nonauth_tries = 0

        # retry loop
        while True:
            try:
                df = get_contest_dataframe(sess, cid)
                break
            except (RemoteDisconnected, zipfile.BadZipFile) as e:
                log.warning(
                    "Auth/ZIP error for %s: %s — rebuilding session",
                    cid, e
                )
                sess = build_requests_session(profile_dir)
                time.sleep(1)
                nonauth_tries = 0
                continue
            except Exception as e:
                nonauth_tries += 1
                if nonauth_tries >= MAX_NONAUTH_RETRIES:
                    log.error(
                        "Failed to download %s after %d retries: %s",
                        cid, nonauth_tries, e
                    )
                    df = None
                    break
                else:
                    log.warning(
                        "Transient error on %s (try %d/%d): %s — retrying…",
                        cid, nonauth_tries, MAX_NONAUTH_RETRIES, e
                    )
                    time.sleep(1)
                    continue

        if df is None:
            continue  # give up on this contest

        # parse & upsert
        lineups, payouts, actuals = parse_df(df, cid)
        try:
            # upsert_payout_changes(session, payouts)
            # upsert_actuals(session, actuals)
            fit_slate_kde(session, hc, lineups)
            session.commit()
            log.info("✔ Modeled contest %s", cid)
        except Exception:
            session.rollback()
            log.exception("✗ Error modeling contest %s", cid)

    log.info("All done.")

if __name__ == "__main__":
    main()
