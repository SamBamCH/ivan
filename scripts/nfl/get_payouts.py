#!/usr/bin/env python
import sys
import logging
import requests

from sqlalchemy import distinct
from sqlalchemy.exc import SQLAlchemyError

from ivan.database.session import init_db, get_session
from ivan.sports.mlb.models import HistoricalContest, ContestPayout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DK_CONTESTDETAIL_URL = "https://api.draftkings.com/contests/v1/contests/{cid}"

def upsert_payouts_for_contest(db_session, contest_id, payload):
    """
    Given the JSON from DraftKings’ “contest detail” endpoint,
    parses contestDetail.payoutSummary and inserts rows into payouts.
    """
    detail  = payload.get("contestDetail", {})
    summary = detail.get("payoutSummary", []) or []

    rows_to_insert = []
    for tier in summary:
        min_pos = tier.get("minPosition")
        max_pos = tier.get("maxPosition")

        pd_list = tier.get("payoutDescriptions", [])
        if not pd_list:
            continue

        first_pd = pd_list[0]
        val       = first_pd.get("value", 0.0)
        text_form = first_pd.get("payoutDescription", "")
        currency  = "USD"

        rows_to_insert.append(
            ContestPayout(
                contest_id    = contest_id,
                min_position  = min_pos,
                max_position  = max_pos,
                payout_amount = float(val),
                payout_text   = text_form,
                currency      = currency,
            )
        )

    if not rows_to_insert:
        logger.info(f" → No payout tiers found for contest {contest_id}. Skipping.")
        return

    try:
        db_session.bulk_save_objects(rows_to_insert)
        db_session.commit()
        logger.info(f" ✓ Inserted {len(rows_to_insert)} payout tiers for {contest_id}")
    except SQLAlchemyError as e:
        db_session.rollback()
        logger.error(f" ✗ DB error inserting payouts for {contest_id}: {e}")


def fetch_and_store_all_payouts():
    # 1) Init DB + session
    engine = init_db("nfl")
    db     = get_session(engine)

    # 2) Query all contest_ids and all payout contest_ids
    all_contests = db.query(HistoricalContest.contest_id).all()
    all_contests = {row.contest_id for row in all_contests}

    existing = db.query(distinct(ContestPayout.contest_id)).all()
    existing = {row[0] for row in existing if row[0] is not None}

    missing = sorted(all_contests - existing)
    logger.info(f"Found {len(all_contests)} contests total, {len(existing)} already have payouts → {len(missing)} to fetch.")

    # 3) Loop over missing contest_ids
    for idx, cid in enumerate(missing, start=1):
        logger.info(f"[{idx}/{len(missing)}] Fetching contest detail for {cid}…")
        url = DK_CONTESTDETAIL_URL.format(cid=cid)
        try:
            resp = requests.get(url, params={"format": "json"})
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f" ✗ HTTP error for contest {cid}: {e}")
            continue

        payload = resp.json()
        upsert_payouts_for_contest(db, cid, payload)

    logger.info("Done.")

if __name__ == "__main__":
    fetch_and_store_all_payouts()
