# src/ivan/sports/mma/contests.py
"""Upcoming DraftKings MMA contests and their payout tables, into the DK tables (same as the other sports)."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from sqlalchemy import distinct
from sqlalchemy.orm import Session

from ivan.sports.common.draftkings import get_upcoming_contest_ids
from ivan.sports.mma.models import ContestPayout, HistoricalContest

log = logging.getLogger(__name__)
DK_CONTESTDETAIL_URL = "https://api.draftkings.com/contests/v1/contests/{cid}"


def _parse_sd(sd_str: str) -> Optional[datetime]:
    m = re.search(r"/Date\((\d+)\)/", sd_str or "")
    return datetime.fromtimestamp(int(m.group(1)) / 1000.0, tz=timezone.utc) if m else None


def fetch_contests(session: Session, days_ahead: int = 2, entry_filter: int = 100) -> dict:
    """Upsert upcoming guaranteed MMA contests. Returns counts."""
    found = get_upcoming_contest_ids(sport="MMA", days_ahead=days_ahead, entry_filter=entry_filter)
    added = 0
    for c in found:
        cid = int(c["id"])
        if session.query(HistoricalContest).filter_by(contest_id=cid).one_or_none():
            continue
        session.add(HistoricalContest(contest_id=cid, draft_group_id=c.get("dg"), name=c.get("n"),
                                      date=_parse_sd(c.get("sd")) or datetime.utcnow(), entry_fee=float(c.get("a", 0)),
                                      contest_type=c.get("gameType"), total_entries=int(c.get("m", 0)), multi_entry=c.get("mec")))
        added += 1
    session.commit()
    return {"upcoming": len(found), "new": added}


def _payout_rows(contest_id: int, payload: dict) -> list[ContestPayout]:
    rows = []
    for tier in (payload.get("contestDetail", {}) or {}).get("payoutSummary", []) or []:
        pd_list = tier.get("payoutDescriptions", [])
        if not pd_list:
            continue
        rows.append(ContestPayout(contest_id=contest_id, min_position=tier.get("minPosition"), max_position=tier.get("maxPosition"),
                                  payout_amount=float(pd_list[0].get("value", 0.0)), payout_text=pd_list[0].get("payoutDescription", ""),
                                  currency="USD"))
    return rows


def fetch_payouts(session: Session, days: Optional[int] = 2, draft_group_id: Optional[int] = None,
                  contest_ids: Optional[list[int]] = None) -> dict:
    """Payout tables for contests that lack one; recent/upcoming only unless days is None."""
    q = session.query(HistoricalContest.contest_id)
    if days is not None:
        q = q.filter(HistoricalContest.date >= datetime.now() - timedelta(days=days))
    if draft_group_id is not None:
        q = q.filter(HistoricalContest.draft_group_id == int(draft_group_id))
    if contest_ids is not None:
        q = q.filter(HistoricalContest.contest_id.in_([int(c) for c in contest_ids]))
    have = {r[0] for r in session.query(distinct(ContestPayout.contest_id)).all() if r[0] is not None}
    missing = sorted({r.contest_id for r in q.all()} - have)
    done = failed = 0
    for cid in missing:
        try:
            resp = requests.get(DK_CONTESTDETAIL_URL.format(cid=cid), params={"format": "json"}, timeout=15)
            resp.raise_for_status()
            rows = _payout_rows(cid, resp.json())
        except (requests.RequestException, ValueError) as e:
            log.warning("payouts for contest %s: %s", cid, e)
            failed += 1
            continue
        if rows:
            session.bulk_save_objects(rows)
            session.commit()
            done += 1
    return {"missing": len(missing), "fetched": done, "failed": failed}


def _contest_from_detail(cid: int, payload: dict) -> Optional[HistoricalContest]:
    """A contest row from the contest-detail endpoint, for a contest the lobby listing left out."""
    d = (payload or {}).get("contestDetail") or {}
    if not d.get("draftGroupId"):
        return None
    start = d.get("contestStartTime")
    try:
        date = datetime.fromisoformat(str(start).replace("Z", "+00:00")) if start else None
    except ValueError:
        date = None
    return HistoricalContest(contest_id=cid, draft_group_id=int(d["draftGroupId"]), name=d.get("name"),
                             date=date or datetime.utcnow(), entry_fee=float(d.get("entryFee") or 0.0),
                             contest_type=d.get("gameType") or d.get("contestTypeDetail"),
                             total_entries=int(d.get("maxEntries") or 0), multi_entry=int(d.get("maxEntriesPerUser") or 1))


def ensure_contests(session: Session, contest_ids) -> dict:
    """
    Make sure every contest id (from the entries file) has a row and a payout
    table: the lobby listing first, then the contest-detail endpoint for any
    the listing left out. Returns the ids still without a payout table.
    """
    ids = sorted({int(c) for c in contest_ids})
    have = {r[0] for r in session.query(HistoricalContest.contest_id).filter(HistoricalContest.contest_id.in_(ids)).all()}
    if set(ids) - have:
        try:
            fetch_contests(session)
        except (requests.RequestException, ValueError, KeyError) as e:
            log.warning("contest listing: %s", e)
        have = {r[0] for r in session.query(HistoricalContest.contest_id).filter(HistoricalContest.contest_id.in_(ids)).all()}
    for cid in sorted(set(ids) - have):
        try:
            resp = requests.get(DK_CONTESTDETAIL_URL.format(cid=cid), params={"format": "json"}, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as e:
            log.warning("contest %s: %s", cid, e)
            continue
        row = _contest_from_detail(cid, payload)
        if row is None:
            log.warning("contest %s: no detail returned", cid)
            continue
        session.add(row)
        session.commit()
        pay = _payout_rows(cid, payload)
        if pay:
            session.bulk_save_objects(pay)
            session.commit()
    fetch_payouts(session, days=None, contest_ids=ids)
    have_pay = {r[0] for r in session.query(distinct(ContestPayout.contest_id)).filter(ContestPayout.contest_id.in_(ids)).all()}
    return {"missing": sorted(set(ids) - have_pay)}


def contests_for_slate(session: Session, draft_group_id: int) -> list[dict]:
    """The contests on one slate with whether a payout table is stored, biggest first."""
    have = {r[0] for r in session.query(distinct(ContestPayout.contest_id)).all()}
    out = []
    for c in (session.query(HistoricalContest).filter(HistoricalContest.draft_group_id == int(draft_group_id))
              .order_by(HistoricalContest.total_entries.desc()).all()):
        out.append({"id": c.contest_id, "name": c.name, "entries": c.total_entries, "fee": c.entry_fee,
                    "type": c.contest_type, "payouts": c.contest_id in have})
    return out
