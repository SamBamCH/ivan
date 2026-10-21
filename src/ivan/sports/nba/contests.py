# src/ivan/sports/nba/contests.py
"""
DraftKings contests and payout tables for NBA, as pipeline steps (free).

upsert_contests   lobby contest dicts (fetch_all_contests' payload) ->
                  historical_contest_information rows; existing rows keep
                  their values except total_entries, which DK revises.
fetch_contests    the lobby for the next `days_ahead` days (raw dk/lobby) -> upsert.
fetch_payouts     the contest-detail endpoint for contests without a payout
                  table (raw dk/payouts) -> payouts rows.
ensure_contest    a contest the entries file names but the table lacks:
                  fetched from the lobby, then from the contest detail.

The capture tick calls upsert_contests + fetch_payouts for the day's slates;
scripts/nba/sim.py calls ensure_contest for every contest it is asked to run.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, List, Optional

from sqlalchemy import distinct
from sqlalchemy.orm import Session

from ivan.sports.common.draftkings import fetch_all_contests, get_upcoming_payouts
from ivan.sports.nba.models import ContestPayout, HistoricalContest

Log = Callable[[str], None]


def parse_sd(sd: Optional[str]) -> Optional[datetime]:
    """'/Date(1747782300000)/' -> naive UTC datetime (the table's convention)."""
    m = re.search(r"/Date\((\d+)\)/", sd or "")
    if not m:
        return None
    return datetime.fromtimestamp(int(m.group(1)) / 1000.0, tz=timezone.utc).replace(tzinfo=None)


def contest_row(c: dict) -> Optional[dict]:
    """A lobby contest dict's fields for historical_contest_information, or None when it lacks the essentials."""
    try:
        cid = int(c["id"]); dg = int(c.get("dg") or 0)
        fee = float(c.get("a") or 0.0); size = int(c.get("m") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if not cid or not dg:
        return None
    return dict(contest_id=cid, draft_group_id=dg, name=c.get("n"), date=parse_sd(c.get("sd")), entry_fee=fee,
                contest_type=c.get("gameType"), total_entries=size, multi_entry=int(c.get("mec") or 1))


def upsert_contests(session: Session, contests: Iterable[dict], draft_groups: Optional[Iterable[int]] = None) -> int:
    """Store the lobby's contests (optionally only those of `draft_groups`). Returns rows added."""
    want = {int(d) for d in draft_groups} if draft_groups is not None else None
    added = 0
    for c in contests:
        row = contest_row(c)
        if row is None or (want is not None and row["draft_group_id"] not in want):
            continue
        hc = session.get(HistoricalContest, row["contest_id"])
        if hc is None:
            session.add(HistoricalContest(**row)); added += 1
        elif hc.total_entries != row["total_entries"] and row["total_entries"]:
            hc.total_entries = row["total_entries"]
    session.commit()
    return added


def fetch_contests(session: Session, days_ahead: int = 1, draft_groups: Optional[Iterable[int]] = None, log: Log = print) -> int:
    """The lobby (raw dk/lobby) -> upsert_contests. Returns rows added."""
    contests = fetch_all_contests("NBA", raw_sport="nba")
    n = upsert_contests(session, contests, draft_groups)
    log(f"contests: {len(contests)} in the lobby, {n} new")
    return n


def payout_rows(contest_id: int, payload: dict) -> List[ContestPayout]:
    detail = (payload or {}).get("contestDetail", {}) or {}
    rows = []
    for tier in detail.get("payoutSummary", []) or []:
        descs = tier.get("payoutDescriptions") or []
        if not descs:
            continue
        rows.append(ContestPayout(contest_id=int(contest_id), min_position=tier.get("minPosition"), max_position=tier.get("maxPosition"),
                                  payout_amount=float(descs[0].get("value", 0.0) or 0.0), payout_text=descs[0].get("payoutDescription", ""),
                                  currency="USD"))
    return rows


def contests_without_payouts(session: Session, contest_ids: Optional[Iterable[int]] = None, since_days: Optional[int] = 1) -> List[int]:
    have = {r[0] for r in session.query(distinct(ContestPayout.contest_id)).all() if r[0] is not None}
    q = session.query(HistoricalContest.contest_id)
    if contest_ids is not None:
        q = q.filter(HistoricalContest.contest_id.in_([int(c) for c in contest_ids]))
    elif since_days is not None:
        q = q.filter(HistoricalContest.date >= datetime.utcnow() - timedelta(days=since_days))
    return sorted(int(c) for (c,) in q.all() if int(c) not in have)


def fetch_payouts(session: Session, contest_ids: Optional[Iterable[int]] = None, since_days: Optional[int] = 1, log: Log = print) -> int:
    """Payout tables (raw dk/payouts) for the contests that lack one. Returns contests filled."""
    missing = contests_without_payouts(session, contest_ids, since_days)
    filled = 0
    for cid in missing:
        payload = get_upcoming_payouts(cid, raw_sport="nba")
        rows = payout_rows(cid, payload) if payload else []
        if not rows:
            log(f"payouts: none for contest {cid}")
            continue
        session.bulk_save_objects(rows); session.commit(); filled += 1
    if missing:
        log(f"payouts: {filled} of {len(missing)} contests filled")
    return filled


def ensure_contest(session: Session, contest_id: int, log: Log = print) -> Optional[HistoricalContest]:
    """The contest row, fetched from the lobby or the contest detail when the table lacks it; None when DK has nothing."""
    hc = session.get(HistoricalContest, int(contest_id))
    if hc is not None:
        return hc
    fetch_contests(session, days_ahead=1, log=log)
    hc = session.get(HistoricalContest, int(contest_id))
    if hc is not None:
        return hc
    payload = get_upcoming_payouts(int(contest_id), raw_sport="nba")
    d = (payload or {}).get("contestDetail") or {}
    if not d:
        return None
    try:
        hc = HistoricalContest(contest_id=int(contest_id), draft_group_id=int(d.get("draftGroupId") or 0), name=d.get("name"),
                               date=parse_sd(d.get("contestStartTime")) if isinstance(d.get("contestStartTime"), str) and "Date(" in d.get("contestStartTime")
                               else None,
                               entry_fee=float(d.get("entryFee") or 0.0), contest_type=d.get("gameTypeName") or d.get("gameType"),
                               total_entries=int(d.get("maximumEntries") or d.get("totalEntries") or 0), multi_entry=int(d.get("maximumEntriesPerUser") or 1))
    except (TypeError, ValueError):
        return None
    session.add(hc)
    rows = payout_rows(int(contest_id), payload)
    if rows:
        session.bulk_save_objects(rows)
    session.commit()
    log(f"contest {contest_id} ({hc.name}) stored from the contest detail")
    return hc
