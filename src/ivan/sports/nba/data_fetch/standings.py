# src/ivan/sports/nba/data_fetch/standings.py
"""
DraftKings' standings exports: which contests to scrape and the scrape itself (the parsing and the derived fields are
data_clean.standings).

    contests_to_scrape   per played slate (archive capture with a lock in the window): every contest we played (the slate's
                         entries file) plus the largest public classic contests, minus those already exported FINAL (a
                         mid-slate export is scraped again)
    scrape               the export of each contest through the logged-in Chrome profile (ivan.sports.common.draftkings),
                         kept raw under dk/standings/<id>
    record_scrape        the outcome noted on every held slate it touched (slates/<dg>/results/scrape.json: the calendar's red)
    our_contest_counts   {contest id: our entries} on a slate, from its entries file

DraftKings keeps standings for about two weeks, so scrape within days.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from ivan.core import archive
from ivan.sports.nba.data_clean.standings import has_final_standings
from ivan.sports.nba.data_fetch.contests import slate_contests

Log = Callable[[str], None]


SPORT = "nba"

PUBLIC_MIN_ENTRIES = 1000

GRACE_HOURS = 5.0           # after lock: the last game must be over before the export is final

def played_slates(session: Session, since: datetime, until: datetime, now: Optional[datetime] = None, grace_hours: float = GRACE_HOURS):
    """Archive captures (one per slate key) whose lock is in the window and at least grace_hours ago."""
    now = now or datetime.now()
    return [c for c in archive.slates(session, SPORT, since, until) if c.lock + timedelta(hours=grace_hours) <= now]

def entries_contest_counts(entries_csv: Optional[Path]) -> Dict[int, int]:
    """{Contest ID: our entries in it} from a DraftKings entries file; empty when there is no file."""
    if not entries_csv or not Path(entries_csv).exists():
        return {}
    try:
        from ivan.sports.nba.contest.export import read_entries
        return {int(c): int(n) for c, n in read_entries(entries_csv).groupby("Contest ID").size().items()}
    except Exception:   # noqa: BLE001
        return {}

def our_contest_counts(draft_group: int, entries_csv: Optional[Path] = None, root: Optional[Path] = None) -> Dict[int, int]:
    """
    {contest id: our entries} on a slate: from the slate's own entries file (data/nba/slates/<dg>/input/entries.csv, the
    DraftKings export the desk works from), else from `entries_csv` when given (a machine that does not hold the slate).
    Until 2026-11-05 this read the pre-split shared path, which no longer exists, so only the public contests were scraped.
    """
    from ivan.sports.nba import slates as SL
    dg = int(draft_group)
    if SL.holds(dg, root):
        ours = entries_contest_counts(SL.paths(dg, root).entries_csv)
        if ours:
            return ours
    return entries_contest_counts(entries_csv)

def our_contest_ids(draft_group: int, entries_csv: Optional[Path] = None, root: Optional[Path] = None) -> set:
    """The contests we played on a slate (our_contest_counts' keys)."""
    return set(our_contest_counts(draft_group, entries_csv, root))

def contests_to_scrape(session: Session, since: datetime, until: datetime, public_n: int = 3, entries_csv: Optional[Path] = None,
                       now: Optional[datetime] = None) -> List[dict]:
    """
    Per played slate: {slate_key, lock, contests: [ids], ours: [ids], scraped: [ids], to_scrape: [ids], names}. The contests are
    every one we played (our_contest_ids) plus the public_n largest public classic contests by size whether we played them or not;
    scraped are those with a FINAL export, to_scrape the rest (a mid-slate export counts as not scraped).
    """
    out = []
    for cap in played_slates(session, since, until, now):
        dg = int(cap.slate_key)
        rows = slate_contests(session, dg)
        ours = our_contest_ids(dg, entries_csv)
        # the lobby leaves contest_type empty on some contests (2026-11-02: the eight largest of slate 156054); a slate key is one
        # game type, so only a contest marked as something other than classic is left out
        pub = sorted([r for r in rows if not (r.contest_type and not str(r.contest_type).lower().startswith("classic")) and int(r.total_entries or 0) >= PUBLIC_MIN_ENTRIES],
                     key=lambda r: -int(r.total_entries or 0))[:public_n]
        ids = sorted({int(r.contest_id) for r in pub} | {int(r.contest_id) for r in rows if int(r.contest_id) in ours})
        scraped = [c for c in ids if has_final_standings(c)]
        out.append(dict(slate_key=cap.slate_key, lock=cap.lock, contests=ids, ours=sorted(c for c in ids if c in ours), scraped=scraped,
                        to_scrape=[c for c in ids if c not in scraped], names={int(r.contest_id): r.name for r in rows if int(r.contest_id) in ids}))
    return out

def record_scrape(plan: List[dict], result: dict, now: Optional[datetime] = None, root: Optional[Path] = None) -> List[Path]:
    """Note the scrape's outcome on every held slate it touched: slates/<dg>/results/scrape.json = {at, contests, ok, failed,
    login_needed}. The readiness of a past day (ivan.sports.nba.backtest.readiness: the calendar's red) reads it; a later scrape that
    succeeds overwrites it."""
    from ivan.sports.nba import slates as SL
    now = now or datetime.now()
    ok = {int(c) for c in result.get("ok") or []}; failed = {int(c) for c in result.get("failed") or []}
    out = []
    for s in plan:
        dg = int(s["slate_key"])
        mine = [int(c) for c in s.get("to_scrape") or []]
        if not mine or not SL.holds(dg, root):
            continue
        rec = dict(at=now.isoformat(timespec="seconds"), contests=mine, ok=sorted(c for c in mine if c in ok),
                   failed=sorted(c for c in mine if c in failed), login_needed=bool(result.get("login_needed")))
        p = SL.paths(dg, root).root / "results" / "scrape.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1))
        out.append(p)
    try:
        from ivan.sports.nba.backtest import readiness
        readiness.forget()
    except Exception:   # noqa: BLE001
        pass
    return out

def scrape(ids: Sequence[int], headless: bool = False, log: Log = print) -> dict:
    """Export every contest in ids through the logged-in Chrome profile. Returns {ok, failed, login_needed}."""
    from ivan.sports.common.draftkings import build_requests_session_from_chrome, download_contest_dataframe
    from ivan.sports.nba.paths import CHROME_PROFILE_DIR
    out = {"ok": [], "failed": [], "login_needed": False}
    if not ids:
        return out
    try:
        sess = build_requests_session_from_chrome(str(CHROME_PROFILE_DIR), headless=headless)
    except RuntimeError as e:
        log(f"login needed: {e}"); out["login_needed"] = True; out["failed"] = list(ids)
        return out
    for cid in ids:
        try:
            df = download_contest_dataframe(sess, int(cid), raw_sport=SPORT)
        except Exception as e:   # noqa: BLE001
            log(f"  {cid}: {type(e).__name__}: {e}"); out["failed"].append(int(cid)); continue
        if df is None or df.empty:
            log(f"  {cid}: no export (not finished, purged, or not logged in)"); out["failed"].append(int(cid)); continue
        out["ok"].append(int(cid)); log(f"  {cid}: {len(df):,} rows")
    return out
