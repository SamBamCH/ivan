# src/ivan/sports/nfl/results.py
"""
What our NFL entries made, from DraftKings' word only: the finish of every entry of ours in a scraped contest
(contest_entry_result: rank, points and payout as the standings page gave them, ties already shared; written by
scripts/nfl/get_contest_data.py --username, which the settle job runs after the slate) against the entry fee of the
contest (historical_contest_information). Nothing here is rescored from the box scores and nothing is a model: this is
the money, the NFL counterpart of ivan.sports.nba.backtest.results. The after-tax backtest (backtest_tax) answers a
different question: what the chooser's picks would have made, played or not.

    contest_rows(session, slate_key)   one row per contest we have finishes for: fees, won, net, ROI, best rank, cashed
    unscored_rows(session, slate_key)  contests the chooser entered on a slate past its lock that were never scraped
    slate_results(session, slate_key)  one slate (a draft group): its contests and their totals (the slate's backtest page)
    contest_entries(session, id)       our entries in one contest, best first (the contest's panel)
    ledger_days(now, session)          every slate summed per day and overall (the picker, the home tile; ivan.core.pnl.roll_up)

A slate is a DraftKings draft group; its day is the local day of its first lock. A contest is unscored when the settle
job's status row lists it as played (the chooser's output named it), no finish of ours is recorded and it has no
standings: what it cost is the chooser's entry count times the fee, what it won is unknown and out of every total.
A contest with standings and no entry of ours is one we chose for and did not enter: not ours, not counted.

    scripts/nfl/results.py [--slate KEY | --contest ID]
"""
from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from ivan.core import dk_history, pnl
from ivan.sports.nfl.backtest_tax import utc_to_local
from ivan.sports.nfl.paths import OUTPUT_DIR

SPORT = "nfl"
SOURCE = ("the finish DraftKings gave each of our entries in the scraped standings (contest_entry_result) against the entry fees "
          "(scripts/nfl/results.py; scripts/nfl/settle.py fetches them)")
MODE_OF_TYPE = {"Showdown Captain Mode": "showdown", "Classic": "classic"}
PAST_LOCK = ("waiting", "login_needed", "settled")          # the settle job's words for a slate whose lock has passed


def _dt(v) -> Optional[datetime]:
    if v is None or isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        return None


def _row(contest_id, draft_group, name, date_utc, fee, ctype, total_entries) -> dict:
    lock = utc_to_local(_dt(date_utc)) if _dt(date_utc) else None
    return dict(contest_id=int(contest_id), draft_group=str(draft_group), name=str(name or f"contest {contest_id}"), entry_fee=float(fee or 0),
                mode=MODE_OF_TYPE.get(str(ctype), str(ctype or "").lower()), field=int(total_entries or 0), lock=lock,
                day=lock.date().isoformat() if lock else None)


# ----------------------------------------------------------------------------- the contests
def contest_rows(session, slate_key: Optional[str] = None) -> List[dict]:
    """One row per contest with a recorded finish of ours (every one final: the scrape runs after the slate), biggest fees first."""
    q = """SELECT h.contest_id, h.draft_group_id, h.name, h.date, h.entry_fee, h.contest_type, h.total_entries,
                  count(*) AS n, sum(e.payout) AS won, min(e.rank) AS best, max(e.points) AS best_points,
                  sum(CASE WHEN e.payout > 0 THEN 1 ELSE 0 END) AS cashed,
                  sum(CASE WHEN e.payout > h.entry_fee THEN e.payout - h.entry_fee ELSE 0 END) AS gross_win,
                  max(e.scraped_at) AS scraped_at
           FROM contest_entry_result e JOIN historical_contest_information h ON h.contest_id = e.contest_id"""
    params = {}
    if slate_key is not None:
        q += " WHERE CAST(h.draft_group_id AS TEXT) = :k"
        params["k"] = str(slate_key)
    out = []
    for r in session.execute(text(q + " GROUP BY h.contest_id").bindparams(**params)).all():
        d = _row(r.contest_id, r.draft_group_id, r.name, r.date, r.entry_fee, r.contest_type, r.total_entries)
        n, won = int(r.n), float(r.won or 0)
        fees = d["entry_fee"] * n
        at = _dt(r.scraped_at)
        d.update(ours=n, found=n, final=True, detail=True, fees=fees, won=won, net=won - fees, roi=(won - fees) / fees if fees else None,
                 best_rank=int(r.best) if r.best is not None else None, best_points=float(r.best_points) if r.best_points is not None else None,
                 cashed=int(r.cashed or 0), min_cash=None, gross_win=float(r.gross_win or 0), gross_loss=float(r.gross_win or 0) - (won - fees),
                 export_at=utc_to_local(at) if at else None, scored="DraftKings' standings" + (f", scraped {utc_to_local(at):%m-%d %H:%M}" if at else ""))
        out.append(d)
    out.sort(key=lambda d: -d["fees"])
    return out


_CHOSEN: Dict[str, Dict[int, int]] = {}          # chooser output file -> {contest id: entries}; the files are written once


def chosen_counts(dirs: Optional[Sequence[Path]] = None) -> Dict[int, int]:
    """{contest id: the entries the chooser's latest run picked for it}, from chosen*.choose.<stamp>.json in the output directories."""
    out: Dict[int, int] = {}
    for d in (dirs if dirs is not None else (OUTPUT_DIR, OUTPUT_DIR / "web")):
        for f in sorted(glob.glob(str(Path(d) / "chosen*.choose.*.json")), key=lambda p: Path(p).name.split(".")[2]):
            if f not in _CHOSEN:
                try:
                    _CHOSEN[f] = {int(cid): int((v or {}).get("entries") or 0) for cid, v in json.load(open(f)).items()}
                except (OSError, ValueError, AttributeError):
                    _CHOSEN[f] = {}
            out.update(_CHOSEN[f])
    return out


def unscored_rows(session, slate_key: Optional[str] = None, dirs: Optional[Sequence[Path]] = None) -> List[dict]:
    """Contests the settle job lists as played on a slate past its lock, with no finish of ours and no standings (see the module doc)."""
    from ivan.core import slates as slate_status
    try:
        status = slate_status.rows(session, SPORT)
    except OperationalError:                         # no slate_status table: the settle job never ran on this database
        return []
    want: Dict[int, str] = {}
    for st in status:
        if st.status not in PAST_LOCK or (slate_key is not None and str(st.slate_key) != str(slate_key)):
            continue
        scraped = set(json.loads(st.scraped or "[]"))
        for cid in json.loads(st.played or "[]"):
            if cid not in scraped:
                want[int(cid)] = str(st.slate_key)
    if not want:
        return []
    ids = ",".join(str(c) for c in want)
    have = {int(r[0]) for r in session.execute(text(f"SELECT DISTINCT contest_id FROM contest_entry_result WHERE contest_id IN ({ids})")).all()}
    counts = None
    out = []
    for r in session.execute(text(f"""SELECT contest_id, draft_group_id, name, date, entry_fee, contest_type, total_entries
                                      FROM historical_contest_information WHERE contest_id IN ({ids})""")).all():
        if int(r.contest_id) in have:
            continue
        counts = chosen_counts(dirs) if counts is None else counts
        d = _row(r.contest_id, want[int(r.contest_id)], r.name, r.date, r.entry_fee, r.contest_type, r.total_entries)
        n = int(counts.get(d["contest_id"], 0))
        d.update(ours=n, found=0, final=False, fees=d["entry_fee"] * n, won=None, net=None, roi=None, best_rank=None, best_points=None, cashed=0,
                 min_cash=None, gross_win=None, gross_loss=None, export_at=None,
                 scored="the chooser entered it; no standings scraped (scripts/nfl/settle.py)")
        out.append(d)
    return out


# ----------------------------------------------------------------------------- one slate, one contest
def slate_results(session, slate_key: str) -> dict:
    """{"slate_key", "contests" (final first, biggest fees first), "summary" (pnl.agg), "complete"} for the slate's backtest page."""
    rows = contest_rows(session, slate_key) + unscored_rows(session, slate_key)
    mine = [s for s in _with_history(session, [dict(key=str(slate_key), day=min([r["day"] for r in rows if r["day"]], default=""), complete=False, contests=rows)])
            if str(s["key"]) == str(slate_key)]
    rows = sorted(mine[0]["contests"], key=lambda r: (not r["final"], -(r["fees"] or 0))) if mine else []
    return dict(slate_key=str(slate_key), contests=rows, summary=pnl.agg(rows), complete=bool(rows) and all(r["final"] for r in rows))


def _with_history(session, slates: List[dict]) -> List[dict]:
    """The slates with DraftKings' entry history laid over them (ivan.core.dk_history.merge): its money where it has the contest, its
    contests we have no standings for under their draft group (historical_contest_information), unscored rows it does not list dropped."""
    ids = [int(c) for c in dk_history.entries(SPORT).contest_id.unique()]
    slate_of: Dict[int, str] = {}
    for i in range(0, len(ids), 900):
        q = f"SELECT contest_id, draft_group_id FROM historical_contest_information WHERE contest_id IN ({','.join(map(str, ids[i:i + 900]))})"
        slate_of.update({int(c): str(dg) for c, dg in session.execute(text(q)).all()})
    return dk_history.merge(SPORT, slates, slate_of=slate_of)


def contest_entries(session, contest_id: int) -> Optional[dict]:
    """One contest's row and our entries in it, best first: rank, the tie it shares, points, payout, net, the lineup, our copies of it."""
    dg = session.execute(text("SELECT draft_group_id FROM historical_contest_information WHERE contest_id = :c").bindparams(c=int(contest_id))).scalar()
    if dg is None:
        return None
    rows = [r for r in contest_rows(session, str(dg)) if r["contest_id"] == int(contest_id)]
    if not rows:
        return None
    c = rows[0]
    es = session.execute(text("""SELECT entry_id, rank, tie_size, points, payout, lineup FROM contest_entry_result
                                 WHERE contest_id = :c ORDER BY rank, entry_id""").bindparams(c=int(contest_id))).all()
    copies: Dict[str, int] = {}
    for e in es:
        copies[e.lineup or ""] = copies.get(e.lineup or "", 0) + 1
    entries = [dict(entry_id=int(e.entry_id), rank=int(e.rank), tie_size=int(e.tie_size or 1), points=e.points, payout=float(e.payout or 0),
                    net=float(e.payout or 0) - c["entry_fee"], lineup=e.lineup or "", copies=copies[e.lineup or ""] if e.lineup else 1) for e in es]
    return dict(contest=c, entries=entries)


# ----------------------------------------------------------------------------- the season
def ledger_days(now: Optional[datetime] = None, session=None, dirs: Optional[Sequence[Path]] = None) -> dict:
    """
    Every slate with a contest of ours, summed: {"overall": agg, "days": {iso day: agg + slates, incomplete}, "slates":
    {draft group: agg + day, complete}, "source"} (ivan.core.pnl.roll_up). `session` is the NFL database's; one is opened when
    none is given.
    """
    own = session is None
    if own:
        from ivan.database.session import get_session, init_db
        session = get_session(init_db(SPORT))
    try:
        rows = contest_rows(session) + unscored_rows(session, dirs=dirs)
        by: Dict[str, List[dict]] = {}
        for r in rows:
            if r["day"]:
                by.setdefault(r["draft_group"], []).append(r)
        slates = [dict(key=k, day=min(r["day"] for r in g), complete=all(r["final"] for r in g), contests=g) for k, g in by.items()]
        slates = _with_history(session, slates)
    finally:
        if own:
            session.close()
    out = pnl.roll_up(slates)
    out["source"] = SOURCE
    return out
