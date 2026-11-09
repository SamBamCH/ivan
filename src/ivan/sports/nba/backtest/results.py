"""
ivan.sports.nba.backtest.results

What our entries made, from DraftKings' word only: the final standings export of each contest (every game over, no LOCKED
slot) and the payout table. Our entries are the ones DraftKings lists under our user name (config backtest.dk_user,
"SamBamCH (3/10)" or "SamBamCH"), plus any Entry ID the desk's upload files name. Nothing here is rescored from the box
scores, and nothing is a model: this is the money.

    score_contest(contest, ours, ent, pay, export_at, user)   one parsed export -> the contest's result (or unscored)
    compute(dg, session)                                      every contest of the slate we played: results with entries, from the raw store
    slate_ledger(dg)                                          the compact result of one slate, cached in slates/<dg>/results/ledger.json
    ledger_days(now)                                          every held past slate's ledger summed per day and overall (the picker, the home tile)

The cache: a slate whose every contest has a final export is complete and never recomputed (past results do not
change). An incomplete one (a contest still mid-slate or unscraped) is re-checked at most every RECHECK minutes, and
recomputed only when a contest's newest final export is newer than the one the ledger saw. So a page that sums the
season reads a few small JSON files, and the first page after a scrape pays the parse once.

The actuals are the local machine's: a consumer desk (the server) does not get them (ivan.web.nba.core leaves the
sport's ledger out and the backtest view keeps only the model's columns).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.core import raw
from ivan.sports.nba.names import normalize_name

SPORT = "nba"
RECHECK = timedelta(minutes=10)          # how often an incomplete slate's exports are looked at again
LEDGER_VERSION = 2                       # bump when the ledger's content changes: older files are recomputed


def our_user() -> Optional[str]:
    """The DraftKings user whose entries are ours (config backtest.dk_user); None when not configured (upload files only)."""
    try:
        from ivan.core.config import load_config
        u = (load_config(SPORT).get("backtest") or {}).get("dk_user")
        return str(u).strip() or None
    except Exception:   # noqa: BLE001 - no config in a test
        return None


_USER_RX = re.compile(r"^(.*?)(?:\s*\(\d+/\d+\))?\s*$")


def entry_user(entry_name: str) -> str:
    """'SamBamCH (3/10)' -> 'SamBamCH'; 'SamBamCH' -> 'SamBamCH'."""
    m = _USER_RX.match(str(entry_name or ""))
    return (m.group(1) if m else str(entry_name or "")).strip()


def ours_mask(ent: pd.DataFrame, ours: Iterable[str] = (), user: Optional[str] = None) -> np.ndarray:
    """Which rows of a parsed export are ours: the Entry IDs given, or the entries under our user name."""
    ids = {str(x) for x in ours}
    m = ent.entry_id.astype(str).isin(ids).to_numpy() if ids else np.zeros(len(ent), dtype=bool)
    if user:
        m = m | (ent.entry_name.map(entry_user).str.lower() == user.lower()).to_numpy()
    return m


# ----------------------------------------------------------------------------- one contest
def payout_table(session: Session, contest_id: int, n: int) -> np.ndarray:
    """Payout by finishing place 1..n (index 0 = 1st), from the payouts table."""
    pay = np.zeros(max(n, 1))
    rows = session.execute(text("select min_position, max_position, payout_amount from payouts where contest_id = :c").bindparams(c=int(contest_id))).all()
    for lo, hi, amt in rows:
        lo, hi = int(lo), min(int(hi), len(pay))
        if lo <= hi:
            pay[lo - 1:hi] = float(amt)
    return pay


def ranked_payouts(points: np.ndarray, pay: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rank, payout) per entry: rank = 1 + the entries strictly ahead; tied entries share the payouts of the places they cover."""
    order = np.argsort(-points, kind="stable")
    rank = np.empty(len(points), dtype=int); won = np.zeros(len(points))
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and points[order[j + 1]] == points[order[i]]:
            j += 1
        places = pay[i:j + 1] if i < len(pay) else np.zeros(0)
        share = float(places.sum()) / (j - i + 1) if len(places) else 0.0
        for k in range(i, j + 1):
            rank[order[k]] = i + 1; won[order[k]] = share
        i = j + 1
    return rank, won


def unscored(contest: dict, ours, why: str, export_at: Optional[datetime] = None, field: int = 0, locked: int = 0) -> dict:
    """A contest without a result yet: what we paid is known (`ours`: our Entry IDs, or their count), what we won is not (won, net
    and ROI are None, never 0)."""
    n = int(ours) if isinstance(ours, (int, np.integer)) else len(ours)
    return dict(contest, ours=n, found=0, final=False, scored=why, export_at=export_at, field=field, locked=locked, entries=[],
                fees=float(contest["entry_fee"]) * n, won=None, net=None, roi=None, best_rank=None, cashed=0, min_cash=None,
                unique=None, max_copies=None, dup_share=None, winner=None, side=None)


def score_contest(contest: dict, ours: set, ent: pd.DataFrame, pay: np.ndarray, export_at: Optional[datetime] = None,
                  user: Optional[str] = None) -> dict:
    """
    One contest from a parsed standings export (standings.parse_export) and its payout table. A final export (every game over,
    no LOCKED slot) is scored as DraftKings scored it: its points, ranks from them, ties sharing the payouts of the places they
    cover; the field's duplication; our entries in it (the Entry IDs in `ours` and the entries under `user`). A mid-slate export
    is not scored (unscored(...)): the late swap's fetch shows every not-yet-tipped player as LOCKED, ours included.
    """
    from ivan.sports.nba.data_clean.standings import export_is_final, locked_entries
    cid = int(contest["contest_id"])
    n_locked = locked_entries(ent)
    if not export_is_final(ent):
        when = f" {export_at:%m-%d %H:%M}" if export_at else ""
        return unscored(contest, ours, f"mid-slate export{when}: {n_locked:,} of {len(ent):,} entries show a LOCKED slot; scrape it final",
                        export_at, len(ent), n_locked)
    out = dict(contest, ours=len(ours), final=True, scored=f"DraftKings' final export{f' {export_at:%m-%d %H:%M}' if export_at else ''}",
               export_at=export_at, field=len(ent), locked=n_locked, side=None)
    pts = ent.points.fillna(0).to_numpy(dtype=float)
    rank, won = ranked_payouts(pts, pay)
    keys = ent.names.map(lambda ns: tuple(sorted(normalize_name(n) for n in ns if n)))
    copies = Counter(keys)
    out["unique"] = len(copies)
    out["max_copies"] = max(copies.values()) if copies else 0
    out["dup_share"] = float(sum(v for v in copies.values() if v > 1)) / len(ent) if len(ent) else None
    out["winner"] = float(pts.max()) if len(pts) else None
    mine = ours_mask(ent, ours, user)
    rows = []
    for i in np.nonzero(mine)[0]:
        rows.append(dict(entry_id=str(ent.entry_id.iloc[i]), names=list(ent.names.iloc[i]), points=float(pts[i]), rank=int(rank[i]),
                         payout=float(won[i]), copies=int(copies[keys.iloc[i]]), contest=contest["name"], contest_id=cid))
    rows.sort(key=lambda r: r["rank"])
    out["entries"] = rows
    out["found"] = len(rows)
    out["ours"] = max(len(ours), len(rows))
    out["won"] = float(sum(r["payout"] for r in rows))
    out["fees"] = float(contest["entry_fee"]) * (len(rows) or len(ours))
    out["net"] = out["won"] - out["fees"]
    out["best_rank"] = rows[0]["rank"] if rows else None
    out["cashed"] = sum(1 for r in rows if r["payout"] > 0)
    out["roi"] = out["net"] / out["fees"] if out["fees"] else None
    out["min_cash"] = float(pts[won > 0].min()) if (won > 0).any() else None
    return out


def summarize(results: List[dict]) -> dict:
    """The slate's tiles over the contests with a final export; the rest listed as unscored with what they cost."""
    scored = [r for r in results if r["final"]]
    fees = sum(r["fees"] for r in scored); won = sum(r["won"] for r in scored)
    claimed = [(r["claim"]["roi"], r["fees"]) for r in scored if r.get("claim") and r["claim"].get("roi") is not None]
    return dict(contests=len(scored), entries=sum(r["found"] for r in scored), fees=fees, won=won, net=won - fees,
                roi=(won - fees) / fees if fees else None, cashed=sum(r["cashed"] for r in scored),
                claimed_roi=(sum(a * w for a, w in claimed) / sum(w for _, w in claimed)) if claimed else None,
                final=bool(results) and len(scored) == len(results),
                unscored=[dict(name=r["name"], contest_id=r["contest_id"], fees=r["fees"], ours=r["ours"], why=r["scored"]) for r in results if not r["final"]],
                fees_unscored=sum(r["fees"] for r in results if not r["final"]))


# ----------------------------------------------------------------------------- one slate
def final_exports(contest_ids: Iterable[int]) -> Dict[int, raw.RawArtifact]:
    """{contest id: its newest standings export} for the ids that have one, from one pass over the raw index."""
    want = {str(int(c)) for c in contest_ids}
    out: Dict[int, raw.RawArtifact] = {}
    for a in raw.index(SPORT):
        if a.kind == "dk/standings" and a.key in want and (int(a.key) not in out or a.stamp > out[int(a.key)].stamp):
            out[int(a.key)] = a
    return out


def compute(dg: int, session: Session, ours_ids: Optional[Dict[int, set]] = None, user: Optional[str] = None,
            exports: Optional[Dict[int, tuple]] = None, root: Optional[Path] = None) -> List[dict]:
    """
    Every contest of the slate we played, scored from its newest export (score_contest): the contests the slate's entries file
    names (standings.our_contest_ids), those the upload files name (`ours_ids`, {contest id: Entry IDs}) and, among the slate's
    contests with a final export, those where our user name appears. `exports` ({contest id: (stamp, text)}) stands in for the
    raw store in tests. Rows carry `entries` (ours, with lineups) and `side` (the export's player table, final exports only).
    """
    from ivan.sports.nba.data_fetch.standings import our_contest_counts
    from ivan.sports.nba.data_clean.standings import parse_export
    from ivan.sports.nba.data_fetch.contests import slate_contests
    user = user if user is not None else our_user()
    ours_ids = {int(k): set(v) for k, v in (ours_ids or {}).items()}
    rows = {int(r.contest_id): dict(contest_id=int(r.contest_id), name=r.name, entry_fee=float(r.entry_fee or 0), total_entries=int(r.total_entries or 0))
            for r in slate_contests(session, int(dg))}
    counts = our_contest_counts(int(dg), root=root)                    # the entries file: how many we have in each contest
    want = set(counts) | set(ours_ids)
    if exports is None:
        arts = final_exports(rows.keys())
        exports = {cid: (a.stamp, None, a.path) for cid, a in arts.items()}
    out = []
    for cid in sorted(set(exports) | want):
        c = rows.get(cid) or dict(contest_id=cid, name=f"contest {cid}", entry_fee=0.0, total_entries=0)
        ours = ours_ids.get(cid, set())
        n_ours = max(len(ours), counts.get(cid, 0))
        ex = exports.get(cid)
        if ex is None:
            out.append(unscored(c, n_ours, "no standings export: scrape it (scripts/nba/standings.py scrape)")); continue
        stamp, txt = ex[0], ex[1]
        if txt is None:
            txt = raw.load(ex[2], SPORT)
        export_at = datetime.strptime(stamp, "%Y%m%d-%H%M%S") if stamp else None
        ent, side = parse_export(txt)
        r = score_contest(c, ours, ent, payout_table(session, cid, len(ent)), export_at, user)
        if r["final"] and r["found"] == 0 and cid not in want:
            continue                                                   # a public contest scraped for the field, not one of ours
        if not r["final"] and cid not in want and not ours_mask(ent, ours, user).any():
            continue                                                   # a mid-slate export of a contest we did not play
        if not r["final"]:
            r["ours"] = n_ours; r["fees"] = float(c["entry_fee"]) * n_ours
        else:
            r["ours"] = max(r["ours"], n_ours); r["side"] = side
        out.append(r)
    out.sort(key=lambda r: (not r["final"], -r["fees"]))
    return out


def _compact(r: dict) -> dict:
    keep = ("contest_id", "name", "entry_fee", "ours", "found", "final", "scored", "field", "fees", "won", "net", "roi", "best_rank", "cashed", "min_cash")
    d = {k: r.get(k) for k in keep}
    d["export_at"] = r["export_at"].isoformat(timespec="minutes") if r.get("export_at") else None
    return d


def _agg(rows: List[dict]) -> dict:
    """fees, won, net, roi, entries, cashed, contests over the final rows; what the unscored ones cost."""
    fin = [r for r in rows if r.get("final")]
    fees = float(sum(r["fees"] for r in fin)); won = float(sum(r["won"] or 0 for r in fin))
    return dict(fees=fees, won=won, net=won - fees, roi=(won - fees) / fees if fees else None, entries=int(sum(r.get("found") or 0 for r in fin)),
                cashed=int(sum(r.get("cashed") or 0 for r in fin)), contests=len(fin), unscored=len(rows) - len(fin),
                fees_unscored=float(sum(r["fees"] for r in rows if not r.get("final"))))


def ledger_path(dg: int, root: Optional[Path] = None) -> Path:
    from ivan.sports.nba import slates as S
    return S.paths(int(dg), root).root / "results" / "ledger.json"


def read_ledger(dg: int, root: Optional[Path] = None) -> Optional[dict]:
    p = ledger_path(dg, root)
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return d if d.get("version") == LEDGER_VERSION else None


def _is_current(led: dict, now: datetime, root: Optional[Path] = None) -> bool:
    """A complete ledger stands; an incomplete one is re-checked every RECHECK minutes against the exports on disk."""
    if led.get("complete"):
        return True
    checked = datetime.fromisoformat(led["checked_at"]) if led.get("checked_at") else None
    if checked is not None and now - checked < RECHECK:
        return True
    seen = {int(k): v for k, v in (led.get("exports") or {}).items()}
    cur = {cid: a.stamp for cid, a in final_exports(set(seen) | set(led.get("wanted") or [])).items()}
    return cur == seen


def slate_ledger(dg: int, now: Optional[datetime] = None, force: bool = False, root: Optional[Path] = None, session: Optional[Session] = None) -> dict:
    """
    The slate's result, compact, from the cache when it is current (see the module doc) else recomputed and written to
    slates/<dg>/results/ledger.json: {draft_group, day, computed_at, checked_at, complete, user, exports, wanted, contests, summary}.
    """
    from ivan.sports.nba import slates as S
    from ivan.sports.nba.data_fetch.standings import our_contest_ids
    now = now or datetime.now()
    dg = int(dg)
    led = None if force else read_ledger(dg, root)
    if led is not None and _is_current(led, now, root):
        if not led.get("complete") and (not led.get("checked_at") or now - datetime.fromisoformat(led["checked_at"]) >= RECHECK):
            led["checked_at"] = now.isoformat(timespec="seconds"); _write(ledger_path(dg, root), led)
        return led
    own = session is None
    sess = session or S.session(dg, root)
    try:
        with S.using(dg):
            rows = compute(dg, sess, root=root)
    finally:
        if own:
            sess.close()
    wanted = sorted(our_contest_ids(dg, root=root))
    exports = {str(r["contest_id"]): r["export_at"].strftime("%Y%m%d-%H%M%S") for r in rows if r.get("export_at")}
    meta = S.read_meta(dg, root) if S.holds(dg, root) else {}
    led = dict(version=LEDGER_VERSION, draft_group=dg, day=meta.get("day"), computed_at=now.isoformat(timespec="seconds"),
               checked_at=now.isoformat(timespec="seconds"), user=our_user(), entries_file=bool(wanted),
               complete=bool(rows) and all(r["final"] for r in rows) and (not wanted or all(any(r["contest_id"] == c and r["final"] for r in rows) for c in wanted)),
               exports=exports, wanted=wanted, contests=[_compact(r) for r in rows], summary=_agg(rows))
    _write(ledger_path(dg, root), led)
    return led


def _write(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1, default=str))


# ----------------------------------------------------------------------------- the season
def ledger_days(now: Optional[datetime] = None, root: Optional[Path] = None, today: Optional[date] = None) -> dict:
    """
    Every held slate whose day has passed, summed: {"overall": agg, "days": {iso day: agg + slates, incomplete}, "slates": {dg: ledger
    summary}}. Reads the cached ledgers; computes only what is missing or incomplete (see slate_ledger). For the picker and the
    home tile: the money per day and over the season, local machine only.
    """
    from ivan.sports.nba import slates as S
    now = now or datetime.now()
    today = today or now.date()
    days: Dict[str, dict] = {}
    slates: Dict[int, dict] = {}
    for dg, meta in S.held_meta(root).items():
        day = str(meta.get("day") or "")[:10]
        if not day or date.fromisoformat(day) >= today:
            continue
        try:
            led = slate_ledger(dg, now, root=root)
        except Exception:   # noqa: BLE001 - one broken slate must not take the page down
            continue
        if not led["contests"]:
            continue
        slates[dg] = dict(led["summary"], day=day, complete=led["complete"], contests_list=led["contests"])
        days.setdefault(day, []).append(led)
    out_days = {}
    for day, leds in days.items():
        rows = [c for led in leds for c in led["contests"]]
        a = _agg(rows)
        a["slates"] = [led["draft_group"] for led in leds]
        a["incomplete"] = sum(1 for led in leds if not led["complete"])
        out_days[day] = a
    overall = _agg([c for led in [slates[d] for d in slates] for c in led["contests_list"]])
    overall["days"] = len(out_days); overall["slates"] = len(slates); overall["incomplete"] = sum(1 for s in slates.values() if not s["complete"])
    for s in slates.values():
        s.pop("contests_list", None)
    return dict(overall=overall, days=dict(sorted(out_days.items(), reverse=True)), slates=slates)
