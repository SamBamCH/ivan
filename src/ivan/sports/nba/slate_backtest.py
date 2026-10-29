"""
ivan.sports.nba.slate_backtest

One past slate, read only: how our entries did, what the sim claimed for them, and how every player went against our
projection at lock. Nothing here touches the working tables (the pool, lineups, sims), so a past slate is looked at
without loading it into the desk. The desk's backtest view (/?view=backtest&dg=...) and scripts/nba/backtest_slate.py
show the same.

    mode_for(day, first_lock, now)   which view a slate opens in: "backtest" (an earlier day), "lateswap" (today, after
                                      the first lock), "prelock" (before the first lock)
    slate_backtest(session, dg)      everything for one slate: summary, contests, our entries, players by team

Where the numbers come from:
  our entries   the Entry IDs in the upload files the desk wrote for the slate's contests (data/nba/output/
                chosen_entries.classic.*.csv and lateswap.*.csv); their final lineups come from DraftKings' export
  the field     the newest standings export of the contest (raw dk/standings/<id>: the late swap's Fetch standings,
                or a scrape after the slate). Exports taken during the games carry live points, so every lineup is
                rescored from the box scores of the slate's day when they are in (ranks and payouts then final; ties
                share the payouts of the places they cover); before the box scores arrive the export's points stand
  payouts       the payouts table (stored with the contest)
  the claim     the chooser's run for the contest closest before lock: chosen_roi and chosen_cash
  players       the scorecard join (validation.build_scorecard): our last projection before lock vs the box score,
                with the desk's notes that stood at lock; our exposure from our entries; the field's ownership from
                the largest contest's export (DK's %drafted)
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.core import raw
from ivan.core.utils import normalize_name
from ivan.sports.nba.paths import OUTPUT_DIR

STAMP = re.compile(r"(\d{8}-\d{6})")


def mode_for(day: Optional[date], first_lock: Optional[datetime], now: Optional[datetime] = None) -> str:
    """The view a slate opens in: an earlier day is a backtest; today after the first lock is the late swap; else pre-lock."""
    now = now or datetime.now()
    if day is not None and day < now.date():
        return "backtest"
    if first_lock is not None and now >= first_lock and (day is None or day == now.date()):
        return "lateswap"
    return "prelock"


def _stamp_of(path: Path) -> Optional[datetime]:
    m = STAMP.search(path.name)
    return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S") if m else None


def our_entry_ids(contest_ids, out_dir: Path = OUTPUT_DIR) -> Dict[int, set]:
    """{contest id: our Entry IDs}, from every upload file the desk wrote (the chooser's and the late swap's)."""
    want = {int(c) for c in contest_ids}
    out: Dict[int, set] = defaultdict(set)
    for f in sorted(list(Path(out_dir).glob("chosen_entries.classic.*.csv")) + list(Path(out_dir).glob("lateswap.*.csv"))):
        try:
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    try:
                        cid = int(str(r.get("Contest ID") or "").strip())
                    except ValueError:
                        continue
                    if cid in want and str(r.get("Entry ID") or "").strip():
                        out[cid].add(str(r["Entry ID"]).strip())
        except (OSError, csv.Error):
            continue
    return dict(out)


def claims(contest_ids, lock: Optional[datetime], out_dir: Path = OUTPUT_DIR) -> Dict[int, dict]:
    """{contest id: the chooser's claim}: the run closest before lock that covered the contest (any run when lock is unknown)."""
    want = {str(int(c)) for c in contest_ids}
    best: Dict[int, tuple] = {}
    for f in Path(out_dir).glob("chosen_entries.choose.*.json"):
        t = _stamp_of(f)
        if t is None or (lock is not None and t > lock):
            continue
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for cid, v in d.items():
            if cid in want and isinstance(v, dict) and (int(cid) not in best or t > best[int(cid)][0]):
                best[int(cid)] = (t, dict(at=t, entries=v.get("entries"), roi=v.get("chosen_roi"), cash=v.get("chosen_cash"),
                                          method=(v.get("book") or {}).get("method")))
    return {k: v[1] for k, v in best.items()}


def payout_table(session: Session, contest_id: int, n: int) -> np.ndarray:
    """Payout by finishing place 1..n (index 0 = 1st)."""
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


class BoxPoints(dict):
    """{box name: fantasy points} for one day, looked up by DK's spelling: the name, the name without suffixes or spaces
    ('Robert Williams' = 'Robert Williams III'), then the one player that day with the last name ('Gregory' = 'GG'
    Jackson). A name still unknown scores 0 (he did not play)."""

    def __init__(self, rows):
        from ivan.sports.nba.experience import last_name, loose
        super().__init__(rows)
        self._loose = {loose(k): v for k, v in rows.items()}
        by_last: Dict[str, list] = defaultdict(list)
        for k, v in rows.items():
            by_last[last_name(k)].append(v)
        self._last = {k: v[0] for k, v in by_last.items() if len(v) == 1}
        self._memo: Dict[str, float] = {}

    def points(self, name: str) -> float:
        if name in self._memo:
            return self._memo[name]
        from ivan.sports.nba.experience import last_name, loose
        v = self.get(name)
        if v is None:
            v = self._loose.get(loose(name))
        if v is None:
            v = self._last.get(last_name(name), 0.0)
        self._memo[name] = float(v)
        return float(v)


def box_points(session: Session, day: date) -> BoxPoints:
    rows = session.execute(text("select player_name, dk_fpts from nba_stats where game_date = :d").bindparams(d=day.isoformat())).all()
    return BoxPoints({str(n): float(f or 0.0) for n, f in rows})


def contest_result(session: Session, contest: dict, ours: set, box) -> dict:
    """One contest: the field from its newest export, rescored from the box scores when they are in; our entries in it."""
    from ivan.sports.nba.standings import parse_export
    cid = int(contest["contest_id"])
    out = dict(contest, ours=len(ours), export_at=None, scored="no export", field=0, entries=[], fees=0.0, won=0.0, net=0.0,
               best_rank=None, cashed=0, side=None)
    out["fees"] = float(contest["entry_fee"]) * len(ours)
    art = raw.latest("nba", "dk/standings", cid)
    if art is None:
        return out
    ent, side = parse_export(raw.load(art.path, "nba"))
    out["export_at"] = datetime.strptime(art.stamp, "%Y%m%d-%H%M%S")
    out["field"] = len(ent); out["side"] = side
    keys = ent.names.map(lambda ns: tuple(sorted(normalize_name(n) for n in ns if n)))
    if box:
        pts = np.array([sum(box.points(n) for n in k) for k in keys])
        out["scored"] = "rescored from the box scores"
    else:
        pts = ent.points.fillna(0).to_numpy(dtype=float)
        out["scored"] = f"DK export {out['export_at']:%m-%d %H:%M} (live points; no box scores yet)"
    pay = payout_table(session, cid, len(ent))
    rank, won = ranked_payouts(pts, pay)
    copies = Counter(keys)
    mine = ent.entry_id.astype(str).isin(ours).to_numpy()
    rows = []
    for i in np.nonzero(mine)[0]:
        rows.append(dict(entry_id=str(ent.entry_id.iloc[i]), names=list(ent.names.iloc[i]), points=float(pts[i]), rank=int(rank[i]),
                         payout=float(won[i]), copies=int(copies[keys.iloc[i]]), contest=contest["name"], contest_id=cid))
    rows.sort(key=lambda r: r["rank"])
    out["entries"] = rows
    out["found"] = len(rows)
    out["won"] = float(sum(r["payout"] for r in rows))
    out["fees"] = float(contest["entry_fee"]) * (len(rows) or len(ours))
    out["net"] = out["won"] - out["fees"]
    out["best_rank"] = rows[0]["rank"] if rows else None
    out["cashed"] = sum(1 for r in rows if r["payout"] > 0)
    out["roi"] = out["net"] / out["fees"] if out["fees"] else None
    out["min_cash"] = float(pts[won > 0].min()) if (won > 0).any() else None
    return out


def slate_backtest(session: Session, dg: int, now: Optional[datetime] = None) -> dict:
    from ivan.sports.nba import validation as V
    now = now or datetime.now()
    ours_cap, _, lock = V.captures_for(session, int(dg))
    day = V.slate_day(lock, ours_cap) if len(ours_cap) else None
    contests = [dict(contest_id=int(c), name=n, entry_fee=float(f or 0), total_entries=int(t or 0)) for c, n, f, t in session.execute(
        text("select contest_id, name, entry_fee, total_entries from historical_contest_information where draft_group_id = :d").bindparams(d=int(dg))).all()]
    if day is None and contests:
        d0 = session.execute(text("select min(date) from historical_contest_information where draft_group_id = :d").bindparams(d=int(dg))).scalar()
        day = pd.Timestamp(d0).date() if d0 else None
    ids = our_entry_ids([c["contest_id"] for c in contests])
    box = box_points(session, day) if day else {}
    claim = claims(ids.keys(), lock)
    results = []
    for c in contests:
        if c["contest_id"] not in ids:
            continue
        r = contest_result(session, c, ids[c["contest_id"]], box)
        r["claim"] = claim.get(c["contest_id"])
        results.append(r)
    results.sort(key=lambda r: -r["fees"])
    fees = sum(r["fees"] for r in results); won = sum(r["won"] for r in results)
    claimed = [(r["claim"]["roi"], r["fees"]) for r in results if r.get("claim") and r["claim"].get("roi") is not None]
    summary = dict(contests=len(results), entries=sum(r.get("found", 0) for r in results), fees=fees, won=won, net=won - fees,
                   roi=(won - fees) / fees if fees else None, cashed=sum(r["cashed"] for r in results),
                   claimed_roi=(sum(a * w for a, w in claimed) / sum(w for _, w in claimed)) if claimed else None,
                   final=bool(box), missing_exports=[r["name"] for r in results if r["export_at"] is None])
    # players: the scorecard join, our exposure, the field's ownership (the largest contest's export)
    players: Dict[str, list] = {}
    card = V.build_scorecard(session, int(dg)) if len(ours_cap) else None
    entries = [e for r in results for e in r["entries"]]
    exp = Counter(normalize_name(n) for e in entries for n in e["names"] if n)
    biggest = max((r for r in results if r.get("side") is not None and len(r["side"])), key=lambda r: r["field"], default=None)
    own = {normalize_name(n): float(p) for n, p in zip(biggest["side"].name, biggest["side"].drafted_pct)} if biggest else {}
    if card is not None and len(card.rows):
        rows = card.rows
        for t, g in rows.groupby("team"):
            lst = []
            for r in g.sort_values("minutes_proj", ascending=False).itertuples():
                n = r.player_name
                lst.append(dict(name=n, min_proj=_f(r.minutes_proj), min_act=_f(getattr(r, "min_act", None)), fpts_proj=_f(r.fpts_proj),
                                fpts_act=_f(getattr(r, "fpts_act", None)), err=_f(getattr(r, "fpts_err", None)), played=bool(getattr(r, "played", False)),
                                starter=str(getattr(r, "starter", "") or ""), note=getattr(r, "note", None) if isinstance(getattr(r, "note", None), str) else None,
                                exposure=(exp[n] / len(entries)) if entries else None, own=own.get(n), salary=_f(getattr(r, "salary", None))))
            players[str(t)] = lst
    for r in results:
        r.pop("side", None)
    return dict(draft_group=int(dg), day=day, lock=lock, mode=mode_for(day, lock, now), summary=summary, contests=results,
                entries=sorted(entries, key=lambda e: (-e["payout"], e["rank"])), players=players,
                scorecard=bool(card is not None and len(card.rows)), notes=(card.summary.get("notes") if card is not None and card.summary else []) or [])


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x
