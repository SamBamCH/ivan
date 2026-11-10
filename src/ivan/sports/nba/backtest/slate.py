"""
ivan.sports.nba.backtest.slate

One past slate, read only: how our entries did, what the sim claimed for them, and how every player went against our
projection at lock. Nothing here touches the working tables (the pool, lineups, sims), so a past slate is looked at
without loading it into the desk. The desk's backtest view (/?view=backtest&dg=...) and scripts/nba/backtest_slate.py
show the same.

    mode_for(day, first_lock, now)   which view a slate opens in: "backtest" (an earlier day), "lateswap" (today, after
                                      the first lock), "prelock" (before the first lock)
    slate_backtest(session, dg)      everything for one slate: summary, contests, players by team and as one table, the
                                      rotations as played over our grid
    contest_detail(session, dg, cid) one contest drilled into: our entries with their claims, the field's ownership against
                                      ours, the payout curve, the duplication
    flat_table(players, positions)   every player of the slate in one list, by points scored

Where the numbers come from:
  our entries   the entries DraftKings lists under our user name (config backtest.dk_user: "SamBamCH (3/10)") plus the
                Entry IDs in the upload files the desk wrote for the slate's contests (the slate's output/
                chosen_entries.classic.*.csv and lateswap.*.csv); their final lineups come from DraftKings' export.
                The money is ivan.sports.nba.backtest.results (compute), shared with the ledger the picker and home page sum
  the field     the contest's newest standings export (raw dk/standings/<id>), and only when it is FINAL: every game
                over, no LOCKED slot (standings.export_is_final). Points are DraftKings' own; ranks follow them, ties
                sharing the payouts of the places they cover. The late swap's fetch (taken minutes after a tip) shows
                every not-yet-tipped player as LOCKED, ours included, so it is never scored: such a contest shows as
                unscored until it is scraped final (scripts/nba/standings.py scrape). Nothing is rescored from the box
                scores (2026-11-04: eleven contests scored on one to four visible players read as a result)
  payouts       the payouts table (stored with the contest)
  the claim     the chooser's run for the contest closest before lock: chosen_roi and chosen_cash; per entry, the lineup's
                own simulated ROI and cash rate (runs from 2026-10-29 keep them, by_lineup), matched through the
                chooser's upload file (the entry's lineup at choose time, before any late swap)
  rotations     per team, the game's stints as a 48-minute grid (rotation.last_game_grid) over our grid: the one drawn on
                the desk (the slate's edits/rotations/<TEAM>.json), else the grid row the archive at tip kept
                (rotation_cells, kept from 2026-10-29), else our minutes only; agreement = the share of the 240
                player-minutes on the floor in both
  players       the scorecard join (validation.build_scorecard): our last projection before lock vs the box score,
                with the desk's notes that stood at lock; our exposure from our entries; the field's ownership from
                the largest contest's export (DK's %drafted); the position from the slate's pool when this machine holds it
  a contest     its own final export: our entries in it (rank, points, payout, copies, the claim), every player's
                ownership in that field against our projected ownership at the first tip and our share of our entries
                there, the payout curve (the places paid, what 1st, 10th ... the last paid place got), the duplication
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

from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.backtest import results as R
from ivan.sports.nba import slates as S

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


def our_entry_ids(contest_ids, out_dir: Optional[Path] = None) -> Dict[int, set]:
    """{contest id: our Entry IDs}, from every upload file the desk wrote (the chooser's and the late swap's)."""
    out_dir = Path(out_dir) if out_dir else S.output_dir()          # the slate's output directory (the current slate)
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


ID_RX = re.compile(r"\((\d+)\)")


def chosen_lineups(contest_ids, lock: Optional[datetime], out_dir: Optional[Path] = None) -> Dict[str, tuple]:
    """{entry id: sorted DK ids of the lineup the chooser gave it}: from the newest chooser upload file before lock that holds it."""
    out_dir = Path(out_dir) if out_dir else S.output_dir()          # the slate's output directory (the current slate)
    want = {int(c) for c in contest_ids}
    out: Dict[str, tuple] = {}
    files = sorted(Path(out_dir).glob("chosen_entries.classic.*.csv"), key=lambda f: _stamp_of(f) or datetime.min)
    for f in files:
        t = _stamp_of(f)
        if t is None or (lock is not None and t > lock):
            continue
        try:
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    try:
                        cid = int(str(r.get("Contest ID") or "").strip())
                    except ValueError:
                        continue
                    if cid not in want:
                        continue
                    ids = sorted(int(m) for k in ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL") for m in ID_RX.findall(str(r.get(k) or ""))[:1])
                    if len(ids) == 8:
                        out[str(r["Entry ID"]).strip()] = tuple(ids)
        except (OSError, csv.Error):
            continue
    return out


def claims(contest_ids, lock: Optional[datetime], out_dir: Optional[Path] = None) -> Dict[int, dict]:
    """{contest id: the chooser's claim}: the run closest before lock that covered the contest (any run when lock is unknown)."""
    out_dir = Path(out_dir) if out_dir else S.output_dir()          # the slate's output directory (the current slate)
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
                by = {tuple(sorted(int(x) for x in c.get("players") or [])): c for c in (v.get("by_lineup") or {}).values() if c.get("players")}
                best[int(cid)] = (t, dict(at=t, entries=v.get("entries"), roi=v.get("chosen_roi"), cash=v.get("chosen_cash"),
                                          method=(v.get("book") or {}).get("method"), by_players=by))
    return {k: v[1] for k, v in best.items()}


# the money is ivan.sports.nba.backtest.results'; the names stay importable here (tests, the web)
from ivan.sports.nba.backtest.results import payout_table, ranked_payouts, score_contest, summarize, unscored  # noqa: E402,F401


def archived_cells(session: Session, dg: int) -> tuple[Dict[str, Dict[str, list]], Optional[datetime]]:
    """{team: {player: grid row}} from the slate's last archive capture with grid rows, at or just after lock; and its time."""
    from ivan.core import archive
    caps = [c for c in archive.captures(session, "nba") if str(c.slate_key) == str(int(dg))]
    pre = [c for c in caps if c.lock is None or c.captured_at <= c.lock + pd.Timedelta(minutes=15)]
    for cap in sorted(pre, key=lambda c: c.captured_at, reverse=True):
        out: Dict[str, Dict[str, list]] = defaultdict(dict)
        for r in archive.load_blobs(session, cap.id).get("players", []):
            if r.get("rotation_cells"):
                out[str(r.get("team"))][normalize_name(r["name"])] = r["rotation_cells"]
        if out:
            return dict(out), cap.captured_at
    return {}, None


def our_grid(session: Session, dg: int, team: str, cells: Optional[tuple] = None) -> tuple[Optional[pd.DataFrame], str]:
    """(our grid for the team on that slate, where it came from): drawn on the desk, else the archive's grid rows, else None."""
    from ivan.sports.nba.projection.rotation import MINUTES
    paths = [S.paths(dg).rotations / f"{team}.json"]                      # the slate's own drawn grid (slates/<dg>/edits/rotations)
    for p in paths:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                g = pd.DataFrame(np.array(d["grid"], dtype=bool).reshape(len(d["players"]), MINUTES), index=d["players"], columns=list(range(MINUTES)))
                return g, f"drawn on the desk ({str(d.get('saved_at') or '')[5:16].replace('T', ' ')})"
            except (OSError, ValueError, KeyError):
                continue
    by_team, at = cells if cells is not None else archived_cells(session, dg)
    rows = by_team.get(team) or {}
    if rows:
        return pd.DataFrame({n: [bool(x) for x in v][:48] for n, v in rows.items()}).T, f"the projection at {at:%H:%M}"
    return None, "no grid of ours on file (none drawn; the archive kept grid rows from 2026-10-29)"


def grid_for_day(session: Session, team: str, day: date, root: Optional[Path] = None) -> tuple[Optional[pd.DataFrame], str]:
    """
    (our grid for the team on `day`, where it came from) whatever slate held the game: the grid drawn on the desk that day
    (data/nba/days/<day>/edits/rotations/<TEAM>.json, shared by the day's slates), else the grid rows of the newest archive
    capture at or just after the lock of a slate of that day whose pool has the team (rotation_cells, kept from 2026-10-29).
    None when neither exists. The team page's "vs our grid" toggle on a past game reads it (Desk.game_grid).
    """
    from ivan.core import archive
    from ivan.sports.nba.projection.rotation import MINUTES
    p = S.days_dir(root) / day.isoformat() / "edits" / "rotations" / f"{team}.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            g = pd.DataFrame(np.array(d["grid"], dtype=bool).reshape(len(d["players"]), MINUTES), index=d["players"], columns=list(range(MINUTES)))
            return g, f"drawn on the desk ({str(d.get('saved_at') or '')[5:16].replace('T', ' ')})"
        except (OSError, ValueError, KeyError):
            pass
    start = datetime.combine(day, datetime.min.time()); end = start + pd.Timedelta(days=1)
    caps = [c for c in archive.captures(session, "nba", since=start, until=end) if c.captured_at <= c.lock + pd.Timedelta(minutes=15)]
    for cap in sorted(caps, key=lambda c: c.captured_at, reverse=True):
        rows = {normalize_name(r["name"]): r["rotation_cells"] for r in archive.load_blobs(session, cap.id).get("players", [])
                if str(r.get("team")) == team and r.get("rotation_cells")}
        if rows:
            return pd.DataFrame({n: [bool(x) for x in v][:MINUTES] for n, v in rows.items()}).T, f"the projection at {cap.captured_at:%H:%M} (slate {cap.slate_key})"
    return None, "no grid of ours on file for that day (none drawn; the archive keeps grid rows from 2026-10-29)"


def rotation_overlay(session: Session, dg: int, team: str, day: date, min_proj: Dict[str, float], cells: Optional[tuple] = None) -> Optional[dict]:
    """The team's game as played over our grid: rows (actual and ours per minute), the agreement, where ours came from."""
    from ivan.sports.nba.projection import rotation as ROT
    gid = session.execute(text("select game_id from nba_stint where team = :t and game_date = :d limit 1").bindparams(t=team, d=day.isoformat())).scalar()
    if gid is None:
        return None
    last = ROT.last_game_grid(session, team, game_id=str(gid))
    if last is None:
        return None
    act = last["grid"]
    ours, src = our_grid(session, dg, team, cells)
    names = list(act.index) + [n for n in (ours.index if ours is not None else []) if n not in act.index and int(ours.loc[n].sum()) > 0]
    rows = []
    both = 0
    for n in names:
        a = [bool(x) for x in act.loc[n].values] if n in act.index else [False] * 48
        o = [bool(x) for x in ours.loc[n].values] if ours is not None and n in ours.index else None
        if o is not None:
            both += sum(1 for x, y in zip(a, o) if x and y)
        cells = [("both" if x and o[m] else "act" if x else "ours" if o[m] else "") if o is not None else ("act" if x else "") for m, x in enumerate(a)]
        rows.append(dict(name=n, cells=cells, min_act=sum(a), min_ours=sum(o) if o is not None else None, min_proj=min_proj.get(n)))
    return dict(team=team, game_id=str(gid), rows=rows, source=src, agreement=(both / 240.0) if ours is not None else None)


def projected_ownership(session: Session, dg: int) -> Dict[str, float]:
    """Our projected ownership at lock: the ownership column of the slate's first archive capture at a tip (else the last before lock)."""
    from ivan.core import archive
    caps = [c for c in archive.captures(session, "nba") if str(c.slate_key) == str(int(dg))]
    tip = sorted([c for c in caps if c.source == "at_tip"], key=lambda c: c.captured_at)
    pre = sorted([c for c in caps if c.lock is None or c.captured_at <= c.lock], key=lambda c: c.captured_at)
    for cap in (tip[:1] or pre[-1:]):
        rows = archive.load_blobs(session, cap.id).get("players", [])
        out = {normalize_name(r["name"]): float(r["ownership"]) for r in rows if r.get("ownership") not in (None, 0)}
        if out:
            return out
    return {}


def slate_backtest(session: Session, dg: int, now: Optional[datetime] = None) -> dict:
    from ivan.sports.nba.backtest import validation as V
    now = now or datetime.now()
    ours_cap, _, lock = V.captures_for(session, int(dg))
    day = V.slate_day(lock, ours_cap) if len(ours_cap) else None
    contests = [dict(contest_id=int(c), name=n, entry_fee=float(f or 0), total_entries=int(t or 0)) for c, n, f, t in session.execute(
        text("select contest_id, name, entry_fee, total_entries from historical_contest_information where draft_group_id = :d").bindparams(d=int(dg))).all()]
    if day is None and contests:
        d0 = session.execute(text("select min(date) from historical_contest_information where draft_group_id = :d").bindparams(d=int(dg))).scalar()
        day = pd.Timestamp(d0).date() if d0 else None
    ids = our_entry_ids([c["contest_id"] for c in contests])
    results = R.compute(int(dg), session, ours_ids=ids)                 # every contest we played: by our user name in the exports + the upload files
    cids = [r["contest_id"] for r in results]
    claim = claims(cids, lock)
    picked = chosen_lineups(cids, lock)
    for r in results:
        r["claim"] = claim.get(r["contest_id"])
        by = (r["claim"] or {}).get("by_players") or {}
        for e in r["entries"]:                                    # each entry's own claim, when the chooser's run kept them
            k = picked.get(e["entry_id"])
            cl = by.get(k) if k else None
            e["claim_roi"] = None if cl is None else cl.get("avg_roi")
            e["claim_cash"] = None if cl is None else cl.get("cash_pct")
        if r["claim"]:
            r["claim"].pop("by_players", None)
    summary = summarize(results)
    scored = [r for r in results if r["final"]]
    # players: the scorecard join, our exposure, the field's ownership (the largest final export)
    players: Dict[str, list] = {}
    card = V.build_scorecard(session, int(dg)) if len(ours_cap) else None
    entries = [e for r in scored for e in r["entries"]]
    exp = Counter(normalize_name(n) for e in entries for n in e["names"] if n)
    biggest = max((r for r in scored if r.get("side") is not None and len(r["side"])), key=lambda r: r["field"], default=None)
    own = {normalize_name(n): float(p) for n, p in zip(biggest["side"].name, biggest["side"].drafted_pct)} if biggest else {}
    own_proj = projected_ownership(session, int(dg))
    if card is not None and len(card.rows):
        rows = card.rows
        for t, g in rows.groupby("team"):
            lst = []
            for r in g.sort_values("minutes_proj", ascending=False).itertuples():
                n = r.player_name
                lst.append(dict(name=n, min_proj=_f(r.minutes_proj), min_act=_f(getattr(r, "min_act", None)), fpts_proj=_f(r.fpts_proj),
                                fpts_act=_f(getattr(r, "fpts_act", None)), err=_f(getattr(r, "fpts_err", None)), played=bool(getattr(r, "played", False)),
                                starter=str(getattr(r, "starter", "") or ""), status=str(getattr(r, "status", "") or "").upper().replace("NONE", ""),
                                priced=bool(getattr(r, "priced", False)) if getattr(r, "priced", None) is not None and getattr(r, "priced") == getattr(r, "priced") else False,
                                note=getattr(r, "note", None) if isinstance(getattr(r, "note", None), str) else None,
                                exposure=(exp[n] / len(entries)) if entries else None, own=own.get(n), own_proj=own_proj.get(n),
                                salary=_f(getattr(r, "salary", None))))
            players[str(t)] = lst
    rotations = {}
    if day is not None:
        cells = archived_cells(session, int(dg))
        for t, lst in players.items():
            try:
                ov = rotation_overlay(session, int(dg), t, day, {p["name"]: p["min_proj"] for p in lst}, cells)
            except Exception:   # noqa: BLE001 - a team without stints or an odd grid: no overlay
                ov = None
            if ov:
                rotations[t] = ov
    for r in results:
        r.pop("side", None)
        r.pop("entries", None)                                    # the contest's own panel (contest_detail) lists them
    return dict(draft_group=int(dg), day=day, lock=lock, mode=mode_for(day, lock, now), summary=summary, contests=results,
                players=players, table=flat_table(players, positions_of(session)), rotations=rotations, field_contest=biggest["name"] if biggest else None,
                scorecard=bool(card is not None and len(card.rows)))

def positions_of(session: Session) -> Dict[str, str]:
    """{normalized name: DraftKings position} from the slate's pool when the session reaches one (a held slate); {} otherwise."""
    try:
        rows = session.execute(text("select name, position from nba_player")).all()
    except Exception:   # noqa: BLE001 - the history engine of a slate not held has no pool
        return {}
    return {normalize_name(n): str(p or "") for n, p in rows if n}


def flat_table(players: Dict[str, list], positions: Optional[Dict[str, str]] = None) -> List[dict]:
    """Every player of the slate in one list, the most points scored first: the team's rows with the team, his position, the miss
    (scored minus projected) and his value as played (points per $1k)."""
    positions = positions or {}
    out = []
    for team, lst in players.items():
        for p in lst:
            d = (p["fpts_act"] - p["fpts_proj"]) if p.get("fpts_act") is not None and p.get("fpts_proj") is not None else None
            sal = p.get("salary")
            val = (p["fpts_act"] / sal * 1000.0) if sal and p.get("fpts_act") is not None else None
            out.append(dict(p, team=team, pos=positions.get(p["name"], ""), delta=d, value=val))
    out.sort(key=lambda r: (-(r["fpts_act"] or 0), -(r["fpts_proj"] or 0)))
    return out


CURVE_PLACES = (1, 2, 3, 5, 10, 25, 50, 100, 250, 1000, 5000)


def payout_curve(pay: np.ndarray) -> List[dict]:
    """A few rungs of the payout table: 1st, 2nd, 3rd, 5th, 10th ... and the last place paid, each with its payout."""
    paid = int((pay > 0).sum())
    if not paid:
        return []
    places = sorted({p for p in CURVE_PLACES if p < paid} | {paid})
    return [dict(place=p, payout=float(pay[p - 1])) for p in places]


def contest_detail(session: Optional[Session], dg: int, cid: int, lock: Optional[datetime] = None, players: Optional[Dict[str, dict]] = None,
                   export: Optional[tuple] = None, contest: Optional[dict] = None, pay: Optional[np.ndarray] = None, out_dir: Optional[Path] = None,
                   user: Optional[str] = None) -> dict:
    """
    One contest of a past slate, read only: the contest scored from its newest export (results.score_contest, final exports only;
    a mid-slate or missing export comes back unscored with the reason), our entries in it (rank, points, payout, copies in the
    field, the chooser's claim for the lineup), every player of the field by ownership (DK's %drafted in this contest, our
    projected ownership at the first tip, our share of our entries here, points projected and scored) with our own players
    merged in, the payout curve and the duplication. `players` is {normalized name: his row from slate_backtest} for the
    projection columns; `export` (stamp, text), `contest` and `pay` stand in for the raw store, the contests table and the payouts
    table in tests.
    """
    from ivan.sports.nba.data_clean.standings import parse_export
    from ivan.core import raw
    cid = int(cid)
    if contest is None:
        from ivan.sports.nba.data_fetch.contests import slate_contests
        row = next((r for r in slate_contests(session, int(dg)) if int(r.contest_id) == cid), None)
        contest = dict(contest_id=cid, name=row.name if row else f"contest {cid}", entry_fee=float(row.entry_fee or 0) if row else 0.0,
                       total_entries=int(row.total_entries or 0) if row else 0)
    ids = our_entry_ids([cid], out_dir).get(cid, set())
    if export is None:
        art = R.final_exports([cid]).get(cid)
        export = (art.stamp, raw.load(art.path, R.SPORT)) if art is not None else None
    if export is None:
        return dict(contest=R.unscored(contest, ids, "no standings export: scrape it (scripts/nba/standings.py scrape)"), field=[], curve=[], paid=0, claim=None)
    stamp, txt = export
    export_at = datetime.strptime(stamp, "%Y%m%d-%H%M%S") if stamp else None
    ent, side = parse_export(txt)
    if pay is None:
        pay = R.payout_table(session, cid, len(ent))
    r = R.score_contest(contest, ids, ent, pay, export_at, user)
    r.pop("side", None)
    claim = claims([cid], lock, out_dir).get(cid)
    picked = chosen_lineups([cid], lock, out_dir)
    by = (claim or {}).get("by_players") or {}
    for e in r["entries"]:
        k = picked.get(e["entry_id"])
        cl = by.get(k) if k else None
        e["claim_roi"] = None if cl is None else cl.get("avg_roi")
        e["claim_cash"] = None if cl is None else cl.get("cash_pct")
    if claim:
        claim = {k: v for k, v in claim.items() if k != "by_players"}
    players = players or {}
    n_ours = len(r["entries"])
    exp = Counter(normalize_name(n) for e in r["entries"] for n in e["names"] if n)
    own = {normalize_name(n): float(p) for n, p in zip(side.name, side.drafted_pct) if p == p} if len(side) else {}
    fpts_side = {normalize_name(n): float(f) for n, f in zip(side.name, side.fpts) if f == f} if len(side) else {}
    field = []
    for n in set(own) | set(exp):
        p = players.get(n, {})
        field.append(dict(name=n, team=p.get("team"), pos=p.get("pos"), salary=p.get("salary"), own=own.get(n), own_proj=p.get("own_proj"),
                          exposure=(exp[n] / n_ours) if n_ours else None, fpts_proj=p.get("fpts_proj"),
                          fpts_act=fpts_side.get(n, p.get("fpts_act")), played=p.get("played")))
    field.sort(key=lambda f: (-(f["own"] or 0), -(f["exposure"] or 0)))
    return dict(contest=r, field=field, curve=payout_curve(pay) if r["final"] else [], paid=int((pay > 0).sum()) if r["final"] else 0, claim=claim, n_ours=n_ours)



def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x
