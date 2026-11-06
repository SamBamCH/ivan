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
  our entries   the entries DraftKings lists under our user name (config backtest.dk_user: "SamBamCH (3/10)") plus the
                Entry IDs in the upload files the desk wrote for the slate's contests (the slate's output/
                chosen_entries.classic.*.csv and lateswap.*.csv); their final lineups come from DraftKings' export.
                The money is ivan.sports.nba.results (compute), shared with the ledger the picker and home page sum
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
                the desk (filed under data/nba/input/edits/<dg>/<TEAM>.json when the slate was left; the live file while the
                slate is still loaded), else the grid row the archive at tip kept (rotation_cells, kept from 2026-10-29),
                else our minutes only; agreement = the share of the 240 player-minutes on the floor in both
  by tip        late news moves the projection after the first lock, so each tip's games are also judged on the newest
                capture taken by that tip (ours or at_tip, up to TIP_GRACE after it) beside the pre-lock one: minutes and
                points MAE, bias, and the players the late news moved most
  ROI timeline  what our entries were expected to make at each point: the chooser's claim before lock, each late swap
                run's expected payout for the entries as held and as swapped (lateswap.<stamp>.json), the final result
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
from ivan.sports.nba import results as R
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


# the money is ivan.sports.nba.results'; the names stay importable here (tests, the web)
from ivan.sports.nba.results import payout_table, ranked_payouts, score_contest, summarize, unscored  # noqa: E402,F401


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


TIP_GRACE = pd.Timedelta(minutes=12)


def by_tip(session: Session, dg: int, box_rows: pd.DataFrame) -> List[dict]:
    """Per tip: the games' players judged on the pre-lock projection and on the newest capture by that tip (see the module doc)."""
    caps = pd.read_sql(text("select captured_at, source, lock, gametime, player_name, team, minutes_proj, fpts_mean from nba_projection_capture "
                            "where draft_group_id = :d and source in ('ours', 'at_tip')").bindparams(d=int(dg)), session.get_bind())
    if caps.empty or box_rows.empty:
        return []
    for c in ("captured_at", "lock", "gametime"):
        caps[c] = pd.to_datetime(caps[c])
    lock = caps["lock"].dropna().min() if caps["lock"].notna().any() else caps.gametime.min()
    pre = caps[(caps.source == "ours") & (caps.captured_at <= lock)]
    if pre.empty:
        pre = caps[caps.captured_at == caps.captured_at.min()]
    pre = pre[pre.captured_at == pre.captured_at.max()].drop_duplicates("player_name", keep="last").set_index("player_name")
    act = box_rows.set_index("player_name")
    out = []
    for tip, g in caps.groupby("gametime"):
        if pd.isna(tip):
            continue
        upto = caps[(caps.captured_at <= tip + TIP_GRACE) & caps.player_name.isin(g.player_name.unique())]
        last = upto[upto.captured_at == upto.captured_at.max()].drop_duplicates("player_name", keep="last").set_index("player_name") if len(upto) else None
        names = [n for n in g.player_name.unique() if n in act.index and n in pre.index]
        rows = []
        for n in names:
            a = act.loc[n]
            if isinstance(a, pd.DataFrame):
                a = a.iloc[0]
            t = last.loc[n] if last is not None and n in last.index else pre.loc[n]
            rows.append(dict(name=n, team=str(pre.loc[n, "team"]), min_act=float(a.min_act or 0), fpts_act=float(a.fpts_act or 0),
                             min_pre=_f(pre.loc[n, "minutes_proj"]) or 0.0, fpts_pre=_f(pre.loc[n, "fpts_mean"]) or 0.0,
                             min_tip=_f(t["minutes_proj"]) or 0.0, fpts_tip=_f(t["fpts_mean"]) or 0.0))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        rot = df[(df.min_pre >= 10) | (df.min_tip >= 10) | (df.min_act >= 10)]
        df["moved"] = df.fpts_tip - df.fpts_pre
        out.append(dict(tip=tip.to_pydatetime(), at=(last.captured_at.max().to_pydatetime() if last is not None and len(last) else None), n=len(df),
                        teams=sorted(df.team.unique()),
                        min_mae_pre=float((rot.min_pre - rot.min_act).abs().mean()) if len(rot) else None,
                        min_mae_tip=float((rot.min_tip - rot.min_act).abs().mean()) if len(rot) else None,
                        fpts_mae_pre=float((rot.fpts_pre - rot.fpts_act).abs().mean()) if len(rot) else None,
                        fpts_mae_tip=float((rot.fpts_tip - rot.fpts_act).abs().mean()) if len(rot) else None,
                        fpts_bias_pre=float((rot.fpts_pre - rot.fpts_act).mean()) if len(rot) else None,
                        fpts_bias_tip=float((rot.fpts_tip - rot.fpts_act).mean()) if len(rot) else None,
                        moved=df.reindex(df.moved.abs().sort_values(ascending=False).index).head(8)[lambda d: d.moved.abs() >= 1.0].to_dict("records")))
    return sorted(out, key=lambda r: r["tip"])


def roi_timeline(results: List[dict], lock: Optional[datetime], day: Optional[date], out_dir: Optional[Path] = None) -> List[dict]:
    """What our entries were expected to make at each point of the night (see the module doc), against the fees."""
    out_dir = Path(out_dir) if out_dir else S.output_dir()          # the slate's output directory (the current slate)
    fees = sum(r["fees"] for r in results)
    if not fees:
        return []
    cids = {str(r["contest_id"]) for r in results}
    fee_of = {str(r["contest_id"]): float(r["entry_fee"]) for r in results}
    ours = {e["entry_id"] for r in results for e in r["entries"]}
    out = []
    claimed = [(r["claim"]["roi"], r["fees"], r["claim"]["at"]) for r in results if r.get("claim") and r["claim"].get("roi") is not None]
    if claimed:
        out.append(dict(at=max(c[2] for c in claimed), what="the chooser's claim before lock", games=None,
                        roi=sum(a * w for a, w, _ in claimed) / sum(w for _, w, _ in claimed), roi_swapped=None, entries=None))
    for f in sorted(Path(out_dir).glob("lateswap.*.json")):
        t = _stamp_of(f)
        if t is None or day is None or t.date() != day:
            continue
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        held = swapped = cost = 0.0; n = 0
        for c in d.get("contests") or []:
            if str(c.get("contest_id")) not in cids:
                continue
            fee = fee_of[str(c["contest_id"])]
            for e in c.get("entries") or []:
                if str(e.get("entry_id")) not in ours:
                    continue
                o = (e.get("original") or {}).get("exp_payout"); b = (e.get("best") or {}).get("exp_payout")
                if o is None:
                    continue
                held += float(o); swapped += float(b if b is not None else o); cost += fee; n += 1
        if n:
            out.append(dict(at=datetime.fromisoformat(d["when"]) if d.get("when") else t, what=f"late swap: {d.get('games_started')} of {d.get('games_total')} games started",
                            games=d.get("games_started"), roi=held / cost - 1, roi_swapped=swapped / cost - 1, entries=n))
    won = sum(r["won"] for r in results)
    out.append(dict(at=None, what="final", games=None, roi=won / fees - 1, roi_swapped=None, entries=sum(len(r["entries"]) for r in results)))
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
                                starter=str(getattr(r, "starter", "") or ""), note=getattr(r, "note", None) if isinstance(getattr(r, "note", None), str) else None,
                                exposure=(exp[n] / len(entries)) if entries else None, own=own.get(n), own_proj=own_proj.get(n),
                                salary=_f(getattr(r, "salary", None))))
            players[str(t)] = lst
    from ivan.sports.nba.validation import actuals_for
    tips = by_tip(session, int(dg), actuals_for(session, day)) if day else []
    timeline = roi_timeline(scored, lock, day)
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
    field = []
    if own:
        allp = {p["name"]: dict(p, team=t) for t, lst in players.items() for p in lst}
        for n, o in sorted(own.items(), key=lambda kv: -kv[1])[:15]:
            p = allp.get(n, {})
            field.append(dict(name=n, team=p.get("team"), own=o, own_proj=own_proj.get(n), exposure=(exp[n] / len(entries)) if entries else None,
                              fpts_act=p.get("fpts_act"), fpts_proj=p.get("fpts_proj")))
    for r in results:
        r.pop("side", None)
    return dict(draft_group=int(dg), day=day, lock=lock, mode=mode_for(day, lock, now), summary=summary, contests=results,
                entries=sorted(entries, key=lambda e: (-e["payout"], e["rank"])), players=players, rotations=rotations, field=field, tips=tips, timeline=timeline, field_contest=biggest["name"] if biggest else None,
                scorecard=bool(card is not None and len(card.rows)), notes=(card.summary.get("notes") if card is not None and card.summary else []) or [])


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x
