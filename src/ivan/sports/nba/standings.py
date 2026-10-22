# src/ivan/sports/nba/standings.py
"""
Real contest fields from DraftKings standings exports (raw dk/standings):
every entry's eight players, points and rank, and the player side table
(drafted % and actual fantasy points). These are what the field generator
is calibrated against and what the dupe and ownership models will be fitted
on.

contests_to_scrape   per played slate (archive capture with a lock in the
                     window), the largest public classic contests plus any
                     contest in our entries file, minus those already exported
scrape               the export through a logged-in Chrome profile
                     (ivan.sports.common.draftkings), raw dk/standings/<id>
derive_contest       the newest export parsed and matched by name to the
                     slate's capture (ids, salaries, teams) and to our last
                     projection before lock:
                       data/nba/derived/contests/<id>/lineups.parquet   one row per unique lineup
                       data/nba/derived/contests/<id>/players.csv       the side table joined to the pool
                       data/nba/derived/contests/<id>/field.json        summary
DraftKings keeps standings for about two weeks, so scrape within days.
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core import archive, raw
from ivan.core.utils import normalize_name
from ivan.sports.nba.models import HistoricalContest, NBAProjectionCapture
from ivan.sports.nba.paths import NBA_DATA_DIR

Log = Callable[[str], None]
SPORT = "nba"
DERIVED_DIR = NBA_DATA_DIR / "derived" / "contests"
SALARY_CAP = 50000
SLOTS = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")
_SLOT_RX = re.compile(r"\b(PG|SG|SF|PF|C|G|F|UTIL)\b")
PUBLIC_MIN_ENTRIES = 1000
GRACE_HOURS = 5.0           # after lock: the last game must be over before the export is final


# ----------------------------------------------------------------------------- the export
def parse_lineup(s: str) -> List[tuple]:
    """'PG Luka Doncic SG Amen Thompson ...' -> [(slot, name or None), ...] in the order written."""
    if not isinstance(s, str) or not s.strip():
        return []
    s = re.sub(r"\s+", " ", s).strip()
    ms = list(_SLOT_RX.finditer(s))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(s)
        val = s[m.end():end].strip().strip(",")
        out.append((m.group(1), None if (not val or val.upper() == "LOCKED") else val))
    return out


def parse_export(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entries, players): entries have rank, entry_id, entry_name, points, names (list of 8); players have name, position, drafted_pct, fpts."""
    df = pd.read_csv(io.StringIO(text.lstrip("﻿")), dtype=str, keep_default_na=False, low_memory=False)
    cols = {str(c).strip().lower().replace(" ", ""): c for c in df.columns}
    need = ["rank", "entryid", "entryname", "points", "lineup"]
    if any(k not in cols for k in need):
        raise ValueError(f"not a DraftKings standings export (columns {list(df.columns)[:8]})")
    ent = pd.DataFrame({"rank": pd.to_numeric(df[cols["rank"]], errors="coerce"), "entry_id": df[cols["entryid"]].str.strip(),
                        "entry_name": df[cols["entryname"]].str.strip(), "points": pd.to_numeric(df[cols["points"]], errors="coerce"),
                        "lineup": df[cols["lineup"]]})
    ent = ent[ent.lineup.str.strip() != ""].copy()
    ent["names"] = ent.lineup.map(lambda s: [n for _, n in parse_lineup(s)])
    side = pd.DataFrame(columns=["name", "position", "drafted_pct", "fpts"])
    if all(k in cols for k in ("player", "%drafted")):
        side = pd.DataFrame({"name": df[cols["player"]].str.strip(), "position": df[cols["rosterposition"]].str.strip() if "rosterposition" in cols else "",
                             "drafted_pct": pd.to_numeric(df[cols["%drafted"]].str.rstrip("%"), errors="coerce"),
                             "fpts": pd.to_numeric(df[cols["fpts"]], errors="coerce") if "fpts" in cols else np.nan})
        side = side[side.name != ""].drop_duplicates("name")
    return ent.reset_index(drop=True), side.reset_index(drop=True)


# ----------------------------------------------------------------------------- which contests
def has_standings(contest_id: int) -> bool:
    return raw.latest(SPORT, "dk/standings", contest_id) is not None


def played_slates(session: Session, since: datetime, until: datetime, now: Optional[datetime] = None, grace_hours: float = GRACE_HOURS):
    """Archive captures (one per slate key) whose lock is in the window and at least grace_hours ago."""
    now = now or datetime.now()
    return [c for c in archive.slates(session, SPORT, since, until) if c.lock + timedelta(hours=grace_hours) <= now]


def entries_contest_ids(entries_csv: Optional[Path]) -> set:
    if not entries_csv or not Path(entries_csv).exists():
        return set()
    try:
        from ivan.sports.nba.lineup_export import read_entries
        return {int(c) for c in read_entries(entries_csv)["Contest ID"].unique()}
    except Exception:   # noqa: BLE001
        return set()


def contests_to_scrape(session: Session, since: datetime, until: datetime, public_n: int = 3, entries_csv: Optional[Path] = None,
                       now: Optional[datetime] = None) -> List[dict]:
    """Per played slate: {slate_key, lock, contests: [ids], scraped: [ids], to_scrape: [ids]} with the largest public contests + ours."""
    ours = entries_contest_ids(entries_csv)
    out = []
    for cap in played_slates(session, since, until, now):
        rows = session.query(HistoricalContest).filter(HistoricalContest.draft_group_id == int(cap.slate_key)).all()
        pub = sorted([r for r in rows if (r.contest_type or "").lower().startswith("classic") and int(r.total_entries or 0) >= PUBLIC_MIN_ENTRIES],
                     key=lambda r: -int(r.total_entries or 0))[:public_n]
        ids = sorted({int(r.contest_id) for r in pub} | {int(r.contest_id) for r in rows if int(r.contest_id) in ours})
        scraped = [c for c in ids if has_standings(c)]
        out.append(dict(slate_key=cap.slate_key, lock=cap.lock, contests=ids, scraped=scraped, to_scrape=[c for c in ids if c not in scraped],
                        names={int(r.contest_id): r.name for r in rows if int(r.contest_id) in ids}))
    return out


# ----------------------------------------------------------------------------- scrape
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


# ----------------------------------------------------------------------------- derive
class NameIndex:
    """normalized name -> capture player row; a last-name fallback when unique on the slate."""

    def __init__(self, players: Iterable[dict]):
        self.by_name: Dict[str, dict] = {}
        self.by_last: Dict[str, List[dict]] = {}
        for p in players:
            n = normalize_name(str(p.get("name") or ""))
            if not n:
                continue
            self.by_name[n] = p
            self.by_last.setdefault(n.split()[-1], []).append(p)

    def find(self, name: str) -> Optional[dict]:
        n = normalize_name(name or "")
        if n in self.by_name:
            return self.by_name[n]
        cands = self.by_last.get(n.split()[-1] if n else "", [])
        if len(cands) == 1:
            return cands[0]
        first = n.split()[0] if n else ""
        cands = [c for c in cands if normalize_name(str(c.get("name"))).split()[0][:3] == first[:3]]
        return cands[0] if len(cands) == 1 else None


def capture_for_contest(session: Session, contest: HistoricalContest):
    """The slate's last archive capture before lock and its player rows."""
    caps = [c for c in archive.slates(session, SPORT, None, None) if str(c.slate_key) == str(contest.draft_group_id)]
    if not caps:
        return None, []
    cap = caps[-1]
    return cap, archive.load_blobs(session, cap.id).get("players", [])


def projection_at_lock(session: Session, draft_group: int) -> pd.DataFrame:
    """Our last 'ours' capture before the slate's lock: player_id -> minutes_proj, fpts_mean, fpts_std."""
    q = session.query(NBAProjectionCapture).filter(NBAProjectionCapture.draft_group_id == int(draft_group), NBAProjectionCapture.source == "ours")
    df = pd.read_sql(q.statement, session.get_bind())
    if df.empty:
        return df
    lock = pd.to_datetime(df["lock"].dropna().max()) if df["lock"].notna().any() else None
    before = df[df.captured_at <= lock] if lock is not None else df
    stamp = (before if len(before) else df).captured_at.max()
    return df[df.captured_at == stamp].set_index("player_id")[["player_name", "minutes_proj", "fpts_mean", "fpts_std", "salary", "starter"]]


def derive_contest(session: Session, contest_id: int, text: Optional[str] = None, out_dir: Path = DERIVED_DIR, log: Log = print) -> Optional[dict]:
    contest = session.get(HistoricalContest, int(contest_id))
    if contest is None:
        log(f"{contest_id}: not in historical_contest_information"); return None
    if text is None:
        art = raw.latest(SPORT, "dk/standings", contest_id)
        if art is None:
            return None
        text = raw.load(art.path, SPORT)
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
    ent, side = parse_export(text)
    cap, players = capture_for_contest(session, contest)
    if not players:
        log(f"{contest_id}: no archive capture for draft group {contest.draft_group_id}"); return None
    idx = NameIndex(players)
    by_id = {int(p["id"]): p for p in players}
    proj = projection_at_lock(session, int(contest.draft_group_id))
    cache: Dict[str, Optional[dict]] = {}

    def find(n):
        if n not in cache:
            cache[n] = idx.find(n)
        return cache[n]

    recs, unmatched, short = [], set(), 0
    for e in ent.itertuples():
        ids = []
        for n in e.names:
            if n is None:
                continue
            p = find(n)
            if p is None:
                unmatched.add(n)
            else:
                ids.append(int(p["id"]))
        if len(ids) != 8:
            short += 1; continue
        recs.append((tuple(sorted(ids)), float(e.points) if pd.notna(e.points) else np.nan, int(e.rank) if pd.notna(e.rank) else 0))
    lu = pd.DataFrame(recs, columns=["key", "points", "rank"])
    g = lu.groupby("key").agg(copies=("key", "size"), points=("points", "first"), best_rank=("rank", "min")).reset_index() if len(lu) else pd.DataFrame(columns=["key", "copies", "points", "best_rank"])
    g["salary"] = g.key.map(lambda k: sum(int(by_id[i].get("salary") or 0) for i in k))
    g["salary_left"] = SALARY_CAP - g.salary
    if len(proj):
        g["proj_sum"] = g.key.map(lambda k: float(sum(float(proj.fpts_mean.get(i, 0.0) or 0.0) for i in k)))
    for j in range(8):
        g[f"p{j + 1}"] = g.key.map(lambda k, j=j: k[j])
    g = g.drop(columns=["key"]).sort_values(["copies", "points"], ascending=[False, False]).reset_index(drop=True)

    side = side.copy()
    side["id"] = side.name.map(lambda n: (find(n) or {}).get("id"))
    side["team"] = side.id.map(lambda i: by_id.get(i, {}).get("team") if pd.notna(i) else None)
    side["salary"] = side.id.map(lambda i: by_id.get(i, {}).get("salary") if pd.notna(i) else None)
    if len(proj):
        side["minutes_proj"] = side.id.map(lambda i: proj.minutes_proj.get(i) if pd.notna(i) else None)
        side["fpts_proj"] = side.id.map(lambda i: proj.fpts_mean.get(i) if pd.notna(i) else None)
        side["starter"] = side.id.map(lambda i: proj.starter.get(i) if pd.notna(i) else None)

    d = Path(out_dir) / str(contest_id); d.mkdir(parents=True, exist_ok=True)
    g.to_parquet(d / "lineups.parquet", index=False)
    side.to_csv(d / "players.csv", index=False)
    summary = {"contest_id": int(contest_id), "name": contest.name, "draft_group_id": int(contest.draft_group_id), "entry_fee": contest.entry_fee,
               "entries": int(len(ent)), "entries_matched": int(g.copies.sum()) if len(g) else 0, "unique": int(len(g)), "short_entries": short,
               "unmatched_names": sorted(unmatched)[:40], "max_copies": int(g.copies.max()) if len(g) else 0,
               "capture_id": cap.id, "capture_at": cap.captured_at.isoformat(timespec="minutes"),
               "top_owned": side.sort_values("drafted_pct", ascending=False).head(12)[["name", "drafted_pct", "fpts"]].to_dict("records"),
               "derived_at": datetime.now().isoformat(timespec="seconds")}
    (d / "field.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"{contest_id} {contest.name}: {summary['entries']:,} entries, {summary['entries_matched']:,} matched into {summary['unique']:,} unique lineups, "
        f"max copies {summary['max_copies']}, {len(unmatched)} unmatched names")
    return summary


def derived_ids(out_dir: Path = DERIVED_DIR) -> set:
    return {int(p.name) for p in Path(out_dir).glob("*") if p.is_dir() and (p / "lineups.parquet").exists() and p.name.isdigit()}


def derive_pending(session: Session, out_dir: Path = DERIVED_DIR, log: Log = print) -> List[dict]:
    """Every exported contest without a derived folder."""
    have = derived_ids(out_dir)
    out = []
    for a in raw.index(SPORT):
        if a.kind == "dk/standings" and int(a.key) not in have:
            s = derive_contest(session, int(a.key), out_dir=out_dir, log=log)
            if s:
                out.append(s); have.add(int(a.key))
    return out


def load_field(contest_id: int, out_dir: Path = DERIVED_DIR) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict]]:
    d = Path(out_dir) / str(contest_id)
    if not (d / "lineups.parquet").exists():
        return None, None, None
    summary = json.loads((d / "field.json").read_text()) if (d / "field.json").exists() else None
    return pd.read_parquet(d / "lineups.parquet"), pd.read_csv(d / "players.csv"), summary
