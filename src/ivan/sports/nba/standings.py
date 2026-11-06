# src/ivan/sports/nba/standings.py
"""
Real contest fields from DraftKings standings exports (raw dk/standings):
every entry's eight players, points and rank, and the player side table
(drafted % and actual fantasy points). These are what the field generator
is calibrated against and what the dupe and ownership models will be fitted
on.

contests_to_scrape   per played slate (archive capture with a lock in the
                     window), every contest we played (the slate's own entries
                     file, data/nba/slates/<dg>/input/entries.csv) plus the
                     largest public classic contests, minus those already
                     exported FINAL (a mid-slate export is scraped again)
scrape               the export through a logged-in Chrome profile
                     (ivan.sports.common.draftkings), raw dk/standings/<id>
derive_contest       the newest FINAL export parsed and matched by name to the
                     slate's capture (ids, salaries, teams) and to our last
                     projection before lock. An export taken while games were on
                     (the late swap's fetch) shows the unplayed slots as LOCKED;
                     it is not a field and is refused (export_is_final):
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
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core import archive, raw
from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.models import NBAProjectionCapture
from ivan.sports.nba.paths import NBA_DATA_DIR

Log = Callable[[str], None]
SPORT = "nba"
DERIVED_DIR = NBA_DATA_DIR / "derived" / "contests"
SALARY_CAP = 50000
SLOTS = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")
_SLOT_RX = re.compile(r"\b(PG|SG|SF|PF|C|G|F|UTIL)\b")
PUBLIC_MIN_ENTRIES = 1000
GRACE_HOURS = 5.0           # after lock: the last game must be over before the export is final
PARTIAL_MAX_SHARE = 0.02    # an export with more than this share of entries showing a LOCKED slot was taken mid-slate


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
        side = side[side.name != ""]
        # the export lists a player once per roster position he was played in (PG, G, UTIL ...): his ownership is the sum of
        # those rows (2026-10-23: Schroder 30.90 + 14.01 + 3.60 = 48.5%, the share of lineups holding him). Keeping one row
        # per name under-counted every multi-slot player and summed the column to ~465 instead of 800.
        side = (side.groupby("name", sort=False)
                .agg(position=("position", lambda x: "/".join(dict.fromkeys(str(v) for v in x if str(v)))), drafted_pct=("drafted_pct", "sum"), fpts=("fpts", "first"))
                .reset_index())
    return ent.reset_index(drop=True), side.reset_index(drop=True)


def locked_entries(ent: pd.DataFrame) -> int:
    """Entries with at least one LOCKED slot: players whose game had not started when the export was taken."""
    return int(ent.names.map(lambda ns: any(n is None for n in ns)).sum()) if len(ent) else 0


def export_is_final(ent: pd.DataFrame, max_share: float = PARTIAL_MAX_SHARE) -> bool:
    """A final export (every game over) shows every slot; the late swap's mid-slate fetch shows the unplayed ones as LOCKED.
    2026-10-24 .. 11-04: the nightly derived those fetches as fields (16,812 entries into 43 lineups), so the share is checked."""
    return len(ent) > 0 and locked_entries(ent) <= max_share * len(ent)


# ----------------------------------------------------------------------------- which contests
CONTEST_COLS = "contest_id, draft_group_id, name, date, entry_fee, contest_type, total_entries, multi_entry"


def _contest_query(where: str, table: str = "historical_contest_information") -> str:
    return f"select {CONTEST_COLS} from {table} where {where}"


def _rows(con, q: str, **params) -> List[SimpleNamespace]:
    from sqlalchemy import text
    try:
        return [SimpleNamespace(**dict(r._mapping)) for r in con.execute(text(q), params)]
    except Exception:   # noqa: BLE001 - a table this connection does not have (no hist attached, a slate file without the table)
        return []


def slate_contests(session: Session, draft_group: int) -> List[SimpleNamespace]:
    """
    The slate's DraftKings contests, wherever they live since the one-directory-per-slate split (2026-11-03): the slate's own
    database when this machine holds it, else the history file's pre-split copy (`_migrated_historical_contest_information`),
    else the session's own table (a test database). Rows have contest_id, draft_group_id, name, entry_fee, contest_type,
    total_entries, multi_entry.
    """
    from ivan.sports.nba import slates as SL
    dg = int(draft_group)
    if SL.holds(dg):
        with SL.engine(dg).connect() as con:
            rows = _rows(con, _contest_query("draft_group_id = :d"), d=dg)
        if rows:
            return rows
    for table in ("historical_contest_information", "hist._migrated_historical_contest_information"):
        rows = _rows(session, _contest_query("draft_group_id = :d", table), d=dg)
        if rows:
            return rows
    return []


def contest_row(session: Session, contest_id: int) -> Optional[SimpleNamespace]:
    """One contest by id, from the session's table, the history file's pre-split copy, or any held slate's database."""
    from ivan.sports.nba import slates as SL
    cid = int(contest_id)
    for table in ("historical_contest_information", "hist._migrated_historical_contest_information"):
        rows = _rows(session, _contest_query("contest_id = :c", table), c=cid)
        if rows:
            return rows[0]
    for dg in SL.held():
        with SL.engine(dg).connect() as con:
            rows = _rows(con, _contest_query("contest_id = :c"), c=cid)
        if rows:
            return rows[0]
    return None


def has_standings(contest_id: int) -> bool:
    return raw.latest(SPORT, "dk/standings", contest_id) is not None


def latest_export(contest_id: int) -> Optional[str]:
    """The newest stored standings export of the contest as text, or None."""
    art = raw.latest(SPORT, "dk/standings", contest_id)
    if art is None:
        return None
    text = raw.load(art.path, SPORT)
    return text.decode("utf-8", errors="replace") if isinstance(text, bytes) else text


def has_final_standings(contest_id: int) -> bool:
    """A stored export that is a final one (export_is_final); a mid-slate export does not count as scraped."""
    text = latest_export(contest_id)
    if text is None:
        return False
    try:
        return export_is_final(parse_export(text)[0])
    except ValueError:
        return False


def played_slates(session: Session, since: datetime, until: datetime, now: Optional[datetime] = None, grace_hours: float = GRACE_HOURS):
    """Archive captures (one per slate key) whose lock is in the window and at least grace_hours ago."""
    now = now or datetime.now()
    return [c for c in archive.slates(session, SPORT, since, until) if c.lock + timedelta(hours=grace_hours) <= now]


def entries_contest_ids(entries_csv: Optional[Path]) -> set:
    """The Contest IDs in a DraftKings entries file; empty when there is no file."""
    if not entries_csv or not Path(entries_csv).exists():
        return set()
    try:
        from ivan.sports.nba.lineup_export import read_entries
        return {int(c) for c in read_entries(entries_csv)["Contest ID"].unique()}
    except Exception:   # noqa: BLE001
        return set()


def entries_contest_counts(entries_csv: Optional[Path]) -> Dict[int, int]:
    """{Contest ID: our entries in it} from a DraftKings entries file; empty when there is no file."""
    if not entries_csv or not Path(entries_csv).exists():
        return {}
    try:
        from ivan.sports.nba.lineup_export import read_entries
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


# ----------------------------------------------------------------------------- scrape
def record_scrape(plan: List[dict], result: dict, now: Optional[datetime] = None, root: Optional[Path] = None) -> List[Path]:
    """Note the scrape's outcome on every held slate it touched: slates/<dg>/results/scrape.json = {at, contests, ok, failed,
    login_needed}. The readiness of a past day (ivan.sports.nba.readiness: the calendar's red) reads it; a later scrape that
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
        from ivan.sports.nba import readiness
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


def capture_for_contest(session: Session, contest):
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


def derive_contest(session: Session, contest_id: int, text: Optional[str] = None, out_dir: Path = DERIVED_DIR, log: Log = print,
                   allow_partial: bool = False) -> Optional[dict]:
    """The contest's derived field from its newest export (or `text`). A mid-slate export (export_is_final false) is refused
    unless allow_partial: it writes nothing and returns None, and a derived folder of this contest that itself came from a
    partial export is removed (it is not a field)."""
    contest = contest_row(session, int(contest_id))
    if contest is None:
        log(f"{contest_id}: not in historical_contest_information"); return None
    if text is None:
        text = latest_export(contest_id)
        if text is None:
            return None
    ent, side = parse_export(text)
    n_locked = locked_entries(ent)
    final = export_is_final(ent)
    if not final and not allow_partial:
        log(f"{contest_id} {contest.name}: the newest export is mid-slate ({n_locked:,} of {len(ent):,} entries show a LOCKED slot); "
            f"not a field, scrape it final")
        if remove_partial(contest_id, out_dir):
            log(f"{contest_id}: removed the derived folder built from a mid-slate export")
        return None
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
               "locked_entries": n_locked, "final": bool(final),
               "capture_id": cap.id, "capture_at": cap.captured_at.isoformat(timespec="minutes"),
               "top_owned": side.sort_values("drafted_pct", ascending=False).head(12)[["name", "drafted_pct", "fpts"]].to_dict("records"),
               "derived_at": datetime.now().isoformat(timespec="seconds")}
    (d / "field.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"{contest_id} {contest.name}: {summary['entries']:,} entries, {summary['entries_matched']:,} matched into {summary['unique']:,} unique lineups, "
        f"max copies {summary['max_copies']}, {len(unmatched)} unmatched names")
    return summary


def derived_ids(out_dir: Path = DERIVED_DIR) -> set:
    return {int(p.name) for p in Path(out_dir).glob("*") if p.is_dir() and (p / "lineups.parquet").exists() and p.name.isdigit()}


def derived_is_final(contest_id: int, out_dir: Path = DERIVED_DIR) -> bool:
    """Whether the contest's derived folder came from a final export. Folders written before the flag existed (2026-11-05) are
    judged by their short-entry share: a final export has (nearly) none."""
    fj = Path(out_dir) / str(contest_id) / "field.json"
    if not fj.exists():
        return False
    try:
        m = json.loads(fj.read_text())
    except ValueError:
        return False
    if "final" in m:
        return bool(m["final"])
    n = int(m.get("entries") or 0)
    return n > 0 and int(m.get("short_entries") or 0) <= PARTIAL_MAX_SHARE * n


def partial_derived_ids(out_dir: Path = DERIVED_DIR) -> set:
    return {c for c in derived_ids(out_dir) if not derived_is_final(c, out_dir)}


def remove_partial(contest_id: int, out_dir: Path = DERIVED_DIR) -> bool:
    """Remove the contest's derived folder when it came from a mid-slate export (never a final one). Returns whether it did."""
    import shutil
    d = Path(out_dir) / str(contest_id)
    if d.is_dir() and (d / "field.json").exists() and not derived_is_final(contest_id, out_dir):
        shutil.rmtree(d); return True
    return False


def derive_pending(session: Session, out_dir: Path = DERIVED_DIR, log: Log = print) -> List[dict]:
    """Every exported contest without a final derived folder: derived when its newest export is final; a partial folder is
    removed when its export is still mid-slate (derive_contest). Returns the summaries derived."""
    have = derived_ids(out_dir) - partial_derived_ids(out_dir)
    out, partial = [], []
    for a in raw.index(SPORT):
        if a.kind != "dk/standings" or int(a.key) in have:
            continue
        cid = int(a.key)
        if not has_final_standings(cid):
            partial.append(cid)
            if remove_partial(cid, out_dir):
                log(f"{cid}: removed the derived folder built from a mid-slate export")
            continue
        s = derive_contest(session, cid, out_dir=out_dir, log=log)
        if s:
            out.append(s); have.add(cid)
    partial = sorted(set(partial))
    if partial:
        log(f"{len(partial)} exported contest(s) have only a mid-slate export, not a field (scrape them final: scripts/nba/standings.py status): "
            + ", ".join(str(c) for c in partial[:12]) + (" ..." if len(partial) > 12 else ""))
    return out


def load_field(contest_id: int, out_dir: Path = DERIVED_DIR) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict]]:
    d = Path(out_dir) / str(contest_id)
    if not (d / "lineups.parquet").exists():
        return None, None, None
    summary = json.loads((d / "field.json").read_text()) if (d / "field.json").exists() else None
    return pd.read_parquet(d / "lineups.parquet"), pd.read_csv(d / "players.csv"), summary
