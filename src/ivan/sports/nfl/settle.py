# src/ivan/sports/nfl/settle.py
"""
Settle the slates we played: after lock, fetch what the backtest needs and
write a status row per slate (ivan.core.slates).

For every archived slate (an ingest or backfill capture) whose lock passed
more than `grace` hours ago:

  played     the contests the chooser entered (chosen_entries.choose.<stamp>.json / web chosen.choose.<stamp>.json
             in the output directories), by draft group
  public     the draft group's largest contests of the slate's mode, even when
             we did not play them (the field models learn from public fields)
  scrape     every played or public contest without standings: through
             scripts/nfl/get_contest_data.py (the saved Chrome profile, with
             our username so our finishes are recorded); a 403 marks the
             slate login_needed instead of retrying forever
  derive     for every raw standings export (raw dk/standings) without derived
             files: lineups.parquet and scores.npy under data/nfl/derived/contests/<id>/
  box scores the week's nfl_stats rows for the slate's teams (fetch_actuals.py
             when a week is missing)

    python scripts/nfl/settle.py [--days 3 | --since --until] [--username] [--public 3] [--no-scrape] [--headless] [--dry-run]

Everything here is idempotent: run it again and it only does what is still
missing.
"""
from __future__ import annotations

import glob
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import text

from ivan.core import archive, raw, slates as slate_status
from ivan.core.utils import PROJECT_ROOT
from ivan.database.session import read_sql
from ivan.sports.nfl import actuals
from ivan.sports.nfl.backtest_tax import utc_to_local
from ivan.sports.nfl.paths import NFL_DATA_DIR, OUTPUT_DIR

SPORT = "nfl"
SCRIPTS = PROJECT_ROOT / "scripts" / "nfl"
DERIVED_DIR = NFL_DATA_DIR / "derived" / "contests"
MODE_TYPES = {"showdown": ("Showdown Captain Mode",), "classic": ("Classic",)}
GRACE_HOURS = 6
PUBLIC_MIN_ENTRIES = 5000


@dataclass
class SlateSettlement:
    slate_key: str
    mode: str
    lock: datetime
    captured_at: Optional[datetime]
    teams: List[str]
    played: List[int] = field(default_factory=list)
    public: List[int] = field(default_factory=list)
    scraped: List[int] = field(default_factory=list)
    to_scrape: List[int] = field(default_factory=list)
    box_scores: bool = False
    season_week: Optional[tuple] = None
    our_entries: int = 0
    status: str = "open"
    missing: List[str] = field(default_factory=list)

    @property
    def contests(self) -> List[int]:
        return sorted(set(self.played) | set(self.public))


# --------------------------------------------------------------------------- what we played
def played_contests(dirs: Sequence[Path] = (OUTPUT_DIR, OUTPUT_DIR / "web"), since: Optional[datetime] = None) -> Dict[int, str]:
    """contest id -> stamp of the choose run that entered it (the latest), from the chooser's json outputs."""
    out: Dict[int, str] = {}
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "chosen*.choose.*.json"))):   # chosen_entries.* (CLI) and chosen.* (web)
            stamp = Path(f).name.split(".")[2]
            try:
                when = datetime.strptime(stamp, "%Y%m%d-%H%M%S")
            except ValueError:
                continue
            if since is not None and when < since:
                continue
            try:
                for cid in json.load(open(f)):
                    out[int(cid)] = stamp
            except (OSError, ValueError):
                continue
    return out


def _contest_rows(session, since: datetime, until: datetime) -> pd.DataFrame:
    q = text("""SELECT h.contest_id, h.draft_group_id, h.name, h.date, h.entry_fee, h.contest_type, h.total_entries,
                       (SELECT count(*) FROM contest_field_scores f WHERE f.contest_id = h.contest_id) AS n_field,
                       (SELECT count(*) FROM showdown_lineup l WHERE l.contest_id = h.contest_id) AS n_lineups,
                       (SELECT count(*) FROM contest_entry_result e WHERE e.contest_id = h.contest_id) AS n_ours
                FROM historical_contest_information h WHERE h.date BETWEEN :a AND :b""")
    df = read_sql(session, q.bindparams(a=(since - timedelta(days=1)).isoformat(sep=" "), b=(until + timedelta(days=1)).isoformat(sep=" ")))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df.date)
    df["lock_local"] = [utc_to_local(d.to_pydatetime()) for d in df.date]
    return df


def has_standings(session, contest_id: int, row=None) -> bool:
    if raw.latest(SPORT, "dk/standings", contest_id) is not None:
        return True
    if row is not None:
        return int(row.n_field) > 0 or int(row.n_lineups) > 0
    return False


# --------------------------------------------------------------------------- derive from raw standings
_SLOT_RX = re.compile(r"\b(CPT|FLEX|QB|RB|WR|TE|DST|K|UTIL|G|F|C|PG|SG|SF|PF|P|1B|2B|3B|SS|OF)\b")


def split_lineup(s: str) -> List[str]:
    """'CPT Josh Allen FLEX Stefon Diggs ...' -> ['CPT Josh Allen', 'FLEX Stefon Diggs', ...]."""
    if not isinstance(s, str) or not s.strip():
        return []
    parts = _SLOT_RX.split(s)
    out, i = [], 1
    while i < len(parts) - 1:
        out.append(f"{parts[i]} {parts[i + 1].strip()}".strip())
        i += 2
    return out


def derive_contest(contest_id: int, csv_text: Optional[str] = None, out_dir: Path = DERIVED_DIR) -> Optional[dict]:
    """lineups.parquet (rank, entry_id, entry_name, points, slot columns) and scores.npy (descending float32) from a standings export."""
    if csv_text is None:
        art = raw.latest(SPORT, "dk/standings", contest_id)
        if art is None:
            return None
        csv_text = raw.load(art.abspath)
    df = pd.read_csv(io.StringIO(csv_text))
    base = df.iloc[:, :-4] if df.shape[1] > 6 else df
    cols = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in base.columns}
    pts_col, lineup_col = cols.get("points"), cols.get("lineup")
    if pts_col is None:
        return None
    d = pd.DataFrame({"rank": pd.to_numeric(base[cols["rank"]], errors="coerce") if cols.get("rank") else np.arange(1, len(base) + 1),
                      "entry_id": pd.to_numeric(base[cols["entryid"]], errors="coerce") if cols.get("entryid") else np.nan,
                      "entry_name": base[cols["entryname"]].astype(str) if cols.get("entryname") else "",
                      "points": pd.to_numeric(base[pts_col], errors="coerce")})
    d = d[d.points.notna()].reset_index(drop=True)
    if lineup_col is not None:
        slots = [split_lineup(s) for s in base.loc[d.index if len(d) == len(base) else base.index[base[pts_col].notna()], lineup_col]]
        width = max((len(s) for s in slots), default=0)
        for k in range(width):
            d[f"slot{k + 1}"] = [s[k] if k < len(s) else None for s in slots]
    out = Path(out_dir) / str(contest_id)
    out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / "lineups.parquet", index=False)
    scores = np.sort(d.points.to_numpy(dtype=np.float32))[::-1]
    np.save(out / "scores.npy", scores)
    meta = {"contest_id": int(contest_id), "entries": int(len(d)), "slots": int(width) if lineup_col is not None else 0,
            "derived_at": datetime.now().isoformat(timespec="seconds")}
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


def derive_from_lineups(session, contest_ids: Optional[Sequence[int]] = None, out_dir: Path = DERIVED_DIR, echo=print) -> int:
    """
    Derived field files for scraped showdown contests that predate the raw standings exports
    (the 2025 fields): one row per entry from showdown_lineup x dupe_count, players named through
    the slate's capture (matched by slot, team, position and salary, since the old scraper's player
    ids are its own). Skips contests that already have derived files. Returns how many were derived.
    """
    from ivan.core.utils import normalize_name
    q = """SELECT DISTINCT l.contest_id, h.draft_group_id FROM showdown_lineup l JOIN historical_contest_information h USING (contest_id)"""
    cands = read_sql(session, text(q))
    if contest_ids is not None:
        cands = cands[cands.contest_id.isin([int(c) for c in contest_ids])]
    caps = {c.slate_key: c for c in archive.slates(session, SPORT)}
    names_by_cap: Dict[int, dict] = {}
    done = 0
    for cid, dg in zip(cands.contest_id.astype(int), cands.draft_group_id):
        out = Path(out_dir) / str(cid)
        if (out / "meta.json").exists():
            continue
        cap = caps.get(str(dg))
        if cap is None:
            echo(f"[settle] derive {cid}: no capture for draft group {dg}; skipped")
            continue
        if cap.id not in names_by_cap:
            names_by_cap[cap.id] = {(int(pl["roster_slot_id"]), str(pl.get("team")), str(pl.get("position", "")).upper(), int(pl.get("salary") or 0)): str(pl.get("name", ""))
                                    for pl in archive.load_blobs(session, cap.id).get("players", [])}
        names = names_by_cap[cap.id]
        sp = read_sql(session, text("SELECT player_id, roster_slot_id AS slot, team, position, salary FROM showdown_player WHERE contest_id = :c").bindparams(c=cid))
        tok = {}
        for r in sp.itertuples():
            nm = names.get((int(r.slot), r.team, str(r.position).upper(), int(r.salary)))
            if nm:
                tok[int(r.player_id)] = ("CPT " if int(r.slot) == 511 else "FLEX ") + normalize_name(nm)
        lus = read_sql(session, text("SELECT cpt_player_id, flex_player_ids, dupe_count FROM showdown_lineup WHERE contest_id = :c").bindparams(c=cid))
        rows, missed = [], 0
        for r in lus.itertuples():
            pids = [int(r.cpt_player_id)] + [int(x) for x in (json.loads(r.flex_player_ids) if isinstance(r.flex_player_ids, str) else list(r.flex_player_ids))]
            toks = [tok.get(pid) for pid in pids]
            if any(t is None for t in toks) or len(toks) != 6:
                missed += 1
                continue
            for _ in range(int(r.dupe_count or 1)):
                rows.append(toks)
        if not rows:
            echo(f"[settle] derive {cid}: no lineup could be named ({len(lus)} lineups); skipped")
            continue
        df = pd.DataFrame(rows, columns=[f"slot{i}" for i in range(1, 7)])
        df.insert(0, "points", np.nan); df.insert(0, "entry_name", ""); df.insert(0, "entry_id", np.arange(1, len(df) + 1)); df.insert(0, "rank", 0)
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "lineups.parquet", index=False)
        (out / "meta.json").write_text(json.dumps({"contest_id": int(cid), "entries": int(len(df)), "slots": 6, "source": "showdown_lineup",
                                                   "lineups_unnamed": int(missed), "derived_at": datetime.now().isoformat(timespec="seconds")}))
        done += 1
        echo(f"[settle] derived contest {cid} from the lineup table: {len(df)} entries, {len(lus)} lineups" + (f", {missed} unnamed" if missed else ""))
    return done


def derive_missing(contest_ids: Optional[Sequence[int]] = None, out_dir: Path = DERIVED_DIR, echo=print) -> int:
    """Derive every raw standings export that has no derived files yet (or only the given ids). Returns how many were derived."""
    arts = raw.find(SPORT, "dk/standings")
    want = {str(c) for c in contest_ids} if contest_ids is not None else None
    done = 0
    seen = set()
    for a in arts:
        if a.key in seen or (want is not None and a.key not in want):
            continue
        seen.add(a.key)
        if (Path(out_dir) / a.key / "meta.json").exists():
            continue
        try:
            m = derive_contest(int(a.key), out_dir=out_dir)
        except Exception as e:                                   # a malformed export must not stop the loop
            echo(f"[settle] derive {a.key} failed: {type(e).__name__}: {e}")
            continue
        if m:
            done += 1
            echo(f"[settle] derived contest {a.key}: {m['entries']} entries, {m['slots']} slots")
    return done


# --------------------------------------------------------------------------- classic contests: player rows from the export
CLASSIC_SLOTS = {"QB": 66, "RB": 67, "WR": 68, "TE": 69, "DST": 71}
FLEX_SLOT = 70


def classic_player_table(csv_text: str) -> pd.DataFrame:
    """The right-hand table of a DK export: Player, Roster Position, %Drafted, FPTS (one row per player, ownership over all slots)."""
    df = pd.read_csv(io.StringIO(csv_text), low_memory=False)
    cols = {str(c).strip().lower().replace(" ", ""): c for c in df.columns}
    need = [cols.get("player"), cols.get("rosterposition"), cols.get("%drafted"), cols.get("fpts")]
    if any(c is None for c in need):
        return pd.DataFrame(columns=["name", "position", "ownership", "points"])
    t = df[need].copy(); t.columns = ["name", "position", "ownership", "points"]
    t = t.dropna(subset=["name"])
    t["ownership"] = pd.to_numeric(t.ownership.astype(str).str.rstrip("%"), errors="coerce")
    t["points"] = pd.to_numeric(t.points, errors="coerce")
    t["position"] = t.position.astype(str).str.upper().str.strip()
    return t.dropna(subset=["ownership"]).reset_index(drop=True)


def derive_classic_players(session, contest_id: int, cap_players: Sequence[dict], lock: datetime, csv_text: Optional[str] = None,
                           echo=print) -> int:
    """
    Per-player rows for a classic contest (the showdown_player table, which the
    validation and the ownership models read): actual ownership and points from
    the export's player table, matched by name to the slate's capture (id, team,
    salary and every roster slot the player had, FLEX included), with our
    projection and projected ownership from the projection snapshot of that
    lock. Replaces the contest's rows. Returns rows written.
    """
    from ivan.core.utils import normalize_name
    from ivan.sports.nfl.models import ShowdownPlayer
    if csv_text is None:
        art = raw.latest(SPORT, "dk/standings", contest_id)
        if art is None:
            return 0
        csv_text = raw.load(art.abspath)
    table = classic_player_table(csv_text)
    if table.empty:
        return 0
    by_name: Dict[str, List[dict]] = {}
    for p in cap_players:
        by_name.setdefault(normalize_name(str(p.get("name", ""))), []).append(p)
    snap = read_sql(session, text("""SELECT normalized_name, team, roster_slot_id, fpts_mean, ownership_proj, snapshot_date FROM projection_snapshot
                                     WHERE substr(snapshot_date, 1, 10) = :d ORDER BY snapshot_date""").bindparams(d=lock.strftime("%Y-%m-%d")))
    proj: Dict[tuple, tuple] = {}
    proj_any: Dict[str, tuple] = {}
    for r in snap.itertuples():
        proj[(str(r.normalized_name), int(r.roster_slot_id))] = (r.fpts_mean, r.ownership_proj)      # the latest snapshot of the day wins
        proj_any[str(r.normalized_name)] = (r.fpts_mean, r.ownership_proj)
    session.query(ShowdownPlayer).filter(ShowdownPlayer.contest_id == int(contest_id)).delete(synchronize_session=False)
    rows, unmatched = [], []
    seen_ids: set = set()
    for r in table.itertuples():
        key = normalize_name(str(r.name))
        cands = [p for p in by_name.get(key, []) if str(p.get("position", "")).upper() == r.position] or by_name.get(key, [])
        if not cands:
            unmatched.append(str(r.name))
            continue
        for p in cands:                                      # one row per roster slot the player had (position slot and FLEX)
            if int(p["id"]) in seen_ids:                     # two export names that resolve to one capture row
                continue
            seen_ids.add(int(p["id"]))
            fm, op = proj.get((key, int(p["roster_slot_id"])), proj_any.get(key, (None, None)))   # FLEX rows take the position slot's projection
            rows.append(ShowdownPlayer(contest_id=int(contest_id), player_id=int(p["id"]), roster_slot_id=int(p["roster_slot_id"]),
                                       salary=int(p.get("salary") or 0), position=str(p.get("position") or r.position), team=str(p.get("team") or ""),
                                       ownership_actual=float(r.ownership), ownership_proj=float(op) if op is not None else None,
                                       fpts_proj=float(fm) if fm is not None else None, fpts_actual=float(r.points) if pd.notna(r.points) else None))
    session.add_all(rows)
    session.commit()
    if unmatched:
        echo(f"[settle] contest {contest_id}: {len(unmatched)} export players not in the capture: {', '.join(unmatched[:6])}{'...' if len(unmatched) > 6 else ''}")
    return len(rows)


def derive_classic_missing(session, since: datetime, until: datetime, echo=print) -> int:
    """Player rows for every classic contest in the window that has a raw standings export but no player rows yet."""
    from ivan.sports.nfl.models import ShowdownPlayer
    contests = _contest_rows(session, since, until)
    if contests.empty:
        return 0
    caps = {c.slate_key: c for c in archive.slates(session, SPORT, since, until)}
    have = {int(c) for (c,) in session.query(ShowdownPlayer.contest_id).distinct().all()}
    done = 0
    for r in contests[contests.contest_type == "Classic"].itertuples():
        cid = int(r.contest_id)
        if cid in have or raw.latest(SPORT, "dk/standings", cid) is None:
            continue
        cap = caps.get(str(r.draft_group_id))
        if cap is None:
            continue
        players = archive.load_blobs(session, cap.id).get("players", [])
        try:
            n = derive_classic_players(session, cid, players, cap.lock, echo=echo)
        except Exception as e:
            echo(f"[settle] classic players for {cid} failed: {type(e).__name__}: {e}")
            session.rollback()
            continue
        if n:
            done += 1
            echo(f"[settle] contest {cid}: {n} player rows from the export")
    return done


# --------------------------------------------------------------------------- showdown points repair
def base_points_from_export(csv_text: str) -> Dict[str, float]:
    """normalized name -> base (FLEX) DraftKings points from a showdown export's player table (a CPT-only row divided by 1.5)."""
    from ivan.core.utils import normalize_name
    t = classic_player_table(csv_text)
    out: Dict[str, float] = {}
    pref: Dict[str, str] = {}
    for r in t.itertuples():
        if pd.isna(r.points):
            continue
        key = normalize_name(str(r.name))
        slot = str(r.position).upper()
        val = round(float(r.points) / 1.5, 2) if slot == "CPT" else float(r.points)
        if key not in out or (slot != "CPT" and pref.get(key) == "CPT"):
            out[key], pref[key] = val, slot
    return out


def repair_showdown_points(session, contest_id: int, cap_players: Sequence[dict], csv_text: Optional[str] = None, echo=print) -> int:
    """Rewrite showdown_player.fpts_actual for a contest from the raw export's base points (both slots). Returns rows updated."""
    from ivan.core.utils import normalize_name
    from ivan.sports.nfl.models import ShowdownPlayer
    if csv_text is None:
        art = raw.latest(SPORT, "dk/standings", contest_id)
        if art is None:
            return 0
        csv_text = raw.load(art.abspath)
    base = base_points_from_export(csv_text)
    if not base:
        return 0
    # the stored rows carry the snapshot's ids, not always the capture's: match on slot, team, position and salary
    name_of: Dict[tuple, str] = {}
    for p in cap_players:
        name_of[(int(p["roster_slot_id"]), str(p.get("team")), str(p.get("position", "")).upper(), int(p.get("salary") or 0))] = normalize_name(str(p.get("name", "")))
    n = 0
    for r in session.query(ShowdownPlayer).filter(ShowdownPlayer.contest_id == int(contest_id)).all():
        key = name_of.get((int(r.roster_slot_id), str(r.team), str(r.position).upper(), int(r.salary)))
        if key is None or key not in base:
            continue
        if r.fpts_actual is None or abs(float(r.fpts_actual) - base[key]) > 1e-6:
            r.fpts_actual = base[key]
            n += 1
    session.commit()
    return n


def repair_showdown_window(session, since: datetime, until: datetime, echo=print) -> int:
    """Repair every showdown contest in the window that has a raw export. Returns contests touched."""
    contests = _contest_rows(session, since, until)
    if contests.empty:
        return 0
    caps = {c.slate_key: c for c in archive.slates(session, SPORT, since, until)}
    done = 0
    for r in contests[contests.contest_type == "Showdown Captain Mode"].itertuples():
        cid = int(r.contest_id)
        if raw.latest(SPORT, "dk/standings", cid) is None:
            continue
        cap = caps.get(str(r.draft_group_id))
        if cap is None:
            continue
        players = archive.load_blobs(session, cap.id).get("players", [])
        n = repair_showdown_points(session, cid, players, echo=echo)
        if n:
            done += 1
            echo(f"[settle] contest {cid}: {n} player points corrected to base (FLEX) points")
    return done


# --------------------------------------------------------------------------- the assessment
def assess(session, since: datetime, until: datetime, *, public_n: int = 3, grace_hours: float = GRACE_HOURS,
           now: Optional[datetime] = None) -> List[SlateSettlement]:
    now = now or datetime.now()
    caps = archive.slates(session, SPORT, since, until)
    contests = _contest_rows(session, since, until)
    played = played_contests(since=since - timedelta(days=1))
    weeks = actuals.week_lookup(session)
    out = []
    for cap in caps:
        players = archive.load_blobs(session, cap.id).get("players", [])
        teams = sorted({str(p.get("team")) for p in players if p.get("team")})
        s = SlateSettlement(cap.slate_key, cap.mode, cap.lock, cap.captured_at, teams)
        if contests.empty:
            mine = contests
        else:
            mine = contests[contests.draft_group_id.astype(str) == str(cap.slate_key)]
            if mine.empty:
                mine = contests[(contests.lock_local - cap.lock).abs() <= timedelta(minutes=10)]
                mine = mine[mine.contest_type.isin(MODE_TYPES.get(cap.mode, ("Classic",)))]
        ids = set(int(c) for c in mine.contest_id) if not mine.empty else set()
        s.played = sorted(c for c in played if c in ids)
        pub = mine[(mine.contest_type.isin(MODE_TYPES.get(cap.mode, ("Classic",)))) & (mine.total_entries >= PUBLIC_MIN_ENTRIES)] if not mine.empty else mine
        s.public = [int(c) for c in pub.sort_values("total_entries", ascending=False).contest_id.head(public_n)] if not pub.empty else []
        by_id = {int(r.contest_id): r for r in mine.itertuples()} if not mine.empty else {}
        s.scraped = [c for c in s.contests if has_standings(session, c, by_id.get(c))]
        s.to_scrape = [c for c in s.contests if c not in s.scraped]
        s.our_entries = int(sum(int(by_id[c].n_ours) for c in s.played if c in by_id))
        s.season_week = actuals.week_for_date(session, cap.lock, weeks)
        s.box_scores = _box_scores_ok(session, s.season_week, teams)
        if cap.lock + timedelta(hours=grace_hours) > now:
            s.status = "open"
        elif not s.contests:
            s.status = "no_contests"
            s.missing.append("no contest of this draft group in the database (get_contest_ids.py runs before lock)")
        else:
            if s.to_scrape:
                s.missing.append(f"{len(s.to_scrape)} contest(s) without standings")
            if not s.box_scores:
                s.missing.append("box scores for the week not loaded")
            if s.played and s.our_entries == 0:
                s.missing.append("our entries' finishes not recorded (scrape with --username)")
            s.status = "settled" if not s.to_scrape and s.box_scores else "waiting"
        out.append(s)
    return out


def _box_scores_ok(session, season_week, teams: Sequence[str]) -> bool:
    if season_week is None:
        return False
    s, w = season_week
    df = read_sql(session, text("SELECT DISTINCT team FROM nfl_stats WHERE season = :s AND week = :w").bindparams(s=s, w=w))
    if df.empty:
        return False
    have = {actuals.dk_team(t) for t in df.team}
    return all(actuals.dk_team(t) in have for t in teams) if teams else True


def write_status(session, items: Sequence[SlateSettlement]) -> int:
    slate_status.ensure_tables(session.get_bind())
    for s in items:
        slate_status.upsert(session, SPORT, s.slate_key, mode=s.mode, lock=s.lock, captured_at=s.captured_at, played=s.played, public=s.public,
                            scraped=s.scraped, box_scores=s.box_scores, our_entries=s.our_entries, status=s.status, missing=s.missing)
    return len(items)


# --------------------------------------------------------------------------- the job
def scrape(ids: Sequence[int], username: Optional[str], headless: bool = False, echo=print) -> tuple[int, bool]:
    """Run the contest scraper for `ids`; returns (exit code, login_needed)."""
    cmd = [sys.executable, "-u", str(SCRIPTS / "get_contest_data.py"), "--contest-ids", ",".join(map(str, ids))]
    if username:
        cmd += ["--username", username]
    if headless:
        cmd.append("--headless")
    echo("[settle] " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in out.splitlines()[-30:]:
        echo("    " + line)
    low = out.lower()
    login_needed = ("403" in out and "forbidden" in low) or "login needed" in low or "authentication login" in low
    return proc.returncode, login_needed


def settle(session, since: datetime, until: datetime, *, username: Optional[str] = None, do_scrape: bool = True, headless: bool = False,
           fetch_box_scores: bool = True, public_n: int = 3, dry_run: bool = False, echo=print) -> List[SlateSettlement]:
    """Assess, fetch what is missing (box scores, standings), derive, re-assess, write the status rows."""
    items = assess(session, since, until, public_n=public_n)
    due = [s for s in items if s.status in ("waiting",)]
    echo(f"[settle] {len(items)} slate(s) {since:%Y-%m-%d}..{until:%Y-%m-%d}: " + ", ".join(f"{s.status} {sum(1 for x in items if x.status == s.status)}"
                                                                                           for s in {x.status: x for x in items}.values()))
    if dry_run:
        for s in items:
            echo(f"  {s.lock:%Y-%m-%d %H:%M} {s.mode:8s} {s.slate_key}: played {len(s.played)} public {len(s.public)} scraped {len(s.scraped)} "
                 f"to scrape {len(s.to_scrape)} box {s.box_scores} ours {s.our_entries} -> {s.status}")
        return items
    if fetch_box_scores and any(not s.box_scores for s in due):
        echo("[settle] fetching box scores")
        subprocess.run([sys.executable, "-u", str(SCRIPTS / "fetch_actuals.py")], cwd=str(PROJECT_ROOT))
    to_scrape = sorted({c for s in due for c in s.to_scrape})
    login_needed = False
    if do_scrape and to_scrape:
        rc, login_needed = scrape(to_scrape, username, headless=headless, echo=echo)
        if login_needed:
            echo("[settle] DraftKings refused the saved session (403): log in once in the Chrome profile and run again")
    n = derive_missing(echo=echo)
    if n:
        echo(f"[settle] derived {n} contest(s)")
    repair_showdown_window(session, since, until, echo=echo)
    n2 = derive_classic_missing(session, since, until, echo=echo)
    if n2:
        echo(f"[settle] classic player rows for {n2} contest(s)")
    items = assess(session, since, until, public_n=public_n)
    if login_needed:
        for s in items:
            if s.to_scrape and s.status == "waiting":
                s.status = "login_needed"
    write_status(session, items)
    for s in items:
        echo(f"  {s.lock:%Y-%m-%d %H:%M} {s.mode:8s} {s.slate_key}: {s.status}" + (f" ({'; '.join(s.missing)})" if s.missing else ""))
    return items
