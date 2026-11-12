# src/ivan/core/dk_history.py
"""
DraftKings' own record of what we played: the entry-history export of the account (My Contests -> History ->
"Download Entry History", draftkings-contest-entry-history.csv), one row per entry of every sport with its fee and its
winnings. It is the money wherever it has the contest: a scraped standings page can be missing (DraftKings drops a
contest's standings after some weeks) or name a contest we chose for and never entered; this file cannot.

    parse(path)                 the export as one normalized row per entry, every sport
    import_file(path, since)    merge it into the store (data/dk/entry_history.csv, by entry id; a later export replaces)
    entries(sport)              the stored entries
    covered_until()             the time of the newest import: what DraftKings had not listed by then, we did not play
    contest_rows(sport)         {contest id: ledger row} (fees, won, net, cashed, gross_win, gross_loss, ... final)
    merge(sport, slates)        a sport's ledger slates with the history laid over them (see below)
    ledger(sport)               a ledger from the history alone (a sport with no results code of its own)
    sports()                    the sports the store holds

merge: a contest the history has takes the history's money (the row keeps its slate, name and day); a contest the
history has and the sport does not becomes a row of the slate `slate_of` names for it, else of a slate of its own day
("dk-<day>"); an unscored row (played by the chooser's word, no result) of a day the history covers and does not list is
dropped: we did not enter it.

    scripts/dk_history.py import [PATH] [--since 2026-01-01 | --all] | (no arguments: what is stored)
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ivan.core import pnl
from ivan.core.utils import PROJECT_ROOT

STORE_DIR = PROJECT_ROOT / "data" / "dk"
STORE = STORE_DIR / "entry_history.csv"
META = STORE_DIR / "entry_history.json"
EXPORT_GLOB = "draftkings-contest-entry-history*.csv"
DK_ZONE = "America/New_York"                 # the export's Contest_Date_EST
ALIASES = {
    "contest_id": ("contest_key", "contestkey", "contest_id", "contestid"),
    "entry_id": ("entry_key", "entrykey", "entry_id", "entryid"),
    "contest_name": ("entry", "contest", "contest_name", "contestname"),
    "game_type": ("game_type", "gametype"),
    "sport": ("sport",),
    "date": ("contest_date_est", "contest_date", "date", "contestdate"),
    "place": ("place", "rank"),
    "points": ("points", "fpts"),
    "winnings": ("winnings_non_ticket", "winnings", "winnings_cash"),
    "winnings_ticket": ("winnings_ticket",),
    "entry_fee": ("entry_fee", "entryfee", "fee"),
    "contest_entries": ("contest_entries", "entries", "total_entries"),
    "places_paid": ("places_paid", "placespaid"),
}
COLUMNS = ("contest_id", "entry_id", "contest_name", "game_type", "sport", "date", "place", "points", "entry_fee", "winnings",
           "contest_entries", "places_paid")


def parse_money(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    s = str(x).replace("$", "").replace(",", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s) if s else 0.0
    except ValueError:
        return 0.0
    return -v if neg else v


# ----------------------------------------------------------------------------- the export
def parse(path: str | Path) -> pd.DataFrame:
    """The export as one row per entry with COLUMNS; ticket winnings are added to cash winnings; the sport is "" when the file has no sport column."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    norm = {re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c for c in raw.columns}

    def col(key):
        for alias in ALIASES[key]:
            if alias in norm:
                return raw[norm[alias]]
        return None
    need = {k: col(k) for k in ("contest_id", "entry_id", "entry_fee", "winnings")}
    missing = [k for k, v in need.items() if v is None]
    if missing:
        raise ValueError(f"{path}: cannot find column(s) {missing}; have {list(raw.columns)}")
    ticket = col("winnings_ticket")
    df = pd.DataFrame({
        "contest_id": pd.to_numeric(need["contest_id"], errors="coerce"),
        "entry_id": pd.to_numeric(need["entry_id"], errors="coerce"),
        "contest_name": col("contest_name") if col("contest_name") is not None else "",
        "game_type": col("game_type") if col("game_type") is not None else "",
        "sport": col("sport").astype(str).str.strip() if col("sport") is not None else "",
        "date": pd.to_datetime(col("date"), errors="coerce") if col("date") is not None else pd.NaT,
        "place": pd.to_numeric(col("place"), errors="coerce") if col("place") is not None else np.nan,
        "points": pd.to_numeric(col("points"), errors="coerce") if col("points") is not None else np.nan,
        "entry_fee": need["entry_fee"].map(parse_money),
        "winnings": need["winnings"].map(parse_money) + (ticket.map(parse_money) if ticket is not None else 0.0),
        "contest_entries": pd.to_numeric(col("contest_entries"), errors="coerce") if col("contest_entries") is not None else np.nan,
        "places_paid": pd.to_numeric(col("places_paid"), errors="coerce") if col("places_paid") is not None else np.nan,
    })
    df = df.dropna(subset=["contest_id", "entry_id"]).copy()
    df["contest_id"] = df.contest_id.astype(int)
    df["entry_id"] = df.entry_id.astype(int)
    return df.reset_index(drop=True)


def newest_export(folder: Optional[Path] = None) -> Optional[Path]:
    """The newest draftkings-contest-entry-history*.csv in ~/Downloads (where the browser leaves it)."""
    files = sorted((folder or Path.home() / "Downloads").glob(EXPORT_GLOB), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


# ----------------------------------------------------------------------------- the store
def _read_store(store: Path) -> pd.DataFrame:
    if not store.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    df = pd.read_csv(store, dtype={"contest_name": str, "game_type": str, "sport": str}, keep_default_na=False, na_values=[""])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("contest_name", "game_type", "sport"):
        df[c] = df[c].fillna("").astype(str)
    return df


def import_file(path: str | Path, since: Optional[date] = None, store: Optional[Path] = None, now: Optional[datetime] = None) -> dict:
    """
    Merge an export into the store: its entries dated `since` or later (default: every entry), by entry id, the export's row
    replacing a stored one. Returns and writes the store's description (entry_history.json): imported_at (the export file's
    time: what the history covers), rows, first and last contest date, entries per sport.
    """
    path = Path(path)
    store = Path(store) if store else STORE
    df = parse(path)
    if since is not None:
        df = df[df.date.notna() & (df.date >= pd.Timestamp(since))]
    old = _read_store(store)
    merged = pd.concat([old[~old.entry_id.isin(df.entry_id)], df], ignore_index=True) if len(old) else df
    merged = merged.sort_values(["date", "contest_id", "entry_id"]).reset_index(drop=True)
    store.parent.mkdir(parents=True, exist_ok=True)
    merged[list(COLUMNS)].to_csv(store, index=False)
    exported = datetime.fromtimestamp(path.stat().st_mtime)
    prev = _meta(store)
    covered = max([exported] + ([datetime.fromisoformat(prev["covered_until"])] if prev.get("covered_until") else []))
    meta = dict(imported_at=(now or datetime.now()).isoformat(timespec="seconds"), source=str(path), covered_until=covered.isoformat(timespec="seconds"),
                since=since.isoformat() if since else None, rows=int(len(merged)), new_rows=int(len(merged) - len(old)), read_rows=int(len(df)),
                first=merged.date.min().isoformat() if len(merged) and merged.date.notna().any() else None,
                last=merged.date.max().isoformat() if len(merged) and merged.date.notna().any() else None,
                sports={str(k): int(v) for k, v in merged.sport.str.upper().value_counts().items()})
    store.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    _CACHE.clear()
    return meta


def _meta(store: Optional[Path] = None) -> dict:
    try:
        return json.loads((Path(store) if store else STORE).with_suffix(".json").read_text())
    except (OSError, ValueError):
        return {}


def covered_until(store: Optional[Path] = None) -> Optional[datetime]:
    """The time of the newest export imported: a contest locked before it that the history does not list, we did not enter."""
    v = _meta(store).get("covered_until")
    return datetime.fromisoformat(v) if v else None


_CACHE: Dict[tuple, pd.DataFrame] = {}


def entries(sport: Optional[str] = None, store: Optional[Path] = None) -> pd.DataFrame:
    """The stored entries (of one sport), read once per change of the store file."""
    store = Path(store) if store else STORE
    if not store.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    key = (str(store), store.stat().st_mtime_ns)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = _read_store(store)
    df = _CACHE[key]
    return df if sport is None else df[df.sport.str.upper() == str(sport).upper()]


def sports(store: Optional[Path] = None) -> List[str]:
    """The sports the store holds, lower case (the keys of the web's sports where they exist)."""
    return sorted({s.lower() for s in entries(store=store).sport.unique() if s})


# ----------------------------------------------------------------------------- the ledger
def _local(ts) -> Optional[datetime]:
    """The export's Eastern time as this machine's naive local time (the slates' days are local)."""
    if ts is None or pd.isna(ts):
        return None
    t = pd.Timestamp(ts).tz_localize(DK_ZONE, ambiguous="NaT", nonexistent="NaT")
    if pd.isna(t):                                   # the hour the clocks change: keep the wall time
        return pd.Timestamp(ts).to_pydatetime()
    return t.tz_convert(datetime.now().astimezone().tzinfo).tz_localize(None).to_pydatetime()


def contest_rows(sport: str, store: Optional[Path] = None) -> Dict[int, dict]:
    """{contest id: the contest as a ledger row} from the stored entries of one sport: every one final, the money DraftKings' own."""
    df = entries(sport, store)
    when = _meta(store).get("covered_until", "")[:10]
    out: Dict[int, dict] = {}
    for cid, g in df.groupby("contest_id"):
        fee = g.entry_fee.to_numpy(dtype=float); won = g.winnings.to_numpy(dtype=float)
        lock = _local(g.date.min())
        fees, total = float(fee.sum()), float(won.sum())
        field = g.contest_entries.max()
        out[int(cid)] = dict(
            contest_id=int(cid), name=str(g.contest_name.iloc[0]) or f"contest {cid}", entry_fee=float(fee.max()) if len(fee) else 0.0,
            mode=str(g.game_type.iloc[0]).lower(), field=int(field) if pd.notna(field) else 0, lock=lock, day=lock.date().isoformat() if lock else None,
            ours=int(len(g)), found=int(len(g)), final=True, fees=fees, won=total, net=total - fees, roi=(total - fees) / fees if fees else None,
            best_rank=int(g.place.min()) if g.place.notna().any() else None, best_points=float(g.points.max()) if g.points.notna().any() else None,
            cashed=int((won > 0).sum()), min_cash=None, gross_win=float(np.clip(won - fee, 0, None).sum()), gross_loss=float(np.clip(fee - won, 0, None).sum()),
            export_at=None, history=True, scored="DraftKings' entry history" + (f", exported {when}" if when else ""))
    return out


MONEY = ("ours", "found", "final", "fees", "won", "net", "roi", "cashed", "gross_win", "gross_loss", "scored", "history")


def merge(sport: str, slates: List[dict], slate_of: Optional[Dict[int, str]] = None, store: Optional[Path] = None) -> List[dict]:
    """
    A sport's ledger slates ([{"key", "day", "complete", "contests": [rows]}]) with the history laid over them (the module
    doc): returns new slates; the input is not changed. Without a store the slates come back as they are.
    """
    hist = contest_rows(sport, store)
    if not hist:
        return slates
    covered = covered_until(store)
    covered_day = covered.date().isoformat() if covered else ""
    used = set()
    out: Dict[str, dict] = {}
    for s in slates:
        rows = []
        for r in s["contests"]:
            h = hist.get(int(r["contest_id"]))
            if h is not None:
                used.add(int(r["contest_id"]))
                r = dict(r, **{k: h[k] for k in MONEY})
                for k in ("best_rank", "best_points", "field"):
                    r[k] = r.get(k) if r.get(k) else h[k]
            elif not r.get("final") and str(r.get("day") or s["day"]) < covered_day:
                continue                                              # DraftKings lists no entry of ours there: not entered
            rows.append(r)
        if rows:
            out[str(s["key"])] = dict(s, key=s["key"], contests=rows)
    for cid, h in hist.items():
        if cid in used or not h["day"]:
            continue
        key = (slate_of or {}).get(cid)
        if key is None:
            key = f"dk-{h['day']}"
        s = out.setdefault(str(key), dict(key=key, day=h["day"], complete=True, contests=[]))
        s["contests"].append(dict(h, draft_group=str(key)))
    for s in out.values():
        s["complete"] = all(r.get("final") for r in s["contests"])
    return list(out.values())


def ledger(sport: str, store: Optional[Path] = None) -> Optional[dict]:
    """A sport's ledger from the history alone (ivan.core.pnl.roll_up), None when the store has nothing of the sport."""
    slates = merge(sport, [], store=store)
    if not slates:
        return None
    out = pnl.roll_up(slates)
    out["source"] = SOURCE
    return out


SOURCE = "DraftKings' entry history of the account (scripts/dk_history.py import)"


