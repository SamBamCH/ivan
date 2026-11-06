#!/usr/bin/env python3
"""
scripts/nba/normalize_names.py

Re-key the player names stored before the NBA name rule (ivan.sports.nba.names: the generational suffix dropped, the NBA
aliases) so every table and file meets on one spelling: 'wendell carter jr' -> 'wendell carter' (what the box scores,
Rotowire and the news now carry).

    scripts/nba/normalize_names.py              # dry run: what would change, per table and file
    scripts/nba/normalize_names.py --apply      # rewrite the history database, every held slate's database, store and edits

History tables: nba_stats, nba_news, nba_stint, nba_foul, nba_sub (player_in, player_out), nba_projection_capture,
nba_player_note. Per held slate: its database (nba_player.name, starting_lineups, nba_market_rate, lineup_pool.name,
projection_snapshot), its projection store (slates/<dg>/projection/<TEAM>.pkl: the frames' index, the rotation inputs),
and the day's edit files (data/nba/days/<day>/edits and published: minutes_overrides.csv, rotations/<TEAM>.json,
scenarios/). A past slate never refreshes, so its pool keeps the old spelling until this runs (2026-11-06: Wendell Carter's
history and minutes were gone from the 11-05 team page after the rule). A row whose new spelling already exists under the
same unique key is left as it is and listed. Idempotent: a second run changes nothing. Stop the desk first (serve.py)
when it runs on the live data, or reload the slate afterwards (the header's "new data · load").
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from pathlib import Path

import pandas as pd

from ivan.core.utils import normalize_name as core_normalize
from ivan.sports.nba import slates as S
from ivan.sports.nba.names import normalize_name

HISTORY_TABLES = [("nba_stats", "player_name"), ("nba_news", "player_name"), ("nba_stint", "player_name"), ("nba_foul", "player_name"),
                  ("nba_sub", "player_in"), ("nba_sub", "player_out"), ("nba_projection_capture", "player_name"), ("nba_player_note", "player_name")]
SLATE_TABLES = [("nba_player", "name"), ("starting_lineups", "player_name"), ("nba_market_rate", "player_name"), ("lineup_pool", "name"),
                ("projection_snapshot", "player_name")]


def maybe(s):
    """A normalized name that the NBA rule spells differently -> the new spelling; anything else (a team code, a note, a stat name,
    a name already keyed the NBA way) unchanged."""
    if not isinstance(s, str) or not s:
        return s
    if s != core_normalize(s):
        return s                                                     # not a normalized name (a team code, free text with case or punctuation)
    n = normalize_name(s)
    return n if n != s else s


def rekey_obj(x, hits: list):
    """Walk a record (dicts, lists, sets, DataFrames, Series) and re-key every name string in it (maybe); `hits` collects the moves."""
    if isinstance(x, str):
        y = maybe(x)
        if y != x:
            hits.append((x, y))
        return y
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            k2 = rekey_obj(k, hits) if isinstance(k, str) else k
            out[k2] = rekey_obj(v, hits)
        return out
    if isinstance(x, list):
        return [rekey_obj(v, hits) for v in x]
    if isinstance(x, tuple):
        return tuple(rekey_obj(v, hits) for v in x)
    if isinstance(x, (set, frozenset)):
        return type(x)(rekey_obj(v, hits) for v in x)
    if isinstance(x, pd.DataFrame):
        idx = [rekey_obj(i, hits) if isinstance(i, str) else i for i in x.index]
        if idx != list(x.index):
            x = x.copy(); x.index = pd.Index(idx, name=x.index.name)
            x = x[~x.index.duplicated(keep="last")]
        return x
    if isinstance(x, pd.Series):
        idx = [rekey_obj(i, hits) if isinstance(i, str) else i for i in x.index]
        if idx != list(x.index):
            x = x.copy(); x.index = pd.Index(idx, name=x.index.name)
            x = x[~x.index.duplicated(keep="last")]
        return x
    return x


def rekey(con: sqlite3.Connection, table: str, col: str, apply: bool, log=print) -> tuple[int, int, int]:
    """(distinct names changed, rows changed, names left for a conflict) in one table's column."""
    try:
        names = [r[0] for r in con.execute(f"select distinct {col} from {table}") if r[0]]
    except sqlite3.OperationalError:
        return 0, 0, 0                                               # no such table in this file
    moves = [(n, normalize_name(n)) for n in names]
    moves = [(a, b) for a, b in moves if a != b]
    if not moves:
        return 0, 0, 0
    changed = rows = conflicts = 0
    for old, new in moves:
        n = con.execute(f"select count(*) from {table} where {col} = ?", (old,)).fetchone()[0]
        if not apply:
            log(f"  {table}.{col}: {old!r} -> {new!r} ({n} rows)"); changed += 1; rows += n; continue
        try:
            con.execute("savepoint one")
            con.execute(f"update {table} set {col} = ? where {col} = ?", (new, old))
            con.execute("release one")
            changed += 1; rows += n
        except sqlite3.IntegrityError as e:
            con.execute("rollback to one"); con.execute("release one")
            conflicts += 1; log(f"  {table}.{col}: {old!r} -> {new!r} left: {e}")
    return changed, rows, conflicts


def run_file(path: Path, tables, apply: bool, log=print) -> None:
    con = sqlite3.connect(path)
    try:
        tot = [0, 0, 0]
        for table, col in tables:
            c, r, k = rekey(con, table, col, apply, log)
            if c or k:
                log(f"{path.name} {table}.{col}: {c} name(s), {r} row(s)" + (f", {k} left for a conflict" if k else "") + ("" if apply else " (dry run)"))
            tot[0] += c; tot[1] += r; tot[2] += k
        if apply:
            con.commit()
        if not any(tot):
            log(f"{path.name}: every name is already keyed the NBA way")
    finally:
        con.close()


def run_store(store: Path, apply: bool, log=print) -> int:
    """The slate's projection records: the frames' index and every name in the rotation inputs. Returns the files changed."""
    n = 0
    for p in sorted(store.glob("*.pkl")):
        try:
            with open(p, "rb") as f:
                rec = pickle.load(f)
        except Exception as e:   # noqa: BLE001
            log(f"  {p.name}: unreadable ({type(e).__name__}); left"); continue
        hits: list = []
        new = rekey_obj(rec, hits)
        if not hits:
            continue
        moves = sorted(set(hits))
        log(f"  {p.name}: " + ", ".join(f"{a!r} -> {b!r}" for a, b in moves[:6]) + ("…" if len(moves) > 6 else "") + ("" if apply else " (dry run)"))
        n += 1
        if apply:
            tmp = p.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(new, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(p)
    return n


def run_edits(day_dir: Path, apply: bool, log=print) -> int:
    """One day's edits and published layer: the pins CSV, the drawn grids, the scenarios. Returns the files changed."""
    n = 0
    for layer in ("edits", "published"):
        root = day_dir / layer
        if not root.is_dir():
            continue
        csv = root / "minutes_overrides.csv"
        if csv.exists():
            try:
                df = pd.read_csv(csv)
            except Exception as e:   # noqa: BLE001
                df = None; log(f"  {csv}: unreadable ({type(e).__name__}); left")
            if df is not None and "player" in df.columns:
                new = df["player"].map(lambda s: maybe(s) if isinstance(s, str) else s)
                if not new.equals(df["player"]):
                    moved = sorted({(a, b) for a, b in zip(df["player"], new) if a != b})
                    log(f"  {csv.relative_to(day_dir.parent)}: " + ", ".join(f"{a!r} -> {b!r}" for a, b in moved) + ("" if apply else " (dry run)"))
                    n += 1
                    if apply:
                        df["player"] = new
                        df.drop_duplicates("player", keep="last").to_csv(csv, index=False)
        for p in sorted(list((root / "rotations").glob("*.json")) + list((root / "scenarios").rglob("*.json"))):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError) as e:
                log(f"  {p}: unreadable ({type(e).__name__}); left"); continue
            hits: list = []
            new = rekey_obj(d, hits)
            if not hits:
                continue
            moves = sorted(set(hits))
            log(f"  {p.relative_to(day_dir.parent)}: " + ", ".join(f"{a!r} -> {b!r}" for a, b in moves[:6]) + ("…" if len(moves) > 6 else "") + ("" if apply else " (dry run)"))
            n += 1
            if apply:
                p.write_text(json.dumps(new, indent=0))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    a = ap.parse_args()
    hist = S.history_db_path()
    print(f"history database {hist}")
    run_file(hist, HISTORY_TABLES, a.apply)
    for dg in S.held():
        sp = S.paths(dg)
        print(f"slate {dg} {sp.db}")
        run_file(sp.db, SLATE_TABLES, a.apply)
        if sp.store.is_dir():
            k = run_store(sp.store, a.apply)
            print(f"slate {dg} projection store: {k} record(s) {'rewritten' if a.apply else 'to rewrite'}" if k else f"slate {dg} projection store: nothing to change")
    days = S.days_dir()
    if days.is_dir():
        for d in sorted(p for p in days.iterdir() if p.is_dir()):
            k = run_edits(d, a.apply)
            if k:
                print(f"day {d.name}: {k} edit file(s) {'rewritten' if a.apply else 'to rewrite'}")
    if not a.apply:
        print("dry run: --apply writes the changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
