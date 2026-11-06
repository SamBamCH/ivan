#!/usr/bin/env python3
"""
scripts/nba/normalize_names.py

Re-key the player names stored before the NBA name rule (ivan.sports.nba.names: the generational suffix dropped, the NBA
aliases) so every table meets on one spelling: 'jimmy butler iii' (nba_api) -> 'jimmy butler' (DraftKings, Rotowire).

    scripts/nba/normalize_names.py              # dry run: what would change, per table
    scripts/nba/normalize_names.py --apply      # rewrite the history database and every held slate's database

History tables: nba_stats, nba_news, nba_stint, nba_foul, nba_sub (player_in, player_out), nba_projection_capture,
nba_player_note. Slate tables (every held slate): starting_lineups, nba_market_rate, projection_snapshot; nba_player's
name is left (the next refresh rewrites the pool from DraftKings). A row whose new spelling already exists under the same
unique key (the same player stored twice for one game) is left as it is and listed. Idempotent: a second run changes nothing.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from ivan.sports.nba import slates as S
from ivan.sports.nba.names import normalize_name

HISTORY_TABLES = [("nba_stats", "player_name"), ("nba_news", "player_name"), ("nba_stint", "player_name"), ("nba_foul", "player_name"),
                  ("nba_sub", "player_in"), ("nba_sub", "player_out"), ("nba_projection_capture", "player_name"), ("nba_player_note", "player_name")]
SLATE_TABLES = [("starting_lineups", "player_name"), ("nba_market_rate", "player_name"), ("projection_snapshot", "player_name")]


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    a = ap.parse_args()
    hist = S.history_db_path()
    print(f"history database {hist}")
    run_file(hist, HISTORY_TABLES, a.apply)
    for dg in S.held():
        p = S.paths(dg).db
        print(f"slate {dg} {p}")
        run_file(p, SLATE_TABLES, a.apply)
    if not a.apply:
        print("dry run: --apply writes the changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
