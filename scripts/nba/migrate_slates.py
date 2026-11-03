#!/usr/bin/env python3
"""
scripts/nba/migrate_slates.py — one-time: the one pool becomes one directory per slate (docs/nba/notes/plan_slates_2026-11-03.md).

Before, nba_history.sqlite held one slate's pool (nba_player, starting_lineups, nba_market_rate) and every slate's lineups,
sims, contests and notes side by side, and data/nba/input held the current slate's edits with older slates' filed under
input/edits/<dg>/. After, each slate has data/nba/slates/<dg>/ (ivan.sports.nba.slates): its own slate.sqlite with those
tables, its edits, published layer, projection store, entries and outputs.

    scripts/nba/migrate_slates.py --dry-run     say what would move
    scripts/nba/migrate_slates.py               move it: the slate tables' rows into each slate's database, the files into
                                                each slate's directory, then the old tables renamed _migrated_<name> in the
                                                history database (a stale reader fails loudly; drop them when satisfied)

Idempotent: a slate directory that already exists is left alone; tables already renamed are skipped. Run once on the laptop
and once on the server (the server's directories fill from the laptop's feed afterwards).
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from ivan.sports.nba import slates as S
from ivan.sports.nba.paths import NBA_DATA_DIR, INPUT_DIR, OUTPUT_DIR

PER_SLATE_OUTPUT_GLOBS = ("chosen_entries.*", "lateswap.*", "lineups*.csv", "projections*.csv", "exposures*", "sim")


def table_exists(con: sqlite3.Connection, name: str, schema: str = "main") -> bool:
    return con.execute(f"select count(*) from {schema}.sqlite_master where type='table' and name=?", (name,)).fetchone()[0] > 0


def draft_groups(con: sqlite3.Connection) -> dict[int, dict]:
    """Every draft group the history database knows a slate's work for, with what it holds and the lock when known."""
    out: dict[int, dict] = {}

    def note(dg, key, n, lock=None):
        if dg is None:
            return
        d = out.setdefault(int(dg), {"lock": None})
        d[key] = d.get(key, 0) + int(n)
        if lock and (d["lock"] is None or str(lock) < str(d["lock"])):
            d["lock"] = str(lock)
    if table_exists(con, "nba_player"):
        for dg, n, lock in con.execute("select draft_group_id, count(*), min(gametime) from nba_player group by draft_group_id"):
            note(dg, "players", n, lock)
    if table_exists(con, "lineup"):
        for dg, n in con.execute("select draft_group_id, count(*) from lineup group by draft_group_id"):
            note(dg, "lineups", n)
    if table_exists(con, "lineup_pool"):
        for dg, n, lock in con.execute("select draft_group_id, count(*), min(lock) from lineup_pool group by draft_group_id"):
            note(dg, "pool_rows", n, lock)
    if table_exists(con, "historical_contest_information"):
        for dg, n, day in con.execute("select draft_group_id, count(*), min(date) from historical_contest_information group by draft_group_id"):
            note(dg, "contests", n, day)
    if table_exists(con, "nba_player_note"):
        for dg, n in con.execute("select draft_group_id, count(*) from nba_player_note group by draft_group_id"):
            note(dg, "notes", n)
    edits = INPUT_DIR / "edits"
    if edits.is_dir():
        for d in edits.iterdir():
            if d.is_dir() and d.name.isdigit():
                note(int(d.name), "edit_files", sum(1 for _ in d.iterdir()))
    return out


def copy_rows(con: sqlite3.Connection, table: str, where: str, params: tuple) -> int:
    """hist.<table> rows into main.<table> (same columns by name; main's table already created by ensure_slate_schema)."""
    cols = [r[1] for r in con.execute(f"pragma main.table_info({table})")]
    hcols = {r[1] for r in con.execute(f"pragma hist.table_info({table})")}
    use = [c for c in cols if c in hcols]
    cl = ", ".join(f'"{c}"' for c in use)
    cur = con.execute(f'insert or ignore into main."{table}" ({cl}) select {cl} from hist."{table}" where {where}', params)
    return cur.rowcount


def move_tree(src: Path, dst: Path, log, dry: bool, pattern: str = "*") -> int:
    if not src.exists():
        return 0
    n = 0
    items = [src] if src.is_file() else sorted(src.glob(pattern))
    for it in items:
        target = dst / it.name if src.is_dir() else dst
        log(f"  move {it.relative_to(NBA_DATA_DIR)} -> {target.relative_to(NBA_DATA_DIR)}")
        if not dry:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if it.is_dir():
                    shutil.copytree(it, target, dirs_exist_ok=True); shutil.rmtree(it)
                else:
                    shutil.move(str(it), str(target))
            else:
                shutil.move(str(it), str(target))
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dry = a.dry_run
    log = print
    hist = S.history_db_path()
    con = sqlite3.connect(hist)
    groups = draft_groups(con)
    held = next((dg for dg, d in groups.items() if d.get("players")), None)
    log(f"history database {hist}: {len(groups)} draft groups with slate work; the pool holds {held}")
    con.close()
    # a directory for every slate that was worked (a pool, lineups, pool snapshots, notes or edits); the contests the lobby listed
    # for every other draft group stay in the history's renamed table
    worked = {dg for dg, d in groups.items() if any(d.get(k) for k in ("players", "lineups", "pool_rows", "notes", "edit_files"))}
    log(f"{len(worked)} slates were worked ({len(groups) - len(worked)} only have contests on file: not migrated)")
    for dg in sorted(worked):
        d = groups[dg]
        if S.holds(dg):
            log(f"slate {dg}: directory exists, left alone"); continue
        log(f"slate {dg}: " + ", ".join(f"{k} {v}" for k, v in d.items() if k != "lock" and v) + (f", lock {d['lock']}" if d.get("lock") else ""))
        if dry:
            continue
        lock = None
        try:
            lock = datetime.fromisoformat(str(d["lock"])[:19]) if d.get("lock") else None
        except ValueError:
            lock = None
        sp = S.create(dg, lock=lock)
        scon = sqlite3.connect(sp.db); scon.execute(f"attach database '{hist}' as hist")
        moved = {}
        try:
            with scon:
                if table_exists(scon, "nba_player", "hist") and d.get("players"):
                    moved["nba_player"] = copy_rows(scon, "nba_player", "draft_group_id = ?", (dg,))
                    if table_exists(scon, "starting_lineups", "hist"):
                        moved["starting_lineups"] = copy_rows(scon, "starting_lineups", "1=1", ())
                    if table_exists(scon, "nba_market_rate", "hist"):
                        moved["nba_market_rate"] = copy_rows(scon, "nba_market_rate", "1=1", ())
                if table_exists(scon, "lineup", "hist"):
                    moved["lineup"] = copy_rows(scon, "lineup", "draft_group_id = ?", (dg,))
                    if table_exists(scon, "lineup_simulation_results", "hist"):
                        moved["lineup_simulation_results"] = copy_rows(scon, "lineup_simulation_results", "lineup_id in (select id from hist.lineup where draft_group_id = ?)", (dg,))
                if table_exists(scon, "lineup_pool", "hist"):
                    moved["lineup_pool"] = copy_rows(scon, "lineup_pool", "draft_group_id = ?", (dg,))
                if table_exists(scon, "historical_contest_information", "hist"):
                    moved["historical_contest_information"] = copy_rows(scon, "historical_contest_information", "draft_group_id = ?", (dg,))
                    for t in ("payouts", "classic_contest_metadata", "contest_sim_report"):
                        if table_exists(scon, t, "hist"):
                            moved[t] = copy_rows(scon, t, "contest_id in (select contest_id from hist.historical_contest_information where draft_group_id = ?)", (dg,))
                if table_exists(scon, "nba_player_note", "hist"):
                    moved["nba_player_note"] = copy_rows(scon, "nba_player_note", "draft_group_id = ?", (dg,))
                if table_exists(scon, "projection_snapshot", "hist") and dg == held:
                    moved["projection_snapshot"] = copy_rows(scon, "projection_snapshot", "1=1", ())
        finally:
            scon.close()
        log("  rows: " + ", ".join(f"{k} {v}" for k, v in moved.items() if v))
        # files: the filed edits of this slate; the live edits, published layer, store, entries and outputs belong to the held slate
        filed = INPUT_DIR / "edits" / str(dg)
        if filed.is_dir():
            for f in sorted(filed.iterdir()):
                if f.name == "minutes_overrides.csv":
                    move_tree(f, sp.overrides_csv, log, dry)
                elif f.suffix == ".json":
                    move_tree(f, sp.rotations / f.name, log, dry)
            if not any(filed.iterdir()):
                filed.rmdir()
        if dg == held:
            move_tree(INPUT_DIR / "minutes_overrides.csv", sp.overrides_csv, log, dry)
            move_tree(INPUT_DIR / "rotations", sp.rotations, log, dry)
            move_tree(INPUT_DIR / "scenarios", sp.scenarios, log, dry)
            move_tree(INPUT_DIR / "published", sp.published, log, dry)
            move_tree(NBA_DATA_DIR / "derived" / "projection", sp.store, log, dry)
            move_tree(INPUT_DIR / "entries.csv", sp.entries_csv, log, dry)
            move_tree(INPUT_DIR / "contest_weights.json", sp.contest_weights_json, log, dry)
            for pat in PER_SLATE_OUTPUT_GLOBS:
                move_tree(OUTPUT_DIR, sp.output, log, dry, pattern=pat)
        games = None
        try:
            scon = sqlite3.connect(sp.db)
            pairs = scon.execute("select count(distinct case when team < opponent then team || opponent else opponent || team end) from nba_player where team is not null and opponent is not null").fetchone()[0]
            games = int(pairs) or None
            scon.close()
        except sqlite3.Error:
            pass
        S.write_meta(dg, games=games, refreshed_at=datetime.now(), refreshed_by="migration") if games else S.write_meta(dg, refreshed_by="migration")
    if dry:
        log("dry run: nothing moved")
        return 0
    # the old tables step aside
    con = sqlite3.connect(hist)
    with con:
        for t in S.SLATE_TABLES:
            if table_exists(con, t) and not table_exists(con, f"_migrated_{t}"):
                con.execute(f'alter table "{t}" rename to "_migrated_{t}"'); log(f"history: {t} -> _migrated_{t}")
    con.close()
    for stale in (NBA_DATA_DIR / "active_slate.json",):
        if stale.exists():
            stale.unlink(); log(f"removed {stale.relative_to(NBA_DATA_DIR)}")
    log(f"done: {len(S.held())} slate directories under {S.SLATES_DIR.relative_to(NBA_DATA_DIR.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
