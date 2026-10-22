# src/ivan/core/feed.py
"""
The data feed: what one Ivan (the producer, which runs the capture loop and
pays for props) publishes so another Ivan (a consumer on someone else's
machine) can run the desk, the generator and the sim on the same inputs
without fetching anything itself.

A bundle is a gzipped SQL dump of named tables of one sport's database plus
files (models, inputs); a manifest lists each bundle's stamp and digest.

  live        the slate as it stands: the pool, props, market rates, game
              lines, starters, contests, payouts. Small; published every tick.
  reference   the history the models read: box scores, stints, projection
              captures, capture metadata, the model parameters and inputs.
              Published nightly.

The consumer replaces those tables in its own database and keeps everything
else (its edits, rotations, lineups, sims, outputs) untouched; the producer's
working tables (lineups, sims, reports) are never published.

publish(sport, out_dir)         write the bundles + manifest.json under out_dir/<sport>
apply(sport, bundle, db_path)   replace the bundle's tables in a database
pull(url, token, sport, ...)    fetch what changed from a feed server and apply it
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ivan.core.utils import PROJECT_ROOT

Log = Callable[[str], None]
FEED_DIR = PROJECT_ROOT / "data" / "feed"

TABLES = {
    "nba": {
        "live": ["nba_player", "odds", "nba_market_rate", "game_odds", "starting_lineups", "historical_contest_information", "payouts"],
        "reference": ["nba_stats", "nba_stint", "nba_projection_capture", "archived_slate", "projection_snapshot", "classic_contest_metadata"],
        "files": ["data/nba/models", "data/nba/input/historical_lines.csv", "data/nba/derived", "data/nba/diagnostics/scorecard"],
        "db": "data/nba/nba_history.sqlite",
    },
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_tables(db_path: Path, tables: List[str], out_gz: Path) -> int:
    """A gzipped .dump of `tables` (with their explicit indexes), read in one transaction so it is consistent under WAL. Returns bytes written."""
    con = sqlite3.connect(db_path)
    have = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    tables = [t for t in tables if t in have]
    marks = ",".join("?" * len(tables)) or "''"
    indexes = [r[0] for r in con.execute(f"select name from sqlite_master where type='index' and sql is not null and tbl_name in ({marks})", tables)] if tables else []
    con.close()
    if not tables:
        out_gz.parent.mkdir(parents=True, exist_ok=True); out_gz.write_bytes(gzip.compress(b"")); return 0
    dump = subprocess.run(["sqlite3", "-readonly", str(db_path), f".dump {' '.join(tables + indexes)}"], check=True, capture_output=True).stdout
    out_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_gz, "wb", compresslevel=3) as f:
        f.write(dump)
    return out_gz.stat().st_size


def publish(sport: str, out_dir: Path = FEED_DIR, which: str = "live", log: Log = print) -> dict:
    """Write the sport's `which` bundle (live | reference | both) and refresh the manifest. Returns the manifest."""
    spec = TABLES[sport]
    d = Path(out_dir) / sport; d.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(d)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = ["live", "reference"] if which == "both" else [which]
    for part in parts:
        gz = d / f"{part}.sql.gz"
        n = dump_tables(PROJECT_ROOT / spec["db"], spec[part], gz)
        manifest[part] = {"stamp": stamp, "sha256": _sha(gz), "bytes": n, "tables": spec[part]}
        log(f"feed {sport} {part}: {n / 1e6:.1f} MB")
        if part == "reference":
            files = d / "files"; shutil.rmtree(files, ignore_errors=True)
            for rel in spec["files"]:
                src = PROJECT_ROOT / rel
                if src.is_dir():
                    shutil.copytree(src, files / rel, dirs_exist_ok=True)
                elif src.exists():
                    (files / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, files / rel)
            listing = {str(p.relative_to(files)): _sha(p) for p in files.rglob("*") if p.is_file()}
            (d / "files.json").write_text(json.dumps(listing, indent=0))
            manifest["files"] = {"stamp": stamp, "count": len(listing)}
    manifest["published_at"] = datetime.now().isoformat(timespec="seconds")
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _read_manifest(d: Path) -> dict:
    p = Path(d) / "manifest.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def apply(sport: str, bundle_gz: Path, db_path: Optional[Path] = None, log: Log = print) -> List[str]:
    """Replace the bundle's tables in the local database (one transaction). Returns the tables replaced."""
    db_path = Path(db_path or PROJECT_ROOT / TABLES[sport]["db"])
    with gzip.open(bundle_gz, "rb") as f:
        sql = f.read().decode("utf-8", errors="replace")
    if not sql.strip():
        return []
    tables = []
    for line in sql.splitlines():
        if line.startswith("CREATE TABLE"):
            name = line.split("CREATE TABLE", 1)[1].strip().split("(")[0].strip().strip('"').strip("[]")
            if name.upper().startswith("IF NOT EXISTS"):
                name = name[13:].strip().strip('"')
            tables.append(name)
    body = sql.replace("BEGIN TRANSACTION;", "").replace("COMMIT;", "")
    con = sqlite3.connect(db_path, timeout=60, isolation_level=None)
    try:
        con.execute("PRAGMA foreign_keys=OFF"); con.execute("PRAGMA synchronous=OFF"); con.execute("PRAGMA temp_store=MEMORY")
        # one transaction for the drops and the whole dump: 300k rows import in seconds rather than one fsync per row
        con.executescript("BEGIN;\n" + "".join(f"DROP TABLE IF EXISTS [{t}];\n" for t in tables) + body + "\nCOMMIT;")
        con.execute("PRAGMA synchronous=FULL"); con.execute("PRAGMA foreign_keys=ON")
    finally:
        con.close()
    log(f"feed {sport}: replaced {len(tables)} tables from {bundle_gz.name}")
    return tables


def fetch(url: str, token: Optional[str], out: Path, timeout: int = 300) -> Path:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)
    return out


def pull(base_url: str, token: Optional[str], sport: str, state_dir: Path, db_path: Optional[Path] = None, which: str = "both",
         log: Log = print) -> dict:
    """
    Fetch the feed's manifest and apply every bundle whose stamp is newer than
    the last pull (state in state_dir/<sport>.json): live tables, reference
    tables, and the reference files into this checkout. Returns what changed.
    """
    base = base_url.rstrip("/") + f"/{sport}"
    state_dir = Path(state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    state_p = state_dir / f"{sport}.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    tmp = state_dir / "tmp"; tmp.mkdir(exist_ok=True)
    manifest = json.loads(fetch(f"{base}/manifest.json", token, tmp / "manifest.json").read_text())
    changed = {}
    parts = ["live", "reference"] if which == "both" else [which]
    for part in parts:
        m = manifest.get(part)
        if not m:
            continue
        if state.get(part, {}).get("stamp") == m["stamp"]:
            continue
        gz = fetch(f"{base}/{part}.sql.gz", token, tmp / f"{part}.sql.gz")
        if _sha(gz) != m["sha256"]:
            raise RuntimeError(f"{part} bundle digest mismatch; try again")
        changed[part] = apply(sport, gz, db_path, log)
        state[part] = {"stamp": m["stamp"], "pulled_at": datetime.now().isoformat(timespec="seconds")}
        if part == "reference" and manifest.get("files"):
            listing = json.loads(fetch(f"{base}/files.json", token, tmp / "files.json").read_text())
            n = 0
            for rel, sha in listing.items():
                dst = PROJECT_ROOT / rel
                if dst.exists() and _sha(dst) == sha:
                    continue
                fetch(f"{base}/files/{rel}", token, dst); n += 1
            log(f"feed {sport}: {n} reference files updated"); changed["files"] = n
            state["files"] = {"stamp": manifest["files"]["stamp"]}
    state_p.write_text(json.dumps(state, indent=1))
    return changed
