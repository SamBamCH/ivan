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
pull(url, token, sport, ...)    fetch what changed from a feed server (http://host:8790) or from a
                                directory the producer pushed to (file:///path/to/data/feed) and apply it
push(sport, dest)               rsync the bundles to a consumer's feed directory over ssh
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from ivan.core.utils import PROJECT_ROOT

Log = Callable[[str], None]
FEED_DIR = PROJECT_ROOT / "data" / "feed"

TABLES = {
    "nba": {
        "live": ["nba_player", "odds", "nba_market_rate", "game_odds", "starting_lineups", "historical_contest_information", "payouts", "nba_news"],
        "reference": ["nba_stats", "nba_stint", "nba_projection_capture", "archived_slate", "projection_snapshot", "classic_contest_metadata"],
        "files": ["data/nba/models", "data/nba/input/historical_lines.csv", "data/nba/derived", "data/nba/diagnostics/scorecard"],
        # the producer's edits (minutes pins, the sim / public overrides, the painted grids) travel with every live bundle;
        # the consumer keeps them under data/nba/input/published/ as the base layer its own edits sit on (projection.live)
        "live_files": ["data/nba/input/minutes_overrides.csv", "data/nba/input/rotations"],
        "live_files_root": "data/nba/input",
        "published_dir": "data/nba/input/published",
        # the projection store (projection.store): the producer's per-team frames, mirrored to the same path on the consumer,
        # so its desk shows the producer's projection exactly and re-projects only the teams it edited
        "live_mirror": ["data/nba/derived/projection"],
        "db": "data/nba/nba_history.sqlite",
    },
    # the NFL: the server runs its own slates (ingest, generate, sim, choose on its own working tables) and takes the laptop's
    # reference tables (history, captures, settled contests, models) when its desk's Refresh from laptop is pressed. No live part.
    "nfl": {
        "reference": "*",                                  # every table but the working set below (and sqlite's own, and MLB's)
        "reference_exclude": ["nfl_player", "odds", "projection_snapshot", "lineup", "lineup_simulation_results", "contest_sim_report"],
        "reference_exclude_like": ["mlb_%"],
        "files": ["data/nfl/models"],
        "files_exclude": ["archive"],                      # the models' archive folder stays on the laptop
        "db": "data/nfl/nfl_history.sqlite",
    },
    # MMA: the whole card (fights, fighters, contests, lineups, sims) and the hand-entered inputs; the server only looks at it
    "mma": {
        "reference": "*",
        "reference_exclude": [],
        "files": ["data/mma/models", "data/mma/input", "data/mma/derived"],
        "db": "data/mma/mma_history.sqlite",
    },
}


def parts_of(sport: str, which: Optional[str] = None) -> List[str]:
    """The bundle parts a sport has ("live" and/or "reference"), or the one asked for; `both` / None = every part the sport has."""
    spec = TABLES[sport]
    have = [p for p in ("live", "reference") if p in spec]
    if which in (None, "both"):
        return have
    return [which] if which in have else []


def resolve_tables(db_path: Path, sport: str, part: str) -> List[str]:
    """The part's table list: as written, or every table in the database when the spec says "*" minus the excludes."""
    spec = TABLES[sport]; want = spec[part]
    if want != "*":
        return list(want)
    con = sqlite3.connect(db_path)
    try:
        names = [r[0] for r in con.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name")]
    finally:
        con.close()
    out = []
    for n in names:
        if n in spec.get("reference_exclude", []):
            continue
        if any(_like(n, pat) for pat in spec.get("reference_exclude_like", [])):
            continue
        out.append(n)
    return out


def _like(name: str, pattern: str) -> bool:
    import fnmatch
    return fnmatch.fnmatchcase(name, pattern.replace("%", "*"))


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


def publish(sport: str, out_dir: Path = FEED_DIR, which: Optional[str] = None, log: Log = print, db_path: Optional[Path] = None,
            producer: Optional[dict] = None, root: Path = PROJECT_ROOT) -> dict:
    """Write the sport's `which` bundle (live | reference | both; None = every part the sport has, parts_of) and refresh the manifest. Returns the manifest.
    `producer` is a note about who published (the desk, the autopilot tick, the nightly) that the consumer shows in its header.
    `root` is the checkout the spec's file paths are read from (the repository; tests pass a temporary one)."""
    spec = TABLES[sport]
    db_path = Path(db_path or root / spec["db"])
    d = Path(out_dir) / sport; d.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(d)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")     # to the microsecond: two publishes in one second are two stamps (a pull compares them)
    parts = parts_of(sport, which)
    def stamped(prev: Optional[dict], listing_digest: str) -> str:
        """The part's stamp: the previous one when the content is the same (a pull compares stamps, so an unchanged publish
        is nothing new to the consumer), else this publish's."""
        return prev["stamp"] if prev and prev.get("sha256") == listing_digest else stamp

    for part in parts:
        gz = d / f"{part}.sql.gz"
        tables = resolve_tables(db_path, sport, part)
        n = dump_tables(db_path, tables, gz)
        sha = _sha(gz)
        manifest[part] = {"stamp": stamped(manifest.get(part), sha), "sha256": sha, "bytes": n, "tables": tables}
        log(f"feed {sport} {part}: {n / 1e6:.1f} MB")
        if part == "live" and spec.get("live_files"):
            lf = d / "live_files"; shutil.rmtree(lf, ignore_errors=True); lf.mkdir(parents=True)
            lroot = root / spec["live_files_root"]
            for rel in spec["live_files"]:
                src = root / rel
                if src.is_dir():
                    shutil.copytree(src, lf / src.relative_to(lroot), dirs_exist_ok=True)
                elif src.exists():
                    (lf / src.relative_to(lroot)).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, lf / src.relative_to(lroot))
            listing = {str(q.relative_to(lf)): _sha(q) for q in lf.rglob("*") if q.is_file()}
            (d / "live_files.json").write_text(json.dumps(listing, indent=0))
            ldig = hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()
            manifest["live_files"] = {"stamp": stamped(manifest.get("live_files"), ldig), "sha256": ldig, "count": len(listing)}
            log(f"feed {sport} live files: {len(listing)} (the producer's pins and grids)")
        if part == "live" and spec.get("live_mirror"):
            # directories mirrored to the same path on the consumer (the projection store), replaced whole on a pull
            mir = d / "live_mirror"; shutil.rmtree(mir, ignore_errors=True); mir.mkdir(parents=True)
            for rel in spec["live_mirror"]:
                src = root / rel
                if src.is_dir():
                    shutil.copytree(src, mir / rel, dirs_exist_ok=True)
            listing = {str(q.relative_to(mir)): _sha(q) for q in mir.rglob("*") if q.is_file()}
            (d / "live_mirror.json").write_text(json.dumps(listing, indent=0))
            mdig = hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()
            manifest["live_mirror"] = {"stamp": stamped(manifest.get("live_mirror"), mdig), "sha256": mdig, "count": len(listing), "dirs": list(spec["live_mirror"])}
            log(f"feed {sport} mirrored files: {len(listing)} (the projection store)")
        if part == "reference":
            files = d / "files"; shutil.rmtree(files, ignore_errors=True)
            ignore = shutil.ignore_patterns(*spec["files_exclude"]) if spec.get("files_exclude") else None
            for rel in spec.get("files", []):
                src = root / rel
                if src.is_dir():
                    shutil.copytree(src, files / rel, dirs_exist_ok=True, ignore=ignore)
                elif src.exists():
                    (files / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, files / rel)
            listing = {str(p.relative_to(files)): _sha(p) for p in files.rglob("*") if p.is_file()}
            (d / "files.json").write_text(json.dumps(listing, indent=0))
            fdig = hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()
            manifest["files"] = {"stamp": stamped(manifest.get("files"), fdig), "sha256": fdig, "count": len(listing)}
    manifest["published_at"] = datetime.now().isoformat(timespec="seconds")
    if producer is not None:
        manifest["producer"] = producer
    else:
        manifest.pop("producer", None)                 # a publisher that says nothing must not leave the previous publisher's note standing
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


SSH_ENV = "IVAN_FEED_SSH"        # the consumer's .env: the producer as ssh reaches it, e.g. sam-pratt@main-thinkpad (the tailnet name)
SSH_KEY_ENV = "IVAN_FEED_SSH_KEY"   # optional: the private key to use (default: ssh's own choice, ~/.ssh/id_ed25519)
URL_ENV = "IVAN_FEED_URL"        # the older source: a directory (file://) or a feed server (http://, with IVAN_FEED_TOKEN)


def producer_source() -> Optional[tuple]:
    """How this machine reaches the producer, from the environment: ("ssh", user@host) when IVAN_FEED_SSH is set, else
    ("url", url) when IVAN_FEED_URL is, else None (this machine is the producer, or stands alone)."""
    host = os.environ.get(SSH_ENV, "").strip()
    if host:
        return ("ssh", host)
    url = os.environ.get(URL_ENV, "").strip()
    if url:
        return ("url", url)
    return None


def _ssh_cmd(host: str, timeout: int = 15) -> List[str]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", "-o", "StrictHostKeyChecking=accept-new"]
    key = os.environ.get(SSH_KEY_ENV, "").strip()
    if key:
        cmd += ["-i", key]
    return cmd + [host]


def pull_ssh(host: str, sport: str, state_dir: Path, db_path: Optional[Path] = None, log: Log = print, root: Path = PROJECT_ROOT,
             timeout: int = 180, slate: Optional[int] = None) -> dict:
    """
    The consumer's Refresh: fetch the producer's feed over ssh and apply it. One ssh command asks the producer to publish its
    live bundle fresh (`publish <sport>`: the forced command of the key, deploy/feed_serve.sh, runs scripts/<sport>/feed.py
    publish there), then rsync copies the producer's feed directory to this machine's data/feed/<sport>/ (the forced
    command answers rsync with rrsync, read only, rooted at the producer's data/feed; only what changed travels, so the
    reference bundle moves once a night), then the usual pull from that directory replaces the feed's tables, the published
    edits and the projection store here. Nothing is pushed from the producer and nothing runs there but on this request.
    With `slate` (a draft group) the command is `publish <sport> <slate>`: the producer loads that slate into its pool first
    when it holds another (the NBA: ingest.refresh_pool, what its desk's slate pick does), so the consumer's pick of a slate
    moves both desks to it; the ssh waits up to ten minutes for the load.
    Returns what pull returned (what changed), plus "published": the producer's answer to publish.
    """
    local = Path(root) / "data" / "feed"
    (local / sport).mkdir(parents=True, exist_ok=True)
    ask = ["publish", sport] + ([str(int(slate))] if slate is not None else [])
    try:
        r = subprocess.run(_ssh_cmd(host) + ask, check=True, capture_output=True, timeout=(600 if slate is not None else timeout), text=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        err = (getattr(e, "stderr", "") or "").strip().splitlines()
        raise RuntimeError(f"the producer ({host}) did not publish: {err[-1] if err else type(e).__name__}") from e
    published = (r.stdout or "").strip().splitlines()[-1:] or [""]
    log(f"feed {sport}: {host} published ({published[0]})")
    rsync = ["rsync", "-az", "--delete", "-e", " ".join(_ssh_cmd(host)[:-1]), f"{host}:{sport}/", f"{local / sport}/"]
    try:
        subprocess.run(rsync, check=True, capture_output=True, timeout=timeout, text=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        err = (getattr(e, "stderr", "") or "").strip().splitlines()
        raise RuntimeError(f"rsync from {host} failed: {err[-1] if err else type(e).__name__}") from e
    changed = pull(f"file://{local}", None, sport, state_dir, db_path=db_path, which="both", log=log, root=root)
    changed["published"] = published[0]
    return changed


def pull_from_producer(sport: str, state_dir: Path, db_path: Optional[Path] = None, log: Log = print, root: Path = PROJECT_ROOT,
                       slate: Optional[int] = None) -> dict:
    """The consumer's pull, whichever way the environment names the producer (producer_source): ssh (pull_ssh) or a URL (pull).
    `slate` asks the producer to load that slate first; only ssh can ask (a URL source serves what the producer holds)."""
    src = producer_source()
    if src is None:
        raise RuntimeError(f"no producer configured: set {SSH_ENV} (user@host) or {URL_ENV} in .env")
    kind, where = src
    if kind == "ssh":
        return pull_ssh(where, sport, state_dir, db_path=db_path, log=log, root=root, slate=slate)
    if slate is not None:
        raise RuntimeError(f"only an ssh producer ({SSH_ENV}) can be asked for another slate; {URL_ENV} serves what the laptop holds")
    return pull(where, os.environ.get("IVAN_FEED_TOKEN"), sport, state_dir, db_path=db_path, which="both", log=log, root=root)


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
         log: Log = print, root: Path = PROJECT_ROOT) -> dict:
    """
    Fetch the feed's manifest and apply every bundle whose stamp is newer than
    the last pull (state in state_dir/<sport>.json): live tables, reference
    tables, and the reference files into this checkout (`root`). Returns what changed.
    """
    base = base_url.rstrip("/") + f"/{sport}"
    spec = TABLES[sport]
    db_path = Path(db_path or root / spec["db"])         # the database under `root`, never another checkout's
    state_dir = Path(state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    state_p = state_dir / f"{sport}.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    tmp = state_dir / "tmp"; tmp.mkdir(exist_ok=True)
    manifest = json.loads(fetch(f"{base}/manifest.json", token, tmp / "manifest.json").read_text())
    # what the producer last published and when this machine last looked, whether or not anything was new (the consumer's header)
    state["manifest"] = {"published_at": manifest.get("published_at"), "producer": manifest.get("producer"),
                         "live_stamp": (manifest.get("live") or {}).get("stamp")}
    state["checked_at"] = datetime.now().isoformat(timespec="seconds")
    changed = {}
    parts = parts_of(sport, which)
    def is_new(key: str) -> bool:
        m = manifest.get(key)
        return bool(m) and state.get(key, {}).get("stamp") != m["stamp"]

    for part in parts:
        m = manifest.get(part)
        if not m:
            continue
        if is_new(part):
            gz = fetch(f"{base}/{part}.sql.gz", token, tmp / f"{part}.sql.gz")
            if _sha(gz) != m["sha256"]:
                raise RuntimeError(f"{part} bundle digest mismatch; try again")
            changed[part] = apply(sport, gz, db_path, log)
            state[part] = {"stamp": m["stamp"], "pulled_at": datetime.now().isoformat(timespec="seconds")}
        # the files beside the tables each have their own stamp (publish keeps a stamp while the content is the same), so a
        # grid saved without a projection change, or a store record dropped, still travels
        if part == "live" and spec.get("published_dir") and is_new("live_files"):
            # the producer's edits become this machine's published base layer (replaced whole: a pin the producer lifted is lifted here too)
            listing = json.loads(fetch(f"{base}/live_files.json", token, tmp / "live_files.json").read_text())
            pub = root / spec["published_dir"]; shutil.rmtree(pub, ignore_errors=True); pub.mkdir(parents=True, exist_ok=True)
            for rel in listing:
                fetch(f"{base}/live_files/{rel}", token, pub / rel)
            log(f"feed {sport}: {len(listing)} published edit file(s) (the producer's pins and grids)"); changed["live_files"] = len(listing)
            state["live_files"] = {"stamp": manifest["live_files"]["stamp"]}
        if part == "live" and is_new("live_mirror"):
            # the mirrored directories (the projection store) replaced whole: the consumer's frames are the producer's until it edits a team
            listing = json.loads(fetch(f"{base}/live_mirror.json", token, tmp / "live_mirror.json").read_text())
            for rel_dir in manifest["live_mirror"].get("dirs") or spec.get("live_mirror") or []:
                shutil.rmtree(root / rel_dir, ignore_errors=True); (root / rel_dir).mkdir(parents=True, exist_ok=True)
            for rel in listing:
                fetch(f"{base}/live_mirror/{rel}", token, root / rel)
            log(f"feed {sport}: {len(listing)} mirrored file(s) (the projection store)"); changed["live_mirror"] = len(listing)
            state["live_mirror"] = {"stamp": manifest["live_mirror"]["stamp"]}
        if part == "reference" and is_new("files"):   # the reference files (models, inputs) beside the tables
            listing = json.loads(fetch(f"{base}/files.json", token, tmp / "files.json").read_text())
            n = 0
            for rel, sha in listing.items():
                dst = root / rel
                if dst.exists() and _sha(dst) == sha:
                    continue
                fetch(f"{base}/files/{rel}", token, dst); n += 1
            log(f"feed {sport}: {n} reference files updated"); changed["files"] = n
            state["files"] = {"stamp": manifest["files"]["stamp"]}
    state_p.write_text(json.dumps(state, indent=1))
    return changed


def status(sport: str, state_dir: Path = PROJECT_ROOT / "data" / "feed-state") -> dict:
    """What a consumer knows about its feed, from the pull state: when it last checked, when the producer last published,
    who published (the manifest's producer note), the stamps applied. {} on a machine that never pulled."""
    p = Path(state_dir) / f"{sport}.json"
    try:
        st = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    m = st.get("manifest") or {}
    return dict(checked_at=st.get("checked_at"), pulled_at=(st.get("live") or {}).get("pulled_at"), live_stamp=(st.get("live") or {}).get("stamp"),
                published_at=m.get("published_at"), producer=m.get("producer") or {}, new=bool(m.get("live_stamp")) and m.get("live_stamp") != (st.get("live") or {}).get("stamp"))
