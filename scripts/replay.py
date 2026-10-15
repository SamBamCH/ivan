#!/usr/bin/env python3
"""
Replay archived slates with today's code, then settle and tax the result.

    python scripts/replay.py --sport nfl --since 2025-11-01 --until 2025-11-30 [--menu config/replay/nfl.json]
        [--lineups 3000 --jobs 4 --sims 500 --field-cap 5000] [--rebuild-db] [--slates KEY,KEY] [--dry-run]
        [--repeats 50 --seed 0] [--workers 3] [--report-only]
        [--no-report] [--bankroll 2000] [--tax-config config/tax.json]

Works on a copy of the sport's database (data/<sport>/replay/<sport>_replay.sqlite,
rebuilt with --rebuild-db or when missing); the live database is never
written. Each slate gets data/<sport>/output/replay/<sport>/<lock>_<mode>_<key>/
with the entries file it entered, the chosen lineups, the sim traces, a log
and replay.json. Afterwards the settlement + tax report runs over every
replayed slate in the window (same report as scripts/nfl/backtest_tax.py).

--repeats N runs generate -> simulate -> choose N times per slate (projection
once), repeat k seeded with IVAN_SEED = --seed + k, into rep_000/, rep_001/,
... of the slate directory. The report then adds the distribution of the
window over the repeats (repeats.csv, repeats_summary.json, repeats.png):
how much of the result survives the pipeline's own randomness. Running
again with a larger --repeats only adds the missing ones; --report-only
re-reads what is there without replaying.

--workers W splits the slates over W replay processes, each on its own copy
of the replay database (<sport>_replay.w<i>.sqlite). Each worker's generate
stage uses --jobs processes, so keep workers x jobs near the core count.

A slate must be in the archive first: ingest archives every live capture,
scripts/nfl/backfill_archive.py rebuilds past ones.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from ivan.core import archive, replay as rp
from ivan.core.utils import PROJECT_ROOT
from ivan.database.session import DB_DIR, get_session, init_db


def adapter_for(sport: str, config: str | None):
    if sport == "nfl":
        from ivan.sports.nfl.replay_adapter import NFLReplayAdapter
        return NFLReplayAdapter(config)
    raise SystemExit(f"no replay adapter for {sport!r} yet (nfl only)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", default="nfl")
    ap.add_argument("--since", required=True, help="first slate lock (local date)")
    ap.add_argument("--until", required=True, help="last slate lock (local date, inclusive)")
    ap.add_argument("--menu", default=None, help="contest menu json (default config/replay/<sport>.json)")
    ap.add_argument("--config", default=None, help="sport config json for the pipeline scripts")
    ap.add_argument("--lineups", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--field-cap", type=int, default=5000)
    ap.add_argument("--repeats", type=int, default=1, help="generate/simulate/choose this many times per slate")
    ap.add_argument("--seed", type=int, default=None, help="IVAN_SEED of repeat k is seed + k (default 0 when --repeats > 1)")
    ap.add_argument("--workers", type=int, default=1, help="replay processes, each on its own copy of the replay database")
    ap.add_argument("--rebuild-db", action="store_true", help="recreate the replay database from the live one")
    ap.add_argument("--replay-db", default=None)
    ap.add_argument("--out", default=None, help="output root (default data/<sport>/output/replay)")
    ap.add_argument("--slates", default=None, help="comma-separated slate keys to replay (default all in window)")
    ap.add_argument("--redo", action="store_true", help="replay slates (and repeats) that already finished")
    ap.add_argument("--dry-run", action="store_true", help="restore and pick contests only")
    ap.add_argument("--report-only", action="store_true", help="skip the replay; report what is already there")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print every repeat's settlement line")
    ap.add_argument("--bankroll", type=float, default=0.0)
    ap.add_argument("--tax-config", default=None)
    ap.add_argument("--tax-year", type=int, default=None)
    return ap.parse_args(argv)


def run_workers(a, sport: str, live_db: Path, replay_db: Path, out_root: Path, adapter, since, until) -> list[dict]:
    """Split the slates over --workers copies of this script, each with its own replay database."""
    if a.rebuild_db or not replay_db.exists():
        rp.make_replay_db(live_db, replay_db, adapter.working_tables)
    session = get_session(init_db(sport, f"sqlite:///{replay_db}"))
    caps = archive.slates(session, sport, since, until)
    session.close()
    if a.slates:
        want = {s.strip() for s in a.slates.split(",")}
        caps = [c for c in caps if c.slate_key in want]
    groups = [[] for _ in range(a.workers)]
    for i, c in enumerate(caps):
        groups[i % a.workers].append(c.slate_key)
    groups = [g for g in groups if g]
    print(f"{len(caps)} slate(s) over {len(groups)} worker(s)")
    procs = []
    for i, keys in enumerate(groups):
        clone = replay_db.with_name(f"{replay_db.stem}.w{i}{replay_db.suffix}")
        rp.clone_replay_db(replay_db, clone)
        log = out_root / sport / f"worker_{i}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--sport", sport, "--since", a.since, "--until", a.until,
               "--slates", ",".join(keys), "--replay-db", str(clone), "--out", str(out_root), "--no-report", "--workers", "1",
               "--lineups", str(a.lineups), "--jobs", str(max(1, min(a.jobs, (os.cpu_count() or 1) // max(1, a.workers)))), "--sims", str(a.sims), "--field-cap", str(a.field_cap),
               "--repeats", str(a.repeats)]
        if a.seed is not None:
            cmd += ["--seed", str(a.seed)]
        for flag, val in (("--menu", a.menu), ("--config", a.config)):
            if val:
                cmd += [flag, val]
        if a.redo:
            cmd.append("--redo")
        print(f"worker {i}: {len(keys)} slate(s) {keys} -> {log}")
        fh = open(log, "w")
        # the sim already runs a process pool per worker; one BLAS thread per process keeps W workers from thrashing
        # each replay worker gets its share of the cores for its own sim pool and lineup jobs
        share = str(max(1, (os.cpu_count() or 1) // max(1, a.workers)))
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "IVAN_SIM_WORKERS": share}
        procs.append((i, subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True), fh))
    try:
        for i, p, fh in procs:
            rc = p.wait()
            print(f"worker {i}: exit {rc}")
    finally:
        # a cancel (SIGTERM / Ctrl-C) must take the workers and their generator pools down too, or they keep
        # writing to clone databases the next run replaces under them ("attempt to write a readonly database")
        import signal
        for i, p, fh in procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
        for i, p, fh in procs:
            if p.poll() is None:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
            fh.close()
    dirs = [d for d in rp.replayed_dirs(out_root, sport) if since <= _lock_of(d) <= until]
    return [json.loads((d / rp.MANIFEST).read_text()) for d in dirs]


def main(argv=None) -> int:
    a = parse_args(argv)
    sport = a.sport.lower()
    live_db = Path(DB_DIR) / sport / f"{sport}_history.sqlite"
    replay_db = Path(a.replay_db) if a.replay_db else Path(DB_DIR) / sport / "replay" / f"{sport}_replay.sqlite"
    out_root = Path(a.out) if a.out else Path(DB_DIR) / sport / "output" / "replay"
    menu = rp.Menu.load(a.menu or PROJECT_ROOT / "config" / "replay" / f"{sport}.json")
    since = datetime.fromisoformat(a.since)
    until = datetime.fromisoformat(a.until) + timedelta(days=1) - timedelta(seconds=1)
    if a.repeats > 1 and a.seed is None:
        a.seed = 0
    opts = rp.ReplayOptions(n_lineups=a.lineups, jobs=a.jobs, n_sims=a.sims, field_cap=a.field_cap, rebuild_db=a.rebuild_db,
                            dry_run=a.dry_run, only_slate_keys=[s.strip() for s in a.slates.split(",")] if a.slates else None,
                            skip_done=not a.redo, repeats=a.repeats, seed=a.seed)
    adapter = adapter_for(sport, a.config)
    if not a.report_only:
        if a.workers > 1 and not a.dry_run:
            manifests = run_workers(a, sport, live_db, replay_db, out_root, adapter, since, until)
        else:
            manifests = rp.run_replay(adapter, live_db, replay_db, out_root, menu, since, until, opts)
        ok = [m for m in manifests if m.get("status") == "ok"]
        bad = [m for m in manifests if m.get("status") == "error"]
        print(f"\nreplayed {len(ok)} slate(s)" + (f", {len(bad)} failed: " + "; ".join(f"{m['slate_key']} ({m['error'][:80]})" for m in bad) if bad else ""))
        if a.no_report or a.dry_run or not ok:
            return 1 if bad and not ok else 0

    if sport != "nfl":
        return 0
    from ivan.sports.nfl import backtest_repeats, backtest_report
    from ivan.sports.nfl.tax import load_tax_config
    # settlement reads the actuals: the live database when only reporting (repairs and later scrapes land there), else the replay copy
    session = get_session(init_db("nfl", f"sqlite:///{live_db}"))      # the actuals always come from the live database
    dirs = [d for d in rp.replayed_dirs(out_root, sport) if since <= _lock_of(d) <= until]
    if not dirs:
        print("nothing replayed in this window")
        return 1
    reps = rp.repeat_dirs(dirs)
    cfg = load_tax_config(a.tax_config)
    report_dir = out_root / sport / f"report_{since:%Y%m%d}_{until:%Y%m%d}"
    title = f"replay {sport} {since:%Y-%m-%d}..{until:%Y-%m-%d}"
    # replayed entries were never in the real field, so nothing of ours is removed from it (in_field="no")
    first = min(reps) if reps else 0
    runs = backtest_repeats.repeat_runs({first: reps.get(first, dirs)})[first]
    backtest_report.run_report(session, runs, cfg, bankroll=a.bankroll, out=report_dir, in_field="no", tax_year=a.tax_year,
                               plot=not a.no_plot, title=title + (f" (repeat {first} of {len(reps)})" if len(reps) > 1 else ""))
    if len(reps) > 1:
        print()
        backtest_repeats.report_repeats(session, reps, cfg, out=report_dir, in_field="no", tax_year=a.tax_year,
                                        plot=not a.no_plot, title=title + f": {len(reps)} repeats", verbose=a.verbose)
    return 0


def _lock_of(d: Path) -> datetime:
    return datetime.fromisoformat(json.loads((d / rp.MANIFEST).read_text())["lock"])


if __name__ == "__main__":
    raise SystemExit(main())
