# src/ivan/core/replay.py
"""
Replay archived slates with the code that is current today.

For every slate the archive holds in a window, the driver restores the raw
inputs the slate was ingested with (pool, props, lines) into a *copy* of the
sport's database, pins the clock to the slate's lock, and runs the sport's
own pipeline on it: project -> generate the field -> simulate the contests
in the menu -> choose. The outputs land in one directory per slate, in the
same files the live pipeline writes, so the settlement and tax report read
them unchanged.

A slate can be replayed several times (ReplayOptions.repeats): the
projection is fitted once, then generate -> simulate -> choose run again per
repeat with their own seed (IVAN_SEED = seed + k, see ivan.core.seed) into
<slate dir>/rep_NNN/. Against the fixed actual results, the spread of the
repeats is the pipeline's own noise: how much of a window's profit came
from which lineups the generator jittered to and which universes the sim
happened to draw.

A sport plugs in through a SportAdapter (see ivan.sports.nfl.replay_adapter);
every stage runs in this process on the replay session, with the seed and run
directory set in the environment around it (the stages read IVAN_SEED and
IVAN_RUN_DIR wherever they draw or write). The live database is never written.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from sqlalchemy import text

from ivan.core import archive, artifacts, clock, seed as seedmod
from ivan.core.gitinfo import git_state
from ivan.database.session import get_session, init_db, add_missing_columns

MANIFEST = "replay.json"
STAGES = ("restore", "project", "generate", "field", "simulate", "choose")


def stage_rank(stage: str) -> int:
    """Position of a stage in the pipeline; ReplayOptions.through names the last one to run."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; one of {', '.join(STAGES)}")
    return STAGES.index(stage)


# --------------------------------------------------------------------------- the adapter contract
class SportAdapter(Protocol):
    sport: str
    working_tables: Sequence[str]          # per-slate state the replay clears between slates
    repeat_tables: Sequence[str]           # the subset rebuilt by every repeat (field, sim results)

    def restore(self, session, blobs: Dict[str, List[dict]], log: Callable[[str], None]) -> dict: ...
    def project(self, session, log: Callable[[str], None]) -> None: ...
    def generate(self, session, n_lineups: int, jobs: int, log: Callable[[str], None]) -> dict: ...
    def contests_for(self, session, lock_local: datetime, mode: str, slate_key: Optional[str] = None) -> List[dict]: ...
    def entries_csv(self, picks: List[dict], path: Path) -> int: ...
    def simulate(self, session, entries_csv: Path, n_sims: int, field_cap: int, log: Callable[[str], None]) -> dict: ...
    def choose(self, session, entries_csv: Path, out_dir: Path, weights: dict, log: Callable[[str], None]) -> dict: ...
    # optional: the "field" stage samples each contest's field from the generated pool and writes it as an
    # artifact without simulating outcomes (ReplayOptions.through == "field"); the validation scores that
    # field on the actual box scores, so the field model is tested on its own
    # def sample_field(self, session, entries_csv: Path, field_cap: int, log: Callable[[str], None]) -> dict: ...


# --------------------------------------------------------------------------- the contest menu
@dataclass
class MenuItem:
    match: str                  # case-insensitive regex against the contest name
    entries: int
    mode: Optional[str] = None  # showdown | classic | None (either)
    max_fee: Optional[float] = None
    min_entries: Optional[int] = None   # contest size floor


@dataclass
class Menu:
    items: List[MenuItem]
    weights: dict = field(default_factory=lambda: {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1})
    max_fee: Optional[float] = None
    picks: Dict[int, int] = field(default_factory=dict)      # explicit contest_id -> entries (the Backtest page's selection)

    @classmethod
    def load(cls, path: str | Path) -> "Menu":
        raw = json.loads(Path(path).read_text())
        picks = {int(p["contest_id"]): int(p.get("entries", 1)) for p in raw.get("picks", [])}
        items = [MenuItem(match=i["match"], entries=int(i["entries"]), mode=i.get("mode"), max_fee=i.get("max_fee"),
                          min_entries=i.get("min_entries")) for i in raw.get("contests", [])]
        return cls(items=items, weights=raw.get("weights") or cls.__dataclass_fields__["weights"].default_factory(),
                   max_fee=raw.get("max_fee"), picks=picks)


def pick_contests(contests: List[dict], menu: Menu, mode: str) -> List[dict]:
    """
    The contests to enter on one slate: for each menu item the largest
    contest whose name matches (and whose fee, size and mode fit), with the
    entry count capped at the contest's max entries. A contest is picked at
    most once. Each contest dict needs contest_id, name, entry_fee,
    total_entries, multi_entry and has_payouts.
    """
    picked, used = [], set()
    if menu.picks:                                   # an explicit selection: exactly these contests, entries capped at the max
        for c in contests:
            cid = int(c["contest_id"])
            if cid in menu.picks and cid not in used:
                cap = int(c.get("multi_entry") or 0) or menu.picks[cid]
                picked.append({**c, "entries": max(1, min(menu.picks[cid], cap)), "menu": "picked"})
                used.add(cid)
        return picked
    for item in menu.items:
        if item.mode and item.mode != mode:
            continue
        rx = re.compile(item.match, re.I)
        cands = [c for c in contests if c["contest_id"] not in used and rx.search(c["name"] or "") and c.get("has_payouts", True)]
        fee_cap = item.max_fee if item.max_fee is not None else menu.max_fee
        if fee_cap is not None:
            cands = [c for c in cands if float(c["entry_fee"]) <= fee_cap]
        if item.min_entries:
            cands = [c for c in cands if int(c["total_entries"]) >= item.min_entries]
        if not cands:
            continue
        c = max(cands, key=lambda c: int(c["total_entries"]))
        cap = int(c.get("multi_entry") or 0) or item.entries
        picked.append({**c, "entries": max(1, min(item.entries, cap)), "menu": item.match})
        used.add(c["contest_id"])
    return picked


# --------------------------------------------------------------------------- the replay database
def make_replay_db(live_db: Path, replay_db: Path, working_tables: Sequence[str], log=print) -> None:
    """A compact copy of the live database with the per-slate working tables emptied."""
    replay_db = Path(replay_db)
    replay_db.parent.mkdir(parents=True, exist_ok=True)
    if replay_db.exists():
        replay_db.unlink()
    log(f"copying {live_db} -> {replay_db} (VACUUM INTO)")
    t0 = time.time()
    con = sqlite3.connect(str(live_db))
    con.execute("VACUUM INTO ?", (str(replay_db),))
    con.close()
    con = sqlite3.connect(str(replay_db))
    for t in working_tables:
        try:
            con.execute(f'DELETE FROM "{t}"')
        except sqlite3.OperationalError:
            pass
    con.commit()
    con.execute("VACUUM")
    con.close()
    log(f"replay database ready ({replay_db.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")


def clone_replay_db(replay_db: Path, clone: Path, log=print) -> None:
    """A worker's private copy of a replay database (plain file copy; it is already compact)."""
    clone = Path(clone)
    clone.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(clone) + suffix)
        if p.exists():
            p.unlink()
    shutil.copyfile(replay_db, clone)
    log(f"cloned {Path(replay_db).name} -> {clone.name}")


def remove_clones(clones: Sequence[Path]) -> int:
    """Delete worker clones (and their -wal/-shm/-journal files) once the workers are gone. Returns the bytes freed."""
    freed = 0
    for clone in clones:
        for suffix in ("", "-wal", "-shm", "-journal"):
            p = Path(str(clone) + suffix)
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
    return freed


def clear_working_tables(session, working_tables: Sequence[str]) -> None:
    for t in working_tables:
        session.execute(text(f'DELETE FROM "{t}"'))
    session.commit()


# --------------------------------------------------------------------------- the driver
@dataclass
class ReplayOptions:
    n_lineups: int = 3000
    jobs: int = 4
    n_sims: int = 500
    field_cap: int = 5000
    rebuild_db: bool = False
    dry_run: bool = False
    only_slate_keys: Optional[Sequence[str]] = None
    skip_done: bool = True
    repeats: int = 1                 # generate -> simulate -> choose this many times per slate
    seed: Optional[int] = None       # IVAN_SEED of repeat k is seed + k; None leaves every stage unseeded
    through: str = "choose"          # the last stage to run: project | generate | field | simulate | choose


def slate_dir_name(cap: archive.ArchivedSlate) -> str:
    return f"{cap.lock:%Y%m%d-%H%M}_{cap.mode}_{cap.slate_key}"


def repeat_dir_name(k: int) -> str:
    return f"rep_{k:03d}"


def repeat_seed(opts: ReplayOptions, k: int) -> Optional[int]:
    return None if opts.seed is None else int(opts.seed) + int(k)


class _env_var:
    """Set one environment variable for the in-process stages, restore it after."""
    def __init__(self, name: str, value: Optional[str]):
        self.name, self.value, self.prev = name, value, None

    def __enter__(self):
        self.prev = os.environ.get(self.name)
        if self.value is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.value

    def __exit__(self, *exc):
        if self.prev is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.prev


def manifest_repeats(m: dict) -> List[dict]:
    """The repeats a manifest records; a manifest from before repeats existed is one repeat in the slate directory."""
    reps = m.get("repeats")
    if reps is None:
        if m.get("status") in ("ok", "error") and "choose" in m.get("stages", {}):
            return [{"k": 0, "dir": ".", "seed": None, "status": m.get("status"), "stages": m.get("stages", {})}]
        return []
    return list(reps)


def _tee(log_path: Path, log):
    """A log function that also appends to the slate's replay.log."""
    def _log(msg: str) -> None:
        with open(log_path, "a") as fh:
            fh.write(str(msg) + "\n")
        log(msg)
    return _log


def _run_repeat(adapter: SportAdapter, session, k: int, seed: Optional[int], out_dir: Path, rep_dir: Path,
                entries_csv: Path, menu: Menu, opts: ReplayOptions, log_path: Path, log) -> dict:
    rep_dir.mkdir(parents=True, exist_ok=True)
    r = {"k": k, "dir": "." if rep_dir == out_dir else rep_dir.name, "seed": seed, "status": "running", "stages": {},
         "started_at": datetime.now().isoformat(timespec="seconds")}
    t_rep = time.time()
    stage_log = _tee(log_path, log)
    through = stage_rank(opts.through)
    try:
        clear_working_tables(session, getattr(adapter, "repeat_tables", ()))
        with _env_var(seedmod.ENV, None if seed is None else str(seed)), _env_var(artifacts.ENV, str(rep_dir)):
            t0 = time.time()
            info = adapter.generate(session, opts.n_lineups, opts.jobs, stage_log) or {}
            r["stages"]["generate"] = {"seconds": round(time.time() - t0, 1), "n_lineups": opts.n_lineups, **info}
            if opts.through == "field":                 # the sim samples its own field, so this stage only runs when it is the last
                sample = getattr(adapter, "sample_field", None)
                if sample is None:
                    raise RuntimeError(f"the {adapter.sport} adapter has no field stage")
                t0 = time.time()
                info = sample(session, entries_csv, opts.field_cap, stage_log) or {}
                r["stages"]["field"] = {"seconds": round(time.time() - t0, 1), "field_cap": opts.field_cap, **info}
            if through >= stage_rank("simulate"):
                t0 = time.time()
                info = adapter.simulate(session, entries_csv, opts.n_sims, opts.field_cap, stage_log) or {}
                r["stages"]["simulate"] = {"seconds": round(time.time() - t0, 1), "n_sims": opts.n_sims, "field_cap": opts.field_cap, **info}
            if through >= stage_rank("choose"):
                t0 = time.time()
                info = adapter.choose(session, entries_csv, rep_dir, menu.weights, stage_log) or {}
                r["stages"]["choose"] = {"seconds": round(time.time() - t0, 1), "weights": menu.weights, **info}
        r["status"] = "ok"
    except Exception as e:
        r["status"] = "error"
        r["error"] = f"{type(e).__name__}: {e}"
        with open(log_path, "a") as fh:
            fh.write(traceback.format_exc())
        log(f"{out_dir.name} {r['dir']}: FAILED {r['error']}")
    r["seconds"] = round(time.time() - t_rep, 1)
    return r


def run_replay(adapter: SportAdapter, live_db: Path, replay_db: Path, out_root: Path, menu: Menu,
               since: Optional[datetime] = None, until: Optional[datetime] = None,
               opts: ReplayOptions = ReplayOptions(), log=print) -> List[dict]:
    """
    Replay every archived slate of adapter.sport in [since, until]. Returns one
    manifest dict per slate (also written as replay.json in the slate's
    directory), including errors, so a failed slate never stops the run.

    With opts.repeats > 1 the slate directory holds rep_000/, rep_001/, ...
    and the manifest lists them under "repeats". A slate that already has
    some repeats done is only extended to the requested count (its
    projection is fitted again, with the same seed).

    opts.through stops the chain after a stage ("project" fits the slate and
    writes the projections only; "field" generates the pool and samples each
    contest's field without simulating). The manifest records it, and a
    slate replayed through a later stage already covers an earlier one.
    """
    db_env = f"IVAN_{adapter.sport.upper()}_DB_URL"
    if opts.rebuild_db or not Path(replay_db).exists():
        make_replay_db(Path(live_db), Path(replay_db), adapter.working_tables, log)
    engine = init_db(adapter.sport, f"sqlite:///{replay_db}")
    archive.ensure_tables(engine)
    added = add_missing_columns(engine, adapter.metadata) if getattr(adapter, "metadata", None) is not None else []
    if added:
        log(f"replay database: added columns {added}")
    session = get_session(engine)
    # the archive lives in the live db; the replay copy carries it over, but a newer backfill may not be there
    caps = archive.slates(session, adapter.sport, since, until)
    if opts.only_slate_keys:
        caps = [c for c in caps if c.slate_key in set(opts.only_slate_keys)]
    n_rep = max(1, int(opts.repeats))
    through = stage_rank(opts.through)
    n_rep_needed = n_rep if through >= stage_rank("generate") else 0      # a projection-only replay has no repeats
    log(f"{len(caps)} archived slate(s) to replay" + (f" between {since:%Y-%m-%d} and {until:%Y-%m-%d}" if since and until else "")
        + (f", {n_rep} repeats each (seed {opts.seed}+k)" if n_rep > 1 else "") + (f", through {opts.through}" if opts.through != "choose" else ""))
    manifests = []
    for cap in caps:
        out_dir = Path(out_root) / adapter.sport / slate_dir_name(cap)
        mpath = out_dir / MANIFEST
        done: List[dict] = []
        if opts.skip_done and mpath.exists():
            prev = json.loads(mpath.read_text())
            # repeats from a run that stopped earlier in the chain are not done for this one
            covers = stage_rank(prev.get("through") or "choose") >= through
            done = [r for r in manifest_repeats(prev) if r.get("status") == "ok"] if covers else []
            if covers and len(done) >= n_rep_needed and prev.get("status") == "ok":
                log(f"skip {out_dir.name}: already replayed ({prev.get('finished_at')}, {len(done)} repeat(s), through {prev.get('through') or 'choose'})")
                manifests.append(prev)
                continue
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "replay.log"
        todo = [k for k in range(n_rep) if k not in {int(r["k"]) for r in done}]
        m = {"sport": adapter.sport, "capture_id": cap.id, "slate_key": cap.slate_key, "mode": cap.mode,
             "lock": cap.lock.isoformat(), "source": cap.source, "started_at": datetime.now().isoformat(timespec="seconds"),
             "options": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in opts.__dict__.items()},
             "through": opts.through, "git": git_state(), "status": "running", "stages": {}, "repeats": done}
        mpath.write_text(json.dumps(m, indent=2))
        as_of = cap.lock - timedelta(minutes=1)
        project_seed = repeat_seed(opts, 0)
        t_slate = time.time()
        try:
            with clock.frozen(as_of), _env_var(db_env, f"sqlite:///{replay_db}"), \
                    _env_var(seedmod.ENV, None if project_seed is None else str(project_seed)), _env_var(artifacts.ENV, str(out_dir)):
                clear_working_tables(session, adapter.working_tables)
                info_fn = getattr(adapter, "models_info", None)
                m["models"] = info_fn() if info_fn else {}
                if m["models"].get("fallback"):
                    log(f"{out_dir.name}: no as-of model version precedes this slate; using the live models (build_models.py --cutoff)")
                blobs = archive.load_blobs(session, cap.id)
                t0 = time.time()
                m["stages"]["restore"] = adapter.restore(session, blobs, log)
                m["stages"]["restore"]["seconds"] = round(time.time() - t0, 1)

                contests = adapter.contests_for(session, cap.lock, cap.mode, cap.slate_key)
                picks = pick_contests(contests, menu, cap.mode)
                m["contests"] = [{k: p[k] for k in ("contest_id", "name", "entry_fee", "total_entries", "entries", "menu")} for p in picks]
                if not picks:
                    raise RuntimeError(f"no contest on this slate matches the menu ({len(contests)} contests at lock)")
                entries_csv = out_dir / "entries.csv"
                n_entries = adapter.entries_csv(picks, entries_csv)
                log(f"{out_dir.name}: {len(picks)} contest(s), {n_entries} entries"
                    + (f", repeats {todo[0]}..{todo[-1]}" if n_rep > 1 and todo else ""))
                if opts.dry_run:
                    m["status"] = "dry_run"
                    continue

                t0 = time.time()
                adapter.project(session, log)
                m["stages"]["project"] = {"seconds": round(time.time() - t0, 1), "seed": project_seed}
                if through <= stage_rank("project"):
                    m["status"] = "ok"
                    continue

                for k in todo:
                    rep_dir = out_dir if n_rep == 1 else out_dir / repeat_dir_name(k)
                    r = _run_repeat(adapter, session, k, repeat_seed(opts, k), out_dir, rep_dir, entries_csv, menu, opts, log_path, log)
                    m["repeats"] = sorted([x for x in m["repeats"] if int(x["k"]) != k] + [r], key=lambda x: int(x["k"]))
                    mpath.write_text(json.dumps(m, indent=2, default=str))
                    if n_rep > 1:
                        log(f"{out_dir.name} {r['dir']}: {r['status']} in {r['seconds']:.0f}s")
                ok_reps = [x for x in m["repeats"] if x.get("status") == "ok"]
                if n_rep == 1 and m["repeats"]:          # the flat layout keeps the stage times at the top level
                    m["stages"].update(m["repeats"][0].get("stages", {}))
                    if m["repeats"][0].get("status") != "ok":
                        raise RuntimeError(m["repeats"][0].get("error", "repeat failed"))
                if not ok_reps:
                    raise RuntimeError(next((x.get("error") for x in m["repeats"] if x.get("error")), "no repeat succeeded"))
                m["status"] = "ok"
        except Exception as e:
            m["status"] = "error"
            m["error"] = f"{type(e).__name__}: {e}"
            with open(log_path, "a") as fh:
                fh.write(traceback.format_exc())
            log(f"{out_dir.name}: FAILED {m['error']}")
        finally:
            m["finished_at"] = datetime.now().isoformat(timespec="seconds")
            m["seconds"] = round(time.time() - t_slate, 1)
            mpath.write_text(json.dumps(m, indent=2, default=str))
            manifests.append(m)
            n_ok = len([x for x in m["repeats"] if x.get("status") == "ok"])
            log(f"{out_dir.name}: {m['status']} in {m['seconds']:.0f}s" + (f" ({n_ok}/{n_rep} repeats ok)" if n_rep > 1 else ""))
    session.close()
    return manifests


def replayed_dirs(out_root: Path, sport: str) -> List[Path]:
    """Slate directories under out_root/<sport> that finished a replay, oldest lock first."""
    root = Path(out_root) / sport
    if not root.exists():
        return []
    dirs = []
    for d in sorted(root.iterdir()):
        mp = d / MANIFEST
        if not (d.is_dir() and mp.exists()):
            continue
        m = json.loads(mp.read_text())
        # a slate still adding repeats counts once one repeat has finished
        if m.get("status") == "ok" or any(r.get("status") == "ok" for r in manifest_repeats(m)):
            dirs.append(d)
    return dirs


def repeat_dirs(slate_dirs: Sequence[Path]) -> Dict[int, List[Path]]:
    """Per repeat index, the directory of that repeat's choose run in every slate that has it (ok repeats only)."""
    out: Dict[int, List[Path]] = {}
    for d in slate_dirs:
        m = json.loads((Path(d) / MANIFEST).read_text())
        reps = manifest_repeats(m)
        if not reps and m.get("status") == "ok":            # a replay that stopped after the projection: the slate directory is the run
            out.setdefault(0, []).append(Path(d))
        for r in reps:
            if r.get("status") == "ok":
                out.setdefault(int(r["k"]), []).append(Path(d) / r.get("dir", "."))
    return dict(sorted(out.items()))
