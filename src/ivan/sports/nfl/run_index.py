# src/ivan/sports/nfl/run_index.py
"""
Every replay of a slate, across every run on disk: the Backtest hub's runs
(output/backtests/<name>), the Models page's runs
(output/models/<component>/<stamp>) and the lab replay roots (output/replay*).

A run writes one directory per slate (<lock>_<mode>_<key>) with a replay.json
manifest: the slate, when it ran, the commit (git), the options, the stage it
ran through, the models version, the contests entered and each repeat's
status. The run's report_* directory has the settlement per contest
(sim_vs_actual.csv) and per repeat (repeats_by_contest.csv). Those files are
the record retention.py keeps for good, so this index reads nothing else:
a purged run still shows what it claimed and what happened.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ivan.sports.nfl.paths import OUTPUT_DIR
from ivan.sports.nfl.retention import run_roots

SPORT = "nfl"


@dataclass
class SlateRun:
    kind: str                       # backtest | models/<component> | replay
    run: str                        # the run's name (the stamp for a models run)
    root: Path
    slate_key: str
    lock: str
    mode: str
    started_at: str
    finished_at: Optional[str]
    status: str                     # ok | error | running | dry_run
    through: str
    commit: str
    dirty: bool
    subject: str
    options: dict                   # n_lineups, n_sims, field_cap, seed, repeats
    method: Optional[str]           # the chooser (from the first repeat's choose stage)
    models: Optional[str]           # the as-of cutoff, or "live"
    repeats_ok: int
    repeats_wanted: int
    contests: int
    entries: int
    fees: Optional[float] = None    # from the run's report, this slate's contests only
    sim: Optional[float] = None
    actual: Optional[float] = None
    repeat_profits: List[float] = field(default_factory=list)   # actual profit per repeat over this slate's contests
    report_id: Optional[str] = None
    validation_id: Optional[str] = None
    purged_at: Optional[str] = None
    pinned: bool = False
    error: Optional[str] = None
    seconds: Optional[float] = None
    label: str = ""

    @property
    def repeat_mean(self) -> Optional[float]:
        return sum(self.repeat_profits) / len(self.repeat_profits) if self.repeat_profits else None

    @property
    def p_profit(self) -> Optional[float]:
        return (sum(1 for p in self.repeat_profits if p > 0) / len(self.repeat_profits)) if self.repeat_profits else None

    def as_dict(self) -> dict:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}
        d["repeat_mean"] = self.repeat_mean
        d["p_profit"] = self.p_profit
        return d


def _json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _csv(p: Path) -> List[dict]:
    try:
        with open(p, newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _num(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _report_dir(root: Path) -> Optional[Path]:
    reps = sorted((root / SPORT).glob("report_*")) if (root / SPORT).is_dir() else []
    reps = [r for r in reps if (r / "summary.json").is_file()]
    return reps[-1] if reps else None


def _rel(p: Path, output_dir: Path) -> Optional[str]:
    try:
        return str(p.relative_to(output_dir)).replace("\\", "/")
    except ValueError:
        return None


def manifests(root: Path) -> List[tuple[Path, dict]]:
    """(slate dir, manifest) for every replayed slate under a run root."""
    sport = root / SPORT
    out = []
    for d in sorted(sport.iterdir()) if sport.is_dir() else []:
        if d.is_dir():
            m = _json(d / "replay.json")
            if m and m.get("slate_key"):
                out.append((d, m))
    return out


def _run_json(root: Path) -> dict:
    return _json(root / "run.json") or {}


def _method(m: dict) -> Optional[str]:
    reps = m.get("repeats") or []
    stages = (reps[0].get("stages") if reps else None) or m.get("stages") or {}
    w = (stages.get("choose") or {}).get("weights") or {}
    return w.get("method")


def slate_run(kind: str, root: Path, d: Path, m: dict, run: dict, report: Optional[Path], output_dir: Path = OUTPUT_DIR) -> SlateRun:
    opts = m.get("options") or {}
    reps = m.get("repeats") or []
    git = m.get("git") or run.get("git") or {}
    wanted = int(opts.get("repeats") or 1)
    ok = sum(1 for r in reps if r.get("status") == "ok") if reps else (1 if m.get("status") == "ok" else 0)
    contests = m.get("contests") or []
    ids = {int(c["contest_id"]) for c in contests if c.get("contest_id") is not None}
    sr = SlateRun(kind=kind, run=root.name, root=root, slate_key=str(m["slate_key"]), lock=str(m.get("lock") or ""), mode=str(m.get("mode") or ""),
                  started_at=str(m.get("started_at") or run.get("created_at") or ""), finished_at=m.get("finished_at"),
                  status=str(m.get("status") or ""), through=str(m.get("through") or "choose"),
                  commit=str(git.get("commit") or ""), dirty=bool(git.get("dirty")), subject=str(git.get("subject") or ""),
                  options={k: opts.get(k) for k in ("n_lineups", "n_sims", "field_cap", "seed", "repeats")},
                  method=_method(m) or (run.get("params") or {}).get("method"),
                  models=((m.get("models") or {}).get("cutoff") or ("live" if m.get("models") else None)),
                  repeats_ok=ok, repeats_wanted=wanted, contests=len(contests), entries=sum(int(c.get("entries") or 0) for c in contests),
                  purged_at=run.get("purged_at"), pinned=bool(run.get("pinned")), error=m.get("error"), seconds=m.get("seconds"),
                  label=str(run.get("label") or ""))
    if report is not None and ids:
        rows = [r for r in _csv(report / "sim_vs_actual.csv") if _num(r.get("contest_id")) in ids]
        if rows:
            sr.fees = sum(_num(r.get("fees")) or 0.0 for r in rows)
            sims = [_num(r.get("sim_expected_profit")) for r in rows]
            acts = [_num(r.get("actual_profit")) for r in rows]
            sr.sim = sum(x for x in sims if x is not None) if any(x is not None for x in sims) else None
            sr.actual = sum(x for x in acts if x is not None) if any(x is not None for x in acts) else None
            sr.report_id = _rel(report, output_dir)
        per: Dict[str, float] = {}
        for r in _csv(report / "repeats_by_contest.csv"):
            if _num(r.get("contest_id")) in ids and _num(r.get("actual_profit")) is not None:
                per[r.get("repeat")] = per.get(r.get("repeat"), 0.0) + _num(r.get("actual_profit"))
        sr.repeat_profits = [per[k] for k in sorted(per, key=lambda x: _num(x) or 0)]
    vdir = root / SPORT / "validation" / "rep_000"
    if (vdir / "summary.json").is_file():
        sr.validation_id = _rel(vdir, output_dir)
    return sr


def all_runs(output_dir: Path = OUTPUT_DIR) -> List[SlateRun]:
    out: List[SlateRun] = []
    for kind, root in run_roots(output_dir):
        run = _run_json(root)
        report = _report_dir(root)
        for d, m in manifests(root):
            out.append(slate_run(kind, root, d, m, run, report, output_dir))
    return out


def slate_runs(slate_key: str, output_dir: Path = OUTPUT_DIR) -> List[SlateRun]:
    """Every replay of one slate, newest first."""
    key = str(slate_key)
    rows = [r for r in all_runs(output_dir) if r.slate_key == key]
    return sorted(rows, key=lambda r: r.started_at, reverse=True)


def slates(output_dir: Path = OUTPUT_DIR) -> List[dict]:
    """One row per slate that any run replayed: lock, mode, how many runs, the newest."""
    by: Dict[str, dict] = {}
    for r in all_runs(output_dir):
        s = by.setdefault(r.slate_key, {"slate_key": r.slate_key, "lock": r.lock, "mode": r.mode, "runs": 0, "last": "", "ok": 0})
        s["runs"] += 1
        s["ok"] += 1 if r.status == "ok" else 0
        s["last"] = max(s["last"], r.started_at)
    return sorted(by.values(), key=lambda s: s["lock"], reverse=True)


def by_slate(output_dir: Path = OUTPUT_DIR) -> Dict[str, List[SlateRun]]:
    out: Dict[str, List[SlateRun]] = {}
    for r in all_runs(output_dir):
        out.setdefault(r.slate_key, []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: r.started_at, reverse=True)
    return out
