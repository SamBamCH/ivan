# src/ivan/web/backtest_view.py
"""
The Backtest page: what the replay / after-tax backtest reports wrote,
read back from their directories. Nothing here computes; the numbers are
the report files of scripts/replay.py and scripts/nfl/backtest_tax.py
(ivan.sports.nfl.backtest_report, backtest_repeats).

A "report" is a directory with summary.json. They live under
    data/nfl/output/replay*/nfl/report_<since>_<until>/   (replays)
    data/nfl/output/backtest/<stamp>/                      (live backtests)
A replay root with slate directories but no report yet shows as a run in
progress, with a button that builds the report from what is there.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nfl.paths import OUTPUT_DIR

FILES = ("summary.json", "ledger.csv", "scenarios.csv", "formats.csv", "sim_vs_actual.csv", "volume_sweep.csv", "entries.csv",
         "repeats.csv", "repeats_by_slate.csv", "repeats_summary.json", "repeats.png", "volume_sweep.png")
REPORT_RX = re.compile(r"^report_(\d{8})_(\d{8})$")


# ------------------------------------------------------------------ formatting
def num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def money_signed(x, places: int = 2) -> str:
    v = num(x)
    if v is None:
        return ""
    s = f"${abs(v):,.{places}f}"
    return f"-{s}" if v < 0 else (f"+{s}" if v > 0 else s)


def pct_signed(x) -> str:
    v = num(x)
    return "" if v is None else f"{v * 100:+.1f}%"


def sign_css(x) -> str:
    v = num(x)
    return "" if v is None or v == 0 else ("pos" if v > 0 else "neg")


# ------------------------------------------------------------------ discovery
def _roots() -> list[Path]:
    return sorted(p for p in OUTPUT_DIR.glob("replay*") if p.is_dir())


def _report_id(path: Path) -> str:
    return str(path.relative_to(OUTPUT_DIR)).replace("\\", "/")


def report_path(report_id: str) -> Optional[Path]:
    """The directory of a report id (a path under the output dir), or None when it is not a report."""
    if not report_id or ".." in report_id or report_id.startswith("/"):
        return None
    p = (OUTPUT_DIR / report_id).resolve()
    try:
        p.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        return None
    return p if p.is_dir() and (p / "summary.json").is_file() else None


def reports() -> list[dict]:
    """Every report directory, newest first."""
    out = []
    for root in _roots():
        for d in sorted((root / "nfl").glob("report_*")) if (root / "nfl").is_dir() else []:
            if (d / "summary.json").is_file():
                out.append(_report_row(d, kind="replay", root=root))
    bt = OUTPUT_DIR / "backtest"
    for d in sorted(bt.iterdir()) if bt.is_dir() else []:
        if (d / "summary.json").is_file():
            out.append(_report_row(d, kind="live", root=None))
    return sorted(out, key=lambda r: r["mtime"], reverse=True)


def _report_row(d: Path, kind: str, root: Optional[Path]) -> dict:
    m = REPORT_RX.match(d.name)
    window = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]} .. {m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:]}" if m else d.name
    n_rep = 0
    rs = d / "repeats_summary.json"
    if rs.is_file():
        try:
            n_rep = int(json.loads(rs.read_text()).get("repeats") or 0)
        except (ValueError, TypeError):
            n_rep = 0
    label = (f"replay {root.name.replace('replay', '') or ''}".strip() + f" {window}" if kind == "replay" else f"live backtest {d.name}")
    if n_rep > 1:
        label += f" · {n_rep} repeats"
    return {"id": _report_id(d), "path": d, "kind": kind, "root": root, "window": window, "label": label,
            "mtime": (d / "summary.json").stat().st_mtime, "repeats": n_rep}


# ------------------------------------------------------------------ runs in progress
def _manifest(d: Path) -> Optional[dict]:
    try:
        return json.loads((d / "replay.json").read_text())
    except (OSError, ValueError):
        return None


def run_progress(root: Path) -> Optional[dict]:
    """How far a replay under this root has got: slates, repeats done / wanted, whether something is still running."""
    sport_dir = root / "nfl"
    if not sport_dir.is_dir():
        return None
    slates = []
    for d in sorted(sport_dir.iterdir()):
        m = _manifest(d) if d.is_dir() else None
        if not m:
            continue
        reps = m.get("repeats")
        if reps is None:
            reps = [{"status": m.get("status")}] if m.get("status") in ("ok", "error") else []
        wanted = int((m.get("options") or {}).get("repeats") or 1)
        ok = sum(1 for r in reps if r.get("status") == "ok")
        bad = sum(1 for r in reps if r.get("status") == "error")
        last = max((r.get("seconds") or 0) for r in reps) if reps else None
        if m.get("repeats") is None and m.get("seconds"):          # a single replay keeps its time at the top level
            last = m.get("seconds")
        slates.append({"dir": d.name, "key": m.get("slate_key"), "lock": m.get("lock"), "status": m.get("status"), "wanted": wanted,
                       "ok": ok, "bad": bad, "seconds": last, "finished_at": m.get("finished_at"),
                       "contests": len(m.get("contests") or []), "error": m.get("error"),
                       "written": (d / "replay.json").stat().st_mtime})
    if not slates:
        return None
    running = [s for s in slates if s["status"] == "running"]
    wanted = sum(s["wanted"] for s in slates)
    done = sum(s["ok"] for s in slates)
    log = root / "nfl" / "replay_repeats.log"
    tail = ""
    if log.is_file():
        tail = "".join(log.read_text(errors="replace").splitlines(keepends=True)[-8:])
    locks = [s["lock"] for s in slates if s["lock"]]
    last_write = max(s["written"] for s in slates)
    stale = bool(running) and (datetime.now().timestamp() - last_write) > 15 * 60
    return {"root": root, "name": root.name, "slates": slates, "wanted": wanted, "done": done, "failed": sum(s["bad"] for s in slates),
            "running": bool(running), "stale": stale, "last_write": last_write, "share": (done / wanted) if wanted else 0.0, "log_tail": tail,
            "since": min(locks)[:10] if locks else None, "until": max(locks)[:10] if locks else None,
            "updated": max((s["finished_at"] or "") for s in slates) or None}


def runs_in_progress() -> list[dict]:
    """Replay roots with slate output (whether or not a report exists yet)."""
    out = []
    for root in _roots():
        p = run_progress(root)
        if p:
            out.append(p)
    return out


def report_command(root: Path, since: str, until: str, bankroll: float = 0.0, tax_year: Optional[int] = None) -> list[str]:
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "replay.py"), "--sport", "nfl", "--since", since, "--until", until,
           "--out", str(root), "--report-only", "--bankroll", str(bankroll)]
    if tax_year:
        cmd += ["--tax-year", str(tax_year)]
    return cmd


# ------------------------------------------------------------------ one report
def _csv(d: Path, name: str) -> list[dict]:
    p = d / name
    if not p.is_file():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    return json.loads(df.to_json(orient="records"))     # NaN -> null


def _hist(values: list[float], bins: int = 12) -> list[dict]:
    """A text histogram: bin label, count, share of the largest bin."""
    vals = [v for v in (num(x) for x in values) if v is not None]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [{"label": money_signed(lo, 0), "n": len(vals), "share": 1.0, "zero": lo == 0}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        i = min(bins - 1, int((v - lo) / width))
        counts[i] += 1
    top = max(counts)
    out = []
    for i, c in enumerate(counts):
        a, b = lo + i * width, lo + (i + 1) * width
        out.append({"label": f"{money_signed(a, 0)} to {money_signed(b, 0)}", "n": c, "share": c / top if top else 0.0,
                    "has_zero": a <= 0 < b or (i == bins - 1 and b == 0)})
    return out


def report_view(d: Path) -> dict:
    summary = json.loads((d / "summary.json").read_text())
    year_end = summary.get("year_end") or {}
    fees = num(summary.get("fees"))
    view = {
        "dir": d, "id": _report_id(d), "title": summary.get("title") or d.name, "summary": summary, "year_end": year_end,
        "fees": fees, "roi": (year_end.get("P") / fees) if fees and year_end.get("P") is not None else None,
        "coverage": summary.get("coverage") or {}, "unsettled": summary.get("unsettled_contests") or [],
        "scenarios": _csv(d, "scenarios.csv"), "ledger": _csv(d, "ledger.csv"), "formats": _csv(d, "formats.csv"),
        "sim_vs_actual": _csv(d, "sim_vs_actual.csv"), "sweep": _csv(d, "volume_sweep.csv"),
        "files": [f for f in FILES if (d / f).is_file()],
        "images": [f for f in ("repeats.png", "volume_sweep.png") if (d / f).is_file()],
        "mtime": (d / "summary.json").stat().st_mtime,
    }
    sva = view["sim_vs_actual"]
    both = [r for r in sva if num(r.get("sim_expected_profit")) is not None and num(r.get("actual_profit")) is not None]
    view["sva_totals"] = {"sim": sum(num(r["sim_expected_profit"]) for r in both), "actual": sum(num(r["actual_profit"]) for r in both),
                          "sim_cash": (sum(num(r.get("sim_cash_pct")) or 0 for r in both) / len(both)) if both else None,
                          "actual_cash": (sum(num(r.get("actual_cash_rate")) or 0 for r in both) / len(both)) if both else None,
                          "n": len(both)}
    rs = d / "repeats_summary.json"
    if rs.is_file():
        try:
            rep = json.loads(rs.read_text())
        except ValueError:
            rep = {}
        if rep.get("repeats"):
            per = _csv(d, "repeats.csv")
            rep["hist"] = _hist([r.get("P") for r in per])
            rep["hist_after"] = _hist([r.get("after_tax_profit") for r in per])
            rep["per_repeat"] = per
            view["repeats"] = rep
    # the run this report came from (progress, repeats still being added)
    root = d.parent.parent if d.parent.name == "nfl" else None
    view["run"] = run_progress(root) if root else None
    m = REPORT_RX.match(d.name)
    view["since"], view["until"] = (f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}", f"{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:]}") if m else (None, None)
    view["root"] = root
    return view


def dist_rows(summary: dict) -> list[dict]:
    """The distribution lines of a repeats summary as table rows."""
    rows = []
    for key, label, fmt in (("P", "pre-tax profit", "money"), ("after_tax_profit", "after-tax profit", "money"), ("est_tax", "estimated tax", "money"),
                            ("roi", "ROI", "pct"), ("cash_rate", "cash rate", "rate"), ("sim_expected_profit", "sim's expected profit", "money")):
        dd = summary.get(key) or {}
        if dd.get("n"):
            rows.append({"label": label, "fmt": fmt, **dd})
    return rows
