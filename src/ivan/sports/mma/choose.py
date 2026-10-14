# src/ivan/sports/mma/choose.py
"""
Pick the lineups to enter from a contest's simulation results. Greedy:
quality = roi_weight * average ROI + (1 - roi_weight) * cash rate; each
next pick maximises quality minus diversity_weight times its largest
fighter overlap with anything already picked. Ported from
mma_dfs Contest.choose_lineups; the overlap is computed on a membership
matrix instead of set intersections so 150 picks over thousands of
candidates take seconds.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.sports.mma.entries import BASE_COLS, SLOT_COLS
from ivan.sports.mma.models import Lineup, LineupSimulationResult, MMAPlayer
from ivan.sports.mma.paths import OUTPUT_DIR

DEFAULTS = {"num_lineups": 150, "roi_weight": 0.8, "diversity_weight": 0.4}


def choose_lineups(session: Session, contest_name: str, cfg: dict, num_lineups: Optional[int] = None,
                   roi_weight: Optional[float] = None, diversity_weight: Optional[float] = None) -> list[LineupSimulationResult]:
    ch = {**DEFAULTS, **cfg.get("choose", {})}
    n = int(num_lineups or ch["num_lineups"])
    w_roi = float(ch["roi_weight"] if roi_weight is None else roi_weight)
    lam = float(ch["diversity_weight"] if diversity_weight is None else diversity_weight)

    results = (session.query(LineupSimulationResult).filter(LineupSimulationResult.contest_name == contest_name)
               .order_by(LineupSimulationResult.lineup_id).all())
    if not results:
        raise SystemExit(f"no simulation results for contest {contest_name!r}; run scripts/mma/sim.py first")
    lineups = {lu.id: lu for lu in session.query(Lineup).filter(Lineup.id.in_([r.lineup_id for r in results])).all()}

    quality = np.array([w_roi * float(r.avg_roi or 0.0) + (1 - w_roi) * float(r.cash_pct or 0.0) for r in results])
    pids = sorted({p for r in results for p in lineups[r.lineup_id].player_ids_list})
    col = {p: j for j, p in enumerate(pids)}
    B = np.zeros((len(results), len(pids)), dtype=np.int16)
    for i, r in enumerate(results):
        for p in lineups[r.lineup_id].player_ids_list:
            B[i, col[p]] = 1

    picked: list[int] = []
    max_overlap = np.zeros(len(results), dtype=float)
    available = np.ones(len(results), dtype=bool)
    first = int(np.argmax(quality))
    picked.append(first)
    available[first] = False
    max_overlap = np.maximum(max_overlap, B @ B[first])
    while len(picked) < min(n, len(results)):
        score = np.where(available, quality - lam * max_overlap, -np.inf)
        k = int(np.argmax(score))
        picked.append(k)
        available[k] = False
        max_overlap = np.maximum(max_overlap, B @ B[k])
    return [results[k] for k in picked]


def lineup_rows(session: Session, results: list[LineupSimulationResult]) -> pd.DataFrame:
    """F1..F6 as 'Name (id)', Avg Payout, Avg ROI, Cash %, Salary Remaining, Dupe Count: the original's export."""
    players = {p.id: p for p in session.query(MMAPlayer).all()}
    lineups = {lu.id: lu for lu in session.query(Lineup).filter(Lineup.id.in_([r.lineup_id for r in results])).all()}
    rows = []
    for r in results:
        ids = list(lineups[r.lineup_id].player_ids_list)
        cells = [f"{players[p].name} ({p})" if p in players else str(p) for p in ids]
        cells += [""] * (6 - len(cells))
        salary = sum(players[p].salary for p in ids if p in players)
        rows.append(cells + [round(float(r.avg_payout or 0), 2), round(float(r.avg_roi or 0), 4),
                             round(100 * float(r.cash_pct or 0), 2), 50000 - salary, int(r.dupes or 1)])
    return pd.DataFrame(rows, columns=["F1", "F2", "F3", "F4", "F5", "F6", "Avg Payout", "Avg ROI", "Cash %",
                                       "Salary Remaining", "Dupe Count"])


def exposures(session: Session, results: list[LineupSimulationResult]) -> pd.DataFrame:
    players = {p.id: p for p in session.query(MMAPlayer).all()}
    lineups = {lu.id: lu for lu in session.query(Lineup).filter(Lineup.id.in_([r.lineup_id for r in results])).all()}
    counts = Counter(p for r in results for p in lineups[r.lineup_id].player_ids_list)
    n = max(1, len(results))
    df = pd.DataFrame([{"name": players[p].name, "salary": players[p].salary, "win_prob": players[p].win_prob,
                        "projection": players[p].projection, "count": c, "exposure": c / n} for p, c in counts.items() if p in players])
    return df.sort_values(["exposure", "salary"], ascending=[False, False]).reset_index(drop=True)


def export(session: Session, results: list[LineupSimulationResult], out_dir: Path = OUTPUT_DIR, stamp: Optional[str] = None,
           base: str = "chosen_lineups") -> dict[str, Path]:
    """Write the detail CSV, the DK upload CSV (six ids per row) and the exposures CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    detail = lineup_rows(session, results)
    lineups = {lu.id: lu for lu in session.query(Lineup).filter(Lineup.id.in_([r.lineup_id for r in results])).all()}
    upload = pd.DataFrame([list(lineups[r.lineup_id].player_ids_list) for r in results], columns=["F"] * 6)
    paths = {"lineups": out_dir / f"{base}.lineups.{stamp}.csv", "upload": out_dir / f"{base}.upload.{stamp}.csv",
             "exposures": out_dir / f"{base}.exposures.{stamp}.csv"}
    detail.to_csv(paths["lineups"], index=False)
    upload.to_csv(paths["upload"], index=False)
    exposures(session, results).to_csv(paths["exposures"], index=False)
    return paths


def format_exposures(df: pd.DataFrame) -> str:
    if df.empty:
        return "no lineups to analyse"
    lines = [f"{'fighter':<24} {'salary':>7} {'win%':>6} {'exposure':>9}"]
    for _, e in df.iterrows():
        lines.append(f"{e['name']:<24} {int(e['salary']):>7} {100 * float(e['win_prob'] or 0):>5.1f}% {100 * e['exposure']:>8.1f}%")
    return "\n".join(lines)


def choose_entries(session: Session, entries: pd.DataFrame, cfg: dict, roi_weight: Optional[float] = None,
                   diversity_weight: Optional[float] = None) -> dict:
    """
    Fill the entries file: for each contest in it, choose as many lineups as
    we hold entries, from that contest's simulation results. Returns the
    DK upload frame (the entry rows with F1..F6 filled as 'Name (id)'), a
    per-entry detail frame, every pick, and one record per contest.
    """
    players = {p.id: p for p in session.query(MMAPlayer).all()}
    upload_rows, detail_rows, picks_all, contests = [], [], [], []
    for cid, group in entries.groupby("Contest ID", sort=False):
        name = group["Contest Name"].iloc[0]
        n = int(len(group))
        rec = {"contest_id": int(cid), "name": name, "entries": n, "chosen": 0, "mean_cash": None, "mean_roi": None, "skipped": None}
        if session.query(LineupSimulationResult.id).filter(LineupSimulationResult.contest_name == name).first() is None:
            rec["skipped"] = "not simulated"
            contests.append(rec)
            continue
        picks = choose_lineups(session, name, cfg, num_lineups=n, roi_weight=roi_weight, diversity_weight=diversity_weight)
        detail = lineup_rows(session, picks)
        lineups = {lu.id: lu for lu in session.query(Lineup).filter(Lineup.id.in_([r.lineup_id for r in picks])).all()}
        for i, (r, (_, erow)) in enumerate(zip(picks, group.iterrows())):
            ids = list(lineups[r.lineup_id].player_ids_list)
            base = [erow[c] for c in BASE_COLS]
            upload_rows.append(base + [f"{players[p].name} ({p})" if p in players else str(p) for p in ids])
            detail_rows.append(dict(zip(BASE_COLS, base), **{"Lineup ID": r.lineup_id}, **detail.iloc[i].to_dict()))
        picks_all.extend(picks)
        rec.update(chosen=len(picks), mean_cash=float(detail["Cash %"].mean()), mean_roi=float(detail["Avg ROI"].mean()))
        contests.append(rec)
    return {"upload": pd.DataFrame(upload_rows, columns=BASE_COLS + SLOT_COLS),
            "detail": pd.DataFrame(detail_rows), "picks": picks_all, "contests": contests}


def export_entries(session: Session, chosen: dict, out_dir: Path = OUTPUT_DIR, stamp: Optional[str] = None,
                   base: str = "chosen_entries") -> dict[str, Path]:
    """Write the DK upload CSV, the per-entry detail CSV and the exposures over every chosen entry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    paths = {"upload": out_dir / f"{base}.{stamp}.csv", "lineups": out_dir / f"{base}.lineups.{stamp}.csv",
             "exposures": out_dir / f"{base}.exposures.{stamp}.csv"}
    chosen["upload"].to_csv(paths["upload"], index=False)
    chosen["detail"].to_csv(paths["lineups"], index=False)
    exposures(session, chosen["picks"]).to_csv(paths["exposures"], index=False)
    return paths


def format_contests(contests: list[dict]) -> str:
    lines = [f"{'contest':<60} {'entries':>7} {'chosen':>6} {'cash':>6} {'roi':>7}"]
    for c in contests:
        if c["skipped"]:
            lines.append(f"{c['name'][:60]:<60} {c['entries']:>7} {'-':>6} {c['skipped']}")
        else:
            lines.append(f"{c['name'][:60]:<60} {c['entries']:>7} {c['chosen']:>6} {c['mean_cash']:>5.1f}% {100 * c['mean_roi']:>+6.1f}%")
    return "\n".join(lines)
