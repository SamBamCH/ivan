#!/usr/bin/env python3
"""
scripts/nfl/compare_choosers.py

Run several chooser settings on one slate's existing simulation results and
settle each book against the real fields. Nothing before the choose stage is
re-run: the candidates, their simulated worlds and the field all stay the
same, so the books differ only by how they were picked.

    scripts/nfl/compare_choosers.py --entries data/nfl/input/entries.csv --artifacts data/nfl/output \\
        --out data/nfl/output/chooser_ab/2026-10-15 --stamp 20261015-190000 \\
        --config greedy --config portfolio:1.2 --config portfolio:3

    scripts/nfl/compare_choosers.py --db-url sqlite:///data/nfl/output/backtests/X/replay.w0.sqlite \\
        --entries data/nfl/output/backtests/X/nfl/<slate>/entries.csv --artifacts data/nfl/output/backtests/X/nfl/<slate>/rep_002 ...

--db-url is the database holding lineup_simulation_results for the slate (the
live one by default; a replay's replay.w0.sqlite for a replayed slate).
--artifacts is the directory holding the sim/<stamp>/ artifacts and the
projections file of that simulation; it is linked into every book's directory
so the validation finds them. --stamp must be before the slate's lock, else
the validation treats the book as a post-lock run and skips it.
Each --config is "greedy" or "portfolio[:bankroll]"; weights (cash_pct,
avg_roi_win, diversity_lambda) can be appended as key=value pairs, e.g.
"greedy:diversity_lambda=1".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from ivan.database.session import get_session, init_db
from ivan.sports.nfl import lineup_export as LE
from ivan.sports.nfl.validate import validate_window


def parse_config(text: str) -> tuple[str, dict]:
    head, *rest = text.split(":")
    w = dict(LE.DEFAULT_WEIGHTS)
    w["method"] = head.strip().lower()
    label = w["method"]
    for part in rest:
        if "=" in part:
            k, v = part.split("=", 1)
            w[k.strip()] = float(v)
            label += f"_{k.strip()}{v}"
        elif part.strip():
            w["bankroll"] = float(part)
            label += f"_b{part.strip()}"
    return label, w


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", required=True)
    ap.add_argument("--artifacts", required=True, help="directory with sim/<stamp>/ and projections.*.csv of the simulation the entries were simulated by")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stamp", required=True, help="YYYYMMDD-HHMMSS before the lock, used as the choose run's stamp")
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--config", action="append", default=[], help='"greedy" or "portfolio[:bankroll]" (+ ":key=value" weights); repeatable')
    ap.add_argument("--in-field", default="no", choices=("auto", "yes", "no"), help="whether the scraped fields already hold these books (no: they are new)")
    a = ap.parse_args()
    configs = [parse_config(c) for c in (a.config or ["greedy", "portfolio"])]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    art = Path(a.artifacts)
    sims = sorted((art / "sim").glob("*")) if (art / "sim").is_dir() else []
    # the projections file sits beside the sim artifacts (live: output/projections/*.csv; replay: the slate directory above the repeat)
    projs = sorted(art.glob("projections*.csv")) or sorted((art / "projections").glob("projections*.csv")) or sorted(art.parent.glob("projections*.csv"))
    if not sims or not projs:
        print(f"no sim/<stamp> or projections file under {art}", file=sys.stderr)
        return 2

    session = get_session(init_db("nfl", a.db_url))
    results = []
    for label, weights in configs:
        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        (d / "sim").mkdir(exist_ok=True)
        for s in sims:
            link(s, d / "sim" / s.name)
        for p in projs:
            link(p, d / p.name)
        print(f"\n##### {label}: {weights}")
        run, paths = LE.run_choose(session, a.entries, d / "chosen_entries.csv", LE.weights_resolver(overrides={"default": weights}),
                                   stamp=a.stamp)
        rep = validate_window(session, [d], d / "validation", in_field=a.in_field, include_snapshots=False, title=label, echo=lambda *x: None)
        lu = pd.read_csv(d / "validation" / "lineups.csv")
        for cid, g in lu.groupby("contest_id"):
            g = g.dropna(subset=["actual_points"])
            if g.empty:
                continue
            fees = float(g.entry_fee.sum())
            results.append({"book": label, "contest_id": int(cid), "contest": g.contest_name.iloc[0], "entries": len(g), "fees": fees,
                            "sim_roi": float(g.sim_avg_roi.mean()), "sim_cash": float(g.sim_cash_pct.mean()),
                            "actual_payout": float(g.actual_payout.sum()), "actual_roi": float(g.actual_payout.sum() / fees - 1) if fees else None,
                            "cashed": int((g.actual_payout > 0).sum()), "median_points": float(g.actual_points.median()),
                            "best_rank": int(g.actual_rank.min()) if g.actual_rank.notna().any() else None,
                            "pred_dupes": float(g.pred_dupes.mean()) if g.pred_dupes.notna().any() else None,
                            "actual_dupes": float(g.actual_dupes.mean()) if g.actual_dupes.notna().any() else None})
    df = pd.DataFrame(results)
    if df.empty:
        print("nothing settled: are the contests scraped and the stamp before the lock?", file=sys.stderr)
        return 1
    df.to_csv(out / "books.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n== Books settled against the real fields")
    print(df[["book", "contest", "entries", "fees", "sim_roi", "actual_roi", "cashed", "median_points", "best_rank", "actual_dupes"]].round(3).to_string(index=False))
    tot = df.groupby("book").agg(fees=("fees", "sum"), payout=("actual_payout", "sum"), entries=("entries", "sum"), cashed=("cashed", "sum"))
    tot["actual_roi"] = tot.payout / tot.fees - 1
    tot["cash_rate"] = tot.cashed / tot.entries
    print("\n== Totals"); print(tot.round(3).to_string())
    (out / "summary.json").write_text(json.dumps({"configs": {l: w for l, w in configs}, "totals": tot.reset_index().to_dict("records")}, indent=1, default=str))
    print(f"\nwrote {out}/books.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
