#!/usr/bin/env python3
"""
Backtest and build the learned showdown ownership model
(ivan.sports.nfl.ownership_model) from the scraped fields.

    scripts/nfl/build_ownership_model.py backtest [--split 2026-01-01 | --test-share 0.3]
    scripts/nfl/build_ownership_model.py build [--cutoff YYYY-MM-DD] [--model-dir DIR]

The backtest scores, on held-out contests, the sampler alone, the sampler
with the per-position calibration, and the learned model, on the same
ownership checks the validation uses (MAE, Spearman, chalk bias). Build
writes ownership_model.pkl to the model directory (an as-of directory with
--cutoff, trained only on contests before it).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ivan.database.session import get_session, init_db
from ivan.sports.nfl import ownership_model as OM
from ivan.sports.nfl.paths import MODELS_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest"); b.add_argument("--split", default=None); b.add_argument("--test-share", type=float, default=0.3); b.add_argument("--db-url", default=None)
    c = sub.add_parser("build"); c.add_argument("--cutoff", default=None); c.add_argument("--model-dir", default=None); c.add_argument("--db-url", default=None)
    a = ap.parse_args()
    session = get_session(init_db("nfl", a.db_url))
    if a.cmd == "backtest":
        df = OM.training_frame(session)
        print(f"{df.contest_id.nunique()} contests, {len(df)} rows, {df.date.min().date()} .. {df.date.max().date()}")
        r = OM.backtest(df, split_date=a.split, test_share=a.test_share)
        if "error" in r:
            print(r["error"]); return 1
        print(f"train {r['train_contests']} contests to {r['train_until']}; test {r['test_contests']} contests from {r['test_from']}")
        for k in ("sampler", "sampler_calibrated", "learned"):
            s = r[k]
            print(f"  {k:19s} MAE {s['mae']:.2f}  bias {s['bias']:+.2f}  Spearman {s['spearman']:.3f}  chalk bias {s['chalk_bias']:+.2f}")
        print("  by slot/position (actual | sampler | calibrated | learned; MAE calibrated | learned):")
        for k, v in sorted(r["by_slot_position"].items()):
            print(f"    {k:8s} n {v['n']:4d}  {v['actual']:5.1f} | {v['sampler']:5.1f} | {v['calibrated']:5.1f} | {v['learned']:5.1f}   {v['mae_calibrated']:.2f} | {v['mae_learned']:.2f}")
        if r.get("importances"):
            print("  importances: " + ", ".join(f"{f} {v:.2f}" for f, v in list(r["importances"].items())[:8]))
        return 0
    df = OM.training_frame(session, a.cutoff)
    if df.contest_id.nunique() < 20:
        print(f"only {df.contest_id.nunique()} contests before the cutoff; not building"); return 1
    m = OM.fit(df, cutoff=a.cutoff)
    path = m.save(Path(a.model_dir or MODELS_DIR))
    print(f"ownership model on {m.n_contests} contests ({m.n_rows} rows) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
