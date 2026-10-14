#!/usr/bin/env python3
"""
scripts/nfl/build_dupe_model.py

The duplication model: copies(L) = N x k x prod(own_i / 100)^beta, fitted on the real
fields the settle job keeps per contest (ivan.sports.nfl.dupe_model).

    build_dupe_model.py backtest [--db-url URL]
        leave-one-slate-out fit per format, printed
    build_dupe_model.py build [--cutoff YYYY-MM-DD] [--model-dir DIR] [--db-url URL]
        fit (on contests before the cutoff when given) and write dupe_model_v2.json
"""
import argparse
from pathlib import Path

from ivan.database.session import get_session, init_db
from ivan.sports.nfl import dupe_model as DM
from ivan.sports.nfl.paths import MODELS_DIR


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit the product-rule duplication model on scraped fields")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest"); b.add_argument("--db-url", default=None)
    c = sub.add_parser("build"); c.add_argument("--cutoff", default=None); c.add_argument("--model-dir", default=None); c.add_argument("--db-url", default=None)
    a = ap.parse_args(argv)
    session = get_session(init_db("nfl", a.db_url))
    if a.cmd == "backtest":
        df = DM.training_frame(session)
        print(f"{df.contest_id.nunique()} fields, {len(df)} unique lineups")
        r = DM.backtest(df)
        for fmt, v in r.items():
            print(f"{fmt}: {v['contests']} fields; fit {v['fit']}")
            for f in v["folds"]:
                print(f"   held-out {f['held_out']} ({f['contests']}): k {f['k']} beta {f['beta']}  copies/entry pred {f['copies_per_entry_pred']:6.1f} actual {f['copies_per_entry_actual']:6.1f}  "
                      f"log MAE {f['log_mae']:.2f} bias {f['log_bias']:+.2f}  top ratio {f['top_lineup_ratio_median']:.2f}")
        return 0
    df = DM.training_frame(session, cutoff=a.cutoff)
    m = DM.fit(df, cutoff=a.cutoff)
    m.cv = {k: {"contests": v["contests"], "folds": v["folds"]} for k, v in DM.backtest(df).items()} if len(df) else {}
    out = Path(a.model_dir or MODELS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = m.save(out)
    print(f"dupe model -> {path}: {m.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
