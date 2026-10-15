#!/usr/bin/env python3
"""
scripts/nfl/build_dupe_model.py

The duplication model: copies(L) = N x k x prod(own_i / 100)^beta, fitted on the real
fields the settle job keeps per contest (ivan.sports.nfl.dupe_model).

    build_dupe_model.py backtest [--db-url URL]
        leave-one-slate-out fit per format, printed
    build_dupe_model.py build [--cutoff YYYY-MM-DD] [--model-dir DIR] [--db-url URL]
        fit (on contests before the cutoff when given) and write dupe_model_v2.json
        (the same step build_models.py runs as "dupe")
"""
import argparse
from pathlib import Path

from ivan.database.session import get_session, init_db
from ivan.sports.nfl import dupe_model as DM
from ivan.sports.nfl.model_build import build_dupe_model
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
                print(f"   held-out {f['held_out']} ({f['contests']}): k {f['k']} beta {f['beta']} cpt {f['beta_cpt']} gamma {f['gamma']} proj {f.get('delta_proj')} logproj {f.get('delta_logproj')} stack {f['stack']}")
                print(f"      copies/entry pred {f['copies_per_entry_pred']:6.1f} actual {f['copies_per_entry_actual']:6.1f}; held-out by terms/method "
                      f"(MAE | bias | top-lineup ratio | log slope, 1 = calibrated):")
                for name, a in f["ablation"].items():
                    print(f"         {name:16s} {a['log_mae']:.2f} | {a['log_bias']:+.2f} | {a['top_lineup_ratio_median']:.2f} | {a['log_slope']:.2f}")
        return 0
    out = Path(a.model_dir or MODELS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    build_dupe_model(session, out, cutoff=a.cutoff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
