#!/usr/bin/env python3
"""
scripts/nfl/build_ownership_calibration.py

Fit the ownership calibration lines from the scraped showdown contests and write
ownership_calibration.json (ivan.sports.nfl.ownership_model.fit_calibration; the
ownership step applies it after the field-sampling projection). The same step
build_models.py runs as "ownership_calibration".

    build_ownership_calibration.py [--cutoff YYYY-MM-DD] [--model-dir DIR] [--min-n 30]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ivan.database.session import get_session, init_db
from ivan.sports.nfl.model_build import build_ownership_calibration
from ivan.sports.nfl.ownership_model import CALIBRATION_FILE
from ivan.sports.nfl.paths import MODELS_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default=None); ap.add_argument("--model-dir", default=None); ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--db-url", default=None)
    a = ap.parse_args()
    session = get_session(init_db("nfl", a.db_url))
    out = Path(a.model_dir or MODELS_DIR)
    build_ownership_calibration(session, out, a.cutoff, a.min_n)
    cal = json.loads((out / CALIBRATION_FILE).read_text())
    for k, m in sorted(cal["maps"].items()):
        print(f"  {k:8s} on {m['on']:14s} actual = {m['a']:+.2f} + {m['b']:.2f} x   n {m['n']:4d} corr {m['corr']:.2f}   proj mean {m['proj_mean']:5.1f} actual mean {m['actual_mean']:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
