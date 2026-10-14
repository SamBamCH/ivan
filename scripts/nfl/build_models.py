#!/usr/bin/env python3
"""
scripts/nfl/build_models.py

Build the models the pipeline reads from data/nfl/models/ (ivan.sports.nfl.model_build):
the DST game table and DST odds fit, the classic field models, the dupe model, the
predictive-sd calibration and the two ownership models.

    build_models.py                          live models, from everything in the database
    build_models.py --cutoff 2025-11-01      an as-of version into data/nfl/models/asof/2025-11-01/,
                                             trained only on data before the cutoff (what a replay may use)
    build_models.py --only dst,classic       a subset (dst_table, dst, classic, dupe, sd, ownership_calibration, ownership_model)
"""
import argparse
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
np.seterr(all="ignore")

from ivan.database.session import get_session, init_db
from ivan.sports.nfl import model_build


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db-url', default=None)
    ap.add_argument('--model-dir', default=None, help="output directory (default the live model directory, or asof/<cutoff> with --cutoff)")
    ap.add_argument('--cutoff', default=None, help="build an as-of version from data before this date (YYYY-MM-DD)")
    ap.add_argument('--only', default=None, help=f"comma-separated subset of {','.join(model_build.STEPS)}")
    ap.add_argument('--sd-since', default=model_build.SD_SINCE, help="first snapshot day the sd calibration scores")
    a = ap.parse_args()
    session = get_session(init_db("nfl", a.db_url))
    m = model_build.build(session, cutoff=date.fromisoformat(a.cutoff) if a.cutoff else None,
                          out_dir=Path(a.model_dir) if a.model_dir else None,
                          only=[s.strip() for s in a.only.split(",") if s.strip()] if a.only else None,
                          sd_since=a.sd_since)
    return 1 if m.get("failed") else 0


if __name__ == '__main__':
    raise SystemExit(main())
