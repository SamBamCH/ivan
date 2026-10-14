#!/usr/bin/env python3
"""
scripts/nfl/backtest_projections.py

Score stored projections against what the players actually did
(ivan.sports.nfl.backtest_projections): bias, error, rank correlation and whether the
stated sd covers the outcomes, per position and by projection tercile.

    backtest_projections.py [--since 2025-09-01] [--until DATE] [--positions K,DST] [--write-calibration]

--write-calibration sets each listed position's predictive-sd factor in
sd_calibration.json to the sd of z = (actual - projected) / model sd over the window
(the step build_models.py runs as "sd"), so the next ingest's spread covers the
outcomes the way the DST residual pool already does.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ivan.database.session import add_missing_columns, get_session, init_db
from ivan.sports.nfl import backtest_projections as BP
from ivan.sports.nfl.model_build import build_sd_calibration
from ivan.sports.nfl.models import NFLBase
from ivan.sports.nfl.paths import MODELS_DIR
from ivan.sports.nfl.refactor.spread import load_sd_calibration


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2025-09-01")
    ap.add_argument("--until", default=str(date.today()))
    ap.add_argument("--positions", default=",".join(BP.POSITIONS))
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--model-dir", default=None, help="where to read/write sd_calibration.json (default the live model directory)")
    ap.add_argument("--write-calibration", action="store_true",
                    help="write each listed position's z sd (on the model sd) to sd_calibration.json")
    a = ap.parse_args()
    positions = [p.strip().upper() for p in a.positions.split(",") if p.strip()]

    engine = init_db("nfl", a.db_url)
    add_missing_columns(engine, NFLBase.metadata)
    session = get_session(engine)
    try:
        scored = BP.scored_projections(session, a.since, a.until, positions)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"snapshots {a.since}..{a.until}: {len(scored)} rows, {scored.actual.notna().sum()} with an actual "
          f"(a missing actual is a bye, an inactive, or a name nfl_stats spells differently)")
    for pos in positions:
        print(BP.summarize(scored[scored.position == pos], pos))

    model_dir = Path(a.model_dir) if a.model_dir else MODELS_DIR
    if a.write_calibration:
        print()
        build_sd_calibration(session, model_dir, a.since, a.until, positions)
    else:
        cal = load_sd_calibration(Path(a.model_dir) if a.model_dir else None)
        if cal.get("factors"):
            print(f"\npredictive sd factors in effect for the next ingest: {cal['factors']} (written {cal.get('fitted_on')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
