#!/usr/bin/env python3
"""
Fit the ownership calibration from scraped showdown contests and write
data/nfl/models/ownership_calibration.json (ivan.sports.nfl.ownership
applies it after the field-sampling projection).

Per roster slot (511 captain, 512 flex) and position, a line from what we
projected to what the field actually did (actual ~ a + b * projected). For
DST and K the public rosters them by their projected points rather than by
value, so their line runs on projected points instead (actual ~ a + b *
fpts_proj), which fits far better (DST flex: corr 0.89 vs 0.74).

    scripts/nfl/build_ownership_calibration.py [--cutoff YYYY-MM-DD] [--model-dir DIR] [--min-n 30]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import text

from ivan.database.session import get_session, init_db, read_sql
from ivan.sports.nfl.paths import MODELS_DIR

FILE = "ownership_calibration.json"
POINTS_POSITIONS = ("DST", "K")       # calibrated on projected points, not projected ownership


def fit(session, cutoff: str | None = None, min_n: int = 30) -> dict:
    q = """SELECT p.contest_id, p.position, p.roster_slot_id AS slot, p.ownership_proj, p.ownership_actual, p.fpts_proj
           FROM showdown_player p JOIN historical_contest_information h ON h.contest_id = p.contest_id
           WHERE p.ownership_proj IS NOT NULL AND p.ownership_actual IS NOT NULL"""
    if cutoff:
        q += " AND h.date < :c"
    stmt = text(q)
    df = read_sql(session, stmt.bindparams(c=cutoff) if cutoff else stmt)
    out = {"fitted_on": datetime.now().strftime("%Y-%m-%d"), "cutoff": cutoff, "contests": int(df.contest_id.nunique()) if not df.empty else 0,
           "rows": int(len(df)), "maps": {}}
    for (slot, pos), g in df.groupby(["slot", "position"]):
        if len(g) < min_n:
            continue
        x = g.fpts_proj.values.astype(float) if pos in POINTS_POSITIONS else g.ownership_proj.values.astype(float)
        y = g.ownership_actual.values.astype(float)
        if x.std() <= 0:
            continue
        b, a = np.polyfit(x, y, 1)
        out["maps"][f"{int(slot)}:{pos}"] = {"on": "fpts_proj" if pos in POINTS_POSITIONS else "ownership_proj", "a": round(float(a), 4),
                                             "b": round(float(b), 4), "n": int(len(g)), "corr": round(float(np.corrcoef(x, y)[0, 1]), 3),
                                             "proj_mean": round(float(g.ownership_proj.mean()), 2), "actual_mean": round(float(y.mean()), 2)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default=None); ap.add_argument("--model-dir", default=None); ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--db-url", default=None)
    a = ap.parse_args()
    session = get_session(init_db("nfl", a.db_url))
    cal = fit(session, a.cutoff, a.min_n)
    out = Path(a.model_dir or MODELS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / FILE).write_text(json.dumps(cal, indent=1))
    print(f"ownership calibration from {cal['contests']} contests ({cal['rows']} rows) -> {out / FILE}")
    for k, m in sorted(cal["maps"].items()):
        print(f"  {k:8s} on {m['on']:14s} actual = {m['a']:+.2f} + {m['b']:.2f} x   n {m['n']:4d} corr {m['corr']:.2f}   proj mean {m['proj_mean']:5.1f} actual mean {m['actual_mean']:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
