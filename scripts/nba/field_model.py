#!/usr/bin/env python3
"""
scripts/nba/field_model.py

The field ownership model (ivan.sports.nba.field_model), fitted on real contests only.

    scripts/nba/field_model.py validate     leave-one-slate-out on every derived contest (frozen features vs baselines)
    scripts/nba/field_model.py fit          fit on every derived contest and save data/nba/models/field_ownership.json
    scripts/nba/field_model.py check        the saved model on slates it was not fitted on (the honest test; run after a scrape)
"""
import argparse

import pandas as pd

from ivan.database.session import get_session, init_db
from ivan.sports.nba import field_model as FM

COLS = ["mae", "mae_top20", "spearman", "top10_overlap", "max_own_err"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("validate", "fit", "check"))
    a = ap.parse_args()
    s = get_session(init_db("nba"))
    pd.set_option("display.width", 200)
    if a.what == "check":
        r = FM.check_prospective(s)
        if r.empty:
            print("no contests on slates the model has not seen; scrape newer slates first"); return 0
        print(r.round(2).to_string(index=False)); print("mean:", r[COLS].mean().round(2).to_dict()); return 0
    df = FM.dataset(s)
    print(f"{len(df)} player rows, {df.contest_id.nunique()} contests, {df.slate.nunique()} slates")
    if a.what == "validate":
        r = FM.loso(df)
        print(r.groupby("variant")[COLS].mean().round(2).to_string())
        return 0
    m = FM.fit(df)
    print("saved", FM.save(m)); print({f: round(t, 2) for f, t in zip(m["features"], m["theta"])}, "temperature", [round(g, 3) for g in m["gamma"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
