#!/usr/bin/env python3
"""
scripts/nba/rerun_slate.py

What the current code says about a past slate (ivan.sports.nba.gamesim.rerun): the World drawn from the slate's tip record
with the history cut at that day, scored against the actuals layer by layer, and written under
data/nba/slates/<dg>/rerun/<stamp>-<commit>/ for the backtest screen's "current code" block.

    scripts/nba/rerun_slate.py --slate 156184 [-n 1000] [--seed 0] [--tip 1800]     the first tip (the state at lock) unless --tip
    scripts/nba/rerun_slate.py --slate 156184 --list                                  the reruns on file
"""
from __future__ import annotations

import argparse

import pandas as pd

from ivan.sports.nba import slates as S
from ivan.sports.nba import tips as T
from ivan.sports.nba.gamesim import rerun as R


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("-n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tip", help="HHMM of the tip record to rerun from")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    dg, sess = S.open(a.slate)
    if a.list:
        for name in R.list_reruns(dg):
            m = R.load(dg, name)["meta"]
            print(f"{name}: tip {m['tip']} n {m['n']} commit {m['commit']}" + ("*" if m.get("dirty") else "") + f" {m.get('subject', '')[:60]}")
        print("tips on file: " + ", ".join(t.strftime("%m-%d %H:%M") for t in T.tips_of(dg)))
        return 0
    tip = None
    if a.tip:
        tip = next((t for t in T.tips_of(dg) if t.strftime("%H%M") == a.tip), None)
        if tip is None:
            print(f"no tip record at {a.tip}; on file: " + ", ".join(t.strftime("%H%M") for t in T.tips_of(dg))); return 1
    try:
        r = R.rerun(sess, dg, n=a.n, seed=a.seed, tip=tip, log=print)
    finally:
        sess.close()
    pd.set_option("display.width", 200)
    card = pd.DataFrame(r["meta"]["scorecard"])
    print(card[["layer", "metric", "n", "mae", "bias", "in_band", "pit", "mae_published", "note"]].round(3).to_string(index=False))
    for g in r["meta"]["games"]:
        act = g["actual"] or {}
        print(f"{g['away']} @ {g['home']}: drawn total {g['drawn_total']:.1f} (line {g['total']}) actual {act.get('pts_home', 0) + act.get('pts_away', 0):.0f}; "
              f"margin {g['drawn_margin']:+.1f} (spread {g['spread_home']}) actual {act.get('pts_home', 0) - act.get('pts_away', 0):+.0f}; poss {g['drawn_poss']:.1f} actual {act.get('poss', float('nan')):.1f}; "
              f"factors {g['factor']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
