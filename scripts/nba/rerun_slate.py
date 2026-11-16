#!/usr/bin/env python3
"""
scripts/nba/rerun_slate.py

What the current code says about a past slate (ivan.sports.nba.backtest.rerun): the World drawn from the slate's tip record
with the history cut at that day, scored against the actuals layer by layer, and written under
data/nba/slates/<dg>/rerun/<stamp>-<commit>/ for the backtest screen's "current code" block.

    scripts/nba/rerun_slate.py --slate 156184 [-n 1000] [--seed 0] [--tip 1800]     the first tip (the state at lock) unless --tip
    scripts/nba/rerun_slate.py --slate 156184 --list                                  the reruns on file
    scripts/nba/rerun_slate.py --slate 156184 --rotation stints --no-write             the same slate with the rotation drawn stint by stint
                                                                                      (gamesim.stints; --stints-model a tables file fitted as of the slate)
    scripts/nba/rerun_slate.py --compare plan,stints [--since 2026-11-01] [-n 400]     every slate with a tip record under each rotation, the scorecards pooled
                                                                                      side by side; nothing is written
"""
from __future__ import annotations

import argparse

import pandas as pd

from ivan.sports.nba import slates as S
from ivan.sports.nba.backtest import tips as T
from ivan.sports.nba.backtest import rerun as R


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("-n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tip", help="HHMM of the tip record to rerun from")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--rotation", choices=["plan", "stints"], help="the World's rotation draw for this run (default: config gamesim.rotation)")
    ap.add_argument("--stints-model", help="the stint tables to use (default data/nba/models/gamesim/stints.json)")
    ap.add_argument("--no-write", action="store_true"); ap.add_argument("--compare", help="comma-separated rotations to pool over every slate with a tip record")
    ap.add_argument("--since", help="with --compare: slates from this day")
    a = ap.parse_args()
    params = {}
    if a.rotation:
        params["rotation"] = a.rotation
    if a.stints_model:
        params["stints_path"] = a.stints_model
    if a.compare:
        return compare(a.compare.split(","), a.n, a.seed, a.since, params)
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
        r = R.rerun(sess, dg, n=a.n, seed=a.seed, tip=tip, log=print, params=params or None, write=not a.no_write)
    finally:
        sess.close()
    pd.set_option("display.width", 200)
    card = pd.DataFrame(r["meta"]["scorecard"])
    print(card[["layer", "metric", "n", "mae", "bias", "in_band", "pit", "mae_published", "note"]].round(3).to_string(index=False))
    for g in ([] if a.no_write else r["meta"]["games"]):
        act = g["actual"] or {}
        print(f"{g['away']} @ {g['home']}: drawn total {g['drawn_total']:.1f} (line {g['total']}) actual {act.get('pts_home', 0) + act.get('pts_away', 0):.0f}; "
              f"margin {g['drawn_margin']:+.1f} (spread {g['spread_home']}) actual {act.get('pts_home', 0) - act.get('pts_away', 0):+.0f}; poss {g['drawn_poss']:.1f} actual {act.get('poss', float('nan')):.1f}; "
              f"factors {g['factor']}")
    return 0


def compare(rotations, n, seed, since, params) -> int:
    """Every slate with a tip record under each rotation: the scorecard rows pooled over the slates (weighted by their n)."""
    from datetime import date
    rows = []
    for dg in sorted(int(p.name) for p in S.SLATES_DIR.iterdir() if p.name.isdigit()):
        tips = T.tips_of(dg)
        if not tips or (since and tips[0].date() < date.fromisoformat(since)):
            continue
        for rot in rotations:
            _, sess = S.open(dg)
            try:
                over = {"rotation": "stints", "stint_cuts": True} if rot == "stints-cuts" else {"rotation": rot}    # stints-cuts: with the soft pool's cut nights on top
                r = R.rerun(sess, dg, n=n, seed=seed, log=lambda *_: None, params={**params, **over}, write=False)
            except Exception as e:   # noqa: BLE001 - a slate that cannot be rerun is said and skipped
                print(f"slate {dg} ({rot}): {type(e).__name__}: {e}"); continue
            finally:
                sess.close()
            for c in r["meta"]["scorecard"]:
                rows.append(dict(slate=dg, day=r["meta"]["day"], rotation=rot, **c))
            print(f"slate {dg} {r['meta']['day']} {rot}: done")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no slates"); return 1
    both = df.groupby("slate").rotation.nunique(); df = df[df.slate.isin(both[both == len(rotations)].index)]
    pd.set_option("display.width", 220)
    w = lambda d, c: float((d[c] * d.n).sum() / d.n[d[c].notna()].sum()) if d[c].notna().any() else float("nan")   # noqa: E731
    out = df.groupby(["layer", "metric", "rotation"], sort=False)[["slate", "n", "mae", "bias", "in_band", "mae_published"]].apply(lambda d: pd.Series(dict(slates=d.slate.nunique(), n=int(d.n.sum()), mae=w(d, "mae"), bias=w(d, "bias"),
                                                                                                  in_band=w(d, "in_band"), mae_published=w(d, "mae_published")))).reset_index()
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
