#!/usr/bin/env python3
"""
scripts/nfl/slate_runs.py

Every replay of a slate across the runs on disk (ivan.sports.nfl.run_index):
when it ran, the commit, the stage, the parameters, the models version, what
the sim claimed and what happened. The same table the Backtest hub's slate
page (/backtest/slate/<key>) shows.

    slate_runs.py 154487               # one slate
    slate_runs.py --list               # every slate any run replayed, with its run count
    slate_runs.py 154487 --json
"""
from __future__ import annotations

import argparse
import json

from ivan.sports.nfl import run_index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slate_key", nargs="?", help="the DraftKings draft group id (the slate's key)")
    ap.add_argument("--list", action="store_true", help="list the slates with runs")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.list or not a.slate_key:
        rows = run_index.slates()
        if a.json:
            print(json.dumps(rows, indent=1))
            return 0
        print(f"{'lock':17s} {'mode':9s} {'slate':8s} {'runs':>4s} {'ok':>3s}  last")
        for s in rows:
            print(f"{s['lock'][:16].replace('T', ' '):17s} {s['mode']:9s} {s['slate_key']:8s} {s['runs']:4d} {s['ok']:3d}  {s['last'][:16].replace('T', ' ')}")
        return 0
    rows = run_index.slate_runs(a.slate_key)
    if a.json:
        print(json.dumps([r.as_dict() for r in rows], indent=1, default=str))
        return 0
    if not rows:
        print(f"no run has replayed slate {a.slate_key}")
        return 1
    r0 = rows[0]
    print(f"slate {r0.slate_key} {r0.mode} lock {r0.lock[:16].replace('T', ' ')}: {len(rows)} run(s)")
    print(f"{'started':17s} {'kind':22s} {'run':32s} {'commit':8s} {'through':8s} {'status':7s} {'reps':>5s} {'models':10s} {'lineups':>7s} {'sims':>5s} {'method':9s} {'fees':>8s} {'sim':>9s} {'actual':>9s} {'mean':>9s}")
    def money(x, w=9):
        return f"{x:{w}.2f}" if x is not None else f"{'-':>{w}s}"
    for r in rows:
        print(f"{r.started_at[:16].replace('T', ' '):17s} {r.kind:22s} {r.run[:32]:32s} {(r.commit + ('*' if r.dirty else '')):8s} {r.through:8s} {r.status:7s} "
              f"{r.repeats_ok:2d}/{r.repeats_wanted:<2d} {(r.models or '-'):10s} {str(r.options.get('n_lineups') or '-'):>7s} {str(r.options.get('n_sims') or '-'):>5s} "
              f"{(r.method or '-'):9s} {money(r.fees, 8)} {money(r.sim)} {money(r.actual)} {money(r.repeat_mean)}"
              + (f"  purged {r.purged_at[:10]}" if r.purged_at else "") + ("  pinned" if r.pinned else "") + (f"  {r.error[:60]}" if r.error else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
