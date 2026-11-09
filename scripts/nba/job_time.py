#!/usr/bin/env python3
"""
scripts/nba/job_time.py — how long a desk job will take (ivan.sports.nba.slate.job_time), from this machine's completed jobs.

    scripts/nba/job_time.py fit                                              the model per step: seconds per unit, the fixed part, runs, p90
    scripts/nba/job_time.py estimate --step sim --sims 1000 --field-cap 10000 --contests 6 [--slate DG]
    scripts/nba/job_time.py estimate --step generate --lineups 20000 --jobs 7
    scripts/nba/job_time.py estimate --step lateswap --candidates 10 --sims 1000 --field-cap 5000 --contests 6

With --slate (default: today's main held slate) the estimate is checked against the slate's lock (the late swap: its next tip),
as the desk does before it runs a job.
"""
from __future__ import annotations

import argparse
from datetime import datetime

from ivan.sports.nba.slate import job_time as JT
from ivan.sports.nba import slates as S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    e = sub.add_parser("estimate")
    e.add_argument("--step", required=True, choices=JT.STEPS)
    for k in ("lineups", "jobs", "sims", "field_cap", "candidates"):
        e.add_argument(f"--{k.replace('_', '-')}", default=None)
    e.add_argument("--contests", type=int, default=None, help="contests in the entries file (sim, late swap)")
    e.add_argument("--margin", type=float, default=JT.MARGIN_MIN, help="minutes the run must leave before lock")
    S.add_arg(e)
    a = ap.parse_args()
    model = JT.fit()
    if a.cmd == "fit":
        print(f"{'step':16s} {'runs':>4s} {'basis':8s} {'fixed s':>8s} {'s/unit':>9s} {'p90 x':>6s}   unit")
        unit = {"generate": "lineups per worker", "sim": "contests x sims x field cap / 1e6", "lateswap": "contests x candidates x sims x field cap / 1e6"}
        for step, m in model.items():
            print(f"{step:16s} {m['n']:4d} {m['basis']:8s} {m['fixed']:8.0f} {m['per_unit']:9.2f} {m['p90_ratio']:6.2f}   {unit.get(step, 'one run')}")
        return 0
    knobs = {k: getattr(a, k) for k in ("lineups", "jobs", "sims", "field_cap", "candidates") if getattr(a, k) is not None}
    est = JT.estimate(a.step, knobs, a.contests, model)
    deadline = None; label = "lock"
    try:
        dg = S.resolve(str(a.slate) if a.slate else None)
        lock = S.meta_time(S.read_meta(dg), "lock")
        deadline = lock
        if a.step == "lateswap" and lock is not None:
            with S.using(dg):
                from ivan.sports.nba.models import NBAPlayer
                sess = S.session(dg)
                tips = sorted({t for (t,) in sess.query(NBAPlayer.gametime).distinct().all() if t and t > datetime.now()})
                sess.close()
            deadline = tips[0] if tips else None; label = "next tip"
    except S.NoSlate:
        pass
    verdict, msg = JT.deadline_check(est, deadline, a.margin, label)
    print(f"{a.step}: {msg}")
    print(f"  work {est['units']:.3g} units, {est['basis']} on {est['n']} run(s); verdict {verdict}")
    return 0 if verdict != "late" else 1


if __name__ == "__main__":
    raise SystemExit(main())
