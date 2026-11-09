#!/usr/bin/env python3
"""
scripts/nba/optimal.py

The optimal lineup in each of the World's worlds and every player's share of them (ivan.sports.nba.field.optimal): the desk's
Solve button. Reads the per-world fantasy points the projection store holds for every team (written when the team was projected),
solves the salary-capped optimal in each world with the generator's solver, writes slates/<dg>/projection/optimal.json (the
desk's **opt** column) and prints the top of the table. About 0.1 s a world per worker.

    scripts/nba/optimal.py [--slate DG] [-n 100] [-j 4] [--min-fpts 15] [--top 30]
"""
import argparse
import time

from ivan.core.config import load_config
from ivan.sports.nba import slates as S
from ivan.sports.nba.field import optimal as OPT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("-n", "--worlds", type=int, default=OPT.DEFAULT_WORLDS, help="worlds to solve (default 100; at most the samples on file)")
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--min-fpts", type=float, default=None, help="pool floor (default config generate.player_min_fpts)")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    dg, session = S.open(a.slate)
    t0 = time.time()
    r = OPT.optimal_shares(session, n_worlds=a.worlds, jobs=a.jobs, config=load_config("nba"), log=lambda m: print(m, flush=True), player_min_fpts=a.min_fpts)
    p = OPT.write(r)
    rows = sorted(r["players"].items(), key=lambda kv: -kv[1]["share"])
    print(f"\n{'player':28s} {'team':5s} {'opt %':>6s} {'mean opt':>8s}")
    for pid, x in rows[:a.top]:
        print(f"{x['name'][:28]:28s} {str(x['team'] or ''):5s} {x['share']:6.1f} {'  yes' if x['in_mean'] else '':>8s}")
    if r.get("mean_optimal"):
        print(f"\nmean-projection optimal: {r['mean_optimal']['total']} fpts: {', '.join(r['mean_optimal']['names'])}")
    print(f"{r['n_worlds']} worlds solved in {time.time() - t0:.1f}s -> {p}", flush=True)
    return 0 if r["n_worlds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
