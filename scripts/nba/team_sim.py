#!/usr/bin/env python3
"""
scripts/nba/team_sim.py

A team's game simulated from the slate's pool as it stands (ivan.sports.nba.desk.teamsim): what the team page's
Simulation view shows, in the terminal. Nothing is written.

    scripts/nba/team_sim.py --slate 156745 --team BOS [-n 2000] [--batch 250] [--seed 0] [--progress]
"""
from __future__ import annotations

import argparse

from ivan.sports.nba import slates as S
from ivan.sports.nba.desk import teamsim as TS


def table(v: dict) -> str:
    g = v["game"]
    line = lambda x: "no line" if x is None else f"{x:+.1f}"   # noqa: E731
    out = [f"{v['team']}: {v['n']} worlds · total {g['total']:.1f} (line {g['total_line']}) · margin {g['margin']:+.1f} (line {line(g['margin_line'])}) · wins {100 * g['p_win']:.0f}% · "
           f"blowout {100 * g['blowout']:.0f}% · overtime {100 * g['p_ot']:.0f}% · team fpts {g['team_fpts']:.1f}",
           f"{'player':24s} {'set':>5s} {'med':>5s} {'10-90':>11s} {'plays':>6s} {'<8':>5s} | {'fpts':>6s} {'sd':>5s} {'p10':>5s} {'p50':>5s} {'p90':>5s} {'5x':>5s} | {'pts':>5s} {'reb':>5s} {'ast':>5s}"]
    for r in v["rows"]:
        val = "" if r["value"] is None else f"{100 * r['value']:.0f}%"
        out.append(f"{r['name'][:24]:24s} {r['target']:5.1f} {r['min_med']:5.1f} {r['min_p10']:5.1f}-{r['min_p90']:<5.1f} {100 * r['plays']:5.0f}% {100 * r['under']:4.0f}% | "
                   f"{r['fpts']:6.1f} {r['sd']:5.1f} {r['p10']:5.1f} {r['p50']:5.1f} {r['p90']:5.1f} {val:>5s} | {r['pts']:5.1f} {r['reb']:5.1f} {r['ast']:5.1f}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("--team", required=True); ap.add_argument("-n", type=int, default=TS.DEFAULT_N); ap.add_argument("--batch", type=int, default=TS.BATCH)
    ap.add_argument("--seed", type=int); ap.add_argument("--progress", action="store_true", help="print the worlds drawn after each batch")
    a = ap.parse_args()
    dg, sess = S.open(a.slate)
    try:
        team = a.team.upper()
        r = TS.run(sess, team, n=a.n, batch=a.batch, seed=a.seed, on_batch=(lambda r: print(f"  {r.n} / {r.target} worlds")) if a.progress else None)
        s = r.summary[team]
        print(table(dict(team=team, n=r.n, rows=s["rows"], game=s["game"])))
    finally:
        sess.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
