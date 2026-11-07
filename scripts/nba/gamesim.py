#!/usr/bin/env python3
"""
scripts/nba/gamesim.py

The NBA game simulation's fitting data and checks (ivan.sports.nba.gamesim; plan docs/nba/notes/game_sim_plan_2026-11-07.md).

    scripts/nba/gamesim.py events [--since 2026-10-01] [--rebuild] [--max-games 50]
        parse every game with stints and a stored feed into data/nba/derived/gamesim/events.parquet (incremental)
    scripts/nba/gamesim.py check [--game-ids 0022500001 ...]
        the parse against the box score: per-stat exact share and gap, counted vs estimated possessions, both teams' counts
"""
from __future__ import annotations

import argparse

import pandas as pd
from sqlalchemy import text

from ivan.database.session import get_session, init_db
from ivan.sports.nba.gamesim import possessions as X


def cmd_events(args) -> None:
    s = get_session(init_db("nba"))
    X.build(s, since=args.since, rebuild=args.rebuild, max_games=args.max_games, log=print)


def cmd_check(args) -> None:
    s = get_session(init_db("nba"))
    ev = X.load_events()
    if args.game_ids:
        ev = ev[ev.game_id.isin(args.game_ids)]
    if ev.empty:
        print("no events stored for those games: run `events` first"); return
    ids = ",".join(repr(str(g)) for g in ev.game_id.unique())
    box = pd.read_sql(text(f"select * from nba_stats where game_id in ({ids})"), s.get_bind())
    r = X.reconcile(ev, box)
    print(r["players"].round(3).to_string(index=False))
    for k, v in r["summary"].items():
        print(f"{k:>24}: {v:.3f}" if isinstance(v, float) else f"{k:>24}: {v}")
    tg = r["teams"]
    print(f"team-games with the two teams' possessions more than 3 apart: {int((tg.poss - tg.poss_opp).abs().gt(3).sum())} of {len(tg)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("events"); e.add_argument("--since"); e.add_argument("--rebuild", action="store_true"); e.add_argument("--max-games", type=int); e.set_defaults(fn=cmd_events)
    c = sub.add_parser("check"); c.add_argument("--game-ids", nargs="*"); c.set_defaults(fn=cmd_check)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
