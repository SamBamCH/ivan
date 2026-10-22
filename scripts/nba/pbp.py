#!/usr/bin/env python3
"""
scripts/nba/pbp.py

Play-by-play stints (ivan.sports.nba.pbp): when each player was on the floor.

    scripts/nba/pbp.py fetch [--since 2026-10-01] [--max-games 60]    games in nba_stats without stints (the nightly job's call)
    scripts/nba/pbp.py fetch --game-ids 0022600002
    scripts/nba/pbp.py rotation --team NYK [--games 10] [--before 2026-10-20]
    scripts/nba/pbp.py check --game-ids 0022600002                  stint minutes vs the box score, per player

nba_api allows roughly one request a second; a full season is ~20 minutes.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date

import pandas as pd
from sqlalchemy import text

from ivan.database.session import get_session, init_db
from ivan.sports.nba import pbp as P


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--since", default=None); f.add_argument("--max-games", type=int, default=60); f.add_argument("--game-ids", default=None)
    f.add_argument("--refetch", action="store_true"); f.add_argument("--sleep", type=float, default=0.7)
    r = sub.add_parser("rotation"); r.add_argument("--team", required=True); r.add_argument("--games", type=int, default=10); r.add_argument("--before", default=None)
    c = sub.add_parser("check"); c.add_argument("--game-ids", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    engine = init_db("nba"); P.ensure_tables(engine); session = get_session(engine)

    if a.cmd == "fetch":
        ids = [x.strip() for x in a.game_ids.split(",")] if a.game_ids else P.games_without_stints(session, date.fromisoformat(a.since) if a.since else None)[: a.max_games]
        print(f"{len(ids)} game(s) to process")
        ok = 0
        for i, gid in enumerate(ids, 1):
            try:
                P.process_game(session, gid, log=print, refetch=a.refetch); ok += 1
            except Exception as e:   # noqa: BLE001
                print(f"{gid}: {type(e).__name__}: {e}")
            if i < len(ids):
                time.sleep(a.sleep)
        print(f"done: {ok} of {len(ids)}")
        return 0
    if a.cmd == "rotation":
        rm = P.rotation_map(session, a.team.upper(), before=date.fromisoformat(a.before) if a.before else None, n_games=a.games)
        print(P.format_rotation(rm) if rm else f"no stints for {a.team}")
        return 0
    if a.cmd == "check":
        for gid in a.game_ids.split(","):
            st = pd.read_sql(text("select player_name, team, sum(minutes) m, count(*) n, max(to_buzzer) buzz from nba_stint where game_id = :g group by player_name, team").bindparams(g=gid.strip()), engine)
            box = P.box_for_game(session, gid.strip()); box["player_name"] = box.player_name.map(lambda n: n.lower())
            j = box.merge(st, on="player_name", how="left")
            j["raw_minutes"] = j.m
            print(f"{gid}: {len(st)} players with stints of {int((box.minutes > 0).sum())} who played; "
                  f"stints per player {st.n.mean():.1f}; players missing: {sorted(set(box[box.minutes > 0].player_name) - set(st.player_name))}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
