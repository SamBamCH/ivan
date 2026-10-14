#!/usr/bin/env python3
"""
Pull the actuals a backtest settles against: the schedule (scores and closing
lines), the player box scores and the DST game table, for every season
through the current one. The first part of the overnight job, without the
clean-up and the server sync, so the backtest plan page can run it on demand.

    scripts/nfl/fetch_actuals.py
"""
from __future__ import annotations

from ivan.database.session import get_session, init_db
from ivan.sports.nfl.dst_scoring import build_dst_game_table
from ivan.sports.nfl.refactor.historical.data_fetch import fetch_boxscores, fetch_schedules, seasons_to_fetch


def main() -> int:
    session = get_session(init_db("nfl"))
    n = fetch_schedules(session)
    print(f"[actuals] nfl_schedule: {n} games")
    n = fetch_boxscores(session)
    print(f"[actuals] nfl_stats: {n} player-weeks")
    n = build_dst_game_table(None, years=seasons_to_fetch(), session=session)
    print(f"[actuals] nfl_dst_game: {n} team-games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
