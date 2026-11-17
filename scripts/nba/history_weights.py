#!/usr/bin/env python3
"""
scripts/nba/history_weights.py

How a player's trailing history is weighted (ivan.sports.nba.backtest.history_weights): the live half-lives and discounts of
projection.history beside the alternatives, walk-forward on the box scores. Minutes: the trailing mean against the minutes
played. Fantasy points: the shrunk per-minute rates times the minutes really played. MAE / bias over every player-game,
over a player's first 15 games for a new team and over his first 15 games of a season.

    scripts/nba/history_weights.py [--start 2025-11-15] [--player "nikola vucevic"]
"""
from __future__ import annotations

import argparse
from datetime import date

from ivan.database.session import get_session, init_db
from ivan.sports.nba.backtest import history_weights as HW
from ivan.sports.nba.projection.history import build_history, load_box_scores
from ivan.sports.nba.projection.rates import fpts_per_minute, shrunk_rates
from ivan.sports.nba.names import normalize_name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2025-11-15"); ap.add_argument("--player", help="also print this player's state today under each setting")
    a = ap.parse_args()
    sess = get_session(init_db("nba"))
    try:
        box = load_box_scores(sess)
    finally:
        sess.close()
    t = HW.compare(box, a.start)
    print(f"{int(t.n.iloc[0])} player-games from {a.start}; {int(t.n_new_team.iloc[0])} in a first-15 stretch for a new team, {int(t.n_new_season.iloc[0])} in the first 15 of a season")
    print(f"{'':38s}{'minutes MAE / bias':^44s}  {'fpts at true minutes MAE / bias':^44s}")
    print(f"{'setting':38s}" + "".join(f"{h:>14s}" for h in ("all", "new team", "new season")) + "    " + "".join(f"{h:>14s}" for h in ("all", "new team", "new season")))
    for name, r in t.iterrows():
        cell = lambda col, k: f"{r[f'{col}_mae_{k}']:6.2f} /{r[f'{col}_bias_{k}']:+5.2f} "     # noqa: E731
        print(f"{name:38s}" + "".join(cell("min", k) for k in ("all", "new_team", "new_season")) + "    " + "".join(cell("fpts", k) for k in ("all", "new_team", "new_season")))
    if a.player:
        nm = normalize_name(a.player); today = date.today()
        for name, kw in HW.SETTINGS:
            h = build_history(box, **kw); st = h.before(today)
            if nm not in st.index:
                print(f"{nm}: no history"); break
            r = st.loc[nm]
            print(f"{name:38s}{nm}: minutes {r.min_ewm:5.1f} (x plays {r.play_ewm:.2f}), fpts per minute {fpts_per_minute(shrunk_rates(r, h.league_rates)):.3f}, minutes behind the rate {r.eff_min:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
