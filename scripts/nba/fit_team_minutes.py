#!/usr/bin/env python3
"""
scripts/nba/fit_team_minutes.py

Fit the team minutes model (ivan.sports.nba.projection.team_minutes) on the box
scores and check it walk-forward against the current independent sampler.

    scripts/nba/fit_team_minutes.py                      fit through --fit-end, evaluate --eval-start..--eval-end, write the params
    scripts/nba/fit_team_minutes.py --no-write           report only
    scripts/nba/fit_team_minutes.py --fit-end 2026-06-30 --no-eval   fit on everything

The report: tau (share noise) and beta (minutes per point of final margin) per
role; for the evaluation window, per role, the minutes MAE of the allocation and
of the draw's mean, and the 10-90 band coverage and width of the team draw next
to the current sampler's normal band; starters' minutes by final-margin bucket.
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from ivan.database.session import get_session, init_db
from ivan.sports.nba.paths import DIAG_DIR
from ivan.sports.nba.projection import team_minutes as TM
from ivan.sports.nba.projection.fpts import minutes_sd
from ivan.sports.nba.projection.history import build_history, load_box_scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-end", default="2026-01-31")
    ap.add_argument("--eval-start", default="2026-02-01"); ap.add_argument("--eval-end", default="2026-03-15")
    ap.add_argument("--no-eval", action="store_true"); ap.add_argument("--no-write", action="store_true")
    ap.add_argument("-n", type=int, default=300, help="worlds per team-game in the evaluation")
    ap.add_argument("--stand-in", action="store_true", help="fit against the scaled-baseline stand-in instead of the real allocator")
    ap.add_argument("--lines", default=None, help="historical lines CSV (scripts/nba/fetch_historical_lines.py); the draw then knows each game's spread")
    a = ap.parse_args()
    session = get_session(init_db("nba"))
    box = load_box_scores(session); hist = build_history(box)
    fit_end = date.fromisoformat(a.fit_end)
    first = box.game_date.min()
    alloc = None if a.stand_in else TM.allocated_frame(box, hist, first, date.fromisoformat(a.eval_end) if not a.no_eval else fit_end)
    params = TM.fit_team_minutes(box[box.game_date <= fit_end], hist.rows[hist.rows.game_date <= fit_end],
                                 alloc=None if alloc is None else alloc[alloc.game_date <= fit_end])
    params["allocator"] = "stand-in" if a.stand_in else "model"
    lines = [f"fit on {params['fit_rows']} player-games through {fit_end} against the {params['allocator']} allocation: P(OT) {params['p_ot']:.3f}, final margin sd {params['margin_sd_unconditional']:.1f} "
             f"(unconditional; the sampler uses {params['margin_sd']} around the spread)"]
    for r in TM.ROLES:
        lines.append(f"  {r:9s} n={params['n'][r]:5d}  tau={params['tau'][r]:.3f}  beta={params['beta'][r]:+.4f} per point of |margin|  alpha={params['alpha'][r]:+.3f}")
    if not a.no_eval:
        spreads = TM.load_historical_lines(a.lines) if a.lines else None
        ev = TM.evaluate(box, hist.rows, params, date.fromisoformat(a.eval_start), date.fromisoformat(a.eval_end), n=a.n, independent_sd=minutes_sd,
                         alloc=alloc, spreads=spreads)
        pd.set_option("display.width", 220)
        lines.append(f"\nevaluation {a.eval_start}..{a.eval_end}, {len(ev)} player-games (oracle availability and starters, spread unknown):")
        lines.append(TM.coverage_table(ev).round(3).to_string())
        ev["bucket"] = pd.cut(ev.margin.abs(), [-1, 5, 12, 20, 60], labels=["0-5", "6-12", "13-20", "20+"])
        st = ev[ev.role.isin(["star", "starter"])]
        lines.append("\nstarters by final margin (the draw does not know the margin; its mean should sit between the buckets):")
        lines.append(st.groupby("bucket", observed=True).agg(n=("actual", "size"), alloc=("alloc", "mean"), draw_mean=("mean", "mean"), actual=("actual", "mean")).round(1).to_string())
        if spreads:
            known = ev[ev.spread.notna()]
            lines.append(f"\nwith the spread known ({known.game_id.nunique()} games priced of {ev.game_id.nunique()}):")
            tg = known.drop_duplicates(["game_id", "team"])
            resid = tg.margin + tg.spread            # final margin minus the expectation (-spread)
            lines.append(f"  final margin vs spread: mean {resid.mean():+.2f}, sd {resid.std():.2f} (the sampler assumes {params['margin_sd']})")
            st2 = known[known.role.isin(["star", "starter"])].copy()
            st2["fav"] = pd.cut(st2.spread.abs(), [-1, 3, 7, 12, 40], labels=["pick'em", "3-7", "7-12", "12+"])
            lines.append("  starters by |spread| (the draw knows the spread): draw mean vs actual")
            lines.append(st2.groupby("fav", observed=True).agg(n=("actual", "size"), alloc=("alloc", "mean"), draw_mean=("mean", "mean"), actual=("actual", "mean"),
                                                                 cover=("actual", lambda x: np.nan)).round(1).to_string())
            lines.append("  coverage by role with the spread known: " + ", ".join(f"{r} {((g.actual >= g.q10) & (g.actual <= g.q90)).mean():.0%}" for r, g in known.groupby("role")))
        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        ev.to_csv(DIAG_DIR / "team_minutes_eval.csv", index=False)
        lines.append(f"\nper player-game rows -> {DIAG_DIR / 'team_minutes_eval.csv'}")
    print("\n".join(lines))
    if not a.no_write:
        TM.save_params(params); print(f"params -> {TM.PARAMS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
