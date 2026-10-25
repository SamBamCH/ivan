#!/usr/bin/env python3
"""
scripts/nba/backtest_skeleton.py

Walk-forward A/B of the rotation grid as the projection (projection/skeleton.py) against the previous live path (the
solve's minutes as the mean, rotation.propose's grid as the shape), on last season's box scores with oracle
availability and starters (walkforward.build_cases). No books in the walk-forward: the solve anchors on history.

    scripts/nba/backtest_skeleton.py --start 2025-12-01 --end 2025-12-21 [--sims 400]

Reports minutes / fantasy-points error and bias (players with 5+ games), the 10-90 coverage of minutes and points, and
what the skeleton drops: players it leaves off the floor who did play, and their minutes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from ivan.database.session import get_session, init_db
from ivan.sports.nba.projection.engine import ProjectionParams, project_team
from ivan.sports.nba.projection.groups import group_of, position_map, rotation_depth
from ivan.sports.nba.projection.history import build_history, load_box_scores
from ivan.sports.nba.projection.minutes import salary_prior_minutes
from ivan.sports.nba.projection.rotation import boundary_spreads, curves, propose
from ivan.sports.nba.pbp import sub_shares
from ivan.sports.nba.projection.walkforward import build_cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2025-12-01"); ap.add_argument("--end", default="2025-12-21")
    ap.add_argument("--sims", type=int, default=400); ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--dnp-available", action="store_true", help="dressed-but-DNP players count as available (live conditions: nobody knows who sits)")
    ap.add_argument("--no-depth", action="store_true", help="skip the rotation depth cut")
    ap.add_argument("--depth-extra", type=int, default=0, help="keep this many players beyond the team's measured depth")
    a = ap.parse_args()
    session = get_session(init_db("nba"))
    box = load_box_scores(session); hist = build_history(box)
    posmap = position_map(session)
    cases = build_cases(box, date.fromisoformat(a.start), date.fromisoformat(a.end), dnp_available=a.dnp_available)
    if a.max_cases:
        cases = cases[: a.max_cases]
    print(f"{len(cases)} team-games {a.start}..{a.end}")
    base = ProjectionParams.from_config(n_sims=a.sims, quantiles=11)
    rows = []; dropped = []
    rng = np.random.default_rng(0)
    cache = {}
    for k, c in enumerate(cases):
        d = c["date"]
        if d not in cache:
            cache = {d: (hist.before(d), hist.team_before(d))}
        state, tf = cache[d]
        roster = [replace(p, position=posmap.get(p.name)) for p in c["roster"]]
        names = [p.name for p in roster]; avail = [p.name for p in roster if p.available]
        curve = curves(session, players=names, before=d, n_games=10)
        if curve.empty:
            continue
        spreads = boundary_spreads(session, avail, before=d)
        shares = sub_shares(session, players=names, before=d)          # the coach's habits as known before the game
        starters = [p.name for p in roster if p.available and p.starter]
        old_grid = propose(curve, avail, starters, salary_prior={p.name: salary_prior_minutes(p.salary) for p in roster})
        variants = {
            "old: solve mean, proposed grid shape": replace(base, minutes_model="rotation", rotation={"grid": old_grid, "spreads": spreads, "curve": curve}),
            "new: skeleton grid is the projection": replace(base, minutes_model="rotation", rotation={"skeleton": True, "curve": curve, "spreads": spreads, "shares": shares,
                                                                                                      "depth": None if a.no_depth else rotation_depth(box, c["team"], before=d) + a.depth_extra,
                                                                                                      "vacated": [p.name for p in roster if not p.available and p.name in curve.index]}),
        }
        for label, params in variants.items():
            proj = project_team(roster, state, hist, tf.get(c["team"]), params, rng=rng, simulate=True)
            j = proj.join(c["actual"], how="outer", rsuffix="_act")
            j = j[j.index.isin(avail)]
            j["label"], j["team"], j["date"] = label, c["team"], d
            rows.append(j.reset_index())
            if label.startswith("new"):
                zero = params.rotation.get("zero_minutes") or []
                for n in zero:
                    act = float(c["actual"].loc[n, "minutes"]) if n in c["actual"].index else 0.0
                    dropped.append(dict(team=c["team"], date=d, player=n, actual=act, fpts=float(c["actual"].loc[n, "dk_fpts"]) if n in c["actual"].index else 0.0))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1} cases", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df["minutes"] = df["minutes"].fillna(0.0); df["fpts_mean"] = df["fpts_mean"].fillna(0.0)
    df["minutes_act"] = df["minutes_act"].fillna(0.0); df["dk_fpts"] = df["dk_fpts"].fillna(0.0)
    out = []
    for label, g in df.groupby("label"):
        h = g[g.games.fillna(0) >= 5]
        em = h.minutes - h.minutes_act; ef = h.fpts_mean - h.dk_fpts
        sim = h.dropna(subset=["minutes_p10", "minutes_p90"])
        q = sim.quantiles.map(lambda x: x if isinstance(x, list) and len(x) == 11 else None)
        ok = q.notna()
        qa = np.vstack(q[ok].values) if ok.any() else np.zeros((0, 11))
        out.append(dict(label=label, n=len(h), min_mae=em.abs().mean(), min_bias=em.mean(), fpts_mae=ef.abs().mean(), fpts_bias=ef.mean(),
                        min_cover=((sim.minutes_act >= sim.minutes_p10) & (sim.minutes_act <= sim.minutes_p90)).mean(),
                        min_band=(sim.minutes_p90 - sim.minutes_p10).mean(),
                        fpts_cover=((sim[ok].dk_fpts >= qa[:, 1]) & (sim[ok].dk_fpts <= qa[:, 9])).mean() if len(qa) else np.nan,
                        fpts_band=float((qa[:, 9] - qa[:, 1]).mean()) if len(qa) else np.nan))
        # by role of actual minutes
        for lo, hi, nm in ((28, 99, "28+ min"), (18, 28, "18-28"), (9, 18, "9-18"), (0, 9, "<9")):
            s = h[(h.minutes_act >= lo) & (h.minutes_act < hi)]
            out[-1][f"mae {nm}"] = (s.minutes - s.minutes_act).abs().mean()
    t = pd.DataFrame(out).set_index("label")
    print(t.T.to_string(float_format=lambda v: f"{v:8.3f}"))
    dd = pd.DataFrame(dropped)
    if len(dd):
        played = dd[dd.actual > 0]
        print(f"\nskeleton dropped {len(dd)} player-games ({len(dd) / len(cases):.1f} per team-game); {len(played)} of them played: "
              f"mean {played.actual.mean():.1f} min, {played.fpts.mean():.1f} fpts; {int((played.actual >= 10).sum())} played 10+ minutes")
        print(played.sort_values('actual', ascending=False).head(12).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
