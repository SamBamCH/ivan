"""
Walk-forward backtest of the minutes-anchored projection on the 2025-26 box
scores in nba_history.sqlite.

For every team-game from --start on, the roster is projected using only games
before that date. Availability and starters are taken from the box score (the
live pipeline gets them from DK status and Rotowire), so the numbers measure
the minutes/rate model, not injury-news timing.

  --grid       search start_shift x alloc x recovery, print the table
  --calibrate  fit per-stat dispersion and minutes sd, write projection_calibration.json
  --compare    MAE vs the Stokastic projections kept in projection_snapshot
Default (no flag): one evaluation with the current parameters plus the
star-out split and the naive baselines.
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ivan.database.session import init_db, get_session
from ivan.sports.nba.paths import MODELS_DIR, DIAG_DIR
from ivan.sports.nba.projection.history import load_box_scores, build_history, STATS, HistoryState
from ivan.sports.nba.projection.engine import project_team, RosterPlayer, ProjectionParams
from ivan.sports.nba.projection.rates import DK_WEIGHTS

ROSTER_WINDOW_GAMES = 10   # a player seen in the team's last N games is "on the roster"
STAR_BASELINE = 28.0       # a missing player projected above this makes it a star-out game


def team_games(box: pd.DataFrame, start: date, end: date) -> list[tuple[str, str, date]]:
    tg = box[["team", "game_id", "game_date"]].drop_duplicates()
    tg = tg[(tg.game_date >= start) & (tg.game_date <= end)]
    return list(tg.sort_values(["game_date", "game_id"]).itertuples(index=False, name=None))


def build_cases(box: pd.DataFrame, start: date, end: date, dnp_available: bool = False) -> list[dict]:
    """One case per team-game: roster with availability/starter flags and actuals."""
    played = box[box.played]
    recent = {}  # team -> list of (game_date, game_id, set(players))
    for (team, gid, d), g in played.groupby(["team", "game_id", "game_date"], sort=False):
        recent.setdefault(team, []).append((d, gid, set(g.player_name)))
    for t in recent:
        recent[t].sort()
    cases = []
    for team, gid, d in team_games(box, start, end):
        today = box[(box.team == team) & (box.game_id == gid)]
        prior = [p for p in recent[team] if p[0] < d][-ROSTER_WINDOW_GAMES:]
        roster_names = set().union(*[p[2] for p in prior]) if prior else set()
        roster_names |= set(today.player_name)
        act = today.set_index("player_name")
        roster = []
        for n in sorted(roster_names):
            if n in act.index and act.loc[n, "minutes"] > 0:
                roster.append(RosterPlayer(n, True, bool(act.loc[n, "starter"] > 0)))
            elif dnp_available and n in act.index:
                # dressed but did not play: live we cannot tell him from the rotation
                roster.append(RosterPlayer(n, True, False))
            else:
                roster.append(RosterPlayer(n, False, None))
        cases.append(dict(team=team, game_id=gid, date=d, roster=roster,
                          actual=act[act.minutes > 0][["minutes", "dk_fpts"] + STATS]))
    return cases


def run(cases, hist: HistoryState, params: ProjectionParams, simulate=False) -> pd.DataFrame:
    rows = []
    cache = {}
    for c in cases:
        d = c["date"]
        if d not in cache:
            cache = {d: hist.before(d)}  # cases are date-sorted; keep one
        state = cache[d]
        team_fpts = hist.team_before(d).get(c["team"])
        proj = project_team(c["roster"], state, hist, team_fpts, params, simulate=simulate)
        star_out = any((not p.available) and (p.name in state.index) and state.loc[p.name, "min_ewm"] >= STAR_BASELINE
                       for p in c["roster"])
        j = proj.join(c["actual"], how="inner", rsuffix="_act")
        j["team"], j["game_id"], j["date"], j["star_out"] = c["team"], c["game_id"], d, star_out
        rows.append(j.reset_index())
    return pd.concat(rows, ignore_index=True)


def metrics(df: pd.DataFrame, label: str) -> dict:
    d = df[df.games >= 5]  # players with some history; rookies' first games are noise for every method
    em = d.minutes - d.minutes_act
    ef = d.fpts_mean - d.dk_fpts
    return dict(label=label, n=len(d),
                min_mae=em.abs().mean(), min_bias=em.mean(),
                fpts_mae=ef.abs().mean(), fpts_rmse=np.sqrt((ef ** 2).mean()), fpts_bias=ef.mean(),
                corr=np.corrcoef(d.fpts_mean, d.dk_fpts)[0, 1])


def fmt(ms: list[dict]) -> str:
    t = pd.DataFrame(ms).set_index("label")
    return t.to_string(float_format=lambda v: f"{v:7.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2025-11-15")
    ap.add_argument("--end", default="2026-01-15")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--dnp-available", action="store_true",
                    help="treat dressed-but-DNP players as available (live conditions; default is oracle availability)")
    ap.add_argument("--sgrid", action="store_true", help="surplus rule grid (needs --dnp-available to matter)")
    ap.add_argument("--kgrid", action="store_true", help="recovery fraction grid")
    ap.add_argument("--rgrid", action="store_true", help="rate halflife / prior grid")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    session = get_session(init_db("nba"))
    box = load_box_scores(session)
    hist = build_history(box)
    cases = build_cases(box, start, end, dnp_available=args.dnp_available)
    print(f"{len(cases)} team-games {start}..{end}; league fpts/min "
          f"{sum(DK_WEIGHTS[s]*hist.league_rates[s] for s in STATS):.3f}; "
          f"within-player minutes sd {hist.league_min_sd:.2f}")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    if args.grid:
        ms = []
        for ss, al, rc in itertools.product([0.0, 4.0, 6.0, 8.0], ["none", "prop", "headroom"], [0.0, 0.5, 1.0]):
            if al == "none" and rc > 0:
                continue
            p = ProjectionParams(start_shift=ss, alloc=al, recovery=rc)
            res = run(cases, hist, p)
            m = metrics(res, f"shift={ss:g} alloc={al} rec={rc:g}")
            so = metrics(res[res.star_out], "x"); m["starout_fpts_mae"] = so["fpts_mae"]; m["starout_min_mae"] = so["min_mae"]
            ms.append(m); print(fmt([m]).split("\n")[-1])
        out = fmt(ms); report.append("GRID\n" + out); print(out)

    if args.sgrid:
        ms = []
        for tau, upp in [(None, True), (4.0, True), (8.0, True), (12.0, True), (8.0, False)]:
            res = run(cases, hist, ProjectionParams(cut_tau=tau, use_play_prob=upp))
            m = metrics(res, f"cut_tau={'prop' if tau is None else tau:} play_prob={upp}")
            big = res[res.baseline >= 28]
            m["stars_min_bias"] = (big.minutes - big.minutes_act).mean(); m["stars_fpts_mae"] = (big.fpts_mean - big.dk_fpts).abs().mean()
            ms.append(m)
        out = fmt(ms); report.append("SURPLUS GRID\n" + out); print(out)

    if args.kgrid:
        ms = []
        for rc in [0.0, 0.25, 0.5, 0.75, 1.0]:
            res = run(cases, hist, ProjectionParams(recovery=rc))
            m = metrics(res, f"recovery={rc:g}")
            so = metrics(res[res.star_out], "x"); m["starout_fpts_mae"] = so["fpts_mae"]; m["starout_bias"] = so["fpts_bias"]
            ms.append(m)
        out = fmt(ms); report.append("RECOVERY GRID\n" + out); print(out)

    if args.rgrid:
        ms = []
        for rh, pm in [(5, 60), (10, 60), (20, 60), (20, 120), (40, 120), (40, 240)]:
            h2 = build_history(box, rate_halflife=rh)
            res = run(cases, h2, ProjectionParams(prior_minutes=pm))
            ms.append(metrics(res, f"rate_halflife={rh} prior_min={pm}"))
        out = fmt(ms); report.append("RATE GRID\n" + out); print(out)

    params = ProjectionParams()
    res = run(cases, hist, params)
    naive = run(cases, hist, ProjectionParams(start_shift=0.0, alloc="none", recovery=0.0))
    ms = [metrics(naive, "naive: ewm minutes x ewm rates"),
          metrics(res, "model: all games"),
          metrics(res[res.star_out], "model: star-out games"),
          metrics(naive[naive.star_out], "naive: star-out games"),
          metrics(res[~res.star_out], "model: other games")]
    # season-average fpts as the dumbest baseline
    played = box[box.played].sort_values("game_date")
    played["season_avg"] = played.groupby("player_name").dk_fpts.transform(lambda s: s.shift(1).expanding().mean())
    sa = res.merge(played[["game_id", "player_name", "season_avg"]], on=["game_id", "player_name"], how="left")
    sa = sa[sa.season_avg.notna()].copy(); sa["fpts_mean"] = sa.season_avg
    ms.append(metrics(sa, "baseline: season-average fpts"))
    out = fmt(ms); report.append("EVALUATION\n" + out); print(out)
    print(f"star-out team-games: {res.drop_duplicates(['team','game_id']).star_out.sum()} of "
          f"{res.drop_duplicates(['team','game_id']).shape[0]}")

    if args.compare:
        snap = pd.read_sql("select player_name, gametime, fpts_mean as stok from projection_snapshot "
                           "where fpts_mean is not null and fpts_mean > 0", session.get_bind())
        snap["date"] = pd.to_datetime(snap.gametime).dt.date
        snap = snap.drop_duplicates(["player_name", "date"])
        j = res.merge(snap[["player_name", "date", "stok"]], on=["player_name", "date"], how="inner")
        j = j[j.games >= 5]
        em, es = j.fpts_mean - j.dk_fpts, j.stok - j.dk_fpts
        lines = [f"matched player-games with a Stokastic projection: {len(j)}",
                 f"  model     MAE {em.abs().mean():.3f}  bias {em.mean():+.3f}  corr {np.corrcoef(j.fpts_mean, j.dk_fpts)[0,1]:.3f}",
                 f"  stokastic MAE {es.abs().mean():.3f}  bias {es.mean():+.3f}  corr {np.corrcoef(j.stok, j.dk_fpts)[0,1]:.3f}",
                 f"  blend 50/50 MAE {((j.fpts_mean+j.stok)/2 - j.dk_fpts).abs().mean():.3f}"]
        so = j[j.star_out]
        if len(so):
            lines.append(f"  star-out subset n={len(so)}: model MAE {(so.fpts_mean-so.dk_fpts).abs().mean():.3f}  "
                         f"stokastic MAE {(so.stok-so.dk_fpts).abs().mean():.3f}")
        # minutes-band view: where does each win
        j["band"] = pd.cut(j.minutes, [0, 15, 25, 32, 50])
        bt = j.groupby("band", observed=True).apply(lambda d: pd.Series({"n": len(d), "model": (d.fpts_mean-d.dk_fpts).abs().mean(),
                                                               "stok": (d.stok-d.dk_fpts).abs().mean()}), include_groups=False)
        lines.append(bt.to_string(float_format=lambda v: f"{v:.3f}"))
        out = "\n".join(lines); report.append("VS STOKASTIC\n" + out); print(out)

    if args.calibrate:
        d = res[res.games >= 5].copy()
        disp = {}
        for s in STATS:
            e = d[f"proj_{s}"]; a = d[s]
            excess = ((a - e) ** 2 - e).clip(lower=0)      # var - mean  ~ mean^2 / k
            k = float((e ** 2).sum() / max(excess.sum(), 1e-9))
            disp[s] = round(min(max(k, 1.0), 200.0), 2)
        em = d.minutes_act - d.minutes
        bands = pd.cut(d.minutes, [0, 10, 20, 30, 50])
        sd_by_band = em.groupby(bands, observed=True).std()
        mids = np.array([5, 15, 25, 36.0]); sds = sd_by_band.values
        slope, floor = np.polyfit(mids[: len(sds)], sds, 1)
        cal = {"dispersion": disp, "min_sd_floor": round(float(floor), 3), "min_sd_slope": round(float(slope), 4),
               "fit_window": [args.start, args.end], "n_player_games": int(len(d))}
        # coverage with simulated std on a sample of cases
        sub = run(cases[-150:], hist, ProjectionParams(dispersion=disp), simulate=True)
        sub = sub[sub.games >= 5]
        z = (sub.dk_fpts - sub.fpts_mean) / sub.fpts_std
        cal["coverage_1sd"] = round(float((z.abs() < 1).mean()), 3)
        cal["coverage_2sd"] = round(float((z.abs() < 2).mean()), 3)
        cal["mean_std"] = round(float(sub.fpts_std.mean()), 3)
        cal["actual_resid_sd"] = round(float((sub.dk_fpts - sub.fpts_mean).std()), 3)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        (MODELS_DIR / "projection_calibration.json").write_text(json.dumps(cal, indent=2))
        out = json.dumps(cal, indent=2); report.append("CALIBRATION\n" + out); print(out)

    tag = "_live" if args.dnp_available else ""
    (DIAG_DIR / f"backtest_projection{tag}_{date.today()}.txt").write_text("\n\n".join(report))


if __name__ == "__main__":
    main()
