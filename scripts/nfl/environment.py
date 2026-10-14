#!/usr/bin/env python3
"""
The projected game environment (ivan.sports.nfl.refactor.environment): fit,
backtest and inspect layer 1 of the game model.

    scripts/nfl/environment.py backtest [--from 2024-1] [--out DIR]   rolling week-by-week refit, projected vs actual
    scripts/nfl/environment.py grid [--from 2024-1]                    team-effect shrinkage grid (chooses the defaults)
    scripts/nfl/environment.py slate                                   the loaded slate's environment (what ingest writes)
    scripts/nfl/environment.py draw TEAM OPP [-n 2000]                 joint draws for one slate game

The backtest writes a validation report (summary.json with reports.environment,
plus environment_backtest.csv) under data/nfl/output/validation/environment_<stamp>/
so the Validation page shows it like any other report.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ivan.core import validation as V
from ivan.core.config import load_config
from ivan.database.session import get_session, init_db
from ivan.sports.nfl.paths import OUTPUT_DIR
from ivan.sports.nfl.refactor import environment as E


def _week(s: str) -> tuple[int, int]:
    a, b = s.split("-")
    return int(a), int(b)


def cmd_backtest(session, a) -> int:
    df = E.team_games(session)
    bt = E.backtest(df, _week(a.test_from))
    if bt.empty:
        print("no test rows"); return 1
    summary = E.backtest_summary(bt)
    out = Path(a.out) if a.out else OUTPUT_DIR / "validation" / f"environment_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    bt.to_csv(out / "environment_backtest.csv", index=False)
    # the same shape the Validation page reads: reports.environment with per-quantity blocks and rows
    env = bt.rename(columns={q: f"actual_{q}" for q in E.QUANTITIES}).copy()
    env["slate"] = env.season.astype(str) + " wk " + env.week.astype(str)
    rep = V.environment_report(env)
    rep["backtest"] = summary
    rep.pop("frame", None)
    rep["rows"] = rep["rows"][-64:]                       # the page shows the latest team-games; the CSV has all
    checks = V.build_checks(environment=rep)
    s = {"title": a.title or f"game environment backtest from {a.test_from}", "created_at": datetime.now().isoformat(timespec="seconds"),
         "since": f"{bt.season.min()}-01-01", "until": f"{bt.season.max()}-12-31", "contests": 0, "coverage": [], "checks": checks,
         "check_counts": V.summarise_checks(checks), "reports": {"environment": rep}}
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(f"{len(bt)} team-games, seasons {summary['seasons']}")
    for q in E.QUANTITIES:
        o = summary["quantities"][q]["overall"]
        print(f"  {q:10s} bias {o['bias']:+.2f}  MAE {o['mae']:.3f}  lines-only {o['mae_lines_only']:.3f}  league {o['mae_league']:.3f}  "
              f"corr {o['corr']:.2f}  sd ratio {o['sd_ratio']:.2f}  10-90 cover {o['coverage_10_90']:.0%}")
    for c in checks:
        print(f"  [{c['status']:4s}] {c['component']:12s} {c['name']:12s} {c['detail']}")
    print(f"wrote {out}/")
    return 0


def cmd_grid(session, a) -> int:
    df = E.team_games(session)
    g = E.grid_backtest(df, _week(a.test_from))
    pd.set_option("display.width", 200)
    for q in E.QUANTITIES:
        sub = g[g.quantity == q].copy(); ref = sub[~sub.team_effects].iloc[0]
        sub["mae_vs_lines"] = sub.mae - ref.mae
        print(f"\n{q}: lines-only MAE {ref.mae:.4f} corr {ref['corr']:.3f}")
        print(sub[sub.team_effects].sort_values("mae")[["k", "window", "mae", "mae_vs_lines", "corr", "bias"]].round(4).to_string(index=False))
    print(f"\ncurrent defaults: {E.DEFAULT_TEAM_EFFECTS}")
    return 0


def cmd_slate(session, a) -> int:
    env = E.slate_environment(session)
    if env.empty:
        print("no slate games with lines (is a slate loaded, with game odds?)"); return 1
    pd.set_option("display.width", 200)
    print(env[["team", "opponent", "is_home", "total", "spread", "implied_points", "points_mean", "plays_mean", "pass_rate_mean", "n_train", "fit_before"]]
          .round(2).to_string(index=False))
    return 0


def cmd_draw(session, a) -> int:
    from ivan.sports.nfl.actuals import history_cutoff
    before = history_cutoff(session)
    fit = E.fit_environment(E.team_games(session, before=before), before=before)
    print(fit.describe())
    games = E.slate_games(session)
    g = games[((games.team == a.team) & (games.opponent == a.opp)) | ((games.team == a.opp) & (games.opponent == a.team))]
    if len(g) != 2:
        print(f"no lines for {a.team} vs {a.opp} on the loaded slate"); return 1
    d = E.draw(fit, g, a.n, np.random.default_rng(a.seed))
    for t, v in d.items():
        print(t, {k: (round(float(x.mean()), 1), round(float(x.std()), 1)) for k, x in v.items()})
    both = np.column_stack([d[a.team]["points"], d[a.team]["plays"], d[a.team]["pass_rate"], d[a.opp]["points"], d[a.opp]["plays"], d[a.opp]["pass_rate"]])
    print("draw correlation (pts, plays, pr) x (team, opp):"); print(np.round(np.corrcoef(both, rowvar=False), 2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-url", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest"); b.add_argument("--from", dest="test_from", default="2024-1"); b.add_argument("--out", default=None); b.add_argument("--title", default="")
    g = sub.add_parser("grid"); g.add_argument("--from", dest="test_from", default="2024-1")
    sub.add_parser("slate")
    d = sub.add_parser("draw"); d.add_argument("team"); d.add_argument("opp"); d.add_argument("-n", type=int, default=2000); d.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config("nfl")
    session = get_session(init_db("nfl", a.db_url or cfg.get("database_url")))
    return {"backtest": cmd_backtest, "grid": cmd_grid, "slate": cmd_slate, "draw": cmd_draw}[a.cmd](session, a)


if __name__ == "__main__":
    raise SystemExit(main())
