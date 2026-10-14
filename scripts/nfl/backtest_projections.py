#!/usr/bin/env python3
"""
Score stored projections against what the players actually did.

Every ingest snapshots each slate player's projected mean and sd into
projection_snapshot. This joins those rows to DraftKings points computed from
nfl_stats (offense, kickers) and nfl_dst_game (defenses) for the week the
snapshot was taken for, and reports bias, error, rank correlation and whether
the stated sd covers the outcomes, per position and by projection tercile.

    python scripts/nfl/backtest_projections.py [--since 2025-09-01] [--until DATE] [--positions K,DST]
                                                [--write-calibration]

Only FLEX/classic rows are scored (captain rows are the same number x1.5).
Weeks are resolved from nfl_schedule, so run the overnight fetch first.

Two spreads are scored. The *model sd* is the fitted distribution's own
spread (fpts_std_model; older snapshots only stored the calibrated number,
which then stands in for it). The *calibrated sd* is what ingest stored after
widening by the per-position factor in data/nfl/models/sd_calibration.json.
--write-calibration sets each listed position's factor to the sd of
z = (actual - projected) / model sd over the window, so the next ingest's
spread covers the outcomes the way the DST residual pool already does.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from ivan.database.session import init_db, get_session, read_sql, add_missing_columns
from ivan.sports.nfl.actuals import dk_kicker_points, dk_offense_points, week_lookup
from ivan.sports.nfl.refactor.spread import load_sd_calibration, write_sd_calibration, MIN_ROWS

OFFENSE = {"QB", "RB", "WR", "TE"}


def assign_week(snapshot_dates: pd.Series, weeks: pd.DataFrame) -> pd.DataFrame:
    """A snapshot is for the first schedule week whose last game is on or after the snapshot date."""
    out = []
    for d in snapshot_dates:
        cand = weeks[weeks["last"] >= d]
        row = cand.iloc[0] if len(cand) else None
        out.append((int(row.season), int(row.week)) if row is not None else (None, None))
    return pd.DataFrame(out, columns=["season", "week"], index=snapshot_dates.index)


def z_scores(m: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(z on the model sd, z on the calibrated sd) for scored rows."""
    err = m.actual - m.fpts_mean
    model_sd = m.fpts_std_model.fillna(m.fpts_std) if "fpts_std_model" in m else m.fpts_std
    return err / model_sd.replace(0, np.nan), err / m.fpts_std.replace(0, np.nan)


def summarize(m: pd.DataFrame, label: str) -> str:
    m = m.dropna(subset=["actual"]).copy()
    if len(m) < 5:
        return f"\n== {label}: {len(m)} scored rows (too few)"
    err = m.actual - m.fpts_mean
    z_model, z = z_scores(m)
    lines = [f"\n== {label}: {len(m)} scored projections over {m.snapshot_day.nunique()} slates",
             f"projected mean {m.fpts_mean.mean():.2f}   actual mean {m.actual.mean():.2f}   "
             f"bias (actual - projected) {err.mean():+.2f}   MAE {err.abs().mean():.2f}",
             f"projected sd {m.fpts_std.mean():.2f}   actual sd {m.actual.std():.2f}   "
             f"z sd {z.std():.2f} (model sd: {z_model.std():.2f})   inside p05-p95 {((z > -1.645) & (z < 1.645)).mean():.0%}",
             f"spearman(projection, actual) {m[['fpts_mean', 'actual']].corr(method='spearman').iloc[0, 1]:+.2f}   "
             f"spearman(salary, actual) {m[['salary', 'actual']].corr(method='spearman').iloc[0, 1]:+.2f}"]
    try:
        m["tercile"] = pd.qcut(m.fpts_mean, 3, labels=["low", "mid", "high"])
        t = m.groupby("tercile", observed=True).agg(n=("actual", "size"), projected=("fpts_mean", "mean"),
                                                    actual=("actual", "mean"), salary=("salary", "mean")).round(2)
        lines.append(t.to_string())
    except ValueError:
        pass
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2025-09-01")
    ap.add_argument("--until", default=str(date.today()))
    ap.add_argument("--positions", default="QB,RB,WR,TE,K,DST")
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--model-dir", default=None, help="where to read/write sd_calibration.json (default the live model directory)")
    ap.add_argument("--write-calibration", action="store_true",
                    help="write each listed position's z sd (on the model sd) to sd_calibration.json")
    args = ap.parse_args()
    positions = [p.strip().upper() for p in args.positions.split(",") if p.strip()]

    engine = init_db("nfl", args.db_url)
    from ivan.sports.nfl.models import NFLBase
    add_missing_columns(engine, NFLBase.metadata)
    session = get_session(engine)
    snap = pd.read_sql(
        text("""SELECT substr(snapshot_date, 1, 10) AS snapshot_day, player_name, team, upper(position) AS position,
                       salary, fpts_mean, fpts_std, fpts_std_model
                FROM projection_snapshot
                WHERE roster_slot_id <> 511 AND snapshot_date BETWEEN :a AND :b"""),
        session.get_bind(), params={"a": args.since, "b": args.until + " 23:59:59"})
    snap = snap[snap.position.isin(positions)].drop_duplicates(["snapshot_day", "player_name"])
    if snap.empty:
        raise SystemExit("no snapshots in that window")
    snap[["season", "week"]] = assign_week(snap.snapshot_day, week_lookup(session))
    snap = snap.dropna(subset=["season"])
    snap["season"] = snap.season.astype(int); snap["week"] = snap.week.astype(int)
    seasons = ",".join(str(s) for s in snap.season.unique())

    st = read_sql(session, text(f"SELECT * FROM nfl_stats WHERE season IN ({seasons})"))
    st["player_name"] = st.player_display_name.astype(str).str.lower().str.strip()
    st["actual"] = np.where(st.position == "K", dk_kicker_points(st), dk_offense_points(st))
    actual_players = st[["player_name", "season", "week", "actual"]].drop_duplicates(["player_name", "season", "week"])
    dst = read_sql(session, text(f"SELECT season, week, team, dst_points AS actual FROM nfl_dst_game WHERE season IN ({seasons})"))

    players = snap[snap.position != "DST"].merge(actual_players, on=["player_name", "season", "week"], how="left")
    defenses = snap[snap.position == "DST"].merge(dst, on=["team", "season", "week"], how="left")
    scored = pd.concat([players, defenses], ignore_index=True)

    print(f"snapshots {args.since}..{args.until}: {len(scored)} rows, {scored.actual.notna().sum()} with an actual "
          f"(a missing actual is a bye, an inactive, or a name nfl_stats spells differently)")
    for pos in positions:
        print(summarize(scored[scored.position == pos], pos))

    if args.write_calibration:
        factors, counts = {}, {}
        for pos in positions:
            m = scored[(scored.position == pos)].dropna(subset=["actual"])
            if len(m) < MIN_ROWS:
                print(f"\ncalibration: {pos} has {len(m)} scored rows (need {MIN_ROWS}); factor unchanged")
                continue
            z_model, _ = z_scores(m)
            factors[pos] = float(z_model.std())
            counts[pos] = int(len(m))
        cal = write_sd_calibration(factors, counts, (args.since, args.until), **({"model_dir": Path(args.model_dir)} if args.model_dir else {}))
        print("\ncalibration written: " + " ".join(f"{p} x{f:.2f} (n={counts[p]})" for p, f in sorted(factors.items()))
              + f"\n  file: data/nfl/models/sd_calibration.json; all factors now {cal['factors']}")
    else:
        cal = load_sd_calibration(Path(args.model_dir) if args.model_dir else None)
        if cal.get("factors"):
            print(f"\npredictive sd factors in effect for the next ingest: {cal['factors']} (written {cal.get('fitted_on')})")


if __name__ == "__main__":
    main()
