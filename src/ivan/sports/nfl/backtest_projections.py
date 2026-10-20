# src/ivan/sports/nfl/backtest_projections.py
"""
Score stored projections against what the players actually did.

Every ingest snapshots each slate player's projected mean and sd into
projection_snapshot. scored_projections() joins those rows to DraftKings points
computed from nfl_stats (offense, kickers) and nfl_dst_game (defenses) for the
week the snapshot was taken for; summarize() reports bias, error, rank
correlation and whether the stated sd covers the outcomes.

Two spreads are scored. The *model sd* is the fitted distribution's own spread
(fpts_std_model; older snapshots only stored the calibrated number, which then
stands in for it). The *calibrated sd* is what the ingest stored after widening
by the per-position factor in sd_calibration.json. sd_calibration_factors()
is that factor: the sd of z = (actual - projected) / model sd over the window,
per position (refactor.spread writes it; model_build.build_sd_calibration
drives it).

Only FLEX/classic rows are scored (captain rows are the same number x1.5).
Weeks are resolved from nfl_schedule, so run the overnight fetch first.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text

from ivan.database.session import read_sql
from ivan.sports.nfl.actuals import dk_kicker_points, dk_offense_points, week_lookup
from ivan.sports.nfl.refactor.spread import POWER_B_RANGE, MIN_ROWS

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


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


def scored_projections(session, since: str, until: str, positions: Iterable[str] = POSITIONS) -> pd.DataFrame:
    """
    Snapshot rows in [since, until] for the positions, each with the player's actual
    DraftKings points for the week it was taken for (NaN for a bye, an inactive, or a
    name nfl_stats spells differently). Raises ValueError when the window has no snapshots.
    """
    positions = [p.upper() for p in positions]
    snap = read_sql(session, text(
        """SELECT substr(snapshot_date, 1, 10) AS snapshot_day, player_name, team, upper(position) AS position,
                  salary, fpts_mean, fpts_std, fpts_std_model
           FROM projection_snapshot
           WHERE roster_slot_id <> 511 AND snapshot_date BETWEEN :a AND :b""").bindparams(a=since, b=until + " 23:59:59"))
    snap = snap[snap.position.isin(positions)].drop_duplicates(["snapshot_day", "player_name"])
    if snap.empty:
        raise ValueError(f"no projection snapshots between {since} and {until}")
    snap[["season", "week"]] = assign_week(snap.snapshot_day, week_lookup(session))
    snap = snap.dropna(subset=["season"]).copy()
    snap["season"] = snap.season.astype(int)
    snap["week"] = snap.week.astype(int)
    seasons = ",".join(str(s) for s in snap.season.unique())

    st = read_sql(session, text(f"SELECT * FROM nfl_stats WHERE season IN ({seasons})"))
    st["player_name"] = st.player_display_name.astype(str).str.lower().str.strip()
    st["actual"] = np.where(st.position == "K", dk_kicker_points(st), dk_offense_points(st))
    actual_players = st[["player_name", "season", "week", "actual"]].drop_duplicates(["player_name", "season", "week"])
    dst = read_sql(session, text(f"SELECT season, week, team, dst_points AS actual FROM nfl_dst_game WHERE season IN ({seasons})"))

    players = snap[snap.position != "DST"].merge(actual_players, on=["player_name", "season", "week"], how="left")
    defenses = snap[snap.position == "DST"].merge(dst, on=["team", "season", "week"], how="left")
    return pd.concat([players, defenses], ignore_index=True)


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


def sd_calibration_powers(scored: pd.DataFrame, positions: Iterable[str] = POSITIONS, min_rows: int = 300,
                          bins: int = 4) -> dict:
    """
    Per position with at least min_rows scored rows: realized sd = a x model sd^b,
    fitted log-log on the realized rmse within each model-sd quantile bin. The
    flat factor is the special case b = 1; b < 1 means the model's spread grows
    faster than reality's across the projection range.
    """
    out = {}
    for pos in positions:
        m = scored[scored.position == pos.upper()].dropna(subset=["actual"]).copy()
        model_sd = m.fpts_std_model.fillna(m.fpts_std) if "fpts_std_model" in m else m.fpts_std
        m = m.assign(model_sd=model_sd)
        m = m[m.model_sd > 0]
        if len(m) < min_rows:
            continue
        m["bin"] = pd.qcut(m.model_sd, bins, labels=False, duplicates="drop")
        g = m.groupby("bin")
        x = np.log(g.model_sd.mean().values)
        y = np.log(np.sqrt(g.apply(lambda h: ((h.actual - h.fpts_mean) ** 2).mean()).values))
        if len(x) < 2 or np.ptp(x) < 0.5:            # the bins must span a real range of model sd for a slope to mean anything
            continue
        b, _ = np.polyfit(x, y, 1)
        b = float(np.clip(b, *POWER_B_RANGE))
        loga = float(np.mean(y - b * x))              # a refitted at the (clipped) slope
        out[pos] = {"a": float(np.exp(loga)), "b": b, "n": int(len(m)), "bins": int(len(x))}
    return out


def sd_calibration_factors(scored: pd.DataFrame, positions: Iterable[str] = POSITIONS,
                           min_rows: int = MIN_ROWS) -> tuple[dict, dict, dict]:
    """
    Per position, the sd of z on the model sd over the scored rows: (factors, counts,
    skipped) where skipped maps a position with fewer than min_rows scored rows to its count.
    """
    factors, counts, skipped = {}, {}, {}
    for pos in positions:
        m = scored[scored.position == pos.upper()].dropna(subset=["actual"])
        if len(m) < min_rows:
            skipped[pos] = int(len(m))
            continue
        z_model, _ = z_scores(m)
        factors[pos] = float(z_model.std())
        counts[pos] = int(len(m))
    return factors, counts, skipped
