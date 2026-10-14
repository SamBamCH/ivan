# src/ivan/core/validation.py
"""
Projected against actual, one component of the pipeline at a time.

Every model in the pipeline makes a claim before lock: a player's score
distribution, a player's ownership, how often a lineup is duplicated, what
the field's cash line will be, which lineups have the best ROI. After the
games this module lines each claim up with what happened and reports the
difference, bucketed so the error shows where it lives, next to a dumb
baseline so "better than nothing" is visible.

The frames are sport-agnostic. A sport's adapter (ivan.sports.<sport>.
validate) builds them from its run artifacts and its actuals and calls:

    player_report(players)   fpts and ownership: bias, MAE, salary baseline,
                             PIT histogram and interval coverage, by
                             position / salary tier / ownership decile
    field_report(field)      per contest: the sim's cash line, winning
                             score, score percentiles and dupe profile
                             against the real field's
    lineup_report(lineups)   our entries: sim cash % / ROI / dupes against
                             the real finish
    pool_report(pool)        every lineup the sim scored: does sim ROI rank
                             lineups the way the real field did, and did
                             the chooser beat a random pick from the pool
    build_checks(reports)    pass / warn / fail rows for the UI

Column contracts are documented on each function. Missing optional columns
are tolerated; a metric that needs them comes back None.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

QGRID = (1, 5, 10, 25, 50, 75, 90, 95, 99)
Q_COLS = [f"q{q:02d}" for q in QGRID]
PIT_BINS = 10


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _corr(a, b) -> Optional[float]:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return None
    return _f(np.corrcoef(a[ok], b[ok])[0, 1])


def _spearman(a, b) -> Optional[float]:
    a = pd.Series(np.asarray(a, dtype=float)); b = pd.Series(np.asarray(b, dtype=float))
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return None
    return _corr(a[ok].rank().values, b[ok].rank().values)


# --------------------------------------------------------------------------- PIT
def pit_values(actual: np.ndarray, quantiles: np.ndarray, qgrid: Sequence[float] = QGRID) -> np.ndarray:
    """
    Where each actual falls in its projected distribution, as a probability
    in [0, 1], by linear interpolation through the stored quantile grid
    (percentiles `qgrid`) and linear extrapolation to 0 / 1 past its ends.
    quantiles[i] is the i-th row's quantile values at qgrid; NaN -> NaN.
    """
    actual = np.asarray(actual, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    p = np.asarray(qgrid, dtype=float) / 100.0
    out = np.full(actual.shape[0], np.nan)
    for i in range(actual.shape[0]):
        a, q = actual[i], quantiles[i]
        if not np.isfinite(a) or not np.isfinite(q).all():
            continue
        q = np.maximum.accumulate(q)                         # a non-decreasing grid, whatever the fit did
        if a <= q[0]:
            span = q[1] - q[0]
            out[i] = max(0.0, p[0] - (q[0] - a) / span * (p[1] - p[0])) if span > 0 else 0.0
        elif a >= q[-1]:
            span = q[-1] - q[-2]
            out[i] = min(1.0, p[-1] + (a - q[-1]) / span * (p[-1] - p[-2])) if span > 0 else 1.0
        else:
            out[i] = float(np.interp(a, q, p))
    return out


def pit_normal(actual: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """PIT under a normal approximation, for projections that stored only mean and sd."""
    from math import erf, sqrt
    actual = np.asarray(actual, dtype=float); mean = np.asarray(mean, dtype=float); sd = np.asarray(sd, dtype=float)
    out = np.full(actual.shape[0], np.nan)
    ok = np.isfinite(actual) & np.isfinite(mean) & np.isfinite(sd) & (sd > 0)
    z = (actual[ok] - mean[ok]) / sd[ok]
    out[ok] = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in z])
    return out


def pit_histogram(pit: np.ndarray, bins: int = PIT_BINS) -> List[dict]:
    """Share of actuals per PIT decile; a calibrated model gives 1/bins each. Includes the ideal share."""
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]
    if pit.size == 0:
        return []
    counts, edges = np.histogram(np.clip(pit, 0, 1 - 1e-9), bins=bins, range=(0, 1))
    return [{"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "lo": float(edges[i]), "hi": float(edges[i + 1]),
             "count": int(c), "share": float(c / pit.size), "ideal": 1.0 / bins} for i, c in enumerate(counts)]


def pit_shape(pit: np.ndarray) -> Optional[str]:
    """A one-word reading of the PIT histogram: calibrated, too_narrow (U), too_wide (hump), biased_low, biased_high."""
    pit = np.asarray(pit, dtype=float); pit = pit[np.isfinite(pit)]
    if pit.size < 30:
        return None
    tails = float(((pit < 0.1) | (pit > 0.9)).mean())        # ideal 0.2
    mean = float(pit.mean())                                   # ideal 0.5
    if mean < 0.42:
        return "biased_high"                                   # actuals land low in the distribution: projections too high
    if mean > 0.58:
        return "biased_low"
    if tails > 0.28:
        return "too_narrow"
    if tails < 0.12:
        return "too_wide"
    return "calibrated"


# --------------------------------------------------------------------------- players
def _salary_baseline(df: pd.DataFrame, by: Sequence[str] = ("slot",), holdout: str = "slate") -> np.ndarray:
    """
    A least-squares line actual ~ a + b * salary per group: the projection you get for free from the price.
    Out of sample when the frame spans several `holdout` groups (slates): each slate's line is fitted on the
    other slates only, so the baseline never sees the scores it is judged on. A single slate falls back to the
    in-sample line (flagged by _salary_baseline_note).
    """
    pred = np.full(len(df), np.nan)
    keys = [k for k in by if k in df.columns]
    groups = df.groupby(keys, dropna=False).indices if keys else {"all": np.arange(len(df))}
    hold = df[holdout].values if holdout in df.columns else None
    for _, idx in groups.items():
        idx = np.asarray(idx)
        x = df.salary.values[idx].astype(float); y = df.actual_fpts.values[idx].astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        folds = [None]
        if hold is not None and len(set(hold[idx])) >= 2:
            folds = sorted(set(hold[idx]), key=str)
        for f in folds:
            test = np.ones(len(idx), dtype=bool) if f is None else (hold[idx] == f)
            train = ok if f is None else (ok & ~test)
            if train.sum() >= 5 and x[train].std() > 0:
                b, a = np.polyfit(x[train], y[train], 1)
                pred[idx[test]] = a + b * x[test]
            elif ok.sum() >= 5 and x[ok].std() > 0:                     # too few other slates in this slot: in-sample line
                b, a = np.polyfit(x[ok], y[ok], 1)
                pred[idx[test]] = a + b * x[test]
            elif ok.sum():
                pred[idx[test]] = y[ok].mean()
    return pred


def _salary_baseline_note(df: pd.DataFrame, holdout: str = "slate") -> str:
    n = df[holdout].nunique() if holdout in df.columns else 1
    return f"leave-one-slate-out over {n} slates" if n >= 2 else "in-sample (one slate)"


def _error_block(df: pd.DataFrame, pit_col: str = "pit") -> dict:
    """bias / mae / rmse / corr / coverage / sd ratio of actual against projected for one group."""
    a = df.actual_fpts.values.astype(float); p = df.proj_mean.values.astype(float)
    ok = np.isfinite(a) & np.isfinite(p)
    out = {"n": int(ok.sum())}
    if not ok.any():
        return out
    err = a[ok] - p[ok]
    out.update(bias=_f(err.mean()), mae=_f(np.abs(err).mean()), rmse=_f(np.sqrt((err ** 2).mean())), corr=_corr(a[ok], p[ok]),
               actual_mean=_f(a[ok].mean()), proj_mean=_f(p[ok].mean()))
    if "baseline" in df.columns:
        b = df.baseline.values.astype(float)
        okb = ok & np.isfinite(b)
        out["baseline_mae"] = _f(np.abs(a[okb] - b[okb]).mean()) if okb.any() else None
    if "proj_sd" in df.columns:
        sd = df.proj_sd.values.astype(float)
        oks = ok & np.isfinite(sd) & (sd > 0)
        if oks.sum() >= 5:
            out["sd_ratio"] = _f((a[oks] - p[oks]).std() / sd[oks].mean())     # > 1: the model's sd is too small
            out["z_abs_mean"] = _f(np.abs((a[oks] - p[oks]) / sd[oks]).mean())  # 0.80 for a normal
    if pit_col in df.columns:
        pit = df[pit_col].values.astype(float); okp = np.isfinite(pit)
        if okp.any():
            out.update(coverage_10_90=_f(((pit[okp] >= 0.10) & (pit[okp] <= 0.90)).mean()),
                       coverage_05_95=_f(((pit[okp] >= 0.05) & (pit[okp] <= 0.95)).mean()),
                       pit_mean=_f(pit[okp].mean()), pit_shape=pit_shape(pit[okp]), n_pit=int(okp.sum()))
    return out


def _ownership_block(df: pd.DataFrame, proj_col: str = "own_proj") -> dict:
    if proj_col not in df.columns or "actual_own" not in df.columns:
        return {"n": 0}
    a = df.actual_own.values.astype(float); p = df[proj_col].values.astype(float)
    ok = np.isfinite(a) & np.isfinite(p)
    out = {"n": int(ok.sum())}
    if ok.sum() < 3:
        return out
    err = p[ok] - a[ok]
    out.update(bias=_f(err.mean()), mae=_f(np.abs(err).mean()), corr=_corr(a[ok], p[ok]), spearman=_spearman(a[ok], p[ok]),
               proj_sum=_f(p[ok].sum()), actual_sum=_f(a[ok].sum()),
               max_abs_err=_f(np.abs(err).max()))
    # the chalk: the top 5 by actual ownership, how far off were we on average
    top = np.argsort(-a[ok])[:5]
    out["chalk_bias"] = _f(err[top].mean())
    return out


def player_report(players: pd.DataFrame, pit_source: str = "auto") -> dict:
    """
    players: one row per player-slot per slate with
      required  slot, position, salary, proj_mean, actual_fpts
      optional  proj_sd, q01..q99 (projected quantiles), own_proj, own_field (the sim's field exposure %),
                actual_own, name, team, slate, contest_id, source
    pit_source: "quantiles" (the stored grid), "normal" (mean/sd), "auto" (grid when present)
    """
    df = players.copy()
    if df.empty:
        return {"n": 0}
    df["actual_fpts"] = pd.to_numeric(df.actual_fpts, errors="coerce")
    df["proj_mean"] = pd.to_numeric(df.proj_mean, errors="coerce")
    has_q = all(c in df.columns for c in Q_COLS) and df[Q_COLS].notna().all(axis=1).any()
    method = pit_source if pit_source != "auto" else ("quantiles" if has_q else "normal")
    if method == "quantiles" and has_q:
        df["pit"] = pit_values(df.actual_fpts.values, df[Q_COLS].values, QGRID)
    elif "proj_sd" in df.columns:
        df["pit"] = pit_normal(df.actual_fpts.values, df.proj_mean.values, pd.to_numeric(df.proj_sd, errors="coerce").values)
    else:
        df["pit"] = np.nan
    df["baseline"] = _salary_baseline(df)
    df["err"] = df.actual_fpts - df.proj_mean
    df["scored"] = np.where(df.actual_fpts > 0, "scored > 0", np.where(df.actual_fpts.notna(), "zero", "no actual"))
    if "salary" in df.columns and df.salary.notna().any():
        try:
            df["salary_tier"] = pd.qcut(df.salary.rank(method="first"), 4, labels=["Q1 cheap", "Q2", "Q3", "Q4 expensive"]).astype(str)
        except ValueError:
            df["salary_tier"] = "all"
    own_col = "actual_own" if "actual_own" in df.columns and df.actual_own.notna().any() else ("own_proj" if "own_proj" in df.columns else None)
    if own_col:
        try:
            df["own_decile"] = pd.qcut(pd.to_numeric(df[own_col], errors="coerce").rank(method="first"), 5,
                                       labels=["D1 lowest", "D2", "D3", "D4", "D5 chalk"]).astype(str)
        except ValueError:
            df["own_decile"] = "all"

    def grouped(key: str) -> List[dict]:
        if key not in df.columns:
            return []
        rows = []
        for k, g in df.groupby(key, dropna=False, sort=True):
            rows.append({key: str(k), **_error_block(g)})
        return rows

    out = {"n": int(len(df)), "pit_method": method, "baseline": _salary_baseline_note(df), "overall": _error_block(df),
           "by_volume_source": grouped("volume_source") if "volume_source" in df.columns and df.volume_source.notna().any() else [],
           "by_scored": [r for r in grouped("scored") if r["scored"] != "no actual"],
           "by_position": grouped("position"), "by_slot": grouped("slot"), "by_salary_tier": grouped("salary_tier"),
           "by_own_decile": grouped("own_decile"), "by_slate": grouped("slate") if "slate" in df.columns else [],
           "pit_hist": pit_histogram(df.pit.values),
           "ownership": _ownership_block(df, "own_proj"),
           "ownership_field": _ownership_block(df, "own_field") if "own_field" in df.columns else {"n": 0}}
    # the ownership error by chalk decile (where the field model misses)
    if own_col and "own_proj" in df.columns and "actual_own" in df.columns:
        rows = []
        for k, g in df.groupby("own_decile", dropna=False, sort=True):
            rows.append({"own_decile": str(k), **_ownership_block(g, "own_proj")})
        out["ownership_by_decile"] = rows
    # the worst misses, for the workspace
    show = [c for c in ("slate", "contest_id", "slot", "name", "team", "position", "salary", "proj_mean", "proj_sd", "actual_fpts", "err", "pit",
                        "own_proj", "own_field", "actual_own", "source") if c in df.columns]
    out["worst"] = df.reindex(df.err.abs().sort_values(ascending=False).index)[show].head(25).to_dict("records")
    out["frame"] = df
    return out


# --------------------------------------------------------------------------- field
def field_report(field: pd.DataFrame) -> dict:
    """
    field: one row per contest with
      sim_cash_p10/p50/p90, actual_cash_line, sim_win_p10/p50/p90, actual_win,
      sim_score_p50/p90/p99, actual_score_p50/p90/p99 (entry score percentiles),
      sim_unique_share, actual_unique_share, sim_dupe_max, actual_dupe_max, sim_dupe_p90, actual_dupe_p90,
      sim_dupe_max_scaled, sim_dupe_p90_scaled, sim_top_share, actual_top_share (dupes scaled to the contest size),
      sim_size, actual_size, paid_places, contest_id, contest_name, slate
    """
    df = field.copy()
    if df.empty:
        return {"n": 0, "rows": []}
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    df["cash_in_band"] = (num("actual_cash_line") >= num("sim_cash_p10")) & (num("actual_cash_line") <= num("sim_cash_p90"))
    df["cash_err"] = num("actual_cash_line") - num("sim_cash_p50")
    df["win_in_band"] = (num("actual_win") >= num("sim_win_p10")) & (num("actual_win") <= num("sim_win_p90"))
    df["win_err"] = num("actual_win") - num("sim_win_p50")
    df["score_p50_err"] = num("actual_score_p50") - num("sim_score_p50")
    df["score_p90_err"] = num("actual_score_p90") - num("sim_score_p90")
    df["unique_share_err"] = num("sim_unique_share") - num("actual_unique_share")       # not scale-free: shown, not checked
    df["top_share_ratio"] = num("sim_top_share") / num("actual_top_share").replace(0, np.nan)   # most-duplicated lineup's share of the field
    df["dupe_max_ratio"] = num("sim_dupe_max_scaled") / num("actual_dupe_max").replace(0, np.nan)
    df["dupe_p90_ratio"] = num("sim_dupe_p90_scaled") / num("actual_dupe_p90").replace(0, np.nan)
    ok_cash = num("actual_cash_line").notna() & num("sim_cash_p50").notna()
    out = {"n": int(len(df)),
           "cash_line": {"n": int(ok_cash.sum()), "in_band_share": _f(df.cash_in_band[ok_cash].mean()) if ok_cash.any() else None,
                         "bias": _f(df.cash_err[ok_cash].mean()) if ok_cash.any() else None,
                         "mae": _f(df.cash_err[ok_cash].abs().mean()) if ok_cash.any() else None},
           "winning_score": {"n": int(df.win_err.notna().sum()), "in_band_share": _f(df.win_in_band[df.win_err.notna()].mean()) if df.win_err.notna().any() else None,
                             "bias": _f(df.win_err.mean())},
           "entry_score": {"p50_bias": _f(df.score_p50_err.mean()), "p90_bias": _f(df.score_p90_err.mean())},
           "dupes": {"unique_share_bias": _f(df.unique_share_err.mean()), "dupe_max_ratio_median": _f(df.dupe_max_ratio.median()),
                     "dupe_p90_ratio_median": _f(df.dupe_p90_ratio.median()), "top_share_ratio_median": _f(df.top_share_ratio.median()),
                     "n": int(df.dupe_max_ratio.notna().sum())},
           "rows": df.to_dict("records"), "frame": df}
    return out


# --------------------------------------------------------------------------- our lineups
def lineup_report(lineups: pd.DataFrame) -> dict:
    """
    lineups: one row per entry we chose with
      contest_id, entry_fee, sim_cash_pct, sim_avg_roi, pred_dupes, actual_points, actual_rank, actual_payout,
      actual_dupes, field_size, slate, name
    """
    df = lineups.copy()
    if df.empty:
        return {"n": 0}
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    fee = num("entry_fee"); pay = num("actual_payout")
    settled = pay.notna()
    out = {"n": int(len(df)), "settled": int(settled.sum())}
    if settled.any():
        sim_exp = (fee * (1 + num("sim_avg_roi")))[settled]
        out.update(fees=_f(fee[settled].sum()), actual_payout=_f(pay[settled].sum()),
                   actual_profit=_f((pay - fee)[settled].sum()), sim_expected_profit=_f((sim_exp - fee[settled]).sum()),
                   actual_roi=_f((pay - fee)[settled].sum() / fee[settled].sum()) if fee[settled].sum() else None,
                   sim_roi=_f((sim_exp - fee[settled]).sum() / fee[settled].sum()) if fee[settled].sum() else None,
                   sim_cash_pct=_f(num("sim_cash_pct")[settled].mean()), actual_cash_rate=_f((pay >= fee)[settled].mean()))
    d_ok = num("pred_dupes").notna() & num("actual_dupes").notna()
    if d_ok.any():
        pdp, ad = num("pred_dupes")[d_ok], num("actual_dupes")[d_ok]
        out["dupes"] = {"n": int(d_ok.sum()), "pred_mean": _f(pdp.mean()), "actual_mean": _f(ad.mean()),
                        "ratio": _f(pdp.mean() / ad.mean()) if ad.mean() > 0 else None,
                        "bias": _f((pdp - ad).mean()), "spearman": _spearman(pdp, ad),
                        "share_unique_actual": _f((ad <= 1).mean()), "share_unique_pred": _f((pdp <= 1.5).mean())}
    by = []
    for cid, g in df.groupby("contest_id"):
        f = pd.to_numeric(g.entry_fee, errors="coerce"); p = pd.to_numeric(g.actual_payout, errors="coerce")
        s = p.notna()
        by.append({"contest_id": int(cid), "contest_name": str(g.contest_name.iloc[0]) if "contest_name" in g.columns else "", "slate": str(g.slate.iloc[0]) if "slate" in g.columns else "",
                   "entries": int(len(g)), "fees": _f(f.sum()),
                   "sim_expected_profit": _f((f * pd.to_numeric(g.sim_avg_roi, errors="coerce")).sum()),
                   "actual_profit": _f((p - f)[s].sum()) if s.any() else None,
                   "sim_cash_pct": _f(pd.to_numeric(g.sim_cash_pct, errors="coerce").mean()),
                   "actual_cash_rate": _f((p >= f)[s].mean()) if s.any() else None,
                   "pred_dupes": _f(pd.to_numeric(g.pred_dupes, errors="coerce").mean()) if "pred_dupes" in g.columns else None,
                   "actual_dupes": _f(pd.to_numeric(g.actual_dupes, errors="coerce").mean()) if "actual_dupes" in g.columns else None,
                   "best_rank": int(pd.to_numeric(g.actual_rank, errors="coerce").min()) if pd.to_numeric(g.actual_rank, errors="coerce").notna().any() else None})
    out["by_contest"] = by
    out["frame"] = df
    return out


# --------------------------------------------------------------------------- the pool
def pool_report(pool: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """
    pool: one row per unique lineup the sim scored, per contest, with
      contest_id, entry_fee, sim_avg_roi, sim_cash_pct, sim_dupes, pred_dupes, actual_points, actual_rank,
      actual_payout, actual_dupes, picked (bool), n_picked (entries of this lineup we played), slate, name
    Per contest: Spearman(sim ROI, actual payout); actual ROI of the picks against the pool mean, the
    top-K by sim ROI and K random lineups (bootstrap p-value); sim-ROI deciles against actual ROI.
    """
    df = pool.copy()
    if df.empty:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    rows, deciles = [], []
    for cid, g in df.groupby("contest_id"):
        fee = pd.to_numeric(g.entry_fee, errors="coerce").iloc[0]
        pay = pd.to_numeric(g.actual_payout, errors="coerce"); roi_sim = pd.to_numeric(g.sim_avg_roi, errors="coerce")
        ok = pay.notna() & roi_sim.notna()
        if ok.sum() < 5 or not fee:
            continue
        gg = g[ok].copy(); gg["actual_roi"] = (pay[ok] - fee) / fee
        picked = gg[gg.picked.astype(bool)] if "picked" in gg.columns else gg.iloc[0:0]
        k = int(picked.n_picked.sum()) if "n_picked" in picked.columns and len(picked) else len(picked)
        pool_roi = float(gg.actual_roi.mean())
        row = {"contest_id": int(cid), "contest_name": str(gg.contest_name.iloc[0]) if "contest_name" in gg.columns else "", "slate": str(gg.slate.iloc[0]) if "slate" in gg.columns else "",
               "pool_lineups": int(len(gg)), "entry_fee": _f(fee), "spearman_sim_roi_vs_actual": _spearman(gg.sim_avg_roi, gg.actual_roi),
               "spearman_sim_cash_vs_actual": _spearman(gg.sim_cash_pct, gg.actual_roi) if "sim_cash_pct" in gg.columns else None,
               "pool_actual_roi": _f(pool_roi), "pool_sim_roi": _f(gg.sim_avg_roi.mean()),
               "picked_lineups": int(len(picked)), "picked_entries": k}
        if len(picked):
            w = picked.n_picked.values.astype(float) if "n_picked" in picked.columns else np.ones(len(picked))
            picked_roi = float(np.average(picked.actual_roi.values, weights=w))
            row.update(picked_actual_roi=_f(picked_roi), picked_sim_roi=_f(np.average(picked.sim_avg_roi.values, weights=w)))
            kk = max(1, min(int(round(w.sum())), len(gg)))
            top = gg.sort_values("sim_avg_roi", ascending=False).head(kk)
            row["topk_by_sim_roi_actual_roi"] = _f(top.actual_roi.mean())
            if "sim_cash_pct" in gg.columns:
                row["topk_by_sim_cash_actual_roi"] = _f(gg.sort_values("sim_cash_pct", ascending=False).head(kk).actual_roi.mean())
            draws = np.array([gg.actual_roi.values[rng.integers(0, len(gg), kk)].mean() for _ in range(n_boot)])
            row.update(random_k_actual_roi=_f(draws.mean()), random_k_p10=_f(np.percentile(draws, 10)), random_k_p90=_f(np.percentile(draws, 90)),
                       p_random_beats_picked=_f((draws >= picked_roi).mean()))
        rows.append(row)
        try:
            gg["sim_decile"] = pd.qcut(gg.sim_avg_roi.rank(method="first"), 10, labels=False) + 1
            for d, h in gg.groupby("sim_decile"):
                deciles.append({"contest_id": int(cid), "sim_decile": int(d), "n": int(len(h)), "sim_roi": _f(h.sim_avg_roi.mean()),
                                "actual_roi": _f(h.actual_roi.mean()), "actual_cash_rate": _f((h.actual_roi >= 0).mean())})
        except ValueError:
            pass
    dec = pd.DataFrame(deciles)
    dec_all = []
    if not dec.empty:
        for d, h in dec.groupby("sim_decile"):
            dec_all.append({"sim_decile": int(d), "contests": int(len(h)), "sim_roi": _f(h.sim_roi.mean()), "actual_roi": _f(h.actual_roi.mean()),
                            "actual_cash_rate": _f(h.actual_cash_rate.mean())})
    r = pd.DataFrame(rows)
    out = {"n": int(len(df)), "contests": int(len(r)), "rows": rows, "deciles": dec_all, "deciles_by_contest": deciles}
    if len(r):
        out["spearman_median"] = _f(r.spearman_sim_roi_vs_actual.median())
        out["spearman_positive_share"] = _f((r.spearman_sim_roi_vs_actual > 0).mean())
        if "picked_actual_roi" in r.columns and r.picked_actual_roi.notna().any():
            w = r.picked_entries.fillna(0).values.astype(float)
            w = w if w.sum() else np.ones(len(r))
            out["picked_actual_roi"] = _f(np.average(r.picked_actual_roi.fillna(0).values, weights=w))
            out["random_k_actual_roi"] = _f(np.average(r.random_k_actual_roi.fillna(0).values, weights=w))
            out["topk_by_sim_roi_actual_roi"] = _f(np.average(r.topk_by_sim_roi_actual_roi.fillna(0).values, weights=w))
            out["pool_actual_roi"] = _f(np.average(r.pool_actual_roi.values, weights=w))
            out["picked_sim_roi"] = _f(np.average(r.picked_sim_roi.fillna(0).values, weights=w))
            out["contests_picked_beats_random"] = int((r.picked_actual_roi > r.random_k_actual_roi).sum())
    out["frame"] = df
    return out


# --------------------------------------------------------------------------- game environment
ENV_QUANTITIES = ("points", "plays", "pass_rate")


def environment_report(env: pd.DataFrame) -> dict:
    """
    env: one row per team-game with, for each of points / plays / pass_rate: <q>_mean, <q>_sd, <q>_league (optional),
    actual_<q>; plus slate, team, opponent, spread, total. Projected vs actual per quantity, with the league mean as
    the baseline and a normal PIT for coverage.
    """
    df = env.copy()
    if df.empty:
        return {"n": 0}
    out = {"n": int(len(df)), "quantities": {}}
    for q in ENV_QUANTITIES:
        a = pd.to_numeric(df.get(f"actual_{q}"), errors="coerce"); m = pd.to_numeric(df.get(f"{q}_mean"), errors="coerce")
        sd = pd.to_numeric(df.get(f"{q}_sd"), errors="coerce"); lg = pd.to_numeric(df.get(f"{q}_league"), errors="coerce")
        ok = a.notna() & m.notna()
        if ok.sum() < 2:
            out["quantities"][q] = {"n": int(ok.sum())}
            continue
        pit = pit_normal(a[ok].values, m[ok].values, sd[ok].values) if sd.notna().any() else np.full(int(ok.sum()), np.nan)
        df.loc[ok, f"{q}_pit"] = pit
        okp = np.isfinite(pit)
        out["quantities"][q] = {"n": int(ok.sum()), "actual_mean": _f(a[ok].mean()), "proj_mean": _f(m[ok].mean()), "bias": _f((a[ok] - m[ok]).mean()),
                                "mae": _f((a[ok] - m[ok]).abs().mean()), "mae_league": _f((a[ok] - lg[ok]).abs().mean()) if lg.notna().any() else None,
                                "corr": _corr(a[ok], m[ok]), "sd_ratio": _f((a[ok] - m[ok]).std() / sd[ok].mean()) if sd.notna().any() and sd[ok].mean() > 0 else None,
                                "coverage_10_90": _f(((pit[okp] >= 0.1) & (pit[okp] <= 0.9)).mean()) if okp.any() else None}
    out["frame"] = df
    show = [c for c in ("slate", "team", "opponent", "spread", "total", "points_mean", "actual_points", "plays_mean", "actual_plays",
                        "pass_rate_mean", "actual_pass_rate") if c in df.columns]
    out["rows"] = df[show].to_dict("records")
    return out


# --------------------------------------------------------------------------- usage (layer 2)
USAGE_STATS = ("receiving_yards", "receptions", "rushing_yards")


def usage_report(usage: pd.DataFrame) -> dict:
    """
    usage: one row per slate, team and stat with total, total_source, fixed, residual, n_unpriced, floored, coverage
    (= fixed / total: how much of the team's volume the settled lines already claim; 1.0 means the priced players
    add up to the whole team, above it their means are collectively too high). Needs no actuals: it is an identity
    check on the marginals, per stat.
    """
    df = usage.copy()
    if df.empty:
        return {"n": 0}
    df["coverage"] = pd.to_numeric(df.get("coverage"), errors="coerce")
    out = {"n": int(len(df)), "n_slates": int(df.slate.nunique()) if "slate" in df.columns else None, "stats": {}}
    for stat in USAGE_STATS:
        g = df[df.stat == stat]
        c = g.coverage.dropna()
        if c.empty:
            continue
        src = g.total_source.value_counts().to_dict() if "total_source" in g.columns else {}
        out["stats"][stat] = {"n": int(len(c)), "coverage_mean": _f(c.mean()), "coverage_median": _f(c.median()), "coverage_min": _f(c.min()),
                              "coverage_max": _f(c.max()), "over_share": _f((c > 1.0).mean()), "floored_share": _f(g.floored.astype(bool).mean()) if "floored" in g.columns else None,
                              "total_mean": _f(g.total.mean()), "fixed_mean": _f(g.fixed.mean()), "residual_mean": _f(g.residual.mean()),
                              "unpriced_mean": _f(g.n_unpriced.mean()) if "n_unpriced" in g.columns else None, "total_source": src}
    out["frame"] = df
    show = [c for c in ("slate", "team", "stat", "total", "total_source", "fixed", "residual", "coverage", "n_unpriced", "floored", "unpriced_volume") if c in df.columns]
    out["rows"] = df[show].to_dict("records")
    return out


# --------------------------------------------------------------------------- checks
def _check(status: str, component: str, name: str, detail: str, value=None) -> dict:
    return {"status": status, "component": component, "name": name, "detail": detail, "value": value}


def build_checks(players: Optional[dict] = None, field: Optional[dict] = None, lineups: Optional[dict] = None,
                 pool: Optional[dict] = None, min_n: int = 30, environment: Optional[dict] = None, usage: Optional[dict] = None) -> List[dict]:
    """pass / warn / fail / na rows over the reports. Thresholds are deliberately loose; the tables carry the detail."""
    out: List[dict] = []
    # ---- game environment (layer 1)
    if environment and environment.get("n"):
        for q, o in environment["quantities"].items():
            if o.get("n", 0) < 20:
                out.append(_check("na", "environment", f"{q}", f"only {o.get('n', 0)} team-games with actuals", o.get("n")))
                continue
            status = "pass"
            if o.get("mae_league") is not None and o["mae"] > o["mae_league"]:
                status = "fail"
            cov = o.get("coverage_10_90")
            if cov is not None and not (0.65 <= cov <= 0.92):
                status = "warn" if status == "pass" else status
            r3 = lambda v: "-" if v is None else f"{v:.3g}"
            out.append(_check(status, "environment", f"{q}",
                              f"bias {o['bias']:+.2f}, MAE {r3(o['mae'])} vs league-mean {r3(o.get('mae_league'))}, corr {r3(o.get('corr'))}, "
                              f"10-90 coverage {'-' if cov is None else f'{cov:.0%}'} over {o['n']} team-games", o.get("mae")))
    # ---- usage (layer 2): the priced marginals should add up to the team, not past it
    if usage and usage.get("n"):
        for stat, o in usage["stats"].items():
            if o.get("n", 0) < 4:
                out.append(_check("na", "usage", f"{stat} coverage", f"only {o.get('n', 0)} team-slates", o.get("n")))
                continue
            med = o["coverage_median"]
            over = o.get("over_share") or 0.0
            status = "pass" if (0.70 <= med <= 1.00 and over <= 0.5) else ("warn" if 0.55 <= med <= 1.15 else "fail")
            out.append(_check(status, "usage", f"{stat} coverage",
                              f"settled lines claim a median {med:.0%} of the team total (mean {o['coverage_mean']:.0%}, max {o['coverage_max']:.0%}); "
                              f"{o['over_share']:.0%} of team-slates past 100%, residual floored on {o['floored_share']:.0%}; "
                              f"{o['unpriced_mean']:.1f} unpriced players share {o['residual_mean']:.1f} per team", med))
    # ---- players
    if players and players.get("n"):
        o = players["overall"]
        if o.get("n", 0) < min_n:
            out.append(_check("na", "projections", "sample", f"only {o.get('n', 0)} player-games with actuals", o.get("n")))
        else:
            bias = o.get("bias")
            out.append(_check("pass" if abs(bias) <= 1.0 else ("warn" if abs(bias) <= 2.0 else "fail"), "projections", "mean bias",
                              f"actual - projected averages {bias:+.2f} fpts over {o['n']} player-games", bias))
            if o.get("baseline_mae") is not None:
                better = o["mae"] < o["baseline_mae"]
                out.append(_check("pass" if better else "fail", "projections", "beats salary baseline",
                                  f"MAE {o['mae']:.2f} vs salary-only {o['baseline_mae']:.2f}", o["mae"] - o["baseline_mae"]))
            cov = o.get("coverage_10_90")
            if cov is not None:
                out.append(_check("pass" if 0.72 <= cov <= 0.88 else ("warn" if 0.65 <= cov <= 0.92 else "fail"), "projections",
                                  "10-90 interval coverage", f"{cov:.0%} of actuals inside the projected 10-90 band (ideal 80%); PIT shape {o.get('pit_shape')}", cov))
            sr = o.get("sd_ratio")
            if sr is not None:
                out.append(_check("pass" if 0.85 <= sr <= 1.15 else ("warn" if 0.7 <= sr <= 1.3 else "fail"), "projections", "spread",
                                  f"realized error sd / projected sd = {sr:.2f} (1 = calibrated; >1 too confident)", sr))
        own = players.get("ownership") or {}
        if own.get("n", 0) >= min_n:
            out.append(_check("pass" if own["mae"] <= 3.0 else ("warn" if own["mae"] <= 6.0 else "fail"), "ownership", "projected ownership",
                              f"MAE {own['mae']:.1f} pts, Spearman {own.get('spearman') if own.get('spearman') is None else round(own['spearman'], 2)}, chalk bias {own.get('chalk_bias'):+.1f}", own["mae"]))
        ownf = players.get("ownership_field") or {}
        if ownf.get("n", 0) >= min_n:
            out.append(_check("pass" if ownf["mae"] <= 3.0 else ("warn" if ownf["mae"] <= 6.0 else "fail"), "field", "sampled field exposure",
                              f"MAE {ownf['mae']:.1f} pts against actual ownership, chalk bias {ownf.get('chalk_bias'):+.1f}", ownf["mae"]))
    # ---- field
    if field and field.get("n"):
        c = field["cash_line"]
        if c.get("n"):
            share = c["in_band_share"]
            out.append(_check("pass" if share >= 0.6 else ("warn" if share >= 0.4 else "fail"), "field", "cash line",
                              f"actual cash line inside the sim's p10-p90 in {share:.0%} of {c['n']} contests; bias {c['bias']:+.1f} pts", share))
        w = field["winning_score"]
        if w.get("n"):
            out.append(_check("pass" if (w["in_band_share"] or 0) >= 0.5 else "warn", "field", "winning score",
                              f"actual winner inside the sim's p10-p90 in {w['in_band_share']:.0%} of contests; bias {w['bias']:+.1f}", w["in_band_share"]))
        d = field["dupes"]
        if d.get("n"):
            r = d.get("top_share_ratio_median")
            out.append(_check("pass" if r is not None and 0.5 <= r <= 2.0 else ("warn" if r is not None and 0.25 <= r <= 4.0 else "fail"), "dupes", "field duplication",
                              f"the most-duplicated lineup's share of the field: sim / actual = {r:.2f}x (median over {d['n']} contests); "
                              f"the sim's field is {'too concentrated' if r > 1 else 'too flat'} at the top" if r is not None else "no dupe data", r))
    # ---- our lineups
    if lineups and lineups.get("settled"):
        gap = (lineups.get("sim_roi") or 0) - (lineups.get("actual_roi") or 0)
        out.append(_check("pass" if abs(gap) <= 0.25 else ("warn" if abs(gap) <= 0.75 else "fail"), "chooser", "sim ROI vs actual ROI",
                          f"sim {lineups['sim_roi']:+.0%} vs actual {lineups['actual_roi']:+.0%} over {lineups['settled']} entries (${lineups['fees']:.0f})", gap))
        cg = (lineups.get("sim_cash_pct") or 0) - (lineups.get("actual_cash_rate") or 0)
        out.append(_check("pass" if abs(cg) <= 0.08 else ("warn" if abs(cg) <= 0.15 else "fail"), "chooser", "sim cash % vs actual",
                          f"sim {lineups['sim_cash_pct']:.0%} vs actual {lineups['actual_cash_rate']:.0%}", cg))
        d = lineups.get("dupes")
        if d and d["n"] >= 10:
            out.append(_check("pass" if abs(d["bias"]) <= 1.0 else ("warn" if abs(d["bias"]) <= 3.0 else "fail"), "dupes", "our lineups' dupes",
                              f"predicted {d['pred_mean']:.1f} vs actual {d['actual_mean']:.1f} copies per entry (Spearman {d.get('spearman')})", d["bias"]))
    # ---- pool
    if pool and pool.get("contests"):
        sp = pool.get("spearman_median")
        out.append(_check("pass" if sp is not None and sp >= 0.15 else ("warn" if sp is not None and sp > 0.0 else "fail"), "simulator", "sim ROI ranks lineups",
                          f"median Spearman(sim ROI, actual ROI) across {pool['contests']} contests = {sp}; positive in {pool.get('spearman_positive_share', 0):.0%}", sp))
        if pool.get("picked_actual_roi") is not None:
            diff = pool["picked_actual_roi"] - pool["random_k_actual_roi"]
            out.append(_check("pass" if diff > 0.05 else ("warn" if diff > -0.05 else "fail"), "chooser", "picks vs random from pool",
                              f"picked {pool['picked_actual_roi']:+.0%} vs random K {pool['random_k_actual_roi']:+.0%} vs top-K by sim ROI {pool['topk_by_sim_roi_actual_roi']:+.0%} (pool {pool['pool_actual_roi']:+.0%}); "
                              f"picks beat random in {pool.get('contests_picked_beats_random')}/{pool['contests']} contests", diff))
    return out


# --------------------------------------------------------------------------- confidence
def _score(x, target: float, tol: float) -> Optional[float]:
    """1 at the target, 0 at target +- tol, linear between."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(np.clip(1.0 - abs(float(x) - target) / tol, 0.0, 1.0))


def _score_log_ratio(x, tol_factor: float) -> Optional[float]:
    """1 at a ratio of 1, 0 at tol_factor or 1/tol_factor."""
    if x is None or not np.isfinite(x) or x <= 0:
        return None
    return float(np.clip(1.0 - abs(np.log(float(x))) / np.log(tol_factor), 0.0, 1.0))


def _score_up(x, lo: float, hi: float) -> Optional[float]:
    """0 at lo, 1 at hi (more is better)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(np.clip((float(x) - lo) / (hi - lo), 0.0, 1.0))


GRADES = ((85, "A"), (70, "B"), (55, "C"), (40, "D"), (0, "F"))
MIN_PLAYERS, MIN_CONTESTS = 200, 10


def confidence_report(reports: dict) -> dict:
    """
    One number for a backtest: how far each stage's projections were from
    what happened, rolled up. Every stage metric becomes a 0-1 score against
    an explicit target, a stage is the mean of its scores, and the confidence
    is the geometric mean of the stages (one broken stage drags it down),
    scaled 0-100 with a grade and the weakest stage named. Stages without
    data are left out and listed; thin samples cap the number.

      projections  where the actual sits in the projected distribution (10-90 coverage -> 80%, sd ratio -> 1),
                   skill (MAE vs the salary line), bias
      ownership    MAE, Spearman, chalk bias of the projected ownership
      field        exposure MAE vs real ownership, cash line and winning score inside the sim's band, chalk bias
      dupes        our lineups' predicted / actual copies, the field's top-lineup share sim / real
      simulator    sim cash % vs actual cash rate, how the sim's ROI ranks lineups (Spearman)
    """
    P = reports.get("players") or {}; o = P.get("overall") or {}
    own = P.get("ownership") or {}; ownf = P.get("ownership_field") or {}
    F = reports.get("field") or {}; cl = F.get("cash_line") or {}; ws = F.get("winning_score") or {}; fd = F.get("dupes") or {}
    L = reports.get("lineups") or {}; ld = L.get("dupes") or {}
    pool = reports.get("pool") or {}
    skill = (o.get("baseline_mae") / o["mae"]) if o.get("baseline_mae") and o.get("mae") else None
    stages = {
        "projections": {"n": o.get("n", 0), "metrics": {
            "coverage_10_90": (o.get("coverage_10_90"), _score(o.get("coverage_10_90"), 0.80, 0.20)),
            "sd_ratio": (o.get("sd_ratio"), _score_log_ratio(o.get("sd_ratio"), 2.0)),
            "skill_vs_salary": (skill, _score_up(skill, 0.85, 1.15)),
            "bias": (o.get("bias"), _score(o.get("bias"), 0.0, 3.0))}},
        "ownership": {"n": own.get("n", 0), "metrics": {
            "mae": (own.get("mae"), _score_up(own.get("mae"), 6.0, 1.5) if own.get("mae") is not None else None),
            "spearman": (own.get("spearman"), _score_up(own.get("spearman"), 0.60, 0.95)),
            "chalk_bias": (own.get("chalk_bias"), _score(own.get("chalk_bias"), 0.0, 10.0))}},
        "field": {"n": cl.get("n", 0), "metrics": {
            "exposure_mae": (ownf.get("mae"), _score_up(ownf.get("mae"), 6.0, 1.5) if ownf.get("mae") is not None else None),
            "cash_line_in_band": (cl.get("in_band_share"), _score(cl.get("in_band_share"), 0.80, 0.30)),
            "winning_score_in_band": (ws.get("in_band_share"), _score(ws.get("in_band_share"), 0.80, 0.30)),
            "exposure_chalk_bias": (ownf.get("chalk_bias"), _score(ownf.get("chalk_bias"), 0.0, 10.0))}},
        "dupes": {"n": fd.get("n", 0), "metrics": {
            "our_copies_ratio": (ld.get("ratio"), _score_log_ratio(ld.get("ratio"), 3.0)),
            "top_share_ratio": (fd.get("top_share_ratio_median"), _score_log_ratio(fd.get("top_share_ratio_median"), 3.0))}},
        "simulator": {"n": L.get("settled", 0), "metrics": {
            "cash_calibration": ((L.get("sim_cash_pct"), L.get("actual_cash_rate")),
                                 _score(L.get("sim_cash_pct") - L.get("actual_cash_rate"), 0.0, 0.15) if L.get("sim_cash_pct") is not None and L.get("actual_cash_rate") is not None else None),
            "roi_rank_spearman": (pool.get("spearman_median"), _score_up(pool.get("spearman_median"), 0.0, 0.30))}},
    }
    out_stages, missing = {}, []
    for name, st in stages.items():
        scored = {k: v for k, v in st["metrics"].items() if v[1] is not None}
        if not scored:
            missing.append(name)
            continue
        out_stages[name] = {"score": round(float(np.mean([v[1] for v in scored.values()])), 3), "n": int(st["n"] or 0),
                            "metrics": {k: {"value": (list(v[0]) if isinstance(v[0], tuple) else v[0]), "score": round(v[1], 3)} for k, v in scored.items()}}
    if not out_stages:
        return {"score": None, "grade": None, "weakest": None, "stages": {}, "missing": missing, "notes": ["no stage has data"]}
    geo = float(np.exp(np.mean([np.log(max(s["score"], 0.01)) for s in out_stages.values()])))
    score = round(100 * geo, 1)
    notes = []
    n_players = out_stages.get("projections", {}).get("n", 0); n_contests = out_stages.get("field", {}).get("n", 0) or out_stages.get("simulator", {}).get("n", 0)
    if n_players and n_players < MIN_PLAYERS:
        score = min(score, 60.0); notes.append(f"only {n_players} player-games: capped at 60")
    if n_contests and n_contests < MIN_CONTESTS:
        score = min(score, 70.0); notes.append(f"only {n_contests} contests: capped at 70")
    if missing:
        notes.append("no data for: " + ", ".join(missing))
    weakest = min(out_stages.items(), key=lambda kv: kv[1]["score"])
    grade = next(g for lo, g in GRADES if score >= lo)
    return {"score": score, "grade": grade, "weakest": weakest[0], "weakest_score": weakest[1]["score"], "stages": out_stages, "missing": missing, "notes": notes}


def summarise_checks(checks: Iterable[dict]) -> Dict[str, int]:
    out = {"pass": 0, "warn": 0, "fail": 0, "na": 0}
    for c in checks:
        out[c["status"]] = out.get(c["status"], 0) + 1
    return out
