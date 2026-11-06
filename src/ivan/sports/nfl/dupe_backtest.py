# src/ivan/sports/nfl/dupe_backtest.py
"""
The season test of the showdown dupe model: every slate scored by a model that saw
only the fields before it.

`build_dupe_model.py backtest` leaves one slate out of a fit on every other field,
which lets the later slates into every fold and weighs the fit toward whichever
season holds the most fields (113 of the live model's 140 are November 2025).
The live pipeline never has that: at lock it holds a model fitted on the past.
This module replays that situation over the scraped fields, slate by slate in
time order (the rolling origin), and answers four questions the Dupes page
cannot:

1. forward error: for each slate, fit on the fields locked before it (all history,
   or this season only) and score the slate: copy-weighted log MAE and bias, the
   calibration slope (1 = the spread of predictions matches the spread of copies; the
   bias is log(actual / pred) as on the Dupes page, > 0 = the field duplicated more),
   the most duplicated lineup's predicted / actual copies, and the share of entries
   the 100 most duplicated lineups hold, predicted and real (what the field sampler's
   concentration follows). The live model file is scored the same way on the slates
   after its fitted_on date: that is the error the chooser actually carried.
2. variants side by side: the live terms on all history (copy-weighted, so the
   biggest fields set the fit), the same with every past slate weighted equally
   (dupe_model.fit_weights "slate": November 2025 held 55 % of the copies in
   2026-11 but 10 of 28 slates), this season only, ownership only, and the live
   terms plus the entry-cap term (log entries per user; dupe_model ALL_TERMS
   "emax"), so a candidate change is judged forward before it is turned on.
3. calibration of the forward predictions pooled over slates: by predicted-copies
   bin, and by the contest's entries-per-user cap (single entry, 5, 20, 150),
   from projected ownership (what the pipeline has) and from actual ownership
   (what the rule would do with perfect ownership), so a miss is attributed to the
   rule or to the ownership upstream.
4. our entries: the lineups we entered (contest_entry_result) found in the real
   field, their predicted copies against the copies the field held, per slate and
   by the contest's cap, next to the field's own copies per entry. This is the
   number the chooser divides the prize by, and whether we are chalkier than the
   field we play in.

Confidence: the per-slate scores are bootstrapped over slates (90 % band of the
mean), so a variant is only called better when the band says so. Output under
data/nfl/diagnostics/dupe_backtest/<stamp>/: report.txt, slates.csv (slate x
variant), calibration.csv, by_entry_max.csv, ours.csv, ours_by_cap.csv, summary.json.

    scripts/nfl/backtest_dupes.py [--since 2026-09-01] [--until ...] [--train-since ...] [--model path] [--no-write]
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from ivan.core.utils import normalize_name
from ivan.database.session import read_sql
from ivan.sports.nfl import dupe_model as DM
from ivan.sports.nfl.paths import DIAG_DIR

OUT_DIR = DIAG_DIR / "dupe_backtest"
SEASON_START = "2026-09-01"
TOP_N = 100                                   # the "chalk" the concentration check looks at
EMAX_BUCKETS = ((1, 1, "single entry"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 10 ** 9, "21-150"))
PRED_BIN_EDGES = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10 ** 9]
N_BOOT = 2000


@dataclass
class Variant:
    key: str
    terms: Optional[tuple]                    # None = a fixed model file, not refitted
    train_since: Optional[str] = None         # fit only on fields locked on or after this date
    model: Optional[DM.ProductDupeModel] = None
    note: str = ""
    balance: Optional[str] = None             # "slate": every slate carries the same weight in the fit (dupe_model.fit_weights)


def default_variants(model: Optional[DM.ProductDupeModel] = None, train_since: Optional[str] = None, ablation: bool = False) -> List[Variant]:
    """
    The live file and the live terms on all history (the "asof" row), then either the alternatives (this season only, slate-balanced,
    ownership only, plus the candidate terms) or, with `ablation`, the live terms with each one dropped in turn and each candidate
    term from ALL_TERMS added in turn: the term-by-term evidence for what stays in TERMS.
    """
    season = train_since or SEASON_START
    out = []
    if model is not None:
        out.append(Variant("live file", None, model=model, note=f"dupe_model_v2.json fitted {model.fitted_on} on {model.contests.get('showdown', '?')} fields; scored after that date"))
    if ablation:
        live = tuple(DM.TERMS)
        out.append(Variant("asof", live, note=f"the live terms {list(live)}, refitted on every field before the slate"))
        out += [Variant(f"asof -{t}", tuple(x for x in live if x != t), note=f"the live terms without {t}") for t in live]
        out += [Variant(f"asof +{t}", live + (t,), note=f"the live terms plus {t}") for t in DM.ALL_TERMS if t not in live]
        out.append(Variant("asof own only", (), note="ownership product alone (k, beta)"))
        return out
    out += [Variant("asof", DM.TERMS, note=f"the live terms {list(DM.TERMS)}, refitted on every field before the slate (copy-weighted: the biggest fields set the fit)"),
            Variant("asof -emax", tuple(t for t in DM.TERMS if t != "emax"), note="the live terms without the entry-cap term (the model before 2026-11-05)"),
            Variant("asof balanced", DM.TERMS, balance="slate", note="the live terms, every past slate weighted the same in the fit"),
            Variant("asof season", DM.TERMS, train_since=season, note=f"the live terms, fields since {season} only"),
            Variant("asof own only", (), note="ownership product alone (k, beta)")]
    return out


# --------------------------------------------------------------------------- the frame
def lineup_key(tokens: Sequence[str]) -> Optional[Tuple[str, frozenset]]:
    """(captain, frozenset of flex) with normalised names from 'CPT name' / 'FLEX name' tokens; None when there is no captain."""
    cpt, flex = None, []
    for tok in tokens:
        tok = tok.strip()
        if not tok or tok in ("None", "nan"):
            continue
        slot, _, name = tok.partition(" ")
        n = normalize_name(name)
        if slot == "CPT":
            cpt = n
        else:
            flex.append(n)
    if cpt is None or not flex:
        return None
    return (cpt, frozenset(flex))


def key_of_entry_string(s: str) -> Optional[Tuple[str, frozenset]]:
    """The key of a contest_entry_result lineup ('CPT A FLEX B FLEX C ...')."""
    return lineup_key(re.split(r"\s(?=(?:CPT|FLEX)\s)", str(s).strip()))


def our_lineups(session, username: Optional[str] = None) -> pd.DataFrame:
    """Our scraped entries: contest_id, entry_id, key. Every username in contest_entry_result when none is given (the table holds only our finishes)."""
    user = username or os.environ.get("IVAN_DK_USERNAME")
    q = "SELECT contest_id, entry_id, username, lineup FROM contest_entry_result" + (" WHERE username = :u" if user else "")
    st = text(q).bindparams(u=user) if user else text(q)
    df = read_sql(session, st)
    if df.empty:
        return pd.DataFrame(columns=["contest_id", "entry_id", "username", "key"])
    df["key"] = df.lineup.map(key_of_entry_string)
    return df[df.key.notna()][["contest_id", "entry_id", "username", "key"]]


def season_frame(session, since: Optional[str] = None, until: Optional[str] = None, username: Optional[str] = None) -> pd.DataFrame:
    """
    Every showdown field's unique lineups (dupe_model.training_frame) with the slate (lock hour), the lineup key,
    the number of our entries on the lineup (`ours`) and the entry-cap bucket. `since` / `until` bound the
    slates scored, not the training history: the frame carries every field so the fits can look back.
    """
    df = DM.training_frame(session)
    df = df[df.is_showdown == 1.0].copy() if len(df) else df
    if df.empty:
        return df
    df["slate"] = df.date.dt.strftime("%Y-%m-%d %H")
    df["key"] = df.lineup.map(lambda s: lineup_key(str(s).split("|")))
    ours = our_lineups(session, username)
    if len(ours):
        n = ours.groupby(["contest_id", "key"]).size().rename("ours").reset_index()
        df = df.merge(n, on=["contest_id", "key"], how="left")
        df["ours"] = df.ours.fillna(0).astype(int)
    else:
        df["ours"] = 0
    df["emax_bucket"] = df.max_entries.map(lambda v: _bucket(v, EMAX_BUCKETS))
    df["in_window"] = True
    if since:
        df.loc[df.date < pd.Timestamp(since), "in_window"] = False
    if until:
        df.loc[df.date >= pd.Timestamp(until) + pd.Timedelta(days=1), "in_window"] = False
    return df.reset_index(drop=True)


def _bucket(v, buckets) -> Optional[str]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return "unknown"
    for lo, hi, label in buckets:
        if lo <= x <= hi:
            return label
    return None


# --------------------------------------------------------------------------- the forward pass
def slate_scores(te: pd.DataFrame, pred: np.ndarray) -> dict:
    """
    dupe_model.scores plus the chalk concentration: the share of entries in the TOP_N most duplicated lineups, predicted
    and real. log_bias here is the copy-weighted mean of log(actual / pred), the Dupes page's sign: > 0 means the field
    duplicated more than the model said.
    """
    sc = DM.scores(te, pred)
    sc.pop("by_contest", None)
    sc["log_bias"] = -sc["log_bias"]          # dupe_model scores log(pred / actual); this report and the Dupes page read log(actual / pred)
    c = te["count"].values.astype(float)
    tops_a, tops_p = [], []
    for _, g in te.assign(pred=pred).groupby("contest_id"):
        n = float(g.entries.iloc[0])
        tops_a.append(np.sort(g["count"].values)[::-1][:TOP_N].sum() / n)
        tops_p.append(np.sort(g.pred.values)[::-1][:TOP_N].sum() / n)
    sc["top100_share_actual"] = round(float(np.mean(tops_a)), 4)
    sc["top100_share_pred"] = round(float(np.mean(tops_p)), 4)
    sc["entries"] = int(te.groupby("contest_id").entries.first().sum())
    sc["unique"] = int(len(te))
    return sc


def forward(frame: pd.DataFrame, variants: Sequence[Variant], min_contests: int = DM.MIN_CONTESTS) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Slate by slate in time order, each variant fitted on the fields locked before the slate (or the fixed file) and
    scored on it. Returns (one row per slate x variant, the forward predictions per variant aligned to the frame's
    rows; NaN where the slate was not scored).
    """
    rows = []
    preds = {v.key: np.full(len(frame), np.nan) for v in variants}
    slates = sorted(frame.loc[frame.in_window, "slate"].unique())
    for s in slates:
        te_mask = (frame.slate == s).values
        te = frame[te_mask]
        lock = te.date.min()
        for v in variants:
            if v.terms is None:
                m = v.model
                if m is None or (m.fitted_on and lock < pd.Timestamp(m.fitted_on) + pd.Timedelta(days=1)):
                    continue                                        # the file saw this slate (or was built the same day): not forward
                p = m.params["showdown"]
                n_train, train_slates = m.contests.get("showdown", 0), None
            else:
                tr = frame[frame.date < lock]
                if v.train_since:
                    tr = tr[tr.date >= pd.Timestamp(v.train_since)]
                if tr.contest_id.nunique() < min_contests:
                    continue
                n_slates = tr.slate.nunique()
                p = DM._fit_format(tr, v.terms if n_slates >= DM.MIN_SLATES else (), balance=v.balance)
                n_train, train_slates = int(tr.contest_id.nunique()), int(n_slates)
            pr = DM.predict(p, te)
            preds[v.key][te_mask] = pr
            sc = slate_scores(te, pr)
            tr_2025 = int(tr[tr.date < pd.Timestamp(SEASON_START)].contest_id.nunique()) if v.terms is not None else None
            rows.append({"slate": s, "lock": lock, "variant": v.key, "contests": int(te.contest_id.nunique()), "train_contests": n_train, "train_slates": train_slates,
                         "train_before_season": tr_2025,
                         **{k: p.get(k, 0.0) for k in DM.PARAM_KEYS}, **sc})
    return pd.DataFrame(rows), preds


def summarize(slates: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Per variant: the mean of each slate score with a 90 % bootstrap band over slates (slates are the unit of noise, not lineups)."""
    rng = np.random.default_rng(seed)
    out = []
    for v, g in slates.groupby("variant", sort=False):
        row = {"variant": v, "slates": int(len(g)), "contests": int(g.contests.sum())}
        for col in ("log_mae", "log_bias", "log_slope", "top_lineup_ratio_median", "top100_share_pred", "top100_share_actual"):
            x = g[col].values.astype(float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                row[col] = np.nan; row[col + "_lo"] = np.nan; row[col + "_hi"] = np.nan
                continue
            row[col] = float(x.mean())
            if x.size > 1:
                b = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
                row[col + "_lo"], row[col + "_hi"] = float(np.percentile(b, 5)), float(np.percentile(b, 95))
            else:
                row[col + "_lo"] = row[col + "_hi"] = row[col]
        out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- calibration of the forward predictions
def calibration(frame: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Lineups, copies, mean predicted and mean actual copies by predicted-copies bin over the scored rows."""
    ok = np.isfinite(pred)
    d = frame[ok].assign(pred=pred[ok])
    if d.empty:
        return pd.DataFrame(columns=["bin", "lineups", "copies", "pred_mean", "actual_mean", "ratio"])
    labels = [f"{PRED_BIN_EDGES[i]}-{PRED_BIN_EDGES[i + 1]}" if PRED_BIN_EDGES[i + 1] < 10 ** 9 else f"{PRED_BIN_EDGES[i]}+" for i in range(len(PRED_BIN_EDGES) - 1)]
    d["bin"] = pd.cut(d.pred, PRED_BIN_EDGES, labels=labels, right=False, include_lowest=True)
    g = d.groupby("bin", observed=True)
    out = pd.DataFrame({"lineups": g.size(), "copies": g["count"].sum(), "pred_mean": g.pred.mean(), "actual_mean": g["count"].mean()}).reset_index()
    out["ratio"] = out.actual_mean / out.pred_mean
    return out


def by_entry_max(frame: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Copy-weighted log(actual / pred) and the top-lineup ratio by the contest's entries-per-user cap."""
    ok = np.isfinite(pred)
    d = frame[ok].assign(pred=pred[ok])
    rows = []
    order = [b[2] for b in EMAX_BUCKETS] + ["unknown"]
    for b in order:
        g = d[d.emax_bucket == b]
        if g.empty:
            continue
        c = g["count"].values.astype(float)
        le = np.log(np.maximum(g.pred.values, 1e-3)) - np.log(c)
        tops = [float(x.pred.max() / x["count"].max()) for _, x in g.groupby("contest_id")]
        rows.append({"entry_max": b, "contests": int(g.contest_id.nunique()), "entries": int(g.groupby("contest_id").entries.first().sum()),
                     "copies_per_entry_actual": float((c * c).sum() / c.sum()), "copies_per_entry_pred": float((g.pred.values * c).sum() / c.sum()),
                     "log_bias_w": float(-(le * c).sum() / c.sum()), "log_mae_w": float((np.abs(le) * c).sum() / c.sum()),
                     "top_lineup_ratio_median": float(np.median(tops))})
    return pd.DataFrame(rows)


def ours_table(frame: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Per slate, our entered lineups: predicted copies (the field's, ourselves included) against the copies the field held."""
    ok = np.isfinite(pred) & (frame.ours.values > 0)
    d = frame[ok].assign(pred=pred[ok])
    rows = []
    for s, g in d.groupby("slate"):
        w = g.ours.values.astype(float)                                   # an entry we made twice counts twice
        ratio = np.log(g["count"].values.astype(float)) - np.log(np.maximum(g.pred.values, 1e-3))
        rows.append({"slate": s, "contests": int(g.contest_id.nunique()), "entries": int(w.sum()), "unique_lineups": int(len(g)),
                     "pred_copies_mean": float((g.pred.values * w).sum() / w.sum()), "actual_copies_mean": float((g["count"].values * w).sum() / w.sum()),
                     "share_alone_actual": float(((g["count"].values == 1) * w).sum() / w.sum()),
                     "share_pred_under_2": float(((g.pred.values < 2.0) * w).sum() / w.sum()),
                     "log_bias": float((ratio * w).sum() / w.sum()), "log_mae": float((np.abs(ratio) * w).sum() / w.sum()),
                     "max_actual": int(g["count"].max()), "pred_at_max": float(g.loc[g["count"].idxmax(), "pred"])})
    return pd.DataFrame(rows)


def ours_by_cap(frame: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Our entered lineups by the contest's entries-per-user cap: where we play, how duplicated we are there, and the model's bias there."""
    ok = np.isfinite(pred) & (frame.ours.values > 0)
    d = frame[ok].assign(pred=pred[ok])
    rows = []
    for b in [x[2] for x in EMAX_BUCKETS] + ["unknown"]:
        g = d[d.emax_bucket == b]
        if g.empty:
            continue
        w = g.ours.values.astype(float)
        ratio = np.log(g["count"].values.astype(float)) - np.log(np.maximum(g.pred.values, 1e-3))
        # the field's own copies per entry (sum c^2 / entries, the copies a random entry of the field has), weighted like our entries
        all_rows = frame[frame.contest_id.isin(g.contest_id.unique())]
        cpe = all_rows.groupby("contest_id").apply(lambda x: (x["count"].astype(float) ** 2).sum() / float(x.entries.iloc[0]), include_groups=False)
        field_cpe = float((g.contest_id.map(cpe).values * w).sum() / w.sum())
        rows.append({"entry_max": b, "contests": int(g.contest_id.nunique()), "entries": int(w.sum()),
                     "pred_copies_mean": float((g.pred.values * w).sum() / w.sum()), "actual_copies_mean": float((g["count"].values * w).sum() / w.sum()),
                     "field_copies_per_entry": field_cpe, "share_alone_actual": float(((g["count"].values == 1) * w).sum() / w.sum()),
                     "log_bias": float((ratio * w).sum() / w.sum()), "log_mae": float((np.abs(ratio) * w).sum() / w.sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- run and report
@dataclass
class Result:
    since: Optional[str]
    until: Optional[str]
    variants: List[Variant]
    slates: pd.DataFrame
    summary: pd.DataFrame
    calibration: Dict[str, pd.DataFrame]          # variant -> table (projected ownership)
    calibration_actual_own: Optional[pd.DataFrame]
    by_entry_max: Dict[str, pd.DataFrame]
    ours: Dict[str, pd.DataFrame]
    ours_cap: Dict[str, pd.DataFrame] = field(default_factory=dict)
    n_fields: int = 0
    n_slates_total: int = 0
    notes: List[str] = field(default_factory=list)


def run(session, since: Optional[str] = SEASON_START, until: Optional[str] = None, train_since: Optional[str] = None,
        model: Optional[DM.ProductDupeModel] = None, username: Optional[str] = None, actual_own: bool = True, ablation: bool = False, log=print) -> Result:
    model = DM.load() if model is None else model
    variants = default_variants(model, train_since, ablation)
    log(f"[dupe backtest] building the frame from every scraped showdown field...")
    frame = season_frame(session, since, until, username)
    if frame.empty:
        raise SystemExit("no scraped showdown fields with projected ownership")
    log(f"[dupe backtest] {frame.contest_id.nunique()} fields, {frame.slate.nunique()} slates, {int(frame.in_window.sum())} rows in the window; "
        f"{int((frame.ours > 0).sum())} of our lineups found in the fields")
    slates, preds = forward(frame, variants)
    if slates.empty:
        raise SystemExit("no slate in the window had enough history before it to fit on")
    summary = summarize(slates)
    cal = {v.key: calibration(frame, preds[v.key]) for v in variants if np.isfinite(preds[v.key]).any()}
    bem = {v.key: by_entry_max(frame, preds[v.key]) for v in variants if np.isfinite(preds[v.key]).any()}
    ours = {v.key: ours_table(frame, preds[v.key]) for v in variants if np.isfinite(preds[v.key]).any()}
    ours_cap = {v.key: ours_by_cap(frame, preds[v.key]) for v in variants if np.isfinite(preds[v.key]).any()}
    cal_act = None
    if actual_own:
        # the same forward rule from the ownership the field actually showed: a miss that stays is the rule's, one that goes was ownership
        log("[dupe backtest] rebuilding the frame from actual ownership...")
        fa = DM.training_frame(session, own_col="ownership_actual")
        fa = fa[fa.is_showdown == 1.0].copy()
        if len(fa):
            fa["slate"] = fa.date.dt.strftime("%Y-%m-%d %H"); fa["in_window"] = frame.set_index(["contest_id", "lineup"]).in_window.reindex(pd.MultiIndex.from_frame(fa[["contest_id", "lineup"]])).fillna(False).values
            fa["ours"] = 0; fa["emax_bucket"] = fa.max_entries.map(lambda v: _bucket(v, EMAX_BUCKETS))
            ref = [v for v in variants if v.key == "asof"]
            _, pa = forward(fa, ref)
            cal_act = calibration(fa, pa["asof"])
    return Result(since, until, variants, slates, summary, cal, cal_act, bem, ours, ours_cap, n_fields=int(frame.contest_id.nunique()), n_slates_total=int(frame.slate.nunique()))


def _fmt_band(row, col, nd=2) -> str:
    v, lo, hi = row[col], row.get(col + "_lo", np.nan), row.get(col + "_hi", np.nan)
    if not np.isfinite(v):
        return "   n/a"
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]" if np.isfinite(lo) else f"{v:+.{nd}f}"


def format_report(r: Result) -> str:
    L = []
    L.append(f"dupe model season backtest: slates {r.since or 'start'} .. {r.until or 'now'}; {r.n_fields} showdown fields over {r.n_slates_total} slates in the archive")
    L.append("each slate scored by a model fitted only on the fields locked before it (the live file: on the slates after its fitted_on date)")
    L.append("")
    L.append("variants:")
    for v in r.variants:
        L.append(f"  {v.key:16s} {v.note}")
    L.append("")
    L.append("forward scores, mean over slates [90 % bootstrap band over slates]; log bias = copy-weighted log(actual / pred), > 0 = the field duplicated "
             "more than predicted; slope 1 = calibrated spread; top100 = share of entries in the 100 most duplicated lineups, pred vs actual")
    L.append(f"  {'variant':16s} {'slates':>6s} {'log MAE':>22s} {'log bias':>22s} {'slope':>22s} {'top lineup pred/act':>22s} {'top100 pred':>11s} {'top100 act':>10s}")
    for _, row in r.summary.iterrows():
        L.append(f"  {row.variant:16s} {int(row.slates):6d} {_fmt_band(row, 'log_mae'):>22s} {_fmt_band(row, 'log_bias'):>22s} {_fmt_band(row, 'log_slope'):>22s} "
                 f"{_fmt_band(row, 'top_lineup_ratio_median'):>22s} {row.top100_share_pred:11.3f} {row.top100_share_actual:10.3f}")
    L.append("")
    L.append("per slate (the asof variant, then the others' log bias):")
    piv = r.slates.pivot_table(index="slate", columns="variant", values="log_bias", aggfunc="first")
    base = r.slates[r.slates.variant == ("asof" if (r.slates.variant == "asof").any() else r.slates.variant.iloc[0])].set_index("slate")
    others = [v.key for v in r.variants if v.key in piv.columns]
    L.append(f"  {'slate':14s} {'fields':>6s} {'entries':>9s} {'train':>6s} {'(2025)':>6s} {'MAE':>5s} {'slope':>6s} {'top1 p/a':>8s} {'top100 p':>8s} {'top100 a':>8s} | bias: " + " ".join(f"{k:>14s}" for k in others))
    for s, row in base.iterrows():
        L.append(f"  {s:14s} {int(row.contests):6d} {int(row.entries):9,d} {int(row.train_contests):6d} {int(row.train_before_season or 0):6d} {row.log_mae:5.2f} {row.log_slope:6.2f} {row.top_lineup_ratio_median:8.2f} "
                 f"{row.top100_share_pred:8.3f} {row.top100_share_actual:8.3f} |       " + " ".join((f"{piv.loc[s, k]:+14.2f}" if np.isfinite(piv.loc[s, k]) else f"{'':>14s}") for k in others))
    L.append("")
    for key in ("asof", "live file"):
        if key in r.calibration:
            L.append(f"calibration of the forward predictions ({key}; bins of predicted copies, pooled over the scored slates):")
            L.append(f"  {'pred':>9s} {'lineups':>9s} {'copies':>10s} {'pred mean':>9s} {'actual':>8s} {'ratio':>6s}")
            for _, c in r.calibration[key].iterrows():
                L.append(f"  {str(c.bin):>9s} {int(c.lineups):9,d} {int(c.copies):10,d} {c.pred_mean:9.2f} {c.actual_mean:8.2f} {c.ratio:6.2f}")
            L.append("")
    if r.calibration_actual_own is not None and len(r.calibration_actual_own):
        L.append("the same (asof) from the ownership the field actually showed: what the rule would do with perfect ownership")
        L.append(f"  {'pred':>9s} {'lineups':>9s} {'copies':>10s} {'pred mean':>9s} {'actual':>8s} {'ratio':>6s}")
        for _, c in r.calibration_actual_own.iterrows():
            L.append(f"  {str(c.bin):>9s} {int(c.lineups):9,d} {int(c.copies):10,d} {c.pred_mean:9.2f} {c.actual_mean:8.2f} {c.ratio:6.2f}")
        L.append("")
    L.append("by the contest's entries-per-user cap (copy-weighted; bias > 0 = the field duplicated more than predicted):")
    for key, t in r.by_entry_max.items():
        L.append(f"  {key}:")
        L.append(f"    {'cap':14s} {'fields':>6s} {'entries':>9s} {'c/e actual':>10s} {'c/e pred':>9s} {'bias':>6s} {'MAE':>5s} {'top1 p/a':>8s}")
        for _, b in t.iterrows():
            L.append(f"    {b.entry_max:14s} {int(b.contests):6d} {int(b.entries):9,d} {b.copies_per_entry_actual:10.1f} {b.copies_per_entry_pred:9.1f} {b.log_bias_w:+6.2f} {b.log_mae_w:5.2f} {b.top_lineup_ratio_median:8.2f}")
    L.append("")
    L.append("our entries (predicted copies of the lineups we entered, ourselves included, vs the copies the field held):")
    for key, t in r.ours.items():
        if t.empty:
            continue
        L.append(f"  {key}:")
        L.append(f"    {'slate':14s} {'fields':>6s} {'entries':>7s} {'pred':>6s} {'actual':>6s} {'alone':>6s} {'bias':>6s} {'MAE':>5s} {'worst':>6s} {'pred@':>6s}")
        for _, o in t.iterrows():
            L.append(f"    {o.slate:14s} {int(o.contests):6d} {int(o.entries):7d} {o.pred_copies_mean:6.2f} {o.actual_copies_mean:6.2f} {o.share_alone_actual:6.2f} {o.log_bias:+6.2f} {o.log_mae:5.2f} {int(o.max_actual):6d} {o.pred_at_max:6.1f}")
        w = t.entries.values.astype(float)
        L.append(f"    {'all':14s} {int(t.contests.sum()):6d} {int(w.sum()):7d} {(t.pred_copies_mean * w).sum() / w.sum():6.2f} {(t.actual_copies_mean * w).sum() / w.sum():6.2f} "
                 f"{(t.share_alone_actual * w).sum() / w.sum():6.2f} {(t.log_bias * w).sum() / w.sum():+6.2f} {(t.log_mae * w).sum() / w.sum():5.2f}")
    L.append("")
    L.append("our entries by the contest's cap (where we play; field c/e = the copies a random entry of the same fields has, weighted like our entries: ours above it = we are chalkier than the field):")
    for key, t in r.ours_cap.items():
        if t.empty:
            continue
        L.append(f"  {key}:")
        L.append(f"    {'cap':14s} {'fields':>6s} {'entries':>7s} {'pred':>7s} {'actual':>7s} {'field c/e':>9s} {'alone':>6s} {'bias':>6s} {'MAE':>5s}")
        for _, o in t.iterrows():
            L.append(f"    {o.entry_max:14s} {int(o.contests):6d} {int(o.entries):7d} {o.pred_copies_mean:7.1f} {o.actual_copies_mean:7.1f} {o.field_copies_per_entry:9.1f} {o.share_alone_actual:6.2f} {o.log_bias:+6.2f} {o.log_mae:5.2f}")
    if r.notes:
        L.append("")
        L += [f"note: {n}" for n in r.notes]
    return "\n".join(L)


def write(r: Result, out_dir: Optional[Path] = None, stamp: Optional[str] = None) -> Path:
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir or OUT_DIR) / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(format_report(r))
    r.slates.to_csv(out / "slates.csv", index=False)
    r.summary.to_csv(out / "summary.csv", index=False)
    pd.concat([t.assign(variant=k) for k, t in r.calibration.items()] + ([r.calibration_actual_own.assign(variant="asof (actual own)")] if r.calibration_actual_own is not None else []),
              ignore_index=True).to_csv(out / "calibration.csv", index=False)
    pd.concat([t.assign(variant=k) for k, t in r.by_entry_max.items()], ignore_index=True).to_csv(out / "by_entry_max.csv", index=False)
    pd.concat([t.assign(variant=k) for k, t in r.ours.items()], ignore_index=True).to_csv(out / "ours.csv", index=False)
    pd.concat([t.assign(variant=k) for k, t in r.ours_cap.items()], ignore_index=True).to_csv(out / "ours_by_cap.csv", index=False)
    summary = {"since": r.since, "until": r.until, "fields": r.n_fields, "slates_total": r.n_slates_total, "written_at": datetime.now().isoformat(timespec="seconds"),
               "variants": [{"key": v.key, "terms": list(v.terms) if v.terms is not None else None, "train_since": v.train_since, "balance": v.balance, "note": v.note} for v in r.variants],
               "summary": json.loads(r.summary.to_json(orient="records"))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return out
