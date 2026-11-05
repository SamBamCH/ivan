# src/ivan/sports/nfl/dupe_model.py
"""
How many copies of a lineup a field holds, fitted on real fields.

The settle job keeps every scraped contest's lineups (data/nfl/derived/
contests/<id>/lineups.parquet). Each unique lineup in a field is a row: its
copy count, the field size and the players' projected ownership (what the
pipeline has before lock). The model is a product rule, four terms:

    copies(L) = N x k x prod_i (own_i / 100) ** beta
                  x exp(gamma x min(salary_left, knee) / 1000)
                  x exp(delta_logproj x log(1 + points below the best lineup))
                  x exp(delta_emax x log(entries per user))

read as the probability that an entry lands on L times the field size N.
Ownership is the base; the public spends to the cap, so money left on the
table cuts copies (gamma < 0, flat past `knee` dollars); optimizer users land
on the top projected lineups far more than ownership implies, so points below
the best lineup of the field cut copies (delta_logproj < 0; at prediction time
the reference is the best lineup of the candidate pool); and a single-entry
crowd lands on the chalk where a 150-max crowd spreads, so the contest's
entries-per-user cap (`historical_contest_information.multi_entry`) sets the
level (delta_emax < 0; EMAX_DEFAULT when a contest's cap is unknown).

Fitted per format by copy-weighted least squares on the log scale (the fit is
about where the entries land). Which terms stay is decided by the season
backtest (ivan.sports.nfl.dupe_backtest, scripts/nfl/backtest_dupes.py
--ablation): every slate scored by a fit on the fields locked before it, the
live terms with each one dropped and each candidate added. On 2026-11-05 over
16 slates: without the salary term the log MAE went 0.80 -> 0.94, without the
projection gap the top lineup's predicted copies fell to 0.22x the real ones,
the entry-cap term took the single-entry bias from +0.51 to -0.02 and the
150-max bias from -0.29 to -0.13 (slope 0.93 -> 0.98); a separate captain
exponent changed nothing, a linear projection-gap term and QB-stack deltas made
it worse, and a contest-size exponent ("logn", kept as a candidate in
ALL_TERMS) improved the pooled error but worsened the slate-level bias. A
zero-truncated Poisson likelihood (2026-10-14) and a two-stage quantile + GBM
model (2026-10) were tried and removed. Classic fields barely duplicate by
ownership (beta ~0.07) and keep the ownership-only rule.

The field samplers use the same function for their lineup weights (so the
sampled field's concentration follows it) and for the predicted copies of
every candidate (what the validation checks against real copies). Re-fit as
slates accumulate:

    scripts/nfl/build_dupe_model.py backtest | build [--cutoff]

`backtest` here is leave-one-slate-out over every field and is weighted
toward whichever season has the most fields; the acceptance test is the
season backtest above.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import text

from ivan.core import archive
from ivan.core.utils import normalize_name
from ivan.database.session import read_sql
from ivan.sports.nfl.paths import MODELS_DIR, NFL_DATA_DIR

FILE = "dupe_model_v2.json"
DERIVED_DIR = NFL_DATA_DIR / "derived" / "contests"
SLOT_IDS = {"CPT": 511, "FLEX": 512, "QB": 66, "RB": 67, "WR": 68, "TE": 69, "DST": 71}      # classic FLEX is 70
MIN_OWN = 0.05
SALARY_CAP = 50000
SALARY_KNEE = 2000.0      # dollars left past which the salary effect stops growing
CPT_MULT = 1.5
EMAX_DEFAULT = 20.0       # entries per user assumed when a contest's cap is unknown (the most common cap on the scraped fields)
N_REF = 50000.0           # the contest size the candidate "logn" term is centred on (copies ~ N^(1 + delta_logn) around it)
TERMS = ("salary", "logproj", "emax")       # the live terms (2026-11-05 forward ablation)
ALL_TERMS = TERMS + ("logn",)               # plus the candidates the ablation keeps testing
PARAM_KEYS = ("k", "beta", "gamma", "knee", "delta_logproj", "delta_emax", "delta_logn")
DEFAULTS = {"showdown": {"k": 0.044, "beta": 0.56, "gamma": 0.0, "knee": SALARY_KNEE, "delta_logproj": 0.0, "delta_emax": 0.0, "delta_logn": 0.0},
            "classic": {"k": 4e-5, "beta": 0.07, "gamma": 0.0, "knee": SALARY_KNEE, "delta_logproj": 0.0, "delta_emax": 0.0, "delta_logn": 0.0}}   # 2026-10-13 ownership-only fits
MIN_CONTESTS = 4
MIN_SLATES = 3            # distinct lock times a format needs before the extra terms are fitted (else ownership only)
FRAME_COLUMNS = ["contest_id", "count", "entries", "is_showdown", "n_players", "sum_log_own", "salary_left", "fpts_proj",
                 "lineup", "max_entries", "entry_fee", "proj_gap", "date"]


# --------------------------------------------------------------------------- lineups -> rows
def _slot_id(token_slot: str, is_showdown: bool) -> int:
    if token_slot == "FLEX":
        return 512 if is_showdown else 70
    return SLOT_IDS.get(token_slot, 70)


def contest_lineup_rows(session, contest_id: int, own_col: str = "ownership_proj", cache: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """
    One row per unique lineup of a scraped contest: copy count, field size, format, sum of log ownership from `own_col`,
    salary left, projected points (captain at 1.5x), the lineup string, the contest's entries-per-user cap and fee.
    `cache` (a dict the caller keeps across contests) holds the slate captures and each draft group's player blob,
    which every contest of a slate shares.
    """
    meta_p = DERIVED_DIR / str(contest_id) / "meta.json"
    if not meta_p.exists():
        return None
    h = read_sql(session, text("SELECT draft_group_id, contest_type, total_entries, entry_fee, multi_entry FROM historical_contest_information WHERE contest_id = :c")
                 .bindparams(c=int(contest_id)))
    if h.empty:
        return None
    h = h.iloc[0]
    is_showdown = str(h.contest_type).startswith("Showdown")
    players = read_sql(session, text("SELECT roster_slot_id AS slot, salary, position, team, ownership_proj, ownership_actual, fpts_proj FROM showdown_player WHERE contest_id = :c")
                       .bindparams(c=int(contest_id)))
    if players.empty or players[own_col].isna().all():
        return None
    cache = cache if cache is not None else {}
    if "caps" not in cache:
        cache["caps"] = {c.slate_key: c for c in archive.slates(session, "nfl")}
    cap = cache["caps"].get(str(h.draft_group_id))
    if cap is None:
        return None
    names = cache.setdefault("names", {}).get(cap.id)
    if names is None:
        names = {}
        for p in archive.load_blobs(session, cap.id).get("players", []):
            names[(int(p["roster_slot_id"]), normalize_name(str(p.get("name", ""))))] = (str(p.get("team")), str(p.get("position", "")).upper(), int(p.get("salary") or 0))
        cache["names"][cap.id] = names
    by_key = {(int(r.slot), r.team, str(r.position).upper(), int(r.salary)): float(getattr(r, own_col)) for r in players.itertuples() if pd.notna(getattr(r, own_col))}
    fp_key = {(int(r.slot), r.team, str(r.position).upper(), int(r.salary)): float(r.fpts_proj) for r in players.itertuples() if pd.notna(r.fpts_proj)}
    lp = pd.read_parquet(DERIVED_DIR / str(contest_id) / "lineups.parquet")
    slot_cols = [c for c in lp.columns if c.startswith("slot")]
    if not slot_cols:
        return None
    counts = lp[slot_cols].astype(str).agg("|".join, axis=1).value_counts()
    max_entries = float(h.multi_entry) if pd.notna(h.multi_entry) and float(h.multi_entry) > 0 else np.nan
    fee = float(h.entry_fee) if pd.notna(h.entry_fee) else np.nan
    rows = []
    for lineup, c in counts.items():
        lo, miss, salary, fpts = [], False, 0, 0.0
        for tok in lineup.split("|"):
            if tok in ("None", "nan", ""):
                continue
            slot, _, name = tok.partition(" ")
            sid = _slot_id(slot, is_showdown)
            info = names.get((sid, normalize_name(name)))
            own = by_key.get((sid, *info)) if info else None
            if own is None:
                miss = True
                break
            lo.append(np.log(max(own, MIN_OWN) / 100.0))
            salary += info[2]
            fpts += fp_key.get((sid, *info), 0.0) * (CPT_MULT if sid == 511 else 1.0)
        if miss or not lo:
            continue
        rows.append({"contest_id": int(contest_id), "count": int(c), "entries": int(len(lp)), "is_showdown": float(is_showdown),
                     "n_players": len(lo), "sum_log_own": float(sum(lo)), "salary_left": int(SALARY_CAP - salary), "fpts_proj": float(fpts),
                     "lineup": str(lineup), "max_entries": max_entries, "entry_fee": fee})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["proj_gap"] = df.fpts_proj.max() - df.fpts_proj          # points below the best lineup in the field
    return df


def training_frame(session, own_col: str = "ownership_proj", cutoff: Optional[str] = None) -> pd.DataFrame:
    ids = sorted(int(Path(p).parent.name) for p in glob.glob(str(DERIVED_DIR / "*" / "meta.json")))
    if not ids:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    q = ",".join(map(str, ids))
    dates = read_sql(session, text(f"SELECT contest_id, date FROM historical_contest_information WHERE contest_id IN ({q})"))
    if cutoff:
        dates = dates[pd.to_datetime(dates.date) < pd.Timestamp(cutoff)]
    frames, cache = [], {}
    for cid in dates.contest_id.astype(int):
        f = contest_lineup_rows(session, cid, own_col, cache)
        if f is not None and len(f):
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    df = pd.concat(frames, ignore_index=True).merge(dates, on="contest_id")
    df["date"] = pd.to_datetime(df.date)
    return df


def lineup_proj_gap(L, fpts, is_cpt) -> np.ndarray:
    """
    Each lineup's projected points below the best lineup of the set: L is lineups x players, fpts the
    projection per column (the captain column's projection is multiplied by CPT_MULT here when is_cpt
    marks it, so pass the plain per-player number for both slots).
    """
    f = np.asarray(fpts, dtype=float) * np.where(np.asarray(is_cpt, dtype=bool), CPT_MULT, 1.0)
    tot = np.asarray(L @ np.nan_to_num(f, nan=0.0)).ravel()
    return (tot.max() if tot.size else 0.0) - tot


def _log_emax(max_entries) -> np.ndarray:
    """log of the entries-per-user cap, the default cap where it is unknown or not positive."""
    x = np.asarray(pd.to_numeric(pd.Series(np.asarray(max_entries, dtype=object).ravel()), errors="coerce").values, dtype=float)
    x = np.where(np.isfinite(x) & (x > 0), x, EMAX_DEFAULT)
    return np.log(x)


def _log_n(entries) -> np.ndarray:
    return np.log(np.maximum(np.asarray(entries, dtype=float), 1.0)) - np.log(N_REF)


# --------------------------------------------------------------------------- the model
@dataclass
class ProductDupeModel:
    params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULTS.items()})
    fitted_on: Optional[str] = None
    cutoff: Optional[str] = None
    contests: Dict[str, int] = field(default_factory=dict)
    cv: Dict[str, dict] = field(default_factory=dict)

    def log_rate(self, is_showdown: bool, sum_log_own, salary_left=None, proj_gap=None, max_entries=None, entries=None) -> np.ndarray:
        """
        log of the lineup's relative weight (no scale): beta x sum log ownership, the salary term, the projection-gap term,
        the entry-cap term (the default cap when none is given) and the candidate contest-size term.
        """
        p = self.params["showdown" if is_showdown else "classic"]
        out = float(p["beta"]) * np.asarray(sum_log_own, dtype=float)
        gamma = float(p.get("gamma", 0.0) or 0.0)
        if gamma and salary_left is not None:
            out = out + gamma * np.minimum(np.maximum(np.asarray(salary_left, dtype=float), 0.0), float(p.get("knee", SALARY_KNEE))) / 1000.0
        dl = float(p.get("delta_logproj", 0.0) or 0.0)
        if dl and proj_gap is not None:
            out = out + dl * np.log1p(np.maximum(np.nan_to_num(np.asarray(proj_gap, dtype=float), nan=0.0), 0.0))
        de = float(p.get("delta_emax", 0.0) or 0.0)
        if de:
            out = out + de * (float(np.log(EMAX_DEFAULT)) if max_entries is None else _log_emax(max_entries))
        dn = float(p.get("delta_logn", 0.0) or 0.0)
        if dn and entries is not None:
            out = out + dn * _log_n(entries)
        return out

    def copies(self, entries: float, is_showdown: bool, sum_log_own: np.ndarray, salary_left=None, proj_gap=None, max_entries=None) -> np.ndarray:
        """
        Expected copies of each lineup in a field of `entries`. sum_log_own = sum over its players of
        log(own/100) (captain included), salary_left in dollars, proj_gap the projected points below the best
        lineup (captain at 1.5x), max_entries the contest's entries-per-user cap. Terms whose inputs are not
        given fall back to the plain rule (the cap to EMAX_DEFAULT).
        """
        p = self.params["showdown" if is_showdown else "classic"]
        return float(entries) * float(p["k"]) * np.exp(self.log_rate(is_showdown, sum_log_own, salary_left, proj_gap, max_entries, entries))

    def weights(self, is_showdown: bool, sum_log_own: np.ndarray, salary_left=None, proj_gap=None, max_entries=None, entries=None) -> np.ndarray:
        """Relative lineup weights for the field sampler (the same function, without the scale)."""
        return np.exp(self.log_rate(is_showdown, sum_log_own, salary_left, proj_gap, max_entries, entries))

    def describe(self) -> str:
        parts = []
        for f, p in self.params.items():
            t = f"{f}: copies = N x {p['k']:.3g} x prod(own)^{p['beta']:.2f}"
            if p.get("gamma"):
                t += f" x exp({p['gamma']:.2f} x min($left, {p.get('knee', SALARY_KNEE):.0f})/1k)"
            if p.get("delta_logproj"):
                t += f" x exp({p['delta_logproj']:.2f} x log(1 + pts below best))"
            if p.get("delta_emax"):
                t += f" x exp({p['delta_emax']:.2f} x log(entries per user))"
            if p.get("delta_logn"):
                t += f" x (N/{N_REF:.0f})^{p['delta_logn']:.2f}"
            t += f" ({self.contests.get(f, 0)} fields)" if self.contests.get(f) else " (default)"
            parts.append(t)
        return "; ".join(parts)

    def save(self, model_dir: Path = MODELS_DIR) -> Path:
        path = Path(model_dir) / FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"params": self.params, "fitted_on": self.fitted_on, "cutoff": self.cutoff, "contests": self.contests, "cv": self.cv}, indent=1))
        return path


def sum_log_own(ownerships: Sequence[float]) -> float:
    return float(sum(np.log(max(float(o), MIN_OWN) / 100.0) for o in ownerships))


def load(model_dir=None) -> ProductDupeModel:
    """The fitted model from the model directory (the as-of version under a replay clock), or the defaults. Keys an older file carries that the model no longer has are ignored."""
    from ivan.sports.nfl.paths import models_dir
    path = Path(models_dir() if model_dir is None else model_dir) / FILE
    if not path.exists():
        return ProductDupeModel()
    try:
        raw = json.loads(path.read_text())
        params = {}
        for fmt, v in raw["params"].items():
            p = {**DEFAULTS.get(fmt, DEFAULTS["showdown"]), **{k: v[k] for k in PARAM_KEYS if k in v and v[k] is not None}}
            if v.get("balance"):
                p["balance"] = v["balance"]
            params[fmt] = p
        return ProductDupeModel(params=params, fitted_on=raw.get("fitted_on"), cutoff=raw.get("cutoff"), contests=raw.get("contests") or {}, cv=raw.get("cv") or {})
    except (OSError, ValueError, KeyError, TypeError):
        return ProductDupeModel()


# --------------------------------------------------------------------------- the fit
def _salary_x(d: pd.DataFrame, knee: float = SALARY_KNEE) -> np.ndarray:
    """The salary regressor: min(salary left, knee) / 1000, 0 when the frame has no salary."""
    if "salary_left" not in d or d.salary_left.isna().all():
        return np.zeros(len(d))
    return np.minimum(np.maximum(d.salary_left.fillna(0).values.astype(float), 0.0), knee) / 1000.0


VARIES = 1e-9             # a regressor enters the fit only when it varies (a constant column would split the intercept with k)


def _design(d: pd.DataFrame, terms, knee: float):
    """Columns of the log-linear fit for the terms present in the frame: (names, matrix)."""
    cols, names = [np.ones(len(d)), d.sum_log_own.values.astype(float)], ["const", "beta"]
    if "salary" in terms:
        sx = _salary_x(d, knee)
        if sx.std() > VARIES:
            cols.append(sx); names.append("gamma")
    if "logproj" in terms and "proj_gap" in d and d.proj_gap.notna().any():
        lg = np.log1p(np.maximum(d.proj_gap.fillna(0).values.astype(float), 0.0))
        if lg.std() > VARIES:
            cols.append(lg); names.append("delta_logproj")
    if "emax" in terms and "max_entries" in d and d.max_entries.notna().any():
        le = _log_emax(d.max_entries.values)
        if le.std() > VARIES:
            cols.append(le); names.append("delta_emax")
    if "logn" in terms:
        ln = _log_n(d.entries.values)
        if ln.std() > VARIES:
            cols.append(ln); names.append("delta_logn")
    return names, np.column_stack(cols)


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return coef


def fit_weights(d: pd.DataFrame, balance: Optional[str] = None) -> np.ndarray:
    """
    The least-squares weights: a lineup's copies (the fit is about where the entries land), or with balance "slate"
    the copies scaled so every slate (lock hour) carries the same total weight. The season backtest of 2026-11-05 found
    balancing worse forward (the small slates' levels are the noisiest), so the live fit is copy-weighted.
    """
    counts = d["count"].values.astype(float)
    if balance != "slate" or "date" not in d:
        return counts
    slate = pd.to_datetime(d.date).dt.strftime("%Y-%m-%d %H").values
    tot = pd.Series(counts).groupby(slate).transform("sum").values
    return counts / tot * (counts.sum() / len(np.unique(slate)))


def _fit_format(d: pd.DataFrame, terms=TERMS, knee: float = SALARY_KNEE, balance: Optional[str] = None) -> Dict[str, float]:
    """One format's parameters from its training rows: the regressors in `terms`, weighted least squares per fit_weights."""
    terms = set(terms or ())
    log_n = np.log(d.entries.values.astype(float))
    counts = d["count"].values.astype(float)
    w = fit_weights(d, balance)
    y = np.log(counts) - log_n
    names, X = _design(d, terms, knee)
    try:
        coef = _wls(X, y, w)
    except np.linalg.LinAlgError:
        coef = np.zeros(X.shape[1]); coef[names.index("beta")] = DEFAULTS["showdown"]["beta"]
    c = dict(zip(names, coef))
    p = {"k": float(np.exp(c["const"])), "beta": float(np.clip(c["beta"], 0.0, 2.0)), "gamma": float(np.clip(c.get("gamma", 0.0), -5.0, 1.0)), "knee": float(knee),
         "delta_logproj": float(np.clip(c.get("delta_logproj", 0.0), -5.0, 0.0)), "delta_emax": float(np.clip(c.get("delta_emax", 0.0), -3.0, 1.0)),
         "delta_logn": float(np.clip(c.get("delta_logn", 0.0), -1.0, 1.0))}
    if balance:
        p["balance"] = balance
    # re-centre k after clipping so the weighted mean is still matched
    m = ProductDupeModel(params={"showdown": p, "classic": p})
    resid = y - m.log_rate(True, d.sum_log_own.values, d.salary_left.values if "salary_left" in d else None,
                           d.proj_gap.values if "proj_gap" in d else None, d.max_entries.values if "max_entries" in d else None, d.entries.values)
    p["k"] = float(np.exp((resid * w).sum() / w.sum()))
    return p


def fit(df: pd.DataFrame, cutoff: Optional[str] = None, terms=TERMS, balance: Optional[str] = None) -> ProductDupeModel:
    """
    Fit per format: showdown with the given terms (ownership only until it has MIN_SLATES slates to generalise across),
    classic with the ownership rule alone (its fields barely duplicate by ownership and the season backtest scores
    showdown only, so no classic term has been earned); a format with too few fields keeps the defaults.
    """
    m = ProductDupeModel(fitted_on=datetime.now().strftime("%Y-%m-%d"), cutoff=cutoff)
    for fmt, flag in (("showdown", 1.0), ("classic", 0.0)):
        d = df[df.is_showdown == flag] if len(df) else df
        if len(d) and d.contest_id.nunique() >= MIN_CONTESTS:
            n_slates = d.date.dt.strftime("%Y-%m-%d %H").nunique() if "date" in d else d.contest_id.nunique()
            m.params[fmt] = _fit_format(d, terms if (fmt == "showdown" and n_slates >= MIN_SLATES) else (), balance=balance)
            m.contests[fmt] = int(d.contest_id.nunique())
    return m


# --------------------------------------------------------------------------- backtest
def scores(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Entry-weighted log error (the error on an entry drawn at random from the field), copies per entry, the top lineup. log_bias is log(pred / actual)."""
    c = df["count"].values.astype(float)
    w = c
    log_err = np.log(np.maximum(pred, 1e-3)) - np.log(c)
    tops = []
    for cid, g in df.assign(pred=pred).groupby("contest_id"):
        tops.append({"contest_id": int(cid), "entries": int(g.entries.iloc[0]), "actual_top": int(g["count"].max()), "pred_top": round(float(g.pred.max()), 1),
                     "actual_copies_per_entry": round(float((g["count"] ** 2).sum() / g.entries.iloc[0]), 1),
                     "pred_copies_per_entry": round(float((g.pred * g["count"]).sum() / g.entries.iloc[0]), 1)})
    lp = np.log(np.maximum(pred, 1e-3))
    lpm, lcm = (lp * w).sum() / w.sum(), (np.log(c) * w).sum() / w.sum()
    var = (w * (lp - lpm) ** 2).sum()
    slope = float((w * (lp - lpm) * (np.log(c) - lcm)).sum() / var) if var > 0 else np.nan      # 1 = calibrated spread of predictions
    return {"n": int(len(df)), "contests": int(df.contest_id.nunique()), "log_slope": round(slope, 3),
            "copies_per_entry_pred": round(float((pred * w).sum() / w.sum()), 2), "copies_per_entry_actual": round(float((c * w).sum() / w.sum()), 2),
            "log_mae": round(float((np.abs(log_err) * w).sum() / w.sum()), 3), "log_bias": round(float((log_err * w).sum() / w.sum()), 3),
            "top_lineup_ratio_median": round(float(np.median([t["pred_top"] / t["actual_top"] for t in tops])), 2), "by_contest": tops}


def predict(params: Dict[str, float], d: pd.DataFrame) -> np.ndarray:
    """Expected copies for the rows of a training frame under `params` (one format's parameters)."""
    m = ProductDupeModel(params={"showdown": params, "classic": params})
    return d.entries.values.astype(float) * float(params["k"]) * np.exp(m.log_rate(True, d.sum_log_own.values,
                                                                                   salary_left=d.salary_left.values if "salary_left" in d else None,
                                                                                   proj_gap=d.proj_gap.values if "proj_gap" in d else None,
                                                                                   max_entries=d.max_entries.values if "max_entries" in d else None,
                                                                                   entries=d.entries.values))


ABLATION = (("none", ()), ("salary", ("salary",)), ("salary+logproj", ("salary", "logproj")), ("live terms", TERMS), ("+logn", ALL_TERMS))


def backtest(df: pd.DataFrame, terms=TERMS) -> dict:
    """
    Leave one slate (lock time) out per format: fit on the rest with `terms`, score the held-out slate.
    Each fold also carries the held-out scores of the ablation under "ablation". The season backtest
    (dupe_backtest) is the acceptance test; this is the quick in-sample ablation the model build records.
    """
    out = {}
    df = df.copy()
    df["slate"] = df.date.dt.strftime("%Y-%m-%d %H")
    keys = ("copies_per_entry_pred", "copies_per_entry_actual", "log_mae", "log_bias", "top_lineup_ratio_median", "log_slope")
    for fmt, flag in (("showdown", 1.0), ("classic", 0.0)):
        d = df[df.is_showdown == flag]
        folds = []
        for held in sorted(d.slate.unique()):
            tr, te = d[d.slate != held], d[d.slate == held]
            if tr.contest_id.nunique() < MIN_CONTESTS or te.empty:
                continue
            p = _fit_format(tr, terms)
            sc = scores(te, predict(p, te))
            fold = {"held_out": held, "contests": int(te.contest_id.nunique()), **{k: round(p[k], 5 if k == "k" else 3) for k in PARAM_KEYS}, **{k: sc[k] for k in keys}}
            fold["ablation"] = {}
            for name, t in ABLATION:
                sa = scores(te, predict(_fit_format(tr, t), te))
                fold["ablation"][name] = {k: sa[k] for k in ("log_mae", "log_bias", "top_lineup_ratio_median", "log_slope")}
            folds.append(fold)
        out[fmt] = {"contests": int(d.contest_id.nunique()), "fit": _fit_format(d, terms) if d.contest_id.nunique() >= MIN_CONTESTS else None, "folds": folds}
    return out
