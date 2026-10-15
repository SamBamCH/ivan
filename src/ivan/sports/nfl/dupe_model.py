# src/ivan/sports/nfl/dupe_model.py
"""
How many copies of a lineup a field holds, fitted on real fields.

The settle job keeps every scraped contest's lineups (data/nfl/derived/
contests/<id>/lineups.parquet). Each unique lineup in a field is a row:
its copy count, the field size and the players' projected ownership (what
the sampler has before lock). The model is the product rule with a fitted
exponent and scale per format, times a salary term:

    copies(L) = N * k * prod_flex (own_i / 100) ** beta * (own_cpt / 100) ** beta_cpt
                  * exp(gamma * min(salary_left, knee) / 1000)
                  * exp(d_mates * min(QB's same-team WR/TE, 3) + d_two_qb * [2+ QBs] + d_no_qb * [no QB])

read as the probability that an entry lands on L: p(L) = k x the terms, and
copies ~ Poisson(N p(L)). A lineup is in the data only when at least one entry
landed on it, so the fit is a zero-truncated Poisson likelihood (METHOD "ztp"):
the lineups nobody entered are accounted for instead of ignored. Fitting by
least squares on the observed lineups ("wls", the earlier method, kept for the
ablation) drags the low end up and flattens the slope, which showed on the
Dupes page as mid-range lineups over-predicted and the chalk under-predicted.
Tried 2026-10-14: the truncated-Poisson fit did not help on the held-out,
copy-weighted scores (bias -0.7 to -1.1, slope unchanged at ~0.7): the counts
are far more dispersed than Poisson (the public clusters on a few lineups), so
METHOD stays "wls" and the slope has to come from a feature, not the likelihood.
The projection term is that feature: how far the lineup's projected points sit
below the best lineup in the field (proj_gap, captain at 1.5x); optimizer users
land on the top projected lineups far more than the ownership product implies.
Held out it took the log MAE from 0.89/0.94/0.83 to 0.80/0.90/0.79 and the
calibration slope from 0.73/0.74/1.06 to 0.79/0.81/1.00, so it is in TERMS. At
prediction time the reference is the best lineup of the candidate pool. The salary term (2026-10-14, from the Dupes page): the public
spends to the cap, so a lineup leaving money on the table is duplicated far
less than its ownership alone says (lineups leaving $2k+ were over-predicted
about 8x without it); the effect flattens past `knee` dollars left. The
captain's ownership gets its own exponent (the public picks captains by points,
not value, so captain chalk concentrates differently), and the stack terms carry
the combinations ownership alone misses: a QB with his own pass catchers, two
QBs, no QB. Each term is kept only while the leave-one-slate-out backtest
(`backtest`, the ablation `build_dupe_model.py backtest` prints) says it helps:
on the 27 October 2026 showdown fields the salary term cut the held-out log MAE
from 1.03/1.14/1.04 to 0.89/0.96/0.85, the captain exponent a little more
(0.89/0.94/0.83, captain ^0.69 vs flex ^0.45), while the stack deltas changed
sign between slates and made one fold worse, so TERMS leaves them out. Measured 2026-10-13 on 50 October fields: showdown k 0.044,
beta 0.56 (leave-one-slate-out bias within +-0.2 in log terms; the extreme
top lineup is still under-predicted, a tail the product rule does not
reach); classic k 4e-5, beta 0.07 on two slates, i.e. classic fields barely
duplicate by ownership (3-6 copies per entry).

The field samplers use the same function for their lineup weights (so the
sampled field's concentration follows it) and for the predicted copies of
every candidate (what the validation checks against real copies). Re-fit
as slates accumulate:

    scripts/nfl/build_dupe_model.py backtest | build [--cutoff]
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
MAX_MATES = 3
STACK_KEYS = ("qb_mates", "two_qb", "no_qb")
DEFAULTS = {"showdown": {"k": 0.044, "beta": 0.56, "beta_cpt": None, "gamma": 0.0, "knee": SALARY_KNEE, "delta_proj": 0.0, "stack": {}},
            "classic": {"k": 4e-5, "beta": 0.07, "beta_cpt": None, "gamma": 0.0, "knee": SALARY_KNEE, "delta_proj": 0.0, "stack": {}}}   # 2026-10-13; extra terms off
ALL_TERMS = ("salary", "cpt", "proj", "stack")
CPT_MULT = 1.5
METHOD = "wls"            # copy-weighted least squares on log copies; "ztp" = zero-truncated Poisson likelihood (held-out worse, 2026-10-14: see docstring)
TERMS = ("salary", "cpt", "proj")   # the default fit; the stack deltas were unstable across held-out slates (2026-10-14) and stay off
PASS_CATCHERS = {"WR", "TE"}
MIN_CONTESTS = 4
MIN_SLATES = 3            # distinct lock times a format needs before the extra terms are fitted (else ownership only)


# --------------------------------------------------------------------------- lineups -> rows
def _slot_id(token_slot: str, is_showdown: bool) -> int:
    if token_slot == "FLEX":
        return 512 if is_showdown else 70
    return SLOT_IDS.get(token_slot, 70)


def contest_lineup_rows(session, contest_id: int, own_col: str = "ownership_proj") -> Optional[pd.DataFrame]:
    """One row per unique lineup of a scraped contest: copy count, field size, format, sum of log ownership from `own_col`."""
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
    caps = {c.slate_key: c for c in archive.slates(session, "nfl")}
    cap = caps.get(str(h.draft_group_id))
    if cap is None:
        return None
    names = {}
    for p in archive.load_blobs(session, cap.id).get("players", []):
        names[(int(p["roster_slot_id"]), normalize_name(str(p.get("name", ""))))] = (str(p.get("team")), str(p.get("position", "")).upper(), int(p.get("salary") or 0))
    by_key = {(int(r.slot), r.team, str(r.position).upper(), int(r.salary)): float(getattr(r, own_col)) for r in players.itertuples() if pd.notna(getattr(r, own_col))}
    fp_key = {(int(r.slot), r.team, str(r.position).upper(), int(r.salary)): float(r.fpts_proj) for r in players.itertuples() if pd.notna(r.fpts_proj)}
    lp = pd.read_parquet(DERIVED_DIR / str(contest_id) / "lineups.parquet")
    slot_cols = [c for c in lp.columns if c.startswith("slot")]
    if not slot_cols:
        return None
    counts = lp[slot_cols].astype(str).agg("|".join, axis=1).value_counts()
    rows = []
    for lineup, c in counts.items():
        lo, miss, salary, cpt_lo, members, fpts = [], False, 0, np.nan, [], 0.0
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
            lv = np.log(max(own, MIN_OWN) / 100.0)
            lo.append(lv)
            if sid == 511:
                cpt_lo = lv
            salary += info[2]
            fpts += fp_key.get((sid, *info), 0.0) * (CPT_MULT if sid == 511 else 1.0)
            members.append((info[0], info[1]))          # (team, position)
        if miss or not lo:
            continue
        n_qb, mates = stack_features(members)
        rows.append({"contest_id": int(contest_id), "count": int(c), "entries": int(len(lp)), "is_showdown": float(is_showdown),
                     "n_players": len(lo), "sum_log_own": float(sum(lo)), "log_own_cpt": float(cpt_lo), "salary_left": int(SALARY_CAP - salary),
                     "n_qb": int(n_qb), "qb_mates": int(mates), "fpts_proj": float(fpts)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["proj_gap"] = df.fpts_proj.max() - df.fpts_proj          # points below the best lineup in the field
    return df


def training_frame(session, own_col: str = "ownership_proj", cutoff: Optional[str] = None) -> pd.DataFrame:
    ids = sorted(int(Path(p).parent.name) for p in glob.glob(str(DERIVED_DIR / "*" / "meta.json")))
    if not ids:
        return pd.DataFrame(columns=["contest_id", "count", "entries", "is_showdown", "n_players", "sum_log_own", "log_own_cpt", "salary_left", "n_qb", "qb_mates", "fpts_proj", "proj_gap", "date"])
    q = ",".join(map(str, ids))
    dates = read_sql(session, text(f"SELECT contest_id, date FROM historical_contest_information WHERE contest_id IN ({q})"))
    if cutoff:
        dates = dates[pd.to_datetime(dates.date) < pd.Timestamp(cutoff)]
    frames = []
    for cid in dates.contest_id.astype(int):
        f = contest_lineup_rows(session, cid, own_col)
        if f is not None and len(f):
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["contest_id", "count", "entries", "is_showdown", "n_players", "sum_log_own", "log_own_cpt", "salary_left", "n_qb", "qb_mates", "fpts_proj", "proj_gap", "date"])
    df = pd.concat(frames, ignore_index=True).merge(dates, on="contest_id")
    df["date"] = pd.to_datetime(df.date)
    return df


def stack_features(members) -> tuple[int, int]:
    """(number of QBs, the most same-team WR/TE any QB has) for a lineup given as (team, position) pairs."""
    qbs = [t for t, pos in members if pos == "QB"]
    mates = max((sum(1 for t, pos in members if t == qt and pos in PASS_CATCHERS) for qt in qbs), default=0)
    return len(qbs), int(mates)


def lineup_stack_features(L, teams, positions) -> tuple[np.ndarray, np.ndarray]:
    """
    stack_features for every row of a lineup matrix: L is lineups x players (sparse or dense),
    teams / positions per column. Returns (n_qb, qb_mates) arrays.
    """
    teams = np.asarray(teams, dtype=object)
    pos = np.asarray([str(p).upper() for p in positions], dtype=object)
    is_qb = (pos == "QB").astype(float)
    is_catcher = np.isin(pos, list(PASS_CATCHERS)).astype(float)
    n_qb = np.asarray(L @ is_qb).ravel()
    mates = np.zeros(L.shape[0])
    for t in set(teams[is_qb > 0].tolist()):
        has_qb = np.asarray(L @ (is_qb * (teams == t))).ravel() > 0
        m = np.asarray(L @ (is_catcher * (teams == t))).ravel()
        mates = np.where(has_qb, np.maximum(mates, m), mates)
    return n_qb.astype(int), mates.astype(int)


def lineup_proj_gap(L, fpts, is_cpt) -> np.ndarray:
    """
    Each lineup's projected points below the best lineup of the set: L is lineups x players, fpts the
    projection per column (the captain column's projection is multiplied by CPT_MULT here when is_cpt
    marks it, so pass the plain per-player number for both slots).
    """
    f = np.asarray(fpts, dtype=float) * np.where(np.asarray(is_cpt, dtype=bool), CPT_MULT, 1.0)
    tot = np.asarray(L @ np.nan_to_num(f, nan=0.0)).ravel()
    return (tot.max() if tot.size else 0.0) - tot


# --------------------------------------------------------------------------- the model
@dataclass
class ProductDupeModel:
    params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULTS.items()})
    fitted_on: Optional[str] = None
    cutoff: Optional[str] = None
    contests: Dict[str, int] = field(default_factory=dict)
    cv: Dict[str, dict] = field(default_factory=dict)

    def _salary_term(self, p: dict, salary_left) -> np.ndarray:
        """exp(gamma x min(salary left, knee) / 1000); 1 when the model has no salary term or no salary is given."""
        gamma = float(p.get("gamma", 0.0))
        if gamma == 0.0 or salary_left is None:
            return 1.0
        knee = float(p.get("knee", SALARY_KNEE))
        return np.exp(gamma * np.minimum(np.maximum(np.asarray(salary_left, dtype=float), 0.0), knee) / 1000.0)

    def _own_term(self, p: dict, sum_log_own, log_own_cpt) -> np.ndarray:
        """beta x sum of log ownership, with the captain's log ownership on its own exponent when the model has one."""
        slo = np.asarray(sum_log_own, dtype=float)
        bc = p.get("beta_cpt")
        if bc is None or log_own_cpt is None:
            return float(p["beta"]) * slo
        cpt = np.asarray(log_own_cpt, dtype=float)
        cpt = np.where(np.isnan(cpt), 0.0, cpt)
        return float(p["beta"]) * (slo - cpt) + float(bc) * cpt

    def _stack_term(self, p: dict, qb_mates, n_qb) -> np.ndarray:
        st = p.get("stack") or {}
        if not st or qb_mates is None or n_qb is None:
            return 0.0
        m = np.minimum(np.asarray(qb_mates, dtype=float), MAX_MATES)
        nq = np.asarray(n_qb, dtype=float)
        return (float(st.get("qb_mates", 0.0)) * m + float(st.get("two_qb", 0.0)) * (nq >= 2) + float(st.get("no_qb", 0.0)) * (nq == 0))

    def _proj_term(self, p: dict, proj_gap) -> np.ndarray:
        """delta_proj x points below the field's best lineup (delta_proj < 0: the top projected lineups draw the optimizer crowd)."""
        d = float(p.get("delta_proj", 0.0) or 0.0)
        if d == 0.0 or proj_gap is None:
            return 0.0
        return d * np.maximum(np.nan_to_num(np.asarray(proj_gap, dtype=float), nan=0.0), 0.0)

    def log_rate(self, is_showdown: bool, sum_log_own, salary_left=None, log_own_cpt=None, qb_mates=None, n_qb=None, proj_gap=None) -> np.ndarray:
        """log of the lineup's relative weight: the ownership, salary, projection and stack terms (no scale)."""
        p = self.params["showdown" if is_showdown else "classic"]
        out = self._own_term(p, sum_log_own, log_own_cpt) + self._stack_term(p, qb_mates, n_qb) + self._proj_term(p, proj_gap)
        return out + np.log(self._salary_term(p, salary_left))

    def copies(self, entries: float, is_showdown: bool, sum_log_own: np.ndarray, salary_left=None, log_own_cpt=None, qb_mates=None, n_qb=None,
               proj_gap=None) -> np.ndarray:
        """
        Expected copies of each lineup in a field of `entries`. sum_log_own = sum over its players of
        log(own/100) (captain included), salary_left in dollars, log_own_cpt the captain's log(own/100),
        qb_mates / n_qb from stack_features, proj_gap the projected points below the best lineup
        (captain at 1.5x). Terms whose inputs are not given fall back to the plain rule.
        """
        p = self.params["showdown" if is_showdown else "classic"]
        return float(entries) * float(p["k"]) * np.exp(self.log_rate(is_showdown, sum_log_own, salary_left, log_own_cpt, qb_mates, n_qb, proj_gap))

    def weights(self, is_showdown: bool, sum_log_own: np.ndarray, salary_left=None, log_own_cpt=None, qb_mates=None, n_qb=None, proj_gap=None) -> np.ndarray:
        """Relative lineup weights for the field sampler (the same function, without the scale)."""
        return np.exp(self.log_rate(is_showdown, sum_log_own, salary_left, log_own_cpt, qb_mates, n_qb, proj_gap))

    def describe(self) -> str:
        parts = []
        for f, p in self.params.items():
            t = f"{f}: copies = N x {p['k']:.3g} x prod(own)^{p['beta']:.2f}"
            if p.get("beta_cpt") is not None:
                t += f" (cpt ^{p['beta_cpt']:.2f})"
            if p.get("gamma"):
                t += f" x exp({p['gamma']:.2f} x min($left, {p.get('knee', SALARY_KNEE):.0f})/1k)"
            if p.get("delta_proj"):
                t += f" x exp({p['delta_proj']:.3f} x pts below best)"
            st = p.get("stack") or {}
            if st:
                t += " x exp(" + " ".join(f"{v:+.2f}*{k}" for k, v in st.items()) + ")"
            t += f" ({self.contests.get(f, 0)} fields, {p.get('method', 'wls')})" if self.contests.get(f) else " (default)"
            parts.append(t)
        return "; ".join(parts)

    def save(self, model_dir: Path = MODELS_DIR) -> Path:
        path = Path(model_dir) / FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"params": self.params, "fitted_on": self.fitted_on, "cutoff": self.cutoff, "contests": self.contests, "cv": self.cv}, indent=1))
        return path


def sum_log_own(ownerships: Sequence[float]) -> float:
    return float(sum(np.log(max(float(o), MIN_OWN) / 100.0) for o in ownerships))


def _salary_x(d: pd.DataFrame, knee: float = SALARY_KNEE) -> np.ndarray:
    """The salary regressor: min(salary left, knee) / 1000, 0 when the frame has no salary."""
    if "salary_left" not in d or d.salary_left.isna().all():
        return np.zeros(len(d))
    return np.minimum(np.maximum(d.salary_left.fillna(0).values.astype(float), 0.0), knee) / 1000.0


def _design(d: pd.DataFrame, terms, knee: float):
    """Columns of the log-linear fit for the terms present in the frame: (names, matrix)."""
    cols, names = [np.ones(len(d))], ["const"]
    slo = d.sum_log_own.values.astype(float)
    cpt = d.log_own_cpt.values.astype(float) if ("cpt" in terms and "log_own_cpt" in d and d.log_own_cpt.notna().any()) else None
    if cpt is not None:
        cpt = np.where(np.isnan(cpt), 0.0, cpt)
        cols += [slo - cpt, cpt]; names += ["beta", "beta_cpt"]
    else:
        cols.append(slo); names.append("beta")
    if "salary" in terms:
        sx = _salary_x(d, knee)
        if sx.std() > 0:
            cols.append(sx); names.append("gamma")
    if "proj" in terms and "proj_gap" in d and d.proj_gap.notna().any():
        pg = np.maximum(d.proj_gap.fillna(0).values.astype(float), 0.0)
        if pg.std() > 0:
            cols.append(pg); names.append("delta_proj")
    if "stack" in terms and "n_qb" in d:
        nq = d.n_qb.values.astype(float)
        for name, x in (("qb_mates", np.minimum(d.qb_mates.values.astype(float), MAX_MATES)), ("two_qb", (nq >= 2).astype(float)), ("no_qb", (nq == 0).astype(float))):
            if x.std() > 0:
                cols.append(x); names.append(name)
    return names, np.column_stack(cols)


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return coef


def _ztp(X: np.ndarray, counts: np.ndarray, log_n: np.ndarray, b0: np.ndarray) -> np.ndarray:
    """
    Zero-truncated Poisson regression: counts_i ~ Poisson(N_i exp(X_i b)) given counts_i >= 1.
    Maximises sum_i [c_i (log N_i + X_i b) - lam_i - log(1 - exp(-lam_i))] from b0 (the least-squares fit).
    """
    from scipy.optimize import minimize
    c = counts.astype(float)

    def f(b):
        eta = log_n + X @ b
        lam = np.exp(np.minimum(eta, 50.0))
        one_minus = -np.expm1(-lam)                                   # 1 - exp(-lam), accurate for small lam
        ll = c * eta - lam - np.log(np.maximum(one_minus, 1e-300))
        g_eta = c - lam - lam / np.maximum(np.expm1(lam), 1e-300)     # d/d eta: lam exp(-lam) / (1 - exp(-lam)) = lam / (exp(lam) - 1)
        return -ll.sum(), -(X.T @ g_eta)

    r = minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x if np.all(np.isfinite(r.x)) else b0


def _fit_format(d: pd.DataFrame, terms=TERMS, knee: float = SALARY_KNEE, method: str = METHOD) -> Dict[str, float]:
    """One format's parameters from its training rows: the regressors in `terms`, fitted by `method` (ztp or wls)."""
    terms = set(terms or ())
    log_n = np.log(d.entries.values.astype(float))
    counts = d["count"].values.astype(float)
    y = np.log(counts) - log_n
    names, X = _design(d, terms, knee)
    try:
        coef = _wls(X, y, counts)
        if method == "ztp":
            coef = _ztp(X, counts, log_n, coef)
    except np.linalg.LinAlgError:
        coef = np.zeros(X.shape[1]); coef[names.index("beta")] = DEFAULTS["showdown"]["beta"]
    c = dict(zip(names, coef))
    p = {"k": float(np.exp(c["const"])), "beta": float(np.clip(c["beta"], 0.0, 2.0)), "beta_cpt": float(np.clip(c["beta_cpt"], 0.0, 2.0)) if "beta_cpt" in c else None,
         "gamma": float(np.clip(c.get("gamma", 0.0), -5.0, 1.0)), "knee": float(knee), "delta_proj": float(np.clip(c.get("delta_proj", 0.0), -2.0, 0.0)),
         "stack": {k: float(np.clip(c[k], -3.0, 3.0)) for k in STACK_KEYS if k in c}, "method": method}
    if method == "wls":
        # least squares: re-centre k after clipping so the copy-weighted mean is still matched
        m = ProductDupeModel(params={"showdown": p, "classic": p})
        resid = y - m.log_rate(True, d.sum_log_own.values, d.salary_left.values if "salary_left" in d else None,
                               d.log_own_cpt.values if "log_own_cpt" in d else None,
                               d.qb_mates.values if "qb_mates" in d else None, d.n_qb.values if "n_qb" in d else None,
                               d.proj_gap.values if "proj_gap" in d else None)
        p["k"] = float(np.exp((resid * counts).sum() / counts.sum()))
    return p


def fit(df: pd.DataFrame, cutoff: Optional[str] = None, terms=TERMS, method: str = METHOD) -> ProductDupeModel:
    """Fit per format with the given terms (default TERMS: salary, cpt; ALL_TERMS adds the stack deltas) and method; formats with too few fields keep the defaults."""
    m = ProductDupeModel(fitted_on=datetime.now().strftime("%Y-%m-%d"), cutoff=cutoff)
    for fmt, flag in (("showdown", 1.0), ("classic", 0.0)):
        d = df[df.is_showdown == flag] if len(df) else df
        if len(d) and d.contest_id.nunique() >= MIN_CONTESTS:
            n_slates = d.date.dt.strftime("%Y-%m-%d %H").nunique() if "date" in d else d.contest_id.nunique()
            m.params[fmt] = _fit_format(d, terms if n_slates >= MIN_SLATES else (), method=method)   # the extra terms need slates to generalise across
            m.contests[fmt] = int(d.contest_id.nunique())
    return m


def load(model_dir=None) -> ProductDupeModel:
    """The fitted model from the model directory (the as-of version under a replay clock), or the defaults."""
    from ivan.sports.nfl.paths import models_dir
    path = Path(models_dir() if model_dir is None else model_dir) / FILE
    if not path.exists():
        return ProductDupeModel()
    try:
        raw = json.loads(path.read_text())
        return ProductDupeModel(params={k: {"beta_cpt": None, "gamma": 0.0, "knee": SALARY_KNEE, "delta_proj": 0.0, "stack": {}, "method": "wls", **dict(v)} for k, v in raw["params"].items()}, fitted_on=raw.get("fitted_on"), cutoff=raw.get("cutoff"),
                                contests=raw.get("contests") or {}, cv=raw.get("cv") or {})
    except (OSError, ValueError, KeyError):
        return ProductDupeModel()


# --------------------------------------------------------------------------- backtest
def scores(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Entry-weighted log error (the error on an entry drawn at random from the field), copies per entry, the top lineup."""
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
    return d.entries.values.astype(float) * m.copies(1.0, True, d.sum_log_own.values,
                                                     salary_left=d.salary_left.values if "salary_left" in d else None,
                                                     log_own_cpt=d.log_own_cpt.values if "log_own_cpt" in d else None,
                                                     qb_mates=d.qb_mates.values if "qb_mates" in d else None,
                                                     n_qb=d.n_qb.values if "n_qb" in d else None,
                                                     proj_gap=d.proj_gap.values if "proj_gap" in d else None)


ABLATION = (("none/wls", (), "wls"), ("salary+cpt/wls", ("salary", "cpt"), "wls"), ("salary+cpt+proj/wls", ("salary", "cpt", "proj"), "wls"),
            ("proj/wls", ("proj",), "wls"), ("all/wls", ALL_TERMS, "wls"), ("salary+cpt+proj/ztp", ("salary", "cpt", "proj"), "ztp"))


def backtest(df: pd.DataFrame, terms=TERMS, method: str = METHOD) -> dict:
    """
    Leave one slate (lock time) out per format: fit on the rest with `terms`, score the held-out slate.
    Each fold also carries the held-out scores of the ablation (none / salary / salary+cpt / all) under "ablation".
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
            p = _fit_format(tr, terms, method=method)
            sc = scores(te, predict(p, te))
            fold = {"held_out": held, "contests": int(te.contest_id.nunique()), "k": round(p["k"], 5), "beta": round(p["beta"], 3),
                    "beta_cpt": None if p["beta_cpt"] is None else round(p["beta_cpt"], 3), "gamma": round(p["gamma"], 3), "delta_proj": round(p["delta_proj"], 4),
                    "stack": {k: round(v, 3) for k, v in p["stack"].items()}, **{k: sc[k] for k in keys}}
            fold["ablation"] = {}
            for name, t, meth in ABLATION:
                sa = scores(te, predict(_fit_format(tr, t, method=meth), te))
                fold["ablation"][name] = {k: sa[k] for k in ("log_mae", "log_bias", "top_lineup_ratio_median", "log_slope")}
            folds.append(fold)
        out[fmt] = {"contests": int(d.contest_id.nunique()), "fit": _fit_format(d, terms, method=method) if d.contest_id.nunique() >= MIN_CONTESTS else None, "folds": folds}
    return out
