# src/ivan/sports/nfl/dupe_model.py
"""
How many copies of a lineup a field holds, fitted on real fields.

The settle job keeps every scraped contest's lineups (data/nfl/derived/
contests/<id>/lineups.parquet). Each unique lineup in a field is a row:
its copy count, the field size and the players' projected ownership (what
the sampler has before lock). The model is the product rule with a fitted
exponent and scale per format:

    copies(L) = N * k * prod_i (own_i / 100) ** beta

fitted by copy-weighted least squares on the log scale, so it targets the
copies of the lineup an entry actually holds (the chalk), not the million
singletons. Measured 2026-10-13 on 50 October fields: showdown k 0.044,
beta 0.56 (leave-one-slate-out bias within +-0.2 in log terms; the extreme
top lineup is still under-predicted, a tail the product rule does not
reach); classic k 4e-5, beta 0.07 on two slates, i.e. classic fields barely
duplicate by ownership (3-6 copies per entry).

The field samplers use the same function for their lineup weights (so the
sampled field's concentration follows it) and for the predicted copies of
every candidate (what the validation checks against real copies). Re-fit
as slates accumulate:

    scripts/nfl/build_dupe_model.py v2 backtest | build [--cutoff]
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
DEFAULTS = {"showdown": {"k": 0.044, "beta": 0.56}, "classic": {"k": 4e-5, "beta": 0.07}}     # 2026-10-13, 27 + 23 fields
MIN_CONTESTS = 4


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
    players = read_sql(session, text("SELECT roster_slot_id AS slot, salary, position, team, ownership_proj, ownership_actual FROM showdown_player WHERE contest_id = :c")
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
    lp = pd.read_parquet(DERIVED_DIR / str(contest_id) / "lineups.parquet")
    slot_cols = [c for c in lp.columns if c.startswith("slot")]
    if not slot_cols:
        return None
    counts = lp[slot_cols].astype(str).agg("|".join, axis=1).value_counts()
    rows = []
    for lineup, c in counts.items():
        lo, miss = [], False
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
        if miss or not lo:
            continue
        rows.append({"contest_id": int(contest_id), "count": int(c), "entries": int(len(lp)), "is_showdown": float(is_showdown),
                     "n_players": len(lo), "sum_log_own": float(sum(lo))})
    return pd.DataFrame(rows) if rows else None


def training_frame(session, own_col: str = "ownership_proj", cutoff: Optional[str] = None) -> pd.DataFrame:
    ids = sorted(int(Path(p).parent.name) for p in glob.glob(str(DERIVED_DIR / "*" / "meta.json")))
    if not ids:
        return pd.DataFrame(columns=["contest_id", "count", "entries", "is_showdown", "n_players", "sum_log_own", "date"])
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
        return pd.DataFrame(columns=["contest_id", "count", "entries", "is_showdown", "n_players", "sum_log_own", "date"])
    df = pd.concat(frames, ignore_index=True).merge(dates, on="contest_id")
    df["date"] = pd.to_datetime(df.date)
    return df


# --------------------------------------------------------------------------- the model
@dataclass
class ProductDupeModel:
    params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULTS.items()})
    fitted_on: Optional[str] = None
    cutoff: Optional[str] = None
    contests: Dict[str, int] = field(default_factory=dict)
    cv: Dict[str, dict] = field(default_factory=dict)

    def copies(self, entries: float, is_showdown: bool, sum_log_own: np.ndarray) -> np.ndarray:
        """Expected copies of each lineup in a field of `entries`; sum_log_own = sum over its players of log(own/100)."""
        p = self.params["showdown" if is_showdown else "classic"]
        return float(entries) * float(p["k"]) * np.exp(float(p["beta"]) * np.asarray(sum_log_own, dtype=float))

    def weights(self, is_showdown: bool, sum_log_own: np.ndarray) -> np.ndarray:
        """Relative lineup weights for the field sampler (the same function, without the scale)."""
        p = self.params["showdown" if is_showdown else "classic"]
        return np.exp(float(p["beta"]) * np.asarray(sum_log_own, dtype=float))

    def describe(self) -> str:
        return "; ".join(f"{f}: copies = N x {p['k']:.3g} x prod(own)^{p['beta']:.2f}" + (f" ({self.contests.get(f, 0)} fields)" if self.contests.get(f) else " (default)")
                         for f, p in self.params.items())

    def save(self, model_dir: Path = MODELS_DIR) -> Path:
        path = Path(model_dir) / FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"params": self.params, "fitted_on": self.fitted_on, "cutoff": self.cutoff, "contests": self.contests, "cv": self.cv}, indent=1))
        return path


def sum_log_own(ownerships: Sequence[float]) -> float:
    return float(sum(np.log(max(float(o), MIN_OWN) / 100.0) for o in ownerships))


def _fit_format(d: pd.DataFrame) -> Dict[str, float]:
    """Copy-weighted least squares of log(copies / N) on sum_log_own."""
    y = np.log(d["count"].values.astype(float)) - np.log(d.entries.values.astype(float))
    x = d.sum_log_own.values.astype(float)
    w = d["count"].values.astype(float)
    xm, ym = (x * w).sum() / w.sum(), (y * w).sum() / w.sum()
    denom = (w * (x - xm) ** 2).sum()
    beta = float(((w * (x - xm) * (y - ym)).sum()) / denom) if denom > 0 else DEFAULTS["showdown"]["beta"]
    beta = float(np.clip(beta, 0.0, 1.5))
    return {"k": float(np.exp(ym - beta * xm)), "beta": beta}


def fit(df: pd.DataFrame, cutoff: Optional[str] = None) -> ProductDupeModel:
    m = ProductDupeModel(fitted_on=datetime.now().strftime("%Y-%m-%d"), cutoff=cutoff)
    for fmt, flag in (("showdown", 1.0), ("classic", 0.0)):
        d = df[df.is_showdown == flag] if len(df) else df
        if len(d) and d.contest_id.nunique() >= MIN_CONTESTS:
            m.params[fmt] = _fit_format(d)
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
        return ProductDupeModel(params={k: dict(v) for k, v in raw["params"].items()}, fitted_on=raw.get("fitted_on"), cutoff=raw.get("cutoff"),
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
    return {"n": int(len(df)), "contests": int(df.contest_id.nunique()),
            "copies_per_entry_pred": round(float((pred * w).sum() / w.sum()), 2), "copies_per_entry_actual": round(float((c * w).sum() / w.sum()), 2),
            "log_mae": round(float((np.abs(log_err) * w).sum() / w.sum()), 3), "log_bias": round(float((log_err * w).sum() / w.sum()), 3),
            "top_lineup_ratio_median": round(float(np.median([t["pred_top"] / t["actual_top"] for t in tops])), 2), "by_contest": tops}


def backtest(df: pd.DataFrame) -> dict:
    """Leave one slate (lock time) out per format: fit on the rest, score the held-out slate."""
    out = {}
    df = df.copy()
    df["slate"] = df.date.dt.strftime("%Y-%m-%d %H")
    for fmt, flag in (("showdown", 1.0), ("classic", 0.0)):
        d = df[df.is_showdown == flag]
        folds = []
        for held in sorted(d.slate.unique()):
            tr, te = d[d.slate != held], d[d.slate == held]
            if tr.contest_id.nunique() < MIN_CONTESTS or te.empty:
                continue
            p = _fit_format(tr)
            pred = te.entries.values * p["k"] * np.exp(p["beta"] * te.sum_log_own.values)
            sc = scores(te, pred)
            folds.append({"held_out": held, "contests": int(te.contest_id.nunique()), "k": round(p["k"], 5), "beta": round(p["beta"], 3),
                          **{k: sc[k] for k in ("copies_per_entry_pred", "copies_per_entry_actual", "log_mae", "log_bias", "top_lineup_ratio_median")}})
        out[fmt] = {"contests": int(d.contest_id.nunique()), "fit": _fit_format(d) if d.contest_id.nunique() >= MIN_CONTESTS else None, "folds": folds}
    return out
