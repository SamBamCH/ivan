# src/ivan/sports/nfl/ownership_model.py
"""
A learned ownership model for showdown slates, fitted on the scraped
fields (showdown_player: what we projected before lock, what the public
actually rostered).

The field sampler projects ownership by generating value-driven lineups
and counting exposures; it has the wrong habits for roster-filler
positions (it never plays a 4-point defense, the public plays it 13% of
the time) and overplays the chalk. This model predicts each player-slot's
actual ownership directly from pre-lock features:

    projected points, salary, value, their ranks within the slate and slot,
    position, slot, the sampler's own projected ownership (so the sampler is
    a feature, not the answer), the team's share of the slate's projection,
    contest size and fee

with a gradient-boosted regressor, then renormalises each slot to one
lineup's worth (captain 100, flex 500). The backtest is a time split: fit
on contests before a date, score after it, against the sampler alone and
the sampler with the per-position calibration, on the ownership checks the
validation uses (MAE, Spearman, chalk bias). The live projection uses it
only when config ownership.model == "learned"; the as-of build writes a
version per cutoff like the other models.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import text

from ivan.database.session import read_sql
from ivan.sports.nfl.paths import MODELS_DIR

FILE = "ownership_model.pkl"
SHOWDOWN_SLOTS = (511, 512)        # the model and the calibration run on showdown slates only (ownership.apply_learned_ownership and
                                   # calibrate_ownership leave classic alone), so they train on showdown rows only. The classic rows
                                   # in showdown_player are per-slot shares of a per-player projection (a different quantity) and
                                   # they share slot_flex=0 with the captain rows: trained with them, the captain fit was biased
                                   # +0.5..+0.9 pts and its held-out MAE 1.58 vs 1.17 without them (2026-10-24, 57 contests)
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
SLOT_TOTAL = {511: 100.0, 512: 500.0}
FEATURES = ["slot_flex", "salary_k", "fpts_proj", "value", "own_sampler", "salary_rank", "proj_rank", "own_rank", "team_share", "n_players",
            "log_entries", "entry_fee"] + [f"pos_{p}" for p in POSITIONS]


# --------------------------------------------------------------------------- frames
def training_frame(session, cutoff: Optional[str] = None) -> pd.DataFrame:
    """One row per (showdown contest, player, slot) with the pre-lock features and the actual ownership (SHOWDOWN_SLOTS only)."""
    q = """SELECT p.contest_id, h.date, h.total_entries, h.entry_fee, p.player_id, p.roster_slot_id AS slot, p.salary, p.position, p.team,
                  p.ownership_proj AS own_sampler, p.fpts_proj, p.ownership_actual
           FROM showdown_player p JOIN historical_contest_information h ON h.contest_id = p.contest_id
           WHERE p.ownership_proj IS NOT NULL AND p.fpts_proj IS NOT NULL AND p.ownership_actual IS NOT NULL
             AND p.roster_slot_id IN (511, 512)"""
    if cutoff:
        q += " AND h.date < :c"
    stmt = text(q)
    df = read_sql(session, stmt.bindparams(c=cutoff) if cutoff else stmt)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df.date)
    return add_features(df, group=["contest_id"])


def slate_frame(session, players: Sequence, contest: Optional[dict] = None) -> pd.DataFrame:
    """The live slate's players (NFLPlayer rows) as a feature frame; contest gives total_entries / entry_fee (defaults otherwise)."""
    rows = [{"contest_id": 0, "player_id": int(p.id), "slot": int(p.roster_slot_id), "salary": float(p.salary or 0), "position": str(p.position),
             "team": str(p.team), "own_sampler": float(p.ownership_proj or 0.0), "fpts_proj": float(p.fpts_mean or 0.0),
             "total_entries": float((contest or {}).get("total_entries") or 50000), "entry_fee": float((contest or {}).get("entry_fee") or 5.0)}
            for p in players]
    return add_features(pd.DataFrame(rows), group=["contest_id"])


def add_features(df: pd.DataFrame, group: List[str]) -> pd.DataFrame:
    d = df.copy()
    d["slot_flex"] = (d.slot == 512).astype(float)
    d["salary_k"] = d.salary / 1000.0
    d["value"] = d.fpts_proj / np.maximum(d.salary_k, 0.1)
    g = group + ["slot"]
    d["salary_rank"] = d.groupby(g).salary.rank(ascending=False, method="first")
    d["proj_rank"] = d.groupby(g).fpts_proj.rank(ascending=False, method="first")
    d["own_rank"] = d.groupby(g).own_sampler.rank(ascending=False, method="first")
    d["n_players"] = d.groupby(g).salary.transform("size").astype(float)
    team_proj = d[d.slot == 512].groupby(group + ["team"]).fpts_proj.sum()
    slate_proj = d[d.slot == 512].groupby(group).fpts_proj.sum()
    share = (team_proj / slate_proj.reindex(team_proj.index.droplevel("team")).values).rename("team_share").reset_index()
    d = d.merge(share, on=group + ["team"], how="left")
    d["team_share"] = d.team_share.fillna(0.5)
    d["log_entries"] = np.log10(np.maximum(d.total_entries.astype(float), 10.0))
    d["entry_fee"] = d.entry_fee.astype(float)
    for p in POSITIONS:
        d[f"pos_{p}"] = (d.position == p).astype(float)
    return d


# --------------------------------------------------------------------------- model
@dataclass
class OwnershipModel:
    model: object                      # sklearn regressor on FEATURES -> ownership %
    features: List[str]
    fitted_on: str
    cutoff: Optional[str]
    n_contests: int
    n_rows: int

    def predict_frame(self, d: pd.DataFrame) -> np.ndarray:
        """Raw predictions renormalised per (contest, slot) to the lineup totals."""
        raw = np.maximum(self.model.predict(d[self.features].values.astype(float)), 0.0)
        out = raw.copy()
        for (cid, slot), idx in d.groupby(["contest_id", "slot"]).indices.items():
            tot = SLOT_TOTAL.get(int(slot))
            ssum = raw[idx].sum()
            if tot and ssum > 0:
                out[idx] = raw[idx] * tot / ssum
        return out

    def save(self, model_dir: Path = MODELS_DIR) -> Path:
        path = Path(model_dir) / FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path


def fit(df: pd.DataFrame, cutoff: Optional[str] = None, seed: int = 0) -> OwnershipModel:
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_leaf=20, random_state=seed)
    m.fit(df[FEATURES].values.astype(float), df.ownership_actual.values.astype(float))
    return OwnershipModel(m, list(FEATURES), datetime.now().strftime("%Y-%m-%d"), cutoff, int(df.contest_id.nunique()), int(len(df)))


def load(model_dir=None) -> Optional[OwnershipModel]:
    from ivan.sports.nfl.paths import models_dir
    path = Path(models_dir() if model_dir is None else model_dir) / FILE
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------- backtest
def _scores(actual: np.ndarray, pred: np.ndarray, cids: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    err = pred - actual
    chalk = []
    for cid in np.unique(cids):
        m = cids == cid
        top = np.argsort(-actual[m])[:5]
        chalk.append(err[m][top].mean())
    return {"n": int(len(actual)), "mae": round(float(np.abs(err).mean()), 3), "bias": round(float(err.mean()), 3),
            "spearman": round(float(spearmanr(actual, pred).correlation), 3), "chalk_bias": round(float(np.mean(chalk)), 3),
            "contests": int(len(np.unique(cids)))}


def _calibrate_like_live(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The per-slot/position calibration lines (ownership.calibrate_ownership), fitted on train, applied to test with renormalisation."""
    pred = test.own_sampler.values.astype(float).copy()
    for (slot, pos), g in train.groupby(["slot", "position"]):
        if len(g) < 30:
            continue
        x = g.fpts_proj.values if pos in ("DST", "K") else g.own_sampler.values
        if x.std() <= 0:
            continue
        b, a = np.polyfit(x.astype(float), g.ownership_actual.values.astype(float), 1)
        m = (test.slot.values == slot) & (test.position.values == pos)
        xt = test.fpts_proj.values[m] if pos in ("DST", "K") else test.own_sampler.values[m]
        pred[m] = np.maximum(a + b * xt.astype(float), 0.0)
    for (cid, slot), idx in test.groupby(["contest_id", "slot"]).indices.items():
        tot = SLOT_TOTAL.get(int(slot)); ssum = pred[idx].sum()
        if tot and ssum > 0:
            pred[idx] = pred[idx] * tot / ssum
    return pred


def backtest(df: pd.DataFrame, split_date: Optional[str] = None, test_share: float = 0.3, seed: int = 0) -> dict:
    """
    Time split when split_date is given (train before, test from it), else the last `test_share` of contests by date.
    Scores the sampler alone, the sampler + calibration (fitted on train) and the learned model on the test contests.
    """
    df = df.sort_values(["date", "contest_id"]).reset_index(drop=True)
    cids = df.drop_duplicates("contest_id")[["contest_id", "date"]]
    if split_date:
        test_ids = set(cids[cids.date >= pd.Timestamp(split_date)].contest_id)
    else:
        k = max(1, int(round(len(cids) * test_share)))
        test_ids = set(cids.contest_id.iloc[-k:])
    train, test = df[~df.contest_id.isin(test_ids)], df[df.contest_id.isin(test_ids)]
    if train.contest_id.nunique() < 10 or test.empty:
        return {"error": f"not enough contests to split (train {train.contest_id.nunique()}, test {test.contest_id.nunique()})"}
    model = fit(train, seed=seed)
    actual = test.ownership_actual.values.astype(float); c = test.contest_id.values
    out = {"train_contests": int(train.contest_id.nunique()), "test_contests": int(test.contest_id.nunique()),
           "train_until": str(train.date.max().date()), "test_from": str(test.date.min().date()),
           "sampler": _scores(actual, test.own_sampler.values.astype(float), c),
           "sampler_calibrated": _scores(actual, _calibrate_like_live(train, test), c),
           "learned": _scores(actual, model.predict_frame(test), c)}
    by_pos = {}
    pred_l = model.predict_frame(test); pred_c = _calibrate_like_live(train, test)
    for (slot, pos), g in test.groupby(["slot", "position"]):
        idx = g.index.values
        pos_idx = test.index.get_indexer(idx)
        by_pos[f"{int(slot)}:{pos}"] = {"n": int(len(g)), "actual": round(float(g.ownership_actual.mean()), 2),
                                        "sampler": round(float(g.own_sampler.mean()), 2), "calibrated": round(float(pred_c[pos_idx].mean()), 2),
                                        "learned": round(float(pred_l[pos_idx].mean()), 2),
                                        "mae_calibrated": round(float(np.abs(pred_c[pos_idx] - g.ownership_actual.values).mean()), 2),
                                        "mae_learned": round(float(np.abs(pred_l[pos_idx] - g.ownership_actual.values).mean()), 2)}
    out["by_slot_position"] = by_pos
    imp = getattr(model.model, "feature_importances_", None)
    if imp is not None:
        out["importances"] = {f: round(float(v), 3) for f, v in sorted(zip(model.features, imp), key=lambda t: -t[1])}
    return out


# --------------------------------------------------------------------------- calibration lines
CALIBRATION_FILE = "ownership_calibration.json"
POINTS_POSITIONS = ("DST", "K")       # calibrated on projected points, not projected ownership


def fit_calibration(session, cutoff: Optional[str] = None, min_n: int = 30) -> dict:
    """
    Per roster slot (511 captain, 512 flex) and position, a line from what we projected
    to what the field actually did (actual ~ a + b * projected), from the scraped showdown
    contests (SHOWDOWN_SLOTS rows only; before `cutoff` when given). The public rosters DST and K by their projected
    points rather than by value, so their line runs on fpts_proj instead (DST flex: corr
    0.89 vs 0.74). ownership.calibrate_ownership applies the result after the sampler.
    """
    q = """SELECT p.contest_id, p.position, p.roster_slot_id AS slot, p.ownership_proj, p.ownership_actual, p.fpts_proj
           FROM showdown_player p JOIN historical_contest_information h ON h.contest_id = p.contest_id
           WHERE p.ownership_proj IS NOT NULL AND p.ownership_actual IS NOT NULL AND p.roster_slot_id IN (511, 512)"""
    if cutoff:
        q += " AND h.date < :c"
    stmt = text(q)
    df = read_sql(session, stmt.bindparams(c=cutoff) if cutoff else stmt)
    out = {"fitted_on": datetime.now().strftime("%Y-%m-%d"), "cutoff": cutoff, "contests": int(df.contest_id.nunique()) if not df.empty else 0,
           "rows": int(len(df)), "maps": {}}
    for (slot, pos), g in df.groupby(["slot", "position"]):
        if len(g) < min_n:
            continue
        x = g.fpts_proj.values.astype(float) if pos in POINTS_POSITIONS else g.ownership_proj.values.astype(float)
        y = g.ownership_actual.values.astype(float)
        if x.std() <= 0:
            continue
        b, a = np.polyfit(x, y, 1)
        out["maps"][f"{int(slot)}:{pos}"] = {"on": "fpts_proj" if pos in POINTS_POSITIONS else "ownership_proj", "a": round(float(a), 4),
                                             "b": round(float(b), 4), "n": int(len(g)), "corr": round(float(np.corrcoef(x, y)[0, 1]), 3),
                                             "proj_mean": round(float(g.ownership_proj.mean()), 2), "actual_mean": round(float(y.mean()), 2)}
    return out


def save_calibration(cal: dict, model_dir: Path = MODELS_DIR) -> Path:
    import json
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / CALIBRATION_FILE
    path.write_text(json.dumps(cal, indent=1))
    return path
