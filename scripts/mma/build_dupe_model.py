#!/usr/bin/env python3
"""
scripts/mma/build_dupe_model.py

Fit the MMA duplication model on real fields (data/mma/derived/contests/*/
lineups.parquet, from get_contest_data.py + standings.derive_contest):

    E[copies] = field_size x exp(a + b_left x min(salary left, 2000)/100 + b_zero x [salary left = 0]
                                   + b_win x summed win probability + b_same x [two fighters from one fight])

a Poisson log-linear model, so the field sampler can use exp(.) as each
candidate's relative weight (the field size cancels). Writes
data/mma/models/dupe_model.json with the coefficients, the fit's deviance
explained, calibration by salary-left bin and the same-fight share.

    scripts/mma/build_dupe_model.py                 # every derived contest
    scripts/mma/build_dupe_model.py --contest-ids 196337922,196426859
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ivan.sports.mma.paths import MMA_DATA_DIR
from ivan.sports.mma.standings import DERIVED_DIR

MODEL_PATH = MMA_DATA_DIR / "models" / "dupe_model.json"
LEFT_CAP = 2000.0


def features(df: pd.DataFrame) -> np.ndarray:
    left = np.minimum(df.salary_left.to_numpy(dtype=float), LEFT_CAP) / 100.0
    return np.column_stack([np.ones(len(df)), left, (df.salary_left.to_numpy() <= 0).astype(float),
                            df.sum_win.to_numpy(dtype=float), df.same_fight.to_numpy().astype(float)])


FEATURE_NAMES = ["intercept", "b_left_per_100", "b_zero_left", "b_win", "b_same_fight"]


def fit_poisson(X: np.ndarray, y: np.ndarray, offset: np.ndarray, iters: int = 50) -> np.ndarray:
    """Poisson regression with log link by iteratively reweighted least squares."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-9)) - offset.mean()
    for _ in range(iters):
        eta = X @ beta + offset
        mu = np.exp(np.clip(eta, -30, 30))
        z = eta - offset + (y - mu) / mu
        W = mu
        XtW = X.T * W
        new = np.linalg.solve(XtW @ X + 1e-9 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def deviance(y: np.ndarray, mu: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(t - (y - mu)))


UNIVERSE_FEATURES = ["intercept", "b_left_per_100", "b_zero_left", "b_win", "b_log_rank", "b_main_event"]


def universe_features(df: pd.DataFrame) -> np.ndarray:
    left = np.minimum(df.salary_left.to_numpy(dtype=float), LEFT_CAP) / 100.0
    return np.column_stack([np.ones(len(df)), left, (df.salary_left.to_numpy() <= 0).astype(float),
                            df.sum_win.to_numpy(dtype=float), np.log(df.rank_win.to_numpy(dtype=float)), df.n_main.to_numpy(dtype=float)])


def fit_alpha(y: np.ndarray, mu: np.ndarray, min_mu: float = 5.0) -> float:
    """
    Gamma overdispersion of copies around the model: among lineups the model
    expects at least min_mu copies for, var(y) = mu + mu^2 / alpha (negative
    binomial). Small alpha = lumpy top (one of several near-identical builds
    becomes the chalk); large alpha = Poisson.
    """
    m = mu >= min_mu
    if m.sum() < 10:
        return 1e9
    extra = float(np.mean((y[m] - mu[m]) ** 2 - mu[m]))
    if extra <= 0:
        return 1e9
    return float(np.mean(mu[m] ** 2) / extra)


def main_universe(a) -> int:
    """Fit on every valid lineup of the slate (zeros included), with the lineup's rank by summed win probability."""
    from ivan.sports.mma.ingest import load_players_dk
    from ivan.sports.mma.standings import win_probs_from_odds_csv
    from ivan.sports.mma.universe import enumerate_lineups
    ids = [int(x) for x in a.contest_ids.split(",")] if a.contest_ids else sorted(int(p.name) for p in DERIVED_DIR.iterdir() if p.name.isdigit())
    pool = load_players_dk(int(a.draft_group))
    wp = win_probs_from_odds_csv(Path(a.player_odds), pool)
    # the main event: the latest fight on the card (ingest's default for the 5-round fight)
    last = max(r["gametime"] for r in pool if r.get("gametime"))
    main = {int(r["id"]) for r in pool if r.get("gametime") == last} if a.main_event is None else {int(x) for x in a.main_event.split(",")}
    U = enumerate_lineups(pool, wp, main_event=main)
    print(f"main-event fighters: {sorted(r['name'] for r in pool if int(r['id']) in main)}")
    frames, outside = [], 0
    for cid in ids:
        p = DERIVED_DIR / str(cid) / "lineups.parquet"
        if not p.exists():
            continue
        g = pd.read_parquet(p)
        g["key"] = g[[f"f{j}" for j in range(1, 7)]].apply(lambda r: tuple(sorted(int(x) for x in r)), axis=1)
        m = U.merge(g[["key", "copies"]], on="key", how="left").fillna({"copies": 0.0})
        outside += len(g) - int((m.copies > 0).sum())
        m["contest_id"] = cid
        m["field_size"] = int(g.copies.sum())
        frames.append(m)
    if not frames:
        print("no derived contests for these ids")
        return 1
    df = pd.concat(frames, ignore_index=True)
    X, y = universe_features(df), df.copies.to_numpy(dtype=float)
    offset = np.log(df.field_size.to_numpy(dtype=float))
    beta = fit_poisson(X, y, offset)
    mu = np.exp(X @ beta + offset)
    null = np.exp(offset) * (y.sum() / np.exp(offset).sum())
    alpha = fit_alpha(y, mu)
    rho = float(pd.Series(y).corr(pd.Series(mu), method="spearman"))
    top = df.assign(mu=mu).sort_values("copies", ascending=False).head(10)
    big = df.contest_id == df.loc[df.field_size.idxmax(), "contest_id"]
    N = int(df.loc[big, "field_size"].iloc[0])
    model = {"version": 2, "features": "universe", "built_at": datetime.now().isoformat(timespec="seconds"), "contests": ids,
             "draft_group": int(a.draft_group), "entries": int(y.sum()), "universe_lineups": int(len(U)), "left_cap": LEFT_CAP,
             "coef": dict(zip(UNIVERSE_FEATURES, map(float, beta))), "alpha": alpha,
             "fit": {"deviance_explained": 1 - deviance(y, mu) / deviance(y, null), "spearman": rho,
                     "top10_real_copies": top.copies.astype(int).tolist(), "top10_predicted": np.round(top.mu, 1).tolist(),
                     "biggest_field": {"size": N, "real_max": int(df.loc[big, "copies"].max()), "predicted_max": float(mu[big.to_numpy()].max()),
                                       "real_top100_share": float(df.loc[big, "copies"].nlargest(100).sum() / N),
                                       "predicted_top100_share": float(np.sort(mu[big.to_numpy()])[-100:].sum() / N)},
                     "real_lineups_outside_universe": outside},
             "same_fight_share": float(outside / max(1, len(df[df.copies > 0]) + outside))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(model, indent=1))
    print(f"universe {len(U):,} lineups x {len(ids)} contests, {int(y.sum()):,} entries; {outside} real lineups outside the universe (same fight)")
    print("coefficients:", {k: round(v, 4) for k, v in model["coef"].items()}, f"alpha {alpha:.2f}")
    print(f"deviance explained {model['fit']['deviance_explained']:.3f}; spearman {rho:.3f}")
    print(f"10 most-copied real lineups: predicted {np.round(top.mu).astype(int).tolist()} vs actual {top.copies.astype(int).tolist()}")
    b = model["fit"]["biggest_field"]
    print(f"biggest field ({N:,}): real max {b['real_max']} vs predicted max {b['predicted_max']:.0f}; top-100 share real {100*b['real_top100_share']:.1f}% vs predicted {100*b['predicted_top100_share']:.1f}%")
    print(f"wrote {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contest-ids", default=None)
    ap.add_argument("--out", default=str(MODEL_PATH))
    ap.add_argument("--universe", action="store_true", help="fit on every valid lineup of the slate (needs --draft-group and --player-odds)")
    ap.add_argument("--draft-group", default=None, help="DK draft group of the contests (universe mode)")
    ap.add_argument("--player-odds", default=None, help="player_odds csv with the Win column for that card (universe mode)")
    ap.add_argument("--main-event", default=None, help="universe mode: comma-separated fighter ids of the 5-round fight (default: the card's last fight)")
    a = ap.parse_args()
    if a.universe:
        if not (a.draft_group and a.player_odds):
            ap.error("--universe needs --draft-group and --player-odds")
        return main_universe(a)
    ids = [int(x) for x in a.contest_ids.split(",")] if a.contest_ids else sorted(int(p.name) for p in DERIVED_DIR.iterdir() if p.name.isdigit())
    frames = []
    for cid in ids:
        p = DERIVED_DIR / str(cid) / "lineups.parquet"
        if p.exists():
            g = pd.read_parquet(p)
            g["contest_id"] = cid
            g["field_size"] = int(g.copies.sum())
            frames.append(g)
    if not frames:
        print("no derived contests; run get_contest_data.py and derive them first")
        return 1
    df = pd.concat(frames, ignore_index=True)
    X, y = features(df), df.copies.to_numpy(dtype=float)
    offset = np.log(df.field_size.to_numpy(dtype=float))
    beta = fit_poisson(X, y, offset)
    mu = np.exp(X @ beta + offset)
    null = np.exp(offset) * (y.sum() / np.exp(offset).sum())
    d_model, d_null = deviance(y, mu), deviance(y, null)
    # what each feature family adds (drop it, refit)
    contrib = {}
    for name, cols in (("salary_left", [1, 2]), ("win", [3]), ("same_fight", [4])):
        keep = [j for j in range(X.shape[1]) if j not in cols]
        b2 = fit_poisson(X[:, keep], y, offset)
        contrib[name] = 1 - deviance(y, np.exp(X[:, keep] @ b2 + offset)) / d_null
    bins = pd.cut(df.salary_left, [-1, 0, 100, 200, 300, 500, 700, 1000, 1500, 2000, 10**6])
    cal = (pd.DataFrame({"bin": bins.astype(str), "actual": y, "predicted": mu}).groupby("bin", observed=True)
           .agg(lineups=("actual", "size"), actual=("actual", "mean"), predicted=("predicted", "mean")).reset_index())
    same_share = float(df.loc[df.same_fight, "copies"].sum() / df.copies.sum())
    rho = float(pd.Series(y).corr(pd.Series(mu), method="spearman"))
    model = {"version": 1, "built_at": datetime.now().isoformat(timespec="seconds"), "contests": ids,
             "entries": int(y.sum()), "unique_lineups": int(len(df)), "left_cap": LEFT_CAP,
             "coef": dict(zip(FEATURE_NAMES, map(float, beta))),
             "fit": {"deviance_explained": 1 - d_model / d_null, "spearman": rho, "explained_by": contrib},
             "same_fight_share": same_share, "calibration_by_salary_left": cal.round(3).to_dict("records")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(model, indent=1))
    print(f"{len(ids)} contests, {int(y.sum()):,} entries, {len(df):,} unique lineups")
    print("coefficients:", {k: round(v, 4) for k, v in model["coef"].items()})
    print(f"deviance explained {model['fit']['deviance_explained']:.3f}; spearman(predicted, actual copies) {rho:.3f}")
    print("explained by (full minus without):", {k: round(1 - d_model / d_null - v, 3) for k, v in contrib.items()})
    print(f"same-fight share of entries {100 * same_share:.2f}%")
    print(cal.round(2).to_string(index=False))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
