# src/ivan/sports/nba/field_model.py
"""
The field model: how DraftKings entrants actually build lineups, fitted on real
contests and kept apart from our projections.

It sees only what the public sees at lock (the slate archive capture the contest
was matched to, standings.derive_contest): salary, position, DraftKings' FPPG,
status and Rotowire starters, the books' player props and the game lines, plus
plain box-score history (recent minutes, how often he starts). Never our
projections, minutes or rates.

features(...)       one row per available player of a slate
dataset()           every derived contest (data/nba/derived/contests/<id>) joined to
                    its capture: features + the real ownership (counted from the lineups)
fit(...)            ownership as a softmax inside each contest scaled to 800% (eight
                    slots): share_i = 8 * exp(theta . x_i / tau_c) / sum_j exp(theta . x_j / tau_c),
                    tau_c = exp(g . [1, log entries, log fee, log games]); fitted by the
                    multinomial likelihood of the drafted shares
predict(...)        ownership % for a slate's players under a contest's size and fee
loso(...)           leave-one-slate-out validation against simple baselines

The props-based public projection ("prop_fpts") is points + 1.25 rebounds + 1.5 assists
+ 0.5 threes + 2 (steals + blocks) at the books' median main lines; it stands in for
the projection sites the field reads. Model file: data/nba/models/field_ownership.json.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core import archive
from ivan.core.utils import normalize_name
from ivan.sports.nba.paths import MODELS_DIR, NBA_DATA_DIR

DERIVED_DIR = NBA_DATA_DIR / "derived" / "contests"
MODEL_JSON = MODELS_DIR / "field_ownership.json"
PROP_WEIGHTS = {"player_points": 1.0, "player_rebounds": 1.25, "player_assists": 1.5, "player_threes": 0.5, "player_steals": 2.0, "player_blocks": 2.0}
OUT = {"OUT", "O"}
# frozen 2026-10-23 by leave-one-slate-out on 7 slates / 29 contests (held-out ownership MAE 5.0 pts, top-20 MAE 11.6,
# rank correlation 0.84 vs 7.8 / 17.9 / 0.81 for a DK-value-only model). Fewer features did worse; more (slate-relative
# value ranks, a news-aware public projection) did not do better, so they are computed but unused. Known bias: the top
# chalk comes out ~13 pts low. Change this list only with the prospective check (check_prospective) on newer slates.
FEATURES = ["value_dk", "fppg_dk", "prop_fpts", "has_prop", "salary_k", "starter", "promoted", "questionable", "freed_min_team", "value_prop"]
CANDIDATE_FEATURES = FEATURES + ["hist_min", "team_total", "pos_c", "pos_g", "log_value_rank", "top3_value", "value_gap",
                                 "value_rank_tier", "pub_proj", "pub_value", "news_min", "no_history"]
TAU_FEATURES = ["one", "log_entries", "log_fee", "log_games"]


# ----------------------------------------------------------------------------- features
def _prop_fpts(odds: List[dict]) -> Dict[str, float]:
    if not odds:
        return {}
    df = pd.DataFrame(odds)
    df = df[df.market.isin(PROP_WEIGHTS)]
    if df.empty:
        return {}
    med = df.groupby(["player", "market"]).point.median().unstack()
    out = {}
    for name, r in med.iterrows():
        out[normalize_name(name)] = float(sum(PROP_WEIGHTS[m] * float(r[m]) for m in PROP_WEIGHTS if m in r and pd.notna(r[m])))
    return out


def _team_totals(lines: List[dict]) -> Dict[str, float]:
    """team -> implied points (median total / 2 minus half the team's median spread)."""
    if not lines:
        return {}
    df = pd.DataFrame(lines)
    out = {}
    for gid, g in df.groupby("game_id"):
        tot = g[(g.market == "totals")].point.dropna()
        sp = g[(g.market == "spreads") & (g.side == "home")].point.dropna()
        if not len(tot):
            continue
        home, away = g.home_abbr.iloc[0], g.away_abbr.iloc[0]
        t, s = float(tot.median()), float(sp.median()) if len(sp) else 0.0
        out[home] = t / 2 - s / 2; out[away] = t / 2 + s / 2
    return out


def features(players: List[dict], odds: List[dict], lines: List[dict], history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    players: the capture's player rows (id, name, team, position, salary, status, dk_avg_fpts, starter, gametime).
    history: optional rows (player_name, hist_min, start_ewm) from box-score history; missing means 0 / unknown.
    """
    df = pd.DataFrame(players).copy()
    df["key"] = df.name.map(normalize_name)
    df["status"] = df.status.fillna("").astype(str).str.upper().replace({"NONE": ""})
    hist = (history.set_index("player_name") if history is not None and len(history) else pd.DataFrame(columns=["hist_min", "start_ewm"]))
    df["hist_min"] = df.key.map(hist["hist_min"]).astype(float).fillna(0.0) if "hist_min" in hist else 0.0
    df["start_ewm"] = df.key.map(hist["start_ewm"]).astype(float).fillna(0.0) if "start_ewm" in hist else 0.0
    out_by_team = df[df.status.isin(OUT)].groupby("team").hist_min.sum()
    df = df[~df.status.isin(OUT)].copy()
    props = _prop_fpts(odds); tt = _team_totals(lines)
    df["salary_k"] = df.salary.astype(float) / 1000.0
    df["fppg_dk"] = df.dk_avg_fpts.astype(float).fillna(0.0)
    df["value_dk"] = df.fppg_dk / df.salary_k
    df["prop_fpts"] = df.key.map(props).astype(float)
    df["has_prop"] = df.prop_fpts.notna().astype(float)
    df["prop_fpts"] = df.prop_fpts.fillna(df.fppg_dk)                  # unpriced: the field falls back on FPPG
    df["value_prop"] = df.prop_fpts / df.salary_k
    st = df.starter.fillna("").astype(str).str.lower()
    df["starter"] = st.isin(["confirmed", "expected", "true", "1", "s"]).astype(float)
    df["promoted"] = ((df.starter > 0) & (df.start_ewm < 0.5)).astype(float)
    df["questionable"] = df.status.isin(["Q", "GTD", "D"]).astype(float)
    df["freed_min_team"] = df.team.map(out_by_team).fillna(0.0) / 48.0
    df["team_total"] = (df.team.map(tt).astype(float) - 114.0).fillna(0.0) / 10.0
    pos = df.position.fillna("").astype(str)
    df["pos_c"] = pos.str.contains("C").astype(float); df["pos_g"] = pos.str.contains("G").astype(float)
    # a news-aware public projection, the way projection sites build one from public data: recent minutes plus a
    # minutes-weighted share of the minutes freed on his team, times his FPPG per minute; salary-implied when he
    # has no FPPG or no minutes history (back from injury, rookies); the books' props replace it where priced
    mins = df.hist_min.clip(lower=0.0)
    rot = df[mins >= 10].groupby("team").hist_min.sum()
    freed = df.team.map(out_by_team).fillna(0.0)
    share = (mins / df.team.map(rot).replace(0, np.nan)).fillna(0.0)
    df["news_min"] = (freed * share).clip(upper=12.0)
    fppm = (df.fppg_dk / mins.where(mins >= 8)).where(df.fppg_dk > 0)
    sal_proj = 5.0 * df.salary_k
    pub = (fppm * (mins + df.news_min)).where(fppm.notna(), sal_proj)
    df["pub_proj"] = np.where(df.has_prop > 0, df.prop_fpts * (1.0 + 0.0 * df.news_min), pub)
    df["no_history"] = (fppm.isna()).astype(float)
    df["pub_value"] = df.pub_proj / df.salary_k
    df["hist_min"] = df.hist_min / 36.0
    df["news_min"] = df.news_min / 12.0
    # relative to the slate: the field piles onto whoever ranks best, not onto a fixed value number
    best = df["pub_value"]
    rank = best.rank(ascending=False, method="first")
    df["log_value_rank"] = np.log(rank)
    df["top3_value"] = (rank <= 3).astype(float)
    df["value_gap"] = best - best.max()
    df["value_rank_tier"] = df.groupby(pd.cut(df.salary_k, [0, 4.5, 6.5, 8.5, 99]), observed=False).pub_value.rank(ascending=False, method="first").map(np.log)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- the dataset
def _history_at(session: Session, draft_group_id: int, lock) -> pd.DataFrame:
    from sqlalchemy import text
    q = text("select player_name, hist_min, start_ewm from nba_projection_capture where draft_group_id = :d and source = 'ours' "
             "and captured_at <= :t order by captured_at desc")
    df = pd.read_sql(q.bindparams(d=int(draft_group_id), t=str(lock)), session.get_bind())
    return df.drop_duplicates("player_name")


def lineup_ownership(contest_dir: Path) -> pd.Series:
    """player id -> % of the contest's entries holding him, counted from the real lineups (copies included).
    DraftKings' %Drafted column is not this: on 2026-10-21 it read 30.9 for a player in 48.5% of the
    95,066 lineups, and its column sums to ~465 rather than 800, so it is never used as the target."""
    L = pd.read_parquet(Path(contest_dir) / "lineups.parquet")
    ids = L[[f"p{i}" for i in range(1, 9)]].to_numpy(); w = np.repeat(L.copies.to_numpy()[:, None], 8, axis=1)
    return pd.Series(w.ravel()).groupby(ids.ravel()).sum() / float(L.copies.sum()) * 100.0


def dataset(session: Session, derived_dir: Path = DERIVED_DIR) -> pd.DataFrame:
    """One row per (contest, available player): FEATURES, own (real %Drafted), contest size / fee / games, slate key."""
    from ivan.sports.nba.models import HistoricalContest
    frames = []
    for d in sorted(Path(derived_dir).iterdir()):
        fj, pc = d / "field.json", d / "players.csv"
        if not fj.exists() or not pc.exists():
            continue
        meta = json.loads(fj.read_text())
        cap = meta.get("capture_id")
        if cap is None:
            continue
        blobs = archive.load_blobs(session, int(cap))
        hc = session.get(HistoricalContest, int(meta["contest_id"]))
        lock = min((p.get("gametime") for p in blobs.get("players", []) if p.get("gametime")), default=None)
        f = features(blobs.get("players", []), blobs.get("odds", []), blobs.get("game_lines", []),
                     _history_at(session, int(meta["draft_group_id"]), meta.get("capture_at") or lock))
        f["own"] = f.id.astype(int).map(lineup_ownership(d)).astype(float).fillna(0.0)
        n_games = len({(p.get("team"), p.get("opponent")) for p in blobs.get("players", [])}) // 2 or 1
        f["contest_id"] = int(meta["contest_id"]); f["slate"] = int(meta["draft_group_id"])
        f["entries"] = int(meta.get("entries") or (hc.total_entries if hc else 0)); f["fee"] = float(meta.get("entry_fee") or (hc.entry_fee if hc else 1.0))
        f["games"] = int(n_games); f["max_entries"] = int(hc.multi_entry) if hc else 150
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------------------------------------------------------- the model
def _tau_x(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), np.log(df.entries.clip(lower=100) / 10000.0), np.log(df.fee.clip(lower=0.1)), np.log(df.games.clip(lower=1) / 6.0)])


def _shares(theta: np.ndarray, gamma: np.ndarray, X: np.ndarray, T: np.ndarray, groups: np.ndarray) -> np.ndarray:
    z = (X @ theta) / np.exp(T @ gamma)
    out = np.empty_like(z)
    for g in np.unique(groups):
        m = groups == g
        e = np.exp(z[m] - z[m].max())
        out[m] = 8.0 * e / e.sum()
    return out


def fit(df: pd.DataFrame, feats: List[str] = FEATURES, l2: float = 0.002) -> dict:
    """theta (per feature) and gamma (temperature by contest) by the drafted-share likelihood, with a little L2 on theta."""
    from scipy.optimize import minimize
    X = df[feats].to_numpy(float); T = _tau_x(df); y = (df.own.to_numpy(float) / 100.0); groups = df.contest_id.to_numpy()
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    k = len(feats)

    def nll(w):
        th, ga = w[:k], np.r_[0.0, w[k:]]                      # the temperature's intercept is fixed: theta carries the scale
        p = _shares(th, ga, Xs, T, groups)
        return -float(np.sum(y * np.log(np.maximum(p, 1e-12)))) / len(np.unique(groups)) + l2 * float(th @ th)
    w0 = np.r_[np.zeros(k), np.zeros(T.shape[1] - 1)]
    w0[feats.index("value_prop") if "value_prop" in feats else 0] = 1.0
    res = minimize(nll, w0, method="L-BFGS-B")
    return dict(features=list(feats), theta=res.x[:k].tolist(), gamma=[0.0] + res.x[k:].tolist(), mu=mu.tolist(), sd=sd.tolist(),
                l2=l2, nll=float(res.fun), n_contests=int(len(np.unique(groups))), n_slates=int(df.slate.nunique()), n_rows=int(len(df)),
                slates=sorted(int(x) for x in df.slate.unique()),
                fitted_at=datetime.now().isoformat(timespec="seconds"))


def predict(model: dict, df: pd.DataFrame, entries: int, fee: float, games: int) -> np.ndarray:
    """Ownership % (summing to 800) for one slate's feature rows under one contest's size, fee and game count."""
    feats = model["features"]
    X = (df[feats].to_numpy(float) - np.array(model["mu"])) / np.array(model["sd"])
    d = df.assign(entries=entries, fee=fee, games=games)
    return 100.0 * _shares(np.array(model["theta"]), np.array(model["gamma"]), X, _tau_x(d), np.zeros(len(df), dtype=int))


def save(model: dict, path: Path = MODEL_JSON) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(model, indent=1)); return path


def load(path: Path = MODEL_JSON) -> Optional[dict]:
    return json.loads(path.read_text()) if Path(path).exists() else None


# ----------------------------------------------------------------------------- validation
def metrics(own: np.ndarray, pred: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    top = np.argsort(-own)[:10]
    return dict(mae=float(np.mean(np.abs(own - pred))), mae_top20=float(np.mean(np.abs(own[np.argsort(-own)[:20]] - pred[np.argsort(-own)[:20]]))),
                spearman=float(spearmanr(own, pred).correlation), top10_overlap=int(len(set(top) & set(np.argsort(-pred)[:10]))),
                max_own_err=float(pred.max() - own.max()))


def loso(df: pd.DataFrame, variants: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """Leave one slate out: fit on the other slates, score every contest of the held-out slate. One row per (variant, contest)."""
    variants = variants or {"frozen": FEATURES, "dk_value": ["value_dk", "salary_k"], "salary_only": ["salary_k"]}
    rows = []
    for name, feats in variants.items():
        for slate in sorted(df.slate.unique()):
            tr, te = df[df.slate != slate], df[df.slate == slate]
            m = fit(tr, feats)
            for cid, g in te.groupby("contest_id"):
                pred = predict(m, g, int(g.entries.iloc[0]), float(g.fee.iloc[0]), int(g.games.iloc[0]))
                rows.append(dict(variant=name, slate=int(slate), contest_id=int(cid), entries=int(g.entries.iloc[0]), **metrics(g.own.to_numpy(float), pred)))
    return pd.DataFrame(rows)


def check_prospective(session: Session, model: Optional[dict] = None, derived_dir: Path = DERIVED_DIR) -> pd.DataFrame:
    """The saved model scored on contests whose slate it was not fitted on (derived after it was fitted): the honest test."""
    model = model or load()
    if model is None:
        raise ValueError(f"no field model at {MODEL_JSON}; fit one first")
    df = dataset(session, derived_dir)
    seen = set(model.get("slates", []))
    new = df[~df.slate.isin(seen)]
    rows = []
    for cid, g in new.groupby("contest_id"):
        pred = predict(model, g, int(g.entries.iloc[0]), float(g.fee.iloc[0]), int(g.games.iloc[0]))
        rows.append(dict(slate=int(g.slate.iloc[0]), contest_id=int(cid), entries=int(g.entries.iloc[0]), **metrics(g.own.to_numpy(float), pred)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- the field of lineups
# NBA fields are convergent: most entrants run an optimizer on one of a few providers' projections and take (close
# to) the optimal lineup; the rest build loosely. One entry:
#   jammer (share alpha):  provider k ~ uniform(K); score_i = P_i * (1 + d_ki + sigma_u * e_i), d_k ~ N(0, sigma_p)
#   casual (1 - alpha):    score_i = P_i * (1 + SIGMA_CASUAL * e_i)
# The knobs come from the slate and the contest (knobs()): slate dynamics (games, chalk concentration) set sigma_p
# and sigma_u, the contest (max entries, fee, size) sets alpha. They are calibrated to real fields' shape only.
N_PROVIDERS = 3
SIGMA_CASUAL = 0.45


def chalk_concentration(own_pct: np.ndarray) -> float:
    """How concentrated the public's chalk is: the top-8 share of predicted ownership (0.125 = flat .. 1 = one lineup)."""
    o = np.sort(np.asarray(own_pct, dtype=float))[::-1]
    return float(o[:8].sum() / max(o.sum(), 1e-9))


def slate_features(games: int, chalk: float) -> dict:
    return dict(log_games=float(np.log(max(games, 1) / 4.0)), chalk=float(chalk - 0.35))


def contest_features(entries: int, fee: float, max_entries: int) -> dict:
    return dict(log_entries=float(np.log(max(entries, 100) / 10000.0)), log_fee=float(np.log(max(fee, 0.1))),
                multi=float(max_entries >= 100))


def knobs(params: dict, slate: dict, contest: dict) -> dict:
    """sigma_p, sigma_u, alpha for one contest from the fitted link (params from calibrate_field_shape)."""
    sp = np.exp(params["sp0"] + params["sp_games"] * slate["log_games"] + params["sp_chalk"] * slate["chalk"])
    su = np.exp(params["su0"] + params["su_games"] * slate["log_games"])
    z = params["a0"] + params["a_multi"] * contest["multi"] + params["a_entries"] * contest["log_entries"] + params["a_fee"] * contest["log_fee"]
    return dict(sigma_p=float(sp), sigma_u=float(su), alpha=float(1.0 / (1.0 + np.exp(-z))))


def _field_chunk(args):
    from ivan.sports.nba.pre_lock.lineup_generation import generate_single_classic_lineup
    players, P, prov, n, sigma_u, alpha, seed = args
    rng = np.random.default_rng(seed)
    ids = [p.id for p in players]; base = np.array([P.get(i, 0.0) for i in ids])
    out = []
    for _ in range(n):
        if rng.random() < alpha:
            sc = base * (1.0 + prov[rng.integers(0, prov.shape[0])] + sigma_u * rng.standard_normal(len(ids)))
        else:
            sc = base * (1.0 + SIGMA_CASUAL * rng.standard_normal(len(ids)))
        lu = generate_single_classic_lineup(players, base_scores=dict(zip(ids, sc.tolist())), max_salary_per_team=50000)
        if lu:
            out.append(tuple(sorted(lu)))
    return out


def simulate_field(players: list, P: Dict[int, float], n: int, sigma_p: float, sigma_u: float, alpha: float,
                   seed: int = 0, workers: int = 8, providers: int = N_PROVIDERS) -> List[tuple]:
    """n field lineups (sorted id tuples, copies kept) under the convergent-field model; players are FieldPlayers."""
    from concurrent.futures import ProcessPoolExecutor
    rng = np.random.default_rng(seed)
    prov = sigma_p * rng.standard_normal((providers, len(players)))
    w = max(1, min(workers, n)); per = [n // w + (1 if i < n % w else 0) for i in range(w)]
    with ProcessPoolExecutor(max_workers=w) as exe:
        parts = list(exe.map(_field_chunk, [(players, P, prov, k, sigma_u, alpha, seed * 1000 + i) for i, k in enumerate(per) if k]))
    return [lu for part in parts for lu in part]


def field_shape(lineups: List[tuple], Q: Dict[int, float], q_opt: float, subsample: Optional[int] = None, seed: int = 0) -> dict:
    """A field's shape under projection Q: strength quantiles relative to Q's optimal, within 5 / 10 %, duplicate share."""
    from collections import Counter
    L = list(lineups)
    if subsample and len(L) > subsample:
        rng = np.random.default_rng(seed); L = [L[i] for i in rng.choice(len(L), size=subsample, replace=False)]
    r = np.nan_to_num(np.array([sum(Q.get(i, 0.0) for i in lu) for lu in L]) / q_opt)
    c = Counter(L)
    return dict(p10=float(np.quantile(r, 0.1)), p50=float(np.quantile(r, 0.5)), p90=float(np.quantile(r, 0.9)),
                within5=float((r >= 0.95).mean()), within10=float((r >= 0.90).mean()),
                dup_share=float(sum(v for v in c.values() if v > 1) / max(1, len(L))))
