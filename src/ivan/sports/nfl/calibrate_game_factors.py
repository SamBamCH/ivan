from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd, pyvinecopulib as pv, joblib
from sqlalchemy.orm import Session
from sqlalchemy import text
import scipy.stats as st
from scipy.optimize import minimize

from ivan.database.session import init_db, get_session

TEAM10 = ["QB","RB1","RB2","WR1","WR2","WR3","TE1","TE2","K","DST"]

def _read_sql(session: Session, sql: str) -> pd.DataFrame:
    with session.bind.connect() as c:
        return pd.read_sql(text(sql), c)

def _target_cross_spearman(session: Session) -> pd.DataFrame:
    df = _read_sql(session, """
        SELECT QB,RB1,RB2,WR1,WR2,WR3,TE1,TE2,K,DST,
               OPPQB,OPPRB1,OPPRB2,OPPWR1,OPPWR2,OPPWR3,OPPTE1,OPPTE2,OPPK,OPPDST
        FROM nfl_team_week_positional_fpts
    """).replace([np.inf,-np.inf], np.nan).fillna(0.0)

    # Ensure all expected columns exist (fill with zeros if missing)
    for c in TEAM10:
        if c not in df.columns:
            df[c] = 0.0
        oppc = f"OPP{c}"
        if oppc not in df.columns:
            df[oppc] = 0.0

    A = df[TEAM10]
    B = df[[f"OPP{c}" for c in TEAM10]]

    S = np.empty((len(TEAM10), len(TEAM10)), dtype=float)

    def safe_spearman(x: pd.Series, y: pd.Series) -> float:
        # Spearman undefined if one side is constant; treat as 0
        if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
            return 0.0
        val = float(x.corr(y, method="spearman"))
        if not np.isfinite(val):  # guard NaN/inf
            return 0.0
        return val

    for i, ai in enumerate(TEAM10):
        for j, bj in enumerate(TEAM10):
            S[i, j] = safe_spearman(A[ai], B[f"OPP{bj}"])

    return pd.DataFrame(S, index=TEAM10, columns=TEAM10)

def _load_team10(model_dir: Path):
    obj = joblib.load(model_dir / "team10_vine.pkl")
    return obj.vine, list(obj.cols)

def _apply_two_factors(UA, UB, lam_off, lam_def, rng):
    zA = st.norm.ppf(UA); zB = st.norm.ppf(UB)
    n, d = zA.shape
    G_off = rng.standard_normal(size=(n,1))
    G_def = rng.standard_normal(size=(n,1))
    lam_off = lam_off.reshape(1,d); lam_def = lam_def.reshape(1,d)
    scale = np.sqrt(np.clip(1.0 - lam_off**2 - lam_def**2, 1e-4, 1.0))
    zA = lam_off*G_off + lam_def*G_def + scale*zA
    zB = lam_off*G_off + lam_def*G_def + scale*zB
    return st.norm.cdf(zA), st.norm.cdf(zB)

def _simulate_cross_spearman(vine, lam_off, lam_def, n, seed=23):
    rng = np.random.default_rng(seed)
    UA = vine.simulate(n); UB = vine.simulate(n)
    UA, UB = _apply_two_factors(UA, UB, lam_off, lam_def, rng)
    S = np.empty((len(TEAM10), len(TEAM10)), dtype=float)
    for i in range(len(TEAM10)):
        for j in range(len(TEAM10)):
            S[i, j] = pd.Series(UA[:, i]).corr(pd.Series(UB[:, j]), method="spearman")
    return S

def calibrate_game_factors(model_dir: str | Path, sims: int = 40000):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    engine = init_db("nfl")
    session = get_session(engine)
    target = _target_cross_spearman(session).values
    vine, order = _load_team10(model_dir)
    assert order == TEAM10, f"Unexpected order: {order}"

    lam_off0 = np.array([0.35,0.25,0.18,0.30,0.28,0.20,0.22,0.18,0.25,0.10], dtype=float)
    lam_def0 = np.array([0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.05,0.35], dtype=float)
    bounds   = [(0.0, 0.70)]*10 + [(-0.70, 0.70)]*10
    x0 = np.concatenate([lam_off0, lam_def0])

    def loss(x):
        lam_off = x[:10]; lam_def = x[10:]
        if np.any(lam_off**2 + lam_def**2 > 0.95):  # feasibility
            return 1e3 + np.maximum(0, lam_off**2 + lam_def**2 - 0.95).sum()*100
        S = _simulate_cross_spearman(vine, lam_off, lam_def, n=sims)
        w = np.ones_like(S); w[9,:] *= 1.5; w[:,9] *= 1.5  # emphasize DST row/col
        return float(np.mean((w*(S - target))**2))

    res = minimize(loss, x0, method="Powell", options={"maxiter": 120, "xtol":1e-3, "ftol":1e-4})
    out = {
        "lam_off": res.x[:10].tolist(),
        "lam_def": res.x[10:].tolist(),
        "order": TEAM10,
        "loss": float(res.fun),
    }
    (model_dir / "game_factors.json").write_text(json.dumps(out, indent=2))
    session.close()
    return out

if __name__ == "__main__":
    calibrate_game_factors("data/nfl/models")
