from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List
import numpy as np, pandas as pd, joblib
from sqlalchemy.orm import Session
from sqlalchemy import text
import pyvinecopulib as pv

from ivan.database.session import init_db, get_session

TEAM10 = ["QB","RB1","RB2","WR1","WR2","WR3","TE1","TE2","K","DST"]

@dataclass
class Team10Vine:
    cols: List[str]
    vine: pv.Vinecop
    n_obs: int

def _read_sql(session: Session, sql: str) -> pd.DataFrame:
    with session.bind.connect() as c:
        return pd.read_sql(text(sql), c)

def _zero_aware_u(col: pd.Series) -> np.ndarray:
    x = pd.to_numeric(col, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
    mask0 = (x == 0)
    n = len(x)
    r = np.empty(n, dtype=float)
    if mask0.any():
        p0 = mask0.mean()
        rng = np.random.default_rng(7)
        r[mask0.to_numpy()] = rng.uniform(1e-6, max(1e-6, p0-1e-6), mask0.sum())
        nz = x[~mask0]
        rnz = nz.rank(pct=True, method="average").to_numpy() if len(nz) else np.full((~mask0).sum(), 0.5)
        r[~mask0.to_numpy()] = p0 + (1.0 - p0) * rnz
    else:
        r = x.rank(pct=True, method="average").to_numpy()
    return np.clip(r, 1e-6, 1-1e-6)

def build_team10_vine(model_dir: str | Path, db_url: str | None = None):
    model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "team10_vine.pkl"

    engine = init_db("nfl", db_url)
    session = get_session(engine)
    df = _read_sql(session, """
        SELECT QB,RB1,RB2,WR1,WR2,WR3,TE1,TE2,K,DST
        FROM nfl_team_week_positional_fpts
    """).replace([np.inf,-np.inf], np.nan)

    df = df.dropna(how="all")
    keep = [c for c in TEAM10 if c in df.columns and not df[c].isna().all()]
    U = np.column_stack([_zero_aware_u(df[c]) for c in keep])
    U = np.ascontiguousarray(U, dtype=float)

    controls = pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.indep,
            pv.BicopFamily.gaussian, pv.BicopFamily.student,
            pv.BicopFamily.clayton,  pv.BicopFamily.gumbel,
            pv.BicopFamily.frank,    pv.BicopFamily.joe,
            pv.BicopFamily.bb1,      pv.BicopFamily.bb6,
            pv.BicopFamily.bb7,      pv.BicopFamily.bb8,
            # If your version errors on tawn, just comment the next line:
            # pv.BicopFamily.tawn
        ],
        num_threads=0,
        select_trunc_lvl=False
    )
    vine = pv.Vinecop(d=U.shape[1])
    vine.select(U, controls)

    obj = Team10Vine(cols=keep, vine=vine, n_obs=len(U))
    joblib.dump(obj, out_path)
    session.close()
    return obj
