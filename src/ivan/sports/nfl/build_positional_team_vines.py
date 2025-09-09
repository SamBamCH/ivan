from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from copulas.multivariate import VineCopula

from ivan.database.session import init_db, get_session

# Fixed slot order (global). K/DST exist but will be dropped if all-null.
SLOTS = [
    "QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "TE2", "K", "DST",
    "OPPQB", "OPPRB1", "OPPRB2", "OPPWR1", "OPPWR2", "OPPWR3", "OPPTE1", "OPPTE2", "OPPK", "OPPDST",
]

@dataclass
class GlobalPositionalVine:
    cols: List[str]        # columns actually fit (subset of SLOTS if some are all-null)
    vine: VineCopula
    n_obs: int

def _read_sql(session: Session, sql: str) -> pd.DataFrame:
    with session.bind.connect() as conn:
        return pd.read_sql(text(sql), conn)

def _zero_aware_uniform(col: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """
    Map a numeric series to U(0,1) while treating structural zeros specially.
    - All zeros get random U in [0, p0), where p0 = zero proportion.
    - Non-zeros get p0 + (1-p0) * rank_pct_among_nonzeros.
    This avoids massive identical ranks across many columns (which inflate correlations).
    """
    x = pd.to_numeric(col, errors="coerce")
    mask0 = (x == 0) | (~np.isfinite(x))
    p0 = float(mask0.mean())
    u = np.empty(len(x), dtype=float)
    if p0 > 0.0:
        n0 = int(mask0.sum())
        u[mask0.to_numpy()] = rng.random(n0) * max(p0, 1e-6)
        nz = x[~mask0]
        if len(nz) > 0:
            ranks = nz.rank(pct=True, method="average").to_numpy()
            u[~mask0.to_numpy()] = p0 + (1.0 - p0) * ranks
        else:
            u[~mask0.to_numpy()] = rng.random((~mask0).sum())
    else:
        u = x.rank(pct=True, method="average").to_numpy()
    return np.clip(u, 1e-6, 1 - 1e-6)

def build_positional_team_vines(model_dir: Path | str, db_url: str | None = None) -> GlobalPositionalVine:
    """
    Build a single global positional vine using all rows from nfl_team_week_positional_fpts.

    Saves to: {model_dir}/positional_global_vine.pkl
    Returns: GlobalPositionalVine
    """
    model_dir = Path(model_dir)
    out_path   = model_dir / "positional_global_vine.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engine  = init_db("nfl", db_url)
    session = get_session(engine)

    sql = (
        "SELECT season, season_type, week, team, opponent, "
        "QB,RB1,RB2,WR1,WR2,WR3,TE1,TE2,K,DST, "
        "OPPQB,OPPRB1,OPPRB2,OPPWR1,OPPWR2,OPPWR3,OPPTE1,OPPTE2,OPPK,OPPDST "
        "FROM nfl_team_week_positional_fpts"
    )
    df = _read_sql(session, sql)

    # 1) choose globally usable columns (drop all-null and low coverage)
    coverage = df[SLOTS].notna().mean()
    keep_cols = [c for c in SLOTS if (c in df.columns) and (not df[c].isna().all()) and (coverage.get(c, 0.0) >= 0.50)]

    mat = df[keep_cols].astype(float)

    # 2) zero-aware rank -> uniforms, keep as DataFrame with names
    rng = np.random.default_rng(42)
    u_map = {c: _zero_aware_uniform(mat[c], rng) for c in keep_cols}
    U_df = pd.DataFrame(u_map, columns=keep_cols)

    # 3) drop rows with any NaN across kept columns (should be rare now)
    U_df = U_df.dropna(axis=0, how="any")

    # 4) drop near-constant columns
    stds = U_df.std(axis=0)
    keep_cols2 = [c for c, s in stds.items() if s > 1e-9]
    U_df = U_df[keep_cols2]

    # 5) tiny jitter to break ties (preserve DataFrame with column names)
    U_df = U_df.clip(lower=1e-6, upper=1-1e-6)
    U_df = U_df + rng.normal(0.0, 1e-7, size=U_df.shape)

    # 6) fit single vine on a DataFrame (copulas expects .columns)
    vine = VineCopula("center")
    vine.fit(U_df)

    obj = GlobalPositionalVine(cols=keep_cols2, vine=vine, n_obs=len(U_df))
    joblib.dump(obj, out_path)

    session.close()
    return obj
