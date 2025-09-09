# src/ivan/sports/nfl/correlated_sampling_validation.py
from __future__ import annotations
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.sports.nfl.correlated_samples import generate_correlated_samples

def _spearman_from_uniform_df(U_df: pd.DataFrame) -> np.ndarray:
    V = U_df.to_numpy(dtype=float)
    ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 0, V).astype(float)
    return np.corrcoef(ranks, rowvar=False)

def print_global_vine_spearman(model_dir: str | Path, n: int = 10000) -> None:
    """Load the single global positional vine and print its Spearman correlation matrix."""
    pkg = joblib.load(Path(model_dir) / "positional_global_vine.pkl")
    cols = list(getattr(pkg, "cols", []))
    vine = getattr(pkg, "vine", None)
    if not cols or vine is None:
        print("[validation] global vine not found or empty")
        return

    U_df = vine.sample(n)
    U_df = U_df[cols].clip(lower=1e-6, upper=1-1e-6)

    R = _spearman_from_uniform_df(U_df)
    with np.printoptions(precision=3, suppress=True):
        print("[validation] Global vine columns:", cols)
        print("[validation] Spearman (vine sample):")
        print(R)

def print_historical_slot_spearman(session: Session) -> None:
    """Compute raw Spearman across the historical positional table for reference."""
    with session.bind.connect() as conn:
        df = pd.read_sql(text("""            SELECT QB,RB1,RB2,WR1,WR2,WR3,TE1,TE2,
                   OPPQB,OPPRB1,OPPRB2,OPPWR1,OPPWR2,OPPWR3,OPPTE1,OPPTE2
            FROM nfl_team_week_positional_fpts
        """), conn)

    cols = [c for c in df.columns if c in (
        "QB","RB1","RB2","WR1","WR2","WR3","TE1","TE2",
        "OPPQB","OPPRB1","OPPRB2","OPPWR1","OPPWR2","OPPWR3","OPPTE1","OPPTE2"
    )]

    X = df[cols].copy()
    X = X.dropna(how="all")
    X = X.fillna(0.0)

    R = X.corr(method="spearman")
    with pd.option_context("display.precision", 3):
        print("[validation] Historical slot Spearman (raw table):")
        print(R)

def run_end_to_end_and_validate(session: Session, model_dir: str | Path, n_samples: int = 2000) -> Dict[int, np.ndarray]:
    """Convenience to run sampling and then print vine & historical Spearman for sanity."""
    samples = generate_correlated_samples(session, model_dir=model_dir, overrides={"n_samples": n_samples, "print_sanity": True})
    print_global_vine_spearman(model_dir, n=max(10000, n_samples))
    print_historical_slot_spearman(session)
    return samples
