"""
NFL Classic field distribution model calculations.

Follows the same pattern as MLB: trains Random Forest regressors to predict
field distribution shapes (projection and ownership histograms) based on
contest parameters (entries, fee, games, ownership_gini).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sqlalchemy.orm import Session

from ivan.sports.nfl.models import ClassicContestMetadata

BINS = 20  # Fixed-width histogram bins


def build_slate_bin_regressors(session: Session, cutoff=None) -> dict:
    """
    Fit regressors mapping slate features to fixed-width bin masses for
    projection and ownership distributions.

    Returns dict with:
        - scaler: StandardScaler for features
        - proj_model: MultiOutputRegressor for projection distribution
        - own_model: MultiOutputRegressor for ownership distribution
        - bin_edges: np.array of bin edges [0, 0.05, 0.10, ..., 1.0]
    """
    # Query historical classic contest metadata
    q = session.query(ClassicContestMetadata).filter(
        ClassicContestMetadata.hist_proj_counts.isnot(None),
        ClassicContestMetadata.hist_own_counts.isnot(None),
    )
    if cutoff is not None:                      # as-of build: only contests that finished before the cutoff
        q = q.filter(ClassicContestMetadata.date < cutoff)
    recs = q.all()

    bin_edges = np.linspace(0.0, 1.0, BINS + 1)

    if not recs:
        print("⚠️  No historical classic contest data: returning fallback models")
        # Return simple models that will work
        scaler = StandardScaler().fit(np.zeros((1, 4)))
        uniform = np.ones(BINS) / BINS
        proj_model = ConstantRegressor(uniform)
        own_model = ConstantRegressor(uniform)
        return {
            'scaler': scaler,
            'proj_model': proj_model,
            'own_model': own_model,
            'bin_edges': bin_edges,
        }

    # 1) Build feature matrix X: 4 slate features
    X = np.vstack([
        [
            r.total_entries or 1,
            r.entry_fee or 1.0,
            r.num_games or 1,
            r.ownership_gini or 0.5,
        ]
        for r in recs
    ]).astype(float)

    # 2) Build target matrices for normalized bin masses
    Y_proj = np.zeros((len(recs), BINS), dtype=float)
    Y_own = np.zeros((len(recs), BINS), dtype=float)

    for i, r in enumerate(recs):
        if r.hist_proj_counts:
            proj_counts = np.array(r.hist_proj_counts, dtype=float)
            if proj_counts.sum() > 0:
                Y_proj[i] = proj_counts / proj_counts.sum()
            else:
                Y_proj[i] = np.ones(BINS) / BINS

        if r.hist_own_counts:
            own_counts = np.array(r.hist_own_counts, dtype=float)
            if own_counts.sum() > 0:
                Y_own[i] = own_counts / own_counts.sum()
            else:
                Y_own[i] = np.ones(BINS) / BINS

    # 3) Standardize features
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 4) Fit multi-output RF regressors
    rf_params = dict(n_estimators=200, random_state=42, n_jobs=-1)
    proj_model = MultiOutputRegressor(RandomForestRegressor(**rf_params)).fit(Xs, Y_proj)
    own_model = MultiOutputRegressor(RandomForestRegressor(**rf_params)).fit(Xs, Y_own)

    print(f"✓ Trained on {len(recs)} historical contests")
    print(f"  Features: entries, fee, games, gini")
    print(f"  Targets: 20-bin proj & ownership distributions")

    return {
        'scaler': scaler,
        'proj_model': proj_model,
        'own_model': own_model,
        'bin_edges': bin_edges,
    }


class ConstantRegressor:
    """Fallback regressor that always predicts a constant distribution."""

    def __init__(self, constant: np.ndarray):
        self.constant = np.asarray(constant, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return constant for each sample."""
        n_samples = len(X) if hasattr(X, '__len__') else 1
        return np.tile(self.constant, (n_samples, 1))
