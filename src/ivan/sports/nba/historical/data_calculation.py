import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from ivan.sports.nba.models import ClassicContestMetadata, HistoricalContest

BINS = 20
PROJECTION_MIN = 200.0  # Fixed minimum projection for normalization

class NormalBinRegressor:
    """Predicts the same normal PDF over fixed bins for every X."""
    def __init__(self, mean: float, std: float, bin_edges: np.ndarray):
        self.mean = mean
        self.std = std
        self.centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        pdf = np.exp(-0.5 * ((self.centers - self.mean) / self.std) ** 2)
        pdf /= (self.std * np.sqrt(2 * np.pi))
        pdf = pdf / pdf.sum()
        return np.tile(pdf, (X.shape[0], 1))

class ConstantRegressor:
    """A regressor that always predicts the same vector y regardless of input X."""
    def __init__(self, y):
        self.y = np.asarray(y)

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        n = X.shape[0]
        return np.tile(self.y.reshape(1, -1), (n, 1))

def build_field_distribution_model(session):
    """
    Fit regressors mapping contest features to ownership and projection distributions.
    
    Each contest has normalized histograms (20 bins from 0-1) for:
    - Ownership distribution (how ownership is spread across the field)
    - Projection distribution (how projections are spread across the field)
    
    Predictors: entry_fee, total_entries, multi_entry
    
    Returns dict with scaler, own_model, proj_model, bin_edges.
    """
    # Query historical contest metadata
    metadata_recs = (
        session.query(ClassicContestMetadata, HistoricalContest)
        .join(HistoricalContest, ClassicContestMetadata.contest_id == HistoricalContest.contest_id)
        .all()
    )
    
    bin_edges = np.linspace(0.0, 1.0, BINS + 1)

    # Build feature matrix X: [entry_fee, total_entries, multi_entry]
    X = np.vstack([
        [
            contest.entry_fee,
            contest.total_entries,
            contest.multi_entry,
        ]
        for metadata, contest in metadata_recs
    ]).astype(float)

    # Build target matrices Y: normalized bin counts
    Y_own  = np.zeros((len(metadata_recs), BINS), dtype=float)
    Y_proj = np.zeros((len(metadata_recs), BINS), dtype=float)
    
    Y_own_raw = Y_own.copy()
    Y_proj_raw = Y_proj.copy()
    
    for i, (metadata, contest) in enumerate(metadata_recs):
        c_own  = np.array(metadata.hist_own_counts,  dtype=float)
        c_proj = np.array(metadata.hist_proj_counts, dtype=float)
        
        # Normalize to sum to 1
        if c_own.sum() > 0:
            normalized_own = c_own / c_own.sum()
            # Apply Gaussian smoothing with sigma=1.5 to reduce noise
            smoothed_own = gaussian_filter1d(normalized_own, sigma=1.5)
            Y_own[i] = smoothed_own / smoothed_own.sum()  # Re-normalize after smoothing
            Y_own_raw[i] = normalized_own
        
        if c_proj.sum() > 0:
            normalized_proj = c_proj / c_proj.sum()
            # Apply Gaussian smoothing with sigma=1.5 to reduce noise
            smoothed_proj = gaussian_filter1d(normalized_proj, sigma=1.5)
            Y_proj[i] = smoothed_proj / smoothed_proj.sum()  # Re-normalize after smoothing
            Y_proj_raw[i] = normalized_proj

    # Standardize features
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Debug: Print average distributions from training data
    print(f"\nTraining data statistics (after smoothing):")
    print(f"  Peak ownership bin: {Y_own.mean(axis=0).argmax()} (value: {Y_own.mean(axis=0).max():.3f})")
    print(f"  Peak projection bin: {Y_proj.mean(axis=0).argmax()} (value: {Y_proj.mean(axis=0).max():.3f})")
    
    # Fit multi-output RF regressors
    rf_params = dict(n_estimators=200, random_state=0)
    own_model  = MultiOutputRegressor(RandomForestRegressor(**rf_params)).fit(Xs, Y_own)
    proj_model = MultiOutputRegressor(RandomForestRegressor(**rf_params)).fit(Xs, Y_proj)

    return {
        'scaler':      scaler,
        'own_model':   own_model,
        'proj_model':  proj_model,
        'bin_edges':   bin_edges,
        'projection_min': PROJECTION_MIN,  # Store for use in sampling
        'Y_own_train': Y_own,  # Store smoothed for diagnostics
        'Y_proj_train': Y_proj,
        'Y_own_raw': Y_own_raw,  # Store raw for comparison
        'Y_proj_raw': Y_proj_raw,
    }



