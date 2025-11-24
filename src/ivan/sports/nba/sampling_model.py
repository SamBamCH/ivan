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

def print_model_diagnostics(model_dict, n_samples=5):
    """Print diagnostics about the fitted model."""
    print("\n=== Field Distribution Model Diagnostics ===")
    print(f"\nScaler fitted on 3 features: [entry_fee, total_entries, multi_entry]")
    print(f"  Mean: {model_dict['scaler'].mean_}")
    print(f"  Std:  {model_dict['scaler'].scale_}")
    
    print(f"\nOwnership Model: {type(model_dict['own_model']).__name__}")
    print(f"Projection Model: {type(model_dict['proj_model']).__name__}")
    
    print(f"\nBin Edges ({len(model_dict['bin_edges'])} edges, {BINS} bins):")
    print(f"  {model_dict['bin_edges']}")
    
    # Check if we have actual training data
    if hasattr(model_dict['own_model'], 'estimators_'):
        print(f"\nNumber of RF estimators: {len(model_dict['own_model'].estimators_)}")
    
    # Show example predictions
    print(f"\n--- Example Predictions ---")
    test_cases = [
        [0.50, 20000, 150],   # $0.50 GPP, 20k entries, 150 max
        [4.0, 5000, 20],      # $4 GPP, 5k entries, 20 max
        [100.0, 500, 1],      # $100 GPP, 500 entries, 1 max
    ]
    
    for i, test_x in enumerate(test_cases[:n_samples]):
        X_test = np.array([test_x])
        X_scaled = model_dict['scaler'].transform(X_test)
        
        own_pred = model_dict['own_model'].predict(X_scaled)[0]
        proj_pred = model_dict['proj_model'].predict(X_scaled)[0]
        
        print(f"\nTest {i+1}: entry_fee=${test_x[0]}, entries={test_x[1]}, multi={test_x[2]}")
        print(f"  Ownership dist - sum: {own_pred.sum():.3f}, mean bin: {np.argmax(own_pred)}, max prob: {own_pred.max():.3f}")
        print(f"  Projection dist - sum: {proj_pred.sum():.3f}, mean bin: {np.argmax(proj_pred)}, max prob: {proj_pred.max():.3f}")
        
        # Show top 3 bins for ownership
        top_own_bins = np.argsort(own_pred)[-3:][::-1]
        print(f"  Top 3 ownership bins: {top_own_bins} with probs {own_pred[top_own_bins]}")

def plot_distributions(model_dict, n_samples=3):
    """Plot predicted distributions for different contest types."""
    test_cases = [
        ([0.50, 20000, 150], "$0.50 GPP (20k entries, 150 max)"),
        ([4.0, 5000, 20], "$4 GPP (5k entries, 20 max)"),
        ([100.0, 500, 1], "$100 GPP (500 entries, 1 max)"),
    ]
    
    bin_centers = (model_dict['bin_edges'][:-1] + model_dict['bin_edges'][1:]) / 2
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, (test_x, label) in enumerate(test_cases[:n_samples]):
        X_test = np.array([test_x])
        X_scaled = model_dict['scaler'].transform(X_test)
        
        own_pred = model_dict['own_model'].predict(X_scaled)[0]
        proj_pred = model_dict['proj_model'].predict(X_scaled)[0]
        
        # Ownership distribution
        axes[i, 0].bar(bin_centers, own_pred, width=0.04, alpha=0.7, color='blue', edgecolor='black')
        axes[i, 0].set_xlabel('Normalized Ownership (0-1)')
        axes[i, 0].set_ylabel('Probability Density')
        axes[i, 0].set_title(f'Ownership Distribution\n{label}')
        axes[i, 0].set_xlim(0, 1)
        axes[i, 0].grid(alpha=0.3)
        
        # Projection distribution
        axes[i, 1].bar(bin_centers, proj_pred, width=0.04, alpha=0.7, color='green', edgecolor='black')
        axes[i, 1].set_xlabel('Normalized Projection (0-1)')
        axes[i, 1].set_ylabel('Probability Density')
        axes[i, 1].set_title(f'Projection Distribution\n{label}')
        axes[i, 1].set_xlim(0, 1)
        axes[i, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/nba/models/field_distribution_plots.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Distribution plots saved to data/nba/models/field_distribution_plots.png")
    plt.show()

def plot_training_data(model_dict):
    """Plot the average training distributions to understand the data."""
    if 'Y_own_train' not in model_dict or 'Y_proj_train' not in model_dict:
        print("Training data not stored in model dict")
        return
    
    Y_own = model_dict['Y_own_train']
    Y_proj = model_dict['Y_proj_train']
    Y_own_raw = model_dict.get('Y_own_raw', Y_own)
    Y_proj_raw = model_dict.get('Y_proj_raw', Y_proj)
    bin_centers = (model_dict['bin_edges'][:-1] + model_dict['bin_edges'][1:]) / 2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Average ownership distribution - raw vs smoothed
    avg_own_raw = Y_own_raw.mean(axis=0)
    avg_own_smooth = Y_own.mean(axis=0)
    axes[0, 0].bar(bin_centers, avg_own_raw, width=0.04, alpha=0.5, color='red', edgecolor='black', label='Raw')
    axes[0, 0].plot(bin_centers, avg_own_smooth, 'b-', linewidth=2, label='Smoothed')
    axes[0, 0].set_xlabel('Normalized Ownership (0-1)')
    axes[0, 0].set_ylabel('Average Probability')
    axes[0, 0].set_title(f'Average Ownership Distribution\nn={Y_own.shape[0]} contests')
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Average projection distribution - raw vs smoothed
    avg_proj_raw = Y_proj_raw.mean(axis=0)
    avg_proj_smooth = Y_proj.mean(axis=0)
    axes[0, 1].bar(bin_centers, avg_proj_raw, width=0.04, alpha=0.5, color='red', edgecolor='black', label='Raw')
    axes[0, 1].plot(bin_centers, avg_proj_smooth, 'g-', linewidth=2, label='Smoothed')
    axes[0, 1].set_xlabel('Normalized Projection (0-1)')
    axes[0, 1].set_ylabel('Average Probability')
    axes[0, 1].set_title(f'Average Projection Distribution\nn={Y_proj.shape[0]} contests')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Show a few individual ownership distributions (smoothed)
    axes[1, 0].set_title('Sample Individual Ownership Distributions (Smoothed)')
    for i in range(min(5, Y_own.shape[0])):
        axes[1, 0].plot(bin_centers, Y_own[i], alpha=0.5, linewidth=1)
    axes[1, 0].set_xlabel('Normalized Ownership (0-1)')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].grid(alpha=0.3)
    
    # Show a few individual projection distributions (smoothed)
    axes[1, 1].set_title('Sample Individual Projection Distributions (Smoothed)')
    for i in range(min(5, Y_proj.shape[0])):
        axes[1, 1].plot(bin_centers, Y_proj[i], alpha=0.5, linewidth=1)
    axes[1, 1].set_xlabel('Normalized Projection (0-1)')
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/nba/models/training_data_distributions.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Training data plots saved to data/nba/models/training_data_distributions.png")
    plt.show()
