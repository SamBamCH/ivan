"""
Test the fitted NBA models by generating sample predictions.

This script:
1. Loads marginals and correlations
2. Generates correlated samples using Gaussian copula
3. Calculates DK fantasy points
4. Displays summary statistics

Usage:
    python tests/nba/test_models.py
"""
from pathlib import Path
import joblib
import numpy as np
import scipy.stats as st


def load_models(model_dir: Path):
    """Load marginals and correlations."""
    marginals = joblib.load(model_dir / 'marginals.pkl')
    correlations = joblib.load(model_dir / 'correlations.pkl')
    return marginals, correlations


def ppf_from_dist(dist_name: str, params: tuple, u: np.ndarray) -> np.ndarray:
    """
    Transform uniform samples to stat values using the fitted distribution.

    Args:
        dist_name: Distribution name (e.g., 'nbinom', 'poisson')
        params: Distribution parameters
        u: Uniform samples [0, 1]

    Returns:
        Stat values
    """
    dist = getattr(st, dist_name)
    return dist.ppf(u, *params).astype(float)


def simulate_player_stats(marginals: dict, correlations: dict, position: str = 'G', n_samples: int = 10000):
    """
    Simulate correlated stat samples for a player using Gaussian copula.

    Args:
        marginals: Marginal distributions dict
        correlations: Correlation matrices dict
        position: Position group ('G', 'F', 'C')
        n_samples: Number of samples to generate

    Returns:
        Dict mapping stat name to samples array
    """
    pos_marginals = marginals[position]
    pos_corr = correlations[position]

    stats = ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks']

    # Generate correlated normal samples
    Z = np.random.multivariate_normal(np.zeros(len(stats)), pos_corr, size=n_samples)

    # Transform to uniform [0, 1]
    U = st.norm.cdf(Z)

    # Transform to stat values using marginals
    samples = {}
    for i, stat in enumerate(stats):
        stat_info = pos_marginals[stat]

        if stat_info['dist'] is None or stat_info['params'] is None:
            # No distribution fitted, use zeros
            samples[stat] = np.zeros(n_samples)
        else:
            samples[stat] = ppf_from_dist(stat_info['dist'], stat_info['params'], U[:, i])

    return samples


def calculate_dk_fpts(samples: dict) -> np.ndarray:
    """
    Calculate DraftKings fantasy points from stat samples.

    DK Scoring:
    - Points: 1.0
    - 3PM: 0.5
    - Rebounds: 1.25
    - Assists: 1.5
    - Steals: 2.0
    - Blocks: 2.0
    - Turnovers: -0.5 (we're not modeling this yet)
    - Double-double bonus: +1.5
    - Triple-double bonus: +3.0 total
    """
    pts = samples['points']
    fg3m = samples['fg3m']
    reb = samples['rebounds']
    ast = samples['assists']
    stl = samples['steals']
    blk = samples['blocks']

    # Base fantasy points
    fpts = (pts * 1.0 +
            fg3m * 0.5 +
            reb * 1.25 +
            ast * 1.5 +
            stl * 2.0 +
            blk * 2.0)

    # Double-double and triple-double bonuses
    # Count stats >= 10
    double_digit_count = (
        (pts >= 10).astype(int) +
        (reb >= 10).astype(int) +
        (ast >= 10).astype(int) +
        (stl >= 10).astype(int) +
        (blk >= 10).astype(int)
    )

    # Double-double: +1.5
    fpts += (double_digit_count >= 2) * 1.5

    # Triple-double: additional +1.5 (total +3.0)
    fpts += (double_digit_count >= 3) * 1.5

    return fpts


def main():
    """Main entry point."""
    model_dir = Path("/home/sam-pratt/ivan/data/nba/models")

    print("="*60)
    print("Testing NBA Models")
    print("="*60)

    # Load models
    marginals, correlations = load_models(model_dir)
    print("✓ Loaded marginals and correlations\n")

    # Test for each position
    for position in ['G', 'F', 'C']:
        print(f"\n{'='*60}")
        print(f"Position: {position}")
        print('='*60)

        # Generate samples
        samples = simulate_player_stats(marginals, correlations, position=position, n_samples=10000)

        # Calculate fantasy points
        fpts = calculate_dk_fpts(samples)

        # Print statistics
        print(f"\nStat Distributions (n=10,000 samples):")
        print(f"{'Stat':<12s} {'Mean':>8s} {'Std':>8s} {'P10':>8s} {'P50':>8s} {'P90':>8s}")
        print("-" * 60)

        for stat in ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks']:
            s = samples[stat]
            print(f"{stat:<12s} {s.mean():8.2f} {s.std():8.2f} "
                  f"{np.percentile(s, 10):8.2f} {np.percentile(s, 50):8.2f} {np.percentile(s, 90):8.2f}")

        print("\nFantasy Points Distribution:")
        print(f"{'Stat':<12s} {'Mean':>8s} {'Std':>8s} {'P10':>8s} {'P50':>8s} {'P90':>8s}")
        print("-" * 60)
        print(f"{'DK FPTS':<12s} {fpts.mean():8.2f} {fpts.std():8.2f} "
              f"{np.percentile(fpts, 10):8.2f} {np.percentile(fpts, 50):8.2f} {np.percentile(fpts, 90):8.2f}")

        # Check for double-doubles and triple-doubles
        pts_dd = (samples['points'] >= 10).sum() / len(samples['points']) * 100
        reb_dd = (samples['rebounds'] >= 10).sum() / len(samples['rebounds']) * 100
        ast_dd = (samples['assists'] >= 10).sum() / len(samples['assists']) * 100

        print(f"\nDouble-Digit Percentages:")
        print(f"  Points >= 10:   {pts_dd:5.1f}%")
        print(f"  Rebounds >= 10: {reb_dd:5.1f}%")
        print(f"  Assists >= 10:  {ast_dd:5.1f}%")

    print("\n" + "="*60)
    print("✓ Model testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
