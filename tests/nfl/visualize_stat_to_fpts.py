#!/usr/bin/env python3
"""
NFL Stat → FPTS Visualization Pipeline

This script creates comprehensive visualizations showing how individual player stats
are modeled and combined into final FPTS projections.

The visualization shows:
1. Historical data distribution for each stat (histogram + KDE)
2. Odds points overlaid on the CDF (showing market-implied probabilities)
3. Calibrated CDF after odds adjustment
4. Sample distributions from the calibrated model
5. Correlation structure between stats
6. Final FPTS distribution after applying scoring weights

Usage:
    python visualize_stat_to_fpts.py "Patrick Mahomes"
    python visualize_stat_to_fpts.py "Christian McCaffrey" --samples 10000
    python visualize_stat_to_fpts.py "Travis Kelce" --output kelce_analysis.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import scipy.stats as st
import seaborn as sns

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLPlayer, NFLStat, NFLOdds
from ivan.sports.nfl.refactor.stat import (
    fit_stat_distribution,
    sample_stat,
    extract_odds_points,
    fit_kde_marginal,
    calibrate_kde_to_odds,
    _numerical_cdf_from_kde,
)
from ivan.sports.nfl.refactor.player import STATS

# DraftKings scoring weights
FPTS_WEIGHTS = {
    'passing_yards': 0.04,
    'passing_tds': 4.0,
    'passing_interceptions': -1.0,
    'rushing_yards': 0.1,
    'rushing_tds': 6.0,
    'receptions': 1.0,
    'receiving_yards': 0.1,
    'receiving_tds': 6.0,
}

# Colors for consistent theming
COLORS = {
    'historical': '#3498db',
    'kde': '#2ecc71',
    'odds': '#e74c3c',
    'calibrated': '#9b59b6',
    'samples': '#f39c12',
    'fpts': '#1abc9c',
}


def plot_stat_distribution(ax, stat_name, historical_values, stat_dist, samples):
    """
    Plot a single stat's distribution showing:
    - Historical data (histogram)
    - Base KDE fit
    - Odds points on CDF
    - Calibrated CDF
    - Sample distribution
    """
    # Filter valid historical data
    hist_data = np.array([v for v in historical_values if v is not None])
    hist_data = hist_data[np.isfinite(hist_data)]
    
    if len(hist_data) == 0:
        ax.text(0.5, 0.5, f'No historical data\nfor {stat_name}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(stat_name.replace('_', ' ').title())
        return
    
    # Create grid for evaluation
    x_min = min(hist_data.min(), samples.min()) - 5
    x_max = max(hist_data.max(), samples.max()) + 5
    x_grid = np.linspace(x_min, x_max, 500)
    
    # Plot histogram of historical data
    ax.hist(hist_data, bins=20, density=True, alpha=0.3, 
            color=COLORS['historical'], label='Historical', edgecolor='black', linewidth=0.5)
    
    # Plot base KDE if available
    if stat_dist.kde_support is not None:
        kde, _ = fit_kde_marginal(hist_data)
        kde_pdf = kde(x_grid)
        ax.plot(x_grid, kde_pdf, color=COLORS['kde'], linewidth=2, 
                label='Base KDE', linestyle='--', alpha=0.7)
    
    # Plot calibrated distribution (from samples)
    ax.hist(samples, bins=30, density=True, alpha=0.4, 
            color=COLORS['calibrated'], label='Calibrated (samples)', 
            edgecolor='black', linewidth=0.3)
    
    # Add mean lines
    ax.axvline(hist_data.mean(), color=COLORS['historical'], 
               linestyle=':', linewidth=1.5, alpha=0.7, label=f'Hist mean: {hist_data.mean():.1f}')
    ax.axvline(samples.mean(), color=COLORS['calibrated'], 
               linestyle=':', linewidth=1.5, alpha=0.7, label=f'Calib mean: {samples.mean():.1f}')
    
    # Formatting
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'{stat_name.replace("_", " ").title()}\n{len(stat_dist.odds_points)} odds points')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)


def plot_stat_cdf(ax, stat_name, historical_values, stat_dist, samples):
    """
    Plot CDF comparison showing:
    - Base KDE CDF
    - Odds points (market-implied probabilities)
    - Calibrated CDF from samples
    """
    # Filter valid historical data
    hist_data = np.array([v for v in historical_values if v is not None])
    hist_data = hist_data[np.isfinite(hist_data)]
    
    if len(hist_data) == 0:
        ax.text(0.5, 0.5, f'No historical data\nfor {stat_name}', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create grid for evaluation
    x_min = min(hist_data.min(), samples.min()) - 5
    x_max = max(hist_data.max(), samples.max()) + 5
    x_grid = np.linspace(x_min, x_max, 500)
    
    # Base KDE CDF
    if stat_dist.kde_support is not None:
        kde, _ = fit_kde_marginal(hist_data)
        base_cdf = _numerical_cdf_from_kde(kde, x_grid)
        ax.plot(x_grid, base_cdf, color=COLORS['kde'], linewidth=2, 
                label='Base KDE CDF', linestyle='--', alpha=0.7)
    
    # Calibrated CDF (from samples)
    sorted_samples = np.sort(samples)
    empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    ax.plot(sorted_samples, empirical_cdf, color=COLORS['calibrated'], 
            linewidth=2, label='Calibrated CDF', alpha=0.9)
    
    # Plot odds points
    if stat_dist.odds_points:
        odds_x = [pt for pt, _ in stat_dist.odds_points]
        odds_y = [cdf for _, cdf in stat_dist.odds_points]
        ax.scatter(odds_x, odds_y, color=COLORS['odds'], s=100, 
                   marker='o', edgecolors='black', linewidth=1.5,
                   label=f'Odds points (n={len(stat_dist.odds_points)})', zorder=5)
        
        # Draw vertical lines from odds points to CDF
        for x, y in zip(odds_x, odds_y):
            ax.plot([x, x], [0, y], color=COLORS['odds'], 
                    linestyle=':', alpha=0.3, linewidth=1)
    
    # Formatting
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'{stat_name.replace("_", " ").title()} - CDF Calibration')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(-0.05, 1.05)


def plot_correlation_matrix(ax, stat_names, correlation_matrix):
    """Plot the correlation matrix used for dependent sampling."""
    # Use seaborn for nicer heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, fmt='.2f', 
                cmap='coolwarm', center=0,
                xticklabels=[s.replace('_', '\n') for s in stat_names],
                yticklabels=[s.replace('_', ' ').title() for s in stat_names],
                cbar_kws={'label': 'Spearman Correlation'},
                ax=ax, vmin=-1, vmax=1, square=True)
    ax.set_title('Stat Correlation Matrix\n(Spearman Rank Correlation)', fontsize=10, pad=10)


def plot_fpts_distribution(ax, fpts_samples, player_name, position):
    """
    Plot the final FPTS distribution after applying scoring weights
    and correlation structure.
    """
    # Histogram
    n, bins, patches = ax.hist(fpts_samples, bins=50, density=True, alpha=0.6, 
                                color=COLORS['fpts'], edgecolor='black', linewidth=0.5)
    
    # Fit normal distribution
    mu, sigma = st.norm.fit(fpts_samples)
    x_grid = np.linspace(fpts_samples.min(), fpts_samples.max(), 500)
    pdf = st.norm.pdf(x_grid, mu, sigma)
    ax.plot(x_grid, pdf, color='darkred', linewidth=3, 
            label=f'Normal fit\nμ={mu:.1f}, σ={sigma:.1f}')
    
    # Add percentile lines
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = np.percentile(fpts_samples, percentiles)
    colors_p = ['#e74c3c', '#f39c12', '#2ecc71', '#f39c12', '#e74c3c']
    
    for p, val, color in zip(percentiles, percentile_values, colors_p):
        ax.axvline(val, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(val, ax.get_ylim()[1] * 0.95, f'P{p}\n{val:.1f}', 
                ha='center', va='top', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Mean line
    ax.axvline(mu, color='darkred', linestyle='-', linewidth=2, alpha=0.9)
    
    # Formatting
    ax.set_xlabel('Fantasy Points')
    ax.set_ylabel('Density')
    ax.set_title(f'Final FPTS Distribution\n{player_name} ({position})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)


def plot_sampling_process(ax, stat_samples_dict, stat_names, weights_dict):
    """
    Plot showing how individual stat samples combine into FPTS.
    Box plots for each stat showing the sample distributions.
    """
    # Prepare data for box plots
    data_to_plot = []
    labels = []
    colors_list = []
    
    for stat in stat_names:
        if stat in stat_samples_dict:
            data_to_plot.append(stat_samples_dict[stat])
            labels.append(f'{stat.replace("_", " ").title()}\n(×{weights_dict.get(stat, 0):.2f})')
            colors_list.append(COLORS['samples'])
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2, linestyle='--'))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Formatting
    ax.set_ylabel('Sampled Value')
    ax.set_title('Stat Sample Distributions\n(before correlation)', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.tick_params(axis='x', rotation=0, labelsize=7)


def create_visualization(player_name, n_samples=10000, output_path=None, seed=42):
    """
    Main function to create comprehensive visualization for a player.
    
    Args:
        player_name: Name of the NFL player
        n_samples: Number of Monte Carlo samples to generate
        output_path: Path to save the figure (if None, displays interactively)
        seed: Random seed for reproducibility
    """
    # Initialize database
    engine = init_db('nfl')
    session = get_session(engine)
    
    print(f"🏈 Loading data for {player_name}...")
    
    # Get player
    player = session.query(NFLPlayer).filter(NFLPlayer.name == player_name).first()
    if not player:
        print(f"❌ Player '{player_name}' not found in database")
        return
    
    print(f"✓ Found {player.name} ({player.position})")
    
    # Get historical stats
    historical_stats = session.query(NFLStat).filter(
        NFLStat.player_display_name == player.name
    ).all()
    
    if not historical_stats or len(historical_stats) < 5:
        print(f"❌ Insufficient historical data (found {len(historical_stats)} games)")
        return
    
    print(f"✓ Loaded {len(historical_stats)} historical games")
    
    # Get odds
    odds = session.query(NFLOdds).filter(NFLOdds.player == player.name).all()
    if not odds:
        print(f"⚠️  No odds found for {player.name}")
        odds_df = pd.DataFrame()
    else:
        print(f"✓ Loaded {len(odds)} odds lines")
        odds_df = pd.DataFrame([
            {
                'market': o.market,
                'side': o.side,
                'point': o.point,
                'price': o.price
            }
            for o in odds
        ])
    
    # Get stats to model for this position
    stats_to_model = STATS.get(player.position)
    if not stats_to_model:
        print(f"❌ No stat model defined for position {player.position}")
        return
    
    print(f"✓ Modeling stats: {', '.join(stats_to_model)}")
    print(f"\n📊 Fitting distributions...")
    
    # Fit distributions for each stat
    stat_distributions = []
    stat_samples_dict = {}
    historical_data_dict = {}
    
    rng = np.random.default_rng(seed)
    
    for stat in stats_to_model:
        print(f"  - {stat}...", end=' ')
        
        # Get historical values
        historical_values = [
            getattr(s, stat) for s in historical_stats 
            if getattr(s, stat) is not None
        ]
        historical_data_dict[stat] = historical_values
        
        if not historical_values:
            print("no data, skipping")
            continue
        
        # Fit distribution
        result = fit_stat_distribution(odds_df, historical_values, stat)
        stat_distributions.append(result)
        
        # Generate samples
        samples = sample_stat(result, n_samples, rng)
        stat_samples_dict[stat] = samples
        
        print(f"✓ ({len(result.odds_points)} odds points)")
    
    print(f"\n📈 Building correlation structure...")
    
    # Build correlation matrix from historical data
    df_hist = pd.DataFrame({
        stat: [getattr(s, stat) for s in historical_stats if getattr(s, stat) is not None]
        for stat in stats_to_model
    })
    df_hist = df_hist.dropna()
    
    if df_hist.shape[0] < 5:
        print(f"⚠️  Insufficient joint history for correlation (only {df_hist.shape[0]} games)")
        corr = np.eye(len(stats_to_model))
    else:
        corr = df_hist.corr(method='spearman').fillna(0.0).values
        print(f"✓ Computed correlation from {df_hist.shape[0]} complete games")
    
    print(f"\n🎲 Generating {n_samples:,} correlated samples...")
    
    # Generate independent samples
    X = np.zeros((n_samples, len(stat_distributions)), dtype=float)
    for j, sd in enumerate(stat_distributions):
        X[:, j] = stat_samples_dict[sd.stat]
    
    # Apply correlation using Gaussian copula (Iman-Conover method)
    U = np.zeros_like(X)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j])
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n_samples)
        U[:, j] = (ranks + 0.5) / n_samples
    
    # Generate correlated normals
    L = np.linalg.cholesky(corr + 1e-6 * np.eye(corr.shape[0]))
    Z = rng.standard_normal(size=(n_samples, corr.shape[0])) @ L.T
    U_target = st.norm.cdf(Z)
    
    # Reorder to impose correlation
    X_dep = np.zeros_like(X)
    for j in range(X.shape[1]):
        x_sorted = np.sort(X[:, j])
        order = np.argsort(U_target[:, j])
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(n_samples)
        X_dep[:, j] = x_sorted[inv_order]
    
    # Compute FPTS
    weights = np.array([FPTS_WEIGHTS.get(s.stat, 0.0) for s in stat_distributions], dtype=float)
    fpts_samples = X_dep @ weights
    
    print(f"✓ Mean FPTS: {fpts_samples.mean():.2f} ± {fpts_samples.std():.2f}")
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    
    # Determine grid size based on number of stats
    n_stats = len(stat_distributions)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 14))
    
    # Use GridSpec for flexible layout
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.05, right=0.95, top=0.94, bottom=0.05)
    
    # Title
    fig.suptitle(f'NFL Stat → FPTS Modeling Pipeline: {player.name} ({player.position})',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Top row: Individual stat PDFs
    stat_pdf_axes = []
    for i in range(min(4, n_stats)):
        ax = fig.add_subplot(gs[0, i])
        stat_pdf_axes.append(ax)
    
    # Second row: Individual stat CDFs
    stat_cdf_axes = []
    for i in range(min(4, n_stats)):
        ax = fig.add_subplot(gs[1, i])
        stat_cdf_axes.append(ax)
    
    # Third row left: Correlation matrix
    ax_corr = fig.add_subplot(gs[2, :2])
    
    # Third row right: Sampling process
    ax_sampling = fig.add_subplot(gs[2, 2:])
    
    # Fourth row: FPTS distribution (spanning all columns)
    ax_fpts = fig.add_subplot(gs[3, :])
    
    # Plot individual stat distributions (PDF)
    for i, sd in enumerate(stat_distributions[:4]):
        if i < len(stat_pdf_axes):
            plot_stat_distribution(
                stat_pdf_axes[i], 
                sd.stat, 
                historical_data_dict[sd.stat],
                sd, 
                stat_samples_dict[sd.stat]
            )
    
    # Plot individual stat CDFs with odds
    for i, sd in enumerate(stat_distributions[:4]):
        if i < len(stat_cdf_axes):
            plot_stat_cdf(
                stat_cdf_axes[i], 
                sd.stat, 
                historical_data_dict[sd.stat],
                sd, 
                stat_samples_dict[sd.stat]
            )
    
    # Plot correlation matrix
    plot_correlation_matrix(
        ax_corr,
        [sd.stat for sd in stat_distributions],
        corr
    )
    
    # Plot sampling process
    plot_sampling_process(
        ax_sampling,
        stat_samples_dict,
        [sd.stat for sd in stat_distributions],
        FPTS_WEIGHTS
    )
    
    # Plot final FPTS distribution
    plot_fpts_distribution(ax_fpts, fpts_samples, player.name, player.position)
    
    # Add methodology text box
    methodology_text = (
        "Methodology:\n"
        "1. Fit KDE to historical stat values\n"
        "2. Extract market odds → CDF points\n"
        "3. Calibrate KDE to match odds\n"
        "4. Sample from calibrated distributions\n"
        "5. Apply Gaussian copula for correlation\n"
        "6. Compute FPTS with DK scoring weights"
    )
    
    fig.text(0.02, 0.02, methodology_text, 
             fontsize=8, va='bottom', ha='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add stats summary
    stats_text = (
        f"Samples: {n_samples:,}\n"
        f"Historical Games: {len(historical_stats)}\n"
        f"Odds Lines: {len(odds)}\n"
        f"Stats Modeled: {len(stat_distributions)}"
    )
    
    fig.text(0.98, 0.02, stats_text, 
             fontsize=8, va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
    else:
        # Try to use interactive backend if available
        backend = matplotlib.get_backend()
        if backend == 'agg' or backend.lower() == 'agg':
            print(f"⚠️  Non-interactive backend detected ({backend})")
            print(f"💡 Tip: Set MPLBACKEND=TkAgg or save to file with --output")
            # Save to temp file instead
            temp_output = f"temp_viz_{player.name.replace(' ', '_')}.png"
            plt.savefig(temp_output, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {temp_output}")
        else:
            print(f"✓ Displaying interactive plot...")
            plt.show()
    
    plt.close()
    print(f"\n✅ Visualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize NFL stat → FPTS modeling pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Patrick Mahomes"
  %(prog)s "Christian McCaffrey" --samples 20000
  %(prog)s "Travis Kelce" --output kelce_viz.png
  %(prog)s "Tyreek Hill" --seed 123
        """
    )
    
    parser.add_argument('player_name', type=str, 
                        help='Name of the NFL player to visualize')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of Monte Carlo samples (default: 10000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (if not provided, shows interactively)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    try:
        create_visualization(
            player_name=args.player_name,
            n_samples=args.samples,
            output_path=args.output,
            seed=args.seed
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
