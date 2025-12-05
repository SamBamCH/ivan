#!/usr/bin/env python3
"""
NFL Player Comparison Visualization

Creates side-by-side comparisons of multiple players' FPTS distributions
and stat modeling quality.

Usage:
    python visualize_player_comparison.py "Patrick Mahomes" "Josh Allen" "Lamar Jackson"
    python visualize_player_comparison.py --position QB --top 5
    python visualize_player_comparison.py --output qb_comparison.png "Patrick Mahomes" "Josh Allen"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as st
import seaborn as sns

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLPlayer, NFLStat, NFLOdds
from ivan.sports.nfl.refactor.stat import fit_stat_distribution, sample_stat
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


def generate_player_fpts_samples(session, player_name, n_samples=10000, seed=42):
    """
    Generate FPTS samples for a single player.
    
    Returns:
        dict with keys: player, fpts_samples, stat_distributions, metadata
    """
    rng = np.random.default_rng(seed)
    
    # Get player
    player = session.query(NFLPlayer).filter(NFLPlayer.name == player_name).first()
    if not player:
        return None
    
    # Get historical stats
    historical_stats = session.query(NFLStat).filter(
        NFLStat.player_display_name == player.name
    ).all()
    
    if not historical_stats or len(historical_stats) < 5:
        return None
    
    # Get odds
    odds = session.query(NFLOdds).filter(NFLOdds.player == player.name).all()
    odds_df = pd.DataFrame([
        {
            'market': o.market,
            'side': o.side,
            'point': o.point,
            'price': o.price
        }
        for o in odds
    ]) if odds else pd.DataFrame()
    
    # Get stats to model
    stats_to_model = STATS.get(player.position)
    if not stats_to_model:
        return None
    
    # Fit distributions
    stat_distributions = []
    X = []
    
    for stat in stats_to_model:
        historical_values = [
            getattr(s, stat) for s in historical_stats 
            if getattr(s, stat) is not None
        ]
        
        if not historical_values:
            continue
        
        result = fit_stat_distribution(odds_df, historical_values, stat)
        stat_distributions.append(result)
        samples = sample_stat(result, n_samples, rng)
        X.append(samples)
    
    if not stat_distributions:
        return None
    
    X = np.column_stack(X)
    
    # Build correlation
    df_hist = pd.DataFrame({
        stat: [getattr(s, stat) for s in historical_stats if getattr(s, stat) is not None]
        for stat in stats_to_model
    }).dropna()
    
    if df_hist.shape[0] >= 5:
        corr = df_hist.corr(method='spearman').fillna(0.0).values
        
        # Apply correlation
        U = np.zeros_like(X)
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j])
            ranks = np.empty_like(order)
            ranks[order] = np.arange(n_samples)
            U[:, j] = (ranks + 0.5) / n_samples
        
        L = np.linalg.cholesky(corr + 1e-6 * np.eye(corr.shape[0]))
        Z = rng.standard_normal(size=(n_samples, corr.shape[0])) @ L.T
        U_target = st.norm.cdf(Z)
        
        X_dep = np.zeros_like(X)
        for j in range(X.shape[1]):
            x_sorted = np.sort(X[:, j])
            order = np.argsort(U_target[:, j])
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(n_samples)
            X_dep[:, j] = x_sorted[inv_order]
    else:
        X_dep = X
    
    # Compute FPTS
    weights = np.array([FPTS_WEIGHTS.get(sd.stat, 0.0) for sd in stat_distributions], dtype=float)
    fpts_samples = X_dep @ weights
    
    # Calculate metadata
    total_odds_points = sum(len(sd.odds_points) for sd in stat_distributions)
    
    return {
        'player': player,
        'fpts_samples': fpts_samples,
        'stat_distributions': stat_distributions,
        'metadata': {
            'n_historical_games': len(historical_stats),
            'n_odds_lines': len(odds),
            'n_stats_modeled': len(stat_distributions),
            'total_odds_points': total_odds_points,
        }
    }


def plot_fpts_comparison(fig, player_results, output_path=None):
    """
    Create comparison visualization of multiple players.
    """
    n_players = len(player_results)
    
    # Create grid
    gs = gridspec.GridSpec(3, n_players, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    fig.suptitle('NFL Player FPTS Distribution Comparison', 
                 fontsize=16, fontweight='bold', y=0.97)
    
    # Colors for each player
    colors = plt.cm.Set2(np.linspace(0, 1, n_players))
    
    # Collect all data for shared axis limits
    all_fpts = np.concatenate([r['fpts_samples'] for r in player_results])
    fpts_min, fpts_max = all_fpts.min(), all_fpts.max()
    
    # Row 1: Individual FPTS distributions
    for i, result in enumerate(player_results):
        ax = fig.add_subplot(gs[0, i])
        fpts = result['fpts_samples']
        player = result['player']
        
        # Histogram
        ax.hist(fpts, bins=40, density=True, alpha=0.6, 
                color=colors[i], edgecolor='black', linewidth=0.5)
        
        # Fit normal
        mu, sigma = st.norm.fit(fpts)
        x_grid = np.linspace(fpts_min, fpts_max, 300)
        pdf = st.norm.pdf(x_grid, mu, sigma)
        ax.plot(x_grid, pdf, color='darkred', linewidth=2, label=f'μ={mu:.1f}, σ={sigma:.1f}')
        
        # Mean line
        ax.axvline(mu, color='darkred', linestyle='--', linewidth=2, alpha=0.7)
        
        # Formatting
        ax.set_title(f'{player.name}\n({player.position})', fontsize=10, fontweight='bold')
        ax.set_xlabel('FPTS')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(fpts_min - 5, fpts_max + 5)
    
    # Row 2: Overlay CDFs for direct comparison
    ax_cdf = fig.add_subplot(gs[1, :])
    
    for i, result in enumerate(player_results):
        fpts = result['fpts_samples']
        player = result['player']
        
        sorted_fpts = np.sort(fpts)
        cdf = np.arange(1, len(sorted_fpts) + 1) / len(sorted_fpts)
        
        ax_cdf.plot(sorted_fpts, cdf, linewidth=2.5, color=colors[i], 
                    label=f'{player.name} (μ={fpts.mean():.1f})', alpha=0.8)
    
    ax_cdf.set_xlabel('Fantasy Points', fontsize=11)
    ax_cdf.set_ylabel('Cumulative Probability', fontsize=11)
    ax_cdf.set_title('FPTS CDF Comparison', fontsize=12, fontweight='bold')
    ax_cdf.legend(fontsize=9, loc='best')
    ax_cdf.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax_cdf.set_xlim(fpts_min - 5, fpts_max + 5)
    
    # Row 3: Summary statistics comparison
    ax_stats = fig.add_subplot(gs[2, :])
    
    # Prepare data
    player_names = [r['player'].name for r in player_results]
    means = [r['fpts_samples'].mean() for r in player_results]
    stds = [r['fpts_samples'].std() for r in player_results]
    p10s = [np.percentile(r['fpts_samples'], 10) for r in player_results]
    p90s = [np.percentile(r['fpts_samples'], 90) for r in player_results]
    
    x_pos = np.arange(len(player_names))
    width = 0.6
    
    # Bar chart with error bars
    bars = ax_stats.bar(x_pos, means, width, yerr=stds, 
                        color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                        error_kw={'linewidth': 2, 'ecolor': 'darkred'})
    
    # Add P10-P90 range as lines
    for i, (p10, p90) in enumerate(zip(p10s, p90s)):
        ax_stats.plot([i, i], [p10, p90], color='blue', linewidth=3, alpha=0.5)
        ax_stats.scatter([i, i], [p10, p90], color='blue', s=50, zorder=5)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax_stats.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                     f'{mean:.1f}\n±{std:.1f}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax_stats.set_xlabel('Player', fontsize=11)
    ax_stats.set_ylabel('Fantasy Points', fontsize=11)
    ax_stats.set_title('FPTS Summary Statistics (Mean ± Std, P10-P90 range)', 
                       fontsize=12, fontweight='bold')
    ax_stats.set_xticks(x_pos)
    ax_stats.set_xticklabels(player_names, fontsize=9)
    ax_stats.grid(alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Add metadata table
    metadata_rows = []
    for result in player_results:
        meta = result['metadata']
        metadata_rows.append([
            result['player'].name,
            meta['n_historical_games'],
            meta['n_odds_lines'],
            meta['n_stats_modeled'],
            meta['total_odds_points']
        ])
    
    table_data = pd.DataFrame(metadata_rows, 
                             columns=['Player', 'Games', 'Odds', 'Stats', 'Odds Pts'])
    
    # Add table below the plot
    table_text = table_data.to_string(index=False)
    fig.text(0.5, 0.02, table_text, 
             fontsize=8, ha='center', va='bottom', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))


def create_comparison_visualization(player_names, n_samples=10000, output_path=None, seed=42):
    """
    Main function to create player comparison visualization.
    """
    engine = init_db('nfl')
    session = get_session(engine)
    
    print(f"🏈 Processing {len(player_names)} players...")
    
    player_results = []
    
    for name in player_names:
        print(f"\n  Processing {name}...", end=' ')
        result = generate_player_fpts_samples(session, name, n_samples, seed)
        
        if result is None:
            print(f"❌ Failed (no data or insufficient data)")
            continue
        
        player_results.append(result)
        print(f"✓ Mean FPTS: {result['fpts_samples'].mean():.1f}")
    
    if not player_results:
        print(f"\n❌ No valid players to visualize")
        return
    
    print(f"\n✓ Successfully processed {len(player_results)} players")
    print(f"\n🎨 Creating comparison visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(6 * len(player_results), 12))
    
    plot_fpts_comparison(fig, player_results, output_path)
    
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
            player_names_str = '_'.join([r['player'].name.replace(' ', '_') for r in player_results[:3]])
            temp_output = f"temp_comparison_{player_names_str}.png"
            plt.savefig(temp_output, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {temp_output}")
        else:
            print(f"✓ Displaying interactive plot...")
            plt.show()
    
    plt.close()
    print(f"\n✅ Comparison complete!")


def get_top_players_by_position(session, position, limit=5):
    """Get top players by position based on salary."""
    players = session.query(NFLPlayer).filter(
        NFLPlayer.position == position
    ).order_by(NFLPlayer.salary.desc()).limit(limit).all()
    
    return [p.name for p in players]


def main():
    parser = argparse.ArgumentParser(
        description='Compare NFL players FPTS distributions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Patrick Mahomes" "Josh Allen" "Lamar Jackson"
  %(prog)s --position QB --top 5
  %(prog)s --output comparison.png "Christian McCaffrey" "Derrick Henry"
        """
    )
    
    parser.add_argument('players', nargs='*', type=str,
                        help='Names of NFL players to compare')
    parser.add_argument('--position', type=str, default=None,
                        help='Position to filter (e.g., QB, RB, WR)')
    parser.add_argument('--top', type=int, default=5,
                        help='Number of top players by salary (requires --position)')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of Monte Carlo samples (default: 10000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (if not provided, shows interactively)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Determine player list
    if args.position:
        engine = init_db('nfl')
        session = get_session(engine)
        player_names = get_top_players_by_position(session, args.position, args.top)
        print(f"📋 Selected top {len(player_names)} {args.position}s by salary:")
        for name in player_names:
            print(f"   - {name}")
    elif args.players:
        player_names = args.players
    else:
        parser.print_help()
        sys.exit(1)
    
    if not player_names:
        print("❌ No players specified")
        sys.exit(1)
    
    try:
        create_comparison_visualization(
            player_names=player_names,
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
