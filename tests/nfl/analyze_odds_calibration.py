#!/usr/bin/env python3
"""
NFL Odds Calibration Quality Analysis

Analyzes how well the calibrated distributions match the market odds
for individual players. Shows before/after calibration accuracy.

Usage:
    python analyze_odds_calibration.py "Patrick Mahomes"
    python analyze_odds_calibration.py --position QB --top 10
    python analyze_odds_calibration.py "Travis Kelce" --output kelce_calibration.png
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

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLPlayer, NFLStat, NFLOdds
from ivan.sports.nfl.refactor.stat import (
    fit_stat_distribution,
    fit_kde_marginal,
    _numerical_cdf_from_kde,
    extract_odds_points,
)
from ivan.sports.nfl.refactor.player import STATS


def analyze_calibration_quality(player_name, output_path=None):
    """
    Analyze and visualize odds calibration quality for a player.
    """
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
    
    # Get odds
    odds = session.query(NFLOdds).filter(NFLOdds.player == player.name).all()
    if not odds:
        print(f"❌ No odds found for {player.name}")
        return
    
    odds_df = pd.DataFrame([
        {
            'market': o.market,
            'side': o.side,
            'point': o.point,
            'price': o.price
        }
        for o in odds
    ])
    
    # Get stats to model
    stats_to_model = STATS.get(player.position)
    if not stats_to_model:
        print(f"❌ No stat model defined for position {player.position}")
        return
    
    print(f"\n📊 Analyzing calibration quality...")
    
    # Analyze each stat
    calibration_results = []
    
    for stat in stats_to_model:
        print(f"  - {stat}...", end=' ')
        
        # Get historical values
        historical_values = [
            getattr(s, stat) for s in historical_stats 
            if getattr(s, stat) is not None
        ]
        
        if not historical_values:
            print("no data, skipping")
            continue
        
        # Fit distribution
        stat_dist = fit_stat_distribution(odds_df, historical_values, stat)
        
        if not stat_dist.odds_points:
            print("no odds points, skipping")
            continue
        
        # Get base KDE
        kde_base, support = fit_kde_marginal(historical_values)
        
        # Compute base CDF at odds points
        grid = np.linspace(support[0], support[1], 1024)
        base_cdf = _numerical_cdf_from_kde(kde_base, grid)
        
        # Compute calibrated CDF at odds points
        calib_cdf = stat_dist.cdf_values
        calib_grid = stat_dist.cdf_grid
        
        # Evaluate at odds points
        odds_x = np.array([pt for pt, _ in stat_dist.odds_points])
        odds_target_cdf = np.array([cdf for _, cdf in stat_dist.odds_points])
        
        base_pred = np.interp(odds_x, grid, base_cdf)
        calib_pred = np.interp(odds_x, calib_grid, calib_cdf)
        
        # Calculate errors
        base_error = np.abs(base_pred - odds_target_cdf)
        calib_error = np.abs(calib_pred - odds_target_cdf)
        
        calibration_results.append({
            'stat': stat,
            'stat_dist': stat_dist,
            'odds_x': odds_x,
            'odds_target_cdf': odds_target_cdf,
            'base_pred': base_pred,
            'calib_pred': calib_pred,
            'base_error': base_error,
            'calib_error': calib_error,
            'base_mae': base_error.mean(),
            'calib_mae': calib_error.mean(),
            'improvement': (base_error.mean() - calib_error.mean()) / base_error.mean() * 100
        })
        
        print(f"✓ Base MAE: {base_error.mean():.4f}, Calib MAE: {calib_error.mean():.4f} ({calibration_results[-1]['improvement']:.1f}% improvement)")
    
    if not calibration_results:
        print("❌ No calibration results to visualize")
        return
    
    print(f"\n🎨 Creating calibration analysis visualization...")
    
    # Create visualization
    n_stats = len(calibration_results)
    fig = plt.figure(figsize=(18, 4 * n_stats))
    
    gs = gridspec.GridSpec(n_stats, 3, figure=fig, hspace=0.4, wspace=0.3,
                           left=0.05, right=0.95, top=0.94, bottom=0.05)
    
    fig.suptitle(f'Odds Calibration Quality Analysis: {player.name} ({player.position})',
                 fontsize=16, fontweight='bold', y=0.98)
    
    for i, result in enumerate(calibration_results):
        stat = result['stat']
        
        # Column 1: Predicted vs Target CDF
        ax1 = fig.add_subplot(gs[i, 0])
        
        ax1.scatter(result['odds_target_cdf'], result['base_pred'], 
                   s=80, alpha=0.6, color='blue', label='Base KDE', edgecolors='black')
        ax1.scatter(result['odds_target_cdf'], result['calib_pred'], 
                   s=80, alpha=0.6, color='green', label='Calibrated', edgecolors='black')
        
        # Perfect calibration line
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
        
        ax1.set_xlabel('Target CDF (from odds)', fontsize=10)
        ax1.set_ylabel('Predicted CDF', fontsize=10)
        ax1.set_title(f'{stat.replace("_", " ").title()}\nCalibration Accuracy', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        
        # Column 2: Error by odds point
        ax2 = fig.add_subplot(gs[i, 1])
        
        x_pos = np.arange(len(result['odds_x']))
        width = 0.35
        
        ax2.bar(x_pos - width/2, result['base_error'], width, 
               label='Base KDE', color='blue', alpha=0.6, edgecolor='black')
        ax2.bar(x_pos + width/2, result['calib_error'], width, 
               label='Calibrated', color='green', alpha=0.6, edgecolor='black')
        
        ax2.set_xlabel('Odds Point', fontsize=10)
        ax2.set_ylabel('Absolute Error', fontsize=10)
        ax2.set_title(f'Absolute Error by Odds Point\nBase MAE: {result["base_mae"]:.4f}, Calib MAE: {result["calib_mae"]:.4f}', 
                     fontsize=11)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{x:.1f}' for x in result['odds_x']], fontsize=8)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        # Column 3: Odds points on stat scale
        ax3 = fig.add_subplot(gs[i, 2])
        
        # Sort by x value
        sorted_idx = np.argsort(result['odds_x'])
        x_sorted = result['odds_x'][sorted_idx]
        target_sorted = result['odds_target_cdf'][sorted_idx]
        base_sorted = result['base_pred'][sorted_idx]
        calib_sorted = result['calib_pred'][sorted_idx]
        
        ax3.plot(x_sorted, target_sorted, 'ro-', linewidth=2, markersize=8,
                label='Target (from odds)', alpha=0.8)
        ax3.plot(x_sorted, base_sorted, 'b^--', linewidth=2, markersize=6,
                label='Base KDE', alpha=0.7)
        ax3.plot(x_sorted, calib_sorted, 'gs-', linewidth=2, markersize=6,
                label='Calibrated', alpha=0.7)
        
        ax3.set_xlabel(f'{stat.replace("_", " ").title()} Value', fontsize=10)
        ax3.set_ylabel('CDF', fontsize=10)
        ax3.set_title(f'CDF Curves\nImprovement: {result["improvement"]:.1f}%', fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.set_ylim(-0.05, 1.05)
    
    # Add summary statistics
    avg_base_mae = np.mean([r['base_mae'] for r in calibration_results])
    avg_calib_mae = np.mean([r['calib_mae'] for r in calibration_results])
    avg_improvement = np.mean([r['improvement'] for r in calibration_results])
    
    summary_text = (
        f"Summary Statistics:\n"
        f"  Average Base MAE: {avg_base_mae:.4f}\n"
        f"  Average Calibrated MAE: {avg_calib_mae:.4f}\n"
        f"  Average Improvement: {avg_improvement:.1f}%\n"
        f"  Total Stats: {n_stats}\n"
        f"  Total Odds Points: {sum(len(r['odds_x']) for r in calibration_results)}"
    )
    
    fig.text(0.02, 0.02, summary_text, 
             fontsize=9, va='bottom', ha='left',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
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
            temp_output = f"temp_calibration_{player.name.replace(' ', '_')}.png"
            plt.savefig(temp_output, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {temp_output}")
        else:
            print(f"✓ Displaying interactive plot...")
            plt.show()
    
    plt.close()
    print(f"\n✅ Calibration analysis complete!")
    
    # Print summary table
    print(f"\n📊 Summary Table:")
    df = pd.DataFrame([
        {
            'Stat': r['stat'],
            'Odds Points': len(r['odds_x']),
            'Base MAE': f"{r['base_mae']:.4f}",
            'Calib MAE': f"{r['calib_mae']:.4f}",
            'Improvement': f"{r['improvement']:.1f}%"
        }
        for r in calibration_results
    ])
    print(df.to_string(index=False))


def batch_analyze_position(position, top_n=10, output_dir=None):
    """
    Analyze calibration quality for top N players at a position.
    """
    engine = init_db('nfl')
    session = get_session(engine)
    
    print(f"🏈 Analyzing top {top_n} {position}s by salary...")
    
    players = session.query(NFLPlayer).filter(
        NFLPlayer.position == position
    ).order_by(NFLPlayer.salary.desc()).limit(top_n).all()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for player in players:
        print(f"\n{'='*60}")
        output_path = output_dir / f"{player.name.replace(' ', '_')}_calibration.png" if output_dir else None
        analyze_calibration_quality(player.name, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze NFL odds calibration quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Patrick Mahomes"
  %(prog)s "Christian McCaffrey" --output cmac_calibration.png
  %(prog)s --position QB --top 10 --output-dir calibration_analysis/
        """
    )
    
    parser.add_argument('player_name', nargs='?', type=str, default=None,
                        help='Name of the NFL player to analyze')
    parser.add_argument('--position', type=str, default=None,
                        help='Position to analyze (e.g., QB, RB, WR)')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of top players by salary (requires --position)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for single player analysis')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for batch analysis')
    
    args = parser.parse_args()
    
    try:
        if args.position:
            batch_analyze_position(args.position, args.top, args.output_dir)
        elif args.player_name:
            analyze_calibration_quality(args.player_name, args.output)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
