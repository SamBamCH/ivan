#!/usr/bin/env python3
"""
Visualization script for Stage 1 Quantile Regression Model

Shows:
1. Predicted vs Actual quantiles for each level
2. Distribution shape predictions for sample contests
3. Error patterns across quantiles
4. Residuals and model calibration
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

def load_models(model_dir: Path):
    """Load all Stage 1 artifacts."""
    quantile_models = pickle.load(open(model_dir / 'showdown_stage1_quantile_models.pkl', 'rb'))
    unique_model = pickle.load(open(model_dir / 'showdown_stage1_unique_model.pkl', 'rb'))
    scaler = pickle.load(open(model_dir / 'showdown_stage1_scaler.pkl', 'rb'))
    feature_cols = pickle.load(open(model_dir / 'showdown_stage1_feature_cols.pkl', 'rb'))
    return quantile_models, unique_model, scaler, feature_cols


def make_predictions(df_contest, quantile_models, scaler, feature_cols):
    """
    Make predictions for all contests in the test set.
    
    Returns:
        predictions: Dict mapping quantile level to arrays of predictions
        actuals: Dict mapping quantile level to arrays of actual values
    """
    X = df_contest[feature_cols].values
    X_scaled = scaler.transform(X)
    
    predictions = {}
    actuals = {}
    
    for quantile_level, model in quantile_models.items():
        predictions[quantile_level] = model.predict(X_scaled)
        # Extract actual values from contest data (should be q05, q10, q25, q50, q75, q90, q95, q99)
        quantile_col = f'q{int(quantile_level*100):02d}'
        if quantile_col in df_contest.columns:
            actuals[quantile_level] = df_contest[quantile_col].values
        else:
            # Try alternative naming
            quantile_pct = int(quantile_level * 100)
            if quantile_pct == 5:
                actuals[quantile_level] = np.percentile(df_contest['median_duplicate_count'].values, quantile_level*100)
    
    return predictions, actuals


def plot_predicted_vs_actual(df_test, quantile_models, scaler, feature_cols):
    """Plot predicted vs actual for each quantile level."""
    X_test = df_test[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    quantile_levels = sorted(quantile_models.keys())
    n_quantiles = len(quantile_levels)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, q in enumerate(quantile_levels):
        model = quantile_models[q]
        y_pred = model.predict(X_test_scaled)
        
        # Get actual values - these are from median_duplicate_count (proxy for quantile value)
        y_actual = df_test['median_duplicate_count'].values
        
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(y_actual, y_pred, alpha=0.6, s=80, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        lim = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
        ax.plot(lim, lim, 'r--', lw=2, label='Perfect')
        
        # Metrics
        mae = np.abs(y_pred - y_actual).mean()
        rmse = np.sqrt(((y_pred - y_actual)**2).mean())
        r2 = 1 - (np.sum((y_actual - y_pred)**2) / np.sum((y_actual - y_actual.mean())**2))
        
        ax.set_xlabel('Actual Duplicate Count', fontsize=10)
        ax.set_ylabel('Predicted Duplicate Count', fontsize=10)
        ax.set_title(f'Q{int(q*100):02d}\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 01_predicted_vs_actual.png")
    plt.close()


def plot_distribution_shapes(df_contest, quantile_models, scaler, feature_cols):
    """
    Show predicted distribution shapes for different contest types.
    """
    X = df_contest[feature_cols].values
    X_scaled = scaler.transform(X)
    
    # Pick 3 representative contests: small, medium, large by total_entries
    indices = [
        np.argsort(df_contest['total_entries'].values)[len(df_contest)//4],      # Small
        np.argsort(df_contest['total_entries'].values)[len(df_contest)//2],      # Medium
        np.argsort(df_contest['total_entries'].values)[-len(df_contest)//4],    # Large
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    quantile_levels = sorted(quantile_models.keys())
    
    for plot_idx, contest_idx in enumerate(indices):
        ax = axes[plot_idx]
        X_single = X_scaled[contest_idx:contest_idx+1]
        
        contest_info = df_contest.iloc[contest_idx]
        
        # Get predictions for all quantiles
        quantile_preds = []
        for q in quantile_levels:
            model = quantile_models[q]
            pred = model.predict(X_single)[0]
            quantile_preds.append(pred)
        
        # Plot the distribution curve
        ax.plot(quantile_levels, quantile_preds, 'o-', linewidth=2.5, markersize=8, 
                label='Predicted Distribution', color='steelblue')
        
        # Add reference line for median
        ax.axhline(y=quantile_preds[3], color='gray', linestyle='--', alpha=0.5, label='Median')
        
        ax.set_xlabel('Quantile Level', fontsize=11)
        ax.set_ylabel('Predicted Duplicate Count', fontsize=11)
        title = (f"Contest Size: {int(contest_info['total_entries']):,}\n"
                f"Entry Fee: ${contest_info['entry_fee']:.2f}\n"
                f"Ownership Gini: {contest_info['ownership_gini']:.3f}")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis
        ax.set_xticks(quantile_levels)
        ax.set_xticklabels([f'{int(q*100):02d}' for q in quantile_levels], rotation=45)
    
    plt.tight_layout()
    plt.savefig('02_distribution_shapes.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 02_distribution_shapes.png")
    plt.close()


def plot_error_analysis(df_test, quantile_models, scaler, feature_cols):
    """Plot error metrics across quantiles."""
    X_test = df_test[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    quantile_levels = sorted(quantile_models.keys())
    
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    
    for q in quantile_levels:
        model = quantile_models[q]
        y_pred = model.predict(X_test_scaled)
        y_actual = df_test['median_duplicate_count'].values
        
        mae = np.abs(y_pred - y_actual).mean()
        rmse = np.sqrt(((y_pred - y_actual)**2).mean())
        r2 = 1 - (np.sum((y_actual - y_pred)**2) / np.sum((y_actual - y_actual.mean())**2))
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # MAE
    axes[0].bar(range(len(quantile_levels)), mae_scores, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xticks(range(len(quantile_levels)))
    axes[0].set_xticklabels([f'Q{int(q*100):02d}' for q in quantile_levels], rotation=45)
    axes[0].set_ylabel('Mean Absolute Error', fontsize=11)
    axes[0].set_title('MAE by Quantile', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # RMSE
    axes[1].bar(range(len(quantile_levels)), rmse_scores, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(quantile_levels)))
    axes[1].set_xticklabels([f'Q{int(q*100):02d}' for q in quantile_levels], rotation=45)
    axes[1].set_ylabel('Root Mean Squared Error', fontsize=11)
    axes[1].set_title('RMSE by Quantile', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # R²
    axes[2].bar(range(len(quantile_levels)), r2_scores, color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[2].set_xticks(range(len(quantile_levels)))
    axes[2].set_xticklabels([f'Q{int(q*100):02d}' for q in quantile_levels], rotation=45)
    axes[2].set_ylabel('R² Score', fontsize=11)
    axes[2].set_title('R² by Quantile', fontsize=12, fontweight='bold')
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('03_error_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 03_error_analysis.png")
    plt.close()


def plot_residuals(df_test, quantile_models, scaler, feature_cols):
    """Plot residuals to check for bias and heteroscedasticity."""
    X_test = df_test[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    quantile_levels = sorted(quantile_models.keys())
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, q in enumerate(quantile_levels):
        model = quantile_models[q]
        y_pred = model.predict(X_test_scaled)
        y_actual = df_test['median_duplicate_count'].values
        residuals = y_actual - y_pred
        
        ax = axes[idx]
        
        # Residuals vs predicted
        ax.scatter(y_pred, residuals, alpha=0.6, s=80, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        # Add bands for ±1 std
        std_resid = np.std(residuals)
        ax.axhline(y=std_resid, color='orange', linestyle=':', alpha=0.5)
        ax.axhline(y=-std_resid, color='orange', linestyle=':', alpha=0.5)
        
        mean_resid = np.mean(residuals)
        ax.set_xlabel('Predicted Value', fontsize=10)
        ax.set_ylabel('Residual', fontsize=10)
        ax.set_title(f'Q{int(q*100):02d}\nMean: {mean_resid:.3f}, Std: {std_resid:.3f}', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_residuals.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 04_residuals.png")
    plt.close()


def plot_feature_impact(df_contest, quantile_models, scaler, feature_cols):
    """Show how each feature affects predictions across quantiles."""
    quantile_levels = sorted(quantile_models.keys())
    
    # Create synthetic contests by varying one feature at a time
    baseline = df_contest.iloc[0:1].copy()
    X_baseline = baseline[feature_cols].values
    X_baseline_scaled = scaler.transform(X_baseline)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for feat_idx, feat_name in enumerate(feature_cols):
        ax = axes[feat_idx]
        
        # Get range of this feature
        feat_min = df_contest[feat_name].min()
        feat_max = df_contest[feat_name].max()
        feat_range = np.linspace(feat_min, feat_max, 5)
        
        for feat_val in feat_range:
            X_test = X_baseline.copy()
            X_test[0, feat_idx] = feat_val
            X_test_scaled = scaler.transform(X_test)
            
            preds = []
            for q in quantile_levels:
                model = quantile_models[q]
                pred = model.predict(X_test_scaled)[0]
                preds.append(pred)
            
            ax.plot(quantile_levels, preds, 'o-', alpha=0.7, 
                   label=f'{feat_name}={feat_val:.1f}', linewidth=2, markersize=6)
        
        ax.set_xlabel('Quantile Level', fontsize=11)
        ax.set_ylabel('Predicted Duplicate Count', fontsize=11)
        ax.set_title(f'Impact of {feat_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(quantile_levels)
        ax.set_xticklabels([f'{int(q*100):02d}' for q in quantile_levels], rotation=45)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('05_feature_impact.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 05_feature_impact.png")
    plt.close()


def main():
    model_dir = Path('data/nfl/models')
    
    print("Loading models...")
    quantile_models, unique_model, scaler, feature_cols = load_models(model_dir)
    print(f"✓ Loaded {len(quantile_models)} quantile models")
    
    # For visualization, we need actual test data
    # Load the contest data that was used in training
    print("\nNote: This script needs the actual test data to visualize.")
    print("To fully visualize, you'll need to:")
    print("  1. Save df_test during training")
    print("  2. Or re-run with test data loaded from database")
    print("\nFor now, creating example visualizations...")
    
    # Create synthetic test data for demonstration
    np.random.seed(42)
    n_test = 21
    
    df_test = pd.DataFrame({
        'total_entries': np.random.uniform(5000, 100000, n_test),
        'entry_fee': np.random.choice([2, 5, 10, 25, 50], n_test),
        'ownership_gini': np.random.uniform(0.3, 0.8, n_test),
        'median_duplicate_count': np.random.uniform(1, 100, n_test),
        'q05': np.random.uniform(0.5, 50, n_test),
        'q10': np.random.uniform(0.5, 50, n_test),
        'q25': np.random.uniform(0.5, 50, n_test),
        'q50': np.random.uniform(0.5, 50, n_test),
        'q75': np.random.uniform(0.5, 50, n_test),
        'q90': np.random.uniform(1, 100, n_test),
        'q95': np.random.uniform(1, 100, n_test),
        'q99': np.random.uniform(10, 400, n_test),
    })
    
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)
    
    plot_predicted_vs_actual(df_test, quantile_models, scaler, feature_cols)
    plot_distribution_shapes(df_test, quantile_models, scaler, feature_cols)
    plot_error_analysis(df_test, quantile_models, scaler, feature_cols)
    plot_residuals(df_test, quantile_models, scaler, feature_cols)
    plot_feature_impact(df_test, quantile_models, scaler, feature_cols)
    
    print("\n" + "="*70)
    print("✓ All visualizations complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  01_predicted_vs_actual.png  - Scatter plots for each quantile")
    print("  02_distribution_shapes.png  - Distribution curves for sample contests")
    print("  03_error_analysis.png       - MAE, RMSE, R² by quantile")
    print("  04_residuals.png            - Residual patterns")
    print("  05_feature_impact.png       - How features affect predictions")


if __name__ == '__main__':
    main()