import json
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import kstest, probplot, norm


def plot_marginal_fit(stat_name: str, values: np.ndarray, dist_info: dict) -> None:
    """
    Histogram of `values` with overlaid fitted PDF, and a QQ-plot.
    `dist_info` should have keys 'dist' (name) and 'params' (tuple).
    """
    dist_name = dist_info['dist']
    params = dist_info['params']
    dist_fn = getattr(st, dist_name)

    # PDF overlay
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_hist, ax_qq = axes

    # histogram
    ax_hist.hist(values, bins=30, density=True, alpha=0.6, edgecolor='k')
    x = np.linspace(values.min(), values.max(), 200)
    ax_hist.plot(x, dist_fn.pdf(x, *params), 'r-', lw=2)
    ax_hist.set_title(f"{stat_name} PDF fit")
    ax_hist.set_xlabel(stat_name)
    ax_hist.set_ylabel("Density")

    # QQ-plot
    probplot(values, dist=dist_fn, sparams=params, plot=ax_qq)
    ax_qq.set_title(f"{stat_name} QQ-plot")

    plt.tight_layout()
    plt.show()


def test_marginal_ks(values: np.ndarray, dist_info: dict) -> float:
    """
    Perform a Kolmogorov–Smirnov test comparing `values` to the fitted distribution.
    Returns the KS p-value.
    """
    dist_name = dist_info['dist']
    params = dist_info['params']
    # kstest using scipy named distribution
    statistic, p_value = kstest(values, dist_name, args=params)
    return p_value


def plot_copula_corr(team: str, team_data: pd.DataFrame) -> None:
    """
    Compute empirical copula correlation among position columns,
    then display as a heatmap.
    """
    # 1) empirical U via rank%
    U = team_data.rank(axis=0, pct=True).clip(1e-6, 1-1e-6)
    # 2) Gaussianize: apply probit to DataFrame, then reconstruct DataFrame
    Z_array = norm.ppf(U.values)
    Z = pd.DataFrame(Z_array, index=U.index, columns=U.columns)
    # 3) compute empirical Pearson corr on Gaussianized scores
    emp_corr = Z.corr()

    # plot
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(emp_corr, vmin=-1, vmax=1, cmap='coolwarm')
    ticks = np.arange(len(emp_corr))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(emp_corr.columns)
    ax.set_yticklabels(emp_corr.index)
    plt.colorbar(cax, ax=ax)
    ax.set_title(f"Empirical Gaussian-Copula Corr: {team}")
    plt.tight_layout()
    plt.show()


def compare_empirical_vs_model_scatter(team: str, team_data: pd.DataFrame, vine) -> None:
    """
    For each pair of positions, scatterplot (U_i, U_j) from data vs samples from `vine`.
    `vine` must implement `.sample(n)` returning a DataFrame.
    """
    # empirical uniforms
    U_emp = team_data.rank(axis=0, pct=True).clip(1e-6, 1-1e-6)
    n = len(U_emp)
    # model uniforms: ensure DataFrame
    U_mod_raw = vine.sample(n)
    if isinstance(U_mod_raw, np.ndarray):
        U_mod = pd.DataFrame(U_mod_raw, columns=U_emp.columns)
    else:
        U_mod = U_mod_raw.copy()

    pairs = list(combinations(team_data.columns, 2))
    num = len(pairs)
    cols = 2
    rows = int(np.ceil(num / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(U_emp[i], U_emp[j], alpha=0.5, label='emp')
        ax.scatter(U_mod[i], U_mod[j], alpha=0.3, label='model')
        ax.set_xlabel(f'U[{i}]')
        ax.set_ylabel(f'U[{j}]')
        ax.set_title(f'{team}: {i} vs {j}')
        ax.legend()

    plt.tight_layout()
    plt.show()


def estimate_tail_dependence(team_data: pd.DataFrame, u: float = 0.95) -> float:
    """
    Estimate average upper-tail dependence across all position pairs.
    Uses empirical uniforms with threshold `u`.
    """
    U = team_data.rank(axis=0, pct=True).clip(1e-6, 1-1e-6)
    pairs = list(combinations(U.columns, 2))
    lambdas = []
    for i, j in pairs:
        ui = U[i] > u
        uj = U[j] > u
        if ui.sum() > 0:
            lambdas.append(((ui & uj).sum()) / ui.sum())
    return float(np.mean(lambdas)) if lambdas else np.nan
