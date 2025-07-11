import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from copulas.multivariate import GaussianMultivariate
from sklearn.metrics import r2_score
from scipy.stats import nbinom

from ivan.database.session import init_db, get_session, Base
from ivan.sports.mlb.models import (
    MLBStat,
    SlateKDEModel,
    ContestPayout
)
from ivan.core.utils import project_root

import warnings
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    # or ignore all warnings from the continuous_distns module
    warnings.filterwarnings(
        "ignore",
        module="scipy.stats"
    )


def fit_nbinom_moments(data: np.ndarray):
    """Method‐of‐moments NB fit (r, p)."""
    M = data.mean()
    V = data.var(ddof=1)
    if V <= M:
        r = 1e6
        p = r / (r + M)
    else:
        r = M**2 / (V - M)
        p = r / (r + M)
    return r, p


def build_nb_model(session, min_samples: int = 5, n_bins: int = 4):
    """
    1) Query MLBStat (PA>0) into a DataFrame,
    2) Compute per‐game OBP/SLG, cum_OPS per player,
    3) Drop first (min_samples−1) games, bin cum_OPS into quantiles,
    4) Fit NB on fpts in each bin.
    Returns: (dict mapping pandas.Interval → (r, p), DataFrame with OPS_bin & fpts)
    """
    stats = session.query(MLBStat).filter(MLBStat.pa > 0).all()
    rows = []
    for s in stats:
        obp = (s.hits + s.bb + s.hbp) / s.pa if s.pa else np.nan
        tb  = s.hits + s.doubles + 2*s.triples + 3*s.hr
        slg = tb / s.ab if s.ab else np.nan
        rows.append({
            'player_id': s.player_id,
            'date':      s.date.date() if hasattr(s.date, 'date') else s.date,
            'obp':       obp,
            'slg':       slg,
            'fpts':      s.dk_fpts
        })

    df = pd.DataFrame(rows).dropna(subset=['obp','slg'])
    df = df.sort_values(['player_id','date'])

    def _cum_ops(grp):
        grp = grp.copy()
        grp['cum_OBP'] = grp['obp'].expanding().mean()
        grp['cum_SLG'] = grp['slg'].expanding().mean()
        grp['cum_OPS'] = grp['cum_OBP'] + grp['cum_SLG']
        grp.loc[grp.index[:min_samples-1], 'cum_OPS'] = np.nan
        return grp

    df = df.groupby('player_id', group_keys=False).apply(_cum_ops)
    df = df.dropna(subset=['cum_OPS']).reset_index(drop=True)
    df['OPS_bin'] = pd.qcut(df['cum_OPS'], q=n_bins)

    model = {}
    for ops_bin, sub in df.groupby('OPS_bin'):
        fpts = sub['fpts'].to_numpy()
        if len(fpts) < min_samples:
            continue
        model[ops_bin] = fit_nbinom_moments(fpts)

    return model, df


def build_team_copulas(session, min_games_per_team: int = 5):
    """
    1) Query MLBStat (PA>0) into a DataFrame,
    2) For each team/date: pick top‐9 by PA, assign slots 1–9, collect fpts,
    3) Drop dates with <9 batters or <min_games_per_team total dates,
    4) Fit a StandardScaler + GaussianMultivariate copula.
    Returns: (dict team→copula, dict team→scaler, dict team→raw slot DataFrame)
    """
    stats = session.query(MLBStat).filter(MLBStat.pa > 0).all()
    rows = [{
        'team':   s.team,
        'date':   s.date.date() if hasattr(s.date, 'date') else s.date,
        'pa':     s.pa,
        'fpts':   s.dk_fpts
    } for s in stats]
    df = pd.DataFrame(rows)

    team_matrices = {}
    copulas = {}
    scalers = {}

    for team, tdf in df.groupby('team'):
        games = []
        for gdate, g in tdf.groupby('date'):
            top9 = g.nlargest(9, 'pa')
            if len(top9) < 9:
                continue
            top9 = top9.sort_values('pa', ascending=False).copy()
            top9['slot'] = range(1,10)
            mat_row = top9.set_index('slot')['fpts'].sort_index().to_list()
            games.append(mat_row)

        if len(games) < min_games_per_team:
            continue

        mat = pd.DataFrame(games, columns=[f"slot_{i}" for i in range(1,10)])
        scaler = StandardScaler().fit(mat)
        scaled = scaler.transform(mat)
        copula = GaussianMultivariate()
        copula.fit(pd.DataFrame(scaled, columns=mat.columns))

        team_matrices[team] = mat
        scalers[team] = scaler
        copulas[team] = copula

    return copulas, scalers, team_matrices


def plot_nb_diagnostics(nb_model, df):
    """Overlay histogram of fpts & fitted NB pmf for each OPS bin."""
    for ops_bin, (r, p) in nb_model.items():
        sub = df[df['OPS_bin'] == ops_bin]
        xmax = int(sub['fpts'].max()) + 1
        x = np.arange(xmax)
        pmf = nbinom.pmf(x, r, p)

        plt.figure(figsize=(6, 4))
        plt.hist(sub['fpts'], bins=30, density=True, alpha=0.6, label='empirical')
        plt.plot(x, pmf, lw=2, label=f'NB(r={r:.1f}, p={p:.2f})')
        plt.title(f'NB Fit — OPS Bin {ops_bin}')
        plt.xlabel('DraftKings fpts')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
    plt.show()


def plot_copula_diagnostics(copulas, scalers, team_matrices, n_samples: int = 500):
    """Scatter real vs simulated slot 1 vs 2 for the first team found."""
    for team, copula in copulas.items():
        if copula is None: 
            print(f'{team} has no copula fit')
            continue
        mat = team_matrices[team]
        scaler = scalers[team]

        sim_scaled = copula.sample(n_samples)
        sim = scaler.inverse_transform(sim_scaled)

        plt.figure(figsize=(5, 5))
        plt.scatter(mat['slot_1'], mat['slot_2'], alpha=0.4, label='real')
        plt.scatter(sim[:, 0], sim[:, 1], alpha=0.4, label='sim')
        plt.title(f'Team {team} — Slot1 vs Slot2')
        plt.xlabel('Slot 1 fpts')
        plt.ylabel('Slot 2 fpts')
        plt.legend()
        plt.tight_layout()
        break
    plt.show()
    real_corr = mat[['slot_1','slot_2']].corr().iloc[0,1]
    sim_corr  = pd.DataFrame(sim, columns=mat.columns)[['slot_1','slot_2']].corr().iloc[0,1]
    print(f'historical corr: {real_corr}, simulated corr: {sim_corr}')
    
BINS = 20
class NormalBinRegressor:
    """Predicts the same normal PDF over fixed bins for every X."""
    def __init__(self, mean: float, std: float, bin_edges: np.ndarray):
        self.mean = mean
        self.std = std
        # compute bin centers once
        self.centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # compute a Gaussian PDF at each bin center
        pdf = np.exp(-0.5 * ((self.centers - self.mean) / self.std) ** 2)
        pdf /= (self.std * np.sqrt(2 * np.pi))
        # normalize to sum=1
        pdf = pdf / pdf.sum()
        # repeat for each sample
        return np.tile(pdf, (X.shape[0], 1))
    
class ConstantRegressor:
    """A regressor that always predicts the same vector y regardless of input X."""
    def __init__(self, y):
        self.y = np.asarray(y)

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # Repeat the constant vector for each row in X
        n = X.shape[0]
        return np.tile(self.y.reshape(1, -1), (n, 1))


def build_slate_bin_regressors(session):
    """
    Fit regressors mapping slate features to fixed-width bin masses for
    ownership and projection, plus stack-type probabilities.
    Returns dict of trained models, scaler, bin_edges, and stack_types.
    """
    # Query historical KDE models
    recs = session.query(SlateKDEModel).all()
    bin_edges = np.linspace(0.0, 1.0, BINS + 1)

    # Fallback when no historical data is available
    if not recs:
        print("⚠️ No historical data: using fallback models")
        # Identity scaler on 9 features
        scaler = StandardScaler().fit(np.zeros((1, 9)))
        # Uniform ownership distribution
        uniform = np.ones(BINS) / BINS
        own_model = ConstantRegressor(uniform)
        # Projection as Normal(μ=0.85, σ=0.1)
        proj_model = NormalBinRegressor(mean=0.85, std=0.1, bin_edges=bin_edges)
        # No stacks
        stack_model = ConstantRegressor(np.zeros((0,)))
        stack_types = []
        return {
            'scaler':      scaler,
            'own_model':   own_model,
            'proj_model':  proj_model,
            'stack_model': stack_model,
            'stack_types': stack_types,
            'bin_edges':   bin_edges
        }

    # --- Real-data branch ---
    # 1) Prepare feature matrix X
    X = np.array([
        [
            r.num_games, r.total_entries, r.max_entries, r.entry_fee,
            r.top_owned, r.top_5_owned, r.top_10_owned,
            r.first_pct, r.top5_pct
        ] for r in recs
    ], dtype=float)

    # 2) Compute normalized bin counts for ownership and projection histograms
    Y_own = np.zeros((len(recs), BINS))
    Y_proj = np.zeros((len(recs), BINS))
    for i, r in enumerate(recs):
        counts_own = np.array(r.hist_own_counts, dtype=float)
        counts_proj = np.array(r.hist_proj_counts, dtype=float)
        if counts_own.sum() > 0:
            Y_own[i] = counts_own / counts_own.sum()
        if counts_proj.sum() > 0:
            Y_proj[i] = counts_proj / counts_proj.sum()

    # 3) Stack-type probabilities
    all_stacks = sorted({st for r in recs for st in r.hist_stack_types})
    S = len(all_stacks)
    Y_stack = np.zeros((len(recs), S), dtype=float)
    for i, r in enumerate(recs):
        pm = dict(zip(r.hist_stack_types, r.hist_stack_probs))
        for j, st in enumerate(all_stacks):
            Y_stack[i, j] = pm.get(st, 0.0)

    # 4) Standardize features
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 5) Train multi-output regressors
    own_model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, random_state=0)
    ).fit(Xs, Y_own)
    proj_model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, random_state=0)
    ).fit(Xs, Y_proj)
    stack_model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, random_state=0)
    ).fit(Xs, Y_stack)

    return {
        'scaler':      scaler,
        'own_model':   own_model,
        'proj_model':  proj_model,
        'stack_model': stack_model,
        'stack_types': all_stacks,
        'bin_edges':   bin_edges
    }