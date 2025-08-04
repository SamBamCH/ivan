# src/ivan/sports/nfl/model_builder.py
"""
Module for fitting and saving NFL stat marginal distributions and copula models.
Supports vanilla, Tweedie, and pure continuous fits for heavy-zero continuous stats.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import norm
import matplotlib.pyplot as plt

from sqlalchemy.orm import Session
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLStat

# Configuration: per-position filter, thresholds, stats, and explicit stat types
position_configs = {
    'QB': {
        'filter_stat': 'attempts',
        'filter_threshold': 15,
        'stats': [
            'passing_yards', 'passing_tds', 'interceptions',
            'rushing_yards', 'rushing_tds'
        ],
        'stat_types': {
            'passing_yards': 'continuous',
            'passing_tds': 'count',
            'interceptions': 'count',
            'rushing_yards': 'continuous',
            'rushing_tds': 'count'
        }, 
        'overrides': {
            'passing_yards': 'gamma',
            'passing_tds': 'poisson',
            'interceptions': 'poisson',
            'rushing_yards': 'gamma',
            'rushing_tds': 'poisson'
        }
    },
    'RB': {
        'filter_stat': 'carries',
        'filter_threshold': 7,
        'stats': [
            'rushing_yards', 'rushing_tds', 'receptions',
            'receiving_yards', 'receiving_tds'
        ],
        'stat_types': {
            'rushing_yards': 'continuous',
            'rushing_tds': 'count',
            'receptions': 'count',
            'receiving_yards': 'continuous',
            'receiving_tds': 'count'
        },
        'overrides': {
            'rushing_yards': 'lognorm',
            'rushing_tds': 'poisson',
            'receptions': 'poisson',
            'receiving_yards': 'lognorm',
            'receiving_tds': 'poisson'
        }
    },
    'WR': {
        'filter_stat': 'targets',
        'filter_threshold': 2,
        'stats': [
            'receptions', 'receiving_yards', 'receiving_tds'
        ],
        'stat_types': {
            'receptions': 'count',
            'receiving_yards': 'continuous',
            'receiving_tds': 'count'
        },
        'overrides': {
            'receptions': 'poisson',
            'receiving_yards': 'gamma',
            'receiving_tds': 'poisson'
        }
    },
    'TE': {
        'filter_stat': 'targets',
        'filter_threshold': 2,
        'stats': [
            'receptions', 'receiving_yards', 'receiving_tds'
        ],
        'stat_types': {
            'receptions': 'count',
            'receiving_yards': 'continuous',
            'receiving_tds': 'count'
        },
        'overrides': {
            'receptions': 'poisson',
            'receiving_yards': 'gamma',
            'receiving_tds': 'poisson'
        }
    }
}

# Candidate distributions for stat types
continuous_dists = ['lognorm', 'gamma', 'weibull_min', 'expon', 'norm']
count_dists = ['nbinom', 'poisson']


def _discrete_ks(data: np.ndarray, dist, args: tuple) -> float:
    """Compute KS statistic between empirical CDF and discrete model CDF."""
    sorted_data = np.sort(data)
    unique_vals = np.unique(sorted_data)
    ecdf = np.searchsorted(sorted_data, unique_vals, side='right') / len(data)
    cdf_model = dist.cdf(unique_vals, *args)
    return float(np.max(np.abs(ecdf - cdf_model)))


def fit_best_distribution(
    data: np.ndarray,
    dist_names: list[str],
    bins: int = 50,
    plot: bool = False,
    plot_path: Path | None = None
) -> tuple[str | None, tuple | None, dict]:
    """
    Fit candidate distributions; select best via KS.
    Returns (best_name, best_params, fit_info).
    """
    if data.size == 0 or np.unique(data).size < 2:
        if plot and plot_path:
            plt.figure(); plt.title(plot_path.stem)
            plt.savefig(plot_path); plt.close()
        return None, None, {}

    fit_info: dict[str, dict] = {}
    # Count distributions: method-of-moments
    for name in dist_names:
        if name == 'poisson':
            lam = data.mean()
            D = _discrete_ks(data, st.poisson, (lam,))
            fit_info[name] = {'params': (lam,), 'ks_stat': D}
        elif name == 'nbinom':
            mean, var = data.mean(), data.var(ddof=0)
            if var > mean:
                p = mean/var
                r = mean * p/(1-p)
                D = _discrete_ks(data, st.nbinom, (r, p))
                fit_info[name] = {'params': (r, p), 'ks_stat': D}
    # Continuous distributions: MLE on positive-only
    for name in dist_names:
        if name in continuous_dists:
            dist = getattr(st, name)
            pos_data = data[data>0]
            if pos_data.size < 2:
                continue
            try:
                params = dist.fit(pos_data, floc=0)
                D, _ = st.kstest(pos_data, name, args=params)
                fit_info[name] = {'params': params, 'ks_stat': float(D)}
            except Exception:
                continue
    if not fit_info:
        return None, None, {}
    best_name = min(fit_info, key=lambda n: fit_info[n]['ks_stat'])
    best_params = fit_info[best_name]['params']
    # Plot all fits
    if plot and plot_path:
        hist, edges = np.histogram(data, bins=bins, density=True)
        x = np.linspace(edges[0], edges[-1], 200)
        plt.figure(); plt.hist(data, bins=bins, density=True, alpha=0.5, label='Data')
        for name, info in fit_info.items():
            fn = getattr(st, name)
            try:
                y = fn.pdf(x, *info['params']) if hasattr(fn, 'pdf') else fn.pmf(np.round(x), *info['params'])
                plt.plot(x, y, label=f"{name} (KS={info['ks_stat']:.3f})")
            except Exception:
                continue
        plt.title(plot_path.stem); plt.legend()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path); plt.close()
    return best_name, best_params, fit_info


def fit_distribution_by_name(
    data: np.ndarray,
    name: str
) -> tuple[tuple | None, dict]:
    """
    Fit a single distribution by name using appropriate method:
    - Count dists: method-of-moments
    - Continuous dists: MLE on positive-only data
    Returns (params, fit_info)
    """
    fit_info: dict[str, dict] = {}
    if name in count_dists:
        if name == 'poisson':
            lam = data.mean()
            fit_info[name] = {'params': (lam,), 'ks_stat': _discrete_ks(data, st.poisson, (lam,))}
            return (lam,), fit_info
        elif name == 'nbinom':
            mean, var = data.mean(), data.var(ddof=0)
            if var > mean:
                p = mean/var
                r = mean*p/(1-p)
                fit_info[name] = {'params': (r, p), 'ks_stat': _discrete_ks(data, st.nbinom, (r, p))}
                return (r, p), fit_info
    elif name in continuous_dists:
        pos_data = data[data>0]
        if pos_data.size < 2:
            return None, {}
        dist = getattr(st, name)
        params = dist.fit(pos_data, floc=0)
        D, _ = st.kstest(pos_data, name, args=params)
        fit_info[name] = {'params': params, 'ks_stat': float(D)}
        return params, fit_info
    return None, {}

def fit_position_marginals(
    session: Session,
    configs: dict[str, dict],
    model_dir: Path,
    do_plot: bool = False
) -> dict[str, dict]:
    """
    Fit marginals, using manual override where specified.
    Returns: {position: {stat: {'dist', 'params', 'fit_info'}}}.
    """
    results: dict[str, dict] = {}
    for pos, cfg in configs.items():
        df = pd.read_sql(
            session.query(NFLStat)
                   .filter(NFLStat.position_group == pos)
                   .filter(getattr(NFLStat, cfg['filter_stat']) > cfg['filter_threshold'])
                   .statement,
            session.bind
        )
        overrides = cfg.get('overrides', {})
        stats_out: dict[str, dict] = {}
        for stat in cfg['stats']:
            data = df[stat].dropna().to_numpy()
            plot_path = Path(model_dir)/'plots'/pos/f"{stat}.png" if do_plot else None
            if stat in overrides:
                name = overrides[stat]
                params, fit_info = fit_distribution_by_name(data, name)
            else:
                # automatically choose best among list
                stype = cfg['stat_types'][stat]
                dist_list = continuous_dists if stype=='continuous' else count_dists
                name, params, info = fit_best_distribution(
                    data, dist_list, bins=50, plot=do_plot, plot_path=plot_path
                )
                fit_info = info
            stats_out[stat] = {'dist': name, 'params': params, 'fit_info': fit_info}
        results[pos] = stats_out
    return results

def fit_position_copulas(session: Session, configs: dict, marginals: dict) -> dict[str, np.ndarray]:
    """
    Fit Gaussian copula for each position’s stats using pooled games across all players.
    Returns {position: corr_matrix}.
    """
    out = {}
    for pos, cfg in configs.items():
        df = pd.read_sql(
            session.query(NFLStat)
                   .filter(NFLStat.position_group == pos)
                   .filter(getattr(NFLStat, cfg['filter_stat']) > cfg['filter_threshold'])
                   .statement,
            session.bind
        )
        Z_cols = []
        for stat in cfg['stats']:
            info = marginals[pos][stat]
            if not info['dist']:
                continue
            dist_fn = getattr(st, info['dist'])
            frozen = dist_fn(*info['params'])
            # convert to numpy array of floats to avoid pandas extension types
            arr = df[stat].clip(lower=0).to_numpy(dtype=float)
            u = frozen.cdf(arr)
            z = norm.ppf(np.clip(u, 1e-6, 1-1e-6))
            Z_cols.append(z)
        if Z_cols:
            Z = np.vstack(Z_cols).T
            out[pos] = np.corrcoef(Z, rowvar=False)
    return out


def fit_player_copulas(session: Session, configs: dict, marginals: dict, model_dir: Path) -> dict[int, np.ndarray]:
    """
    Fit a Gaussian copula for each player's stats across games.
    Saves individual player matrices.
    Returns {player_id: corr_matrix}.
    """
    player_dir = Path(model_dir) / 'player_copulas'
    player_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    players = (
        session.query(NFLStat.player_id, NFLStat.position_group)
               .filter(NFLStat.position_group.in_(configs.keys()))
               .distinct()
    )
    for player_id, pos in players:
        cfg = configs[pos]
        df = pd.read_sql(
            session.query(NFLStat)
                   .filter(NFLStat.player_id == player_id)
                   .filter(getattr(NFLStat, cfg['filter_stat']) > cfg['filter_threshold'])
                   .statement,
            session.bind
        )
        Z_cols = []
        for stat in cfg['stats']:
            info = marginals[pos][stat]
            if not info['dist']:
                continue
            dist_fn = getattr(st, info['dist'])
            frozen = dist_fn(*info['params'])
            arr = df[stat].clip(lower=0).to_numpy(dtype=float)
            u = frozen.cdf(arr)
            z = norm.ppf(np.clip(u, 1e-6, 1-1e-6))
            Z_cols.append(z)
        if Z_cols:
            Z = np.vstack(Z_cols).T
            corr = np.corrcoef(Z, rowvar=False)
            out[player_id] = corr
            joblib.dump(corr, player_dir / f"player_{player_id}_copula.pkl")
    return out


def fit_stacked_player_copulas(session: Session, configs: dict, marginals: dict, model_dir: Path) -> dict[str, np.ndarray]:
    """
    Stack each player's stat-scores (Z-values) for a position, then fit a single copula.
    Saves one correlation matrix per position.
    Returns {position: corr_matrix}.
    """
    out = {}
    stacked_dir = Path(model_dir) / 'stacked_player_copulas'
    stacked_dir.mkdir(parents=True, exist_ok=True)
    for pos, cfg in configs.items():
        Z_all = []
        players = (
            session.query(NFLStat.player_id)
                   .filter(NFLStat.position_group == pos)
                   .distinct()
        )
        for (player_id,) in players:
            df = pd.read_sql(
                session.query(NFLStat)
                       .filter(NFLStat.player_id == player_id)
                       .filter(getattr(NFLStat, cfg['filter_stat']) > cfg['filter_threshold'])
                       .statement,
                session.bind
            )
            Z_cols = []
            for stat in cfg['stats']:
                info = marginals[pos][stat]
                if not info['dist']:
                    continue
                dist_fn = getattr(st, info['dist'])
                frozen = dist_fn(*info['params'])
                arr = df[stat].clip(lower=0).to_numpy(dtype=float)
                u = frozen.cdf(arr)
                z = norm.ppf(np.clip(u, 1e-6, 1-1e-6))
                Z_cols.append(z)
            if Z_cols:
                Z_player = np.vstack(Z_cols).T
                Z_all.append(Z_player)
        if Z_all:
            Z_stack = np.vstack(Z_all)
            corr = np.corrcoef(Z_stack, rowvar=False)
            out[pos] = corr
            joblib.dump(corr, stacked_dir / f"{pos}_stacked_copula.pkl")
    return out


def build_models(db_url: str, model_dir: Path | str, do_plot: bool = False) -> None:
    """
    Master function: fits and saves marginals + copulas.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    engine = init_db('nfl', db_url)
    session = get_session(engine)

    print('Fitting marginals...')
    marginals = fit_position_marginals(session, position_configs, model_dir, do_plot)
    joblib.dump(marginals, model_dir / 'marginals.pkl')

    print('Fitting position copulas...')
    pos_copulas = fit_position_copulas(session, position_configs, marginals)
    joblib.dump(pos_copulas, model_dir / 'position_copulas.pkl')

    print('Fitting player copulas...')
    fit_player_copulas(session, position_configs, marginals, model_dir)

    print('Fitting stacked player copulas...')
    fit_stacked_player_copulas(session, position_configs, marginals, model_dir)
    print(f'Stacked player copulas saved under {model_dir / 'stacked_player_copulas'}')


