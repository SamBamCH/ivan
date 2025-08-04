from pathlib import Path
import joblib
import numpy as np
from scipy.stats import norm
import scipy.stats as st
from sqlalchemy.orm import Session

from ivan.sports.nfl.player_level_calculations import (
    position_configs,
    continuous_dists,
    fit_distribution_by_name,
    fit_best_distribution
)
from ivan.sports.nfl.models import NFLPlayer


def load_models(model_dir: Path):
    """
    Load base marginal fits and stacked position copula matrices from disk.
    Returns:
      - marginals: {pos: {stat: {'dist','params',...}}}
      - stacked_copulas: {pos: corr_matrix}
    """
    marginals = joblib.load(model_dir / 'marginals.pkl')
    # load stacked position copulas
    stacked_copulas = {}
    stacked_dir = model_dir / 'stacked_player_copulas'
    if stacked_dir.exists():
        for path in stacked_dir.glob('*_stacked_copula.pkl'):
            pos = path.stem.split('_')[0]
            stacked_copulas[pos] = joblib.load(path)
    return marginals, stacked_copulas


def simulate_player_fpts(
    session: Session,
    player_id: int,
    marginals: dict,
    stacked_copulas: dict,
    n_samples: int = 10000
) -> np.ndarray:
    """
    Simulate fantasy points for one player by sampling correlated stats.
    Uses the position's stacked copula matrix.
    """
    player = session.get(NFLPlayer, player_id)
    pos = player.position
    stats = position_configs[pos]['stats']

    # use only stacked copula for the position
    corr = stacked_copulas.get(pos)
    if corr is None:
        raise ValueError(f"No stacked copula found for position {pos}")

    # sample latent normals
    Z = np.random.multivariate_normal(np.zeros(len(stats)), corr, size=n_samples)
    U = norm.cdf(Z)

    # transform to stat values via inverse marginals
    X = np.zeros_like(U)
    for j, stat in enumerate(stats):
        info = marginals[pos][stat]
        dist_name = info['dist']
        params = info['params']
        dist_fn = getattr(st, dist_name)
        X[:, j] = dist_fn.ppf(U[:, j], *params)

        # compute fpts using configurable weights for each stat
    # default DraftKings scoring weights
    fpts_weights = {
        'passing_yards': 0.04,
        'passing_tds': 4,
        'interceptions': -2,
        'rushing_yards': 0.1,
        'rushing_tds': 6,
        'receptions': 1,
        'receiving_yards': 0.1,
        'receiving_tds': 6
    }
    # ensure stats order aligns with weights
    weights = np.array([fpts_weights.get(stat, 0.0) for stat in stats])
    # dot product across samples
    fpts = X.dot(weights)
    return fpts


def fit_and_store_fpts_distribution(
    session: Session,
    player_id: int,
    fpts_samples: np.ndarray
) -> None:
    """
    Fit a best marginal to the simulated fpts samples and persist
    the distribution name and parameters on the NFLPlayer model.
    """
    # Only continuous dists for fpts
    name, params, _ = fit_best_distribution(
        fpts_samples,
        continuous_dists,
        bins=50,
        plot=False,
        plot_path=None
    )
    player = session.get(NFLPlayer, player_id)
    player.fpts_dist_name = name
    player.fpts_dist_params = params  # expects JSON-serializable tuple
    session.commit()


def generate_and_fit_all(
    session: Session,
    model_dir: Path,
    n_samples: int = 10000
) -> None:
    """
    For every player in NFLPlayer, simulate and fit their fpts distribution.
    """
    marginals, stacked_copulas = load_models(model_dir)
    for player in session.query(NFLPlayer).all():
        fpts = simulate_player_fpts(
            session,
            player.id,
            marginals,
            stacked_copulas,
            n_samples
        )
        fit_and_store_fpts_distribution(session, player.id, fpts) #TODO: DST projections

        
