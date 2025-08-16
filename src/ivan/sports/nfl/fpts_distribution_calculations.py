from pathlib import Path
import joblib
import numpy as np
from scipy.stats import norm
import scipy.stats as st
from sqlalchemy.orm import Session
from sqlalchemy import text

from ivan.sports.nfl.player_level_calculations import (
    position_configs,
    continuous_dists,
    fit_distribution_by_name,
    fit_best_distribution
)
from ivan.sports.nfl.models import NFLPlayer


# ------ NEW: DST helpers ------------------------------------------------------

DST_POS = {"DST", "D/ST", "DEF"}

def _fetch_dst_points(session: Session, team: str | None, opponent: str | None) -> np.ndarray:
    """
    Pull historical dst_points from nfl_dst_game.
    Preference:
      1) (team, opponent) exact matchup
      2) team-only
      3) league-wide
    """
    with session.bind.connect() as conn:
        if team and opponent:
            res = conn.execute(
                text("""SELECT dst_points FROM nfl_dst_game
                        WHERE team = :t AND opponent = :o AND dst_points IS NOT NULL"""),
                {"t": team, "o": opponent},
            ).fetchall()
            vals = np.array([r[0] for r in res], dtype=float)
            if vals.size >= 5:
                return vals

        if team:
            res = conn.execute(
                text("""SELECT dst_points FROM nfl_dst_game
                        WHERE team = :t AND dst_points IS NOT NULL"""),
                {"t": team},
            ).fetchall()
            vals = np.array([r[0] for r in res], dtype=float)
            if vals.size >= 5:
                return vals

        res = conn.execute(
            text("SELECT dst_points FROM nfl_dst_game WHERE dst_points IS NOT NULL")
        ).fetchall()
        vals = np.array([r[0] for r in res], dtype=float)
        return vals


def _sample_dst_fpts(session: Session, team: str | None, opponent: str | None, n_samples: int) -> np.ndarray:
    """
    Empirical bootstrap of DST fpts from nfl_dst_game with a safe fallback.
    """
    vals = _fetch_dst_points(session, team, opponent)
    if vals.size > 0:
        # optional tiny jitter helps continuous fitting stay well-behaved
        samples = np.random.choice(vals, size=n_samples, replace=True).astype(float)
        return samples

    # fallback if the table is empty for a new DB
    rng = np.random.default_rng()
    fpts = rng.normal(loc=5.0, scale=4.0, size=n_samples)
    return np.clip(fpts, -6.0, None)


# -----------------------------------------------------------------------------

def load_models(model_dir: Path):
    """
    Load base marginal fits and stacked position copulas from disk.
    Returns:
      - marginals: {pos: {stat: {'dist','params',...}}}
      - stacked_copulas: {pos: corr_matrix}
    """
    marginals = joblib.load(model_dir / 'marginals.pkl')
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
    Simulate fantasy points for one player.
    - Non-DST: sample correlated stats via position's stacked copula + inverse marginals, then score.
    - DST: BYPASS marginals entirely; empirical bootstrap from nfl_dst_game (fpts only).
    """
    player = session.get(NFLPlayer, player_id)
    raw_pos = player.position
    pos = 'DST' if raw_pos in DST_POS else raw_pos

    # ----- DST branch: skip marginals & stacked copulas completely ------------
    if pos == 'DST':
        return _sample_dst_fpts(session, getattr(player, "team", None), getattr(player, "opponent", None), n_samples)

    # ----- Existing logic for skill positions ---------------------------------
    stats = position_configs[pos]['stats']

    # use only stacked copula for the position (fall back to independence if missing)
    corr = stacked_copulas.get(pos)
    if corr is None:
        corr = np.eye(len(stats))

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

    # compute fpts using configurable weights for each stat (DK defaults)
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
    weights = np.array([fpts_weights.get(stat, 0.0) for stat in stats])
    fpts = X.dot(weights)
    return fpts


def fit_and_store_fpts_distribution(
    session: Session,
    player_id: int,
    fpts_samples: np.ndarray
) -> None:
    """
    Fit a best marginal to the simulated fpts samples and persist it.
    (DST included — we fit directly on empirical fpts samples)
    """
    name, params, _ = fit_best_distribution(
        fpts_samples,
        continuous_dists,
        bins=50,
        plot=False,
        plot_path=None
    )
    player = session.get(NFLPlayer, player_id)
    player.fpts_dist_name = name
    player.fpts_dist_params = params
    session.commit()


def generate_and_fit_all(
    session: Session,
    model_dir: Path,
    n_samples: int = 10000
) -> None:
    """
    For every player in NFLPlayer, simulate and fit their fpts distribution.
    - Skill positions: correlated stat sampling (marginals + stacked copula).
    - DST: empirical bootstrap on historical dst_points from nfl_dst_game.
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
        fit_and_store_fpts_distribution(session, player.id, fpts)  # includes DST now
