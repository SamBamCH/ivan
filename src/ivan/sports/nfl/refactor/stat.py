"""
Stat distribution fitting with KDE + odds calibration.

Workflow:
1. fit_base_distribution()            - Simple parametric fallback (norm/poisson)
2. fit_kde_marginal()                 - Nonparametric KDE marginal with support controls
3. extract_odds_points()              - Extract all 'over' odds → CDF targets
4. calibrate_kde_to_odds()            - Monotone quantile mapping to hit odds CDFs
5. fit_stat_distribution()            - Main entry point returning sampler config
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import gaussian_kde

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

MARKET_TO_STAT = {
    "passing_yards":            ("player_pass_yds", "player_pass_yds_alternate"),
    "rushing_yards":            ("player_rush_yds", "player_rush_yds_alternate"),
    "receiving_yards":          ("player_reception_yds", "player_reception_yds_alternate"),
    "receptions":               ("player_receptions", "player_receptions_alternate"),

    "passing_tds":              ("player_pass_tds", "player_pass_tds_alternate"),
    "rushing_tds":              ("player_anytime_td", "player_rush_tds_alternate", "player_rush_tds"), #TODO: infer based on anytimeTD odds
    "receiving_tds":            ("player_anytime_td", "player_reception_tds_alternate", "player_reception_tds"), #TODO: infer based on anytimeTD odds

    "passing_interceptions":    ("player_pass_interceptions", "player_pass_interceptions_alternate"),
    "fg_made":                  ("player_field_goals", "player_field_goals_alternate"),
    "pat_made":                 (),  # No direct odds market, derived via historical stats
    "kicker_pts":               ("player_kicking_points", "player_kicking_points_alternate")

}




def _split_params(dist_fn, params: Tuple) -> Tuple[Tuple, float, float | None]:
    """
    Parse distribution params into (shapes, loc, scale_or_none).
    Works for both continuous (with scale) and discrete (no scale).
    """
    params = tuple(params)
    
    if dist_fn == st.poisson:
        # Poisson: (mu,) or (mu, loc)
        if len(params) == 1:
            return (params[0],), 0.0, None
        return (params[0],), float(params[1]), None
    
    # Continuous (norm, gamma, etc.): params = (*shapes, loc, scale)
    if len(params) < 2:
        return params, 0.0, 1.0
    
    shapes = params[:-2]
    loc = float(params[-2])
    scale = float(params[-1])
    return shapes, loc, scale


@dataclass
class StatDistribution:
    """Result of fitting a stat distribution with odds calibration.

    For KDE, we store bandwidth and support plus the calibrated CDF grid.
    """
    stat: str
    dist_name: str                  # 'kde' or parametric name
    params: Tuple[float, ...]       # For parametric fallback
    odds_points: list[Tuple[float, float]]
    num_odds_points: int
    kde_support: Tuple[float, float] | None = None
    kde_bandwidth: float | None = None
    # Quantile calibration: list of (point, target_cdf) used to warp
    quantile_map: list[Tuple[float, float]] | None = None
    # Odds-adjusted CDF grid for direct sampling
    cdf_grid: np.ndarray | None = None
    cdf_values: np.ndarray | None = None


# ============================================================================
#                        STEP 1: BASE DISTRIBUTION
# ============================================================================

def _american_to_prob(american: int | float) -> float:
    """Convert American odds to implied probability (with vig)."""
    a = float(american)
    if a >= 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)


def _devig_two_way(p_over: float, p_under: float) -> Tuple[float, float]:
    """Simple de-vig: normalize over/under to sum to 1."""
    total = p_over + p_under
    if total <= 0:
        return p_over, p_under
    return p_over / total, p_under / total


def fit_base_distribution(
    historical_stat_values: list[float] | np.ndarray,
    stat_name: str = "unknown"
) -> Tuple[str, Tuple[float, ...]]:
    """
    Fit a scipy distribution to historical stat values.
    
    Tries: norm (default for continuous), then gamma (if all positive).
    For discrete/count stats (TDs, receptions with low variance), uses poisson.
    
    Args:
        historical_stat_values: List/array of historical values
        stat_name: Name of stat (for logging/debugging)
        
    Returns:
        (dist_name, params) - name like 'norm' or 'gamma', params tuple
    """
    data = np.asarray(historical_stat_values, dtype=float)
    data = data[np.isfinite(data)]
    
    if len(data) < 2:
        # Degenerate case: return normal with std=1
        return "norm", (float(np.mean(data)) if len(data) == 1 else 0.0, 1.0)
    
    # Try to detect if this is a count variable
    unique_vals = np.unique(np.round(data))
    is_likely_count = (
        len(unique_vals) <= 20 and 
        np.allclose(data, np.round(data)) and 
        (np.min(data) >= -0.5)  # mostly non-negative
    )
    
    if is_likely_count:
        # Use Poisson for count data
        mu = float(np.mean(data))
        return "poisson", (mu,)
    
    # Otherwise fit normal (allows negative support, simple and robust)
    loc, scale = st.norm.fit(data)
    return "norm", (float(loc), float(scale))


# ==========================================================================
#                      KDE MARGINAL + SAMPLER UTILITIES
# ==========================================================================

def fit_kde_marginal(
    historical_stat_values: list[float] | np.ndarray,
    positive_only: bool = False,
    bw_method: str | float | None = "scott"
) -> tuple[gaussian_kde, Tuple[float, float]]:
    """Fit a KDE to historical values and return the KDE and support bounds.

    If positive_only=True, we clamp support at [0, max].
    """
    data = np.asarray(historical_stat_values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        # Fallback tiny jitter
        m = float(np.mean(data) if data.size == 1 else 0.0)
        data = np.array([m - 1e-3, m + 1e-3])
    
    # Check for zero/near-zero variance (all values identical or nearly so)
    if np.std(data) < 1e-9:
        # Add tiny jitter to avoid singular covariance
        m = float(np.mean(data))
        jitter = max(abs(m) * 1e-6, 1e-3)  # relative jitter or absolute minimum
        data = data + np.random.RandomState(42).normal(0, jitter, size=data.shape)
    
    try:
        kde = gaussian_kde(data, bw_method=bw_method)
    except (np.linalg.LinAlgError, ValueError) as e:
        # Singular covariance: fallback to jittered data
        m = float(np.mean(data))
        jitter = max(abs(m) * 0.01, 0.1)  # 1% jitter or 0.1 minimum
        data = data + np.random.RandomState(42).normal(0, jitter, size=data.shape)
        kde = gaussian_kde(data, bw_method=bw_method)
    
    lo = float(np.min(data))
    hi = float(np.max(data))
    if positive_only:
        lo = max(0.0, lo)
    return kde, (lo, hi)


def _numerical_cdf_from_kde(kde: gaussian_kde, grid: np.ndarray) -> np.ndarray:
    """Approximate CDF via numerical integration on grid using KDE PDF values."""
    pdf_vals = kde(grid)
    # simple trapezoidal integration
    cdf = np.cumsum((pdf_vals[:-1] + pdf_vals[1:]) * 0.5 * np.diff(grid))
    cdf = np.concatenate([[0.0], cdf])
    # normalize
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return np.clip(cdf, 0.0, 1.0)


def _grid_mean(grid: np.ndarray, cdf: np.ndarray) -> float:
    """Mean of the distribution described by a (grid, cdf) pair."""
    mids = 0.5 * (grid[1:] + grid[:-1])
    return float(np.sum(mids * np.diff(cdf)))


def _inverse_cdf_numeric(grid: np.ndarray, cdf: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Numerical inverse CDF by interpolation on grid."""
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    return np.interp(u, cdf, grid)


def _get_anytime_td_prob_from_odds(odds_df: pd.DataFrame) -> float | None:
    """Consensus anytime-TD probability: mean of the books' Yes prices with the
    same one-sided vig haircut extract_odds_points applies."""
    if odds_df is None or odds_df.empty:
        return None
    m = odds_df[
        (odds_df['market'].str.contains('anytime', case=False, na=False)) &
        (odds_df['side'].str.contains('Yes', case=False, na=False))
    ]
    if m.empty:
        return None
    probs = m['price'].astype(float).map(_american_to_prob)
    return float(np.clip(probs.mean() * 0.955, 0.0, 1.0))


# Touchdown-type counts are 0 most games with a spike at zero; a KDE warped
# through one anchor at 0.5 misbehaves there. These stats are modelled as a
# Poisson fit to the market anchors instead.
PARAMETRIC_COUNT_STATS = {"rushing_tds", "receiving_tds", "passing_tds", "passing_interceptions"}


def _poisson_mu_from_anchors(odds_points: list[Tuple[float, float]]) -> float | None:
    """Poisson mean whose CDF best matches the market anchors (point, P(X<=point))."""
    pts = [(int(np.floor(x)), float(np.clip(p, 1e-4, 1 - 1e-4))) for x, p in odds_points]
    if not pts:
        return None
    if len(pts) == 1:
        k, p = pts[0]
        if k == 0:
            return float(-np.log(p))          # P(X>=1) = 1 - p
    from scipy.optimize import minimize_scalar
    ks = np.array([k for k, _ in pts]); ps = np.array([p for _, p in pts])
    res = minimize_scalar(lambda mu: float(np.sum((st.poisson.cdf(ks, mu) - ps) ** 2)),
                          bounds=(1e-3, 12.0), method="bounded")
    return float(res.x)


def _discrete_stat_distribution(stat: str, pmf: np.ndarray, odds_points: list[Tuple[float, float]]) -> 'StatDistribution':
    """Step CDF on an integer grid so _inverse_cdf_numeric returns integers.

    For u in [F(k-1), F(k)) the inverse must return k, so the CDF breakpoints
    sit at the jumps and the grid value is held flat across each step.
    """
    cdf = np.cumsum(pmf) / float(np.sum(pmf))
    n = len(pmf)
    delta = 1e-9
    # xp (cdf values) strictly increasing: 0, F0-d, F0, F1-d, F1, ...
    xp = np.empty(2 * n); fp = np.empty(2 * n)
    xp[0::2] = np.concatenate([[0.0], cdf[:-1]])
    xp[1::2] = cdf - delta
    fp[0::2] = np.arange(n, dtype=float)
    fp[1::2] = np.arange(n, dtype=float)
    xp[-1] = 1.0
    xp = np.maximum.accumulate(xp)
    for i in range(1, len(xp)):
        if xp[i] <= xp[i - 1]:
            xp[i] = xp[i - 1] + delta
    return StatDistribution(
        stat=stat, dist_name="kde", params=(), odds_points=odds_points,
        num_odds_points=len(odds_points), kde_support=(0.0, float(n - 1)),
        kde_bandwidth=None, quantile_map=odds_points, cdf_grid=fp, cdf_values=xp,
    )


def _fit_parametric_count(stat: str, odds_points, hist_data: np.ndarray, target_mean: float | None) -> 'StatDistribution | None':
    mu = _poisson_mu_from_anchors(odds_points) if odds_points else None
    if mu is None and target_mean is not None:
        mu = float(target_mean)
    if mu is None and len(hist_data) >= 5:
        mu = float(np.mean(hist_data))
    if mu is None:
        return None
    mu = max(mu, 1e-3)
    kmax = int(max(6, np.ceil(mu + 6 * np.sqrt(mu))))
    return _discrete_stat_distribution(stat, st.poisson.pmf(np.arange(kmax + 1), mu), odds_points)


# Game logs per position, loaded once per process: {position: DataFrame}.
_POSITION_LOGS: dict[str, pd.DataFrame] = {}
# Per player-season means per (position, stat): DataFrame[player_id, season, mean, n].
_SEASON_MEANS: dict[tuple[str, str], pd.DataFrame] = {}

NEIGHBOR_MIN_LOGS = 200      # game logs in a neighbor pool
NEIGHBOR_MIN_SEASONS = 8     # distinct player-seasons in a neighbor pool
NEIGHBOR_MIN_GAMES = 4       # games before a player-season's mean counts as a role


def position_logs(session: 'Session', position: str) -> pd.DataFrame:
    """All NFLStat rows for a position as a frame (cached per process)."""
    if position not in _POSITION_LOGS:
        from ivan.sports.nfl.models import NFLStat
        from sqlalchemy import select
        stmt = select(NFLStat).where(NFLStat.position == position)
        with session.bind.connect() as conn:
            _POSITION_LOGS[position] = pd.read_sql(stmt, conn)
    return _POSITION_LOGS[position]


def clear_neighbor_cache() -> None:
    _POSITION_LOGS.clear()
    _SEASON_MEANS.clear()


def _season_means(session: 'Session', position: str, stat: str) -> pd.DataFrame:
    key = (position, stat)
    if key not in _SEASON_MEANS:
        logs = position_logs(session, position)
        if stat not in logs.columns:
            _SEASON_MEANS[key] = pd.DataFrame(columns=['player_id', 'season', 'mean', 'n'])
        else:
            g = logs.dropna(subset=[stat]).groupby(['player_id', 'season'])[stat]
            sm = g.agg(mean='mean', n='size').reset_index()
            _SEASON_MEANS[key] = sm[sm['n'] >= NEIGHBOR_MIN_GAMES].reset_index(drop=True)
    return _SEASON_MEANS[key]


def neighbor_values_by_mean(
    session: 'Session',
    position: str,
    stat: str,
    target_mean: float,
) -> np.ndarray:
    """Game logs from the player-seasons whose per-game mean of `stat` is
    closest to `target_mean`, then rescaled so the pool mean hits the target.

    A player-season is the unit because it captures a *role*: the pool for a
    44-yard-per-game target is made of backs who averaged about 44 yards over
    a season, complete with their zero-touch and breakout games, so the
    resulting KDE has a realistic spread rather than a shifted starter's.
    """
    sm = _season_means(session, position, stat)
    if sm.empty:
        return np.array([], dtype=float)
    order = (sm['mean'] - float(target_mean)).abs().argsort().values
    logs = position_logs(session, position).dropna(subset=[stat])
    picked: list[np.ndarray] = []
    seasons = 0
    total = 0
    for i in order:
        row = sm.iloc[i]
        vals = logs.loc[(logs['player_id'] == row['player_id']) & (logs['season'] == row['season']), stat].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        picked.append(vals)
        seasons += 1
        total += vals.size
        if total >= NEIGHBOR_MIN_LOGS and seasons >= NEIGHBOR_MIN_SEASONS:
            break
    if not picked:
        return np.array([], dtype=float)
    pool = np.concatenate(picked)
    pool_mean = float(pool.mean())
    if pool_mean > 0 and target_mean >= 0:
        # Nudge the pool onto the target; the clip keeps the shape honest when
        # the target sits outside anything the history contains.
        pool = pool * float(np.clip(target_mean / pool_mean, 0.8, 1.25))
    return pool


# ============================================================================
#                     STEP 2: EXTRACT ODDS POINTS
# ============================================================================


def extract_odds_points(
    filtered_odds: pd.DataFrame
) -> list[Tuple[float, float]]:
    """
    Extract all 'over'/'yes' odds and convert to CDF points.
    
    For each unique 'over'/'yes' point across all books, de-vig the over/under or yes/no odds
    to get the true probability P(X > point).
    
    Args:
        filtered_odds: DataFrame with columns: market, side, point, price
                      (already filtered to single stat, preferably only Over/Yes sides)
        
    Returns:
        List of (point, cdf_value) tuples, where cdf_value = P(X <= point)
        Sorted by point ascending.
        
    Example:
        Input odds: 4.5 receptions over -110, under -110
        => P(over 4.5) = 0.5 (de-vigged)
        => P(X <= 4.5) = 0.5
        => Returns (4.5, 0.5)
        
        Or anytime TD yes +250, no -300
        => P(yes TD) = de-vigged probability
        => P(X <= 0.5) = 1 - P(yes)
        => Returns (0.5, cdf_value)
    """
    if filtered_odds.empty:
        return []
    
    # Ensure lowercase sides
    filtered_odds = filtered_odds.copy()
    filtered_odds['side'] = filtered_odds['side'].str.lower().str.strip()
    
    # Group by point to collect all books at each point
    by_point = {}
    for point in filtered_odds['point'].unique():
        point_data = filtered_odds[filtered_odds['point'] == point]
        sides_at_point = {}
        
        for _, row in point_data.iterrows():
            side = row['side']
            prob = _american_to_prob(row['price'])
            if side not in sides_at_point:
                sides_at_point[side] = []
            sides_at_point[side].append(prob)
        
        # Average within each side across books
        side_probs = {s: np.mean(probs) for s, probs in sides_at_point.items()}
        
        by_point[point] = side_probs
    
    # Extract all 'over'/'yes' points with de-vigged probabilities
    odds_points = []
    
    for point in sorted(by_point.keys()):
        side_probs = by_point[point]
        
        # Handle both Over/Under and Yes/No markets
        if "over" in side_probs and "under" in side_probs:
            # Both sides available: de-vig
            p_over, p_under = _devig_two_way(side_probs["over"], side_probs["under"])
        elif "yes" in side_probs and "no" in side_probs:
            # Yes/No market (e.g., anytime TD): de-vig
            p_over, p_under = _devig_two_way(side_probs["yes"], side_probs["no"])
        elif "over" in side_probs:
            # Only over: rough de-vig with haircut
            p_over = side_probs["over"] * 0.955
        elif "yes" in side_probs:
            # Only yes: rough de-vig with haircut
            p_over = side_probs["yes"] * 0.955
        else:
            # No over/yes at this point, skip
            continue
        
        # CDF = P(X <= point) = 1 - P(X > point)
        cdf_value = 1.0 - p_over
        odds_points.append((float(point), float(cdf_value)))
    
    return odds_points


# ============================================================================
#                    STEP 3: FIT TO ODDS DISTRIBUTION
# ============================================================================

def calibrate_kde_to_odds(
    kde: gaussian_kde,
    support: Tuple[float, float],
    odds_points: list[Tuple[float, float]],
    num_grid: int = 1024
) -> Dict[str, np.ndarray]:
    """Build a calibrated inverse CDF for the KDE that passes through every
    odds-derived CDF point.

    The base KDE CDF is computed on a fine grid, then the grid is warped by a
    monotone piecewise-linear map so that CDF(x_odds[i]) == p_odds[i] exactly.
    Between odds points the base shape is stretched linearly. Beyond the
    outermost odds points the tails are scaled *proportionally* to the
    outermost anchor: if the market puts this player's median at 0.6x the base
    median, his 90th percentile is 0.6x the base 90th percentile too. (An
    identity tail would keep a starter's upside on a player the market has
    priced as a backup, and can even fold the map back on itself when the base
    quantile sits above the line.)

    Returns a dict with 'grid' and 'cdf' arrays. Use _inverse_cdf_numeric.
    """
    grid = np.linspace(support[0], support[1], num_grid)
    base_cdf = _numerical_cdf_from_kde(kde, grid)
    if not odds_points:
        return {"grid": grid, "cdf": base_cdf}

    pts = sorted(odds_points, key=lambda x: x[0])
    x_odds = np.array([pt for pt, _ in pts], dtype=float)
    p_odds = np.array([p for _, p in pts], dtype=float)
    # Books occasionally quote a lower CDF at a higher line; a CDF must not
    # decrease, so take the running maximum and drop exact repeats.
    p_odds = np.maximum.accumulate(np.clip(p_odds, 1e-4, 1 - 1e-4))
    keep = np.concatenate([[True], np.diff(x_odds) > 0])
    x_odds, p_odds = x_odds[keep], p_odds[keep]

    # Base quantiles that must land on the odds lines
    x_base = np.interp(p_odds, base_cdf, grid)
    x_base = np.maximum.accumulate(x_base)

    eps = 1e-6 * max(1.0, float(support[1] - support[0]))

    def _ratio(num: float, den: float) -> float:
        # Clipped: a base quantile inside a spike at zero would otherwise
        # stretch the tail without limit.
        return float(np.clip(num / den, 0.5, 2.0)) if den > eps and num > 0 else 1.0

    lo_from, hi_from = float(grid[0]), float(grid[-1])
    lo_to = lo_from * _ratio(x_odds[0], x_base[0])
    hi_to = hi_from * _ratio(x_odds[-1], x_base[-1])
    lo_to = min(lo_to, float(x_odds[0]) - eps)
    hi_to = max(hi_to, float(x_odds[-1]) + eps)

    warp_from = np.concatenate([[lo_from], x_base, [hi_from]])
    warp_to = np.concatenate([[lo_to], x_odds, [hi_to]])

    # np.interp needs strictly increasing sample points
    for arr in (warp_from, warp_to):
        for i in range(1, len(arr)):
            if arr[i] <= arr[i - 1]:
                arr[i] = arr[i - 1] + eps

    warped_grid = np.interp(grid, warp_from, warp_to)
    return {"grid": warped_grid, "cdf": base_cdf.copy()}


# ============================================================================
#                        STEP 4: MAIN ENTRY POINT
# ============================================================================

def fit_stat_distribution(
    odds: pd.DataFrame,
    historical_stat_values: list[float] | np.ndarray,
    stat: str,
    session: 'Session | None' = None,
    position: str | None = None,
    target_mean: float | None = None,
) -> StatDistribution | None:
    """
    Complete workflow: fit base distribution to history, extract all odds points,
    then adjust distribution to match the odds-derived CDF.
    
    If no odds are available for this stat (odds_df empty or no matching markets),
    the player's own history is NOT used for the centre: it reflects whatever
    role he had last season, on whatever team. Instead `target_mean` (the
    salary- and TD-odds-anchored expectation from refactor.volume) selects a
    neighbor pool of same-position player-seasons with that role, and the KDE
    is fit to their game logs. Without a target the fitter falls back to the
    player's own history if he has enough of it.
    
    Returns None if insufficient data for fitting.
    
    Special case for pat_made: No direct odds used — we rely solely on the
    base KDE fitted to historical values (i.e., no odds calibration).
    
    Strategy:
    1. Fit base scipy distribution to historical values
    2. Extract all 'over' odds points across all books/prices
    3. Use optimization to adjust distribution location/scale to fit all odds points
    4. Return distribution that balances historical data and odds constraints
    
    Args:
        odds: DataFrame with columns: market, side, point, price
        historical_stat_values: List of historical values for this stat
        stat: Stat name (e.g., 'passing_yards')
        session: Optional DB session for neighbor lookup when odds missing
        position: Player position for neighbor lookup
        target_mean: Expected per-game mean for this stat when no market prices it
        
    Returns:
        StatDistribution with fitted dist that matches both history and all odds points,
        or None if insufficient data
    """
    # Convert to numpy array for consistency
    hist_data = np.asarray(historical_stat_values, dtype=float)
    hist_data = hist_data[np.isfinite(hist_data)]
    
    # Special handling for pat_made: ignore odds, use base KDE CDF
    if stat == 'pat_made':
        odds_points = []
        use_neighbor_fallback = False
    else:
        # Standard flow: extract odds for this stat
        valid_markets = MARKET_TO_STAT.get(stat)
        if valid_markets is None:
            print(f'no markets found for {stat}')
            return None
        
        filtered_odds = odds[odds['market'].isin(valid_markets)] if not odds.empty else pd.DataFrame()
        odds_points = extract_odds_points(filtered_odds)
        
        # No market for this stat: anchor on the salary/TD-odds target instead
        use_neighbor_fallback = (
            len(odds_points) == 0 and target_mean is not None
            and session is not None and position is not None
        )

    if stat in PARAMETRIC_COUNT_STATS:
        return _fit_parametric_count(stat, odds_points, hist_data, target_mean)

    if use_neighbor_fallback:
        neighbor_values = neighbor_values_by_mean(session, position, stat, float(target_mean))
        if len(neighbor_values) >= 10:
            hist_data = neighbor_values
        elif len(hist_data) < 5:
            return None
    elif odds_points and len(hist_data) < 5 and session is not None and position is not None:
        # Priced stat but no usable history (rookie, or a name the stats table
        # spells differently): the lines fix the centre, so borrow the shape
        # from player-seasons whose mean sits at the market's main line.
        main_line = min(odds_points, key=lambda pt: abs(pt[1] - 0.5))[0]
        neighbor_values = neighbor_values_by_mean(session, position, stat, float(main_line))
        if len(neighbor_values) >= 10:
            hist_data = neighbor_values
    
    # If still too few samples, return None
    if len(hist_data) < 5:
        return None
    
    # Step 2: Fit KDE marginal
    is_count_like = np.allclose(hist_data, np.round(hist_data))
    kde, support = fit_kde_marginal(hist_data, positive_only=not is_count_like)

    # Step 3: Calibrate KDE to odds → build inverse CDF grid
    calib = calibrate_kde_to_odds(kde, support, odds_points)

    if use_neighbor_fallback and target_mean is not None and target_mean > 0:
        # A KDE smooths past the data and is truncated at zero, which lifts the
        # mean of a right-skewed pool by 10-50%. Nothing prices this stat, so
        # the target *is* the centre: rescale the grid onto it.
        got = _grid_mean(calib["grid"], calib["cdf"])
        if got > 0:
            calib["grid"] = calib["grid"] * (float(target_mean) / got)
            support = (float(calib["grid"][0]), float(calib["grid"][-1]))

    return StatDistribution(
        stat=stat,
        dist_name="kde",
        params=(),
        odds_points=odds_points,
        num_odds_points=len(odds_points),
        kde_support=support,
        kde_bandwidth=float(kde.factor) if hasattr(kde, "factor") else None,
        quantile_map=odds_points,
        cdf_grid=calib["grid"],
        cdf_values=calib["cdf"],
    )

# Sampling helper exposed for player-level dependent sampling
def sample_stat(stat_dist: StatDistribution, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample n values from a fitted StatDistribution (KDE path)."""
    if rng is None:
        rng = np.random.default_rng()
    if stat_dist.dist_name == "kde" and stat_dist.kde_support is not None and stat_dist.cdf_grid is not None and stat_dist.cdf_values is not None:
        # Use the stored, odds-adjusted CDF grid for sampling
        u = rng.uniform(0.0, 1.0, size=n)
        return _inverse_cdf_numeric(stat_dist.cdf_grid, stat_dist.cdf_values, u)
    
    # Fallback parametric sampling
    dist_fn = getattr(st, stat_dist.dist_name)
    shapes, loc, scale = _split_params(dist_fn, stat_dist.params)
    if scale is None:
        return dist_fn.rvs(*shapes, loc=loc, size=n).astype(float)
    return dist_fn.rvs(*shapes, loc=loc, scale=scale, size=n).astype(float)
