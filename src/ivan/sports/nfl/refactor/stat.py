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


def _american_to_prob_helper(american: float) -> float:
    """Convert American odds to implied probability (with vig)."""
    a = float(american)
    if a >= 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)

def _get_anytime_td_prob_from_odds(odds_df: pd.DataFrame) -> float | None:
    """Extract anytime TD probability from odds_df."""
    if odds_df is None or odds_df.empty:
        return None
    m = odds_df[
        (odds_df['market'].str.contains('anytime', case=False, na=False)) & 
        (odds_df['side'].str.contains('Yes', case=False, na=False))
    ]
    if m.empty:
        return None
    price = m['price'].astype(float).min()
    return _american_to_prob_helper(price)

def _td_col_for_position(position: str) -> str | None:
    """Map position to relevant TD stat column."""
    if position in ('WR', 'TE'):
        return 'receiving_tds'
    if position == 'RB':
        return 'rushing_tds'
    return None

def _fetch_similar_player_stats(
    session: 'Session',
    position: str,
    stat_name: str,
    target_td_prob: float | None,
    base_window: float = 0.05
) -> list[float]:
    """
    Fetch historical stat values from similar players at same position.
    Filters by per-game TD% if target_td_prob provided (from anytime TD odds).
    Returns list of stat values for fitting.
    """
    from ivan.sports.nfl.models import NFLStat
    
    td_col = _td_col_for_position(position)
    q = session.query(NFLStat).filter(NFLStat.position == position).all()
    if not q:
        return []
    
    # Build DataFrame
    rows = []
    for r in q:
        val = getattr(r, stat_name, None)
        if val is None:
            continue
        row = {'value': float(val), 'player_id': getattr(r, 'player_id', None)}
        if td_col:
            td_val = getattr(r, td_col, None)
            row['_had_td'] = 1 if (td_val is not None and float(td_val) > 0) else 0
        rows.append(row)
    
    if not rows:
        return []
    
    df = pd.DataFrame(rows)
    
    # If no target TD prob, return all position values
    if target_td_prob is None or '_had_td' not in df:
        return df['value'].dropna().tolist()
    
    # Compute per-player TD rate and filter
    rates = df.groupby('player_id')['_had_td'].mean().rename('td_rate')
    df = df.merge(rates, on='player_id', how='left')
    
    # Widen window if needed
    for w in (base_window, 0.10, 0.15, 0.25):
        lower = max(0.0, target_td_prob - w)
        upper = min(1.0, target_td_prob + w)
        mask = (df['td_rate'] >= lower) & (df['td_rate'] <= upper)
        filtered = df.loc[mask, 'value'].dropna()
        if len(filtered) >= 50:  # enough samples
            return filtered.tolist()
    
    # Fallback to all position
    return df['value'].dropna().tolist()


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


def _inverse_cdf_numeric(grid: np.ndarray, cdf: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Numerical inverse CDF by interpolation on grid."""
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    return np.interp(u, cdf, grid)


# ============================================================================
#                     STEP 2: EXTRACT ODDS POINTS
# ============================================================================

def _american_to_prob_helper(american: float) -> float:
    """Convert American odds to implied probability (with vig)."""
    a = float(american)
    if a >= 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)

def _get_anytime_td_prob_from_odds(odds_df: pd.DataFrame) -> float | None:
    """Extract anytime TD probability from odds_df."""
    if odds_df is None or odds_df.empty:
        return None
    m = odds_df[
        (odds_df['market'].str.contains('anytime', case=False, na=False)) & 
        (odds_df['side'].str.contains('Yes', case=False, na=False))
    ]
    if m.empty:
        return None
    price = m['price'].astype(float).min()
    return _american_to_prob_helper(price)

def _td_col_for_position(position: str) -> str | None:
    """Map position to relevant TD stat column."""
    if position in ('WR', 'TE'):
        return 'receiving_tds'
    if position == 'RB':
        return 'rushing_tds'
    return None

def _fetch_similar_player_stats(
    session: 'Session',
    position: str,
    stat_name: str,
    target_td_prob: float | None,
    base_window: float = 0.05
) -> list[float]:
    """
    Fetch historical stat values from similar players at same position.
    Filters by per-game TD% if target_td_prob provided (from anytime TD odds).
    Returns list of stat values for fitting.
    """
    from ivan.sports.nfl.models import NFLStat
    
    td_col = _td_col_for_position(position)
    q = session.query(NFLStat).filter(NFLStat.position == position).all()
    if not q:
        return []
    
    # Build DataFrame
    rows = []
    for r in q:
        val = getattr(r, stat_name, None)
        if val is None:
            continue
        row = {'value': float(val), 'player_id': getattr(r, 'player_id', None)}
        if td_col:
            td_val = getattr(r, td_col, None)
            row['_had_td'] = 1 if (td_val is not None and float(td_val) > 0) else 0
        rows.append(row)
    
    if not rows:
        return []
    
    df = pd.DataFrame(rows)
    
    # If no target TD prob, return all position values
    if target_td_prob is None or '_had_td' not in df:
        return df['value'].dropna().tolist()
    
    # Compute per-player TD rate and filter
    rates = df.groupby('player_id')['_had_td'].mean().rename('td_rate')
    df = df.merge(rates, on='player_id', how='left')
    
    # Widen window if needed
    for w in (base_window, 0.10, 0.15, 0.25):
        lower = max(0.0, target_td_prob - w)
        upper = min(1.0, target_td_prob + w)
        mask = (df['td_rate'] >= lower) & (df['td_rate'] <= upper)
        filtered = df.loc[mask, 'value'].dropna()
        if len(filtered) >= 50:  # enough samples
            return filtered.tolist()
    
    # Fallback to all position
    return df['value'].dropna().tolist()

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
    """Build a calibrated inverse CDF for KDE to satisfy odds CDF points.

    Strategy: Use monotone piecewise-linear interpolation to warp the base CDF
    to exactly hit the odds-derived CDF targets.
    
    Method:
      1. Compute base KDE CDF on a fine grid
      2. Create anchor points: (x_odds, p_odds) from betting lines
      3. Add boundary anchors to preserve shape outside odds range
      4. Build piecewise-linear warping function: x_base -> x_calibrated
      5. Apply warping to create calibrated CDF that hits all odds points
    
    This ensures the calibrated distribution EXACTLY matches market probabilities
    at the odds points while smoothly interpolating between them.

    Returns a dict with 'grid' and 'cdf' arrays. Use _inverse_cdf_numeric.
    """
    if not odds_points:
        grid = np.linspace(support[0], support[1], num_grid)
        cdf = _numerical_cdf_from_kde(kde, grid)
        return {"grid": grid, "cdf": cdf}

    # Sort odds points by x value
    odds_points = sorted(odds_points, key=lambda x: x[0])
    x_odds = np.array([pt for pt, _ in odds_points], dtype=float)
    p_odds = np.array([p for _, p in odds_points], dtype=float)
    
    # Build fine grid for base KDE
    grid = np.linspace(support[0], support[1], num_grid)
    base_cdf = _numerical_cdf_from_kde(kde, grid)
    
    # Create anchor points for monotone mapping
    # We need: for each odds point (x_i, p_i), find x_base such that base_CDF(x_base) = p_i
    # Then warp: x_base -> x_i (the odds point)
    
    # Inverse CDF of base distribution at target probabilities
    x_base_at_odds = np.interp(p_odds, base_cdf, grid)
    
    # Build warping function anchor points
    # Format: (x_before_warp, x_after_warp)
    warp_x_from = []
    warp_x_to = []
    
    # Add lower boundary anchor (preserve shape below lowest odds point)
    if x_odds[0] > support[0]:
        # Find a point well below the first odds point
        p_low = max(0.01, p_odds[0] * 0.5)
        x_low_base = np.interp(p_low, base_cdf, grid)
        x_low_target = x_low_base  # No warping in lower tail
        warp_x_from.append(x_low_base)
        warp_x_to.append(x_low_target)
    
    # Add all odds points as anchors
    for x_base, x_target in zip(x_base_at_odds, x_odds):
        warp_x_from.append(x_base)
        warp_x_to.append(x_target)
    
    # Add upper boundary anchor (preserve shape above highest odds point)
    if x_odds[-1] < support[1]:
        # Find a point well above the last odds point
        p_high = min(0.99, p_odds[-1] + (1.0 - p_odds[-1]) * 0.5)
        x_high_base = np.interp(p_high, base_cdf, grid)
        x_high_target = x_high_base  # No warping in upper tail
        warp_x_from.append(x_high_base)
        warp_x_to.append(x_high_target)
    
    warp_x_from = np.array(warp_x_from)
    warp_x_to = np.array(warp_x_to)
    
    # Apply piecewise-linear warping to grid
    # This maps each x in the base grid to its warped location
    warped_grid = np.interp(grid, warp_x_from, warp_x_to)
    
    # The calibrated CDF is the base CDF evaluated at the warped grid
    # This ensures CDF(x_odds[i]) = p_odds[i] by construction
    calibrated_cdf = base_cdf.copy()
    
    return {"grid": warped_grid, "cdf": calibrated_cdf}


# ============================================================================
#                        STEP 4: MAIN ENTRY POINT
# ============================================================================

def fit_stat_distribution(
    odds: pd.DataFrame,
    historical_stat_values: list[float] | np.ndarray,
    stat: str,
    session: 'Session | None' = None,
    position: str | None = None
) -> StatDistribution | None:
    """
    Complete workflow: fit base distribution to history, extract all odds points,
    then adjust distribution to match the odds-derived CDF.
    
    If no odds are available for this stat (odds_df empty or no matching markets),
    and session+position are provided, fetches similar player stats filtered by
    anytime TD probability and fits distribution to those values.
    
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
        
        # If no odds found and we have session+position, use neighbor fallback
        use_neighbor_fallback = (len(odds_points) == 0 and session is not None and position is not None)
    
    # Neighbor fallback: fetch similar player stats instead of player's own history
    if use_neighbor_fallback:
        target_td_prob = _get_anytime_td_prob_from_odds(odds)
        neighbor_values = _fetch_similar_player_stats(session, position, stat, target_td_prob)
        if len(neighbor_values) >= 10:
            hist_data = np.array(neighbor_values, dtype=float)
        elif len(hist_data) < 5:
            # Insufficient data even with neighbors
            return None
    
    # If still too few samples, return None
    if len(hist_data) < 5:
        return None
    
    # Step 2: Fit KDE marginal
    is_count_like = np.allclose(hist_data, np.round(hist_data))
    kde, support = fit_kde_marginal(hist_data, positive_only=not is_count_like)

    # Step 3: Calibrate KDE to odds → build inverse CDF grid
    calib = calibrate_kde_to_odds(kde, support, odds_points)

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
