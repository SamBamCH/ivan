# Configure logging
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np
from scipy.stats import rankdata
from scipy.interpolate import interp1d
from datetime import datetime
import pandas as pd

from ivan.sports.nba.models import NBAPlayer, NBAOdds, NBAStat
from sqlalchemy import select, func

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

MARKET_MAP = {
    'points': ('player_points', 'player_points_alternate'),
    'rebounds': ('player_rebounds', 'player_rebounds_alternate'),
    'assists': ('player_assists', 'player_assists_alternate'),
    'fg3m': ('player_threes', 'player_threes_alternate'),
    'steals': ('player_steals', 'player_steals_alternate'),
    'blocks': ('player_blocks', 'player_blocks_alternate'),
}
STATS = ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks']

DK_SCORING = {
    'points': 1.0,
    'fg3m': 0.5,
    'rebounds': 1.25,
    'assists': 1.5,
    'steals': 2.0,
    'blocks': 2.0,
    'turnovers': -0.5,
    'double_double': 1.5,
    'triple_double': 3.0,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _american_to_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def _parse_position(position: str) -> Optional[str]:
    """Parse position string (PG, SG, etc.) into simplified form (G, F, C)."""
    if not position:
        return None
    position = position.strip().upper()

    guard_positions = {'PG', 'SG'}
    forward_positions = {'SF', 'PF'}
    center_positions = {'C'}

    if '/' in position:
        positions = [p.strip() for p in position.split('/')]
        for pos in positions:
            if pos in guard_positions:
                return 'G'
            elif pos in forward_positions:
                return 'F'
            elif pos in center_positions:
                return 'C'
    else:
        if position in guard_positions:
            return 'G'
        elif position in forward_positions:
            return 'F'
        elif position in center_positions:
            return 'C'

    return None


# ============================================================================
# POSITION-LEVEL STATS (FALLBACK FOR ROOKIES/NO HISTORY)
# ============================================================================

def _get_position_stats(session, position: str, season: str = "2024-25") -> pd.DataFrame:
    """
    Fetch all stats for players in a given position (used as fallback for rookies).
    
    Args:
        session: Database session
        position: Simplified position ('G', 'F', 'C')
        season: Season string
    
    Returns:
        DataFrame with all stats for position, or empty DataFrame if insufficient data
    """
    # Get all players with this position
    stmt = select(NBAPlayer).where(func.lower(NBAPlayer.position).like(f'%{position}%'))
    players = session.execute(stmt).scalars().all()
    
    if not players:
        logger.warning(f"No players found with position matching '{position}'")
        return pd.DataFrame()
    
    player_names = [p.name for p in players]
    
    # Get stats for all these players
    stats_stmt = select(NBAStat).where(
        (func.lower(func.trim(NBAStat.player_name)).in_([func.lower(p.strip()) for p in player_names])) &
        (NBAStat.season == season)
    )
    
    with session.bind.connect() as conn:
        records = pd.read_sql(stats_stmt, conn)
    
    if len(records) == 0:
        logger.warning(f"No historical stats found for any player with position '{position}'")
        return pd.DataFrame()
    
    stat_cols = [s for s in STATS if s in records.columns]
    data = records[stat_cols].dropna()
    
    logger.info(f"Position {position}: Using {len(data)} stat records from {len(records)} games for fallback")
    
    return data


# ============================================================================
# ODDS PROCESSING
# ============================================================================

def _get_player_consensus_odds(session, player_name: str) -> Dict[str, List[Dict]]:
    """
    Fetch all odds for a player and group by (stat, line, side).
    Only keep groups with consensus from 2+ sportsbooks.
    
    Returns: {stat: [{'line', 'side', 'prob'}, ...]}
    """
    stmt = select(NBAOdds).where(NBAOdds.player == player_name)
    odds_records = session.execute(stmt).scalars().all()

    logger.info(f"Found {len(odds_records)} total odds records for player: {player_name}")

    if not odds_records:
        logger.warning(f"No odds records found for player: {player_name}")
        return {}

    odds_by_stat_line_side = {}

    # Group odds by (stat, line, side)
    for record in odds_records:
        stat = None
        for s, (primary, alternate) in MARKET_MAP.items():
            if record.market in (primary, alternate):
                stat = s
                break

        if stat is None:
            logger.debug(f"Skipping unrecognized market: {record.market}")
            continue

        side = record.side.lower()
        key = (stat, record.point, side)
        if key not in odds_by_stat_line_side:
            odds_by_stat_line_side[key] = []

        prob = _american_to_prob(record.price)
        logger.debug(f"  {stat} @ {record.point} {side}: {record.sportsbook}, prob={prob:.4f}")
        odds_by_stat_line_side[key].append({'sportsbook': record.sportsbook, 'prob': prob})

    logger.info(f"Grouped {len(odds_by_stat_line_side)} unique (stat, line, side) combinations")

    # Filter: only keep groups with 2+ books (consensus)
    filtered_odds = {}
    for key, odds_list in odds_by_stat_line_side.items():
        if len(odds_list) >= 2:
            filtered_odds[key] = odds_list
        else:
            stat, line, side = key
            logger.debug(f"Filtered out {stat} @ {line} {side}: only {len(odds_list)} book(s)")

    # Reorganize into {stat: [{'line', 'side', 'prob'}, ...]}
    consensus_by_stat = {}
    for (stat, line, side), odds_list in filtered_odds.items():
        probs = [o['prob'] for o in odds_list]
        avg_prob = np.mean(probs)

        if stat not in consensus_by_stat:
            consensus_by_stat[stat] = []

        consensus_by_stat[stat].append({
            'line': line,
            'side': side,
            'prob': avg_prob,
        })

    return consensus_by_stat


# ============================================================================
# HISTORICAL DATA & CORRELATION - ROBUST EDGE CASE HANDLING
# ============================================================================

def _get_player_all_historical_stats(
        session,
        player_name: str,
        season: str = "2024-25"
) -> pd.DataFrame:
    """
    Fetch all historical stats for a player in one query.
    
    Returns: DataFrame with columns [points, fg3m, rebounds, assists, steals, blocks]
    """
    stmt = select(NBAStat).where(
        (func.lower(func.trim(NBAStat.player_name)) == func.lower(player_name.strip())) &
        (NBAStat.season == season)
    )

    with session.bind.connect() as conn:
        records = pd.read_sql(stmt, conn)

    if len(records) == 0:
        logger.warning(f"No historical data for player {player_name}")
        return pd.DataFrame()

    # Extract only the stats we care about
    stat_cols = [s for s in STATS if s in records.columns]
    data = records[stat_cols].dropna()
    logger.info(f"Player {player_name}: {len(data)} games with complete stat data")

    return data


def _compute_player_correlation_matrix(historical_data: pd.DataFrame) -> np.ndarray:
    """
    Compute rank-order (Spearman) correlation matrix from historical data.
    Robust handling of edge cases: zero variance, NaN values, insufficient data.
    
    KEY: Clean data BEFORE computing correlation to avoid NaN results.
    
    Args:
        historical_data: DataFrame with stat columns
    
    Returns:
        (n_stats, n_stats) correlation matrix
    """
    if len(historical_data) < 3:
        logger.warning(f"Insufficient data ({len(historical_data)} games) for correlation, using identity matrix")
        return np.eye(len(STATS))

    # STEP 1: Clean data BEFORE correlation computation to avoid NaN in results
    cleaned_data = historical_data.copy()
    problematic_cols = []
    
    for col in cleaned_data.columns:
        # Skip if all NaN
        if cleaned_data[col].isna().all():
            logger.debug(f"Dropping {col}: all NaN values")
            problematic_cols.append(col)
            continue
        
        # Skip if zero variance (all same value) - causes NaN in Spearman correlation
        try:
            std_val = cleaned_data[col].std()
            if std_val == 0 or np.isnan(std_val):
                unique_vals = cleaned_data[col].unique()
                logger.debug(f"Dropping {col}: zero variance (unique values: {unique_vals})")
                problematic_cols.append(col)
                continue
        except Exception as e:
            logger.debug(f"Dropping {col}: error computing std: {e}")
            problematic_cols.append(col)
            continue
        
        # Skip if too many NaN (>50% missing)
        nan_pct = cleaned_data[col].isna().sum() / len(cleaned_data)
        if nan_pct > 0.5:
            logger.debug(f"Dropping {col}: too many NaN ({nan_pct*100:.1f}% missing)")
            problematic_cols.append(col)
            continue
    
    if problematic_cols:
        logger.info(f"Removing {len(problematic_cols)} problematic columns: {problematic_cols}")
        cleaned_data = cleaned_data.drop(columns=problematic_cols)
    
    # If we don't have enough columns, fallback to identity
    if len(cleaned_data.columns) < 2:
        logger.warning(f"Insufficient valid stats for correlation (only {len(cleaned_data.columns)} columns left), using identity matrix")
        return np.eye(len(STATS))
    
    # STEP 2: Compute correlation on cleaned data
    try:
        logger.debug(f"Computing correlation from {len(cleaned_data)} games, {len(cleaned_data.columns)} stats")
        # Compute Spearman correlation (rank-based, more robust to outliers)
        corr_matrix = cleaned_data.corr(method='spearman').values
        
        num_nonfinite = np.sum(~np.isfinite(corr_matrix))
        logger.debug(f"Raw correlation matrix: shape={corr_matrix.shape}, non-finite values={num_nonfinite}")
        
        # Double-check for any remaining NaN/Inf and replace
        if num_nonfinite > 0:
            logger.warning(f"Correlation matrix contains {num_nonfinite} NaN/Inf values, replacing with zeros")
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            np.fill_diagonal(corr_matrix, 1.0)
            logger.debug(f"After replacement: non-finite values={np.sum(~np.isfinite(corr_matrix))}")
    
    except Exception as e:
        logger.error(f"Failed to compute correlation: {type(e).__name__}: {e}, using identity matrix")
        return np.eye(len(STATS))
    
    # STEP 3: Ensure positive semi-definite
    try:
        np.linalg.cholesky(corr_matrix)
        logger.debug(f"Correlation matrix is positive semi-definite")
    except np.linalg.LinAlgError:
        logger.warning(f"Correlation matrix not positive semi-definite, applying regularization")
        try:
            # Add small ridge to diagonal to ensure PSD
            corr_matrix += np.eye(corr_matrix.shape[0]) * 0.01
            # Verify it worked
            np.linalg.cholesky(corr_matrix)
            logger.debug(f"Regularization successful")
        except np.linalg.LinAlgError as e:
            logger.error(f"Failed to regularize: {type(e).__name__}: {e}, using identity matrix")
            return np.eye(len(STATS))

    logger.debug(f"Final correlation matrix: shape={corr_matrix.shape}, all finite={np.all(np.isfinite(corr_matrix))}")
    return corr_matrix


# ============================================================================
# STAT DISTRIBUTION BUILDING - FIXED VERSION
# ============================================================================

def _build_stat_distribution_with_constraints(
        historical_values: np.ndarray,
        consensus_odds: List[Dict],
        n_samples: int = 10000
) -> Tuple[float, float]:
    """
    Fit a distribution to historical values, then adjust to satisfy odds constraints.
    
    FIXES:
    1. Use ONLY "over" odds per line (not both over/under as separate constraints)
    2. Blend empirical data with constraints when historical data is insufficient
    3. Clamp extrapolation to avoid unrealistic tail values
    4. Better handling of small sample edge cases
    
    Args:
        historical_values: Array of historical stat values
        consensus_odds: List of {'line', 'side', 'prob'} dicts
        n_samples: Number of samples to use for fitting
    
    Returns:
        (mean, std) of adjusted distribution
    """
    if len(historical_values) == 0:
        logger.warning("No historical values provided, using defaults")
        return 0.0, 1.0

    # Sample from historical distribution
    sampled = np.random.choice(historical_values, size=n_samples, replace=True)
    sorted_samples = np.sort(sampled)
    
    # Compute empirical statistics for blending later
    n_historical = len(historical_values)
    empirical_mean = float(np.mean(sampled))
    empirical_std = float(np.std(sampled, ddof=1))
    insufficient_data = n_historical < 10
    
    if insufficient_data:
        logger.warning(f"Insufficient historical data ({n_historical} games), will blend empirical with constraints")

    # ========================================================================
    # BUILD CONSTRAINTS - ONLY USE "OVER" ODDS
    # ========================================================================
    # Key fix: Each line should have exactly ONE constraint
    # If we see over 1.5 with prob 0.50, it means P(X > 1.5) = 0.50
    # So CDF at 1.5 is 1 - 0.50 = 0.50
    
    constraints_by_line = {}  # line -> probability of exceeding that line
    
    for odds in consensus_odds:
        line = odds['line']
        side = odds['side']
        prob = odds['prob']

        # ONLY use "over" bets; skip "under"
        # This avoids the duplicate constraint problem
        if side.lower() != 'over':
            logger.debug(f"Skipping {side} @ {line}: using only 'over' odds to avoid duplicate constraints")
            continue
        
        # P(X > line) = prob  =>  CDF(line) = 1 - prob
        cdf_at_line = 1.0 - prob
        
        if line in constraints_by_line:
            # If we somehow see the same line twice, average the CDF values
            old_cdf = constraints_by_line[line]
            constraints_by_line[line] = (old_cdf + cdf_at_line) / 2.0
            logger.debug(f"Averaged CDF for line {line}: {constraints_by_line[line]:.4f}")
        else:
            constraints_by_line[line] = cdf_at_line
            logger.debug(f"Added constraint: line {line:.4f} -> CDF {cdf_at_line:.4f}")
    
    if not constraints_by_line:
        logger.debug("No valid constraints found, returning empirical distribution")
        return empirical_mean, empirical_std
    
    # Convert to (quantile, value) pairs
    quantile_points = sorted(constraints_by_line.values())
    value_points = [line for line, _ in sorted([(k, v) for k, v in constraints_by_line.items()], key=lambda x: x[1])]
    
    logger.debug(f"Constraint points: {list(zip(quantile_points, value_points))}")
    
    # ========================================================================
    # ENSURE MONOTONICITY - Better approach
    # ========================================================================
    # Make sure values are strictly increasing
    for i in range(1, len(value_points)):
        if value_points[i] <= value_points[i - 1]:
            # Adjust value to maintain monotonicity with minimal disturbance
            min_gap = (value_points[-1] - value_points[0]) / (len(value_points) - 1) * 0.1
            value_points[i] = max(value_points[i], value_points[i - 1] + min_gap)
            logger.debug(f"Adjusted value_points[{i}] for monotonicity: {value_points[i]:.4f}")
    
    # Skip endpoints - avoid constraints too close to 0 or 1
    valid_constraints = []
    for q, v in zip(quantile_points, value_points):
        if 0.01 < q < 0.99:
            valid_constraints.append((q, v))
        else:
            logger.debug(f"Skipping constraint at quantile {q:.4f}: too close to endpoints")
    
    if not valid_constraints:
        logger.debug("No valid constraints after endpoint filtering, returning empirical")
        return empirical_mean, empirical_std
    
    quantile_points = [q for q, v in valid_constraints]
    value_points = [v for q, v in valid_constraints]
    
    # ========================================================================
    # APPLY QUANTILE TRANSFORMATION WITH CLAMPING
    # ========================================================================
    # Use clamp instead of extrapolate to avoid unrealistic tail values
    f_quantile_to_value = interp1d(
        quantile_points, 
        value_points,
        kind='linear', 
        fill_value=(value_points[0], value_points[-1]),  # Clamp to endpoint values
        bounds_error=False
    )

    sample_quantiles = np.arange(n_samples) / n_samples
    adjusted_sorted = f_quantile_to_value(sample_quantiles)
    
    # Ensure non-negative
    adjusted_sorted = np.maximum(adjusted_sorted, 0)

    # Unsort back to original order
    sorted_indices = np.argsort(sampled)
    adjusted = np.zeros_like(sampled, dtype=float)
    adjusted[sorted_indices] = adjusted_sorted

    mean = float(np.mean(adjusted))
    std = float(np.std(adjusted, ddof=1))
    
    # ========================================================================
    # BLEND WITH EMPIRICAL FOR SMALL SAMPLES
    # ========================================================================
    # If we have very little historical data, trust the empirical distribution more
    if insufficient_data:
        # Weight blend: with 8 games, use ~40% empirical, 60% constraint
        # With 9 games, use ~50% empirical, 50% constraint
        blend_weight = min(n_historical / 20.0, 0.5)
        mean_blended = blend_weight * empirical_mean + (1 - blend_weight) * mean
        std_blended = blend_weight * empirical_std + (1 - blend_weight) * std
        
        logger.info(f"Blended empirical (weight={blend_weight:.2f}): "
                   f"mean {mean:.4f} → {mean_blended:.4f}, "
                   f"std {std:.4f} → {std_blended:.4f}")
        
        mean = mean_blended
        std = std_blended
    
    logger.debug(f"Final adjusted distribution: mean={mean:.4f}, std={std:.4f}")
    
    return mean, std


# ============================================================================
# SAMPLING & CORRELATION
# ============================================================================

def _sample_from_distributions(
        stat_distributions: Dict[str, Tuple[float, float]],
        n_samples: int
) -> np.ndarray:
    """
    Sample from player's stat distributions using appropriate distribution types.
    
    Distribution choices:
    - Poisson: fg3m, rebounds, assists, steals, blocks (count data)
    - Normal: points (continuous)
    
    Args:
        stat_distributions: {stat: (mean, std), ...}
        n_samples: Number of samples
    
    Returns:
        (n_samples, len(STATS)) array of samples
    """
    # Stats that are counts - use Poisson distribution
    POISSON_STATS = {'fg3m', 'rebounds', 'assists', 'steals', 'blocks'}
    
    samples = np.zeros((n_samples, len(STATS)))
    for i, stat in enumerate(STATS):
        if stat in stat_distributions:
            mean, std = stat_distributions[stat]
            
            if stat in POISSON_STATS:
                # Count data: Use Poisson distribution
                # Lambda parameter is the mean
                samples[:, i] = np.random.poisson(lam=mean, size=n_samples)
                logger.debug(f"  {stat}: sampled from Poisson(λ={mean:.4f})")
            else:
                # Continuous data (points): Use Normal distribution
                samples[:, i] = np.random.normal(loc=mean, scale=std, size=n_samples)
                # Ensure non-negative (can't have negative points)
                samples[:, i] = np.maximum(samples[:, i], 0)
                logger.debug(f"  {stat}: sampled from Normal(μ={mean:.4f}, σ={std:.4f})")
        else:
            logger.warning(f"  {stat}: no distribution found, using zeros")
            samples[:, i] = 0

    return samples


def _impose_correlation_structure(
        samples: np.ndarray,
        target_corr: np.ndarray
) -> np.ndarray:
    """
    Reorder samples to match target correlation structure using rank-based method.
    
    Args:
        samples: (n_samples, n_stats) array
        target_corr: (n_stats, n_stats) correlation matrix
    
    Returns:
        (n_samples, n_stats) reordered array
    """
    n_samples, n_stats = samples.shape
    mean = np.zeros(n_stats)
    
    try:
        corr_normal = np.random.multivariate_normal(mean, target_corr, size=n_samples)
    except np.linalg.LinAlgError:
        logger.warning("Failed to generate multivariate normal, using random reordering")
        corr_normal = np.random.randn(n_samples, n_stats)

    target_ranks = np.apply_along_axis(rankdata, 0, corr_normal)
    sample_ranks = np.apply_along_axis(rankdata, 0, samples)

    reordered = np.zeros_like(samples)
    for j in range(n_stats):
        sorted_idx = np.argsort(sample_ranks[:, j])
        sorted_samples = samples[sorted_idx, j]
        target_order = np.argsort(target_ranks[:, j])
        reordered[target_order, j] = sorted_samples

    return reordered


# ============================================================================
# FPTS CALCULATION
# ============================================================================

def _calculate_fpts(samples: np.ndarray) -> np.ndarray:
    """
    Calculate fantasy points for each sample.
    Includes double-double and triple-double bonuses.
    
    Args:
        samples: (n_samples, n_stats) array
    
    Returns:
        (n_samples,) array of FPTS
    """
    fpts = np.zeros(samples.shape[0])
    
    # Base stat scoring
    for i, stat in enumerate(STATS):
        fpts += samples[:, i] * DK_SCORING[stat]

    # Double-double / Triple-double bonuses
    double_stats = (samples >= 10).sum(axis=1)
    fpts += (double_stats >= 3) * DK_SCORING['triple_double']
    fpts += (double_stats >= 2) * DK_SCORING['double_double']

    return fpts


# ============================================================================
# PLAYER-LEVEL PROCESSING
# ============================================================================

def _process_player(
        session,
        player: NBAPlayer,
        position_stats_cache: Dict[str, pd.DataFrame],
        n_samples: int = 10000
) -> Optional[Tuple[float, float]]:
    """
    Process a single player: build stat distributions, sample, apply correlation, calculate FPTS.
    Uses position-level stats as fallback if no individual history.
    
    Args:
        session: Database session
        player: NBAPlayer object
        position_stats_cache: Cached position-level stats
        n_samples: Number of samples
    
    Returns: (fpts_mean, fpts_std) or None if processing fails
    """
    logger.info(f"[PROCESS] Starting player {player.name} (position: {player.position})")

    # Validate and parse player info
    position = _parse_position(player.position)
    if position is None:
        logger.warning(f"[PROCESS] Could not parse position for {player.name}: '{player.position}'")
        return None

    logger.info(f"[PROCESS] Position: {position}")

    # Fetch individual historical data
    historical_data = _get_player_all_historical_stats(session, player.name, season="2024-25")
    
    # If no individual data, use position fallback
    if historical_data.empty:
        logger.info(f"[PROCESS] No individual history for {player.name}, using position fallback ({position})")
        
        # Get or compute position stats
        if position not in position_stats_cache:
            position_stats_cache[position] = _get_position_stats(session, position, season="2024-25")
        
        historical_data = position_stats_cache[position]
        
        if historical_data.empty:
            logger.warning(f"[PROCESS] No individual history AND no position stats available for {player.name}")
            return None
        
        logger.info(f"[PROCESS] Using {len(historical_data)} position-level records for {player.name}")

    # Get consensus odds for this player
    consensus_odds = _get_player_consensus_odds(session, player.name)
    if not consensus_odds:
        logger.warning(f"[DELETE] {player.name}: No consensus odds found - marking for deletion")
        return None

    logger.info(f"[PROCESS] Found consensus odds for {len(consensus_odds)} stats")

    # For each stat, build distribution and adjust with odds
    stat_distributions = {}
    for stat in STATS:
        logger.info(f"[PROCESS] Building distribution for {stat}")
        
        # Get historical values for this stat
        if stat in historical_data.columns:
            historical_values = historical_data[stat].values
        else:
            logger.warning(f"  {stat}: not in historical data")
            historical_values = np.array([])
        
        # Get odds constraints for this stat
        stat_odds = consensus_odds.get(stat, [])
        
        # Build adjusted distribution
        mean, std = _build_stat_distribution_with_constraints(
            historical_values,
            stat_odds,
            n_samples=n_samples
        )
        stat_distributions[stat] = (mean, std)
        logger.info(f"  {stat}: mean={mean:.4f}, std={std:.4f}")

    # Compute correlation matrix from historical data
    logger.info(f"[PROCESS] Computing correlation matrix from historical data")
    player_corr_matrix = _compute_player_correlation_matrix(historical_data)

    # Sample all stats using their distributions
    logger.info(f"[PROCESS] Sampling {n_samples} from distributions")
    samples = _sample_from_distributions(stat_distributions, n_samples)

    # Apply correlation structure
    logger.info(f"[PROCESS] Applying correlation structure")
    try:
        correlated_samples = _impose_correlation_structure(samples, player_corr_matrix)
    except Exception as e:
        logger.error(f"[PROCESS] Failed to impose correlation: {e}")
        correlated_samples = samples

    # Calculate FPTS
    logger.info(f"[PROCESS] Calculating FPTS")
    fpts_samples = _calculate_fpts(correlated_samples)
    logger.debug(f"  FPTS stats: min={fpts_samples.min():.4f}, max={fpts_samples.max():.4f}, "
                 f"mean={fpts_samples.mean():.4f}, std={fpts_samples.std():.4f}")

    # Fit normal distribution
    fpts_mean = float(np.mean(fpts_samples))
    fpts_std = float(np.std(fpts_samples, ddof=1))
    logger.info(f"[PROCESS] Final FPTS: mean={fpts_mean:.4f}, std={fpts_std:.4f}")

    # Store to database
    player.fpts_mean = fpts_mean
    player.fpts_std = fpts_std
    player.fpts_updated_at = datetime.utcnow()
    session.merge(player)
    session.commit()

    logger.info(f"[PROCESS] Completed {player.name}")
    return fpts_mean, fpts_std


# ============================================================================
# PLAYER CLEANUP
# ============================================================================

def _delete_players_without_odds(session) -> List[str]:
    """
    Delete players that don't have any odds (benchwarmers, etc).
    Log all deleted players.
    
    Returns: List of deleted player names
    """
    # Get all unique player names from odds
    odds_stmt = select(NBAOdds.player).distinct()
    players_with_odds = set(session.execute(odds_stmt).scalars().all())
    
    # Get all players from database
    player_stmt = select(NBAPlayer)
    all_players = session.execute(player_stmt).scalars().all()
    
    deleted = []
    for player in all_players:
        if player.name not in players_with_odds:
            deleted.append(player.name)
            logger.info(f"[DELETE] {player.name}: No odds found - deleted from database")
            session.delete(player)
    
    session.commit()
    return deleted


def _delete_out_status_players(session) -> List[str]:
    """
    Delete players with status "OUT".
    Log all deleted players.
    
    Returns: List of deleted player names
    """
    # Get all players with OUT status
    stmt = select(NBAPlayer).where(
        func.upper(NBAPlayer.status) == 'OUT'
    )
    out_players = session.execute(stmt).scalars().all()
    
    deleted = []
    for player in out_players:
        deleted.append(player.name)
        logger.info(f"[DELETE] {player.name}: Status is OUT - deleted from database")
        session.delete(player)
    
    session.commit()
    return deleted


# ============================================================================
# UNMATCHED ODDS TRACKING
# ============================================================================

def _find_unmatched_odds(session) -> List[Dict]:
    """
    Find all odds records that don't match to a player in our database.
    This helps identify name mismatches.
    
    Returns: List of unmatched odds with details
    """
    stmt = select(NBAOdds).distinct()
    odds_records = session.execute(stmt).scalars().all()
    
    # Get all unique player names from our database
    player_stmt = select(NBAPlayer.name).distinct()
    db_players = set(session.execute(player_stmt).scalars().all())
    
    unmatched = []
    for odds in odds_records:
        if odds.player not in db_players:
            unmatched.append({
                'player': odds.player,
                'market': odds.market,
                'line': odds.point,
                'side': odds.side,
                'sportsbook': odds.sportsbook,
                'price': odds.price
            })
    
    return unmatched


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def fit_fpts_distributions(session, n_samples: int = 10000):
    """
    Main function: Process all players and fit FPTS distributions.
    Includes:
    - Deletion of players with OUT status
    - Deletion of players without odds
    - Comprehensive tracking of all deletions and unmatched odds
    """
    logger.info("[FIT] Starting FPTS distribution fitting")
    logger.info("[FIT] Data sources: NBAPlayer, NBAStat, NBAOdds")

    # ========================================================================
    # PHASE 1: CLEANUP - Remove OUT status and no-odds players
    # ========================================================================
    logger.info("[FIT] ===== PHASE 1: CLEANUP =====")
    
    out_deleted = _delete_out_status_players(session)
    logger.info(f"[FIT] Deleted {len(out_deleted)} players with OUT status")
    
    no_odds_deleted = _delete_players_without_odds(session)
    logger.info(f"[FIT] Deleted {len(no_odds_deleted)} players without odds")
    
    # ========================================================================
    # PHASE 2: PROCESSING - Calculate FPTS distributions
    # ========================================================================
    logger.info("[FIT] ===== PHASE 2: PROCESSING =====")
    
    # Get all remaining players
    stmt = select(NBAPlayer).distinct()
    players = session.execute(stmt).scalars().all()
    logger.info(f"[FIT] Found {len(players)} remaining players to process")

    # Cache for position-level stats
    position_stats_cache = {}
    
    # Track results
    processed_players = []
    failed_players = []

    # Process each player
    successful = 0
    failed = 0
    for idx, player in enumerate(players, 1):
        logger.info(f"[FIT] ===== Player {idx}/{len(players)} =====")
        result = _process_player(session, player, position_stats_cache, n_samples)
        if result is not None:
            successful += 1
            processed_players.append(player.name)
        else:
            failed += 1
            failed_players.append(player.name)

    logger.info(f"[FIT] ===== PHASE 2 COMPLETE =====")
    logger.info(f"[FIT] Successful: {successful}, Failed: {failed}")
    
    # ========================================================================
    # PHASE 3: SUMMARY
    # ========================================================================
    logger.info(f"[FIT] ===== PHASE 3: SUMMARY =====")
    
    logger.info(f"[FIT] Players deleted (OUT status): {len(out_deleted)}")
    if out_deleted:
        for player in sorted(out_deleted):
            logger.info(f"[FIT]   - {player}")
    
    logger.info(f"[FIT] Players deleted (no odds): {len(no_odds_deleted)}")
    if no_odds_deleted:
        for player in sorted(no_odds_deleted)[:20]:  # Show first 20
            logger.info(f"[FIT]   - {player}")
        if len(no_odds_deleted) > 20:
            logger.info(f"[FIT]   ... and {len(no_odds_deleted) - 20} more")
    
    logger.info(f"[FIT] Players processed successfully: {successful}")
    
    logger.info(f"[FIT] Players failed to process: {failed}")
    if failed_players:
        for player in sorted(failed_players):
            logger.info(f"[FIT]   - {player}")
    
    # ========================================================================
    # UNMATCHED ODDS - Indicates potential name mismatches
    # ========================================================================
    logger.info(f"[FIT] Checking for unmatched odds...")
    unmatched_odds = _find_unmatched_odds(session)
    
    if unmatched_odds:
        logger.warning(f"[FIT] ===== UNMATCHED ODDS ({len(unmatched_odds)}) - POTENTIAL NAME MISMATCHES =====")
        
        # Group by player
        by_player = {}
        for odd in unmatched_odds:
            player = odd['player']
            if player not in by_player:
                by_player[player] = []
            by_player[player].append(odd)
        
        # Log unmatched odds grouped by player
        for player_name in sorted(by_player.keys())[:20]:  # Show first 20
            odds_for_player = by_player[player_name]
            logger.warning(f"[FIT] UNMATCHED: {player_name} ({len(odds_for_player)} odds)")
            for odd in odds_for_player[:2]:  # Show first 2
                logger.warning(f"[FIT]   {odd['market']} @ {odd['line']} {odd['side']} "
                             f"({odd['sportsbook']})")
        
        if len(by_player) > 20:
            logger.warning(f"[FIT] ... and {len(by_player) - 20} more unmatched players")
    else:
        logger.info(f"[FIT] ✓ All odds matched to players")
    
    logger.info(f"[FIT] ===== FIT COMPLETE =====")
    logger.info(f"[FIT] Total processed: {successful}")
    logger.info(f"[FIT] Total deleted: {len(out_deleted) + len(no_odds_deleted)}")
    logger.info(f"[FIT] Total unmatched odds: {len(unmatched_odds)}")