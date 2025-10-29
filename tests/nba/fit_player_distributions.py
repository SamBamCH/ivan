"""
Fit player-specific fantasy point distributions using odds constraints.

This script:
1. Loads unique player names from NBAPlayer table
2. Fetches odds data for each player
3. Adjusts base marginals by position to match odds probability distributions
4. Reorders samples using historical rank correlations
5. Calculates fpts from reordered samples
6. Fits distribution to fpts and stores parameters in NBAPlayer table

Usage:
    python tests/nba/fit_player_distributions.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar
from sqlalchemy import select, update, func
from tqdm import tqdm

from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import NBAPlayer, NBAOdds, NBAStat


# DraftKings scoring
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

STATS = ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks']
NUM_SAMPLES = 10000

# Market mapping: stat -> (primary_market, alternate_market)
MARKET_MAP = {
    'points': ('player_points', 'player_points_alternate'),
    'rebounds': ('player_rebounds', 'player_rebounds_alternate'),
    'assists': ('player_assists', 'player_assists_alternate'),
    'fg3m': ('player_threes', 'player_threes_alternate'),
    'steals': ('player_steals', 'player_steals_alternate'),
    'blocks': ('player_blocks', 'player_blocks_alternate'),
}


def load_models(model_dir: Path) -> tuple[dict, dict, dict, dict]:
    """
    Load base marginals and correlation matrices for both roles.

    Returns:
        (marginals_starters, marginals_backups, correlations_starters, correlations_backups)
    """
    marginals_starters = joblib.load(model_dir / "marginals_starters.pkl")
    marginals_backups = joblib.load(model_dir / "marginals_backups.pkl")
    correlations_starters = joblib.load(model_dir / "correlations_starters.pkl")
    correlations_backups = joblib.load(model_dir / "correlations_backups.pkl")

    return marginals_starters, marginals_backups, correlations_starters, correlations_backups


def get_player_odds(session, player_name: str) -> dict:
    """
    Fetch all odds for a player from NBAOdds table.

    Returns:
        {stat: [(line, over_prob, under_prob), ...], ...}
    """
    stmt = select(NBAOdds).where(NBAOdds.player == player_name)
    odds_records = session.execute(stmt).scalars().all()

    odds_by_stat = {}

    for record in odds_records:
        # Find which stat this market corresponds to
        stat = None
        for s, (primary, alternate) in MARKET_MAP.items():
            if record.market in (primary, alternate):
                stat = s
                break

        if stat is None:
            continue

        if stat not in odds_by_stat:
            odds_by_stat[stat] = []

        # Convert american odds to probability
        # Assume we have over/under lines
        if record.side == 'over':
            prob = american_to_prob(record.price)
            line = record.point
            # Store as (line, over_prob, under_prob=None) - we'll match pairs later
            odds_by_stat[stat].append((line, prob, None, 'over'))
        elif record.side == 'under':
            prob = american_to_prob(record.price)
            line = record.point
            odds_by_stat[stat].append((line, None, prob, 'under'))

    # Match over/under pairs at same line
    matched_odds = {}
    for stat, odds_list in odds_by_stat.items():
        # Group by line
        lines = {}
        for item in odds_list:
            line = item[0]
            if line not in lines:
                lines[line] = {'over': None, 'under': None}

            if item[3] == 'over':
                lines[line]['over'] = item[1]
            elif item[3] == 'under':
                lines[line]['under'] = item[2]

        # Create matched pairs
        matched = []
        for line, probs in lines.items():
            over_prob = probs['over']
            under_prob = probs['under']

            # If missing one side, assume fair odds
            if over_prob is None and under_prob is not None:
                over_prob = 1.0 - under_prob
            elif under_prob is None and over_prob is not None:
                under_prob = 1.0 - over_prob

            if over_prob is not None and under_prob is not None:
                matched.append((line, over_prob, under_prob))

        if matched:
            matched_odds[stat] = matched

    return matched_odds


def american_to_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def sample_base_marginals(marginals: dict, position: str, n_samples: int) -> np.ndarray:
    """
    Sample from base marginal distributions for a position.

    Returns:
        (n_samples, n_stats) array of independent samples
    """
    pos_marginals = marginals[position]
    samples = np.zeros((n_samples, len(STATS)))

    for i, stat in enumerate(STATS):
        dist_info = pos_marginals[stat]
        dist_name = dist_info['dist']
        params = dist_info['params']

        if dist_name is None:
            # Fallback to zeros
            samples[:, i] = 0
        else:
            dist = getattr(st, dist_name)
            samples[:, i] = dist.rvs(*params, size=n_samples)

    return samples


def adjust_marginal_with_odds(base_samples: np.ndarray, odds_constraints: list) -> np.ndarray:
    """
    Adjust samples to pass through odds constraints via quantile stretching.

    Uses a piecewise linear quantile transformation that:
    1. Preserves sample ordering (monotonic)
    2. Matches implied quantiles from odds probabilities
    3. Smoothly interpolates between constraint points
    4. Handles contradictory constraints by keeping only the most extreme

    Args:
        base_samples: (n_samples,) array from base distribution
        odds_constraints: [(line, over_prob, under_prob), ...]

    Returns:
        (n_samples,) adjusted samples with matched quantiles
    """
    from scipy.interpolate import interp1d

    if not odds_constraints:
        return base_samples.copy()

    sorted_indices = np.argsort(base_samples)
    sorted_samples = base_samples[sorted_indices]
    n_samples = len(base_samples)

    # Build constraint mapping: (quantile -> target_value)
    # Start with endpoints to anchor the transformation
    constraints_dict = {
        0.0: sorted_samples[0],
        1.0: sorted_samples[-1]
    }

    # Add constraints from odds, handling contradictions
    for line, over_prob, under_prob in odds_constraints:
        # under_prob tells us what quantile this line should be at
        target_quantile = under_prob

        # Skip if quantile is already at an endpoint
        if target_quantile <= 0.01 or target_quantile >= 0.99:
            continue

        # For conflicting constraints at same quantile, keep the one
        # that maintains monotonicity
        if target_quantile in constraints_dict:
            existing_value = constraints_dict[target_quantile]
            # Keep the constraint that maintains order relative to neighbors
            # (simpler approach: just keep first one for consistency)
            continue

        # Store this constraint: at target_quantile position, value should be line
        constraints_dict[target_quantile] = line

    # Sort constraints by quantile
    quantile_points = sorted(constraints_dict.keys())
    value_points = [constraints_dict[q] for q in quantile_points]

    # Ensure monotonicity: if values aren't strictly increasing, adjust
    for i in range(1, len(value_points)):
        if value_points[i] <= value_points[i - 1]:
            # Shift violating point slightly
            value_points[i] = value_points[i - 1] + 1e-6

    # Create interpolation function that maps quantile -> value
    f_quantile_to_value = interp1d(quantile_points, value_points, kind='linear',
                                    fill_value='extrapolate')

    # Create new samples by interpolating at each sorted sample's quantile position
    # This creates a smooth, order-preserving transformation
    sample_quantiles = np.arange(n_samples) / n_samples
    adjusted_sorted = f_quantile_to_value(sample_quantiles)

    # Unsort to match original order
    adjusted = np.zeros_like(base_samples)
    adjusted[sorted_indices] = adjusted_sorted

    return adjusted


def impose_correlation_structure(samples: np.ndarray, target_corr: np.ndarray) -> np.ndarray:
    """
    Reorder samples to match target rank correlation structure.

    Uses the Iman-Conover method:
    1. Rank each column independently
    2. Create target multivariate normal with desired correlation
    3. Reorder samples to match ranks of target normal

    Args:
        samples: (n_samples, n_stats) array with marginals already set
        target_corr: (n_stats, n_stats) target Spearman correlation matrix

    Returns:
        (n_samples, n_stats) reordered samples
    """
    n_samples, n_stats = samples.shape

    # Generate correlated normal samples
    mean = np.zeros(n_stats)
    try:
        corr_normal = np.random.multivariate_normal(mean, target_corr, size=n_samples)
    except:
        # If correlation matrix is not positive definite, use identity
        corr_normal = np.random.randn(n_samples, n_stats)

    # Get ranks of correlated normal
    target_ranks = np.apply_along_axis(rankdata, 0, corr_normal)

    # Get ranks of our samples
    sample_ranks = np.apply_along_axis(rankdata, 0, samples)

    # Reorder samples to match target ranks
    reordered = np.zeros_like(samples)
    for j in range(n_stats):
        # Sort samples by their current ranks
        sorted_idx = np.argsort(sample_ranks[:, j])
        sorted_samples = samples[sorted_idx, j]

        # Assign sorted samples to positions defined by target ranks
        target_order = np.argsort(target_ranks[:, j])
        reordered[target_order, j] = sorted_samples

    return reordered


def calculate_fpts(samples: np.ndarray) -> np.ndarray:
    """
    Calculate DraftKings fantasy points from stat samples.

    Args:
        samples: (n_samples, n_stats) array where columns are STATS

    Returns:
        (n_samples,) array of fpts
    """
    fpts = np.zeros(samples.shape[0])

    for i, stat in enumerate(STATS):
        fpts += samples[:, i] * DK_SCORING[stat]

    # Check for double-doubles and triple-doubles
    # For simplicity, approximate: if 2+ stats >= 10, add DD bonus
    # if 3+ stats >= 10, add TD bonus instead
    double_stats = (samples >= 10).sum(axis=1)
    fpts += (double_stats == 2) * DK_SCORING['double_double']
    fpts += (double_stats >= 3) * DK_SCORING['triple_double']

    return fpts


def fit_normal_distribution(fpts_samples: np.ndarray) -> tuple[float, float]:
    """
    Fit normal distribution to fpts samples.

    Returns:
        (mean, std)
    """
    mean = float(np.mean(fpts_samples))
    std = float(np.std(fpts_samples, ddof=1))
    return mean, std


def parse_position(position: str) -> str:
    """
    Parse NBA player position to normalized category.

    Handles:
    - Guard positions (PG, SG) -> 'G'
    - Forward positions (SF, PF) -> 'F'
    - Center (C) -> 'C'
    - Dual positions (e.g., 'SG/SF', 'PF/C') -> tries primary, falls back to secondary

    Args:
        position: Raw position string from database (e.g., 'SG', 'SF/PF', 'C')

    Returns:
        Normalized position string ('G', 'F', or 'C')
    """
    if not position:
        return None

    position = position.strip().upper()

    # Position mapping
    guard_positions = {'PG', 'SG'}
    forward_positions = {'SF', 'PF'}
    center_positions = {'C'}

    # Handle dual positions (e.g., 'SG/SF')
    if '/' in position:
        positions = [p.strip() for p in position.split('/')]
        # Try to find best match, preferring the order
        for pos in positions:
            if pos in guard_positions:
                return 'G'
            elif pos in forward_positions:
                return 'F'
            elif pos in center_positions:
                return 'C'
    else:
        # Single position
        if position in guard_positions:
            return 'G'
        elif position in forward_positions:
            return 'F'
        elif position in center_positions:
            return 'C'

    return None


def get_player_avg_minutes(session, player_name: str) -> float | None:
    """
    Calculate a player's average minutes from their game history.
    Normalized comparison (case-insensitive, trimmed whitespace).

    Returns average minutes, or None if no data found.
    """
    stmt = select(func.avg(NBAStat.minutes)).where(
        func.lower(func.trim(NBAStat.player_name)) == func.lower(player_name.strip())
    )

    try:
        result = session.execute(stmt).scalar()
        return result
    except Exception:
        return None



def detect_player_role(session, player_name: str) -> str:
    """
    Detect if a player is a starter or backup based on average minutes.

    Returns:
        "starters" if avg minutes > 25
        "backups" if avg minutes <= 25 (or no data)
    """
    avg_minutes = get_player_avg_minutes(session, player_name)

    if avg_minutes is None:
        return "backups"  # Default to backups if no history

    if avg_minutes > 25:
        return "starters"
    else:
        return "backups"


def process_player(session, player: NBAPlayer, marginals_starters: dict, marginals_backups: dict,
                   correlations_starters: dict, correlations_backups: dict) -> tuple[float, float] | None:
    """
    Process a single player: get odds, adjust marginals, impose correlations, fit fpts.

    Automatically detects if player is a starter or backup and uses appropriate models.

    Returns:
        (fpts_mean, fpts_std) or None if failed
    """
    position = parse_position(player.position)
    if position is None:
        print(f'could not parse position {player.position} for player {player.name}')
        return None

    # Detect player role and select appropriate models
    role = detect_player_role(session, player.name)

    if role == "starters":
        marginals = marginals_starters
        correlations = correlations_starters
    else:
        marginals = marginals_backups
        correlations = correlations_backups

    if position not in marginals:
        print(f'position {position} (parsed from {player.position}) for player {player.name} not in marginals')
        return None

    # Get odds for this player
    player_odds = get_player_odds(session, player.name)

    if player_odds is None:
        print(f'no odds found for player {player.name}')
        return None

    # Sample base marginals
    base_samples = sample_base_marginals(marginals, position, NUM_SAMPLES)

    if base_samples is None:
        print(f'error generating base samples for player {player.name}')

    # Adjust each stat's marginal using odds constraints
    adjusted_samples = base_samples.copy()
    for i, stat in enumerate(STATS):
        if stat in player_odds:
            adjusted_samples[:, i] = adjust_marginal_with_odds(
                base_samples[:, i],
                player_odds[stat]
            )

    # Impose correlation structure
    target_corr = correlations[position]
    correlated_samples = impose_correlation_structure(adjusted_samples, target_corr)

    # Calculate fpts
    fpts_samples = calculate_fpts(correlated_samples)

    # Fit distribution
    mean, std = fit_normal_distribution(fpts_samples)

    return mean, std


def update_player_distributions(session, player_id: int, fpts_mean: float, fpts_std: float):
    """
    Update NBAPlayer table with fitted distribution parameters.

    Note: You may need to add fpts_mean and fpts_std columns to NBAPlayer model first.
    """
    # For now, just print - you'll need to add these columns to the model
    # stmt = (
    #     update(NBAPlayer)
    #     .where(NBAPlayer.id == player_id)
    #     .values(fpts_mean=fpts_mean, fpts_std=fpts_std)
    # )
    # session.execute(stmt)
    # session.commit()
    pass


def main():
    """Main entry point."""
    # Initialize database
    engine = init_db("nba")
    session = get_session(engine)

    # Load models
    model_dir = Path("/home/sam-pratt/ivan/data/nba/models")
    print("Loading base models...")
    marginals_starters, marginals_backups, correlations_starters, correlations_backups = load_models(model_dir)

    # Get unique players
    stmt = select(NBAPlayer).distinct()
    players = session.execute(stmt).scalars().all()

    print(f"Processing {len(players)} players...")

    results = []
    for player in tqdm(players):
        result = process_player(
            session, player,
            marginals_starters, marginals_backups,
            correlations_starters, correlations_backups
        )
        if result:
            mean, std = result
            results.append({
                'player_id': player.id,
                'name': player.name,
                'position': player.position,
                'fpts_mean': mean,
                'fpts_std': std,
            })

            # Update database (need to add columns first)
            # update_player_distributions(session, player.id, mean, std)

    # Save results to CSV for now
    results_df = pd.DataFrame(results)
    output_path = model_dir / "player_fpts_distributions.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Processed {len(results)} players")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Show sample results
    if len(results) > 0:
        print("\nSample results:")
        print(results_df.head(10).to_string(index=False))

    session.close()


if __name__ == "__main__":
    main()
