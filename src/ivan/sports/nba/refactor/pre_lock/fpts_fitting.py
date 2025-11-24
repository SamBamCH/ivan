from ivan.sports.nba.models import NBAPlayer, NBAStat, NBAOdds, StartingLineups
from ivan.sports.nba.refactor.utils import parse_position, american_to_prob
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
from scipy.optimize import minimize

STATS = ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers']

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

MARKET_MAP = {
    'points': ('player_points', 'player_points_alternate'),
    'rebounds': ('player_rebounds', 'player_rebounds_alternate'),
    'assists': ('player_assists', 'player_assists_alternate'),
    'fg3m': ('player_threes', 'player_threes_alternate'),
    'steals': ('player_steals', 'player_steals_alternate'),
    'blocks': ('player_blocks', 'player_blocks_alternate'),
}

def _get_market_info(session, player_name):
    """
    Returns a dictionary of market info for a player.
    {
        'points': [{'line': 20.5, 'prob_over': 0.55}, ...],
        ...
    }
    """
    market_info = {}
    for stat, markets in MARKET_MAP.items():
        if isinstance(markets, str):
            markets = (markets,)
            
        odds_records = session.query(NBAOdds).filter(
            NBAOdds.player == player_name, 
            NBAOdds.market.in_(markets)
        ).all()
        
        if not odds_records:
            continue
            
        # Group by line to get fair odds
        lines = {}
        for rec in odds_records:
            key = (rec.sportsbook, rec.point)
            if key not in lines:
                lines[key] = {}
            lines[key][rec.side.lower()] = rec.price
            
        stat_targets = []
        for (book, point), prices in lines.items():
            if 'over' in prices and 'under' in prices:
                p_over = american_to_prob(prices['over'])
                p_under = american_to_prob(prices['under'])
                # Remove vig
                fair_over = p_over / (p_over + p_under)
                stat_targets.append({'line': point, 'prob_over': fair_over})
        
        if stat_targets:
            market_info[stat] = stat_targets
            
    return market_info

def _apply_iman_conover(marginal_samples, rank_corr_matrix):
    """
    Rearranges marginal samples to match the rank correlation structure.
    marginal_samples: DataFrame of shape (N, D)
    rank_corr_matrix: DataFrame of shape (D, D)
    """
    N, D = marginal_samples.shape
    cols = marginal_samples.columns
    
    # 1. Generate random normal samples with the desired correlation
    # We treat the rank correlation as if it were Pearson for the normal copula
    # This is an approximation but standard for Iman-Conover
    def _make_psd(C: np.ndarray) -> np.ndarray:
        """Return a symmetric positive semi-definite approximation of C.
        Adds minimal jitter if needed. Falls back to identity on failure."""
        C = np.array(C, dtype=float)
        # Force symmetry to avoid complex eigenvalues from numerical noise
        C = (C + C.T) / 2.0
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            return np.eye(C.shape[0])
        min_eig = eigvals.min()
        if min_eig < 1e-8:
            # Jitter the diagonal just enough to make all eigenvalues >= 1e-8
            C += (1e-8 - min_eig) * np.eye(C.shape[0])
        # Clip any correlations slightly outside [-1,1] due to rounding
        C = np.clip(C, -1.0, 1.0)
        # Ensure diagonal is exactly 1 (correlation matrix property)
        np.fill_diagonal(C, 1.0)
        return C

    C = _make_psd(rank_corr_matrix.values)
    try:
        Z = np.random.multivariate_normal(np.zeros(D), C, size=N)
    except np.linalg.LinAlgError:
        # Fallback: independent normals if still not PD
        Z = np.random.normal(size=(N, D))
    
    # 2. Find the ranks of Z
    # argsort().argsort() gives the rank (0-based)
    ranks = np.argsort(np.argsort(Z, axis=0), axis=0)
    
    # 3. Reorder marginals
    ordered_samples = np.zeros_like(marginal_samples.values)
    for i, col in enumerate(cols):
        # Sort the marginal samples
        sorted_marginal = np.sort(marginal_samples[col].values)
        # Place them according to the ranks of Z
        ordered_samples[:, i] = sorted_marginal[ranks[:, i]]
        
    return pd.DataFrame(ordered_samples, columns=cols)

def fit_fpts_distributions(session, logging_level: str = "INFO"):
    """
    main function: process all players and fit fpts distirbutions.
    """
    logging.basicConfig(level=f"logging.{logging_level}")
    logger = logging.getLogger(__name__)
    players = session.query(NBAPlayer).all()
    
    num_samples = 10000
    
    for player in players:
        position = parse_position(player.position)
        if position is None:
            logger.error(f'{player.name} has unparsable position')
            continue

        # Check if player is projected to start
        is_projected_starter = session.query(StartingLineups).filter(
            StartingLineups.player_name == player.name
        ).first() is not None
        
        # Check starting frequency: look at last 10 games, count where position != 'BENCH'
        recent_games = session.query(NBAStat).filter(NBAStat.player_name == player.name).order_by(NBAStat.game_date.desc()).limit(10).all()
        
        if recent_games:
            started_count = sum(1 for g in recent_games if g.position and g.position.upper() != 'BENCH')
            starting_freq = started_count / len(recent_games)
        else:
            starting_freq = 0.0
        
        usually_starts = starting_freq >= 0.6  # 60% threshold
        
        # SCENARIO ROUTING: Choose appropriate historical data source
        if is_projected_starter and not usually_starts:
            # Scenario 2: Newly starting player - use positional starter stats
            logger.warning(f"{player.name}: Starting today but usually benched ({starting_freq:.1%} start rate)")
            logger.warning(f"  → Using positional starter stats for {player.team} {position}")
            
            # Get positional starter stats (position = G/F/C, not BENCH)
            pos_records = session.query(NBAStat).filter(NBAStat.team_abbreviation == player.team, NBAStat.position == position).order_by(NBAStat.game_date.desc()).limit(30).all()
            
            if pos_records:
                # Convert to DataFrame
                data = [{stat: getattr(rec, stat, 0) for stat in STATS} for rec in pos_records]
                df = pd.DataFrame(data).dropna()
                logger.info(f"  Team {player.team} Position {position}: {len(df)} starter games")
            else:
                df = pd.DataFrame()
            
            # Fallback to player's own stats if insufficient
            if df.empty or len(df) < 5:
                logger.warning(f"  → Insufficient positional data, using player's own stats")
                player_records = session.query(NBAStat).filter(NBAStat.player_name == player.name).all()
                data = [{stat: getattr(rec, stat, 0) for stat in STATS} for rec in player_records]
                df = pd.DataFrame(data).dropna()
        
        else:
            # Scenario 1 or 3: Use player's own stats
            if is_projected_starter:
                logger.info(f"{player.name}: Regular starter ({starting_freq:.1%} start rate)")
            else:
                logger.info(f"{player.name}: Bench role ({starting_freq:.1%} start rate)")
            
            # Get player's historical stats
            player_records = session.query(NBAStat).filter(NBAStat.player_name == player.name).all()
            data = [{stat: getattr(rec, stat, 0) for stat in STATS} for rec in player_records]
            df = pd.DataFrame(data).dropna()
        
        if df.empty or len(df) < 5:
             logger.error(f'insufficient historical data found for {player.name}')
             continue

        market_info = _get_market_info(session, player.name)
        if not market_info:
            logger.warning(f'no odds found for {player.name}, using history only')
            # We can still proceed with history only if desired, or skip. 
            # User prompt implied we use odds, but let's be robust.

        # 1. Calculate Rank Correlation Matrix from History
        # Handle constant columns (std=0) which cause NaN in corr
        df_std = df[STATS].std()
        valid_stats = [s for s in STATS if df_std[s] > 0]
        
        if not valid_stats:
            logger.error(f"All stats are constant for {player.name}")
            continue
            
        rank_corr = df[valid_stats].corr(method='spearman').fillna(0)
        
        # 2. Generate Marginal Samples
        marginal_samples = {}
        
        for stat in STATS:
            h_vals = df[stat].values
            
            # If constant history, just repeat it
            if np.std(h_vals) == 0:
                marginal_samples[stat] = np.full(num_samples, h_vals[0])
                continue
                
            # Fit KDE
            try:
                kde = gaussian_kde(h_vals)
                # Resample
                samples = kde.resample(num_samples).flatten()
            except Exception as e:
                logger.warning(f"KDE failed for {player.name} {stat}: {e}")
                # Fallback to bootstrap with noise
                samples = np.random.choice(h_vals, size=num_samples, replace=True)
                samples += np.random.normal(0, 0.1, size=num_samples)

            # Adjust to Market
            if stat in market_info:
                targets = market_info[stat]
                shifts = []
                for target in targets:
                    line = target['line']
                    prob_over = target['prob_over']
                    
                    # Find quantile in historical distribution (using the samples we just generated)
                    # We want P(X + s > line) = prob_over
                    # => P(X > line - s) = prob_over
                    # => line - s = Quantile(1 - prob_over)
                    # => s = line - Quantile(1 - prob_over)
                    
                    current_quantile = np.percentile(samples, (1 - prob_over) * 100)
                    shifts.append(line - current_quantile)
                
                avg_shift = np.mean(shifts)
                samples += avg_shift
            
            # Enforce non-negativity for stats that can't be negative
            if stat != 'turnovers': # turnovers can be positive, but scoring is negative. The stat itself is count >= 0
                 samples = np.maximum(samples, 0)
            else:
                 samples = np.maximum(samples, 0) # turnovers are also counts >= 0

            marginal_samples[stat] = samples

        marginals_df = pd.DataFrame(marginal_samples)
        
        # 3. Apply Iman-Conover to induce correlation
        # We only apply it to the valid stats that had correlation
        if len(valid_stats) > 1:
            correlated_part = _apply_iman_conover(marginals_df[valid_stats], rank_corr)
            # Update the marginals df
            for col in valid_stats:
                marginals_df[col] = correlated_part[col]
        
        samples = marginals_df.values
        
        # 4. Calculate FPTS
        fpts_samples = np.zeros(num_samples)
        
        # Base scoring
        for i, stat in enumerate(STATS):
            fpts_samples += samples[:, i] * DK_SCORING.get(stat, 0)
            
        # Bonuses
        # STATS = ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers']
        # Indices: 0, 2, 3, 4, 5
        bonus_indices = [0, 2, 3, 4, 5]
        bonus_stats = samples[:, bonus_indices]
        
        double_digits = (bonus_stats >= 10).sum(axis=1)
        fpts_samples += (double_digits >= 2) * DK_SCORING['double_double']
        fpts_samples += (double_digits >= 3) * DK_SCORING['triple_double']
        
        # Update Player
        player.fpts_mean = float(np.mean(fpts_samples))
        player.fpts_std = float(np.std(fpts_samples))
        
    session.commit()
