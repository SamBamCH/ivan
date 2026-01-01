from ivan.sports.nfl.models import NFLPlayer, NFLStat, NFLOdds
from ivan.sports.nfl.refactor.stat import fit_stat_distribution, StatDistribution, sample_stat
import pandas as pd
import numpy as np
import scipy.stats as st

STATS = {
    'QB': ['passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards', 'rushing_tds'],
    'RB': ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards'],
    'WR': ['receptions', 'receiving_yards', 'receiving_tds'],
    'TE': ['receptions', 'receiving_yards', 'receiving_tds'],
    'K':  ['fg_made', 'pat_made'],
    'DST': ['def_tds', 'def_sacks', 'def_interceptions', 'special_teams_tds', 'points_allowed'] 
}

# Roster slot IDs for slate type detection
SHOWDOWN_SLOTS = {511, 512}
CLASSIC_SLOTS = {67, 68, 69, 70, 71}

# Odds filtering thresholds
# Prices more extreme than these are likely data errors (American odds)
MIN_PRICE = -1000  # Exclude odds like -1000 (implies 90%+ prob, usually bad data)
MAX_PRICE = 10000  # Exclude extreme longshots that are likely errors


def _filter_outlier_odds(odds: list[NFLOdds]) -> list[NFLOdds]:
    """
    Filter out outlier odds that are likely data errors.
    
    Removes:
    - Extreme prices (e.g., -1000 which implies 90%+ probability)
    - Point outliers that are far from the median for that market
    """
    if not odds:
        return odds
    
    # First pass: filter by price
    filtered = [o for o in odds if MIN_PRICE <= o.price <= MAX_PRICE]
    
    if len(filtered) < len(odds):
        removed = len(odds) - len(filtered)
        print(f"  Filtered {removed} odds with extreme prices")
    
    # Second pass: filter point outliers within each market
    # Group by market and remove points that are >3 std from median
    from collections import defaultdict
    by_market = defaultdict(list)
    for o in filtered:
        by_market[o.market].append(o)
    
    final = []
    for market, market_odds in by_market.items():
        if len(market_odds) < 2:
            # Not enough data to detect outliers
            final.extend(market_odds)
            continue
        
        points = np.array([o.point for o in market_odds])
        median = np.median(points)
        std = np.std(points)
        
        if std == 0:
            final.extend(market_odds)
            continue
        
        for o in market_odds:
            z_score = abs(o.point - median) / std
            if z_score <= 5:
                final.append(o)
            else:
                print(f"  Filtered outlier: {o.market} point={o.point} (median={median:.1f}, z={z_score:.1f})")
    
    return final


def _is_classic_slate(players: list[NFLPlayer]) -> bool:
    """Check if slate is Classic based on roster_slot_ids present."""
    slot_ids = {p.roster_slot_id for p in players}
    return bool(slot_ids & CLASSIC_SLOTS) and not bool(slot_ids & SHOWDOWN_SLOTS)


def _dedupe_classic_players(session, players: list[NFLPlayer]) -> list[NFLPlayer]:
    """For Classic slates, keep only one player per name (remove duplicates)."""
    seen_names = set()
    keep = []
    remove = []
    
    for player in players:
        if player.name in seen_names:
            remove.append(player)
        else:
            seen_names.add(player.name)
            keep.append(player)
    
    # Delete duplicates from DB
    for player in remove:
        session.delete(player)
    session.commit()
    
    print(f"Removed {len(remove)} duplicate players for Classic slate")
    return keep


def _populate_fpts_stats(player: NFLPlayer, samples: np.ndarray) -> None:
    """
    Populate all fpts statistical fields on NFLPlayer from samples.
    """
    player.fpts_n = len(samples)
    player.fpts_mean = float(np.mean(samples))
    player.fpts_std = float(np.std(samples))
    player.fpts_min = float(np.min(samples))
    player.fpts_max = float(np.max(samples))
    player.fpts_skew = float(st.skew(samples))
    player.fpts_kurtosis = float(st.kurtosis(samples))
    
    # Percentiles
    player.fpts_p01 = float(np.percentile(samples, 1))
    player.fpts_p05 = float(np.percentile(samples, 5))
    player.fpts_p10 = float(np.percentile(samples, 10))
    player.fpts_p25 = float(np.percentile(samples, 25))
    player.fpts_p50 = float(np.percentile(samples, 50))
    player.fpts_p75 = float(np.percentile(samples, 75))
    player.fpts_p90 = float(np.percentile(samples, 90))
    player.fpts_p95 = float(np.percentile(samples, 95))
    player.fpts_p99 = float(np.percentile(samples, 99))


def _fit_dst_by_salary(player: NFLPlayer, rng: np.random.Generator, n_samples: int = 10000) -> np.ndarray:
    """
    Generate DST fpts samples scaled by salary.
    Higher salary = slightly higher projection.
    
    DST salaries typically range from ~2000 to ~4000.
    Average DST scores around 6-8 fpts with high variance.
    
    Returns the samples array for further processing.
    """
    # Normalize salary to a 0-1 scale (assuming 2000-4500 range)
    salary_norm = (player.salary - 2000) / 2500
    salary_norm = np.clip(salary_norm, 0, 1)
    
    # Base mean around 7, scaled by salary (+/- 2 pts)
    base_mean = 5.0 + (salary_norm * 4.0)  # Range: 5 to 9
    
    # DST has high variance
    sigma = 5.0
    
    # Generate samples
    samples = rng.normal(base_mean, sigma, n_samples)
    
    return samples


def fit_fpts_distributions(session):
    players = session.query(NFLPlayer).all()
    
    # Dedupe for Classic slates
    if _is_classic_slate(players):
        print("Detected Classic slate - removing duplicate players")
        players = _dedupe_classic_players(session, players)

    rng = np.random.default_rng(42)
    
    for player in players:
        print(f'processing player {player.name}')
        
        # DST: use simple salary-based projection for now
        if player.position == 'DST':
            dst_samples = _fit_dst_by_salary(player, rng)
            mu, sigma = st.norm.fit(dst_samples)
            player.fpts_dist_name = 'norm'
            player.fpts_dist_params = (float(mu), float(sigma))
            _populate_fpts_stats(player, dst_samples)
            print(f'{player.name} (DST): salary={player.salary}, mu={mu:.2f}, sigma={sigma:.2f}')
            session.commit()
            continue            
        
        # Get player historical stats
        historical_stats = session.query(NFLStat).filter(
            NFLStat.player_display_name == player.name
        ).all()
        
        has_historical = historical_stats and len(historical_stats) >= 5
        
        # Get player odds (only Over/Yes sides)
        odds = session.query(NFLOdds).filter(
            NFLOdds.player == player.name,
            NFLOdds.side.in_(['Over', 'Yes'])
        ).all()
        
        # Filter out outlier odds (bad data from some sportsbooks)
        odds = _filter_outlier_odds(odds)
        
        if not odds:
            print(f'skipping {player.name}: no odds found')
            continue
        
        # Check if we have any non-anytime TD odds
        non_anytime_odds = [o for o in odds if 'anytime' not in o.market.lower()]
        has_non_anytime_odds = len(non_anytime_odds) > 0
        
        # Skip only if we have neither historical stats nor non-anytime odds
        if not has_historical and not has_non_anytime_odds:
            print(f'skipping {player.name}: no historical stats and only anytime TD odds')
            continue
        
        # If no player-specific historical stats but we have good odds, fall back to team/position
        if not has_historical:
            print(f'{player.name}: no player historical stats, using team/position fallback')
            historical_stats = session.query(NFLStat).filter(
                NFLStat.team == player.team,
                NFLStat.position == player.position
            ).all()
            if not historical_stats or len(historical_stats) < 10:
                print(f'skipping {player.name}: team/position fallback also insufficient')
                continue
        
        odds_df = pd.DataFrame([
            {
                'market': o.market,
                'side': o.side,
                'point': o.point,
                'price': o.price
            }
            for o in odds
        ])
        
        stats_to_model = STATS.get(player.position)
        stat_distributions: list[StatDistribution | None] = []
        
        # Fit each stat distribution - stat.py handles missing odds via neighbor lookup
        for stat in stats_to_model:
            historical_stat_values = [
                getattr(s, stat) for s in historical_stats 
                if getattr(s, stat) is not None
            ]
            
            result = fit_stat_distribution(
                odds_df, 
                historical_stat_values, 
                stat,
                session=session,  # Pass session for neighbor lookup
                position=player.position  # Pass position for neighbor lookup
            )
            stat_distributions.append(result)
        
        # Filter out None distributions
        valid_stats = [(s, sd) for s, sd in zip(stats_to_model, stat_distributions) if sd is not None]
        if not valid_stats:
            print(f"no valid stat distributions for {player.name}; skipping")
            continue
        
        stats_to_model_valid = [s for s, _ in valid_stats]
        stat_distributions_valid = [sd for _, sd in valid_stats]
        
        # Build correlation from historical stats (only valid stats)
        df_hist = pd.DataFrame({
            stat: [getattr(s, stat) for s in historical_stats if getattr(s, stat) is not None]
            for stat in stats_to_model_valid
        })
        df_hist = df_hist.dropna()
        
        if df_hist.shape[0] < 5:
            print(f"insufficient joint history for correlation: {player.name}")
            continue
        
        corr = df_hist.corr(method='spearman').fillna(0.0).values
        
        # Sample independent marginals
        n_samples = 10000
        rng = np.random.default_rng(42)
        X = np.zeros((n_samples, len(stats_to_model_valid)), dtype=float)
        for j, sd in enumerate(stat_distributions_valid):
            X[:, j] = sample_stat(sd, n_samples, rng)
        
        # Impose dependence using Gaussian copula on ranks
        U = np.zeros_like(X)
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j])
            ranks = np.empty_like(order)
            ranks[order] = np.arange(n_samples)
            U[:, j] = (ranks + 0.5) / n_samples
        
        L = np.linalg.cholesky(corr + 1e-6 * np.eye(corr.shape[0]))
        Z = rng.standard_normal(size=(n_samples, corr.shape[0])) @ L.T
        U_target = st.norm.cdf(Z)
        
        X_dep = np.zeros_like(X)
        for j in range(X.shape[1]):
            x_sorted = np.sort(X[:, j])
            order = np.argsort(U_target[:, j])
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(n_samples)
            X_dep[:, j] = x_sorted[inv_order]
        
        # Compute player FPTS using DraftKings scoring weights
        FPTS_WEIGHTS = {
            'passing_yards': 0.04,
            'passing_tds': 4.0,
            'passing_interceptions': -1.0,
            'rushing_yards': 0.1,
            'rushing_tds': 6.0,
            'receptions': 1.0,
            'receiving_yards': 0.1,
            'receiving_tds': 6.0,
            'fg_made': 4.2,
            'pat_made': 1.0,
        }
        weights = np.array([FPTS_WEIGHTS.get(s, 0.0) for s in stats_to_model_valid], dtype=float)
        fpts_samples = X_dep @ weights
        
        # Fit normal to FPTS and populate all stats
        mu, sigma = st.norm.fit(fpts_samples)
        player.fpts_dist_name = 'norm'
        player.fpts_dist_params = (float(mu), float(sigma))
        _populate_fpts_stats(player, fpts_samples)
        session.commit()
    
    # Clean up: remove players with no fpts distribution
    players_without_fpts = session.query(NFLPlayer).filter(
        NFLPlayer.fpts_dist_params.is_(None)
    ).all()
    

    if players_without_fpts:
        print(f"\nRemoving {len(players_without_fpts)} players with no fpts distribution:")
        for p in players_without_fpts:
            print(f"  - {p.name} ({p.position})")
            session.delete(p)
        session.commit()


