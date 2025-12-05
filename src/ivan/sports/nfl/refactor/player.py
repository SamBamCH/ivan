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

def fit_fpts_distributions(session):
    players = session.query(NFLPlayer).all()

    for player in players:
        print(f'processing player {player.name}')
        
        # Get player historical stats
        historical_stats = session.query(NFLStat).filter(
            NFLStat.player_display_name == player.name
        ).all()
        
        if not historical_stats or len(historical_stats) < 10:
            print(f'no historical stats found for {player.name} OR len(historical_stats) < 10, using default team/position values')
            historical_stats = session.query(NFLStat).filter(
                NFLStat.team == player.team, 
                NFLStat.position == player.position
            ).all()
            if not historical_stats or len(historical_stats) < 10: 
                print(f'still no historical values pulled for {player.name} after filtering NFLStat for {player.position} and {player.team}')
                continue
        
        # Get player odds (only Over/Yes sides)
        odds = session.query(NFLOdds).filter(
            NFLOdds.player == player.name,
            NFLOdds.side.in_(['Over', 'Yes'])
        ).all()
        
        if not odds:
            print(f'no odds found for {player.name}')
        
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
            'fg_made': 3.0,
            'pat_made': 1.0,
        }
        weights = np.array([FPTS_WEIGHTS.get(s, 0.0) for s in stats_to_model_valid], dtype=float)
        fpts_samples = X_dep @ weights
        
        # Fit normal to FPTS
        mu, sigma = st.norm.fit(fpts_samples)
        player.fpts_dist_name = 'norm'
        player.fpts_dist_params = (float(mu), float(sigma))
        session.commit()