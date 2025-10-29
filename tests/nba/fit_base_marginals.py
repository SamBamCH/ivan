"""
Fit base marginal distributions for NBA stats by position.

This script:
1. Loads historical box score data from NBAStat table
2. Groups by position (G, F, C)
3. Fits normal and poisson distributions to each stat
4. Stores base marginals for use in odds adjustment
5. Saves marginals.pkl to /data/nba/models/

The stored distributions are flexible for later adjustment to known odds.

Usage:
    python tests/nba/fit_base_marginals.py
"""
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy.stats as st
from sqlalchemy import select

from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import NBAStat


@dataclass
class Distribution:
    """
    Flexible distribution representation for base marginals.

    Supports:
    - 'normal': params = (loc, scale)
    - 'poisson': params = (lambda,)
    """
    dist_type: str  # 'normal' or 'poisson'
    params: tuple

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from this distribution."""
        if self.dist_type == 'normal':
            loc, scale = self.params
            return np.random.normal(loc=loc, scale=scale, size=n_samples)
        elif self.dist_type == 'poisson':
            lam = self.params[0]
            return np.random.poisson(lam=lam, size=n_samples)
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")

    def quantile(self, q: float) -> float:
        """Get quantile value."""
        if self.dist_type == 'normal':
            loc, scale = self.params
            return st.norm.ppf(q, loc=loc, scale=scale)
        elif self.dist_type == 'poisson':
            lam = self.params[0]
            return st.poisson.ppf(q, lam)
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")

    def cdf(self, x: float) -> float:
        """Get cumulative distribution function value."""
        if self.dist_type == 'normal':
            loc, scale = self.params
            return st.norm.cdf(x, loc=loc, scale=scale)
        elif self.dist_type == 'poisson':
            lam = self.params[0]
            return st.poisson.cdf(x, lam)
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")


# Role configurations: separate thresholds for starters vs backups
ROLE_CONFIGS = {
    'starters': {
        'min_threshold': 25,  # Starters play >25 min
        'description': 'Primary players (>25 min)',
    },
    'backups': {
        'min_threshold': (15, 25),  # Backups play 15-25 min
        'description': 'Backup players (15-25 min)',
    },
}

# Position configuration
# We'll group guards (G, PG, SG), forwards (F, SF, PF), and centers (C)
POSITION_CONFIGS = {
    'G': {  # Guards
        'stats': ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks'],
        'stat_types': {
            'points': 'continuous',
            'fg3m': 'count',
            'rebounds': 'count',
            'assists': 'count',
            'steals': 'count',
            'blocks': 'count',
        },
    },
    'F': {  # Forwards
        'stats': ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks'],
        'stat_types': {
            'points': 'continuous',
            'fg3m': 'count',
            'rebounds': 'count',
            'assists': 'count',
            'steals': 'count',
            'blocks': 'count',
        },
    },
    'C': {  # Centers
        'stats': ['points', 'fg3m', 'rebounds', 'assists', 'steals', 'blocks'],
        'stat_types': {
            'points': 'continuous',
            'fg3m': 'count',
            'rebounds': 'count',
            'assists': 'count',
            'steals': 'count',
            'blocks': 'count',
        },
    },
}


def map_position_to_group(position: str) -> str:
    """Map NBA positions to position groups."""
    pos_upper = str(position).upper()

    if pos_upper in ('PG', 'SG', 'G'):
        return 'G'
    elif pos_upper in ('SF', 'PF', 'F'):
        return 'F'
    elif pos_upper in ('C',):
        return 'C'
    else:
        # Default to forward for unknown
        return 'F'


def fit_distribution(data: np.ndarray, stat_type: str) -> Distribution | None:
    """
    Fit distribution for a stat using simple, reliable methods.

    For normal distributions, we fit to all data (including non-positive values).
    For count distributions, we use Poisson with lambda = mean.

    Args:
        data: Array of stat values
        stat_type: 'count' or 'continuous'

    Returns:
        Distribution object or None if fit fails
    """
    if data.size == 0 or np.unique(data).size < 2:
        return None

    if stat_type == 'count':
        # Poisson: lambda = mean (simple, reliable)
        lam = float(np.mean(data))
        return Distribution(dist_type='poisson', params=(lam,))

    elif stat_type == 'continuous':
        # Normal: fit to all data, allowing location to vary freely
        try:
            loc = float(np.mean(data))
            scale = float(np.std(data, ddof=1))

            # Ensure reasonable scale
            if scale < 1e-6:
                return None

            return Distribution(dist_type='normal', params=(loc, scale))
        except Exception:
            return None

    return None


def fit_position_marginals(session, season: str = "2024-25", role: str = "starters") -> dict:
    """
    Fit base marginal distributions for each position and stat, filtered by role.

    Args:
        session: Database session
        season: NBA season to fit (e.g., "2024-25")
        role: "starters" (>25 min) or "backups" (15-25 min)

    Returns:
        {
            'G': {
                'points': Distribution(...),
                'fg3m': Distribution(...),
                ...
            },
            'F': {...},
            'C': {...}
        }
    """
    results = {}

    if role not in ROLE_CONFIGS:
        raise ValueError(f"Invalid role: {role}. Must be one of {list(ROLE_CONFIGS.keys())}")

    role_config = ROLE_CONFIGS[role]

    with session.bind.connect() as conn:
        # Load all stats for the season
        stmt = select(NBAStat).where(NBAStat.season == season)
        df = pd.read_sql(stmt, conn)

    print(f"Loaded {len(df)} player-game records for season {season}")

    # Filter to only players with position data
    df = df[df['position'].isin(['G', 'F', 'C'])]
    print(f"Filtered to {len(df)} records with valid positions (G/F/C)")

    # Apply role filter
    if isinstance(role_config['min_threshold'], tuple):
        # Backups: between min and max
        min_val, max_val = role_config['min_threshold']
        df = df[(df['minutes'] > min_val) & (df['minutes'] <= max_val)]
        print(f"Filtered to {len(df)} records with minutes in ({min_val}, {max_val}]")
    else:
        # Starters: above min
        min_val = role_config['min_threshold']
        df = df[df['minutes'] > min_val]
        print(f"Filtered to {len(df)} records with minutes > {min_val}")

    for pos_group, config in POSITION_CONFIGS.items():
        print(f"Fitting {role} marginals for position: {pos_group}")

        # Filter by position
        pos_df = df[df['position'] == pos_group]
        print(f"  {len(pos_df)} games for {pos_group}")

        stats_out = {}

        for stat in config['stats']:
            if stat not in df.columns:
                print(f"  ⚠ Stat '{stat}' not found in data, skipping")
                stats_out[stat] = None
                continue

            data = pos_df[stat].dropna().to_numpy()

            if len(data) == 0:
                print(f"  ⚠ No data for {stat}, skipping")
                stats_out[stat] = None
                continue

            stat_type = config['stat_types'][stat]
            dist = fit_distribution(data, stat_type)

            if dist:
                if dist.dist_type == 'normal':
                    print(f"  ✓ {stat:12s}: {dist.dist_type:10s} loc={dist.params[0]:.4f} scale={dist.params[1]:.4f}")
                elif dist.dist_type == 'poisson':
                    print(f"  ✓ {stat:12s}: {dist.dist_type:10s} lambda={dist.params[0]:.4f}")
                stats_out[stat] = dist
            else:
                print(f"  ✗ {stat:12s}: fit failed")
                stats_out[stat] = None

        results[pos_group] = stats_out
        print()

    return results


def convert_distributions_to_dicts(marginals: dict) -> dict:
    """
    Convert Distribution objects to plain dicts for pickling.

    This avoids pickle issues where Distribution class references become invalid
    when loading from a different script context.

    Args:
        marginals: {position: {stat: Distribution, ...}, ...}

    Returns:
        {position: {stat: {'dist': dist_type, 'params': params}, ...}, ...}
    """
    # Map our distribution names to scipy.stats names
    dist_name_map = {
        'normal': 'norm',
        'poisson': 'poisson'
    }

    converted = {}
    for position, stats_dict in marginals.items():
        converted[position] = {}
        for stat, dist_obj in stats_dict.items():
            if dist_obj is None:
                converted[position][stat] = {'dist': None, 'params': None}
            else:
                scipy_dist_name = dist_name_map.get(dist_obj.dist_type, dist_obj.dist_type)
                converted[position][stat] = {
                    'dist': scipy_dist_name,
                    'params': dist_obj.params
                }
    return converted


def main():
    """Main entry point."""
    # Initialize database
    engine = init_db("nba")
    session = get_session(engine)

    output_dir = Path("/home/sam-pratt/ivan/data/nba/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit marginals for each role
    for role in ['starters', 'backups']:
        print("="*60)
        print(f"Fitting NBA base marginal distributions for {role}")
        print("="*60)
        marginals = fit_position_marginals(session, season="2024-25", role=role)

        # Convert Distribution objects to plain dicts before pickling
        marginals_for_pickle = convert_distributions_to_dicts(marginals)

        # Save to disk
        output_path = output_dir / f"marginals_{role}.pkl"
        joblib.dump(marginals_for_pickle, output_path)

        print("="*60)
        print(f"✓ Saved {role} marginals to: {output_path}")
        print(f"  Positions: {list(marginals_for_pickle.keys())}")
        print(f"  Format: {{position: {{stat: {{'dist': str, 'params': tuple}}, ...}}}}")
        print("="*60)
        print()

    session.close()


if __name__ == "__main__":
    main()
