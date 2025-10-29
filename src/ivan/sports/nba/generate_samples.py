from sqlalchemy.orm import Session
from sqlalchemy import select
import numpy as np
from ivan.sports.nba.models import NBAPlayer

def generate_player_samples(
    session: Session,
    n_samples: int = 1000,
    randomness: float = 1.0,
    seed: int | None = None,
) -> dict[int, np.ndarray]:
    """
    Generate independent FPTS samples for all players in the NBAPlayer table.

    Each player's samples are drawn from their fitted Normal(mean, std) distribution
    stored in the NBAPlayer table (fpts_mean, fpts_std columns).

    Args:
        session: Database session
        n_samples: Number of samples to generate per player (default: 1000)
        randomness: Controls distribution spread (0.0 = deterministic at mean,
                   1.0 = full distribution). Values clamped to [0.0, 1.0].
        seed: Random seed for reproducibility (optional)

    Returns:
        Dictionary mapping player_id -> np.ndarray of shape (n_samples,) containing
        FPTS samples for each player

    Raises:
        ValueError: If a player has missing fpts_mean or fpts_std
    """
    rng = np.random.default_rng(seed)

    # Clamp randomness to [0.0, 1.0]
    randomness = float(np.clip(randomness, 0.0, 1.0))

    # Query all players with valid FPTS distributions
    stmt = select(NBAPlayer).where(
        (NBAPlayer.fpts_mean.isnot(None)) &
        (NBAPlayer.fpts_std.isnot(None))
    )
    players = session.execute(stmt).scalars().all()

    if not players:
        return {}

    samples_by_player = {}

    for player in players:
        # Get the player's fitted mean and std
        mean = float(player.fpts_mean)
        std = float(player.fpts_std)

        # Apply randomness scaling: reduce std when randomness < 1.0
        effective_std = std * randomness

        # Generate independent samples from Normal(mean, effective_std)
        player_samples = rng.normal(loc=mean, scale=effective_std, size=n_samples)

        # Clip to non-negative (FPTS can't be negative)
        player_samples = np.clip(player_samples, 0, None)

        samples_by_player[player.id] = player_samples

    return samples_by_player
