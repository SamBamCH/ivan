"""
Fantasy points sample generation for NBA players.

This module provides Monte Carlo sampling of player fantasy point outcomes
based on their fitted Normal(mean, std) distributions.
"""

from typing import Optional, Dict
from datetime import date, datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select

from ivan.sports.nba.models import NBAPlayer, ProjectionSnapshot


def generate_player_samples(
    session: Session,
    n_samples: int = 1000,
    randomness: float = 1.0,
    seed: Optional[int] = None,
    snapshot_date: Optional[date] = None,
    include_active_players: bool = True,
    name_to_id: Optional[Dict[str, int]] = None,
) -> Dict[int, np.ndarray]:
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
        snapshot_date: Optional date to include ProjectionSnapshot rows for locked players
        include_active_players: Whether to include current active players from NBAPlayer
        name_to_id: Optional name->id mapping (unused, for API compatibility)

    Returns:
        Dictionary mapping player_id -> np.ndarray of shape (n_samples,) containing
        FPTS samples for each player
    """
    rng = np.random.default_rng(seed)

    # Clamp randomness to [0.0, 1.0]
    randomness = float(np.clip(randomness, 0.0, 1.0))

    samples_by_player: Dict[int, np.ndarray] = {}

    # Include current active (unlocked) players from NBAPlayer
    if include_active_players:
        stmt = select(NBAPlayer).where(
            (NBAPlayer.fpts_mean.isnot(None)) &
            (NBAPlayer.fpts_std.isnot(None))
        )
        players = session.execute(stmt).scalars().all()
        for player in players:
            mean = float(player.fpts_mean)
            std = float(player.fpts_std)
            effective_std = std * randomness
            player_samples = rng.normal(loc=mean, scale=effective_std, size=n_samples)
            samples_by_player[int(player.id)] = np.clip(player_samples, 0, None)

    # Add ProjectionSnapshot rows for the specified date to cover locked players
    if snapshot_date is not None:
        day_start = datetime(snapshot_date.year, snapshot_date.month, snapshot_date.day)
        day_end = day_start + timedelta(days=1)

        snaps = (
            session.query(ProjectionSnapshot)
            .filter(
                ProjectionSnapshot.snapshot_date >= day_start,
                ProjectionSnapshot.snapshot_date < day_end,
                ProjectionSnapshot.fpts_mean.isnot(None),
                ProjectionSnapshot.fpts_std.isnot(None),
            )
            .all()
        )
        for snap in snaps:
            pid = getattr(snap, 'id', None)
            if pid is None:
                continue
            pid = int(pid)
            if pid in samples_by_player:
                continue

            mean = float(snap.fpts_mean)
            std = float(snap.fpts_std)
            effective_std = std * randomness
            player_samples = rng.normal(loc=mean, scale=effective_std, size=n_samples)
            samples_by_player[pid] = np.clip(player_samples, 0, None)

    return samples_by_player

