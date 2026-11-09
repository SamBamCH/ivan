"""
Player fantasy-point samples for lineup generation and contest simulation.

generate_player_samples() is the one draw the field, the contest sim and the chooser read: the slate's games simulated
minute by minute by the game simulation (ivan.sports.nba.gamesim.world.World: the script's possessions, the rotation under
fouls and garbage time, the five on the floor sharing them, makes and misses, rebounds, steals and blocks on both sides),
so every player's fantasy points in world w come from the same game. Locked players from ProjectionSnapshot are
independent Normals (they have no game left to simulate).
"""
from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Dict, Optional

import numpy as np
from sqlalchemy.orm import Session

from ivan.sports.nba.models import ProjectionSnapshot

WORLD_CAP = 2000     # worlds drawn at most per call; a larger request resamples them (the generator asks for thousands of jittered worlds)


def generate_player_samples(
    session: Session,
    n_samples: int = 1000,
    randomness: float = 1.0,
    seed: Optional[int] = None,
    snapshot_date: Optional[date] = None,
    include_active_players: bool = True,
    name_to_id: Optional[Dict[str, int]] = None,
    **_ignored,
) -> Dict[int, np.ndarray]:
    """
    Returns player_id -> np.ndarray(n_samples,) of DK fantasy points.

    randomness scales each player's spread around his mean (0 = deterministic, 1 = the game's). Up to WORLD_CAP worlds are
    simulated; a larger n_samples reuses them (worlds are exchangeable, and the generator jitters each one anyway).
    """
    rng = np.random.default_rng(seed)
    randomness = float(np.clip(randomness, 0.0, 1.0))
    samples: Dict[int, np.ndarray] = {}

    if include_active_players:
        from ivan.sports.nba.gamesim.world import World
        n_world = int(min(n_samples, WORLD_CAP))
        world = World.draw(session, n=n_world, seed=None if seed is None else int(rng.integers(0, 2**31 - 1)))
        idx = np.arange(n_world) if n_world == n_samples else rng.integers(0, n_world, size=int(n_samples))
        for pid, x in world.to_player_samples().items():
            x = x[idx]
            if randomness < 1.0:
                mu = float(x.mean())
                x = mu + randomness * (x - mu)
            samples[int(pid)] = np.clip(x, 0.0, None)

    if snapshot_date is not None:
        day_start = datetime(snapshot_date.year, snapshot_date.month, snapshot_date.day)
        snaps = (session.query(ProjectionSnapshot)
                 .filter(ProjectionSnapshot.snapshot_date >= day_start,
                         ProjectionSnapshot.snapshot_date < day_start + timedelta(days=1),
                         ProjectionSnapshot.fpts_mean.isnot(None), ProjectionSnapshot.fpts_std.isnot(None)).all())
        for snap in snaps:
            pid = getattr(snap, "id", None)
            if pid is None or int(pid) in samples:
                continue
            x = rng.normal(float(snap.fpts_mean), float(snap.fpts_std) * randomness, n_samples)
            samples[int(pid)] = np.clip(x, 0, None)
    return samples
