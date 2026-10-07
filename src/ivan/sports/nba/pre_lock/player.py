"""
Player fantasy-point samples for lineup generation and contest simulation.

generate_player_samples() draws one correlated universe per game (see
projection/correlation.py): every player on the slate with a stored marginal
is sampled through his own quantiles on a Gaussian copula with the role-pair
correlation; players without quantiles are Normal(fpts_mean, fpts_std) on the
same copula; locked players from ProjectionSnapshot are independent Normals.
"""
from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Dict, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ivan.core.utils import normalize_name
from ivan.sports.nba.models import NBAPlayer, ProjectionSnapshot
from ivan.sports.nba.projection.correlation import (load_league, PairHistory, build_game_correlation,
                                                    draw_from_marginals)


def generate_player_samples(
    session: Session,
    n_samples: int = 1000,
    randomness: float = 1.0,
    seed: Optional[int] = None,
    snapshot_date: Optional[date] = None,
    include_active_players: bool = True,
    name_to_id: Optional[Dict[str, int]] = None,
    correlated: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Returns player_id -> np.ndarray(n_samples,) of DK fantasy points.

    randomness scales the spread around the mean (0 = deterministic, 1 = full).
    correlated=False draws every player independently (the old behaviour).
    """
    rng = np.random.default_rng(seed)
    randomness = float(np.clip(randomness, 0.0, 1.0))
    samples: Dict[int, np.ndarray] = {}

    if include_active_players:
        players = session.execute(select(NBAPlayer).where(
            NBAPlayer.fpts_mean.isnot(None), NBAPlayer.fpts_std.isnot(None))).scalars().all()
        league = load_league() if correlated else {}
        pairs = PairHistory.load() if correlated else None
        # one universe per game: group by the unordered (team, opponent) pair
        games: Dict[tuple, list] = {}
        for p in players:
            key = tuple(sorted([p.team or "", p.opponent or ""])) if correlated else (p.id,)
            games.setdefault(key, []).append(p)
        for key, ps in games.items():
            names = [normalize_name(p.name) for p in ps]
            teams = [p.team for p in ps]
            mins = [float(p.minutes_proj) if p.minutes_proj else (float(p.fpts_mean) / 0.95) for p in ps]
            means = [float(p.fpts_mean) for p in ps]
            sds = [float(p.fpts_std) for p in ps]
            quants = [p.fpts_quantiles if isinstance(p.fpts_quantiles, list) else None for p in ps]
            R = build_game_correlation(names, teams, mins, league, pairs) if correlated and len(ps) > 1 else np.eye(len(ps))
            X = draw_from_marginals(R, quants, means, sds, n_samples, rng, randomness)
            for j, p in enumerate(ps):
                samples[int(p.id)] = X[:, j]

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
