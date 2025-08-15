# src/ivan/sports/nfl/correlated_sampling.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import joblib
import scipy.stats as st
from sqlalchemy.orm import Session

from ivan.sports.nfl.models import NFLPlayer
from ivan.sports.nfl.fpts_distribution_calculations import (
    load_models as load_player_models,
    simulate_player_fpts,
)

try:
    # pip install copulas
    from copulas.multivariate import VineCopula
except Exception:  # pragma: no cover
    VineCopula = object  # type: ignore

# 1) Keep full list (for grouping/looping), but define a separate vine order.
_POSITIONS = ["QB", "RB", "WR", "TE", "DST"]
_VINE_POSITIONS = ["QB", "RB", "WR", "TE"]  # <-- vine dimension/order is 4


@dataclass(frozen=True)
class SamplingConfig:
    n_samples: int = 1
    alpha: float = 0.85  # strength of team-position dependence (1.0 = full, 0 = iid)
    jitter: float = 0.0  # tiny Gaussian jitter on uniforms to avoid ties
    seed: int | None = None


def _load_team_vines(model_dir: Path) -> Dict[str, VineCopula]:
    """Load per-team vine copulas trained on (QB, RB, WR, TE) uniforms."""
    vines: Dict[str, VineCopula] = {}
    vine_dir = Path(model_dir) / "team_vines"
    if not vine_dir.exists():
        return vines
    for p in vine_dir.glob("*_vine.pkl"):
        team = p.stem.replace("_vine", "")
        vines[team] = joblib.load(p)
    return vines


def _group_players_by_team_pos(session: Session) -> Dict[str, Dict[str, List[NFLPlayer]]]:
    teams: Dict[str, Dict[str, List[NFLPlayer]]] = {}
    for player in session.query(NFLPlayer).all():  # "slate" = rows currently in NFLPlayer
        pos = (player.position or "").upper()
        if pos not in _POSITIONS:
            continue  # ignore K, DST for now
        teams.setdefault(player.team, {}).setdefault(pos, []).append(player)
    return teams


def _player_dist(player: NFLPlayer):
    """Return (scipy_dist, params) or None if missing."""
    name = getattr(player, "fpts_dist_name", None)
    params = getattr(player, "fpts_dist_params", None)
    if not name or params is None:
        return None
    dist = getattr(st, name)
    if isinstance(params, list):
        params = tuple(params)
    return dist, tuple(params)


def _ppf_with_fallback(
    u: np.ndarray,
    player: NFLPlayer,
    session: Session,
    player_models: tuple[dict, dict] | None,
    n_samples: int,
) -> np.ndarray:
    """
    Transform uniforms to fpts using the player's fitted fpts distribution.
    If missing, fall back to simulating fpts via stat-level models and
    do empirical-quantile mapping.
    """
    u = np.clip(u, 1e-6, 1 - 1e-6)
    dp = _player_dist(player)
    if dp:
        dist, params = dp
        return dist.ppf(u, *params)

    # Fallback path: simulate fpts from stat marginals + stacked copula
    if player_models is None:
        raise ValueError(
            f"No fpts distribution for {player.name} and no stat-level models loaded for fallback."
        )
    marginals, stacked = player_models
    # oversample to get a dense empirical CDF for smooth quantile mapping
    sim = simulate_player_fpts(
        session, player.id, marginals, stacked, n_samples=n_samples * 2
    )
    sim = np.asarray(sim)
    sim.sort()
    idx = np.minimum((u * (len(sim) - 1)).astype(int), len(sim) - 1)
    return sim[idx]


def generate_correlated_samples(
    session: Session,
    model_dir: Path | str = 'data/nfl/models',
    cfg: SamplingConfig = SamplingConfig(),
) -> Dict[int, np.ndarray]:

    model_dir = Path(model_dir)
    rng = np.random.default_rng(cfg.seed)

    vines = _load_team_vines(model_dir)
    try:
        player_models = load_player_models(model_dir)
    except Exception:
        player_models = None

    teams = _group_players_by_team_pos(session)

    # Column index only for the 4 vine positions
    pos_to_col = {p: i for i, p in enumerate(_VINE_POSITIONS)}

    out: Dict[int, np.ndarray] = {}

    for team, pos_players in teams.items():
        vine = vines.get(team)

        # Sample team-level uniforms for ONLY (QB,RB,WR,TE)
        if isinstance(vine, VineCopula):
            U_team = np.asarray(vine.sample(cfg.n_samples))  # shape (n, 4)
        else:
            # Fallback: iid uniforms per vine position
            U_team = rng.random((cfg.n_samples, len(_VINE_POSITIONS)))
        U_team = np.clip(U_team, 1e-6, 1 - 1e-6)

        for pos, players in pos_players.items():
            if pos in pos_to_col:
                # Use the appropriate vine dimension
                col = pos_to_col[pos]
                u_pos_base = U_team[:, col]
            elif pos == "DST":
                # TEMP: treat DST as independent of the team vine (no copula yet)
                # If you want a soft anti-correlation with opponent QB later,
                # we can add an optional mode that uses 1 - opp_QB with a small alpha.
                u_pos_base = rng.random(cfg.n_samples)
            else:
                # Any other positions (e.g., K) → independent for now
                u_pos_base = rng.random(cfg.n_samples)

            for player in players:
                # Blend team-level dependence with per-player noise
                if cfg.alpha >= 1.0 and pos in pos_to_col:
                    u = u_pos_base.copy()
                elif cfg.alpha <= 0.0 or pos not in pos_to_col:
                    # If not in the vine (e.g., DST), just iid + jitter
                    u = u_pos_base.copy()
                else:
                    u = cfg.alpha * u_pos_base + (1.0 - cfg.alpha) * rng.random(cfg.n_samples)

                if cfg.jitter > 0:
                    u = u + rng.normal(0.0, cfg.jitter, size=u.shape)
                u = np.clip(u, 1e-6, 1 - 1e-6)

                x = _ppf_with_fallback(u, player, session, player_models, cfg.n_samples)
                out[player.id] = x

    return out