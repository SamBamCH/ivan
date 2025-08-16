# src/ivan/sports/nfl/correlated_sampling.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
# add imports at top
import pandas as pd
from sqlalchemy import text


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
_VINE_POSITIONS = ["QB", "RB", "WR", "TE", "OppDST"]  # 5D vine


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
    u = np.clip(u, 1e-6, 1 - 1e-6)
    dp = _player_dist(player)
    if dp:
        dist, params = dp
        return dist.ppf(u, *params)

    pos = (player.position or "").upper()

    # --- NEW: DST uses empirical quantiles from nfl_dst_game ---
    if pos == "DST":
        with session.bind.connect() as conn:
            df = pd.read_sql(
                text("SELECT dst_points FROM nfl_dst_game WHERE team = :t AND dst_points IS NOT NULL"),
                conn,
                params={"t": player.team},
            )
            if df.empty:
                # fallback to league-wide empirical if team history is sparse
                df = pd.read_sql(
                    text("SELECT dst_points FROM nfl_dst_game WHERE dst_points IS NOT NULL"),
                    conn,
                )

        vals = df["dst_points"].astype(float).to_numpy()
        if vals.size == 0:
            return np.zeros_like(u)  # ultimate fallback
        vals.sort()
        idx = np.minimum((u * (vals.size - 1)).astype(int), vals.size - 1)
        return vals[idx]
    # --- end DST path ---

    # Non-DST fallback via stat-level models
    if player_models is None:
        raise ValueError(
            f"No fpts distribution for {player.name} and no stat-level models loaded for fallback."
        )
    marginals, stacked = player_models
    sim = simulate_player_fpts(session, player.id, marginals, stacked, n_samples=n_samples * 2)
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

    oppdst_uniforms: dict[tuple[str, str], np.ndarray] = {}  # key = (dst_team, vs_team)

    pos_to_col = {p: i for i, p in enumerate(_VINE_POSITIONS)}


    out: Dict[int, np.ndarray] = {}

    for team, pos_players in teams.items():
        vine = vines.get(team)
        # uniforms for (QB,RB,WR,TE,OppDST)
        if isinstance(vine, VineCopula):
            U_team = np.asarray(vine.sample(cfg.n_samples))
        else:
            U_team = rng.random((cfg.n_samples, len(_VINE_POSITIONS)))
        U_team = np.clip(U_team, 1e-6, 1 - 1e-6)

        # find opponent for this team (take from any player row on this team)
        opponent = None
        for lst in pos_players.values():
            for pl in lst:
                if getattr(pl, "opponent", None):
                    opponent = pl.opponent
                    break
            if opponent:
                break

        # stash OppDST uniforms for the opponent’s DST vs this team
        if opponent is not None and 'OppDST' in pos_to_col:
            opp_col = pos_to_col['OppDST']
            oppdst_uniforms[(opponent, team)] = U_team[:, opp_col].copy()

        # now fan out uniforms to players
        for pos, players in pos_players.items():
            col = pos_to_col.get(pos)  # col is None for DST
            # offense positions use team vine column
            if pos in ('QB', 'RB', 'WR', 'TE') and col is not None:
                u_pos_base = U_team[:, col]
            else:
                u_pos_base = None  # DST handled below

            for player in players:
                if pos == 'DST':
                    # use the OppDST uniforms stashed by the opponent team loop
                    key = (player.team, getattr(player, "opponent", None))
                    u = oppdst_uniforms.get(key)
                    if u is None:
                        u = rng.random(cfg.n_samples)  # fallback iid
                else:
                    # regular alpha‐mixing for offense positions
                    if cfg.alpha >= 1.0 and u_pos_base is not None:
                        u = u_pos_base.copy()
                    elif cfg.alpha <= 0.0 or u_pos_base is None:
                        u = rng.random(cfg.n_samples)
                    else:
                        u = cfg.alpha * u_pos_base + (1.0 - cfg.alpha) * rng.random(cfg.n_samples)

                if cfg.jitter > 0:
                    u = u + rng.normal(0.0, cfg.jitter, size=u.shape)
                u = np.clip(u, 1e-6, 1 - 1e-6)

                x = _ppf_with_fallback(u, player, session, player_models, cfg.n_samples)
                out[player.id] = x

    return out