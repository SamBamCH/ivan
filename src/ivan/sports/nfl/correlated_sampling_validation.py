# src/ivan/sports/nfl/correlated_sampling_validation.py
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import scipy.stats as st
from sqlalchemy.orm import Session

from ivan.sports.nfl.correlated_samples import (
    generate_correlated_samples, _player_dist, _group_players_by_team_pos, _POSITIONS
)
from ivan.sports.nfl.fpts_distribution_calculations import load_models as load_player_models

def _player_cdf(x: np.ndarray, player, session: Session, player_models) -> np.ndarray:
    """Evaluate CDF for a player's fpts marginal; fallback uses empirical CDF from simulated fpts."""
    dp = _player_dist(player)
    if dp:
        dist, params = dp
        return dist.cdf(x, *params)
    # fallback: empirical CDF from simulated fpts (same logic as sampling fallbacks)
    from ivan.sports.nfl.fpts_distribution_calculations import simulate_player_fpts
    sim = np.asarray(simulate_player_fpts(session, player.id, player_models[0], player_models[1], n_samples=5000))
    sim.sort()
    # empirical CDF via searchsorted
    ranks = np.searchsorted(sim, x, side="right")
    return np.clip(ranks / sim.size, 1e-6, 1 - 1e-6)

def validate_marginals(session: Session, samples: Dict[int, np.ndarray], model_dir: str | Path) -> List[Tuple[int, float, float]]:
    """Returns list of (player_id, ks_stat, ks_pvalue) for PIT ~ U(0,1)."""
    player_models = None
    try:
        player_models = load_player_models(model_dir)
    except Exception:
        pass

    results = []
    # build id->player map once
    from ivan.sports.nfl.models import NFLPlayer
    players = {p.id: p for p in session.query(NFLPlayer).all()}

    for pid, x in samples.items():
        player = players.get(pid)
        if player is None:
            continue
        u = _player_cdf(x, player, session, player_models)
        # KS against Uniform(0,1)
        ks = st.kstest(u, 'uniform')
        results.append((pid, float(ks.statistic), float(ks.pvalue)))
    return results

def validate_quantiles(session: Session, samples: Dict[int, np.ndarray], model_dir: str | Path) -> List[Tuple[int, float]]:
    """For each player, compare empirical quantiles vs theoretical PPF at a grid; return median absolute error."""
    player_models = None
    try:
        player_models = load_player_models(model_dir)
    except Exception:
        pass

    from ivan.sports.nfl.models import NFLPlayer
    players = {p.id: p for p in session.query(NFLPlayer).all()}

    grid = np.linspace(0.05, 0.95, 19)  # avoid extreme tails
    out = []
    for pid, x in samples.items():
        player = players.get(pid)
        if player is None or x.size < 100:
            continue
        x_sorted = np.sort(x)
        emp_q = np.quantile(x_sorted, grid)
        dp = _player_dist(player)
        if dp:
            dist, params = dp
            th_q = dist.ppf(grid, *params)
        else:
            # fall back to empirical from simulated fpts
            from ivan.sports.nfl.fpts_distribution_calculations import simulate_player_fpts
            sim = np.asarray(simulate_player_fpts(session, pid, player_models[0], player_models[1], n_samples=10000))
            th_q = np.quantile(sim, grid)
        mae = float(np.median(np.abs(emp_q - th_q)))
        out.append((pid, mae))
    return out

def validate_dependence(session: Session, samples: Dict[int, np.ndarray], model_dir: str | Path, alpha: float) -> Dict[str, dict]:
    """
    For each team, convert player samples -> PIT uniforms, aggregate by position,
    and compare pairwise Spearman correlations against vine (expected ≈ alpha^2 * vine_corr).
    Returns {team: {"emp_corr": 4x4, "expected_corr": 4x4}}
    """
    # load vines
    from ivan.sports.nfl.correlated_samples import _load_team_vines
    vines = _load_team_vines(Path(model_dir))
    # for PIT
    player_models = None
    try:
        player_models = load_player_models(model_dir)
    except Exception:
        pass

    teams = _group_players_by_team_pos(session)
    from ivan.sports.nfl.models import NFLPlayer
    players = {p.id: p for p in session.query(NFLPlayer).all()}

    pos_idx = {p: i for i, p in enumerate(_POSITIONS)}
    results = {}

    for team, pos_players in teams.items():
        # Build position-level uniforms by averaging PITs across players in the pos
        # (robust to having multiple WRs/RBs on slate)
        pos_U = []
        for pos in _POSITIONS:
            plist = pos_players.get(pos, [])
            if not plist:
                pos_U.append(None)
                continue
            # stack PITs for this position (align by sample index)
            pit_stack = []
            for pl in plist:
                x = samples.get(pl.id)
                if x is None:
                    continue
                u = _player_cdf(x, players[pl.id], session, player_models)
                pit_stack.append(u)
            if not pit_stack:
                pos_U.append(None)
                continue
            Upos = np.mean(np.vstack(pit_stack), axis=0)  # aggregate to a single series per position
            pos_U.append(Upos)

        # compute empirical Spearman corr on available positions
        valid = [(i, u) for i, u in enumerate(pos_U) if u is not None]
        if len(valid) < 2:
            continue
        mat = np.eye(len(_POSITIONS))
        for i, ui in valid:
            for j, uj in valid:
                if j <= i: 
                    continue
                rho = st.spearmanr(ui, uj).correlation
                mat[i, j] = mat[j, i] = float(rho)

        # expected corr: alpha^2 * vine Spearman (approx; vine library may expose Kendall tau—scale accordingly)
        exp = np.zeros_like(mat)
        vine = vines.get(team)
        if vine is not None and hasattr(vine, "tau"):
            # vine.tau() may return Kendall's tau matrix; convert to Spearman approx: rho_s ≈ (3/2) * tau
            tau = np.asarray(vine.tau())
            rho_s = 1.5 * tau
            exp = (alpha ** 2) * rho_s
        results[team] = {"emp_corr": mat, "expected_corr": exp}
    return results
