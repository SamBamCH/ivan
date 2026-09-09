from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import json, joblib, numpy as np
import scipy.stats as st
from sqlalchemy.orm import Session
from sqlalchemy import and_
import pyvinecopulib as pv

from ivan.sports.nfl.paths import MODELS_DIR
from ivan.sports.nfl.models import NFLPlayer

# Player marginals
try:
    from ivan.sports.nfl.fpts_distribution_calculations import load_models as load_player_models, simulate_player_fpts
except Exception:
    from ivan.sports.nfl.stat_distribution_calculations import load_models as load_player_models, simulate_player_fpts

TEAM10 = ["QB","RB1","RB2","WR1","WR2","WR3","TE1","TE2","K","DST"]
QB_SLOT, RB_SLOT, WR_SLOT, TE_SLOT, DST_SLOT = 66, 67, 68, 69, 71

def generate_correlated_samples_gaussian(
    session: Session,
    model_dir: str | Path = MODELS_DIR,
    n_samples: int = 1000,
    seed: int | None = None,
    use_hgv20: bool = False,  # Default to team-specific now
    use_team_specific_corr: bool = True,  # NEW: use team-specific correlations from historical data
    randomness: float = 1.0,
) -> Dict[int, np.ndarray]:
    """
    Simpler sampler: treat all fpts marginals as Normal(μ_i, σ_i) and draw
    a single multivariate normal per *game* using a slot-structured covariance.

    Requires NFLPlayer.slot_label ∈ {QB,RB1,RB2,WR1,WR2,WR3,TE1,TE2,K,DST}.
    Players deeper than modeled slots should have been pooled at fit time.

    Args:
        use_team_specific_corr: If True (default), build correlation matrices from each team's
                                historical data. This captures team-specific offensive structures.
        use_hgv20: If True (and use_team_specific_corr=False), use HGV-20 vine copula correlations.
        randomness: Controls distribution spread (0.0 = deterministic at median, 1.0 = full distribution).
                    Values are clamped to [0.0, 1.0].
    """
    from pathlib import Path
    import numpy as np
    import scipy.stats as st
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    model_dir = Path(model_dir)

    # Clamp randomness to [0.0, 1.0]
    randomness = float(np.clip(randomness, 0.0, 1.0))

    # --- 1) load all players on the slate with means/sigmas/slot_label ---
    q = session.query(NFLPlayer)
    players = q.all()
    if not players:
        return {}

    # Build (team -> opp) via first seen opponent
    team_to_opp = {}
    by_team = defaultdict(list)
    for p in players:
        by_team[p.team].append(p)
        if p.team not in team_to_opp and p.opponent:
            team_to_opp[p.team] = p.opponent

    seen = set()
    matchups = []
    for t, o in team_to_opp.items():
        if not o: 
            continue
        a, b = sorted([t, o])
        if (a, b) in seen: 
            continue
        seen.add((a, b)); matchups.append((a, b))

    # --- 2) slot-slot correlations ---
    TEAM10 = ["QB","RB1","RB2","WR1","WR2","WR3","TE1","TE2","K","DST"]
    
    # Spearman -> Pearson (bivariate normal relationship)
    def s2p(Rs: np.ndarray) -> np.ndarray:
        return 2.0 * np.sin(np.clip(np.pi * Rs / 6.0, -np.pi/2, np.pi/2))

    # Cache for team-specific correlations (avoid recomputing)
    team_corr_cache: Dict[str, np.ndarray] = {}
    cross_corr_cache: Dict[Tuple[str, str], np.ndarray] = {}
    
    if use_team_specific_corr:
        # Import team correlation builder
        from ivan.sports.nfl.refactor.team_correlation import build_team_correlation, build_cross_team_correlation
        print("Using team-specific correlations from historical data")
    elif use_hgv20:
        # compute once from your HGV-20 (vine+factors) the two 10x10 Spearman matrices
        from .correlated_sampling_validation import print_hgv20_spearman
        hgv = print_hgv20_spearman(model_dir, n=20_000)

        # Accept (within, cross) or (within, cross, order) or a dict
        if isinstance(hgv, tuple):
            if len(hgv) == 2:
                Rw_S, Rx_S = hgv
            elif len(hgv) >= 3:
                Rw_S, Rx_S = hgv[0], hgv[1]
            else:
                raise RuntimeError("print_hgv20_spearman returned an unexpected tuple length.")
        elif isinstance(hgv, dict):
            Rw_S = np.asarray(hgv["within"], dtype=float)
            Rx_S = np.asarray(hgv["cross"], dtype=float)
        else:
            raise RuntimeError("print_hgv20_spearman returned an unexpected type.")
            
        Rw = s2p(Rw_S)
        Rx = s2p(Rx_S)
        np.fill_diagonal(Rw, 1.0)
    else:
        # Or load a JSON/NPZ you precomputed and checked in
        d = json.loads((Path(model_dir) / "team10_spearman.json").read_text())
        Rw_S = np.array(d["within"], float)
        Rx_S = np.array(d["cross"], float)
        Rw = s2p(Rw_S)
        Rx = s2p(Rx_S)
        np.fill_diagonal(Rw, 1.0)

    slot_idx = {s:i for i,s in enumerate(TEAM10)}

    # --- 3) sample per game ---
    out: Dict[int, np.ndarray] = {}

    def near_spd(A: np.ndarray) -> np.ndarray:
        # Simple Higham projection (very small & fast for d<=40)
        B = (A + A.T) / 2
        eigval, eigvec = np.linalg.eigh(B)
        eigval[eigval < 1e-8] = 1e-8
        return (eigvec * eigval) @ eigvec.T

    for a, b in matchups:
        A = [p for p in by_team.get(a, []) if p.slot_label in slot_idx]
        B = [p for p in by_team.get(b, []) if p.slot_label in slot_idx]
        if not A and not B:
            continue

        # Order players (stable): slot order first, then by mean desc within slot
        def sort_team(pl):
            d = {}
            for s in TEAM10:
                group = [p for p in pl if p.slot_label == s]
                group.sort(key=lambda p: (p.fpts_mean if p.fpts_mean is not None else -1e9), reverse=True)
                for p in group:
                    d.setdefault(s, []).append(p)
            # Flatten but keep slot blocks in TEAM10 order
            seq = []
            for s in TEAM10:
                seq.extend(d.get(s, []))
            return seq

        A = sort_team(A); B = sort_team(B)
        players_game = A + B
        if not players_game:
            continue

        # Build μ and σ, falling back to slot-level sigma if missing
        mu = []
        sigma = []
        slot_sigma_default = {
            "QB": 6.0, "RB1": 6.0, "RB2": 5.5, "WR1": 6.5, "WR2": 6.0, "WR3": 5.5,
            "TE1": 5.5, "TE2": 5.0, "K": 3.0, "DST": 6.0,
        }
        for p in players_game:
            # fpts_dist_params for norm is (mu, sigma)
            prm = tuple(p.fpts_dist_params) if isinstance(p.fpts_dist_params, list) else p.fpts_dist_params
            mu_i = float(p.fpts_mean if p.fpts_mean is not None else (prm[0] if prm else 0.0))
            sd_i = float(prm[1]) if (prm and len(prm) >= 2 and prm[1] is not None) else float(slot_sigma_default[p.slot_label])
            mu.append(mu_i); sigma.append(sd_i)
        mu = np.array(mu, dtype=float)
        sigma = np.array(sigma, dtype=float)

        # Build the *correlation* matrix by slots — block structure:
        dA, dB = len(A), len(B)
        # Map each player to its slot index (first occurrence uses the slot's corr; extra players in same slot share that corr)
        slots_A = [slot_idx[p.slot_label] for p in A]
        slots_B = [slot_idx[p.slot_label] for p in B]

        # Get correlation matrices (team-specific or generic)
        if use_team_specific_corr:
            # Build/retrieve team-specific correlations
            if a not in team_corr_cache:
                Rw_a_S, _ = build_team_correlation(session, a)
                team_corr_cache[a] = s2p(Rw_a_S)
                np.fill_diagonal(team_corr_cache[a], 1.0)
            if b not in team_corr_cache:
                Rw_b_S, _ = build_team_correlation(session, b)
                team_corr_cache[b] = s2p(Rw_b_S)
                np.fill_diagonal(team_corr_cache[b], 1.0)
            
            cross_key = (a, b) if a < b else (b, a)
            if cross_key not in cross_corr_cache:
                Rx_S, _ = build_cross_team_correlation(session, a, b)
                cross_corr_cache[cross_key] = s2p(Rx_S)
            
            Rw_a = team_corr_cache[a]
            Rw_b = team_corr_cache[b]
            Rx_ab = cross_corr_cache[cross_key]
            
            # Within-team corr for A and B (using team-specific matrices)
            Corr_A = Rw_a[np.ix_(slots_A, slots_A)]
            Corr_B = Rw_b[np.ix_(slots_B, slots_B)]
            # Cross-team corr
            Corr_X = Rx_ab[np.ix_(slots_A, slots_B)]
        else:
            # Use generic (HGV20 or JSON) correlations
            Corr_A = Rw[np.ix_(slots_A, slots_A)]
            Corr_B = Rw[np.ix_(slots_B, slots_B)]
            Corr_X = Rx[np.ix_(slots_A, slots_B)]

        # Assemble full correlation:
        Corr = np.block([[Corr_A, Corr_X],
                         [Corr_X.T, Corr_B]])
        # Clamp numeric noise and project to SPD if needed
        Corr = np.clip(Corr, -0.999, 0.999)
        Corr = near_spd(Corr)

        # Covariance Σ = D * Corr * D
        D = np.diag(sigma)
        Sigma = D @ Corr @ D

        # Apply randomness scaling to covariance (scale variance by randomness²)
        Sigma_scaled = Sigma * (randomness ** 2)

        # Sample
        Z = rng.multivariate_normal(mean=np.zeros(len(players_game)), cov=Sigma_scaled, size=n_samples)
        X = (Z + mu)  # broadcasting: add μ row-wise

        # Store back by player id (apply 1.5x multiplier for captains)
        for j, p in enumerate(players_game):
            samples = X[:, j].astype(np.float32, copy=False)
            # Apply 1.5x multiplier for captain slot (roster_slot_id == 511)
            if p.roster_slot_id == 511:
                samples = samples * 1.5
            out[p.id] = samples

    # Teams with no paired opponent: fall back to independent N(μ,σ)
    for p in players:
        if p.id not in out:
            prm = tuple(p.fpts_dist_params) if isinstance(p.fpts_dist_params, list) else p.fpts_dist_params
            mu_i = float(p.fpts_mean if p.fpts_mean is not None else (prm[0] if prm else 0.0))
            sd_i = float(prm[1]) if (prm and len(prm) >= 2 and prm[1] is not None) else 6.0
            # Apply randomness scaling to standard deviation
            sd_scaled = sd_i * randomness
            samples = rng.normal(loc=mu_i, scale=sd_scaled, size=n_samples).astype(np.float32, copy=False)
            # Apply 1.5x multiplier for captain slot (roster_slot_id == 511)
            if p.roster_slot_id == 511:
                samples = samples * 1.5
            out[p.id] = samples

    return out
