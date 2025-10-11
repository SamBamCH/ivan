#!/usr/bin/env python3
# scripts/nfl/sample_field.py
"""
NFL field sampler with support for both Showdown and Classic formats.

Showdown: Uses captain-aware sampling with salary bins and template penalties
Classic: Weighted sampling based on projections with optional duplication

Key features:
- Learned duplication model for Showdown (uses historical patterns)
- Heuristic sampler with diversity optimization
- Captain ownership and template diversity controls
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
from scipy import sparse as sp
from sqlalchemy.orm import Session

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLPlayer, Lineup
from ivan.sports.nfl.correlated_samples import generate_correlated_samples_gaussian

# CLASSIC SAMPLER (NO DUPLICATION)
# ==============================================================================

def sample_classic_field(
    session: Session,
    field_size: int,
    config: dict,
    contest_id: Optional[int] = None,
) -> Tuple[Dict[int, int], Dict]:
    """
    Classic sampler: weighted sampling WITHOUT replacement (unique lineups only).
    
    Strategy:
    1. Build lineup matrix and get player projections
    2. Calculate lineup scores (using mean/median/percentile of player sims)
    3. Apply temperature-based softmax to create weights
    4. Sample without replacement for unique lineups
    
    Config options (under classic.sampling):
        statistic: "mean", "median", or "pNN" (e.g. "p75" for 75th percentile)
        temperature: Controls chalkiness (lower = more chalk, higher = more spread)
        epsilon: Floor probability to avoid completely starving tail lineups
    """
    def _get(d: dict, path: List[str], default=None):
        """Helper to safely navigate nested config dicts."""
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    statistic = str(_get(config, ["classic", "sampling", "statistic"], "median")).lower()
    temperature = float(_get(config, ["classic", "sampling", "temperature"], 1.10))
    epsilon = float(_get(config, ["classic", "sampling", "epsilon"], 0.02))

    # Build lineup matrix
    L, lineup_ids, spends, pid_to_col, _, _ = build_lineup_matrix(session)
    n_lineups, n_players = L.shape

    # Get player simulations
    samples_by_pid = generate_correlated_samples_gaussian(session, n_samples=100)

    # Reduce to per-player projection
    def _reduce(arr: np.ndarray) -> float:
        """Reduce player sample array to single projection value."""
        if statistic == "mean":
            return float(np.mean(arr))
        if statistic.startswith("p"):
            try:
                q = float(statistic[1:]) / 100.0
                return float(np.quantile(arr, q))
            except Exception:
                pass
        return float(np.median(arr))

    proj_vec = np.zeros(n_players, dtype=np.float32)
    present = np.zeros(n_players, dtype=bool)
    
    for pid, col in pid_to_col.items():
        arr = samples_by_pid.get(pid)
        if isinstance(arr, np.ndarray) and arr.size:
            proj_vec[col] = _reduce(arr)
            present[col] = True

    # Filter out lineups with missing player projections
    missing_mask = (~present).astype(np.int8)
    invalid_counts = np.asarray(L @ missing_mask, dtype=np.int32).ravel()
    valid_row_idx = np.where(invalid_counts == 0)[0]
    
    if valid_row_idx.size == 0:
        raise ValueError("No valid classic lineups after filtering missing projections.")

    # Calculate lineup scores and apply temperature
    lineup_scores_all = np.asarray(L @ proj_vec, dtype=np.float64).ravel()
    s = lineup_scores_all[valid_row_idx]

    mu = float(s.mean())
    sd = float(s.std()) or 1.0
    T = max(temperature, 1e-6)

    # Temperature-scaled logits
    logits = (s - mu) / (sd * T)
    logits -= float(logits.max())  # Numerical stability
    w = np.exp(logits)
    
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w /= w_sum

    # Add epsilon floor to avoid starving tail
    eps = max(0.0, min(0.5, epsilon))
    if eps > 0:
        w = (1.0 - eps) * w + eps / len(w)

    # Sample without replacement
    k = int(min(field_size, len(valid_row_idx)))
    rng = np.random.default_rng()
    chosen_local = rng.choice(len(valid_row_idx), size=k, replace=False, p=w)
    chosen_rows = valid_row_idx[chosen_local]

    # Each lineup selected at most once
    counts = {int(lineup_ids[i]): 1 for i in chosen_rows}

    meta = {
        "sampler": "classic_unique",
        "num_candidates": int(n_lineups),
        "num_valid": int(len(valid_row_idx)),
        "selected": int(k),
        "temperature": float(temperature),
        "epsilon": float(eps),
        "score_mean": float(s.mean()),
        "score_std": float(s.std()),
    }
    
    return counts, meta

# ==============================================================================
# CLASSIC SAMPLER (WITH DUPLICATION)
# ==============================================================================

def _minmax01(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-12:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _smooth_lam(dupe_lambda: np.ndarray, *,
                strength: float = 0.6,
                sigma: float = 0.18,
                eps_frac: float = 0.06) -> np.ndarray:
    """
    Smooth duplication lambda to fatten the middle percentiles.
    
    Applies a Gaussian bump centered at median to reduce extreme concentration
    on the very top chalk lineups.
    """
    def _percentiles_01(x: np.ndarray) -> np.ndarray:
        r = np.argsort(np.argsort(x))
        return (r + 0.5) / float(x.size)
    
    lam = np.clip(dupe_lambda.astype(np.float64), 1e-12, None)
    p = _percentiles_01(lam)
    s = max(1e-3, float(sigma))
    temp = 1.0 + float(strength) * np.exp(-0.5 * ((p - 0.5) / s) ** 2)
    lam = np.power(lam, 1.0 / temp)
    lam += float(eps_frac) * lam.mean()
    return np.clip(lam, 1e-12, None)


def sample_classic_field_dupe(
    session: Session,
    field_size: int,
    config: dict | None = None,
    contest_id: Optional[int] = None,
) -> Tuple[Dict[int, int], Dict]:
    """
    Classic field sampler WITH duplication.
    
    Good for small slates (2-gamers) where duplication is realistic.
    
    Strategy:
    1. Use salary-remaining bins (like Showdown)
    2. Within each bin, weight by: projection^alpha * near_cap^beta
    3. Sample with replacement to allow duplicates
    
    Config options (under classic.dupe):
        alpha_fpts: Projection power (higher = more chalk)
        alpha_cap: Near-cap power (higher = more near-cap preference)
        epsilon: Floor weight inside bins
        dup_sharpness: >1 to concentrate on chalk
        smooth_mid: Whether to flatten middle percentiles
    """
    cfg = config or {}
    
    def _get(d: dict, path: List[str], default=None):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    alpha_fpts = float(_get(cfg, ["classic", "dupe", "alpha_fpts"], 1.25))
    alpha_cap = float(_get(cfg, ["classic", "dupe", "alpha_cap"], 0.90))
    epsilon = float(_get(cfg, ["classic", "dupe", "epsilon"], 1e-6))
    sharpness = float(_get(cfg, ["classic", "dupe", "dup_sharpness"], 1.0))
    smooth_mid = bool(_get(cfg, ["classic", "dupe", "smooth_mid"], True))
    mid_strength = float(_get(cfg, ["classic", "dupe", "mid_boost_strength"], 0.5))
    mid_sigma = float(_get(cfg, ["classic", "dupe", "mid_boost_sigma"], 0.18))
    lam_floor = float(_get(cfg, ["classic", "dupe", "lam_floor_frac"], 0.02))
    seed = _get(cfg, ["classic", "dupe", "seed"], None)

    rng = np.random.default_rng(seed)

    # Build lineup matrix
    L, lineup_ids, spends, pid_to_col, _, _ = build_lineup_matrix(session)
    n_lineups = L.shape[0]
    remaining = (SALARY_CAP - spends).astype(np.int32)

    # Get player projections
    P = build_player_samples(session, pid_to_col)
    per_player_mean = np.asarray(P.mean(axis=0), dtype=np.float64)
    lineup_proj = np.asarray(L @ per_player_mean, dtype=np.float64)

    # Normalized features
    proj_pct = _percentile_rank(lineup_proj, lineup_proj)
    rem01 = _minmax01(remaining)
    near_cap = np.clip(1.0 - rem01, 0.0, 1.0)

    # Base duplication lambda
    lam = np.power(np.clip(proj_pct, 1e-12, 1.0), alpha_fpts) * \
          np.power(np.clip(near_cap, 1e-12, 1.0), alpha_cap)

    if sharpness > 1.0:
        lam = np.power(lam, sharpness)

    if smooth_mid:
        lam = _smooth_lam(lam, strength=mid_strength, sigma=mid_sigma, eps_frac=lam_floor)

    lam = np.clip(lam, 1e-12, None)

    # Salary-remaining bins
    bins = list(SALARY_PRIOR)
    bin_masks = [(remaining >= lo) & (remaining < hi) for lo, hi, _ in bins]

    weights = np.array([w for _, _, w in bins], dtype=np.float64)
    has_any = np.array([m.any() for m in bin_masks], dtype=bool)
    
    if not has_any.any():
        raise RuntimeError("No lineups in any salary bin. Check SALARY_PRIOR.")
    
    w_avail = weights * has_any.astype(np.float64)
    if w_avail.sum() <= 0:
        w_avail = has_any.astype(np.float64)
    w_avail /= w_avail.sum()

    draws_per_bin = rng.multinomial(int(field_size), w_avail)
    idx_per_bin = [np.where(m)[0] for m in bin_masks]

    # Sample with replacement inside each bin
    counts = np.zeros(n_lineups, dtype=np.int64)
    
    for b, n_draws in enumerate(draws_per_bin):
        if n_draws <= 0:
            continue
        
        idxs = idx_per_bin[b]
        if idxs.size == 0:
            continue
        
        w = lam[idxs].astype(np.float64, copy=True)
        w += epsilon * float(w.mean() if w.size else 1.0)
        
        p = _normalize_probs(w, idxs.size)
        if p is None:
            p = np.full(idxs.size, 1.0 / idxs.size, dtype=np.float64)

        picks = rng.choice(idxs.size, size=int(n_draws), replace=True, p=p)
        np.add.at(counts, idxs[picks], 1)

    counts_by_lineup = {int(lineup_ids[i]): int(counts[i]) for i in range(n_lineups)}

    # Meta diagnostics
    used_mask = counts > 0
    unique_ids = lineup_ids[used_mask].astype(np.int64).tolist()
    
    meta = {
        "sampler": "classic_dupe",
        "field_size": int(field_size),
        "num_unique": int(used_mask.sum()),
        "unique_ids": unique_ids,
        "proj_pct_stats": {
            "mean": float(proj_pct.mean()), 
            "std": float(proj_pct.std())
        },
        "near_cap_stats": {
            "mean": float(near_cap.mean()), 
            "std": float(near_cap.std())
        },
        "lam_stats": {
            "min": float(lam.min()), 
            "max": float(lam.max()), 
            "mean": float(lam.mean())
        },
    }
    
    return counts_by_lineup, meta

# ==============================================================================
# UNIFIED ENTRY POINT
# ==============================================================================

def sample_field(
    session: Session,
    field_size: int,
    config: dict,
    mode: str,
    contest_id: Optional[int] = None,
    use_classic_dupes: bool = False,
) -> Tuple[Dict[int, int], Dict]:
    """
    Unified field sampler entry point.
    
    Args:
        session: Database session
        field_size: Number of entries to generate
        config: Configuration dictionary
        mode: "showdown" or "classic"
        contest_id: Optional contest ID for parameters
        use_classic_dupes: If True, allow duplication in classic mode
    
    Returns:
        counts_by_lineup: {lineup_id: count}
        meta: Diagnostic information
    """
    mode = (mode or "showdown").lower()
    
    if mode == "showdown":
        # Check if learned model is enabled
        use_learned = config.get('showdown', {}).get('use_learned_model', True)
        
        if use_learned:
            print("Using learned duplication model for showdown.")
            return sample_showdown_field_learned(
                session,
                field_size=field_size,
                config=config,
                contest_id=contest_id,
                model_dir=config.get('model_dir', 'data/nfl/models'),
            )
        else:
            print("Using heuristic sampler for showdown.")
            return sample_showdown_field_heuristic(
                session, 
                field_size=field_size, 
                config=config
            )
    
    # Classic mode
    if use_classic_dupes:
        print("Using classic sampler with duplication.")
        return sample_classic_field_dupe(
            session, 
            field_size=field_size, 
            config=config, 
            contest_id=contest_id
        )
    else:
        print("Using classic sampler (unique lineups only).")
        return sample_classic_field(
            session, 
            field_size=field_size, 
            config=config, 
            contest_id=contest_id
        )


# ==============================================================================
# MAIN (FOR TESTING)
# ==============================================================================

if __name__ == "__main__":
    """Quick test of samplers."""
    from ivan.database.session import get_session
    
    with get_session() as session:
        # Test showdown sampler
        config = {
            'showdown': {
                'use_learned_model': False,  # Use heuristic for testing
            }
        }
        
        print("Testing Showdown sampler...")
        counts, meta = sample_field(
            session=session,
            field_size=1000,
            config=config,
            mode='showdown'
        )
        
        print(f"\nResults:")
        print(f"  Unique lineups: {meta.get('unique_selected', 'N/A')}")
        print(f"  Total entries: {sum(counts.values())}")
        
        # Test classic sampler
        print("\n" + "="*60)
        print("Testing Classic sampler...")
        counts, meta = sample_field(
            session=session,
            field_size=1000,
            config={},
            mode='classic',
            use_classic_dupes=False
        )
        
        print(f"\nResults:")
        print(f"  Unique lineups: {meta.get('selected', 'N/A')}")
        print(f"  Total entries: {sum(counts.values())}")
# CONSTANTS
# ==============================================================================

SALARY_CAP = 50000
CPT_SLOT = 511
FLEX_SLOT = 512

# ==============================================================================
# SHOWDOWN SAMPLING PARAMETERS
# ==============================================================================
# Control field distribution for Showdown contests
# Formula: weight = (projection ** ALPHA) * (cpt_percentile ** BETA) * template_penalty
# template_penalty = exp(-GAMMA * max(0, flex_chalk - TAU) * (1 - cpt_pct))

ALPHA = 1.6   # Projection power (1.0-2.0): higher = more chalk
BETA = 1.2    # Captain popularity power (0.5-1.5): higher = more captain concentration
GAMMA = 2.0   # Template penalty strength (1.0-4.0): higher = stronger anti-template bias
TAU = 0.90    # Template chalk threshold (0.5-0.85): when to start penalizing chalk FLEXes

# Preset configurations:
# CHALK:       ALPHA=1.5, BETA=1.0, GAMMA=1.0, TAU=0.80
# CONTRARIAN:  ALPHA=0.9, BETA=0.6, GAMMA=3.0, TAU=0.60
# BALANCED:    ALPHA=1.2, BETA=0.8, GAMMA=2.0, TAU=0.70

# Salary remaining bins: [low, high, weight]
# Distribution over how much salary is left unused
SALARY_PRIOR: List[Tuple[int, int, float]] = [
    (0,    200,  0.30),   # Near cap (< $200 remaining)
    (200,  300,  0.20),
    (300,  400,  0.13),
    (400,  500,  0.13),
    (500,  600,  0.07),
    (600,  700,  0.05),
    (700,  800,  0.03),
    (800,  900,  0.03),
    (900,  1000, 0.03),
    (1000, 10**9, 0.03),  # >= $1000 remaining
]

# ==============================================================================
# DEFAULT CONFIGURATIONS
# ==============================================================================

DEFAULTS = {
    "pool": {
        "size": 10000,
        "embed_dims": 24,      # JL embedding dimensions for diversity
        "beta_fpts": 0.5,      # Quality vs diversity tradeoff
        "seed": None
    },
    "dupe_model": {
        "theta0": -1.8,                 # Base log-rate
        "theta_remain": -0.6,           # Salary remaining coefficient
        "theta_cpt": 1.0,               # Captain strength coefficient
        "theta_fpts": 1.3,              # Projection coefficient
        "min_count": 1,
        "mid_boost_strength": 0.6,
        "mid_boost_sigma": 0.18,
        "lam_floor_frac": 0.06,
        "stochastic_rounding": True
    },
    "independent_draws": {
        "n_sims": 64
    }
}

# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_showdown_models(model_dir: str | Path = "data/nfl/models"):
    """Load trained showdown duplication models from disk."""
    model_dir = Path(model_dir)
    
    try:
        with open(model_dir / 'showdown_duplication_model.pkl', 'rb') as f:
            dupe_model = pickle.load(f)
        
        with open(model_dir / 'showdown_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open(model_dir / 'showdown_feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        
        return dupe_model, scaler, feature_cols
    
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Showdown models not found in {model_dir}. "
            f"Run 'python scripts/nfl/train_showdown_model.py' first."
        ) from e

# ==============================================================================
# LINEUP MATRIX CONSTRUCTION
# ==============================================================================

def _coerce_player_ids(val) -> List[int]:
    """Extract player IDs from various formats (list, JSON string, etc)."""
    if val is None:
        return []
    if isinstance(val, list):
        return [int(x) for x in val]
    if isinstance(val, str):
        try:
            data = json.loads(val)
            if isinstance(data, list):
                return [int(x) for x in data]
        except Exception:
            return []
    return []


def build_lineup_matrix(session: Session):
    """
    Build sparse lineup matrix and metadata.
    
    Returns:
        L: csr_matrix [n_lineups x n_players] - binary lineup membership
        lineup_ids: array of lineup IDs
        spends: array of total salary per lineup
        pid_to_col: dict mapping player_id -> column index
        cpt_col_by_row: array [n_lineups] -> captain column index (or -1)
        flex_cols_by_row: list of lists - FLEX column indices per lineup
    """
    rows = session.query(Lineup.id, Lineup.player_ids_list).all()
    if not rows:
        raise RuntimeError("No lineups found in database.")
    
    lineup_ids, parsed, all_pids = [], [], set()
    for lid, player_ids_json in rows:
        pids = _coerce_player_ids(player_ids_json)
        parsed.append((lid, pids))
        lineup_ids.append(lid)
        all_pids.update(pids)

    lineup_ids = np.array(lineup_ids, dtype=np.int64)
    pid_list = sorted(all_pids)
    pid_to_col = {pid: i for i, pid in enumerate(pid_list)}

    # Build sparse lineup matrix
    nnz = sum(len(pids) for _, pids in parsed)
    row_indices = np.empty(nnz, dtype=np.int32)
    col_indices = np.empty(nnz, dtype=np.int32)
    values = np.ones(nnz, dtype=np.float32)
    
    k = 0
    for i, (_, pids) in enumerate(parsed):
        for pid in pids:
            row_indices[k] = i
            col_indices[k] = pid_to_col[pid]
            k += 1
    
    L = sp.csr_matrix((values, (row_indices, col_indices)), 
                      shape=(len(parsed), len(pid_list)))

    # Get player metadata (salary and roster slot)
    player_data = session.query(
        NFLPlayer.id, 
        NFLPlayer.salary, 
        NFLPlayer.roster_slot_id
    ).filter(NFLPlayer.id.in_(pid_list)).all()
    
    salaries = {}
    roster_slot = {}
    for pid, sal, slot in player_data:
        salaries[int(pid)] = int(sal)
        roster_slot[int(pid)] = int(slot) if slot is not None else -1

    # Calculate lineup spends
    salary_vec = np.array([int(salaries.get(pid, 0)) for pid in pid_list], dtype=np.int32)
    spends = (L @ salary_vec).astype(np.int32)

    # Identify captain and flex positions
    cpt_col_by_row = np.full(L.shape[0], -1, dtype=np.int32)
    flex_cols_by_row = [[] for _ in range(L.shape[0])]
    
    for row_idx, (_, pids) in enumerate(parsed):
        for pid in pids:
            col = pid_to_col[pid]
            slot = roster_slot.get(pid, -1)
            if slot == CPT_SLOT:
                cpt_col_by_row[row_idx] = col
            elif slot == FLEX_SLOT or slot == -1:
                flex_cols_by_row[row_idx].append(col)

    return L, lineup_ids, spends, pid_to_col, cpt_col_by_row, flex_cols_by_row

# ==============================================================================
# PLAYER SAMPLES
# ==============================================================================

def build_player_samples(session: Session, pid_to_col: Dict[int, int]) -> np.ndarray:
    """
    Build correlated player samples matrix [n_sims x n_players].
    Uses the correlated_samples module for realistic covariance.
    """
    raw = generate_correlated_samples_gaussian(session)
    sizes = [len(x) for x in raw.values() if isinstance(x, np.ndarray)]
    if not sizes:
        raise RuntimeError("No player samples generated.")
    
    n_sims = min(sizes)
    P = np.zeros((n_sims, len(pid_to_col)), dtype=np.float32)
    
    for pid, col in pid_to_col.items():
        vec = raw.get(pid)
        if isinstance(vec, np.ndarray) and vec.ndim == 1 and len(vec) >= n_sims:
            P[:, col] = vec[:n_sims].astype(np.float32)
    
    return P


def build_player_samples_independent(
    session: Session, 
    pid_to_col: Dict[int, int], 
    n_sims: int | None = None,
    seed: int | None = None
) -> np.ndarray:
    """
    Build independent (uncorrelated) player samples by shuffling each player separately.
    This breaks correlations while preserving individual player distributions.
    
    Used for popularity-based features where we want player means without correlation effects.
    """
    raw = generate_correlated_samples_gaussian(session)
    sizes = [len(v) for v in raw.values() if isinstance(v, np.ndarray) and v.size]
    if not sizes:
        raise RuntimeError("No player samples available.")
    
    base_ns = int(min(sizes))
    ns = int(n_sims or base_ns)

    P = np.zeros((ns, len(pid_to_col)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    
    for pid, col in pid_to_col.items():
        v = raw.get(pid)
        if not isinstance(v, np.ndarray) or v.size == 0:
            continue
        
        w = v.astype(np.float32, copy=False)
        
        # Resample to match desired n_sims
        if w.shape[0] < ns:
            reps = int(np.ceil(ns / w.shape[0]))
            w = np.tile(w, reps)[:ns]
        else:
            w = w[:ns]
        
        # Shuffle to break correlations
        rng.shuffle(w)
        P[:, col] = w
    
    return P

# ==============================================================================
# OWNERSHIP EXTRACTION
# ==============================================================================

def get_player_ownership(session: Session, pid_to_col: Dict[int, int]) -> np.ndarray:
    """
    Extract actual player ownership from NFLPlayer table.
    
    Returns array [n_players] with ownership percentages (0-100).
    Missing ownership defaults to 0.
    """
    # Try both column names (ownership_proj and ownership)
    player_data = session.query(
        NFLPlayer.id,
        NFLPlayer.ownership_proj  # Changed from ownership to ownership_proj
    ).filter(NFLPlayer.id.in_(pid_to_col.keys())).all()
    
    ownership = np.zeros(len(pid_to_col), dtype=np.float64)
    missing_count = 0
    
    for pid, own in player_data:
        if own is not None and not np.isnan(own):
            ownership[pid_to_col[pid]] = float(own)
        else:
            missing_count += 1
    
    if missing_count > 0:
        print(f"  WARNING: {missing_count} players missing ownership_proj data")
    
    return ownership

# ==============================================================================
# FEATURE COMPUTATION FOR LEARNED MODEL
# ==============================================================================

def compute_lineup_features_for_model(
    lineup_ids: np.ndarray,
    L,  # sparse lineup matrix [n_lineups x n_players]
    spends: np.ndarray,
    cpt_col_by_row: np.ndarray,
    P_indep: np.ndarray,  # [n_sims x n_players] independent samples
    player_ownership: np.ndarray,  # [n_players] actual ownership
    total_entries: int,
    entry_fee: float,
    ownership_gini: float,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute feature matrix for trained duplication model.
    
    Features include:
    - Contest metadata (entries, fee, ownership distribution)
    - Correlation features (placeholder for now)
    - Captain ownership (actual, not projected)
    - Lineup projection strength
    - Total lineup ownership
    - Salary remaining
    - Interaction terms
    
    Returns: [n_lineups x n_features] array ready for model prediction
    """
    n_lineups = L.shape[0]
    
    if verbose:
        print("\n" + "="*70)
        print("FEATURE CALCULATION DEBUG")
        print("="*70)
        print(f"Number of lineups: {n_lineups:,}")
        print(f"Number of players: {L.shape[1]:,}")
        print(f"P_indep shape: {P_indep.shape}")
        print(f"player_ownership shape: {player_ownership.shape}")
        print(f"player_ownership stats: min={player_ownership.min():.2f}, "
              f"max={player_ownership.max():.2f}, mean={player_ownership.mean():.2f}")
        print(f"Non-zero ownership players: {(player_ownership > 0).sum()} / {len(player_ownership)}")
    
    # Per-player projection means from independent samples
    per_player_mean = np.asarray(P_indep.mean(axis=0), dtype=np.float64)
    
    if verbose:
        print(f"\nPer-player projection stats:")
        print(f"  min={per_player_mean.min():.2f}, max={per_player_mean.max():.2f}, "
              f"mean={per_player_mean.mean():.2f}")
    
    # 1. Lineup mean projections
    mean_proj = np.asarray(L @ per_player_mean, dtype=np.float64)
    
    if verbose:
        print(f"\nLineup projection stats:")
        print(f"  min={mean_proj.min():.2f}, max={mean_proj.max():.2f}, "
              f"mean={mean_proj.mean():.2f}")
    
    # 2. Captain projections
    cpt_means = np.zeros(n_lineups, dtype=np.float64)
    # This extracts captain's mean projection for each lineup
    # mask filters out lineups with invalid captain indices
    mask = cpt_col_by_row >= 0
    if mask.any():
        cpt_means[mask] = per_player_mean[cpt_col_by_row[mask]]
    
    if verbose:
        print(f"\nCaptain projection stats:")
        print(f"  Valid captains: {mask.sum()} / {n_lineups}")
        print(f"  min={cpt_means[mask].min():.2f}, max={cpt_means[mask].max():.2f}, "
              f"mean={cpt_means[mask].mean():.2f}")
    
    # 3. Captain ownership (THE FIX for your bug)
    cpt_ownership = np.zeros(n_lineups, dtype=np.float64)
    if mask.any():
        cpt_ownership[mask] = player_ownership[cpt_col_by_row[mask]]
    
    if verbose:
        print(f"\nCaptain ownership stats:")
        print(f"  min={cpt_ownership[mask].min():.2f}, max={cpt_ownership[mask].max():.2f}, "
              f"mean={cpt_ownership[mask].mean():.2f}")
        print(f"  Zero-ownership captains: {(cpt_ownership[mask] == 0).sum()} / {mask.sum()}")
        
        # Show some examples
        print(f"\nTop 5 captain ownership values:")
        top_cpt_idx = np.argsort(cpt_ownership[mask])[-5:]
        for idx in top_cpt_idx:
            lineup_row = np.where(mask)[0][idx]
            print(f"    Lineup {lineup_ids[lineup_row]}: captain_own={cpt_ownership[lineup_row]:.1f}%, "
                  f"captain_proj={cpt_means[lineup_row]:.1f}")
        
        print(f"\nBottom 5 captain ownership values:")
        bottom_cpt_idx = np.argsort(cpt_ownership[mask])[:5]
        for idx in bottom_cpt_idx:
            lineup_row = np.where(mask)[0][idx]
            print(f"    Lineup {lineup_ids[lineup_row]}: captain_own={cpt_ownership[lineup_row]:.1f}%, "
                  f"captain_proj={cpt_means[lineup_row]:.1f}")
    
    # 4. Total lineup ownership (sum of all players' ownership)
    lineup_total_ownership = np.asarray(L @ player_ownership, dtype=np.float64)
    
    if verbose:
        print(f"\nTotal lineup ownership stats:")
        print(f"  min={lineup_total_ownership.min():.2f}, max={lineup_total_ownership.max():.2f}, "
              f"mean={lineup_total_ownership.mean():.2f}")
        print(f"  Zero-ownership lineups: {(lineup_total_ownership == 0).sum()} / {n_lineups}")
    
    # 5. Salary remaining
    remaining = (SALARY_CAP - spends).astype(np.int32)
    salary_remaining_pct = remaining / SALARY_CAP
    
    if verbose:
        print(f"\nSalary remaining stats:")
        print(f"  min=${remaining.min()}, max=${remaining.max()}, mean=${remaining.mean():.0f}")
        print(f"  Percentage: min={salary_remaining_pct.min():.3f}, "
              f"max={salary_remaining_pct.max():.3f}, mean={salary_remaining_pct.mean():.3f}")
    
    # Convert to percentile ranks within this lineup pool
    def _percentile_rank(arr):
        """Convert values to percentile ranks [0, 1]."""
        if arr.size <= 1:
            return np.full_like(arr, 0.5, dtype=np.float64)
        sorted_arr = np.sort(arr)
        ranks = np.searchsorted(sorted_arr, arr, side='right')
        return ranks / float(arr.size)
    
    cpt_proj_percentile = _percentile_rank(cpt_means)
    cpt_own_percentile = _percentile_rank(cpt_ownership)
    proj_percentile = _percentile_rank(mean_proj)
    own_percentile = _percentile_rank(lineup_total_ownership)
    
    if verbose:
        print(f"\nPercentile conversions:")
        print(f"  Captain ownership percentiles: min={cpt_own_percentile.min():.3f}, "
              f"max={cpt_own_percentile.max():.3f}, mean={cpt_own_percentile.mean():.3f}")
        print(f"  Captain projection percentiles: min={cpt_proj_percentile.min():.3f}, "
              f"max={cpt_proj_percentile.max():.3f}, mean={cpt_proj_percentile.mean():.3f}")
        print(f"  Lineup ownership percentiles: min={own_percentile.min():.3f}, "
              f"max={own_percentile.max():.3f}, mean={own_percentile.mean():.3f}")
        print(f"  Lineup projection percentiles: min={proj_percentile.min():.3f}, "
              f"max={proj_percentile.max():.3f}, mean={proj_percentile.mean():.3f}")
    
    # Contest-level features
    log_total_entries = np.log10(max(1, total_entries))
    
    if verbose:
        print(f"\nContest features:")
        print(f"  log10(total_entries) = {log_total_entries:.3f} (from {total_entries:,})")
        print(f"  entry_fee = ${entry_fee:.2f}")
        print(f"  ownership_gini = {ownership_gini:.3f}")
    
    # Build feature matrix (must match training order!)
    features = []
    for i in range(n_lineups):
        # Placeholder correlation features (can be computed later)
        correlation_score = 0.0
        same_team_count = 6
        qb_wr_stack = 0
        rb_dst_negative = 0
        
        row = [
            # Contest features
            log_total_entries,
            entry_fee,
            ownership_gini,
            
            # Dimension 1: Correlation (placeholders)
            correlation_score,
            same_team_count,
            qb_wr_stack,
            rb_dst_negative,
            
            # Dimension 2: Captain ownership (USING ACTUAL OWNERSHIP NOW)
            cpt_own_percentile[i],
            
            # Dimension 3: Projection
            proj_percentile[i],
            
            # Dimension 4: Total ownership (USING ACTUAL OWNERSHIP NOW)
            own_percentile[i],
            
            # Dimension 5: Salary
            salary_remaining_pct[i],
            
            # Interactions
            cpt_own_percentile[i] * own_percentile[i],  # cpt_own_x_total_own
            1.0 - salary_remaining_pct[i],  # near_cap
            cpt_own_percentile[i] * own_percentile[i] * proj_percentile[i],  # chalk_score
        ]
        features.append(row)
    
    feature_array = np.array(features, dtype=np.float64)
    
    if verbose:
        print(f"\nFinal feature matrix shape: {feature_array.shape}")
        print(f"\nSample features for top 3 most-duped lineups (if available):")
        print(f"{'Feature':<25} {'Lineup 1':>12} {'Lineup 2':>12} {'Lineup 3':>12}")
        print("-" * 70)
        feature_names = [
            'log_total_entries',
            'entry_fee', 
            'ownership_gini',
            'correlation_score',
            'same_team_count',
            'qb_wr_stack',
            'rb_dst_negative',
            'cpt_own_pct',
            'proj_pct',
            'total_own_pct',
            'salary_remain_pct',
            'cpt_x_total_own',
            'near_cap',
            'chalk_score'
        ]
        for feat_idx, feat_name in enumerate(feature_names):
            vals = [f"{feature_array[i, feat_idx]:12.4f}" for i in range(min(3, n_lineups))]
            print(f"{feat_name:<25} {' '.join(vals)}")
        
        print("="*70 + "\n")
    
    # Return features and diagnostics for later analysis
    diagnostics = {
        'per_player_mean': per_player_mean,
        'mean_proj': mean_proj,
        'cpt_ownership': cpt_ownership,
        'lineup_total_ownership': lineup_total_ownership,
    }
    
    return feature_array, diagnostics

# ==============================================================================
# LEARNED MODEL SAMPLER (SHOWDOWN)
# ==============================================================================

def sample_showdown_field_learned(
    session: Session,
    field_size: int,
    config: dict,
    contest_id: Optional[int] = None,
    model_dir: str | Path = "data/nfl/models",
) -> Tuple[Dict[int, int], Dict]:
    """
    Sample showdown field using trained duplication models.
    
    Uses historical patterns to predict lineup duplication based on:
    - Contest size and entry fee
    - Captain and player ownership
    - Lineup projections and salary usage
    
    Returns:
        counts_by_lineup: {lineup_id: duplicate_count}
        meta: Diagnostic info
    """
    # Load trained models
    try:
        dupe_model, scaler, feature_cols = load_showdown_models(model_dir)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Falling back to heuristic sampler")
        return sample_showdown_field_heuristic(session, field_size=field_size, config=config)
    
    # Get contest parameters
    if contest_id:
        from ivan.sports.nfl.models import HistoricalContest
        contest = session.query(HistoricalContest).get(contest_id)
        
        if contest:
            total_entries = contest.total_entries
            entry_fee = contest.entry_fee
            # Calculate ownership gini from current player pool
            ownership_gini = 0.5  # Default if calculation fails
        else:
            total_entries = field_size
            entry_fee = 3.0
            ownership_gini = 0.5
    else:
        total_entries = field_size
        entry_fee = 3.0
        ownership_gini = 0.5
    
    # Build lineup matrix
    L, lineup_ids, spends, pid_to_col, cpt_col_by_row, _ = build_lineup_matrix(session)
    n_lineups = L.shape[0]
    
    if n_lineups == 0:
        print("  WARNING: No lineups found in database")
        return {}, {'error': 'no_lineups'}
    
    print(f"  Candidate lineups: {n_lineups:,}")
    
    # Get player samples (independent for popularity features)
    P_indep = build_player_samples_independent(session, pid_to_col, n_sims=64, seed=42)
    
    # Get actual player ownership
    player_ownership = get_player_ownership(session, pid_to_col)
    
    # Compute features for model (returns features + diagnostics)
    X, feature_diagnostics = compute_lineup_features_for_model(
        lineup_ids=lineup_ids,
        L=L,
        spends=spends,
        cpt_col_by_row=cpt_col_by_row,
        P_indep=P_indep,
        player_ownership=player_ownership,
        total_entries=total_entries,
        entry_fee=entry_fee,
        ownership_gini=ownership_gini,
        verbose=True,  # Enable detailed logging
    )
    
    # Unpack diagnostics
    per_player_mean = feature_diagnostics['per_player_mean']
    mean_proj = feature_diagnostics['mean_proj']
    lineup_total_ownership = feature_diagnostics['lineup_total_ownership']
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    predicted_dupes = dupe_model.predict(X_scaled)
    predicted_dupes = np.clip(predicted_dupes, 0.01, None)
    
    print(f"  Predicted duplicates: min={predicted_dupes.min():.2f}, "
          f"mean={predicted_dupes.mean():.2f}, max={predicted_dupes.max():.2f}")
    
    # Show top predicted lineups BEFORE sampling
    print("\n" + "="*70)
    print("TOP 10 PREDICTED DUPLICATIONS (before sampling)")
    print("="*70)
    top_predicted_idx = np.argsort(predicted_dupes)[-10:][::-1]
    
    feature_names = ['log_entries', 'entry_fee', 'own_gini', 'corr', 'same_team', 
                     'qb_wr', 'rb_dst', 'cpt_own_pct', 'proj_pct', 'total_own_pct',
                     'sal_rem_pct', 'cpt_x_own', 'near_cap', 'chalk']
    
    for rank, idx in enumerate(top_predicted_idx, 1):
        lid = lineup_ids[idx]
        pred = predicted_dupes[idx]
        
        # Get captain info
        cpt_col = cpt_col_by_row[idx]
        if cpt_col >= 0:
            cpt_own_raw = player_ownership[cpt_col]
            cpt_proj_raw = per_player_mean[cpt_col]
            total_own_raw = lineup_total_ownership[idx]
            proj_raw = mean_proj[idx]
            
            # Get percentile features from X
            cpt_own_pct = X[idx, 7]  # cpt_own_pct is 8th feature (index 7)
            proj_pct = X[idx, 8]
            total_own_pct = X[idx, 9]
            
            print(f"#{rank:2d} | LU {lid:5d} | Pred: {pred:6.1f} dupes")
            print(f"     Raw: CPT_own={cpt_own_raw:5.1f}% | CPT_proj={cpt_proj_raw:5.1f} | "
                  f"Total_own={total_own_raw:6.1f} | Proj={proj_raw:5.1f}")
            print(f"     Pct: CPT_own={cpt_own_pct:.3f} | Proj={proj_pct:.3f} | "
                  f"Total_own={total_own_pct:.3f}")
            print(f"     Features: {' | '.join([f'{name}={X[idx, i]:.2f}' for i, name in enumerate(feature_names)])}")
            print()
    print("="*70 + "\n")
    
    # Convert predictions to sampling weights
    weights = predicted_dupes / predicted_dupes.sum()
    
    # Sample field with replacement
    rng = np.random.default_rng(42)
    sampled_indices = rng.choice(n_lineups, size=field_size, replace=True, p=weights)
    
    # Count duplicates
    counts = Counter(sampled_indices)
    counts_by_lineup = {int(lineup_ids[i]): int(count) for i, count in counts.items()}
    
    # Diagnostics
    unique_selected = len(counts)
    actual_dupes = np.array(list(counts.values()))
    
    meta = {
        'sampler': 'showdown_learned',
        'model_path': str(model_dir),
        'total_candidate_lineups': n_lineups,
        'unique_selected': unique_selected,
        'uniqueness_rate': unique_selected / field_size,
        'predicted_dupes': {
            'min': float(predicted_dupes.min()),
            'max': float(predicted_dupes.max()),
            'mean': float(predicted_dupes.mean()),
        },
        'actual_dupes': {
            'min': int(actual_dupes.min()),
            'max': int(actual_dupes.max()),
            'mean': float(actual_dupes.mean()),
            'median': float(np.median(actual_dupes)),
        },
        'contest_params': {
            'total_entries': total_entries,
            'entry_fee': entry_fee,
            'ownership_gini': ownership_gini,
        }
    }
    
    print(f"  ✓ Sampled field: {unique_selected:,} unique lineups "
          f"({meta['uniqueness_rate']:.1%})")
    print(f"  Duplicate range: {actual_dupes.min()}-{actual_dupes.max()} "
          f"(mean={actual_dupes.mean():.1f})")
    
    return counts_by_lineup, meta

# ==============================================================================
# HEURISTIC SAMPLER (SHOWDOWN)
# ==============================================================================

def _percentile_rank(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Calculate percentile rank of each value vs reference array.
    Returns values in [0, 1].
    """
    ref_sorted = np.sort(ref[~np.isnan(ref)])
    if ref_sorted.size <= 1 or np.allclose(ref_sorted[0], ref_sorted[-1]):
        return np.full_like(values, 0.5, dtype=np.float64)
    
    idx = np.searchsorted(ref_sorted, values, side="right")
    return np.clip(idx / ref_sorted.size, 0.0, 1.0).astype(np.float64)


def _normalize_probs(w: np.ndarray, size: int) -> np.ndarray | None:
    """Convert non-negative weights to valid probability vector."""
    p = np.asarray(w, dtype=np.float64).copy()
    p[~np.isfinite(p)] = 0.0
    p = np.clip(p, 0.0, None)
    
    s = p.sum()
    if s <= 0.0:
        return None
    
    p /= s
    
    # Force exact sum=1.0
    residual = 1.0 - p.sum()
    if abs(residual) > 1e-12:
        j = int(np.argmax(p))
        p[j] += residual
        if p[j] < 0.0:
            return None
    
    return p


def sample_showdown_field_heuristic(
    session: Session,
    field_size: int,
    config: dict,
) -> Tuple[Dict[int, int], Dict]:
    """
    Heuristic showdown sampler with captain-aware weighting.
    
    Strategy:
    1. Bin lineups by salary remaining
    2. Within each bin, weight by:
       - Projection strength (ALPHA)
       - Captain popularity (BETA)
       - Template penalty (GAMMA, TAU) - penalizes chalk flex + weak captain
    """
    L, lineup_ids, spends, pid_to_col, cpt_col_by_row, flex_cols_by_row = \
        build_lineup_matrix(session)
    
    n_lineups = L.shape[0]
    remaining = (SALARY_CAP - spends).astype(np.int32)

    # Build salary bins
    bins = list(SALARY_PRIOR)
    bin_masks = [(remaining >= lo) & (remaining < hi) for lo, hi, _ in bins]

    # Get player samples and projections
    P = build_player_samples(session, pid_to_col)
    per_player_mean = np.asarray(P.mean(axis=0), dtype=np.float64)
    mean_proj = np.asarray(L @ per_player_mean, dtype=np.float64)

    # Captain percentiles
    valid_cpt_cols = np.unique(cpt_col_by_row[cpt_col_by_row >= 0])
    cpt_ref = per_player_mean[valid_cpt_cols] if valid_cpt_cols.size else np.array([np.nan])
    
    cpt_means = np.full(n_lineups, np.nan, dtype=np.float64)
    mask_has_cpt = cpt_col_by_row >= 0
    cpt_means[mask_has_cpt] = per_player_mean[cpt_col_by_row[mask_has_cpt]]
    
    cpt_pct = np.zeros(n_lineups, dtype=np.float64)
    if np.isfinite(cpt_ref).any():
        cpt_pct = _percentile_rank(np.nan_to_num(cpt_means, nan=np.nanmin(cpt_ref)), cpt_ref)

    # FLEX chalk score (mean of top-3 FLEX projection percentiles)
    all_flex_cols = np.array(sorted({c for cols in flex_cols_by_row for c in cols}), 
                            dtype=np.int32)
    flex_ref = per_player_mean[all_flex_cols] if all_flex_cols.size else np.array([np.nan])

    player_pct_global = np.zeros_like(per_player_mean, dtype=np.float64)
    if np.isfinite(flex_ref).any():
        player_pct_global = _percentile_rank(per_player_mean, flex_ref)

    flex_topk_mean_pct = np.zeros(n_lineups, dtype=np.float64)
    for i, cols in enumerate(flex_cols_by_row):
        if not cols:
            flex_topk_mean_pct[i] = 0.0
            continue
        vals = player_pct_global[cols]
        if vals.size <= 3:
            flex_topk_mean_pct[i] = float(vals.mean())
        else:
            flex_topk_mean_pct[i] = float(np.sort(vals)[-3:].mean())

    counts = np.zeros(n_lineups, dtype=np.int64)

    # Allocate entries across bins
    weights = np.array([w for _, _, w in bins], dtype=np.float64)
    has_any = np.array([m.any() for m in bin_masks], dtype=bool)
    
    if not has_any.any():
        raise RuntimeError("No lineups in any salary bin. Check SALARY_PRIOR.")
    
    w_avail = weights * has_any.astype(np.float64)
    if w_avail.sum() <= 0:
        w_avail = has_any.astype(np.float64)
    w_avail /= w_avail.sum()

    idx_per_bin = [np.where(m)[0] for m in bin_masks]
    draws_per_bin = np.random.multinomial(int(field_size), w_avail)

    # Sample within each bin
    for b, n_draws in enumerate(draws_per_bin):
        if n_draws <= 0:
            continue
        
        idxs = idx_per_bin[b]
        if idxs.size == 0:
            continue

        # Captain-aware weighting formula
        base = np.maximum(mean_proj[idxs], 0.0)
        cpt_term = np.power(np.clip(cpt_pct[idxs], 0.0, 1.0), BETA)
        
        # Template penalty: chalk FLEX + weak captain = bad
        template_penalty = np.exp(
            -GAMMA * np.maximum(0.0, flex_topk_mean_pct[idxs] - TAU) * (1.0 - cpt_pct[idxs])
        )

        w = np.power(base, ALPHA) * cpt_term * template_penalty

        probs = _normalize_probs(w, idxs.size)
        if probs is None:
            probs = np.full(idxs.size, 1.0 / idxs.size, dtype=np.float64)

        picks = np.random.choice(idxs.size, size=int(n_draws), replace=True, p=probs)
        np.add.at(counts, idxs[picks], 1)

    counts_by_lineup = {int(lineup_ids[i]): int(counts[i]) for i in range(n_lineups)}
    
    meta = {
        "sampler": "showdown_heuristic",
        "params": {"ALPHA": ALPHA, "BETA": BETA, "GAMMA": GAMMA, "TAU": TAU},
        "unique_selected": int((counts > 0).sum()),
        "mean_proj_stats": {"mean": float(mean_proj.mean()), "std": float(mean_proj.std())},
    }
    
    return counts_by_lineup, meta

# ==============================================================================