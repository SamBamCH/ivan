#!/usr/bin/env python3
# scripts/nfl/sample_field.py
"""
NFL field sampler with support for both Showdown and Classic formats.

Showdown: Uses captain-aware sampling with salary bins and template penalties
Classic: Weighted sampling based on projections with optional duplication

Key features:
- Learned duplication model for Showdown (SIMPLIFIED: uses 4 features only)
- Heuristic sampler with diversity optimization
- Captain ownership and template diversity controls

UPDATED: Simplified duplication model now uses only 4 features:
  1. salary_remaining_pct - % of salary cap unused
  2. ownership_sum - Sum of all player ownership percentages
  3. cpt_ownership_pct - Captain's ownership percentage
  4. log_total_entries - Log10 of contest total entries
  
This replaces the old 14-feature model for easier verification and debugging.
Retrain with: python scripts/nfl/train_showdown_model_simplified.py
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
    total_entries: Optional[int] = None,
    entry_fee: Optional[float] = None,
    num_games: Optional[int] = None,
    ownership_gini: Optional[float] = None,
) -> Tuple[Dict[int, int], Dict]:
    """
    Unified field sampler entry point.

    Args:
        session: Database session
        field_size: Number of entries to generate
        config: Configuration dictionary
        mode: "showdown" or "classic"
        contest_id: Optional contest ID for parameters
        use_classic_dupes: Deprecated - now handled automatically by learned sampler
        total_entries: Total entries in the contest (for learned classic sampler)
        entry_fee: Entry fee in dollars (for learned classic sampler)
        num_games: Number of games on slate (for learned classic sampler)
        ownership_gini: Ownership concentration (for learned classic sampler)

    Returns:
        counts_by_lineup: {lineup_id: count}
        meta: Diagnostic information
    """
    mode = (mode or "showdown").lower()

    if mode == "showdown":
        return sample_showdown_field_learned(
            session,
            field_size=field_size,
            config=config,
            contest_id=contest_id,
            model_dir=config.get('model_dir', 'data/nfl/models'),
        )

    return sample_classic_field_learned(
        session,
        field_size=field_size,
        config=config,
        contest_id=contest_id,
        total_entries=total_entries,
        entry_fee=entry_fee,
        num_games=num_games,
        ownership_gini=ownership_gini,
    )

# CONSTANTS
# ==============================================================================

SALARY_CAP = 50000
CPT_SLOT = 511
FLEX_SLOT = 512


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_showdown_models(model_dir: str | Path = "data/nfl/models"):
    """Load trained showdown duplication models from disk.
    
    First tries to load two-stage models (new), falls back to single-stage (legacy).
    
    Returns:
        For two-stage:
            {
                'stage1_model': MultiOutputRegressor,
                'stage1_scaler': StandardScaler,
                'stage1_feature_cols': list,
                'stage2_model': GradientBoostingRegressor,
                'stage2_scaler': StandardScaler,
                'stage2_feature_cols': list,
                'mode': 'two_stage',
            }
        For legacy:
            {
                'dupe_model': GradientBoostingRegressor,
                'scaler': StandardScaler,
                'feature_cols': list,
                'mode': 'legacy',
            }
    """
    model_dir = Path(model_dir)
    
    # Try two-stage models first
    # try:
    with open(model_dir / 'showdown_stage1_model.pkl', 'rb') as f:
        stage1_model = pickle.load(f)
    with open(model_dir / 'showdown_stage1_scaler.pkl', 'rb') as f:
        stage1_scaler = pickle.load(f)
    with open(model_dir / 'showdown_stage1_feature_cols.pkl', 'rb') as f:
        stage1_feature_cols = pickle.load(f)
    
    with open(model_dir / 'showdown_stage2_model.pkl', 'rb') as f:
        stage2_model = pickle.load(f)
    with open(model_dir / 'showdown_stage2_scaler.pkl', 'rb') as f:
        stage2_scaler = pickle.load(f)
    with open(model_dir / 'showdown_stage2_feature_cols.pkl', 'rb') as f:
        stage2_feature_cols = pickle.load(f)
    
    print("✓ Loaded TWO-STAGE hierarchical models")
    
    return {
        'stage1_model': stage1_model,
        'stage1_scaler': stage1_scaler,
        'stage1_feature_cols': stage1_feature_cols,
        'stage2_model': stage2_model,
        'stage2_scaler': stage2_scaler,
        'stage2_feature_cols': stage2_feature_cols,
        'mode': 'two_stage',
    }
    
    # except FileNotFoundError:
    #     # Fall back to legacy single-stage model
    #     try:
    #         print("⚠️  Two-stage models not found, trying legacy single-stage model...")
    #         with open(model_dir / 'showdown_duplication_model.pkl', 'rb') as f:
    #             dupe_model = pickle.load(f)
            
    #         with open(model_dir / 'showdown_scaler.pkl', 'rb') as f:
    #             scaler = pickle.load(f)
            
    #         with open(model_dir / 'showdown_feature_cols.pkl', 'rb') as f:
    #             feature_cols = pickle.load(f)
            
    #         print("✓ Loaded legacy single-stage model")
            
    #         return {
    #             'dupe_model': dupe_model,
    #             'scaler': scaler,
    #             'feature_cols': feature_cols,
    #             'mode': 'legacy',
    #         }
    #     except FileNotFoundError as e:
    #         raise FileNotFoundError(
    #             f"Showdown models not found in {model_dir}. "
    #             f"Run 'python scripts/nfl/train_showdown_model.py' first."
    #         ) from e

# ==============================================================================
# TWO-STAGE PREDICTION FUNCTION
# ==============================================================================

def predict_lineups_with_twostage(
    session: Session,
    L,
    lineup_ids: np.ndarray,
    spends: np.ndarray,
    cpt_col_by_row: np.ndarray,
    player_ownership: np.ndarray,
    total_entries: int,
    models: dict,
    contest_features: dict,
) -> np.ndarray:
    """
    UNIFIED TWO-STAGE PREDICTION
    
    Stage 1: Contest Distribution Model
      Input: Contest parameters (field_size, ownership_gini, projection_std, etc.)
      Output: Quantile distribution [Q25, Q50, Q75, Q90, Q95, Q99]
      
    Stage 2: Lineup Positioning Model  
      Input: Lineup features (cpt_ownership_pct, ownership_sum, salary_remaining_pct)
      Output: Duplication scores → percentile ranks relative to candidate pool
      
    Combined: Interpolate percentile ranks into Stage 1's quantiles to get final predictions
    """
    from scipy.special import expit
    
    n_lineups = L.shape[0]
    
    print(f"\n{'='*70}")
    print("UNIFIED TWO-STAGE PREDICTION")
    print(f"{'='*70}")
    print(f"Contest size: {total_entries:,} entries")
    print(f"Candidate lineups: {n_lineups:,}")
    
    # ========================================================================
    # STAGE 1: Contest Distribution Model
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 1: Contest Distribution Quantiles")
    print("="*70)
    
    # Build contest feature vector
    stage1_features = np.array([[
        contest_features['total_entries'],
        contest_features['field_density'],
        contest_features['ownership_gini'],
        contest_features['projection_std'],
    ]])
    
    # Scale and predict quantiles
    stage1_scaler = models['stage1_scaler']
    stage1_model = models['stage1_model']
    
    stage1_features_scaled = stage1_scaler.transform(stage1_features)
    quantile_predictions = stage1_model.predict(stage1_features_scaled)  # shape [1, 6]
    
    # Extract quantiles [Q25, Q50, Q75, Q90, Q95, Q99]
    quantile_levels = np.array([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    quantile_values = quantile_predictions[0]  # shape [6]
    
    # Clip to realistic range
    quantile_values = np.clip(quantile_values, 1.0, total_entries * 0.01)
    
    print(f"Predicted duplication distribution:")
    for q_level, q_val in zip(quantile_levels, quantile_values):
        print(f"  Q{q_level*100:.0f}: {q_val:.2f} duplicates")
    
    # ========================================================================
    # STAGE 2: Lineup Positioning Model
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 2: Lineup Duplication Scores")
    print("="*70)
    
    # Compute 3 lineup features
    remaining = (SALARY_CAP - spends).astype(np.int32)
    salary_remaining_pct = remaining / SALARY_CAP
    
    ownership_sum = np.asarray(L @ player_ownership, dtype=np.float64)
    
    cpt_ownership = np.zeros(n_lineups, dtype=np.float64)
    mask = cpt_col_by_row >= 0
    if mask.any():
        cpt_ownership[mask] = player_ownership[cpt_col_by_row[mask]]
    
    # Build feature matrix (3 features in correct order)
    X_lineup = np.column_stack([
        cpt_ownership,
        ownership_sum,
        salary_remaining_pct,
    ])
    
    # Scale and predict duplication scores (in logit space)
    stage2_scaler = models['stage2_scaler']
    stage2_model = models['stage2_model']
    
    X_lineup_scaled = stage2_scaler.transform(X_lineup)
    dupe_scores_logit = stage2_model.predict(X_lineup_scaled)
    dupe_scores = expit(dupe_scores_logit)  # Convert from logit to [0, 1]
    
    print(f"Raw duplication scores (probability of duplication):")
    print(f"  Min:    {dupe_scores.min():.3f}")
    print(f"  Mean:   {dupe_scores.mean():.3f}")
    print(f"  Median: {np.median(dupe_scores):.3f}")
    print(f"  Max:    {dupe_scores.max():.3f}")
    
    # ========================================================================
    # COMBINE: Map scores to percentiles, then to quantiles
    # ========================================================================
    print("\n" + "="*70)
    print("COMBINATION: Scores → Percentiles → Duplicates")
    print("="*70)
    
    # Compute percentile rank of each lineup relative to the candidate pool
    # (Higher score = higher percentile = more duplication)
    lineup_percentiles = np.argsort(np.argsort(dupe_scores)) / (n_lineups - 1) if n_lineups > 1 else np.ones(n_lineups) * 0.5
    lineup_percentiles = np.clip(lineup_percentiles, 0.0, 1.0)
    
    print(f"Lineup percentile distribution (relative to candidate pool):")
    print(f"  Min:    {lineup_percentiles.min():.3f}")
    print(f"  Mean:   {lineup_percentiles.mean():.3f}")
    print(f"  Median: {np.median(lineup_percentiles):.3f}")
    print(f"  Max:    {lineup_percentiles.max():.3f}")
    
    # Interpolate percentiles into Stage 1's quantile distribution
    # This gives us the predicted duplicate count for each lineup
    predicted_dupes = np.interp(
        lineup_percentiles,
        quantile_levels,
        quantile_values,
    )
    
    print(f"\nFinal duplicate predictions:")
    print(f"  Min:    {predicted_dupes.min():.2f}")
    print(f"  Mean:   {predicted_dupes.mean():.2f}")
    print(f"  Median: {np.median(predicted_dupes):.2f}")
    print(f"  Max:    {predicted_dupes.max():.2f}")
    print(f"  P90:    {np.percentile(predicted_dupes, 90):.2f}")
    print(f"  P95:    {np.percentile(predicted_dupes, 95):.2f}")
    print(f"  P99:    {np.percentile(predicted_dupes, 99):.2f}")
    
    print(f"{'='*70}\n")
    
    return predicted_dupes

def compute_contest_features_from_lineups(
    L: sp.csr_matrix,
    player_ownership: np.ndarray,
    total_entries: int,
    n_lineups: int,
) -> dict:
    """
    Compute contest-level features from candidate lineup pool.
    
    Returns:
        dict with keys: total_entries, field_density, ownership_gini, projection_std
    """
    # Field density: % of lineups that are unique
    field_density = min(n_lineups / total_entries, 1.0)
    
    # Ownership concentration (Gini coefficient)
    lineup_ownership = np.asarray(L @ player_ownership, dtype=np.float64)
    ownership_gini = calculate_gini(lineup_ownership) if len(lineup_ownership) > 1 else 0.5
    
    # Projection std (estimate from ownership distribution)
    # This is a proxy: more concentrated ownership → lower projection variance
    # We use ownership std as a stand-in
    projection_std = float(np.std(lineup_ownership)) if len(lineup_ownership) > 1 else 5.0
    projection_std = np.clip(projection_std, 0.1, 50.0)
    
    return {
        'total_entries': total_entries,
        'field_density': field_density,
        'ownership_gini': ownership_gini,
        'projection_std': projection_std,
    }


def calculate_gini(values: np.ndarray) -> float:
    """Calculate Gini coefficient for concentration."""
    if len(values) < 2:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))

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

def get_player_ownership_heuristic(
    session: Session,
    pid_to_col: Dict[int, int],
    alpha: float = 3.0,  # Changed: Higher default for better concentration
    salary_weight: float = 0.2,  # Changed: Reduced salary impact
    captain_concentration: float = 4.0,  # NEW: Separate concentration for captains
) -> np.ndarray:
    """
    FIXED: Calculate realistic heuristic player ownership with proper captain concentration.

    Key improvements:
    - Uses exponential concentration instead of buggy power formula
    - Captain ownership properly concentrated (top 2-3 get 40-60% combined)
    - Accounts for 1.5x captain scoring multiplier
    - Separate concentration parameters for captain vs flex

    Args:
        session: Database session
        pid_to_col: Player ID to column index mapping
        alpha: Base concentration for flex (higher = more chalk)
        salary_weight: How much salary matters (0-1)
        captain_concentration: Extra concentration for captains (higher = steeper)

    Returns:
        Array [n_players] with realistic ownership percentages (0-100)
    """
    from ivan.sports.nfl.models import NFLPlayer
    
    player_data = session.query(
        NFLPlayer.id,
        NFLPlayer.fpts_mean,
        NFLPlayer.salary,
        NFLPlayer.roster_slot_id,
    ).filter(NFLPlayer.id.in_(pid_to_col.keys())).all()

    # Build arrays
    n_players = len(pid_to_col)
    projections = np.zeros(n_players, dtype=np.float64)
    salaries = np.zeros(n_players, dtype=np.float64)
    is_cpt = np.zeros(n_players, dtype=bool)

    for pid, fpts_mean, salary, roster_slot_id in player_data:
        col = pid_to_col[pid]
        projections[col] = float(fpts_mean or 0.0)
        salaries[col] = float(salary or 0.0)
        is_cpt[col] = (roster_slot_id == CPT_SLOT)

    ownership = np.zeros(n_players, dtype=np.float64)

    # =========================================================================
    # CAPTAIN OWNERSHIP - FIXED with exponential concentration
    # =========================================================================
    cpt_mask = is_cpt
    if cpt_mask.any():
        cpt_projs = projections[cpt_mask]
        cpt_sals = salaries[cpt_mask]

        # Account for 1.5x captain scoring multiplier
        cpt_projs_adjusted = cpt_projs * 1.5
        
        # Captains: projection matters much more than salary
        captain_salary_weight = salary_weight * 0.5
        
        # Calculate value scores (not percentiles - actual values for better differentiation)
        if cpt_projs_adjusted.max() > 0:
            proj_score = cpt_projs_adjusted / cpt_projs_adjusted.max()
        else:
            proj_score = np.ones_like(cpt_projs_adjusted)
        
        # Inverse salary score (lower salary = better value)
        if cpt_sals.max() > cpt_sals.min():
            sal_score = 1 - (cpt_sals - cpt_sals.min()) / (cpt_sals.max() - cpt_sals.min())
        else:
            sal_score = np.ones_like(cpt_sals)
        
        # Combined value score
        value_scores = (
            proj_score * (1 - captain_salary_weight) + 
            sal_score * captain_salary_weight
        )
        
        # FIXED: Apply exponential transformation for proper concentration
        ownership_scores = np.exp(captain_concentration * value_scores)
        
        # Add small minimum ownership
        ownership_scores = ownership_scores + 0.01
        
        # Normalize to 100% total
        cpt_ownership = 100.0 * ownership_scores / ownership_scores.sum()
        ownership[cpt_mask] = cpt_ownership
        
        # Log top captains for verification
        if len(cpt_ownership) >= 3:
            top_3_idx = np.argsort(cpt_ownership)[-3:]
            top_3_own = cpt_ownership[top_3_idx]
            print(f"  Top 3 captain ownership: {top_3_own[-1]:.1f}%, {top_3_own[-2]:.1f}%, {top_3_own[-3]:.1f}%")

    # =========================================================================
    # FLEX OWNERSHIP - More distributed than captain
    # =========================================================================
    flex_mask = ~is_cpt
    if flex_mask.any():
        flex_projs = projections[flex_mask]
        flex_sals = salaries[flex_mask]

        # Flex: more balanced projection/salary weighting
        if flex_projs.max() > 0:
            proj_score = flex_projs / flex_projs.max()
        else:
            proj_score = np.ones_like(flex_projs)
        
        if flex_sals.max() > flex_sals.min():
            sal_score = 1 - (flex_sals - flex_sals.min()) / (flex_sals.max() - flex_sals.min())
        else:
            sal_score = np.ones_like(flex_sals)
        
        value_scores = (
            proj_score * (1 - salary_weight) + 
            sal_score * salary_weight
        )
        
        # FIXED: Use base alpha for flex (less concentration than captain)
        ownership_scores = np.exp(alpha * value_scores)
        ownership_scores = ownership_scores + 0.1
        
        # Normalize to 500% total (5 flex per lineup)
        flex_ownership = 500.0 * ownership_scores / ownership_scores.sum()
        ownership[flex_mask] = flex_ownership

    return ownership


def get_player_ownership(session: Session, pid_to_col: Dict[int, int]) -> np.ndarray:
    """
    Extract actual player ownership from NFLPlayer table.

    Returns array [n_players] with ownership percentages (0-100).
    Missing ownership defaults to 0.

    NOTE: This should only be used for actual DK ownership data, NOT for model inputs
    (to avoid circular dependency with ownership_proj).
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
    player_ownership: np.ndarray,  # [n_players] actual ownership
    total_entries: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute simplified 4-feature matrix for trained duplication model.
    
    Features (must match training order):
        1. salary_remaining_pct - % of salary cap unused
        2. ownership_sum - Sum of all player ownership percentages
        3. cpt_ownership_pct - Captain's ownership percentage
        4. log_total_entries - Log10 of contest total entries
    
    Returns: [n_lineups x 4] array ready for model prediction
    """
    n_lineups = L.shape[0]
    
    if verbose:
        print("\n" + "="*70)
        print("SIMPLIFIED FEATURE COMPUTATION (4 features)")
        print("="*70)
        print(f"Number of lineups: {n_lineups:,}")
        print(f"Number of players: {L.shape[1]:,}")
        print(f"Contest size: {total_entries:,} entries")
    
    # ====================================================================
    # Feature 1: Salary remaining (% of cap unused)
    # ====================================================================
    remaining = (SALARY_CAP - spends).astype(np.int32)
    salary_remaining_pct = remaining / SALARY_CAP
    
    if verbose:
        print(f"\n1. SALARY REMAINING:")
        print(f"   Min: ${remaining.min()} ({salary_remaining_pct.min():.3%})")
        print(f"   Max: ${remaining.max()} ({salary_remaining_pct.max():.3%})")
        print(f"   Mean: ${remaining.mean():.0f} ({salary_remaining_pct.mean():.3%})")
    
    # ====================================================================
    # Feature 2: Total lineup ownership (sum of all players)
    # ====================================================================
    ownership_sum = np.asarray(L @ player_ownership, dtype=np.float64)
    
    if verbose:
        print(f"\n2. TOTAL LINEUP OWNERSHIP:")
        print(f"   Min: {ownership_sum.min():.2f}%")
        print(f"   Max: {ownership_sum.max():.2f}%")
        print(f"   Mean: {ownership_sum.mean():.2f}%")
        print(f"   Zero-ownership lineups: {(ownership_sum == 0).sum()} / {n_lineups}")
    
    # ====================================================================
    # Feature 3: Captain ownership
    # ====================================================================
    cpt_ownership = np.zeros(n_lineups, dtype=np.float64)
    mask = cpt_col_by_row >= 0
    if mask.any():
        cpt_ownership[mask] = player_ownership[cpt_col_by_row[mask]]
    
    if verbose:
        print(f"\n3. CAPTAIN OWNERSHIP:")
        print(f"   Valid captains: {mask.sum()} / {n_lineups}")
        if mask.any():
            print(f"   Min: {cpt_ownership[mask].min():.2f}%")
            print(f"   Max: {cpt_ownership[mask].max():.2f}%")
            print(f"   Mean: {cpt_ownership[mask].mean():.2f}%")
            print(f"   Zero-ownership captains: {(cpt_ownership[mask] == 0).sum()}")
    
    # ====================================================================
    # Feature 4: Contest size (log scale)
    # ====================================================================
    log_total_entries = np.log10(max(1, total_entries))
    log_total_entries_array = np.full(n_lineups, log_total_entries, dtype=np.float64)
    
    if verbose:
        print(f"\n4. CONTEST SIZE:")
        print(f"   Total entries: {total_entries:,}")
        print(f"   Log10(entries): {log_total_entries:.3f}")
    
    # ====================================================================
    # Build feature matrix (ORDER MATTERS - must match training!)
    # ====================================================================
    feature_array = np.column_stack([
        salary_remaining_pct,
        ownership_sum,
        cpt_ownership,
        log_total_entries_array,
    ])
    
    if verbose:
        print(f"\n{'='*70}")
        print("FEATURE MATRIX SUMMARY")
        print(f"{'='*70}")
        print(f"Shape: {feature_array.shape}")
        
        # Show sample features for first 3 lineups
        if n_lineups >= 3:
            print(f"\nSample features (first 3 lineups):")
            print(f"{'Feature':<25} {'Lineup 1':>12} {'Lineup 2':>12} {'Lineup 3':>12}")
            print("-" * 70)
            feature_names = ['salary_remaining_pct', 'ownership_sum', 
                           'cpt_ownership_pct', 'log_total_entries']
            for i, name in enumerate(feature_names):
                vals = [f"{feature_array[j, i]:12.4f}" for j in range(3)]
                print(f"{name:<25} {' '.join(vals)}")
        
        print(f"{'='*70}\n")
    
    return feature_array

# ==============================================================================
# DIVERSITY SAMPLING WITH EUCLIDEAN DISTANCES
# ==============================================================================

def compute_lineup_diversity_matrix(
    session: Session,
    L: sp.csr_matrix,
    lineup_ids: np.ndarray,
    pid_to_col: dict,
    n_sims: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute pairwise Euclidean distances between lineups based on simulated fpts traces.

    Uses shallow correlated samples to build fpts traces quickly, then calculates
    Euclidean distances in the fpts space (similar to choose.py method).

    Args:
        session: Database session
        L: Lineup matrix [n_lineups x n_players]
        lineup_ids: Array of lineup IDs
        pid_to_col: Dict mapping player ID to column index in L
        n_sims: Number of correlated simulations (64-128 recommended for speed)
        seed: Random seed for reproducibility

    Returns:
        dist_mat: Pairwise distance matrix [n_lineups x n_lineups]
        meta: Diagnostic info about distances
    """
    from scipy.spatial.distance import pdist, squareform

    n_lineups = L.shape[0]

    if n_lineups == 0:
        return np.array([]), {'error': 'no_lineups', 'n_lineups': 0}

    if n_lineups == 1:
        return np.array([[0.0]]), {'n_lineups': 1, 'n_sims': n_sims}

    try:
        # Generate shallow correlated samples
        print(f"  Generating {n_sims} correlated samples for diversity calculation...")
        player_samples = generate_correlated_samples_gaussian(
            session,
            n_samples=n_sims,
            seed=seed,
            use_hgv20=True,
            randomness=1.0,
        )

        if not player_samples:
            print("  WARNING: Failed to generate correlated samples, using uniform diversity")
            return np.ones((n_lineups, n_lineups)), {'error': 'no_samples'}

        # Build fpts matrix: each row is a lineup's simulated fpts across n_sims scenarios
        # Shape: [n_lineups x n_sims]
        print(f"  Building fpts trace matrix ({n_lineups} lineups x {n_sims} sims)...")

        # P should have shape [n_players x n_sims], indexed by column numbers in L
        n_players = L.shape[1]
        P = np.zeros((n_players, n_sims), dtype=np.float64)

        for pid, fpts_array in player_samples.items():
            if pid in pid_to_col and isinstance(fpts_array, np.ndarray) and fpts_array.size >= n_sims:
                col_idx = pid_to_col[pid]
                P[col_idx, :] = fpts_array[:n_sims]

        # Convert sparse L to CSR if needed for efficient multiplication
        if not isinstance(L, sp.csr_matrix):
            L = sp.csr_matrix(L)

        fpts_matrix = np.asarray(L @ P, dtype=np.float64)  # [n_lineups x n_sims]

        # Compute pairwise Euclidean distances
        print(f"  Computing pairwise Euclidean distances ({n_lineups} lineups)...")
        dist_vec = pdist(fpts_matrix, metric="euclidean")
        dist_mat = squareform(dist_vec)

        meta = {
            'n_lineups': n_lineups,
            'n_sims': n_sims,
            'fpts_shape': fpts_matrix.shape,
            'distance_stats': {
                'min': float(dist_mat[dist_mat > 0].min()) if (dist_mat > 0).any() else 0.0,
                'max': float(dist_mat.max()),
                'mean': float(dist_mat[np.triu_indices_from(dist_mat, k=1)].mean()),
                'median': float(np.median(dist_mat[np.triu_indices_from(dist_mat, k=1)])),
            }
        }

        print(f"  ✓ Distance matrix computed: min={meta['distance_stats']['min']:.2f}, "
              f"mean={meta['distance_stats']['mean']:.2f}, max={meta['distance_stats']['max']:.2f}")

        return dist_mat, meta

    except Exception as e:
        print(f"  WARNING: Diversity computation failed ({e}), using uniform diversity")
        return np.ones((n_lineups, n_lineups)), {'error': str(e)}


def apply_diversity_penalty_to_weights(
    predicted_dupes: np.ndarray,
    dist_mat: np.ndarray,
    diversity_lambda: float = 0.1,
) -> np.ndarray:
    """
    Adjust predicted duplicate weights using diversity penalty based on Euclidean distances.

    Lineups that are far apart in fpts space get higher relative weight.
    Lineups that are close together get lower relative weight (penalized).

    Formula:
        adjusted_weight[i] = predicted_dupes[i] * (1 + lambda * mean_distance_to_all[i])

    Args:
        predicted_dupes: Original predicted duplicate counts [n_lineups]
        dist_mat: Pairwise distance matrix [n_lineups x n_lineups]
        diversity_lambda: Strength of diversity penalty (0.0 = no penalty, higher = stronger)

    Returns:
        adjusted_weights: Normalized weights [n_lineups] that sum to 1
    """
    n_lineups = len(predicted_dupes)

    if n_lineups <= 1 or dist_mat.size == 0:
        return predicted_dupes / (predicted_dupes.sum() + 1e-9)

    # Compute mean distance from each lineup to all others
    mean_distances = np.zeros(n_lineups, dtype=np.float64)
    for i in range(n_lineups):
        # Average distance to all other lineups (excluding self, which is 0)
        other_dists = np.delete(dist_mat[i], i)
        if other_dists.size > 0:
            mean_distances[i] = float(np.mean(other_dists))

    # Normalize distances to [0, 1]
    if mean_distances.max() > 0:
        norm_distances = mean_distances / (mean_distances.max() + 1e-9)
    else:
        norm_distances = np.zeros_like(mean_distances)

    # Apply diversity penalty: increase weight for diverse lineups
    diversity_multiplier = 1.0 + diversity_lambda * norm_distances

    adjusted_weights = predicted_dupes * diversity_multiplier
    adjusted_weights = adjusted_weights / (adjusted_weights.sum() + 1e-9)

    return adjusted_weights

# ==============================================================================
# UNIFIED DIVERSITY-GATED DUPLICATION SAMPLER
# ==============================================================================

def sample_showdown_field_unified(
    session: Session,
    field_size: int,
    L,  # sparse lineup matrix
    lineup_ids: np.ndarray,
    spends: np.ndarray,
    pid_to_col: Dict[int, int],
    cpt_col_by_row: np.ndarray,
    predicted_dupes: np.ndarray,  # NEW: pass predicted dupes directly
    player_ownership: np.ndarray,
    total_entries: int,
) -> Tuple[Dict[int, int], Dict]:
    """
    Unified sampler combining diversity and duplication in one gate.

    Strategy:
    1. Compute diversity matrix (Euclidean distances in fpts space)
    2. Use provided predicted duplication counts
    3. Combine: weight = (1 + diversity_bonus) * predicted_dupes
    4. Sample with combined weights

    This ensures:
    - Diverse lineups are prioritized relative to similar ones
    - Historical duplication patterns are preserved
    - Both work together (no post-hoc modification)

    Args:
        predicted_dupes: [n_lineups] array of predicted duplicate counts

    Returns:
        counts_by_lineup: {lineup_id: count}
        meta: Diagnostic info
    """
    n_lineups = L.shape[0]

    print(f"\n" + "="*70)
    print("UNIFIED DIVERSITY-GATED DUPLICATION SAMPLER")
    print("="*70)

    # ========================================================================
    # Step 1: Compute diversity matrix (Euclidean distances)
    # ========================================================================
    print("\nStep 1: Computing diversity matrix...")
    dist_mat, diversity_meta = compute_lineup_diversity_matrix(
        session=session,
        L=L,
        lineup_ids=lineup_ids,
        pid_to_col=pid_to_col,
        n_sims=32,  # Fast: only 32 sims for diversity
        seed=42,
    )

    if dist_mat.size == 0 or 'error' in diversity_meta:
        print("  WARNING: Diversity computation failed, using uniform diversity")
        dist_mat = np.ones((n_lineups, n_lineups))
        diversity_meta = {'error': 'computation_failed'}

    # ========================================================================
    # Step 2: Use predicted duplication from caller
    # ========================================================================
    print("\nStep 2: Using predicted duplication...")
    
    predicted_dupes_clipped = np.clip(predicted_dupes, 1.0, None)
    
    print(f"  Predicted dupes: min={predicted_dupes_clipped.min():.2f}, "
          f"mean={predicted_dupes_clipped.mean():.2f}, max={predicted_dupes_clipped.max():.2f}")

    # ========================================================================
    # Step 3: Compute diversity bonuses
    # ========================================================================
    print("\nStep 3: Computing diversity bonuses...")

    # For each lineup, compute average distance to others
    diversity_scores = np.zeros(n_lineups, dtype=np.float64)
    for i in range(n_lineups):
        other_dists = np.delete(dist_mat[i], i)
        if other_dists.size > 0:
            diversity_scores[i] = float(np.mean(other_dists))

    # Normalize to [0, 1] for interpretability
    if diversity_scores.max() > diversity_scores.min():
        diversity_norm = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min())
    else:
        diversity_norm = np.ones_like(diversity_scores) * 0.5

    # Diversity bonus: lineups far from others get higher bonus
    diversity_bonus = 1.0 + 0.5 * (diversity_norm - 0.5)

    print(f"  Diversity scores: min={diversity_scores.min():.2f}, "
          f"mean={diversity_scores.mean():.2f}, max={diversity_scores.max():.2f}")
    print(f"  Diversity bonus range: [{diversity_bonus.min():.3f}, {diversity_bonus.max():.3f}]")

    # ========================================================================
    # Step 4: Combine diversity and duplication
    # ========================================================================
    print("\nStep 4: Combining diversity and duplication weights...")

    # Combined weight: higher dupes + higher diversity = higher sampling prob
    combined_weights = predicted_dupes_clipped * diversity_bonus
    combined_weights = np.clip(combined_weights, 1e-12, None)
    combined_weights = combined_weights / combined_weights.sum()

    print(f"  Combined weights: min={combined_weights.min():.2e}, "
          f"mean={combined_weights.mean():.2e}, max={combined_weights.max():.2e}")

    # ========================================================================
    # Step 5: Sample field
    # ========================================================================
    print(f"\nStep 5: Sampling {field_size} lineups...")
    rng = np.random.default_rng(42)
    sampled_indices = rng.choice(n_lineups, size=field_size, replace=True, p=combined_weights)

    # Count duplicates
    counts = Counter(sampled_indices)
    counts_by_lineup = {int(lineup_ids[i]): int(count) for i, count in counts.items()}
    unique_selected = len(counts)
    actual_dupes = np.array(list(counts.values()))

    print(f"  ✓ Selected {unique_selected:,} unique lineups ({100*unique_selected/field_size:.1f}%)")
    print(f"  Duplicate range: {actual_dupes.min()}-{actual_dupes.max()} (mean={actual_dupes.mean():.1f})")

    meta = {
        'sampler': 'showdown_unified_diversity_gated',
        'total_candidate_lineups': n_lineups,
        'unique_selected': unique_selected,
        'uniqueness_rate': unique_selected / field_size,
        'predicted_dupes': {
            'min': float(predicted_dupes_clipped.min()),
            'max': float(predicted_dupes_clipped.max()),
            'mean': float(predicted_dupes_clipped.mean()),
        },
        'diversity_scores': {
            'min': float(diversity_scores.min()),
            'max': float(diversity_scores.max()),
            'mean': float(diversity_scores.mean()),
        },
        'diversity_bonus': {
            'min': float(diversity_bonus.min()),
            'max': float(diversity_bonus.max()),
            'mean': float(diversity_bonus.mean()),
        },
        'actual_dupes': {
            'min': int(actual_dupes.min()),
            'max': int(actual_dupes.max()),
            'mean': float(actual_dupes.mean()),
            'median': float(np.median(actual_dupes)),
        },
        'diversity_meta': diversity_meta,
    }

    print("="*70 + "\n")

    return counts_by_lineup, meta


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
    Sample showdown field using trained duplication models with diversity control.

    Supports both:
    - NEW: Two-stage hierarchical model (preferred)
    - LEGACY: Single-stage model (fallback)

    Uses historical patterns to predict lineup duplication based on:
    - Contest size and entry fee
    - Captain and player ownership
    - Lineup projections and salary usage

    Returns:
        counts_by_lineup: {lineup_id: duplicate_count}
        meta: Diagnostic info
    """
    # Load trained models
    models = load_showdown_models(model_dir)
    model_mode = models.get('mode', 'unknown')
    
    # Get contest parameters
    if contest_id:
        from ivan.sports.nfl.models import HistoricalContest
        contest = session.query(HistoricalContest).get(contest_id)
        
        if contest:
            total_entries = contest.total_entries
        else:
            total_entries = field_size
    else:
        total_entries = field_size
    
    # Build lineup matrix
    L, lineup_ids, spends, pid_to_col, cpt_col_by_row, _ = build_lineup_matrix(session)
    n_lineups = L.shape[0]

    if n_lineups == 0:
        print("  WARNING: No lineups found in database")
        return {}, {'error': 'no_lineups'}

    print(f"  Candidate lineups: {n_lineups:,}")

    # Get heuristic player ownership with FIXED concentration
    alpha = 3.0  # Changed from 2.0 - better flex concentration
    salary_weight = 0.2  # Changed from 0.3 - less salary emphasis
    captain_concentration = 4.0  # NEW - proper captain concentration

    player_ownership = get_player_ownership_heuristic(
        session,
        pid_to_col,
        alpha=alpha,
        salary_weight=salary_weight,
        captain_concentration=captain_concentration,
    )

    print(f"  Ownership range: [{player_ownership.min():.2f}%, {player_ownership.max():.2f}%], "
          f"mean={player_ownership.mean():.2f}%")
    
    # ================================================================
    # PREDICT DUPLICATION BASED ON MODEL MODE
    # ================================================================
    
    if model_mode == 'two_stage':
        print(f"\n↓ Using TWO-STAGE hierarchical model...")
        
        # Compute contest features from the candidate lineup pool
        contest_features = compute_contest_features_from_lineups(
            L=L,
            player_ownership=player_ownership,
            total_entries=total_entries,
            n_lineups=n_lineups,
        )
        
        print(f"  Contest features:")
        print(f"    Field density: {contest_features['field_density']:.3f}")
        print(f"    Ownership Gini: {contest_features['ownership_gini']:.3f}")
        print(f"    Projection std: {contest_features['projection_std']:.3f}")
        
        predicted_dupes = predict_lineups_with_twostage(
            session=session,
            L=L,
            lineup_ids=lineup_ids,
            spends=spends,
            cpt_col_by_row=cpt_col_by_row,
            player_ownership=player_ownership,
            total_entries=total_entries,
            models=models,
            contest_features=contest_features,
        )
    else:
        print(f"\n↓ Using LEGACY single-stage model...")
        X = compute_lineup_features_for_model(
            lineup_ids=lineup_ids,
            L=L,
            spends=spends,
            cpt_col_by_row=cpt_col_by_row,
            player_ownership=player_ownership,
            total_entries=total_entries,
            verbose=False,
        )
        X_scaled = models['scaler'].transform(X)
        predicted_dupes = models['dupe_model'].predict(X_scaled)
        predicted_dupes = np.expm1(predicted_dupes)  # Convert from log space
        predicted_dupes = np.clip(predicted_dupes, 1.0, None)  # Minimum 1 dupe
    
    # Use unified diversity-gated duplication sampler
    counts_by_lineup, meta = sample_showdown_field_unified(
        session=session,
        field_size=field_size,
        L=L,
        lineup_ids=lineup_ids,
        spends=spends,
        pid_to_col=pid_to_col,
        cpt_col_by_row=cpt_col_by_row,
        predicted_dupes=predicted_dupes,  # NEW: pass predicted dupes directly
        player_ownership=player_ownership,
        total_entries=total_entries,
    )
    
    # Add metadata
    meta['model_path'] = str(model_dir)
    meta['model_mode'] = model_mode
    meta['contest_params'] = {
        'total_entries': total_entries,
    }

    return counts_by_lineup, meta

# ==============================================================================
# LEARNED CLASSIC SAMPLER (MODEL-BASED DISTRIBUTION PREDICTION)
# ==============================================================================

def sample_classic_field_learned(
    session: Session,
    field_size: int,
    config: dict,
    contest_id: Optional[int] = None,
    total_entries: Optional[int] = None,
    entry_fee: Optional[float] = None,
    num_games: Optional[int] = None,
    ownership_gini: Optional[float] = None,
    model_dir: str | Path = "data/nfl/models",
) -> Tuple[Dict[int, int], Dict]:
    """
    Model-based classic field sampler.

    Strategy:
    1. Load trained Random Forest regressors (projection & ownership distributions)
    2. Predict distribution shapes based on contest parameters
    3. Sample lineups with replacement matching predicted distributions
    4. Natural duplication occurs when distributions are concentrated

    Reuses the proven MLB pattern for NFL Classic contests.
    """
    import pickle
    from pathlib import Path

    def _get(d: dict, path: List[str], default=None):
        """Helper to safely navigate nested config dicts."""
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _calculate_gini(values: List[float]) -> float:
        """Calculate Gini coefficient for ownership concentration."""
        if not values or len(values) < 2:
            return 0.0
        sorted_values = np.array(sorted(values))
        n = len(sorted_values)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n)

    # Load trained models
    model_dir = Path(model_dir)
    try:
        with open(model_dir / "classic_proj_model.pkl", "rb") as f:
            proj_model = pickle.load(f)
        with open(model_dir / "classic_own_model.pkl", "rb") as f:
            own_model = pickle.load(f)
        with open(model_dir / "classic_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(model_dir / "classic_bin_edges.pkl", "rb") as f:
            bin_edges = pickle.load(f)
    except FileNotFoundError:
        print(f"  ⚠️  Models not found in {model_dir}")
        print("  Run: python -m ivan.sports.nfl.build_models")
        return {}, {'error': 'models_not_found'}

    # Build lineup matrix
    L, lineup_ids, spends, pid_to_col, _, _ = build_lineup_matrix(session)
    n_lineups, n_players = L.shape

    if n_lineups == 0:
        print("  WARNING: No lineups found in database")
        return {}, {'error': 'no_lineups'}

    print(f"  Candidate lineups: {n_lineups:,}")

    # Calculate ownership_gini and num_games from NFLPlayers if not provided
    if ownership_gini is None or num_games is None:
        players = session.query(
            NFLPlayer.ownership_proj,
            NFLPlayer.team,
        ).all()

        # Calculate ownership_gini from ownership_proj values
        if ownership_gini is None:
            ownership_values = [p.ownership_proj or 0.0 for p in players if p.ownership_proj is not None]
            if ownership_values:
                ownership_gini = _calculate_gini(ownership_values)
                print(f"  Calculated ownership_gini: {ownership_gini:.4f} from {len(ownership_values)} players")
            else:
                ownership_gini = 0.5
                print(f"  No ownership data found, using default ownership_gini: {ownership_gini}")

        # Calculate num_games from unique gametimes
        if num_games is None:
            teams = set(p.team for p in players if p.team is not None)
            num_games = len(teams) / 2
            print(f"  Calculated num_games: {num_games} unique gametimes")

    # Get player projections for weighting
    P = build_player_samples(session, pid_to_col)
    per_player_mean = np.asarray(P.mean(axis=0), dtype=np.float64)
    lineup_proj_raw = np.asarray(L @ per_player_mean, dtype=np.float64)

    # Get player ownership for weighting
    pid_to_col_inv = {v: k for k, v in pid_to_col.items()}
    player_ownership = get_player_ownership(session, pid_to_col)
    lineup_own_raw = np.asarray(L @ player_ownership, dtype=np.float64)

    # Prepare feature vector for model
    # Features: [total_entries, entry_fee, num_games, ownership_gini]
    X_test = np.array([
        [
            total_entries or field_size,
            entry_fee or 1.0,
            num_games or 1,
            ownership_gini or 0.5,
        ]
    ]).astype(float)

    # Scale features
    X_test_scaled = scaler.transform(X_test)

    # Predict distributions
    proj_probs = proj_model.predict(X_test_scaled)[0]  # [BINS]
    own_probs = own_model.predict(X_test_scaled)[0]   # [BINS]

    # Ensure valid probabilities
    proj_probs = np.clip(proj_probs, 0.0, None)
    own_probs = np.clip(own_probs, 0.0, None)
    if proj_probs.sum() > 0:
        proj_probs = proj_probs / proj_probs.sum()
    else:
        proj_probs = np.ones_like(proj_probs) / len(proj_probs)
    if own_probs.sum() > 0:
        own_probs = own_probs / own_probs.sum()
    else:
        own_probs = np.ones_like(own_probs) / len(own_probs)

    print(f"  Predicted distributions:")
    print(f"    Projection: {proj_probs[:3]}... (first 3 bins)")
    print(f"    Ownership:  {own_probs[:3]}... (first 3 bins)")

    # Normalize lineup projections and ownerships to [0, 1]
    if lineup_proj_raw.max() > lineup_proj_raw.min():
        proj_norm = (lineup_proj_raw - lineup_proj_raw.min()) / (lineup_proj_raw.max() - lineup_proj_raw.min())
    else:
        proj_norm = np.full_like(lineup_proj_raw, 0.5)

    if lineup_own_raw.max() > lineup_own_raw.min():
        own_norm = (lineup_own_raw - lineup_own_raw.min()) / (lineup_own_raw.max() - lineup_own_raw.min())
    else:
        own_norm = np.full_like(lineup_own_raw, 0.5)

    # Assign each lineup to bins
    BINS = len(bin_edges) - 1
    proj_bins = np.clip(np.digitize(proj_norm, bin_edges) - 1, 0, BINS - 1)
    own_bins = np.clip(np.digitize(own_norm, bin_edges) - 1, 0, BINS - 1)

    # Combine probabilities: use product (both must match)
    # For each lineup i, weight = proj_prob[bin_i] * own_prob[bin_i]
    weights = np.zeros(n_lineups, dtype=np.float64)
    for i in range(n_lineups):
        weights[i] = proj_probs[proj_bins[i]] * own_probs[own_bins[i]]

    # Ensure nonzero weights
    weights = np.clip(weights, 1e-12, None)
    weights = weights / weights.sum()

    print(f"  Weight stats: min={weights.min():.2e}, mean={weights.mean():.2e}, max={weights.max():.2e}")

    # Sample with replacement
    rng = np.random.default_rng(42)
    sampled_indices = rng.choice(n_lineups, size=field_size, replace=True, p=weights)

    # Count duplicates
    from collections import Counter
    counts = Counter(sampled_indices)
    counts_by_lineup = {int(lineup_ids[i]): int(count) for i, count in counts.items()}

    # Diagnostics
    unique_selected = len(counts)
    actual_dupes = np.array(list(counts.values()))

    meta = {
        'sampler': 'classic_learned_model',
        'total_candidate_lineups': n_lineups,
        'unique_selected': unique_selected,
        'uniqueness_rate': unique_selected / field_size,
        'actual_dupes': {
            'min': int(actual_dupes.min()),
            'max': int(actual_dupes.max()),
            'mean': float(actual_dupes.mean()),
            'median': float(np.median(actual_dupes)),
        },
        'contest_params': {
            'total_entries': total_entries,
            'entry_fee': entry_fee,
            'num_games': num_games,
            'ownership_gini': ownership_gini,
        },
    }

    print(f"  ✓ Sampled field: {unique_selected:,} unique lineups "
          f"({meta['uniqueness_rate']:.1%})")
    print(f"  Duplicate range: {actual_dupes.min()}-{actual_dupes.max()} "
          f"(mean={actual_dupes.mean():.1f})")

    return counts_by_lineup, meta

# ==============================================================================