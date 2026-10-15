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
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
from scipy import sparse as sp
from sqlalchemy.orm import Session

from ivan.core.utils import coerce_player_ids as _coerce_player_ids
from ivan.core.seed import seed_for
from ivan.sports.nfl.paths import models_dir
from ivan.sports.nfl.models import NFLPlayer, Lineup
from ivan.sports.nfl.correlated_samples import generate_correlated_samples_gaussian

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
        allow_duplicates=use_classic_dupes,
    )

# CONSTANTS
# ==============================================================================

SALARY_CAP = 50000
CPT_SLOT = 511
FLEX_SLOT = 512


def _stats(x: np.ndarray) -> dict:
    """min / mean / quantiles of a vector, JSON-ready."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {}
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {"min": float(x.min()), "mean": float(x.mean()), "max": float(x.max()),
            "p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p90": float(q[4])}

def _dupe_features(session: Session, L, pid_to_col: Dict[int, int], player_ownership: np.ndarray, cpt_col_by_row) -> dict:
    """The dupe model's captain and stack inputs for every lineup row (ivan.sports.nfl.dupe_model)."""
    from ivan.sports.nfl import dupe_model as DMv2
    cols = sorted(pid_to_col.items(), key=lambda kv: kv[1])
    meta = {int(pid): (team, pos, fp, slot) for pid, team, pos, fp, slot in
            session.query(NFLPlayer.id, NFLPlayer.team, NFLPlayer.position, NFLPlayer.fpts_mean, NFLPlayer.roster_slot_id).filter(NFLPlayer.id.in_([p for p, _ in cols])).all()}
    teams = [meta.get(pid, ("", "", None, None))[0] for pid, _ in cols]
    positions = [meta.get(pid, ("", "", None, None))[1] for pid, _ in cols]
    fpts = [meta.get(pid, ("", "", None, None))[2] or 0.0 for pid, _ in cols]
    is_cpt = [meta.get(pid, ("", "", None, None))[3] == CPT_SLOT for pid, _ in cols]
    n_qb, mates = DMv2.lineup_stack_features(L, teams, positions)
    out = {"qb_mates": mates, "n_qb": n_qb, "proj_gap": DMv2.lineup_proj_gap(L, fpts, is_cpt)}
    if cpt_col_by_row is not None:
        own = np.maximum(np.nan_to_num(np.asarray(player_ownership, dtype=float), nan=DMv2.MIN_OWN), DMv2.MIN_OWN) / 100.0
        cpt = np.asarray(cpt_col_by_row)
        out["log_own_cpt"] = np.where(cpt >= 0, np.log(own[np.maximum(cpt, 0)]), np.nan)
    return out


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
    # generate's workers commit in parallel, so ids interleave differently run to run; the seeded sampler below
    # indexes this list, so order it by content to keep a seed reproducible
    rows = sorted(rows, key=lambda r: (tuple(sorted(_coerce_player_ids(r[1]))), int(r[0])))
    
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
    raw = generate_correlated_samples_gaussian(session, seed=seed_for("field/samples"))
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


# ==============================================================================
# OWNERSHIP EXTRACTION
# ==============================================================================

def get_player_ownership_heuristic(
    session: Session,
    pid_to_col: Dict[int, int],
    alpha: float = 2.0,  # Changed: Higher default for better concentration
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

        return dist_mat, meta

    except Exception as e:
        print(f"  WARNING: Diversity computation failed ({e}), using uniform diversity")
        return np.ones((n_lineups, n_lineups)), {'error': str(e)}


# ==============================================================================
# UNIFIED DIVERSITY-GATED DUPLICATION SAMPLER
# ==============================================================================

OWNERSHIP_CALIBRATION_DEFAULTS = {"enabled": True, "iterations": 30, "damping": 0.7, "max_ratio": 4.0, "min_target_pct": 0.05}


def rake_weights_to_ownership(weights: np.ndarray, L, targets_pct: np.ndarray, *, iterations: int = 30, damping: float = 0.7,
                              max_ratio: float = 4.0, min_target_pct: float = 0.05) -> Tuple[np.ndarray, dict]:
    """
    Adjust lineup sampling weights so the field's player exposures match the
    projected ownership (iterative proportional fitting on the lineup x player
    matrix). The duplication model still decides how concentrated the field is
    on its top lineups; this decides who the field is concentrated on.

    weights      [n_lineups] sampling weights (any scale)
    L            [n_lineups, n_players] 0/1 lineup matrix (one column per roster slot id)
    targets_pct  [n_players] projected ownership in % of entries; <= min_target_pct is
                 treated as "as little as possible" (ratio floored at 1/max_ratio)
    Each pass multiplies a lineup's weight by the geometric product of its players'
    (target / current) ratios, damped and capped, so no single pass can swing a
    lineup more than max_ratio ** 6. Returns (weights, info) with the exposure MAE
    before and after.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    w = np.clip(w, 1e-12, None); w /= w.sum()
    t = np.asarray(targets_pct, dtype=np.float64)
    present = np.asarray((L != 0).sum(axis=0)).ravel() > 0        # a target no candidate carries cannot be reached
    reachable = present & np.isfinite(t)
    def exposure(w):
        return 100.0 * np.asarray(L.T @ w).ravel()
    before = exposure(w)
    def mae(e):
        return float(np.abs(e[reachable] - t[reachable]).mean()) if reachable.any() else float("nan")
    mae0 = mae(before)
    # keep the best iterate: with a sparse candidate pool (classic slates, few lineups) the simultaneous
    # update can overshoot, and the result must never be worse than the weights it started from
    best_w, best_mae, best_iter = w.copy(), mae0, 0
    tt = np.where(t > min_target_pct, t, 0.0)
    for k in range(1, max(0, int(iterations)) + 1):
        cur = exposure(w)
        ratio = np.ones_like(cur)
        ok = reachable & (cur > 1e-9)
        ratio[ok] = np.clip(np.where(tt[ok] > 0, tt[ok] / cur[ok], 1.0 / max_ratio), 1.0 / max_ratio, max_ratio)
        logw = np.log(w) + damping * np.asarray(L @ np.log(ratio)).ravel()
        w = np.exp(logw - logw.max()); w = np.clip(w, 1e-300, None); w /= w.sum()
        m = mae(exposure(w))
        if m < best_mae - 1e-9:
            best_w, best_mae, best_iter = w.copy(), m, k
    w = best_w
    after = exposure(w)
    mae1 = mae(after)
    top = np.argsort(-t)[:5]
    per_lineup = float(np.asarray((L != 0).sum(axis=1)).ravel().mean()) if L.shape[0] else 0.0
    info = {"iterations": int(iterations), "best_iteration": int(best_iter), "damping": damping, "max_ratio": max_ratio, "players": int(reachable.sum()),
            # targets are consistent when they sum to 100 x players per lineup; a gap here means they cannot all be met
            "target_sum_pct": float(np.nansum(t[reachable])), "players_per_lineup": per_lineup,
            "unreachable_targets_pct": float(t[~present & np.isfinite(t)].sum()) if (~present).any() else 0.0,
            "exposure_mae_before": mae0, "exposure_mae_after": mae1,
            "chalk_bias_before": float((before[top] - t[top]).mean()), "chalk_bias_after": float((after[top] - t[top]).mean()),
            "max_weight_before": float(np.asarray(weights, dtype=float).max() / np.asarray(weights, dtype=float).sum()),
            "max_weight_after": float(w.max())}
    return w, info


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
    ownership_calibration: Optional[dict] = None,
    dupe_clip_floor: float = 1.0,
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

    # ========================================================================
    # Step 1: Compute diversity matrix (Euclidean distances)
    # ========================================================================
    dist_mat, diversity_meta = compute_lineup_diversity_matrix(
        session=session,
        L=L,
        lineup_ids=lineup_ids,
        pid_to_col=pid_to_col,
        n_sims=32,  # Fast: only 32 sims for diversity
        seed=seed_for("field/diversity", 42),
    )

    if dist_mat.size == 0 or 'error' in diversity_meta:
        print("  WARNING: Diversity computation failed, using uniform diversity")
        dist_mat = np.ones((n_lineups, n_lineups))
        diversity_meta = {'error': 'computation_failed'}

    # ========================================================================
    # Step 2: Use predicted duplication from caller
    # ========================================================================
    predicted_dupes_clipped = np.clip(predicted_dupes, dupe_clip_floor, None)

    # ========================================================================
    # Step 3: Compute diversity bonuses
    # ========================================================================
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

    # ========================================================================
    # Step 4: Combine diversity and duplication
    # ========================================================================
    # Combined weight: higher dupes + higher diversity = higher sampling prob
    combined_weights = predicted_dupes_clipped * diversity_bonus
    combined_weights = np.clip(combined_weights, 1e-12, None)
    combined_weights = combined_weights / combined_weights.sum()

    # ========================================================================
    # Step 4b: Rake the weights so the field's exposures match projected ownership
    # ========================================================================
    # Without this the field over-rosters the chalk: optimizer-built candidates
    # lean on the top projections and the dupe model rewards the same lineups,
    # so the sampled field's top-5 exposure ran ~24 points above actual
    # ownership on the Nov 2025 slates (validation 2026-10-11). Projected
    # ownership itself was within ~3 points; the field should be built on it.
    cal = {**OWNERSHIP_CALIBRATION_DEFAULTS, **(ownership_calibration or {})}
    rake_info: dict = {"enabled": bool(cal.get("enabled", True))}
    if rake_info["enabled"] and player_ownership is not None and np.nanmax(player_ownership) > 0:
        combined_weights, rake_info = rake_weights_to_ownership(
            combined_weights, L, player_ownership, iterations=int(cal["iterations"]), damping=float(cal["damping"]),
            max_ratio=float(cal["max_ratio"]), min_target_pct=float(cal["min_target_pct"]))
        rake_info["enabled"] = True
        print(f"  ownership raking: exposure MAE {rake_info['exposure_mae_before']:.1f} -> {rake_info['exposure_mae_after']:.1f} pts, "
              f"chalk bias {rake_info['chalk_bias_before']:+.1f} -> {rake_info['chalk_bias_after']:+.1f}")

    # ========================================================================
    # Step 5: Sample field
    # ========================================================================
    rng = np.random.default_rng(seed_for("field/sample", 42))
    sampled_indices = rng.choice(n_lineups, size=field_size, replace=True, p=combined_weights)

    # Count duplicates
    counts = Counter(sampled_indices)
    counts_by_lineup = {int(lineup_ids[i]): int(count) for i, count in counts.items()}
    unique_selected = len(counts)
    actual_dupes = np.array(list(counts.values()))

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
        'combined_weights': {
            'min': float(combined_weights.min()),
            'max': float(combined_weights.max()),
            'mean': float(combined_weights.mean()),
        },
        'diversity_meta': diversity_meta,
        'ownership_raking': rake_info,
        # arrays for the run artifact (ivan.sports.nfl.run_artifacts); the simulator pops them before the report is stored
        '_arrays': {'lineup_id': np.asarray(lineup_ids, dtype=np.int64), 'predicted_dupes': predicted_dupes_clipped,
                    'combined_weights': combined_weights},
    }

    return counts_by_lineup, meta


# ==============================================================================
# LEARNED MODEL SAMPLER (SHOWDOWN)
# ==============================================================================

def sample_showdown_field_learned(
    session: Session,
    field_size: int,
    config: dict,
    contest_id: Optional[int] = None,
    model_dir: str | Path | None = None,
) -> Tuple[Dict[int, int], Dict]:
    """
    Sample a showdown field: every candidate's expected copies from the product
    dupe model (ivan.sports.nfl.dupe_model), then the diversity-gated sampler.

    Returns:
        counts_by_lineup: {lineup_id: duplicate_count}
        meta: Diagnostic info
    """
    # Get contest parameters
    entry_fee = 5.0  # Default entry fee
    if contest_id:
        from ivan.sports.nfl.models import HistoricalContest
        contest = session.query(HistoricalContest).get(contest_id)
        
        if contest:
            total_entries = contest.total_entries
            entry_fee = contest.entry_fee if contest.entry_fee else 5.0
        else:
            total_entries = field_size
    else:
        total_entries = field_size
    
    # Build lineup matrix
    L, lineup_ids, spends, pid_to_col, cpt_col_by_row, flex_cols_by_row = build_lineup_matrix(session)
    n_lineups = L.shape[0]

    if n_lineups == 0:
        print("  WARNING: No lineups found in database")
        return {}, {'error': 'no_lineups'}

    # FIXED: Try to use actual ownership first, fall back to heuristic if not available
    # The model was trained on actual DK ownership, so using heuristic causes distribution mismatch
    player_ownership = get_player_ownership(session, pid_to_col)

    # Check if we have valid ownership data
    has_ownership = (player_ownership.max() > 0) and not np.isnan(player_ownership).any()

    if not has_ownership:
        print("  WARNING: no projected ownership on this slate; using a salary/projection heuristic for the field")

        # FIXED: Increased captain_concentration from 2 to 4 for realistic chalk
        alpha = 2.0  # Flex concentration
        salary_weight = 0.3  # Reduced salary impact
        captain_concentration = 4.0  # FIXED: Back to 4 for realistic captain chalk

        player_ownership = get_player_ownership_heuristic(
            session,
            pid_to_col,
            alpha=alpha,
            salary_weight=salary_weight,
            captain_concentration=captain_concentration,
        )
    ownership_info = {
        'source': 'projected' if has_ownership else 'heuristic',
        'min': float(player_ownership.min()), 'max': float(player_ownership.max()),
        'mean': float(player_ownership.mean()),
    }
    dupe_info: dict = {}
    dupe_mode = str(((config or {}).get("field") or {}).get("dupe_model", "product")).lower()
    dupe_clip_floor = 1.0

    # ================================================================
    # PREDICT DUPLICATION BASED ON MODEL MODE
    # ================================================================

    if dupe_mode == "product":
        # copies = N x k x prod(own)^beta x salary term, fitted on real fields (ivan.sports.nfl.dupe_model); the same function gives the
        # sampling weights, so the sampled field's concentration and every candidate's predicted copies agree
        from ivan.sports.nfl import dupe_model as DMv2
        dm = DMv2.load()
        slo = np.asarray(L @ np.log(np.maximum(player_ownership, DMv2.MIN_OWN) / 100.0), dtype=np.float64)
        feats = _dupe_features(session, L, pid_to_col, player_ownership, cpt_col_by_row)
        predicted_dupes = dm.copies(total_entries, True, slo, salary_left=DMv2.SALARY_CAP - np.asarray(spends, dtype=float), **feats)
        dupe_clip_floor = 1e-6
        dupe_info = {"model": "product", "params": dm.params["showdown"], "fitted_on": dm.fitted_on, "contests": dm.contests.get("showdown"),
                     "predicted_dupes": _stats(predicted_dupes), "total_entries": int(total_entries)}
        print(f"  dupes: {dm.describe().split(';')[0]}; candidates' expected copies at {int(total_entries)} entries: "
              f"median {np.median(predicted_dupes):.2f}, p90 {np.percentile(predicted_dupes, 90):.1f}, max {predicted_dupes.max():.1f}")
    else:
        raise ValueError(f"config field.dupe_model must be 'product' (got {dupe_mode!r}); the two-stage model was removed")

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
        dupe_clip_floor=dupe_clip_floor,
        player_ownership=player_ownership,
        total_entries=total_entries,
        ownership_calibration=(config or {}).get("field", {}).get("ownership_calibration"),
    )
    
    # Add metadata
    meta['model_path'] = str(Path(models_dir() if model_dir is None else model_dir))
    meta['model_mode'] = dupe_mode
    meta['contest_params'] = {
        'total_entries': total_entries,
        'entry_fee': entry_fee,
    }
    meta['ownership'] = ownership_info
    meta['dupe_model'] = dupe_info

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
    model_dir: str | Path | None = None,
    allow_duplicates: bool = True,
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
    model_dir = Path(models_dir() if model_dir is None else model_dir)
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
            else:
                ownership_gini = 0.5
                print("  WARNING: no projected ownership on this slate; using default ownership_gini=0.5")

        # Calculate num_games from unique gametimes
        if num_games is None:
            teams = set(p.team for p in players if p.team is not None)
            num_games = len(teams) / 2

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
    classic_dupe_mode = str(((config or {}).get("field") or {}).get("dupe_model", "product")).lower()
    predicted_copies_classic = None
    if classic_dupe_mode == "product" and player_ownership is not None and np.nanmax(player_ownership) > 0:
        # the fitted product rule decides how much the field concentrates on the chalk; the bin weights keep the shape of
        # the projection / ownership distributions and are multiplied in
        from ivan.sports.nfl import dupe_model as DMv2
        dm = DMv2.load()
        slo = np.asarray(L @ np.log(np.maximum(np.nan_to_num(player_ownership, nan=DMv2.MIN_OWN), DMv2.MIN_OWN) / 100.0), dtype=np.float64)
        n_real = float(total_entries or field_size)
        salary_left = DMv2.SALARY_CAP - np.asarray(spends, dtype=float)
        feats = _dupe_features(session, L, pid_to_col, player_ownership, None)
        predicted_copies_classic = dm.copies(n_real, False, slo, salary_left=salary_left, **feats)
        weights = weights * dm.weights(False, slo, salary_left=salary_left, **feats)
        print(f"  dupes (classic): {dm.params['classic']}; candidates' expected copies at {int(n_real)} entries: "
              f"median {np.median(predicted_copies_classic):.2f}, max {predicted_copies_classic.max():.1f}")

    # Ensure nonzero weights
    weights = np.clip(weights, 1e-12, None)
    weights = weights / weights.sum()

    # Rake the weights so the field's player exposures match projected ownership (same step as the showdown sampler)
    cal = {**OWNERSHIP_CALIBRATION_DEFAULTS, **((config or {}).get("field", {}).get("ownership_calibration") or {})}
    rake_info: dict = {"enabled": bool(cal.get("enabled", True))}
    if rake_info["enabled"] and player_ownership is not None and np.nanmax(player_ownership) > 0:
        weights, rake_info = rake_weights_to_ownership(
            weights, L, player_ownership, iterations=int(cal["iterations"]), damping=float(cal["damping"]),
            max_ratio=float(cal["max_ratio"]), min_target_pct=float(cal["min_target_pct"]))
        rake_info["enabled"] = True
        print(f"  ownership raking: exposure MAE {rake_info['exposure_mae_before']:.1f} -> {rake_info['exposure_mae_after']:.1f} pts, "
              f"chalk bias {rake_info['chalk_bias_before']:+.1f} -> {rake_info['chalk_bias_after']:+.1f}")

    # Sampling (with or without replacement)
    rng = np.random.default_rng(seed_for("field/sample", 42))
    if not allow_duplicates:
        if field_size > n_lineups:
            print(f"  ⚠️  Requested field_size {field_size} exceeds unique candidate lineups {n_lineups}; falling back to sampling with replacement.")
            replace_flag = True
        else:
            replace_flag = False
    else:
        replace_flag = True
    sampled_indices = rng.choice(n_lineups, size=field_size, replace=replace_flag, p=weights)

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
        'dedup_strategy': 'no_replacement' if (not allow_duplicates and field_size <= n_lineups) else 'with_replacement',
        'allow_duplicates': allow_duplicates,
        'ownership': {'source': 'projected' if player_ownership.max() > 0 else 'none',
                      'min': float(player_ownership.min()), 'max': float(player_ownership.max()),
                      'mean': float(player_ownership.mean())},
        'predicted_distributions': {'projection_bins': [float(x) for x in proj_probs],
                                    'ownership_bins': [float(x) for x in own_probs]},
        'weights': {'min': float(weights.min()), 'mean': float(weights.mean()), 'max': float(weights.max())},
        'ownership_raking': rake_info,
        # arrays for the run artifact (ivan.sports.nfl.run_artifacts); the simulator pops them before the report is stored
        '_arrays': {'lineup_id': np.asarray(lineup_ids, dtype=np.int64), 'combined_weights': weights,
                    **({'predicted_dupes': predicted_copies_classic} if predicted_copies_classic is not None else {})},
    }

    return counts_by_lineup, meta

# ==============================================================================