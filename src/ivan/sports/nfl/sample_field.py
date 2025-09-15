#!/usr/bin/env python3
# scripts/nfl/sample_field_simple.py
"""
Simple field sampler:
1) Choose a salary-remaining bin by a fixed prior (nearly exponential toward near-cap).
2) Within that bin, choose lineups proportional to their projected score (from correlated samples).
3) Sample WITH REPLACEMENT, aggregate dupes.

Edit SALARY_PRIOR below to shape the field. We normalize weights automatically.
"""

from __future__ import annotations
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import sparse as sp
from sqlalchemy.orm import Session

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLPlayer, Lineup
from ivan.sports.nfl.correlated_samples import generate_correlated_samples_gaussian

SALARY_CAP = 50000

# --- Your nearly-exponential prior over *salary remaining* in dollars ---
# Bins are [lo, hi), in dollars remaining = SALARY_CAP - spend.
# Feel free to edit the weights; they will be normalized to sum to 1.
# Note: I added an explicit 200–299 bin (weight 0.0 by default) so ranges are complete.
SALARY_PRIOR: List[Tuple[int, int, float]] = [
    (0,   200, 0.30),   # < $200 remaining
    (200, 300, 0.20),   # $200–$299 (you can bump this if you want)
    (300, 400, 0.13),
    (400, 500, 0.13),
    (500, 600, 0.07),
    (600, 700, 0.05),
    (700, 800, 0.03),
    (800, 900, 0.03),
    (900, 1000, 0.03),
    (1000, 10**9, 0.03),  # >= $1000 remaining
]

DEFAULTS = {
    "pool": {
        "size": 10000,          # default computed later
        "embed_dims": 24,      # dimensions for JL embedding (Lower = faster, slightly coarser diversity; higher = slower, crisper diversity.)
        "beta_fpts": 0.5,      # tradeoff for quality vs distance
        "seed": None           # RNG seed for reproducibility
    },
    "dupe_model": {
        "theta0": -1.8,                 # base log-rate (theta0 shifts base level (think “global field density”).)
        "theta_remain": -0.6,           # more remaining salary -> fewer dupes (is usually negative; more salary remaining → fewer dupes.)
        "theta_cpt": 1.0,               # stronger CPT -> more dupes (strong CPTs duplicate more (Showdown-like behavior).)
        "theta_fpts": 1.3,              # stronger lineup -> more dupes (high-projection lineups get allocated more often)
        "link": "exp",                  # "exp" (Poisson-like) or "logit"
        "min_count": 1,                 # at least one copy if chosen into pool
        "mid_boost_strength": 0.6,      # ↑ to fatten mid more (0.4–0.9 typical)
        "mid_boost_sigma": 0.18,        # widen/narrow the boosted band (0.12–0.25)
        "lam_floor_frac": 0.06,         # adds a small uniform floor (0.03–0.10)
        "stochastic_rounding": True     # turn on the +1 Bernoulli bump
    },
    "independent_draws": {
        "n_sims": 64          # sims to build lineup-embedding; keep modest for performance
    }
}

def _percentiles_01(x: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(x))
    return (r + 0.5) / float(x.size)

def _smooth_lam(dupe_lambda: np.ndarray, *,
                strength: float = 0.6,   # how much to flatten the mid (0..1.0+)
                sigma: float = 0.18,     # width of the mid bump (0.10..0.30 works)
                eps_frac: float = 0.06) -> np.ndarray:  # additive floor as % of mean
    """
    Flatten the **middle percentiles** of lambda while leaving the head/tail similar.
    lam' = lam ** (1 / temp(p)) + eps_frac * mean(lam)
    where temp(p) = 1 + strength * exp(-((p-0.5)^2)/(2*sigma^2)).
    """
    lam = np.clip(dupe_lambda.astype(np.float64), 1e-12, None)
    p = _percentiles_01(lam)
    s = max(1e-3, float(sigma))
    temp = 1.0 + float(strength) * np.exp(-0.5 * ((p - 0.5) / s) ** 2)
    lam = np.power(lam, 1.0 / temp)
    lam += float(eps_frac) * lam.mean()
    return np.clip(lam, 1e-12, None)


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-12:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (x - lo) / (hi - lo)

def _dupe_lambda_simple(mean_proj: np.ndarray,
                        cpt_means: np.ndarray,
                        remaining: np.ndarray,
                        params: dict) -> np.ndarray:
    """
    Super-simple, fast model using [0,1] features + temperature (dup_sharpness).
    More projection / stronger CPT / closer to cap -> more dupes.
    """
    a0   = float(params.get("a0", -1.5))
    af   = float(params.get("a_fpts",  2.2))
    ac   = float(params.get("a_cpt",   1.2))
    ar   = float(params.get("a_remain",1.6))   # positive on (1 - remain01)

    # features -> [0,1]
    f01  = _minmax01(mean_proj)
    c01  = _minmax01(cpt_means)
    r01  = _minmax01(remaining)       # 0 -> low remaining, 1 -> high remaining
    near_cap = 1.0 - r01              # near_cap = 1 when remaining is small

    z = a0 + af * f01 + ac * c01 + ar * near_cap
    lam = np.exp(np.clip(z, -20.0, 20.0))

    # sharpness ("temperature") to concentrate mass on chalk
    sharp = float(params.get("dup_sharpness", 1.75))
    if sharp > 1.0:
        lam = np.power(lam, sharp)

    # small uniform floor to avoid starving tail
    floor_frac = float(params.get("lam_floor_frac", 0.02))
    lam = lam + floor_frac * lam.mean()

    return np.clip(lam, 1e-12, None).astype(np.float64)

def _counts_topup_proportional(indices: np.ndarray, weights: np.ndarray, total: int) -> np.ndarray:
    """
    Allocate `total` integer units over `indices` proportionally to `weights` (>=0).
    Exact fill via floor + largest-fraction remainder.
    """
    w = np.clip(weights, 0.0, None).astype(np.float64)
    if w.sum() <= 0 or total <= 0 or indices.size == 0:
        return np.zeros(indices.size, dtype=int)
    p = w / w.sum()
    exp = p * int(total)
    base = np.floor(exp).astype(int)
    diff = int(total) - int(base.sum())
    if diff != 0:
        frac = exp - base
        order = np.argsort(-frac)[:abs(diff)]
        base[order] += 1
    return base

def _allocate_counts_two_phase(order_idx: np.ndarray,
                               dupe_lambda: np.ndarray,
                               total_M: int,
                               *,
                               max_unique: int | None = None,
                               min_count: int = 1,
                               stochastic_rounding: bool = True,
                               seed: int | None = None) -> dict[int, int]:
    """
    Phase A: walk ORDER; accept a lineup only if its FULL integer need fits; stop after `max_unique` accepts.
    Phase B: top-up remaining among ACCEPTED ONLY, proportionally to their dupe_lambda.
    """
    rng = np.random.default_rng(seed)
    remaining = int(total_M)

    accepted_positions: list[int] = []    # positions in ORDER we accepted
    counts_by_pos: dict[int, int] = {}    # ORDER position -> count

    # ----- Phase A: accept up to max_unique lineups -----
    pos_to_idx = {pos: int(order_idx[pos]) for pos in range(order_idx.size)}
    picked = np.zeros(order_idx.size, dtype=bool)

    while remaining > 0:
        # denom over still-available
        avail = ~picked
        if not avail.any():
            break
        denom = float(dupe_lambda[order_idx[avail]].sum())
        if denom <= 0:
            break

        progressed = False
        for pos in range(order_idx.size):
            if remaining <= 0:
                break
            if picked[pos]:
                continue
            if max_unique is not None and len(accepted_positions) >= int(max_unique):
                break

            i = pos_to_idx[pos]
            mu = remaining * (float(dupe_lambda[i]) / denom)
            base = np.floor(mu)
            frac = mu - base
            bump = 1 if (stochastic_rounding and rng.random() < frac) else 0
            need = int(max(min_count, base + bump))

            if need <= remaining and need > 0:
                counts_by_pos[pos] = need
                accepted_positions.append(pos)
                remaining -= need
                picked[pos] = True
                progressed = True

        if not progressed:
            # Either we reached max_unique or no item fit its FULL ask. Stop Phase A.
            break

    # ----- Phase B: top-up among ACCEPTED ONLY -----
    if remaining > 0 and accepted_positions:
        acc_idx = np.array([pos_to_idx[p] for p in accepted_positions], dtype=int)
        acc_weights = dupe_lambda[acc_idx]
        topup = _counts_topup_proportional(
            indices=acc_idx,
            weights=acc_weights,
            total=remaining
        )
        for j, pos in enumerate(accepted_positions):
            counts_by_pos[pos] = counts_by_pos.get(pos, 0) + int(topup[j])
        remaining = 0

    return counts_by_pos

def _allocate_counts_greedy_capacity(order_idx: np.ndarray, dupe_lambda: np.ndarray,
                                     total_M: int, *, min_count: int = 1,
                                     stochastic_rounding: bool = True,
                                     seed: int | None = None) -> dict[int, int]:
    """
    Walk the ORDER. For each lineup:
      - compute expected copies against current remaining + current denom
      - stochastically round (floor + Bernoulli(frac))
      - accept only if it fits; else skip
    """
    rng = np.random.default_rng(seed)
    remaining = int(total_M)
    accepted: dict[int, int] = {}
    picked = np.zeros(order_idx.shape[0], dtype=bool)

    while remaining > 0:
        changed = False
        avail_mask = ~picked
        if not avail_mask.any():
            break

        denom = float(dupe_lambda[order_idx[avail_mask]].sum())
        if denom <= 0:
            break

        for t, idx in enumerate(order_idx):
            if remaining <= 0:
                break
            if picked[t]:
                continue

            mu = remaining * (float(dupe_lambda[idx]) / denom)
            base = np.floor(mu)
            frac = mu - base
            bump = 1 if (stochastic_rounding and rng.random() < frac) else 0
            need = int(max(min_count, base + bump))

            if need <= remaining:
                accepted[t] = need
                remaining -= need
                picked[t] = True
                changed = True

        if not changed:
            break
    return accepted

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd

def _embed_lineups_fast(L, P_corr, d_embed: int = 24, *, seed: int | None = None):
    """
    Returns S_embed [n_lineups x d_embed] without forming S = L @ P_corr.T.
    We compute S_embed = L @ (P_corr.T @ R), with JL random projection R.
    """
    rng = np.random.default_rng(seed)
    D = int(P_corr.shape[0])            # n_sims
    d = int(max(8, min(64, d_embed)))   # keep small
    # JL random projection
    R = rng.standard_normal((D, d)).astype(np.float32) / np.sqrt(d)
    # [n_players x d]
    G = (P_corr.T @ R).astype(np.float32)
    # sparse @ dense -> [n_lineups x d]
    if hasattr(L, "sort_indices"):
        L.sort_indices()
        if hasattr(L, "sum_duplicates"):
            L.sum_duplicates()
    S = (L @ G).astype(np.float32)
    # z-score columns to make Euclidean distances well-conditioned
    S -= S.mean(axis=0, keepdims=True)
    std = S.std(axis=0, keepdims=True)
    std[std < 1e-5] = 1.0
    S /= std
    return S

def build_player_samples_independent(session, pid_to_col: dict[int, int], n_sims: int | None = None,
                                     *, seed: int | None = None) -> np.ndarray:
    """
    Build an [n_sims x n_players] matrix of per-player **independent** fpts draws by
    column-wise shuffling the marginal samples (breaks cross-player correlation,
    preserves player marginals).
    """
    from ivan.sports.nfl.correlated_samples import generate_correlated_samples_gaussian  # reuse your source
    raw = generate_correlated_samples_gaussian(session)  # {pid: np.ndarray}
    sizes = [len(v) for v in raw.values() if isinstance(v, np.ndarray) and v.size]
    if not sizes:
        raise RuntimeError("No player samples available for independent draws.")
    base_ns = int(min(sizes))
    ns = int(n_sims or base_ns)

    P = np.zeros((ns, len(pid_to_col)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for pid, col in pid_to_col.items():
        v = raw.get(pid)
        if not isinstance(v, np.ndarray) or v.size == 0:
            continue
        w = v.astype(np.float32, copy=False)
        # normalize to ns
        if w.shape[0] < ns:
            reps = int(np.ceil(ns / w.shape[0]))
            w = np.tile(w, reps)[:ns]
        else:
            w = w[:ns]
        # BREAK correlation by shuffling independently per player
        rng.shuffle(w)
        P[:, col] = w
    return P

def _lineup_core_stats(L, spends, cpt_col_by_row, P_indep: np.ndarray, P_corr: np.ndarray):
    """
    Returns: mean_proj, cpt_means, remaining, features_dict, S (lineup embeddings in correlated sample space)
    - mean_proj uses independent draws (fair for 'popularity weights')
    - S uses correlated draws to space lineups in realistic sample space
    """
    per_player_mean_indep = np.asarray(P_indep.mean(axis=0), dtype=np.float64)  # [n_players]
    mean_proj = np.asarray(L @ per_player_mean_indep, dtype=np.float64)         # [n_lineups]

    # CPT means for each lineup
    n_lineups = L.shape[0]
    cpt_means = np.zeros(n_lineups, dtype=np.float64)
    mask = cpt_col_by_row >= 0
    if mask.any():
        cpt_means[mask] = per_player_mean_indep[cpt_col_by_row[mask]]

    remaining = (SALARY_CAP - spends).astype(np.int32)

    features = {
        "remaining_z": _zscore(remaining),
        "cpt_z": _zscore(cpt_means),
        "fpts_z": _zscore(mean_proj),
    }

    # Embedding for diversity: lineup-by-sim scores under correlated space
    # P_corr: [n_sims x n_players]  -> L @ P_corr.T == [n_lineups x n_sims]
    S = L @ P_corr.T
    # z-score each dimension for fair Euclidean distances
    mu = S.mean(axis=0)
    sd = S.std(axis=0)
    sd[sd < 1e-6] = 1.0
    S = (S - mu) / sd
    return mean_proj, cpt_means, remaining, features, S

def _lineup_core_features(L, spends, cpt_col_by_row, P_indep: np.ndarray):
    """
    Returns: mean_proj, cpt_means, remaining, features_dict
    Uses INDEPENDENT per-player sims for popularity/dupe features.
    """
    per_player_mean_indep = np.asarray(P_indep.mean(axis=0), dtype=np.float64)  # [n_players]
    mean_proj = np.asarray(L @ per_player_mean_indep, dtype=np.float64)         # [n_lineups]

    # CPT mean per lineup
    n_lineups = L.shape[0]
    cpt_means = np.zeros(n_lineups, dtype=np.float64)
    mask = cpt_col_by_row >= 0
    if mask.any():
        cpt_means[mask] = per_player_mean_indep[cpt_col_by_row[mask]]

    remaining = (SALARY_CAP - spends).astype(np.int32)

    features = {
        "remaining_z": _zscore(remaining),
        "cpt_z": _zscore(cpt_means),
        "fpts_z": _zscore(mean_proj),
    }
    return mean_proj, cpt_means, remaining, features

def _dupe_lambda_from_features(features: dict[str, np.ndarray], theta: dict) -> np.ndarray:
    z = (
        float(theta.get("theta0", -2.0))
        + float(theta.get("theta_remain", -0.7)) * features["remaining_z"]
        + float(theta.get("theta_cpt", 1.1))      * features["cpt_z"]
        + float(theta.get("theta_fpts", 1.4))     * features["fpts_z"]
    )
    link = str(theta.get("link", "exp")).lower()
    if link == "logit":
        lam = 1.0 / (1.0 + np.exp(-z))
        lam = np.clip(lam, 1e-9, 1.0)
    else:
        lam = np.exp(np.clip(z, -20.0, 20.0))  # positive rate
    return lam.astype(np.float64)

def _pick_diverse_pool(S: np.ndarray, mean_proj: np.ndarray, k: int, *,
                       beta_fpts: float = 0.4, seed: int | None = None) -> np.ndarray:
    """
    Farthest-first with static scaling + dot-product distance updates.
    Distances are squared L2 in float32. No per-iter min/max scans.
    """
    n, d = S.shape
    k = int(min(max(1, k), n))
    rng = np.random.default_rng(seed)

    # precompute norms and fpts scale once
    S = S.astype(np.float32, copy=False)
    norms = np.einsum("ij,ij->i", S, S).astype(np.float32)

    f = mean_proj.astype(np.float32, copy=False)
    f_min, f_max = float(np.min(f)), float(np.max(f))
    f_scaled = (f - f_min) / max(1e-8, (f_max - f_min))

    # seed with best fpts
    first = int(np.argmax(f))
    selected = np.empty(k, dtype=np.int32)
    selected[0] = first

    # init dmin to distance from first
    Sj = S[first]                                  # [d]
    dot = S @ Sj                                   # [n]
    dmin = norms + float(norms[first]) - 2.0 * dot
    dmin[first] = -np.inf

    # static denominator from initial state
    denom = max(1e-8, float(np.nanmax(dmin)))

    for t in range(1, k):
        # one static rescale of distance + fpts bonus
        score = (dmin / denom) + float(beta_fpts) * f_scaled
        j = int(np.argmax(score))
        selected[t] = j

        # update distances with dot trick
        Sj = S[j]
        dot = S @ Sj
        dj = norms + float(norms[j]) - 2.0 * dot
        np.minimum(dmin, dj, out=dmin)
        dmin[selected[:t+1]] = -np.inf

    return selected


def _counts_from_pool(pool_idx: np.ndarray, dupe_lambda: np.ndarray, total_M: int, *,
                      min_count: int = 1) -> np.ndarray:
    """
    Convert dupe propensities into integer counts that sum to total_M.
    Uses proportional allocation with rounding + remainder fixups. Ensures >= min_count.
    """
    r = np.clip(dupe_lambda[pool_idx].astype(np.float64), 1e-12, None)
    p = r / r.sum()
    expected = p * int(total_M)

    counts = np.floor(expected).astype(int)
    counts = np.maximum(counts, int(min_count))

    diff = int(total_M) - int(counts.sum())
    if diff != 0:
        # greedily assign remainder by largest fractional parts
        frac = expected - np.floor(expected)
        order = np.argsort(-frac) if diff > 0 else np.argsort(frac)
        i = 0
        while diff != 0 and i < len(order):
            t = order[i]
            if diff > 0:
                counts[t] += 1
                diff -= 1
            else:
                if counts[t] > int(min_count):
                    counts[t] -= 1
                    diff += 1
            i += 1
    return counts

CPT_SLOT = 511
FLEX_SLOT = 512

def _normalize_probs(w: np.ndarray, size: int) -> np.ndarray | None:
    """
    Turn arbitrary nonnegative weights into a valid probability vector of length `size`.
    Returns None if sum==0 after sanitization.
    """
    p = np.asarray(w, dtype=np.float64).copy()
    # sanitize
    p[~np.isfinite(p)] = 0.0
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0.0:
        return None
    p /= s
    # force exact sum=1.0 to satisfy numpy
    residual = 1.0 - p.sum()
    if abs(residual) > 1e-12:
        j = int(np.argmax(p))
        p[j] += residual
        if p[j] < 0.0:  # pathological; fall back to uniform
            return None
    return p


def _coerce_player_ids(val) -> List[int]:
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
    Returns:
      L: csr [n_lineups x n_players] with 1 if lineup contains player
      lineup_ids: np.array of lineup ids
      spends: np.array DK spend per lineup
      pid_to_col: dict player_id -> col
      cpt_col_by_row: np.ndarray [n_lineups] -> column index of the CPT player (or -1)
      flex_cols_by_row: list[list[int]] column indices of FLEX players per lineup
    """
    rows = session.query(Lineup.id, Lineup.player_ids_list).all()
    if not rows:
        raise RuntimeError("No lineups found.")
    lineup_ids, parsed, all_pids = [], [], set()
    for lid, j in rows:
        pids = _coerce_player_ids(j)
        parsed.append((lid, pids))
        lineup_ids.append(lid)
        all_pids.update(pids)

    lineup_ids = np.array(lineup_ids, dtype=np.int64)
    pid_list = sorted(all_pids)
    pid_to_col = {pid: i for i, pid in enumerate(pid_list)}

    # Build sparse lineup matrix
    nnz = sum(len(p) for _, pids in parsed for p in [pids])
    r = np.empty(nnz, dtype=np.int32)
    c = np.empty(nnz, dtype=np.int32)
    v = np.ones(nnz, dtype=np.float32)
    k = 0
    for i, (_, pids) in enumerate(parsed):
        for pid in pids:
            r[k] = i
            c[k] = pid_to_col[pid]
            k += 1
    L = sp.csr_matrix((v, (r, c)), shape=(len(parsed), len(pid_list)))

    # Grab salaries + roster_slot for each player id
    q = session.query(NFLPlayer.id, NFLPlayer.salary, NFLPlayer.roster_slot_id)\
               .filter(NFLPlayer.id.in_(pid_list)).all()
    salaries = {}
    roster_slot = {}
    for pid, sal, slot in q:
        salaries[int(pid)] = int(sal)
        roster_slot[int(pid)] = int(slot) if slot is not None else -1

    # spend vector
    sal_vec = np.array([int(salaries.get(pid, 0)) for pid in pid_list], dtype=np.int32)
    spends = (L @ sal_vec).astype(np.int32)

    # Identify CPT & FLEX columns per-lineup
    cpt_col_by_row = np.full(L.shape[0], -1, dtype=np.int32)
    flex_cols_by_row: list[list[int]] = [[] for _ in range(L.shape[0])]
    for row_idx, (_, pids) in enumerate(parsed):
        for pid in pids:
            col = pid_to_col[pid]
            slot = roster_slot.get(pid, -1)
            if slot == CPT_SLOT:
                cpt_col_by_row[row_idx] = col
            elif slot == FLEX_SLOT or slot == -1:
                flex_cols_by_row[row_idx].append(col)

    return L, lineup_ids, spends, pid_to_col, cpt_col_by_row, flex_cols_by_row

def _percentile_rank(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Percentile rank of each value vs ref (both 1D).
    Returns [0,1]. If ref has 1 unique value, returns 0.5.
    """
    ref_sorted = np.sort(ref[~np.isnan(ref)])
    if ref_sorted.size <= 1 or np.allclose(ref_sorted[0], ref_sorted[-1]):
        return np.full_like(values, 0.5, dtype=np.float64)
    # Use searchsorted for vectorized ranks
    idx = np.searchsorted(ref_sorted, values, side="right")
    return np.clip(idx / ref_sorted.size, 0.0, 1.0).astype(np.float64)


def build_player_samples(session: Session, pid_to_col: Dict[int, int]) -> np.ndarray:
    """
    Returns P [n_sims x n_players]; if the generator has fewer sims, resample with replacement.
    """
    raw = generate_correlated_samples_gaussian(session)  # {pid: np.ndarray}
    sizes = [len(x) for x in raw.values() if isinstance(x, np.ndarray)]
    if not sizes:
        raise RuntimeError("generate_correlated_samples returned no vectors.")
    have = min(sizes)
    idx = np.arange(have)

    P = np.zeros((have, len(pid_to_col)), dtype=np.float32)
    for pid, col in pid_to_col.items():
        vec = raw.get(pid)
        if isinstance(vec, np.ndarray) and vec.ndim == 1 and len(vec) >= have:
            P[:, col] = vec[idx].astype(np.float32, copy=False)
    return P


def sample_sd_field(
    session: Session,
    field_size: int = 10000,
    config: dict = None,
    mode = "showdown"
):
    L, lineup_ids, spends, pid_to_col, cpt_col_by_row, flex_cols_by_row = build_lineup_matrix(session)
    n_lineups = L.shape[0]
    remaining = (SALARY_CAP - spends).astype(np.int32)

    # Bin masks
    bins = list(SALARY_PRIOR)
    bin_masks = [(remaining >= lo) & (remaining < hi) for lo, hi, _ in bins]

    # Player samples (sims x n_players)
    P = build_player_samples(session, pid_to_col)

    # Lineup and player means
    per_player_mean = np.asarray(P.mean(axis=0), dtype=np.float64)             # [n_players]
    mean_proj       = np.asarray(L @ per_player_mean, dtype=np.float64)        # [n_lineups]

    # Captain projection percentiles
    valid_cpt_cols = np.unique(cpt_col_by_row[cpt_col_by_row >= 0])
    cpt_ref = per_player_mean[valid_cpt_cols] if valid_cpt_cols.size else np.array([np.nan])
    cpt_means = np.full(n_lineups, np.nan, dtype=np.float64)
    mask_has_cpt = cpt_col_by_row >= 0
    cpt_means[mask_has_cpt] = per_player_mean[cpt_col_by_row[mask_has_cpt]]
    cpt_pct = np.zeros(n_lineups, dtype=np.float64)
    if np.isfinite(cpt_ref).any():
        cpt_pct = _percentile_rank(np.nan_to_num(cpt_means, nan=np.nanmin(cpt_ref)), cpt_ref)  # [0,1]
    # else: leave as zeros -> conservative (weak CPT => tiny weight)

    # FLEX chalk score: mean of top-3 FLEX projection percentiles
    # Build reference across all FLEX columns present
    all_flex_cols = np.array(sorted({c for cols in flex_cols_by_row for c in cols}), dtype=np.int32)
    flex_ref = per_player_mean[all_flex_cols] if all_flex_cols.size else np.array([np.nan])

    # Precompute player-level percentiles once for speed
    player_pct_global = np.zeros_like(per_player_mean, dtype=np.float64)
    if np.isfinite(flex_ref).any():
        # percentile vs all FLEX only (closer to real template chalk)
        # For non-FLEX columns, we still assign a percentile vs flex_ref as a rough global scale.
        player_pct_global = _percentile_rank(per_player_mean, flex_ref)

    # For each lineup: take FLEX columns, pick top-3 percentiles, average
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

    # Prior over available bins
    weights = np.array([w for _, _, w in bins], dtype=np.float64)
    has_any = np.array([m.any() for m in bin_masks], dtype=bool)
    if not has_any.any():
        raise RuntimeError("No lineups fall into any salary-remaining bin. Check SALARY_PRIOR/bins.")
    w_avail = weights * has_any.astype(np.float64)
    if w_avail.sum() <= 0:
        w_avail = has_any.astype(np.float64)
    w_avail /= w_avail.sum()

    idx_per_bin = [np.where(m)[0] for m in bin_masks]
    draws_per_bin = np.random.multinomial(int(field_size), w_avail)

    # --- NEW: captain-aware weights inside each bin ---
    for b, n_draws in enumerate(draws_per_bin):
        if n_draws <= 0:
            continue
        idxs = idx_per_bin[b]
        if idxs.size == 0:
            continue

        # Base score: lineup mean projection
        base = np.maximum(mean_proj[idxs], 0.0)

        # CPT popularity term (0..1) ^ BETA
        cpt_term = np.power(np.clip(cpt_pct[idxs], 0.0, 1.0), BETA)

        # Template penalty: hits hardest when FLEX are very chalky AND CPT is weak
        template_penalty = np.exp(-GAMMA * np.maximum(0.0, flex_topk_mean_pct[idxs] - TAU) * (1.0 - cpt_pct[idxs]))

        w = np.power(base, ALPHA) * cpt_term * template_penalty

        probs = _normalize_probs(w, idxs.size)
        if probs is None:
            probs = np.full(idxs.size, 1.0/idxs.size, dtype=np.float64)

        picks = np.random.choice(idxs.size, size=int(n_draws), replace=True, p=probs)
        np.add.at(counts, idxs[picks], 1)

    counts_by_lineup = {int(lineup_ids[i]): int(counts[i]) for i in range(n_lineups)}
    meta = {
        "lineup_ids": lineup_ids,
        "spend": spends,
        "remaining": (SALARY_CAP - spends).astype(np.int32),
        "mean_proj": mean_proj.astype(np.float32),
        "cpt_pct": cpt_pct.astype(np.float32),
        "flex_topk_mean_pct": flex_topk_mean_pct.astype(np.float32),
    }
    return counts_by_lineup, meta

def sample_field_diverse_dupe(
    session: Session,
    *,
    field_size: int,
    config: dict | None = None,
    mode: str = "classic",     # only one sampler now; we ignore mode semantics
):
    """
    Capacity-aware sampler:
      (1) build L/IDs and per-player sims
      (2) JL embed for diversity geometry from CORRELATED sims
      (3) features from INDEPENDENT sims -> dupe propensities
      (4) farthest-first ORDER
      (5) walk the ORDER, allocating FULL dupe counts only if they fit
    """
    cfg = config or {}

    # 1) matrix + metadata
    L, lineup_ids, spends, pid_to_col, cpt_col_by_row, _ = build_lineup_matrix(session)
    n_lineups = L.shape[0]

    # 2) correlated sims -> geometry via JL
    P_corr = build_player_samples(session, pid_to_col)  # [n_sims x n_players]
    pool_cfg  = {**DEFAULTS["pool"], **(cfg.get("pool") or {})}
    seed      = pool_cfg.get("seed", None)
    embed_dims = int(pool_cfg.get("embed_dims", 24))
    S = _embed_lineups_fast(L, P_corr, d_embed=embed_dims, seed=seed)

    # INDEPENDENT sims -> features for dupe weights
    indep_cfg = {**DEFAULTS["independent_draws"], **(cfg.get("independent_draws") or {})}
    P_indep   = build_player_samples_independent(session, pid_to_col,
                                                 n_sims=int(indep_cfg["n_sims"]), seed=seed)
    mean_proj, cpt_means, remaining_vec, _features = _lineup_core_features(L, spends, cpt_col_by_row, P_indep)

    # === SIMPLE DUPE MODEL (default) =========================================
    dm = {**DEFAULTS.get("dupe_model_simple", {}), **(cfg.get("dupe_model_simple") or {})}
    dupe_lambda = _dupe_lambda_simple(mean_proj, cpt_means, remaining_vec, dm)

    # 4) farthest-first ORDER (pool cap)
    pool_size = int(pool_cfg.get("size") or min(n_lineups, max(3000, int(0.10 * field_size))))
    beta      = float(pool_cfg.get("beta_fpts", 0.5))
    order_idx = _pick_diverse_pool(S, mean_proj, k=pool_size, beta_fpts=beta, seed=seed)

    # 5) capacity-aware, TWO-PHASE allocation with UNIQUE CAP
    field_cfg = cfg.get("field", {}) or {}
    max_unique = field_cfg.get("max_unique", None)
    if max_unique is None:
        tur = float(field_cfg.get("target_unique_ratio", 0.12))  # ~12% by default (e.g., 20k/190k ≈ 10.5%)
        max_unique = int(max(1, min(pool_size, round(tur * field_size))))
    min_count  = int(dm.get("min_count", 1))
    accepted_local = _allocate_counts_two_phase(
        order_idx=order_idx,
        dupe_lambda=dupe_lambda,
        total_M=field_size,
        max_unique=max_unique,
        min_count=min_count,
        stochastic_rounding=bool(dm.get("stochastic_rounding", True)),
        seed=seed,
    )

    # map local order positions -> lineup_ids and counts
    counts_by_lineup = {int(lid): 0 for lid in lineup_ids}
    for pos, need in accepted_local.items():
        row_i = int(order_idx[pos])
        lid = int(lineup_ids[row_i])
        counts_by_lineup[lid] = int(need)
        
    # meta
    used = sum(accepted_local.values())
    meta = {
        "sampler": "diverse_dupe_capacity",
        "pool_size": int(len(order_idx)),
        "accepted": int(len(accepted_local)),
        "lineups_used": int(used),
        "remaining": int(field_size - used),
        "mean_proj": {"mean": float(mean_proj.mean()), "std": float(mean_proj.std())},
        "dupe_lambda": {"min": float(dupe_lambda.min()),
                        "max": float(dupe_lambda.max()),
                        "mean": float(dupe_lambda.mean())}
    }
    return counts_by_lineup, meta


def sample_field(
    session: Session,
    *,
    field_size: int,
    config: dict,
    mode: str,
    contest_id: Optional[int] = None,
    use_classic_dupes: bool = False,
) -> Tuple[Dict[int, int], Dict]:
    """
    Unified sampler:
      - showdown -> delegates to sample_sd_field (unchanged)
      - classic  -> weighted sample WITHOUT replacement (unique lineups)
    """
    mode = (mode or "showdown").lower()
    if mode == "showdown":
        return sample_field_diverse_dupe(session, field_size=field_size, config=config, mode=mode)
    if use_classic_dupes:
        return sample_field_diverse_dupe(session, field_size=field_size, config=config, mode=mode)
    return sample_classic_field(session, field_size=field_size, config=config, contest_id=contest_id)


def sample_classic_field(
    session: Session,
    *,
    field_size: int,
    config: dict,
    contest_id: Optional[int] = None,
) -> Tuple[Dict[int, int], Dict]:
    """
    Classic sampler (NO DUPES):
      1) Build lineup matrix L and IDs.
      2) Get per-player simulations via generate_correlated_samples(mode='classic').
      3) Reduce to per-player projections (median by default).
      4) Compute lineup scores = L @ player_proj.
      5) Temperature-softmax -> probabilities.
      6) Weighted sample WITHOUT replacement for `field_size` unique lineups.
    """
    # ---- knobs (under classic.sampling) ----
    def _get(d: dict, path: List[str], default=None):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    statistic   = str(_get(config, ["classic", "sampling", "statistic"], "median")).lower()
    temperature = float(_get(config, ["classic", "sampling", "temperature"], 1.3))  # lower = peakier
    epsilon     = float(_get(config, ["classic", "sampling", "epsilon"], 0.02))     # floor mass

    # ---- 1) lineup matrix & metadata ----
    L, lineup_ids, spends, pid_to_col, _, _ = build_lineup_matrix(session)
    n_lineups, n_players = L.shape

    # ---- 2) player simulations ----
    samples_by_pid = generate_correlated_samples_gaussian(
        session=session,
        n_samples=100
    )  # {pid: np.ndarray}

    # ---- 3) reduce to per-player projection ----
    def _reduce(arr: np.ndarray) -> float:
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
    present  = np.zeros(n_players, dtype=bool)
    for pid, col in pid_to_col.items():
        arr = samples_by_pid.get(pid)
        if isinstance(arr, np.ndarray) and arr.size:
            proj_vec[col] = _reduce(arr)
            present[col]  = True

    # Drop lineups containing any player that didn't get a projection (fast via sparse matvec)
    missing_mask      = (~present).astype(np.int8)           # [n_players]
    invalid_counts    = np.asarray(L @ missing_mask, dtype=np.int32).ravel()
    valid_row_idx     = np.where(invalid_counts == 0)[0]
    if valid_row_idx.size == 0:
        raise ValueError("No valid classic lineups after filtering (check min_fpts/statistic/config).")

    # ---- 4) lineup scores & probabilities ----
    lineup_scores_all = np.asarray(L @ proj_vec,   dtype=np.float64).ravel()
    s                 = lineup_scores_all[valid_row_idx]           # valid subset

    mu  = float(s.mean())
    sd  = float(s.std()) or 1.0
    T   = max(temperature, 1e-6)

    logits = (s - mu) / (sd * T)
    logits -= float(logits.max())  # numerical stability
    w = np.exp(logits)
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w /= w_sum

    # epsilon floor to avoid starving the tail entirely
    eps = max(0.0, min(0.5, epsilon))
    if eps > 0:
        w = (1.0 - eps) * w + eps / len(w)

    # ---- 5) weighted sample WITHOUT replacement ----
    k   = int(min(field_size, len(valid_row_idx)))
    rng = np.random.default_rng()
    chosen_local = rng.choice(len(valid_row_idx), size=k, replace=False, p=w)
    chosen_rows  = valid_row_idx[chosen_local]

    # Each lineup selected at most once (no dupes)
    counts = {int(lineup_ids[i]): 1 for i in chosen_rows}

    meta = {
        "num_candidates": int(n_lineups),
        "num_valid": int(len(valid_row_idx)),
        "selected": int(k),
        "temperature": float(temperature),
        "epsilon": float(eps),
        "score_mean": float(s.mean()),
        "score_std": float(s.std()),
    }
    # If you *really* need to hit an oversized field_size, you can
    # optionally top up with replacement here — but default is no dupes.
    return counts, meta

# --- NEW: Classic with-replacement sampler that blends projection + near-cap ---
def sample_classic_field_dupe(
    session: Session,
    *,
    field_size: int,
    config: dict | None = None,
    contest_id: Optional[int] = None,
) -> Tuple[Dict[int, int], Dict]:
    """
    Classic field sampler (WITH replacement, i.e., allows dupes).

    Strategy (small slates like 2-gamers):
      - Use salary-remaining bins (like Showdown) to control coarse cap behavior.
      - Within each bin, weight by:
            proj_pct ** alpha_fpts  *  near_cap ** alpha_cap
        where:
            proj_pct  = percentile rank of lineup's projected points vs the full field
            near_cap  = 1 - normalized(salary_remaining)
      - Optional smoothing and sharpness let you fatten the middle or crank chalk.

    Returns:
      counts_by_lineup: {lineup_id: count}  (dupes allowed)
      meta: diagnostics incl. num_unique and unique_ids (to "write unique after duping").
    """
    cfg = config or {}
    # ---- knobs (under classic.dupe) ----
    def _get(d: dict, path: List[str], default=None):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    alpha_fpts  = float(_get(cfg, ["classic", "dupe", "alpha_fpts"], 1.25))  # ↑ -> more projection chalk
    alpha_cap   = float(_get(cfg, ["classic", "dupe", "alpha_cap"],  0.90))  # ↑ -> more near-cap chalk
    epsilon     = float(_get(cfg, ["classic", "dupe", "epsilon"],    1e-6))  # floor mass inside bins
    sharpness   = float(_get(cfg, ["classic", "dupe", "dup_sharpness"], 1.0))# >1 to crank chalk
    smooth_mid  = bool(_get(cfg, ["classic", "dupe", "smooth_mid"], True))
    mid_strength= float(_get(cfg, ["classic", "dupe", "mid_boost_strength"], 0.5))
    mid_sigma   = float(_get(cfg, ["classic", "dupe", "mid_boost_sigma"], 0.18))
    lam_floor   = float(_get(cfg, ["classic", "dupe", "lam_floor_frac"], 0.02))
    seed        = _get(cfg, ["classic", "dupe", "seed"], None)

    rng = np.random.default_rng(seed)

    # ---- lineup matrix, salaries, etc. ----
    L, lineup_ids, spends, pid_to_col, _cpt_cols, _flex_cols = build_lineup_matrix(session)
    n_lineups = L.shape[0]
    remaining = (SALARY_CAP - spends).astype(np.int32)

    # ---- per-player sims -> lineup projections (mean works well for popularity) ----
    P = build_player_samples(session, pid_to_col)              # [n_sims x n_players]
    per_player_mean = np.asarray(P.mean(axis=0), dtype=np.float64)
    lineup_proj     = np.asarray(L @ per_player_mean, dtype=np.float64)  # [n_lineups]

    # ---- normalized features ----
    proj_pct  = _percentile_rank(lineup_proj, lineup_proj)                # [0,1]
    rem01     = _minmax01(remaining)                                      # [0,1]
    near_cap  = np.clip(1.0 - rem01, 0.0, 1.0)                            # 0 low spend, 1 near cap

    # ---- global base lambda (before binning) ----
    lam = np.power(np.clip(proj_pct, 1e-12, 1.0), alpha_fpts) * \
          np.power(np.clip(near_cap,  1e-12, 1.0), alpha_cap)

    if sharpness > 1.0:
        lam = np.power(lam, sharpness)

    if smooth_mid:
        lam = _smooth_lam(lam, strength=mid_strength, sigma=mid_sigma, eps_frac=lam_floor)

    lam = np.clip(lam, 1e-12, None)

    # ---- salary-remaining bins (Showdown-like coarse prior) ----
    bins = list(SALARY_PRIOR)
    bin_masks = [(remaining >= lo) & (remaining < hi) for lo, hi, _ in bins]

    weights = np.array([w for _, _, w in bins], dtype=np.float64)
    has_any = np.array([m.any() for m in bin_masks], dtype=bool)
    if not has_any.any():
        raise RuntimeError("No lineups fall into any salary-remaining bin. Check SALARY_PRIOR.")
    w_avail = weights * has_any.astype(np.float64)
    if w_avail.sum() <= 0:
        w_avail = has_any.astype(np.float64)
    w_avail /= w_avail.sum()

    draws_per_bin = rng.multinomial(int(field_size), w_avail)
    idx_per_bin   = [np.where(m)[0] for m in bin_masks]

    # ---- with-replacement sampling inside each bin ----
    counts = np.zeros(n_lineups, dtype=np.int64)
    for b, n_draws in enumerate(draws_per_bin):
        if n_draws <= 0:
            continue
        idxs = idx_per_bin[b]
        if idxs.size == 0:
            continue
        w = lam[idxs].astype(np.float64, copy=True)
        # small floor so tails are reachable
        w += epsilon * float(w.mean() if w.size else 1.0)
        p = _normalize_probs(w, idxs.size)
        if p is None:
            p = np.full(idxs.size, 1.0 / idxs.size, dtype=np.float64)

        picks = rng.choice(idxs.size, size=int(n_draws), replace=True, p=p)
        np.add.at(counts, idxs[picks], 1)

    counts_by_lineup: Dict[int, int] = {int(lineup_ids[i]): int(counts[i]) for i in range(n_lineups)}

    # ---- meta (helps you “write unique after duping”) ----
    used_mask  = counts > 0
    unique_ids = lineup_ids[used_mask].astype(np.int64).tolist()
    meta = {
        "sampler": "classic_dupe_projection_x_nearcap",
        "field_size": int(field_size),
        "num_unique": int(used_mask.sum()),
        "unique_ids": unique_ids,
        "proj_pct": {"mean": float(proj_pct.mean()), "std": float(proj_pct.std())},
        "near_cap": {"mean": float(near_cap.mean()), "std": float(near_cap.std())},
        "lam": {"min": float(lam.min()), "max": float(lam.max()), "mean": float(lam.mean())},
        "bins": {
            "schema": [(int(lo), int(hi), float(w)) for (lo, hi, w) in bins],
            "draws": [int(x) for x in draws_per_bin.tolist()],
        },
    }
    return counts_by_lineup, meta

