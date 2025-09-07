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
from ivan.sports.nfl.correlated_samples import generate_correlated_samples

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

# --- NEW CONFIG KNOBS (tune to taste) ----------------------------------------
ALPHA = 1.0   # impact of lineup mean projection
BETA  = 2.0   # how strongly captain projection percentile drives popularity/dupes
GAMMA = 3.5   # penalty strength for chalky FLEX when CPT is weak
TAU   = 0.85  # threshold (percentile) above which FLEX are considered "very chalky"

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


def build_player_samples(session: Session, pid_to_col: Dict[int, int], config, mode, process) -> np.ndarray:
    """
    Returns P [n_sims x n_players]; if the generator has fewer sims, resample with replacement.
    """
    raw = generate_correlated_samples(session, config=config, mode=mode, process=process)  # {pid: np.ndarray}
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
    P = build_player_samples(session, pid_to_col, config=config, mode=mode, process='sampling')

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

def sample_field(
    session: Session,
    *,
    field_size: int,
    config: dict,
    mode: str,
    contest_id: Optional[int] = None,
) -> Tuple[Dict[int, int], Dict]:
    """
    Unified sampler:
      - showdown -> delegates to sample_sd_field (unchanged)
      - classic  -> weighted sample WITHOUT replacement (unique lineups)
    """
    mode = (mode or "showdown").lower()
    if mode == "showdown":
        return sample_sd_field(session, field_size=field_size, config=config, mode=mode)
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
    samples_by_pid = generate_correlated_samples(
        session=session,
        config=config,
        mode="classic",
        process="sampling",
        overrides=None,
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


