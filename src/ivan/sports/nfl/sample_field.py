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
from typing import Dict, List, Tuple

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
    (0,   200, 0.40),   # < $200 remaining
    (200, 300, 0.20),   # $200–$299 (you can bump this if you want)
    (300, 400, 0.13),
    (400, 500, 0.10),
    (500, 600, 0.05),
    (600, 700, 0.03),
    (700, 800, 0.03),
    (800, 900, 0.03),
    (900, 1000, 0.02),
    (1000, 10**9, 0.01),  # >= $1000 remaining
]

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
    """
    rows = session.query(Lineup.id, Lineup.player_ids_list).all()
    if not rows:
        raise RuntimeError("No lineups found.")
    lineup_ids = []
    parsed = []
    all_pids = set()
    for lid, j in rows:
        pids = _coerce_player_ids(j)
        parsed.append((lid, pids))
        lineup_ids.append(lid)
        all_pids.update(pids)

    lineup_ids = np.array(lineup_ids, dtype=np.int64)
    pid_list = sorted(all_pids)
    pid_to_col = {pid: i for i, pid in enumerate(pid_list)}

    nnz = sum(len(p) for _, p in parsed)
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

    # spend vector
    salaries = dict(session.query(NFLPlayer.id, NFLPlayer.salary)
                           .filter(NFLPlayer.id.in_(pid_list)).all())
    sal_vec = np.array([int(salaries.get(pid)) for pid in pid_list], dtype=np.int32)
    spends = (L @ sal_vec).astype(np.int32)

    return L, lineup_ids, spends, pid_to_col


def build_player_samples(session: Session, pid_to_col: Dict[int, int], n_sims: int) -> np.ndarray:
    """
    Returns P [n_sims x n_players]; if the generator has fewer sims, resample with replacement.
    """
    raw = generate_correlated_samples(session)  # {pid: np.ndarray}
    sizes = [len(x) for x in raw.values() if isinstance(x, np.ndarray)]
    if not sizes:
        raise RuntimeError("generate_correlated_samples returned no vectors.")
    have = min(sizes)
    use = n_sims
    idx = np.arange(use) if have >= use else np.random.choice(have, size=use, replace=True)

    P = np.zeros((use, len(pid_to_col)), dtype=np.float32)
    for pid, col in pid_to_col.items():
        vec = raw.get(pid)
        if isinstance(vec, np.ndarray) and vec.ndim == 1 and len(vec) >= have:
            P[:, col] = vec[idx].astype(np.float32, copy=False)
    return P


def sample_sd_field(
    session: Session,
    n_sims: int = 100,
    field_size: int = 10000,
):
    """
    Simple two-stage mixture:
      1) Draw bin counts ~ Multinomial(field_size, normalized SALARY_PRIOR over non-empty bins)
      2) For each bin, sample that many lineups WITH REPLACEMENT, weights ∝ projection (per sim)
    Aggregates counts across sims. Returns (counts_by_lineup, meta)
    """
    # Lineup matrix & metadata
    print(f'sampling field of size {field_size}')
    L, lineup_ids, spends, pid_to_col = build_lineup_matrix(session)
    n_lineups = L.shape[0]
    remaining = (SALARY_CAP - spends).astype(np.int32)

    # Build bin masks once
    bins = list(SALARY_PRIOR)
    bin_masks: List[np.ndarray] = []
    for lo, hi, _w in bins:
        mask = (remaining >= lo) & (remaining < hi)
        bin_masks.append(mask)

    # Player samples
    P = build_player_samples(session, pid_to_col, n_sims=n_sims)

    mean_proj = np.asarray(L @ P.T, dtype=np.float32).mean(axis=1) # [n_lineups]

    counts = np.zeros(n_lineups, dtype=np.int64)

    # Normalize prior over bins that actually have lineups
    weights = np.array([w for _, _, w in bins], dtype=np.float64)
    has_any = np.array([m.any() for m in bin_masks], dtype=bool)
    if not has_any.any():
        raise RuntimeError("No lineups fall into any salary-remaining bin. Check SALARY_PRIOR/bins.")
    w_avail = weights * has_any.astype(np.float64)
    if w_avail.sum() <= 0:
        # If all provided weights were zero, fall back to uniform over available bins
        w_avail = has_any.astype(np.float64)
    w_avail /= w_avail.sum()

    # Convenience: precompute indices per bin
    idx_per_bin = [np.where(m)[0] for m in bin_masks]

    draws_per_bin = np.random.multinomial(int(field_size), w_avail)


    for b, n_draws in enumerate(draws_per_bin):
        if n_draws <= 0:
            continue
        idxs = idx_per_bin[b]
        if idxs.size == 0:
            continue

        # weights within bin ∝ mean projections
        bin_proj = mean_proj[idxs]
        shifted  = bin_proj - np.nanmin(bin_proj) + 1e-9
        probs    = _normalize_probs(shifted, idxs.size)
        if probs is None:
            probs = np.full(idxs.size, 1.0/idxs.size, dtype=np.float64)

        picks = np.random.choice(idxs.size, size=int(n_draws), replace=True, p=probs)
        np.add.at(counts, idxs[picks], 1)

    counts_by_lineup = {int(lineup_ids[i]): int(counts[i]) for i in range(n_lineups)}
    meta = {
        "lineup_ids": lineup_ids,
        "spend": spends,
        "remaining": (SALARY_CAP - spends).astype(np.int32),
        "mean_proj": mean_proj,
    }
    return counts_by_lineup, meta

