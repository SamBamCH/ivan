"""
Contest simulation and lineup selection logic for NBA Classic contests.

This module provides:
- sample_field: Sample field lineups based on projection likelihood
- sample_field_with_projection_model: Sample using trained distribution model
- simulate_one_contest: Run Monte Carlo simulation on a single contest
- choose_lineups_for_contest: Select optimal diverse lineups for a single contest
"""

import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rich.console import Console
from scipy.spatial.distance import pdist, squareform
from sqlalchemy.orm import Session

from ivan.sports.nba.models import (
    HistoricalContest,
    ContestPayout,
    Lineup,
    LineupSimulationResult,
    NBAPlayer,
)
from ivan.sports.nba.refactor.pre_lock.player import generate_player_samples
from ivan.sports.mlb.payouts import scale_payouts_correctly


def _coerce_player_ids(val) -> List[int]:
    """Convert various player_ids representations to List[int]."""
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


def _load_and_scale_payouts_from_db(
    session: Session,
    contest_id: int,
    num_field_lineups: int,
    original_field_size: int,
    entry_fee: float,
) -> Dict[int, float]:
    """Load and scale payouts from database for a contest."""
    rows = (
        session.query(ContestPayout)
        .filter(ContestPayout.contest_id == contest_id)
        .order_by(ContestPayout.min_position.asc())
        .all()
    )
    
    if not rows:
        raise ValueError(f"No payouts found for contest_id={contest_id}")
    
    payouts_df = pd.DataFrame(
        [(r.min_position, r.max_position, r.payout_amount) for r in rows],
        columns=["Start Rank", "End Rank", "Payout"]
    )
    return scale_payouts_correctly(
        num_field_lineups=num_field_lineups,
        payouts_df=payouts_df,
        original_field_size=original_field_size,
        entry_fee=entry_fee
    )


def sample_field(session: Session, field_size: int, cfg: dict) -> Dict[int, int]:
    """
    Sample field with replacement based on projection likelihood.
    
    Args:
        session: Database session
        field_size: Number of field entries to sample
        cfg: Config dict with optional 'projection_exponent'
        
    Returns:
        Dict mapping lineup_id -> count of times sampled
    """
    all_lineups = session.query(Lineup).all()
    if not all_lineups:
        raise ValueError("No lineups found in database")
    
    lineup_ids = []
    lineup_projection_sums = []
    
    for lineup in all_lineups:
        player_ids = lineup.player_ids_list
        if not player_ids:
            total_projection = 0.0
        else:
            if isinstance(player_ids, str):
                try:
                    player_ids = json.loads(player_ids)
                except:
                    player_ids = []
            if not isinstance(player_ids, list):
                player_ids = list(player_ids) if hasattr(player_ids, '__iter__') else []
            player_ids = [int(pid) for pid in player_ids if pid]
            players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(player_ids)).all()
            player_dict = {p.id: p for p in players}
            total_projection = sum(
                player_dict[pid].fpts_mean
                for pid in player_ids
                if pid in player_dict and player_dict[pid].fpts_mean is not None
            )
        lineup_ids.append(lineup.id)
        lineup_projection_sums.append(float(total_projection))
    
    lineup_projection_sums = np.array(lineup_projection_sums, dtype=np.float64)
    min_projection = np.min(lineup_projection_sums)
    if min_projection <= 0:
        lineup_projection_sums = lineup_projection_sums - min_projection + 1.0
    
    total_projection = np.sum(lineup_projection_sums)
    if total_projection <= 0:
        probabilities = np.ones(len(lineup_ids)) / len(lineup_ids)
    else:
        probabilities = lineup_projection_sums / total_projection
    
    exponent = cfg.get('projection_exponent', 1.0) if cfg else 1.0
    if exponent != 1.0:
        probabilities = np.power(probabilities, exponent)
        probabilities = probabilities / np.sum(probabilities)
    
    sampled_indices = np.random.choice(len(lineup_ids), size=field_size, replace=True, p=probabilities)
    counts_by_lineup = {}
    unique_indices, counts = np.unique(sampled_indices, return_counts=True)
    for idx, count in zip(unique_indices, counts):
        counts_by_lineup[lineup_ids[idx]] = int(count)
    
    return counts_by_lineup


def sample_field_with_projection_model(
    session: Session,
    field_size: int,
    cfg: dict,
    model_path: str = "data/nba/models/field_distribution_model.pkl",
    contest_params: dict = None
) -> Dict[int, int]:
    """
    Sample field using trained projection-based field distribution model.
    
    Args:
        session: Database session
        field_size: Number of field entries to sample
        cfg: Configuration dict
        model_path: Path to pickled field distribution model
        contest_params: Dict with keys [entry_fee, total_entries, multi_entry]
        
    Returns:
        Dict mapping lineup_id -> count of times sampled
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler = model['scaler']
    proj_model = model['proj_model']
    bin_edges = model['bin_edges']
    projection_min = model.get('projection_min', 200.0)

    entry_fee = contest_params.get('entry_fee', 4.0)
    total_entries = contest_params.get('total_entries', 20000)
    multi_entry = contest_params.get('multi_entry', 150)

    X = np.array([[entry_fee, total_entries, multi_entry]])
    X_scaled = scaler.transform(X)
    proj_dist = proj_model.predict(X_scaled)[0]

    # Get all lineups and their projection sums
    all_lineups = session.query(Lineup).all()
    if not all_lineups:
        raise ValueError("No lineups found in database")
    
    lineup_ids = []
    lineup_projection_sums = []
    
    for lineup in all_lineups:
        player_ids = lineup.player_ids_list
        if not player_ids:
            total_projection = 0.0
        else:
            if isinstance(player_ids, str):
                try:
                    player_ids = json.loads(player_ids)
                except:
                    player_ids = []
            if not isinstance(player_ids, list):
                player_ids = list(player_ids) if hasattr(player_ids, '__iter__') else []
            player_ids = [int(pid) for pid in player_ids if pid]
            players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(player_ids)).all()
            player_dict = {p.id: p for p in players}
            total_projection = sum(
                player_dict[pid].fpts_mean
                for pid in player_ids
                if pid in player_dict and player_dict[pid].fpts_mean is not None
            )
        
        # Filter lineups below projection_min
        if total_projection >= projection_min:
            lineup_ids.append(lineup.id)
            lineup_projection_sums.append(float(total_projection))
    
    if not lineup_ids:
        raise ValueError(f"No lineups found with projection >= {projection_min}")
    
    lineup_projection_sums = np.array(lineup_projection_sums, dtype=np.float64)

    # Normalize to [0, 1]
    max_proj = lineup_projection_sums.max()
    if max_proj > projection_min:
        norm_proj = (lineup_projection_sums - projection_min) / (max_proj - projection_min)
    else:
        norm_proj = np.zeros_like(lineup_projection_sums)
    
    norm_proj = np.clip(norm_proj, 0, 1 - 1e-8)
    
    # Bin each lineup
    bin_indices = np.floor(norm_proj * (len(bin_edges) - 1)).astype(int)
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

    # Group lineups by bin
    lineups_by_bin = {}
    for idx, bin_idx in enumerate(bin_indices):
        if bin_idx not in lineups_by_bin:
            lineups_by_bin[bin_idx] = []
        lineups_by_bin[bin_idx].append(idx)

    # Sample bins according to model distribution
    num_bins = len(proj_dist)
    bin_probs = proj_dist / proj_dist.sum() if proj_dist.sum() > 0 else np.ones(num_bins) / num_bins
    sampled_bins = np.random.choice(num_bins, size=field_size, replace=True, p=bin_probs)
    
    counts_by_lineup = {}
    for bin_idx in sampled_bins:
        if bin_idx in lineups_by_bin and lineups_by_bin[bin_idx]:
            lineup_idx = np.random.choice(lineups_by_bin[bin_idx])
            lineup_id = lineup_ids[lineup_idx]
            counts_by_lineup[lineup_id] = counts_by_lineup.get(lineup_id, 0) + 1
        else:
            # Fallback to random
            lineup_idx = np.random.choice(len(lineup_ids))
            lineup_id = lineup_ids[lineup_idx]
            counts_by_lineup[lineup_id] = counts_by_lineup.get(lineup_id, 0) + 1
    
    unique_lineups = len(counts_by_lineup)
    total_entries_sampled = sum(counts_by_lineup.values())
    avg_dupe = total_entries_sampled / unique_lineups if unique_lineups > 0 else 0
    
    print(f"Sampled from {len(lineup_ids)} lineups (dropped {len(all_lineups) - len(lineup_ids)} below {projection_min:.0f})")
    print(f"Field: {unique_lineups} unique, {total_entries_sampled} total (avg {avg_dupe:.1f}x duplication)")
    
    return counts_by_lineup


def _expand_sampled_field(
    session: Session,
    counts_by_lineup: Dict[int, int],
) -> Tuple[List[int], List[List[int]], List[int], Dict[int, int], np.ndarray, np.ndarray]:
    """Expand sampled field counts into simulation-ready arrays."""
    
    if not counts_by_lineup:
        return [], [], [], {}, np.array([], dtype=int), np.array([], dtype=int)

    u_lids = sorted([lid for lid, c in counts_by_lineup.items() if c > 0])
    cnts = np.array([int(counts_by_lineup[lid]) for lid in u_lids], dtype=np.int64)
    starts = np.cumsum(np.r_[0, cnts[:-1]])
    ends = starts + cnts
    M = int(cnts.sum())

    # Fetch lineups in batches
    BATCH_SIZE = 900
    pid_lists_by_lid = {}
    
    for i in range(0, len(u_lids), BATCH_SIZE):
        batch = u_lids[i:i + BATCH_SIZE]
        rows = (
            session.query(Lineup.id, Lineup.player_ids_list)
            .filter(Lineup.id.in_(batch))
            .all()
        )
        for lid, j in rows:
            pid_lists_by_lid[int(lid)] = _coerce_player_ids(j)

    all_pids = set()
    for lid in u_lids:
        all_pids.update(pid_lists_by_lid.get(lid, []))
    unique_pids = sorted(all_pids)
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

    entry_lineup_ids: List[int] = [0] * M
    lineup_ix: List[List[int]] = [None] * M
    
    for u, lid in enumerate(u_lids):
        pids = pid_lists_by_lid.get(lid, [])
        idxs = [pid_to_idx[p] for p in pids if p in pid_to_idx]
        a, b = int(starts[u]), int(ends[u])
        for pos in range(a, b):
            entry_lineup_ids[pos] = lid
            lineup_ix[pos] = idxs

    return entry_lineup_ids, lineup_ix, unique_pids, pid_to_idx, starts, ends


def _build_player_samples_matrix(
    session: Session,
    pid_to_idx: Dict[int, int],
    n_samples: int = 1000,
) -> Tuple[np.ndarray, int]:
    """Build player samples matrix for simulation."""
    
    raw = generate_player_samples(
        session=session,
        n_samples=n_samples,
        randomness=1.0
    )
    
    ns = 0
    for v in raw.values():
        if isinstance(v, np.ndarray):
            ns = int(v.shape[0])
            break
    
    nP = len(pid_to_idx)
    P = np.zeros((ns, nP), dtype=np.float32)

    for pid, col in pid_to_idx.items():
        v = raw.get(pid)
        if not isinstance(v, np.ndarray):
            continue
        vv = v.astype(np.float32, copy=False)

        if vv.shape[0] < ns:
            reps = int(np.ceil(ns / vv.shape[0]))
            vv = np.tile(vv, reps)[:ns]
        elif vv.shape[0] > ns:
            vv = vv[:ns]

        P[:, col] = vv
    
    return P, ns


def _simulate_chunk(
    start: int,
    end: int,
    lineup_ix: List[List[int]],
    P_chunk: np.ndarray,
    entry_fee: float,
    scaled_payouts: Dict[int, float],
) -> Dict[str, np.ndarray]:
    """Simulate a chunk of Monte Carlo iterations."""

    slice_len = end - start
    M = len(lineup_ix)

    scores = np.zeros((slice_len, M), dtype=np.float64)
    payouts = np.zeros_like(scores)
    rois = np.zeros_like(scores)
    ranks = np.zeros((slice_len, M), dtype=np.int32)

    pay_arr = np.zeros(M, dtype=np.float64)
    for r, a in scaled_payouts.items():
        if 1 <= r <= M:
            pay_arr[r - 1] = float(a)
    pay_csum = np.cumsum(pay_arr, dtype=np.float64)

    atol = 1e-8

    for local_i in range(slice_len):
        row = P_chunk[local_i]

        # Score each lineup
        for m in range(M):
            idxs = lineup_ix[m]
            if idxs:
                scores[local_i, m] = row[idxs].sum(dtype=np.float64)

        # Sort by score desc
        order = np.argsort(-scores[local_i], kind="stable")
        sc_sorted = scores[local_i][order]

        # Find tie groups
        diffs = np.abs(np.diff(sc_sorted))
        edges = np.concatenate(([0], np.nonzero(diffs > atol)[0] + 1, [M]))

        start_rank = edges[:-1] + 1
        end_rank = edges[1:]
        sizes = end_rank - (start_rank - 1)

        # Calculate pool payouts per tie group
        j2 = end_rank - 1
        j1m1 = start_rank - 2
        pools = pay_csum[j2] - np.where(j1m1 >= 0, pay_csum[j1m1], 0.0)

        with np.errstate(invalid='ignore', divide='ignore'):
            per_entry = np.divide(
                pools, sizes,
                out=np.zeros_like(pools, dtype=np.float64),
                where=sizes > 0
            )

        # Assign payouts and ranks
        for g in range(len(start_rank)):
            i0, i1 = edges[g], edges[g + 1]
            idx_in_field = order[i0:i1]
            if idx_in_field.size == 0:
                continue
            payouts[local_i, idx_in_field] = per_entry[g]
            ranks[local_i, idx_in_field] = int(start_rank[g])

        # Calculate ROI
        rois[local_i] = (payouts[local_i] - entry_fee) / entry_fee

    return {
        "start": start,
        "end": end,
        "scores": scores.astype(np.float32, copy=False),
        "payouts": payouts.astype(np.float32, copy=False),
        "rois": rois.astype(np.float32, copy=False),
        "ranks": ranks,
    }


def simulate_one_contest(
    session: Session,
    contest_id: int,
    contest_name: str,
    config_path: str,
    field_cap: int,
    num_sims: int = 1000,
) -> None:
    """
    Run Monte Carlo simulation on a single contest.
    
    Args:
        session: Database session
        contest_id: Contest ID to simulate
        contest_name: Contest name for result storage
        config_path: Path to configuration JSON
        field_cap: Maximum field size to cap at
        num_sims: Number of Monte Carlo simulations
    """
    cfg = json.load(open(config_path, encoding="utf-8"))
    contest = session.query(HistoricalContest).filter_by(contest_id=contest_id).one()
    field_sz = int(contest.total_entries)
    capped_field_sz = min(field_cap, field_sz)
    entry_fee = float(contest.entry_fee)

    contest_params = {
        "entry_fee": float(getattr(contest, "entry_fee", 4.0)),
        "total_entries": int(getattr(contest, "total_entries", 5000)),
        "multi_entry": int(getattr(contest, "multi_entry", 150)),
    }

    counts_by_lineup = sample_field_with_projection_model(
        session,
        field_size=capped_field_sz,
        cfg=cfg,
        contest_params=contest_params
    )
    
    entry_lids, lineup_ix, unique_pids, pid_to_idx, starts, ends = _expand_sampled_field(session, counts_by_lineup)

    M = len(entry_lids)
    scaled = _load_and_scale_payouts_from_db(
        session=session,
        contest_id=contest_id,
        num_field_lineups=M,
        original_field_size=field_sz,
        entry_fee=entry_fee
    )

    P, num_sims = _build_player_samples_matrix(
        session=session,
        pid_to_idx=pid_to_idx,
        n_samples=num_sims
    )

    chunk_size = math.ceil(num_sims / min(12, num_sims))
    full_scores = np.zeros((num_sims, M), dtype=np.float32)
    full_payouts = np.zeros_like(full_scores)
    full_rois = np.zeros_like(full_scores)
    full_ranks = np.zeros((num_sims, M), dtype=int)

    with ProcessPoolExecutor(max_workers=8) as exe:
        futures = []
        for start in range(0, num_sims, chunk_size):
            end = min(start + chunk_size, num_sims)
            futures.append(
                exe.submit(
                    _simulate_chunk,
                    start,
                    end,
                    lineup_ix,
                    P[start:end],
                    float(entry_fee),
                    scaled
                )
            )
        for fut in as_completed(futures):
            part = fut.result()
            s, e = part["start"], part["end"]
            full_scores[s:e] = part["scores"]
            full_payouts[s:e] = part["payouts"]
            full_rois[s:e] = part["rois"]
            full_ranks[s:e] = part["ranks"]
    
    u_lids = sorted([lid for lid, c in counts_by_lineup.items() if c > 0])

    for u, lid in enumerate(u_lids):
        a, b = int(starts[u]), int(ends[u])
        session.add(LineupSimulationResult(
            lineup_id=int(lid),
            contest_id=contest_id,
            contest_name=contest_name,
            fpts=full_scores[:, a:b].mean(axis=1).tolist(),
            payout=full_payouts[:, a:b].sum(axis=1).tolist(),
            roi=full_rois[:, a:b].mean(axis=1).tolist(),
            rank=full_ranks[:, a:b].mean(axis=1).tolist(),
        ))

    session.commit()
    print(f"✓ Simulated (NBA): {num_sims} sims × {M} entries | {len(u_lids)} unique lineups for contest {contest_id}")


def choose_lineups_for_contest(
    session: Session,
    contest_id: int,
    num_to_choose: int,
    weights: Optional[Dict[str, float]] = None,
    console: Optional[Console] = None,
) -> List[int]:
    """
    Select optimal diverse lineups for a single contest using greedy diversity-aware selection.
    
    The algorithm balances:
    - Expected ROI / cash rate (quality)
    - Lineup diversity (measured by cosine distance in FPTS space)
    
    Args:
        session: Database session
        contest_id: Contest ID to choose lineups for
        num_to_choose: Number of lineups to select
        weights: Dict with keys: cash_pct, avg_roi_win, diversity_lambda
        console: Rich console for output (optional)
        
    Returns:
        List of selected lineup IDs in selection order
    """
    if console is None:
        console = Console()
    
    if weights is None:
        weights = {
            "cash_pct": 0.0,
            "avg_roi_win": 1.0,
            "diversity_lambda": 0.5,
        }
    
    # Get simulation results for this contest
    sims = (
        session.query(LineupSimulationResult)
        .filter(LineupSimulationResult.contest_id == contest_id)
        .all()
    )
    
    if not sims:
        console.print(f"[red]No simulation results for contest {contest_id}[/red]")
        return []
    
    console.print(f"[dim]Found {len(sims)} simulation results[/dim]")
    
    # Filter to positive ROI if large result set
    if len(sims) > 5000:
        sims = [
            res for res in sims
            if np.array(res.roi, dtype=float).mean() >= 0.0
        ]
        console.print(f"[yellow]Filtered to {len(sims)} lineups with non-negative ROI[/yellow]")
    
    if not sims:
        return []
    
    # Filter out lineups with missing players
    all_pids = {pid for res in sims for pid in res.lineup.player_ids_list}
    existing = {
        pid for (pid,) in
        session.query(NBAPlayer.id).filter(NBAPlayer.id.in_(all_pids)).all()
    }
    
    filtered = []
    for res in sims:
        missing = set(res.lineup.player_ids_list) - existing
        if not missing:
            filtered.append(res)
    sims = filtered
    
    if not sims:
        console.print(f"[red]No valid lineups after dropping missing players[/red]")
        return []
    
    # Remove duplicates
    unique = []
    seen_comps = set()
    for res in sims:
        comp = tuple(sorted(res.lineup.player_ids_list))
        if comp not in seen_comps:
            seen_comps.add(comp)
            unique.append(res)
    sims = unique
    
    if not sims:
        console.print(f"[red]No unique lineups remain[/red]")
        return []
    
    N = len(sims)
    console.print(f"[cyan]Choosing {num_to_choose} from {N} unique lineups[/cyan]")
    
    # Build feature matrix: [cash%, avg_roi]
    raw = np.zeros((N, 2), dtype=float)
    for i, res in enumerate(sims):
        payouts = np.array(res.payout, dtype=float)
        rois = np.array(res.roi, dtype=float)
        raw[i, 0] = (payouts > 0).mean() if payouts.size else 0.0  # cash%
        raw[i, 1] = rois.mean()  # avg ROI
    
    # Build FPTS matrix for diversity calculation
    fpts_rows = [np.array(res.fpts, dtype=float) for res in sims]
    fpts_matrix = np.vstack(fpts_rows)
    
    # Normalize features to [0, 1]
    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    norm = (raw - mins) / (maxs - mins + 1e-6)
    
    # Blend into single quality score q
    w_feat = np.array([weights.get("cash_pct", 0), weights.get("avg_roi_win", 1)], dtype=float)
    q = (norm * w_feat).sum(axis=1)
    
    # Build diversity matrix using cosine distance
    dist_mat = squareform(pdist(fpts_matrix, metric="cosine"))
    
    lam = float(weights.get("diversity_lambda", 0.5))
    
    # Greedy diversity-aware selection
    remaining = set(range(N))
    selected = []
    
    # Start with the highest quality lineup
    first = int(np.argmax(q))
    selected.append(first)
    remaining.remove(first)
    
    def lineup_score(j: int, S: list) -> float:
        """Score a lineup based on quality + diversity from already selected."""
        if S:
            diversity_gain = np.mean(dist_mat[j, S])
        else:
            diversity_gain = 0.0
        return q[j] + lam * diversity_gain
    
    # Greedily select remaining lineups
    while len(selected) < num_to_choose and remaining:
        best_j = max(remaining, key=lambda j: lineup_score(j, selected))
        selected.append(best_j)
        remaining.remove(best_j)
    
    # Fallback: if not enough, fill with top remaining by quality
    if len(selected) < num_to_choose:
        console.print(f'[yellow]Not enough lineups ({len(selected)}), filling with top remaining[/yellow]')
        for j in np.argsort(-q):
            if j not in selected:
                selected.append(j)
            if len(selected) >= num_to_choose:
                break
    
    # Return lineup IDs in selection order
    chosen_ids = [sims[i].lineup_id for i in selected]
    
    return chosen_ids

