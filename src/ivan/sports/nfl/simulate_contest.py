from __future__ import annotations
import json
import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Dict, Tuple, List
from sqlalchemy.orm import Session
from concurrent.futures import ProcessPoolExecutor, as_completed


from ivan.sports.nfl.models import HistoricalContest, ContestPayout, Lineup, LineupSimulationResult
from ivan.sports.nfl.sample_field import (
    sample_sd_field,
    build_lineup_matrix,
    build_player_samples,
)
from ivan.sports.nfl.correlated_samples import generate_correlated_samples, SamplingConfig
# Reuse your MLB payout scaler (sport-agnostic)
from ivan.sports.mlb.payouts import scale_payouts_correctly

def _coerce_player_ids(val) -> List[int]:
    import json as _json
    if val is None:
        return []
    if isinstance(val, list):
        return [int(x) for x in val]
    if isinstance(val, str):
        try:
            data = _json.loads(val)
            if isinstance(data, list):
                return [int(x) for x in data]
        except Exception:
            return []
    return []

def _expand_sampled_field(
        session: Session,
        counts_by_lineup: Dict[int, int],
) -> Tuple[List[int], List[List[int]], List[int], Dict[int, int], np.ndarray, np.ndarray]:
    """
    Turn sampled counts (with replacement) into: 
      - entry_lineup_ids:   one per entry (length M)
      - lineup_ix:          list of per-entry player-index lists (using pid_to_idx)
      - unique_pids:        compact list of player IDs used across the sampled field
      - pid_to_idx:         map pid -> [0..nP-1]
      - starts, ends:       per-unique-lineup slices in the expanded entries (for aggregations)
    """

    if not counts_by_lineup:
        print('no counts_by_lineup returned from sampling algorithm.')
        return [], [], [], {}, np.array([], dtype=int), np.array([], dtype=int)

    u_lids = sorted([lid for lid, c in counts_by_lineup.items() if c > 0])
    cnts   = np.array([int(counts_by_lineup[lid]) for lid in u_lids], dtype=np.int64)
    starts = np.cumsum(np.r_[0, cnts[:-1]])
    ends   = starts + cnts
    M      = int(cnts.sum())

    # fetch lineups

    rows = (session.query(Lineup.id, Lineup.player_ids_list)
                  .filter(Lineup.id.in_(u_lids))
                  .all())
    
    pid_lists_by_lid = {int(lid): _coerce_player_ids(j) for (lid, j) in rows}

    all_pids = set()
    for lid in u_lids:
        all_pids.update(pid_lists_by_lid.get(lid, []))
    unique_pids = sorted(all_pids)
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

    entry_lineup_ids: List[int]     = [0] * M
    lineup_ix: List[List[int]]      = [None] * M
    for u, lid in enumerate(u_lids): 
        pids = pid_lists_by_lid.get(lid, [])
        idxs = [pid_to_idx[p] for p in pids if p in pid_to_idx]
        a, b = int(starts[u]), int(ends[u])
        for pos in range(a, b): 
            entry_lineup_ids[pos] = lid
            lineup_ix[pos] = idxs

    return entry_lineup_ids, lineup_ix, unique_pids, pid_to_idx, starts, ends  

def _load_and_scale_payouts_from_db(
        session: Session,
        contest_id: int, 
        num_field_lineups: int, 
        original_field_size: int,
        entry_fee: float,
) -> Dict[int, float]: 
    """
    scale payouts to desired contest size
    """
    rows = (session.query(ContestPayout)
                .filter(ContestPayout.contest_id == contest_id)
                .order_by(ContestPayout.min_position.asc())
                .all())
    
    if not rows:
        raise ValueError(f"No payouts found for contest_id={contest_id}")
    
    payouts_df = pd.DataFrame(
        [(r.min_position, r.max_position, r.payout_amount) for r in rows],
        columns=["Start Rank", "End Rank", "Payout"]
    )

    print(f'original payouts_df = {payouts_df}')

    print(f'scaling parameters: num_field_lineups={num_field_lineups}, original_field_size={original_field_size}, entry_fee={entry_fee}')
    return scale_payouts_correctly(
        num_field_lineups=num_field_lineups,
        payouts_df=payouts_df,
        original_field_size=original_field_size,
        entry_fee=entry_fee
    )

def _build_player_samples_matrix(
        session: Session,
        pid_to_idx: Dict[int, int],
        sampling_cfg: SamplingConfig | None = None,
        model_dir: str | Path = "data/nfl/models"
) -> np.ndarray:
    """
    NFL scoring matrix P: [num_sims x num_players] aligned to 'pid_to_idx'.
    """
    cfg = sampling_cfg or SamplingConfig(n_samples=100)
    raw = generate_correlated_samples(session, model_dir=model_dir, cfg=cfg)
    use = int(cfg.n_samples)
    nP  = len(pid_to_idx)
    P   = np.zeros((use, nP), dtype=np.float32)

    for pid, col in pid_to_idx.items():
        v = raw.get(pid)
        if not isinstance(v, np.ndarray):
            continue
        
        if v.shape[0] != use:
            if v.shape[0] > use:
                v = v[:use]

            else:
                reps = int(np.ceil(use / v.shape[0]))
                v = np.tile(v, reps)[:use]
            
        P[:, col] = v.astype(np.float32, copy=False)

    return P

def _simulate_chunk(
    start: int, 
    end: int,
    lineup_ix: List[List[int]],
    P_chunk: np.ndarray,
    entry_fee: float,
    scaled_payouts: Dict[int, float],
) -> Dict[str, np.ndarray]:

    slice_len = end - start
    M = len(lineup_ix)

    # work in float64 for stability at scale
    scores  = np.zeros((slice_len, M), dtype=np.float64)
    payouts = np.zeros_like(scores)
    rois    = np.zeros_like(scores)
    ranks   = np.zeros((slice_len, M), dtype=np.int32)

    # zero-based payout array: index 0 -> rank 1
    pay_arr = np.zeros(M, dtype=np.float64)
    for r, a in scaled_payouts.items():
        if 1 <= r <= M:
            pay_arr[r - 1] = float(a)
    pay_csum = np.cumsum(pay_arr, dtype=np.float64)

    atol = 1e-8  # tighter since float64

    for local_i in range(slice_len):
        row = P_chunk[local_i]  # [nP]

        # 1) score each entry for THIS sim
        for m in range(M):
            idxs = lineup_ix[m]
            if idxs:  # list truthiness is fine
                scores[local_i, m] = row[idxs].sum(dtype=np.float64)

        # 2) sort by score desc
        order = np.argsort(-scores[local_i], kind="stable")
        sc_sorted = scores[local_i][order]

        # 3) tie groups via explicit boundaries
        diffs = np.abs(np.diff(sc_sorted))
        edges = np.concatenate(([0], np.nonzero(diffs > atol)[0] + 1, [M]))

        start_rank = edges[:-1] + 1                  # 1-indexed
        end_rank   = edges[1:]                       # 1-indexed
        sizes      = end_rank - (start_rank - 1)     # == edges[1:] - edges[:-1]

        # --- FIXED: vectorized pool sums with prefix sums (no array "if") ---
        j2   = end_rank - 1                          # 0-based inclusive end
        j1m1 = start_rank - 2                        # one before start (can be -1)
        pools = pay_csum[j2] - np.where(j1m1 >= 0, pay_csum[j1m1], 0.0)

        with np.errstate(invalid='ignore', divide='ignore'):
            per_entry = np.divide(
                pools, sizes,
                out=np.zeros_like(pools, dtype=np.float64),
                where=sizes > 0
            )

        # write per group
        for g in range(len(start_rank)):
            i0, i1 = edges[g], edges[g+1]
            idx_in_field = order[i0:i1]
            if idx_in_field.size == 0:
                continue
            payouts[local_i, idx_in_field] = per_entry[g]
            ranks[local_i, idx_in_field] = int(start_rank[g])

        # 5) ROI
        rois[local_i] = (payouts[local_i] - entry_fee) / entry_fee

        # ======= invariants / anomaly prints =======
        total_payout_this_sim = float(payouts[local_i].sum())
        expected_pool = float(pay_arr.sum())
        if not np.isfinite(total_payout_this_sim) or abs(total_payout_this_sim - expected_pool) > 1e-4:
            print(f"[WARN sim {start + local_i}] payout conservation: got={total_payout_this_sim:.6f}, "
                  f"expected={expected_pool:.6f}")

        for g in range(len(start_rank)):
            r1, r2, sz = int(start_rank[g]), int(end_rank[g]), int(sizes[g])
            if sz <= 0:
                print(f"[WARN sim {start + local_i}] non-positive group size at ranks {r1}..{r2}")
                continue
            sl = pay_arr[r1-1:r2]
            if sl.size == 0:
                print(f"[WARN sim {start + local_i}] empty pay slice {r1}..{r2} (size={sz})")
                continue
            pe = float(per_entry[g])
            lo, hi = float(sl.min()), float(sl.max())
            if (pe < lo - 1e-8 or pe > hi + 1e-8) and (lo != hi):
                print(f"[WARN sim {start + local_i}] per-entry {pe:.6f} outside [{lo:.6f},{hi:.6f}] "
                      f"for tie ranks {r1}..{r2} (size={sz})")

        mx = float(payouts[local_i].max())
        top = float(pay_arr[0]) if M else 0.0
        if mx > top + 1e-8:
            print(f"[WARN sim {start + local_i}] max payout {mx:.6f} > top prize {top:.6f}")

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
    num_sims: int,
    config_path: str,
    field_cap: int
) -> None:
    # 1) contest info
    cfg      = json.load(open(config_path, encoding="utf-8"))
    contest  = session.query(HistoricalContest).filter_by(contest_id=contest_id).one()
    field_sz = int(contest.total_entries)
    capped_field_sz = min(field_cap, field_sz)
    entry_fee = float(contest.entry_fee)

    # 1) sample the field (WITH replacement). This returns counts per lineup_id.
    counts_by_lineup, meta = sample_sd_field(
        session,
        field_size=capped_field_sz,
        n_sims=min(100, num_sims)
    )
    total = sum(counts_by_lineup.values())
    assert total == capped_field_sz, (total, capped_field_sz)


    #TODO: add a quick sanity check and print the top duped lineups in the sample along with their sample chacteristics (i.e. salary remaining, proj/max([projs]))
    # Expand counts into entry-level arrays + compact player universe
    entry_lids, lineup_ix, unique_pids, pid_to_idx, starts, ends = _expand_sampled_field(session, counts_by_lineup)

    M = len(entry_lids)
    if M == 0:
        print(f"[{contest_id}] no sampled entries; nothing to simulate.")

    print(f'M = {M}')
    
    # 2) payouts scaled to M (simulated field size)
    scaled = _load_and_scale_payouts_from_db(
        session=session,
        contest_id=contest_id,
        num_field_lineups=M,
        original_field_size=field_sz,   # scale from original → simulated
        entry_fee=entry_fee,
    )

    # --- debug/inspection: top payouts + last paid position ---
    paid_items = [(rank, amt) for rank, amt in scaled.items() if amt > 0]
    top10 = sorted(paid_items, key=lambda x: (-x[1], x[0]))[:10]

    print("\nTop 10 payouts (rank → payout):")
    for r, a in top10:
        print(f"  {r:>6} → ${a:,.2f}")

    # last position that pays (max rank with payout > 0)
    last_paid = max((r for r, a in scaled.items() if a > 0), default=None)
    if last_paid is None:
        print("No paid positions (all payouts are $0).")
    else:
        print(f"Last paid position: {last_paid}  (pays ${scaled[last_paid]:,.2f})")

    # (optional) how many places are paid and what percent of the field that is
    paid_count = sum(1 for r in range(1, M + 1) if scaled.get(r, 0.0) > 0.0)
    print(f"Paid places: {paid_count}/{M} ({paid_count / max(1, M):.1%})\n")

    sim_cfg = SamplingConfig(
        n_samples   = num_sims,
        alpha       = 0.85,
        jitter      = 0.05     
    )
    # 3) correlated NFL projections for the compact player set

    P = _build_player_samples_matrix(
        session=session,
        pid_to_idx=pid_to_idx,
        sampling_cfg=sim_cfg
    )

    # 4) run simulations in parallel (same “chunk” pattern as MLB)
    chunk_size   = math.ceil(num_sims / min(8, num_sims))   # mirror MLB limiter
    full_scores  = np.zeros((num_sims, M), dtype=np.float32)
    full_payouts = np.zeros_like(full_scores)
    full_rois    = np.zeros_like(full_scores)
    full_ranks   = np.zeros((num_sims, M), dtype=int)

    with ProcessPoolExecutor(max_workers=1) as exe:
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
            full_scores[s:e]  = part["scores"]
            full_payouts[s:e] = part["payouts"]
            full_rois[s:e]    = part["rois"]
            full_ranks[s:e]   = part["ranks"]

    
    # 5) persist per-UNIQUE-lineup aggregates (dupe-aware)
    #    We aggregate across an individual lineup’s duplicate entries using starts/ends.
    u_lids = sorted([lid for lid, c in counts_by_lineup.items() if c > 0])
    counts_U = np.array([int(counts_by_lineup[l]) for l in u_lids], dtype=np.int64)

    for u, lid in enumerate(u_lids):
        a, b = int(starts[u]), int(ends[u])
        session.add(LineupSimulationResult(
            lineup_id    = int(lid),
            contest_id   = contest_id,
            contest_name = contest_name,
            fpts         = full_scores[:, a:b].mean(axis=1).tolist(),
            payout       = full_payouts[:, a:b].sum(axis=1).tolist(),   # total payout over dupes
            roi          = full_rois[:, a:b].mean(axis=1).tolist(),     # mean ROI per entry
            rank         = full_ranks[:, a:b].mean(axis=1).tolist(),    # mean rank across dupes
        ))

    session.commit()
    print(f"Done (NFL): {num_sims} sims × {M} entries | unique lineups={len(u_lids)} for contest {contest_id}")
        



