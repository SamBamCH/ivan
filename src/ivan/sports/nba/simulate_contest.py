import json
import math
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sqlalchemy.orm import Session
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import pickle

try:
    import plotext as plt
    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False

from ivan.sports.nba.models import HistoricalContest, ContestPayout, Lineup, LineupSimulationResult, NBAPlayer
from ivan.sports.nba.sample_field import sample_field_with_projection_model
from ivan.sports.nba.generate_samples import generate_player_samples
# Reuse your MLB payout scaler (sport-agnostic)
from ivan.sports.mlb.payouts import scale_payouts_correctly

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
    return scale_payouts_correctly(
        num_field_lineups=num_field_lineups,
        payouts_df=payouts_df,
        original_field_size=original_field_size,
        entry_fee=entry_fee
    )

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

def _expand_sampled_field(
        session: Session,
        counts_by_lineup: Dict[int, int],
) -> Tuple[List[int], List[List[int]], List[int], Dict[int, int], np.ndarray, np.ndarray]:
    
    if not counts_by_lineup:
        print('no counts_by_lineup returned from sampling algorithm.')
        return [], [], [], {}, np.array([], dtype=int), np.array([], dtype=int)

    u_lids = sorted([lid for lid, c in counts_by_lineup.items() if c > 0])
    cnts   = np.array([int(counts_by_lineup[lid]) for lid in u_lids], dtype=np.int64)
    starts = np.cumsum(np.r_[0, cnts[:-1]])
    ends   = starts + cnts
    M      = int(cnts.sum())

    # FIXED: Fetch lineups in batches to avoid SQLite variable limit
    BATCH_SIZE = 900  # Stay safely under SQLite's 999 limit
    pid_lists_by_lid = {}
    
    for i in range(0, len(u_lids), BATCH_SIZE):
        batch = u_lids[i:i + BATCH_SIZE]
        rows = (session.query(Lineup.id, Lineup.player_ids_list)
                      .filter(Lineup.id.in_(batch))
                      .all())
        for lid, j in rows:
            pid_lists_by_lid[int(lid)] = _coerce_player_ids(j)

    # Rest of the function remains the same...
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

def _build_player_samples_matrix(
    session: Session,
    pid_to_idx: Dict[int, int],
    n_samples: int = 1000,
) -> np.ndarray:
    
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
            vv - vv[:ns]

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


def plot_field_sampling_diagnostics(session, counts_by_lineup, contest_params, model_path="data/nba/models/field_distribution_model.pkl"):
    """
    Plot the model's predicted projection distribution vs. the empirical distribution from the sampled field.
    Prints in terminal using plotext if available, otherwise prints text summary.
    Also prints player exposures.
    """
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler = model['scaler']
    proj_model = model['proj_model']
    bin_edges = model['bin_edges']
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    BINS = len(bin_centers)

    # Predict model distribution for this contest
    X = np.array([[contest_params['entry_fee'], contest_params['total_entries'], contest_params['multi_entry']]])
    X_scaled = scaler.transform(X)
    proj_dist = proj_model.predict(X_scaled)[0]

    # Gather all sampled lineups and their projection sums
    lineup_ids = list(counts_by_lineup.keys())
    lineup_counts = list(counts_by_lineup.values())
    all_lineups = session.query(Lineup).filter(Lineup.id.in_(lineup_ids)).all()
    id_to_lineup = {l.id: l for l in all_lineups}
    lineup_proj = []
    for lid in lineup_ids:
        l = id_to_lineup[lid]
        pids = l.player_ids_list
        if isinstance(pids, str):
            import json
            try:
                pids = json.loads(pids)
            except:
                pids = []
        pids = [int(x) for x in (pids or []) if x is not None]
        players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(pids)).all()
        player_dict = {p.id: p for p in players}
        total_proj = sum(player_dict[pid].fpts_mean for pid in pids if pid in player_dict and player_dict[pid].fpts_mean is not None)
        # Add the projection sum for each time this lineup was sampled
        lineup_proj.extend([total_proj] * counts_by_lineup[lid])

    lineup_proj = np.array(lineup_proj)
    
    # Load projection_min from model
    projection_min = model.get('projection_min', 200.0)
    
    # Filter lineups below projection_min (matching sampling logic)
    valid_mask = lineup_proj >= projection_min
    lineup_proj = lineup_proj[valid_mask]
    
    # Normalize lineup projections to [0, 1] using projection_min as min and max as max
    if len(lineup_proj) > 0:
        max_proj = lineup_proj.max()
        if max_proj > projection_min:
            norm_proj = (lineup_proj - projection_min) / (max_proj - projection_min)
        else:
            norm_proj = np.zeros_like(lineup_proj)
        
        # Empirical histogram (normalized to sum to 1 for comparison)
        hist, _ = np.histogram(norm_proj, bins=bin_edges, density=False)
        hist = hist / hist.sum() if hist.sum() > 0 else hist

        # Map bin centers to projection space for plotting on the new range
        proj_bin_centers = projection_min + bin_centers * max(1e-6, (max_proj - projection_min))

        # Print terminal plot using projection-space x-axis to "shrink" to the active range
        print("\n" + "="*70)
        print("Field Sampling Diagnostics: Model vs Sampled Distribution")
        print("="*70)
        
        if HAS_PLOTEXT:
            plt.clf()
            plt.bar(proj_bin_centers.tolist(), proj_dist.tolist(), label='Model Prediction', color='green')
            plt.bar(proj_bin_centers.tolist(), hist.tolist(), label='Sampled Field', color='blue', fill=False)
            plt.title(f'Projection Distribution (sum): {projection_min:.0f}–{max_proj:.0f}')
            plt.xlabel('Lineup Projection Sum')
            plt.ylabel('Probability Density')
            plt.plotsize(100, 25)
            # limit x-axis to the active projection range for better detail
            try:
                plt.xlim(projection_min, max_proj)
            except Exception:
                pass
            plt.show()
        else:
            print("\n(Install plotext for terminal graphs: pip install plotext)")
            m_idx = int(np.argmax(proj_dist)) if proj_dist.size else 0
            s_idx = int(np.argmax(hist)) if hist.size else 0
            m_x = proj_bin_centers[m_idx] if m_idx < len(proj_bin_centers) else float('nan')
            s_x = proj_bin_centers[s_idx] if s_idx < len(proj_bin_centers) else float('nan')
            print(f"Range: [{projection_min:.1f}, {max_proj:.1f}] (projection sums)")
            print(f"Model peak at {m_x:.1f}, max prob: {proj_dist.max():.3f}")
            print(f"Sampled peak at {s_x:.1f}, max prob: {hist.max():.3f}")

    # --- Player exposures ---
    player_counter = Counter()
    for lid in lineup_ids:
        l = id_to_lineup[lid]
        pids = l.player_ids_list
        if isinstance(pids, str):
            import json
            try:
                pids = json.loads(pids)
            except:
                pids = []
        pids = [int(x) for x in (pids or []) if x is not None]
        for pid in pids:
            player_counter[pid] += counts_by_lineup[lid]

    # Print top 20 player exposures
    print("\n" + "="*70)
    print("Top 20 Player Exposures in Sampled Field")
    print("="*70)
    if player_counter:
        player_ids = list(player_counter.keys())
        players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(player_ids)).all()
        id_to_name = {p.id: p.name for p in players}
        # Total field size = sum of all lineup counts (accounts for duplicates)
        total_field_size = sum(counts_by_lineup.values())
        top = player_counter.most_common(20)
        for pid, count in top:
            pct = 100.0 * count / total_field_size
            name = id_to_name.get(pid, str(pid))
            print(f"  {name:30s}  {count:6d}  ({pct:5.2f}%)")
    else:
        print("  (No player exposures found)")
    print("="*70 + "\n")


def simulate_one_contest(
    session: Session,
    contest_id: int,
    contest_name: str,
    config_path: str,
    field_cap: int,
    num_sims: int = 1000
):
    cfg         = json.load(open(config_path, encoding="utf-8"))
    contest     = session.query(HistoricalContest).filter_by(contest_id=contest_id).one()
    field_sz    = int(contest.total_entries)
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
    
    # Plot field sampling diagnostics in terminal
    plot_field_sampling_diagnostics(session, counts_by_lineup, contest_params)
    
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

    chunk_size      = math.ceil(num_sims / min(12, num_sims)) #TODO: pull out 'jobs' to config for smaller machines? 
    full_scores     = np.zeros((num_sims, M), dtype=np.float32)
    full_payouts    = np.zeros_like(full_scores)
    full_rois       = np.zeros_like(full_scores)
    full_ranks      = np.zeros((num_sims, M), dtype=int)

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
            full_scores[s:e]    = part["scores"]
            full_payouts[s:e]   = part["payouts"]
            full_rois[s:e]      = part["rois"]
            full_ranks[s:e]     = part["ranks"]
    
    u_lids = sorted([lid for lid, c in counts_by_lineup.items() if c > 0])

    for u, lid in enumerate(u_lids):
        a, b = int(starts[u]), int(ends[u])
        session.add(LineupSimulationResult(
            lineup_id       = int(lid),
            contest_id      = contest_id,
            contest_name    = contest_name,
            fpts         = full_scores[:, a:b].mean(axis=1).tolist(),
            payout       = full_payouts[:, a:b].sum(axis=1).tolist(),   # total payout over dupes
            roi          = full_rois[:, a:b].mean(axis=1).tolist(),     # mean ROI per entry
            rank         = full_ranks[:, a:b].mean(axis=1).tolist(),    # mean rank across dupes
        ))

    session.commit()
    print(f"Done (NBA): {num_sims} sims × {M} entries | unique lineups={len(u_lids)} for contest {contest_id}")
