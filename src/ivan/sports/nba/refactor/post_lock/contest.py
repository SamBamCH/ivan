from sqlalchemy.orm import Session
from typing import List, Tuple, Dict
import logging

from ivan.sports.nba.refactor.post_lock.entry_parsing import OriginalEntry
import numpy as np
from ivan.sports.nba.models import NBAPlayer, HistoricalContest, ContestPayout
from ivan.sports.nba.simulate_contest import _load_and_scale_payouts_from_db as _scale_payouts
from ivan.sports.nba.refactor.pre_lock.lineup_generation import FieldPlayer
from ivan.sports.nba.utils import _coerce_positions
from ivan.sports.common.draftkings import download_contest_dataframe
from ivan.sports.nba.refactor.post_lock.lineup import finish_lineup, generate_candidate
from ivan.sports.nba.refactor.pre_lock.fpts_samples import generate_player_samples

WORKERS = 8

def _sample_field_lineups(
    session: Session,
    contest_id: int,
    http_sess,
    name_map,
    num_field_samples: int = 10000,
) -> Tuple[List[List[int]], Dict[int, FieldPlayer], Dict[int, float]]: 
    
    all_lineups: List[List[int]] = []
    seen: set[tuple[int, ...]] = set()
    lineups_completed = 0
    lineups_attempted = 0

    raw_players: List[NBAPlayer] = session.query(NBAPlayer).all()
    player_pool: List[FieldPlayer] = []
    base_scores: Dict[int, float] = {}

    #wrap all players into FieldPlayer
    for rp in raw_players:
        pos = getattr(rp, "position", None)
        if pos is None:
            logging.error(f'no position found for {rp.name}')
            continue
        pos_tuple = _coerce_positions(pos)

        if rp.team and rp.opponent:
            game_id = "-".join(sorted([rp.team.upper(), rp.opponent.upper()]))
        else:
            game_id = None

        player_pool.append(FieldPlayer(
            id=rp.id,
            name=rp.name,
            salary=rp.salary,
            team=rp.team,
            opponent=rp.opponent,
            position=pos_tuple,
            fpts_mean=rp.fpts_mean,
            game_id=game_id,
            gametime=rp.gametime
        ))
        
    id_to_fp: Dict[int, FieldPlayer] = {fp.id: fp for fp in player_pool}
    # Precompute one sample per player (randomness=1.0) for efficiency
    raw_samples = generate_player_samples(session=session, n_samples=1, randomness=1.0)
    for fp in player_pool:
        val = raw_samples.get(fp.id)
        score: float
        if val is None:
            score = float(getattr(fp, 'fpts_mean', 0.0) or 0.0)
        else:
            try:
                if hasattr(val, '__len__') and len(val) > 0:
                    score = float(val[0])
                else:
                    score = float(val)
            except Exception:
                score = float(getattr(fp, 'fpts_mean', 0.0) or 0.0)
        base_scores[fp.id] = score

    #fetch the entire payload from draftkings endpoint
    df = download_contest_dataframe(sess=http_sess, contest_id=int(contest_id))
    if df is None or df.empty:
        print('no data returned from draftkings')
        return

    cols = [str(c).strip() for c in df.columns]
    lineup_col = next((c for c in cols if c.lower() == 'lineup'), None)
    if not lineup_col:
        print('no lineup col found in dataset from DK')
        return
    
    tmp = df[[lineup_col]].dropna().drop_duplicates() #TODO: should we be dropping duplicates? How are we going to persist dupes from real world to sim?

    #filter down to num_field_samples by randomely selecting rows (maybe remove botton n% of lineups first?)
    if len(tmp) > int(num_field_samples):
         tmp = tmp.sample(n=int(num_field_samples))

    completed: List[List[int]] = []
        
    #for each lineup:
    for _, row in tmp.iterrows():
        lineup_str = str(row[lineup_col])
        completed_lineup = finish_lineup(session, lineup_str, name_map=name_map, id_to_fp=id_to_fp, base_scores=base_scores)
        if completed_lineup:
            completed.append(completed_lineup)

    return completed, id_to_fp, base_scores

def lateswap_one_contest(
    session: Session,
    contest_id: int,
    entries: List['OriginalEntry'],
    http_sess,
    name_map,
    num_candidates_per_entry: int = 10,
    num_field_samples: int = 10000,
    num_sims: int = 1000,
) -> Tuple[List[Tuple['OriginalEntry', List[int], float, Dict]], List[Tuple[int, int, int, List[int], float, Dict]]]:
    """Generate late-swap candidates for each entry, simulate against sampled field,
    assign payouts and choose best candidate per entry.

    Returns:
        best_per_entry: List of (OriginalEntry, best_candidate_lineup_player_ids, expected_payout, id_to_fp)
        all_candidate_stats: List of (entry_id, contest_id, candidate_index, lineup_player_ids, expected_payout, id_to_fp)
    """

    # 1. Sample real field lineups from DK
    sampled = _sample_field_lineups(
        session=session,
        contest_id=contest_id,
        http_sess=http_sess,
        name_map=name_map,
        num_field_samples=num_field_samples,
    )
    if not sampled:
        return [], []
    field_lineups, id_to_fp, base_scores = sampled

    # 2. Build candidate lineups per entry
    entry_to_candidates: Dict[int, List[List[int]]] = {}
    for entry in entries:
        candidates: List[List[int]] = []
        for _ in range(num_candidates_per_entry):
            try:
                cand = generate_candidate(session, entry, id_to_fp, base_scores)
                if cand and len(cand) == 8:  # basic sanity
                    candidates.append(cand)
            except Exception as e:
                logging.error(f"candidate generation failed for entry {entry.entry_id}: {e}")
                continue
        # Always include the original lineup as baseline candidate 0
        if entry.player_ids and len(entry.player_ids) == 8:
            candidates.insert(0, entry.player_ids)
        entry_to_candidates[entry.entry_id] = candidates

    # If no candidates generated for any entry, early exit
    if not any(entry_to_candidates.values()):
        return [], []

    # 3. Aggregate all lineups (field + candidates) and metadata
    all_lineups: List[List[int]] = []
    lineup_meta: List[Tuple[int, int]] = []  # (entry_id, cand_idx) or (-1,-1) for field
    for fl in field_lineups:
        if fl and len(fl) == 8:
            all_lineups.append(fl)
            lineup_meta.append((-1, -1))
    for entry in entries:
        cands = entry_to_candidates.get(entry.entry_id, [])
        for ci, cand in enumerate(cands):
            all_lineups.append(cand)
            lineup_meta.append((entry.entry_id, ci))
    if not all_lineups:
        return [], []

    # 4. Build player sample matrix reusing approach from simulate_contest
    from ivan.sports.nba.refactor.pre_lock.fpts_samples import generate_player_samples
    unique_pids = sorted({pid for ln in all_lineups for pid in ln})
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    raw_samples = generate_player_samples(session=session, n_samples=num_sims, randomness=1.0)
    n_players = len(unique_pids)
    P = np.zeros((num_sims, n_players), dtype=np.float32)
    for pid, col in pid_to_idx.items():
        arr = raw_samples.get(pid)
        if isinstance(arr, np.ndarray) and arr.shape[0] == num_sims:
            P[:, col] = arr.astype(np.float32, copy=False)
        else:
            mean = id_to_fp.get(pid).fpts_mean if pid in id_to_fp else 0.0
            P[:, col] = mean

    # 5. Build lineup index list (list of player index lists)
    lineup_ix: List[List[int]] = [[pid_to_idx[p] for p in ln if p in pid_to_idx] for ln in all_lineups]

    # 6. Load contest + scaled payouts using existing scaler
    contest = session.query(HistoricalContest).filter(HistoricalContest.contest_id == contest_id).first()
    if contest is None:
        total_entries = len(field_lineups)
        entry_fee = 0.0
    else:
        total_entries = int(getattr(contest, 'total_entries', len(field_lineups)) or len(field_lineups))
        entry_fee = float(getattr(contest, 'entry_fee', 0.0) or 0.0)
    # scaled payouts sized to current simulated field (field + candidates)
    M = len(all_lineups)
    scaled_payouts = {}
    try:
        scaled_payouts = _scale_payouts(
            session=session,
            contest_id=contest_id,
            num_field_lineups=M,
            original_field_size=total_entries,
            entry_fee=entry_fee
        )
    except Exception as e:
        logging.error(f"payout scaling failed, defaulting to zeros: {e}")
        scaled_payouts = {}

    # 7. Simulate using tie/pool logic from _simulate_chunk (single-process)
    scores = np.zeros((num_sims, M), dtype=np.float64)
    payouts = np.zeros_like(scores)
    ranks = np.zeros((num_sims, M), dtype=np.int32)
    pay_arr = np.zeros(M, dtype=np.float64)
    for r, amt in scaled_payouts.items():
        if 1 <= r <= M:
            pay_arr[r-1] = float(amt)
    pay_csum = np.cumsum(pay_arr, dtype=np.float64)
    atol = 1e-8
    for sim_i in range(num_sims):
        row = P[sim_i]
        for m in range(M):
            idxs = lineup_ix[m]
            if idxs:
                scores[sim_i, m] = row[idxs].sum(dtype=np.float64)
        order = np.argsort(-scores[sim_i], kind='stable')
        sc_sorted = scores[sim_i][order]
        diffs = np.abs(np.diff(sc_sorted))
        edges = np.concatenate(([0], np.nonzero(diffs > atol)[0] + 1, [M]))
        start_rank = edges[:-1] + 1
        end_rank = edges[1:]
        sizes = end_rank - (start_rank - 1)
        j2 = end_rank - 1
        j1m1 = start_rank - 2
        pools = pay_csum[j2] - np.where(j1m1 >= 0, pay_csum[j1m1], 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            per_entry = np.divide(pools, sizes, out=np.zeros_like(pools, dtype=np.float64), where=sizes>0)
        for g in range(len(start_rank)):
            i0, i1 = edges[g], edges[g+1]
            idxs_group = order[i0:i1]
            if idxs_group.size == 0:
                continue
            payouts[sim_i, idxs_group] = per_entry[g]
            ranks[sim_i, idxs_group] = int(start_rank[g])

    expected_payout = payouts.mean(axis=0)

    # 9. Choose best candidate per entry
    best_per_entry: List[Tuple['OriginalEntry', List[int], float, Dict]] = []
    all_candidate_stats: List[Tuple[int, int, int, List[int], float, Dict]] = []

    # Build mapping entry_id -> OriginalEntry for quick access
    entry_map = {e.entry_id: e for e in entries}

    for global_idx, meta in enumerate(lineup_meta):
        entry_id, cand_idx = meta
        if entry_id == -1:
            continue  # field lineup
        payout_val = float(expected_payout[global_idx])
        lineup_players = all_lineups[global_idx]
        all_candidate_stats.append((entry_id, contest_id, cand_idx, lineup_players, payout_val, id_to_fp))

    # Determine best per entry
    from collections import defaultdict
    tmp_best: Dict[int, Tuple[int, List[int], float, Dict]] = defaultdict(lambda: (-1, [], -1.0, {}))
    for entry_id, _, cand_idx, lineup_players, payout_val, _ in all_candidate_stats:
        _, _, current_best, _ = tmp_best[entry_id]
        if payout_val > current_best:
            tmp_best[entry_id] = (cand_idx, lineup_players, payout_val, id_to_fp)

    for entry_id, (cand_idx, lineup_players, payout_val, _) in tmp_best.items():
        entry_obj = entry_map.get(entry_id)
        if entry_obj:
            best_per_entry.append((entry_obj, lineup_players, payout_val, id_to_fp))

    return best_per_entry, all_candidate_stats


