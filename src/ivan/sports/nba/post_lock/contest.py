from sqlalchemy.orm import Session
from typing import List, Tuple, Dict
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from rich.console import Console
from ivan.sports.nba.post_lock.entry_parsing import OriginalEntry
import numpy as np
from ivan.sports.nba.models import NBAPlayer, HistoricalContest, ContestPayout
from ivan.sports.nba.simulate_contest import load_scaled_payouts as _scale_payouts
from ivan.sports.nba.pre_lock.lineup_generation import FieldPlayer
from ivan.sports.nba.utils import _coerce_positions
from ivan.sports.common.draftkings import download_contest_dataframe
from ivan.sports.nba.post_lock.lineup import finish_lineup, generate_candidate
from ivan.sports.nba.pre_lock.player import generate_player_samples

WORKERS = 8
console = Console()


def _process_lineup_chunk(
    chunk_data: Tuple[List[str], Dict[str, int], Dict[int, FieldPlayer], np.ndarray, List[int]]
) -> List[List[int]]:
    """Worker function to process a chunk of lineup strings in parallel.
    
    Args:
        chunk_data: Tuple of (lineup_strings, name_map, id_to_fp, samples_matrix, player_ids)
            - samples_matrix: shape (n_samples, n_players) with random scores
            - player_ids: list of player IDs corresponding to columns in samples_matrix
    
    Returns:
        List of completed lineups (each is a list of 8 player IDs). Duplicates preserved.
    """
    lineup_strs, name_map, id_to_fp, samples_matrix, player_ids = chunk_data
    n_samples = samples_matrix.shape[0]
    results: List[List[int]] = []
    
    for lineup_str in lineup_strs:
        # Pick a random sample row for this lineup
        sample_idx = np.random.randint(0, n_samples)
        sample_row = samples_matrix[sample_idx]
        
        # Build base_scores dict from this sample
        base_scores: Dict[int, float] = {pid: float(sample_row[i]) for i, pid in enumerate(player_ids)}
        
        # finish_lineup's session param is unused, pass None
        completed_lineup = finish_lineup(None, lineup_str, name_map=name_map, id_to_fp=id_to_fp, base_scores=base_scores)
        if completed_lineup:
            results.append(completed_lineup)
    
    return results

def _sample_field_lineups(
    session: Session,
    contest_id: int,
    http_sess,
    name_map,
    num_field_samples: int = 10000,
) -> Tuple[List[List[int]], Dict[int, FieldPlayer], Dict[int, float]]: 
    
    # Get contest info for logging
    contest = session.query(HistoricalContest).filter(HistoricalContest.contest_id == contest_id).first()
    contest_name = contest.name if contest else f"Contest {contest_id}"
    target_entries = contest.total_entries if contest else num_field_samples
    
    console.print(f"\n[bold cyan]═══ Sampling Field for: {contest_name} ═══[/bold cyan]")
    console.print(f"[dim]Contest ID: {contest_id} | Target entries: {target_entries:,} | Sampling up to: {num_field_samples:,}[/dim]\n")

    raw_players: List[NBAPlayer] = session.query(NBAPlayer).all()
    player_pool: List[FieldPlayer] = []

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
    
    # Generate many samples for variety in field lineup completion
    # Each unique lineup will randomly pick one sample row
    NUM_FIELD_SAMPLES_MATRIX = 500  # enough variety for field diversity
    console.print(f"[dim]Generating {NUM_FIELD_SAMPLES_MATRIX} sample sets for field variety...[/dim]")
    
    raw_samples = generate_player_samples(session=session, n_samples=NUM_FIELD_SAMPLES_MATRIX, randomness=1.0)
    
    # Build ordered player ID list and samples matrix
    player_ids: List[int] = sorted(id_to_fp.keys())
    n_players = len(player_ids)
    samples_matrix = np.zeros((NUM_FIELD_SAMPLES_MATRIX, n_players), dtype=np.float32)
    
    for col_idx, pid in enumerate(player_ids):
        arr = raw_samples.get(pid)
        if arr is not None and isinstance(arr, np.ndarray) and arr.shape[0] == NUM_FIELD_SAMPLES_MATRIX:
            samples_matrix[:, col_idx] = arr.astype(np.float32, copy=False)
        else:
            # Fallback to fpts_mean
            mean_val = float(id_to_fp[pid].fpts_mean or 0.0)
            samples_matrix[:, col_idx] = mean_val
    
    # Also build a single base_scores dict for later use (using mean of samples)
    base_scores: Dict[int, float] = {pid: float(samples_matrix[:, i].mean()) for i, pid in enumerate(player_ids)}

    # Fetch the entire payload from draftkings endpoint
    df = download_contest_dataframe(sess=http_sess, contest_id=int(contest_id))
    if df is None or df.empty:
        console.print('[red]✗ No data returned from DraftKings[/red]')
        return None

    console.print(f"[dim]Downloaded {len(df):,} total rows from DraftKings[/dim]")

    cols = [str(c).strip() for c in df.columns]
    lineup_col = next((c for c in cols if c.lower() == 'lineup'), None)
    if not lineup_col:
        console.print('[red]✗ No lineup column found in DK data[/red]')
        console.print(f"[dim]Available columns: {cols}[/dim]")
        return None
    
    # Log how many rows have valid lineup data
    total_rows = len(df)
    non_null_rows = df[lineup_col].notna().sum()
    console.print(f"[dim]Rows with lineup data: {non_null_rows:,} / {total_rows:,}[/dim]")
    
    # Get all lineup strings (including duplicates)
    tmp = df[[lineup_col]].dropna()
    all_lineup_strs: List[str] = tmp[lineup_col].tolist()
    total_field_entries = len(all_lineup_strs)
    
    console.print(f"[dim]Field: {total_field_entries:,} entries[/dim]")
    
    # Random sample rows if we have more than the cap (no deduplication)
    if total_field_entries > int(num_field_samples):
        console.print(f"[dim]Randomly sampling {num_field_samples:,} rows from {total_field_entries:,}...[/dim]")
        sampled_indices = np.random.choice(total_field_entries, size=int(num_field_samples), replace=False)
        all_lineup_strs = [all_lineup_strs[i] for i in sampled_indices]
    
    console.print(f"[dim]Processing {len(all_lineup_strs):,} lineup rows...[/dim]")
    
    # Split lineup strings into chunks for parallel processing (includes duplicates)
    n_workers = WORKERS
    chunk_size = max(1, len(all_lineup_strs) // n_workers)
    chunks: List[List[str]] = []
    for i in range(0, len(all_lineup_strs), chunk_size):
        chunks.append(all_lineup_strs[i:i + chunk_size])
    
    # Prepare chunk data tuples for each worker
    chunk_data_list = [(chunk, name_map, id_to_fp, samples_matrix, player_ids) for chunk in chunks]
    
    # Process all lineups in parallel (each gets its own random sample for variety)
    all_results: List[List[int]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_process_lineup_chunk, chunk_data) for chunk_data in chunk_data_list]
        for future in as_completed(futures):
            try:
                chunk_results = future.result()  # List[List[int]]
                all_results.extend(chunk_results)
            except Exception as e:
                logging.error(f"Worker failed: {e}")
                continue

    completed: List[List[int]] = all_results
    failed_count = len(all_lineup_strs) - len(completed)
    
    if failed_count > 0:
        console.print(f"[yellow]⚠ {failed_count:,} lineups dropped (name lookup failures)[/yellow]")
    
    console.print(f"[green]✓ Field sampling complete:[/green] {len(completed):,} field entries")
    
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
    player_ids_all: List[int] = sorted(id_to_fp.keys())
    cand_nsamples = max(num_candidates_per_entry * 4, 200)  # more than needed for diversity
    cand_raw = generate_player_samples(session=session, n_samples=cand_nsamples, randomness=1.0)

    cand_matrix = np.zeros((cand_nsamples, len(player_ids_all)), dtype=np.float32)
    for j, pid in enumerate(player_ids_all):
        arr = cand_raw.get(pid)
        if isinstance(arr, np.ndarray) and arr.shape[0] == cand_nsamples:
            cand_matrix[:, j] = arr.astype(np.float32, copy=False)
        else:
            mean_val = float(id_to_fp[pid].fpts_mean or 0.0)
            cand_matrix[:, j] = mean_val  

    entry_to_candidates: Dict[str, List[List[int]]] = {}  # FIX typing to str keys
    for entry in entries:
        seen = set()
        candidates: List[List[int]] = []
        orig = entry.player_ids()
        if orig and len(orig) == 8:
            candidates.append(orig)
            seen.add(tuple(orig))
            # Generate unique candidates using different sample rows
        jitter_sd = 0.25  # small tie-break noise; tweak as needed
        row_order = np.random.permutation(cand_nsamples)
        for ridx in row_order:
            sample_row = cand_matrix[ridx]
            base_scores_iter = {pid: float(sample_row[j]) for j, pid in enumerate(player_ids_all)}

            try:
                cand = generate_candidate(session, entry, id_to_fp, base_scores_iter, jitter=jitter_sd)
            except Exception as e:
                logging.error(f"candidate generation failed for entry {entry.entry_id}: {e}")
                continue

            if not cand or len(cand) != 8:
                continue

            key = tuple(cand)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)

            if len(candidates) >= (num_candidates_per_entry + 1):  # +1 for baseline
                break

        entry_to_candidates[entry.entry_id] = candidates

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

    # Print chosen entry exposures
    if best_per_entry:
        # Get contest name for display
        contest = session.query(HistoricalContest).filter(HistoricalContest.contest_id == contest_id).first()
        contest_name = contest.name if contest else f"Contest {contest_id}"
        
        chosen_lineups = [lineup for (_, lineup, _, _) in best_per_entry]
        console.print(f"\n[bold green]═══ Chosen Entries for: {contest_name} ═══[/bold green]")
        console.print(f"[dim]Selected {len(best_per_entry)} optimal lineups from {len(entries)} entries[/dim]")
        

    return best_per_entry, all_candidate_stats


