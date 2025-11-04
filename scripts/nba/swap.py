#!/usr/bin/env python3
"""
Late-swap module for DraftKings NBA DFS contests.

This version handles DraftKings' native CSV export format where:
- Entries are shown with their current lineups in columns PG, SG, SF, PF, C, G, F, UTIL
- Empty/blank entries indicate no lineup assigned yet (skip these)
- Format: "Name (ID) (STATUS)" e.g. "Luka Doncic (40618539) (LOCKED)"
"""

import os
import json
import argparse
import math
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional
from sqlalchemy.orm import Session
from concurrent.futures import ProcessPoolExecutor, as_completed

from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import (
    HistoricalContest, ContestPayout, Lineup, LineupSimulationResult,
    NBAPlayer, NBAStat
)
from ivan.sports.nba.sample_field import sample_field
from ivan.sports.nba.generate_samples import generate_player_samples
from ivan.sports.mlb.payouts import scale_payouts_correctly
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class OriginalEntry:
    """Represents a single original entry from the CSV."""
    def __init__(self, entry_id: str, contest_id: int, lineup_id: int, player_ids: List[int]):
        self.entry_id = entry_id
        self.contest_id = contest_id
        self.lineup_id = lineup_id
        self.player_ids = player_ids  # All 8 players in original lineup
        self.locked_player_ids: Set[int] = set()  # Subset that are locked
        self.swappable_player_ids: Set[int] = set()  # Can be swapped out
        self.best_candidate: Optional['SwapCandidate'] = None


class SwapCandidate:
    """Represents a potential swap lineup for an entry."""
    def __init__(
        self,
        original_entry_id: str,
        original_lineup_id: int,
        locked_player_ids: List[int],
        swapped_player_ids: List[int],
        candidate_lineup_id: int,
    ):
        self.original_entry_id = original_entry_id
        self.original_lineup_id = original_lineup_id
        self.locked_player_ids = locked_player_ids
        self.swapped_player_ids = swapped_player_ids
        self.candidate_lineup_id = candidate_lineup_id
        
        # Filled during simulation
        self.mean_fpts: Optional[float] = None
        self.mean_payout: Optional[float] = None
        self.mean_roi: Optional[float] = None
        self.mean_rank: Optional[float] = None

    def full_lineup(self) -> List[int]:
        """Return complete lineup: locked + swapped players."""
        return self.locked_player_ids + self.swapped_player_ids


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _identify_locked_players(
    session: Session,
    player_ids: List[int],
    lock_time: datetime,
) -> Tuple[Set[int], Set[int]]:
    """
    Identify which players are locked (games started) vs. swappable.
    """
    if not player_ids:
        return set(), set()
        
    players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(player_ids)).all()

    locked = set()
    swappable = set()
    
    for player in players:
        if player.gametime and player.gametime <= lock_time:
            locked.add(player.id)
        else:
            swappable.add(player.id)
    
    return locked, swappable


def _generate_swap_candidates(
    session: Session,
    original_entry: OriginalEntry,
    num_candidates: int = 50,
    allow_duplicates: bool = True,
) -> List[SwapCandidate]:
    """
    Generate candidate lineups by keeping locked players and generating new swappable slots.
    """
    locked_pids = list(original_entry.locked_player_ids)
    n_swappable = len(original_entry.swappable_player_ids)
    
    if n_swappable == 0:
        # No swaps needed; just use original
        candidate = SwapCandidate(
            original_entry_id=original_entry.entry_id,
            original_lineup_id=original_entry.lineup_id,
            locked_player_ids=locked_pids,
            swapped_player_ids=list(original_entry.swappable_player_ids),
            candidate_lineup_id=original_entry.lineup_id,
        )
        return [candidate]
    
    original_pids_set = set(original_entry.player_ids)
    
    available_players = session.query(NBAPlayer).filter(
        ~NBAPlayer.id.in_(original_pids_set)
    ).all()
    
    if not available_players:
        # Fallback: just return original if no options
        candidate = SwapCandidate(
            original_entry_id=original_entry.entry_id,
            original_lineup_id=original_entry.lineup_id,
            locked_player_ids=locked_pids,
            swapped_player_ids=list(original_entry.swappable_player_ids),
            candidate_lineup_id=original_entry.lineup_id,
        )
        return [candidate]
    
    candidates = []
    avail_ids = np.array([p.id for p in available_players])
    n_available = len(avail_ids)
    
    # Generate candidates by randomly sampling swappable positions
    for cand_idx in range(num_candidates):
        swapped = list(np.random.choice(avail_ids, size=n_swappable, replace=allow_duplicates))
        
        candidate = SwapCandidate(
            original_entry_id=original_entry.entry_id,
            original_lineup_id=original_entry.lineup_id,
            locked_player_ids=locked_pids,
            swapped_player_ids=swapped,
            candidate_lineup_id=-1,
        )
        candidates.append(candidate)
    
    return candidates


def _load_and_scale_payouts_from_db(
    session: Session,
    contest_id: int,
    num_field_lineups: int,
    original_field_size: int,
    entry_fee: float,
) -> Dict[int, float]:
    """Scale payouts to desired contest size."""
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


def _build_player_samples_matrix(
    session: Session,
    pid_to_idx: Dict[int, int],
    n_samples: int = 1000,
) -> Tuple[np.ndarray, int]:
    """Build player FPTS samples matrix."""
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


def _expand_candidates_for_simulation(
    candidates: List[SwapCandidate],
) -> Tuple[List[SwapCandidate], List[List[int]], Dict[int, int], np.ndarray, np.ndarray]:
    """Expand candidates into a simulation matrix format."""
    all_pids = set()
    for cand in candidates:
        all_pids.update(cand.full_lineup())
    
    unique_pids = sorted(all_pids)
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    
    lineup_ix = []
    for cand in candidates:
        full_lineup = cand.full_lineup()
        idxs = [pid_to_idx[p] for p in full_lineup]
        lineup_ix.append(idxs)
    
    M = len(candidates)
    starts = np.arange(M, dtype=np.int64)
    ends = np.arange(1, M + 1, dtype=np.int64)
    
    return candidates, lineup_ix, pid_to_idx, starts, ends


def _simulate_lateswap_chunk(
    start: int,
    end: int,
    lineup_ix: List[List[int]],
    P_chunk: np.ndarray,
    entry_fee: float,
    scaled_payouts: Dict[int, float],
) -> Dict[str, np.ndarray]:
    """Simulate one chunk of swap candidates."""
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

        # Score each candidate
        for m in range(M):
            idxs = lineup_ix[m]
            if idxs:
                scores[local_i, m] = row[idxs].sum(dtype=np.float64)

        # Rank by score
        order = np.argsort(-scores[local_i], kind="stable")
        sc_sorted = scores[local_i][order]

        diffs = np.abs(np.diff(sc_sorted))
        edges = np.concatenate(([0], np.nonzero(diffs > atol)[0] + 1, [M]))

        start_rank = edges[:-1] + 1
        end_rank = edges[1:]
        sizes = end_rank - (start_rank - 1)

        j2 = end_rank - 1
        j1m1 = start_rank - 2
        pools = pay_csum[j2] - np.where(j1m1 >= 0, pay_csum[j1m1], 0.0)

        with np.errstate(invalid='ignore', divide='ignore'):
            per_entry = np.divide(
                pools, sizes,
                out=np.zeros_like(pools, dtype=np.float64),
                where=sizes > 0
            )

        for g in range(len(start_rank)):
            i0, i1 = edges[g], edges[g+1]
            idx_in_field = order[i0:i1]
            if idx_in_field.size == 0:
                continue
            payouts[local_i, idx_in_field] = per_entry[g]
            ranks[local_i, idx_in_field] = int(start_rank[g])

        rois[local_i] = (payouts[local_i] - entry_fee) / entry_fee

    return {
        "start": start,
        "end": end,
        "scores": scores.astype(np.float32, copy=False),
        "payouts": payouts.astype(np.float32, copy=False),
        "rois": rois.astype(np.float32, copy=False),
        "ranks": ranks,
    }


# ============================================================================
# MAIN LATESWAP FLOW
# ============================================================================

def parse_entries_csv(
    csv_path: str,
    session: Session,
) -> Dict[int, List[OriginalEntry]]:
    """
    Parse DraftKings entries CSV in the native export format.
    
    Expects columns: Entry ID, Contest Name, Contest ID, Entry Fee, PG, SG, SF, PF, C, G, F, UTIL
    
    Entries with blank lineups (all position columns empty) are skipped.
    """
    df = pd.read_csv(csv_path, usecols=range(12))
    
    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    
    console = Console()
    entries_by_contest = {}
    
    # Expected position columns
    position_cols = ['pg', 'sg', 'sf', 'pf', 'c', 'g', 'f', 'util']
    
    # Verify all position columns exist
    available_cols = [c for c in df.columns]
    missing_cols = [col for col in position_cols if col not in df.columns]
    if missing_cols:
        console.print(
            f"[red]✗ Missing columns: {missing_cols}. "
            f"Available: {', '.join(available_cols)[:100]}...[/red]"
        )
        return {}
    
    console.print(f"[cyan]Processing {len(df)} rows from CSV...[/cyan]")
    
    for idx, row in df.iterrows():
        # Parse entry metadata
        entry_id = row.get('entry id', '')
        if pd.isna(entry_id):
            continue
        
        entry_id = str(entry_id).strip()
        
        contest_name = str(row.get('contest name', '')).strip()
        contest_id_str = row.get('contest id', '0')
        
        if pd.isna(contest_id_str):
            continue
        
        if not entry_id or not contest_id_str:
            continue
        
        try:
            contest_id = int(float(contest_id_str))
        except (ValueError, TypeError):
            continue
        
        # Check if entry has any players (all NaN means it's blank)
        has_any_player = False
        for pos_col in position_cols:
            val = row.get(pos_col, '')
            if not pd.isna(val) and str(val).strip():
                has_any_player = True
                break
        
        if not has_any_player:
            console.print(f"[yellow]⊘ Entry {entry_id}: blank lineup, skipping[/yellow]")
            continue
        
        # Parse player IDs from position columns
        # Format: "Name (ID) (STATUS)" e.g. "Luka Doncic (40618539) (LOCKED)"
        player_ids = []
        skip_entry = False
        
        for pos_idx, pos_col in enumerate(position_cols):
            player_val = row.get(pos_col, '')
            
            # Handle NaN
            if pd.isna(player_val):
                player_val = ''
            else:
                player_val = str(player_val).strip()
            
            # All positions must be filled
            if not player_val:
                console.print(
                    f"[red]✗ Entry {entry_id}: missing player at position {pos_idx} ({pos_col.upper()})[/red]"
                )
                skip_entry = True
                break
            
            player_id = None
            
            # Try to extract player ID from format "Name (ID) (STATUS)"
            match = re.search(r'\((\d+)\)', player_val)
            if match:
                try:
                    player_id = int(match.group(1))
                    # Verify player exists in DB
                    player = session.query(NBAPlayer).filter_by(id=player_id).first()
                    if not player:
                        console.print(
                            f"[red]✗ Entry {entry_id} pos {pos_idx}: player ID {player_id} not in database[/red]"
                        )
                        skip_entry = True
                        break
                except ValueError as e:
                    console.print(
                        f"[red]✗ Entry {entry_id} pos {pos_idx}: failed to parse ID from '{player_val}': {e}[/red]"
                    )
                    skip_entry = True
                    break
            else:
                # No ID found; try parsing as pure integer
                try:
                    player_id = int(player_val)
                    player = session.query(NBAPlayer).filter_by(id=player_id).first()
                    if not player:
                        console.print(
                            f"[red]✗ Entry {entry_id} pos {pos_idx}: player ID {player_id} not in database[/red]"
                        )
                        skip_entry = True
                        break
                except ValueError:
                    # Try fuzzy name matching
                    player_name = player_val
                    player = session.query(NBAPlayer).filter(
                        NBAPlayer.name.ilike(f"%{player_name}%")
                    ).first()
                    
                    if not player:
                        console.print(
                            f"[red]✗ Entry {entry_id} pos {pos_idx}: player '{player_name}' not in database[/red]"
                        )
                        skip_entry = True
                        break
                    
                    player_id = player.id
            
            if player_id is not None:
                player_ids.append(player_id)
            else:
                skip_entry = True
                break
        
        if skip_entry or len(player_ids) != 8:
            continue
        
        console.print(f"[green]✓ Entry {entry_id}: {player_ids}[/green]")
        
        # Create entry
        entry = OriginalEntry(
            entry_id=entry_id,
            contest_id=contest_id,
            lineup_id=-1,
            player_ids=player_ids,
        )
        
        if contest_id not in entries_by_contest:
            entries_by_contest[contest_id] = []
        entries_by_contest[contest_id].append(entry)
    
    total_entries = sum(len(v) for v in entries_by_contest.values())
    console.print(
        f"[green]✓ Parsed {total_entries} entries from {csv_path} "
        f"across {len(entries_by_contest)} contests[/green]"
    )
    
    return entries_by_contest


def lateswap_one_contest(
    session: Session,
    contest_id: int,
    entries: List[OriginalEntry],
    config_path: str,
    lock_time: datetime,
    num_sims: int = 1000,
    num_swap_candidates: int = 50,
) -> List[Tuple[str, SwapCandidate]]:
    """Run lateswap logic for a single contest."""
    cfg = json.load(open(config_path, encoding="utf-8"))
    
    contest = session.query(HistoricalContest).filter_by(contest_id=contest_id).one()
    field_sz = int(contest.total_entries)
    entry_fee = float(contest.entry_fee)
    
    console = Console()
    
    # Identify locked vs swappable
    console.print(f"[cyan]Identifying locked players for {len(entries)} entries...")
    for entry in entries:
        locked, swappable = _identify_locked_players(session, entry.player_ids, lock_time)
        entry.locked_player_ids = locked
        entry.swappable_player_ids = swappable
        console.print(f"[cyan]  Entry {entry.entry_id}: {len(locked)} locked, {len(swappable)} swappable[/cyan]")
    
    # Generate candidates
    console.print(f"[cyan]Generating {num_swap_candidates} candidates per entry...")
    all_candidates = []
    entry_to_candidates = {}
    
    for entry in entries:
        candidates = _generate_swap_candidates(
            session,
            entry,
            num_candidates=num_swap_candidates,
            allow_duplicates=True,
        )
        entry_to_candidates[entry.entry_id] = candidates
        all_candidates.extend(candidates)
    
    console.print(f"[cyan]Total candidates: {len(all_candidates)}")
    
    # Expand for simulation
    candidates_list, lineup_ix, pid_to_idx, starts, ends = _expand_candidates_for_simulation(all_candidates)
    
    M = len(candidates_list)
    
    # Scale payouts
    scaled = _load_and_scale_payouts_from_db(
        session=session,
        contest_id=contest_id,
        num_field_lineups=M,
        original_field_size=field_sz,
        entry_fee=entry_fee
    )
    
    # Generate FPTS samples
    console.print(f"[cyan]Generating {num_sims} FPTS samples...")
    P, num_sims_actual = _build_player_samples_matrix(
        session=session,
        pid_to_idx=pid_to_idx,
        n_samples=num_sims
    )
    
    # Run simulations
    console.print(f"[cyan]Running {num_sims_actual} simulations across {M} candidates...")
    chunk_size = math.ceil(num_sims_actual / min(12, num_sims_actual))
    
    full_scores = np.zeros((num_sims_actual, M), dtype=np.float32)
    full_payouts = np.zeros_like(full_scores)
    full_rois = np.zeros_like(full_scores)
    full_ranks = np.zeros((num_sims_actual, M), dtype=int)

    with ProcessPoolExecutor(max_workers=8) as exe:
        futures = []
        for start in range(0, num_sims_actual, chunk_size):
            end = min(start + chunk_size, num_sims_actual)
            futures.append(
                exe.submit(
                    _simulate_lateswap_chunk,
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
    
    # Compute aggregates per candidate
    console.print(f"[cyan]Computing mean metrics per candidate...")
    for i, candidate in enumerate(candidates_list):
        candidate.mean_fpts = float(full_scores[:, i].mean())
        candidate.mean_payout = float(full_payouts[:, i].mean())
        candidate.mean_roi = float(full_rois[:, i].mean())
        candidate.mean_rank = float(full_ranks[:, i].mean())
    
    # Select best candidate per entry
    console.print(f"[cyan]Selecting best candidate per entry (by ROI)...")
    results = []
    
    for entry in entries:
        entry_candidates = entry_to_candidates.get(entry.entry_id, [])
        if not entry_candidates:
            continue
        
        best = max(entry_candidates, key=lambda c: c.mean_roi if c.mean_roi else -float('inf'))
        entry.best_candidate = best
        results.append((entry.entry_id, best))
        console.print(
            f"[green]  Entry {entry.entry_id}: best ROI={best.mean_roi:.2%}[/green]"
        )
    
    console.print(f"[green]✓ Completed contest {contest_id}: {len(results)} entries processed[/green]")
    
    return results


def write_lateswap_csv(
    results: List[Tuple[str, SwapCandidate]],
    output_path: str,
    session: Session,
) -> None:
    """Write lateswap results to CSV for DraftKings upload."""
    rows = []
    
    for entry_id, candidate in results:
        full_lineup = candidate.full_lineup()
        
        row = {
            'Entry ID': entry_id,
            'Original Lineup ID': candidate.original_lineup_id,
            'Locked Players': ','.join(map(str, candidate.locked_player_ids)),
            'Swapped Players': ','.join(map(str, candidate.swapped_player_ids)),
        }
        
        for i, pid in enumerate(full_lineup, 1):
            row[f'Player {i}'] = pid
        
        row['Mean FPTS'] = candidate.mean_fpts
        row['Mean Payout'] = candidate.mean_payout
        row['Mean ROI'] = candidate.mean_roi
        row['Mean Rank'] = candidate.mean_rank
        
        rows.append(row)
    
    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run late-swap optimization for NBA DFS contests"
    )
    parser.add_argument(
        "--output-csv",
        default="lateswap_results.csv",
        help="Output CSV path for optimized lineups"
    )
    parser.add_argument(
        "--config",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../config/nba.json")
        ),
        help="Config path"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1000,
        help="Number of simulation samples per contest (default: 1000)"
    )
    parser.add_argument(
        "--lock-time",
        type=str,
        help="Lock time (ISO format, e.g., '2025-10-31T19:15:00'). If not provided, uses current time."
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=50,
        help="Number of swap candidates per entry (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Parse lock time
    if args.lock_time:
        lock_time = datetime.fromisoformat(args.lock_time)
    else:
        lock_time = datetime.now()
    
    console = Console()
    console.print(f"[bold]NBA Late-Swap Optimizer[/bold]")
    console.print(f"Lock time: {lock_time}")
    console.print(f"Simulations per contest: {args.n}")
    console.print(f"Candidates per entry: {args.candidates}\n")
    
    # Initialize DB
    engine = init_db("nba")
    session = get_session(engine)
    entries_csv = '/home/sam-pratt/ivan/data/nba/input/entries.csv'
    
    # Parse entries
    console.print("[cyan]Parsing entries CSV...\n[/cyan]")
    entries_by_contest = parse_entries_csv(entries_csv, session)
    console.print(f"\nFound {sum(len(v) for v in entries_by_contest.values())} "
                  f"entries across {len(entries_by_contest)} contests\n")
    
    if not entries_by_contest:
        console.print("[red]✗ No valid entries found in CSV. Exiting.[/red]")
        return
    
    all_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            f"[cyan]Processing {len(entries_by_contest)} contests...",
            total=len(entries_by_contest)
        )
        
        for contest_id, entries in entries_by_contest.items():
            progress.update(
                overall_task,
                description=f"[cyan]Contest {contest_id}: {len(entries)} entries..."
            )
            
            try:
                results = lateswap_one_contest(
                    session=session,
                    contest_id=contest_id,
                    entries=entries,
                    config_path=args.config,
                    lock_time=lock_time,
                    num_sims=args.n,
                    num_swap_candidates=args.candidates,
                )
                all_results.extend(results)
            except Exception as e:
                console.print(f"[red]✗ Error on contest {contest_id}: {e}[/red]")
                import traceback
                traceback.print_exc()
            
            progress.advance(overall_task)
    
    # Write output
    console.print(f"\n[cyan]Writing results to {args.output_csv}...")
    write_lateswap_csv(all_results, args.output_csv, session)
    
    console.print(f"[bold green]✓ Late-swap complete: {len(all_results)} entries optimized[/bold green]")


if __name__ == "__main__":
    main()