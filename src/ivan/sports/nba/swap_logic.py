from __future__ import annotations
from sqlalchemy.orm import Session
from typing import List, Tuple, Set, Optional, Dict, Iterable
from collections import defaultdict
from datetime import datetime, timedelta
import json
import math
import csv
import numpy as np
import pandas as pd
from rich.console import Console

from ivan.sports.nba.models import HistoricalContest, NBAPlayer, Lineup, ProjectionSnapshot, ContestPayout
from ivan.core.utils import normalize_name
from ivan.sports.nba.lineup_generation import (
    generate_single_classic_lineup, 
    FieldPlayer,
    ROSTER_STRUCTURE,
    SLOT_NAMES
)
from ivan.sports.mlb.payouts import scale_payouts_correctly
from ivan.sports.nba.generate_samples import generate_player_samples
from ivan.sports.nba.utils import _coerce_positions
from ivan.sports.nba.sample_field import sample_field
import csv
from ivan.sports.nba.sample_field import sample_field


console = Console()


class OriginalEntry:
    """Represents a single original entry from the CSV."""
    def __init__(
        self,
        entry_id: str,
        contest_id: int,
        lineup_id: int,
        player_ids: List[int],
        player_names: Optional[List[str]] = None,
    ):
        self.entry_id = entry_id
        self.contest_id = contest_id
        self.lineup_id = lineup_id
        self.player_ids = player_ids  # All 8 players in original lineup (roster order)
        self.player_names = player_names or [""] * len(player_ids)
        # Derived during processing
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


def _map_players_to_slots(
    player_ids: List[int],
    session: Session,
) -> Dict[str, int]:
    """
    Map player IDs to roster slots based on the CSV order (PG, SG, SF, PF, C, G, F, UTIL).

    Rationale: The entries CSV already encodes the slot order. Using greedy eligibility-based
    assignment can fail in edge cases (e.g., multi-eligible players consumed earlier), causing
    infeasible mappings. We trust DraftKings export order and assign by index.
    """
    # Ensure exactly 8 players; if not, raise a clear error
    if len(player_ids) != len(SLOT_NAMES):
        raise ValueError(f"Expected {len(SLOT_NAMES)} players, got {len(player_ids)}")

    return {slot: pid for slot, pid in zip(SLOT_NAMES, player_ids)}


def _identify_locked_players(
    session: Session,
    player_ids: List[int],
    lock_time: datetime,
) -> Tuple[Set[int], Set[int]]:
    """Identify which players are locked (games started) vs swappable.

    Assumes that players removed from NBAPlayer are considered locked (post-lock cleanup).
    """
    if not player_ids:
        return set(), set()

    rows = session.query(NBAPlayer).filter(NBAPlayer.id.in_(player_ids)).all()
    found_ids = {p.id for p in rows}

    locked: Set[int] = set()
    swappable: Set[int] = set()

    for p in rows:
        if p.gametime and p.gametime <= lock_time:
            locked.add(p.id)
        else:
            swappable.add(p.id)

    # Any IDs not present in NBAPlayer: attempt classification via ProjectionSnapshot on same day
    missing = set(player_ids) - found_ids
    if missing:
        day_start = datetime(lock_time.year, lock_time.month, lock_time.day)
        day_end = day_start + timedelta(days=1)
        snaps = (
            session.query(ProjectionSnapshot)
            .filter(
                ProjectionSnapshot.id.in_(list(missing)),
                ProjectionSnapshot.snapshot_date >= day_start,
                ProjectionSnapshot.snapshot_date < day_end,
            )
            .all()
        )
        found_snap_ids = set()
        for s in snaps:
            found_snap_ids.add(s.id)
            if s.gametime and s.gametime <= lock_time:
                locked.add(s.id)
            else:
                swappable.add(s.id)
        # Any still missing after snapshots → conservatively mark locked
        locked.update(missing - found_snap_ids)

    return locked, swappable


def _generate_swap_candidates(
    session: Session,
    original_entry: OriginalEntry,
    lock_time: datetime,
    config: dict,
    num_candidates: int = 50,
    debug: bool = False,
) -> Tuple[List[SwapCandidate], Dict[int, Tuple[float, float]]]:
    """
    Simplified candidate generation matching field sampling flow:
    1. Build unified player pool (NBAPlayer + same-day ProjectionSnapshot)
    2. Map locked players to their roster slots
    3. Pass locked assignments to optimizer
    
    Args:
        debug: If True, print detailed diagnostic information
    """
    locked_pids = list(original_entry.locked_player_ids)
    n_swappable = len(original_entry.swappable_player_ids)

    console.print(f"[dim]Entry {original_entry.entry_id}: {len(locked_pids)} locked, {n_swappable} swappable[/dim]")

    if n_swappable == 0:
        if len(locked_pids) == 0:
            console.print(
                f"[red]  → Warning: entry {original_entry.entry_id} has 0 locked & 0 swappable players. Returning original lineup.[/red]"
            )
        else:
            console.print(f"[yellow]  → All {len(locked_pids)} players locked; returning original lineup only[/yellow]")
        return [SwapCandidate(
            original_entry_id=original_entry.entry_id,
            original_lineup_id=original_entry.lineup_id,
            locked_player_ids=original_entry.player_ids,
            swapped_player_ids=[],
            candidate_lineup_id=original_entry.lineup_id
        )], {}
    
    # === STEP 1: Build unified player pool (NBAPlayer + same-day ProjectionSnapshot) ===
    # Same pattern as sample_field.fetch_dk_field_lineups
    player_pool: List[FieldPlayer] = []
    seen_ids: Set[int] = set()
    fallback_stats: Dict[int, Tuple[float, float]] = {}
    
    def _to_field_player(p, is_snapshot=False):
        """Convert NBAPlayer or ProjectionSnapshot to FieldPlayer."""
        pos_tuple = _coerce_positions(getattr(p, 'position', []) or [])
        if isinstance(pos_tuple, str):
            pos_tuple = (pos_tuple,)
        elif not isinstance(pos_tuple, tuple):
            pos_tuple = tuple(pos_tuple) if pos_tuple else ()
        
        team = getattr(p, 'team', '') or ''
        opponent = getattr(p, 'opponent', None)
        game_id = "-".join(sorted([team.upper(), opponent.upper()])) if team and opponent else None
        
        pid = int(getattr(p, 'id'))
        name = getattr(p, 'player_name' if is_snapshot else 'name', f"PID {pid}")
        gametime = getattr(p, 'gametime', None)
        
        return FieldPlayer(
            id=pid,
            name=name,
            salary=int(getattr(p, 'salary') or 0),
            team=team,
            opponent=opponent,
            position=pos_tuple,
            fpts_mean=float(getattr(p, 'fpts_mean') or 0.0),
            game_id=game_id,
            gametime=gametime
        )
    
    # Load NBAPlayer rows (current active players)
    nba_players = session.query(NBAPlayer).all()
    for p in nba_players:
        fp = _to_field_player(p, is_snapshot=False)
        player_pool.append(fp)
        seen_ids.add(fp.id)
        fallback_stats[fp.id] = (
            float(getattr(p, 'fpts_mean') or 0.0),
            float(getattr(p, 'fpts_std', None) or 0.0),
        )
    
    # Load same-day ProjectionSnapshot rows (point-in-time projections for this date)
    day_start = datetime(lock_time.year, lock_time.month, lock_time.day)
    day_end = day_start + timedelta(days=1)
    snapshots = (
        session.query(ProjectionSnapshot)
        .filter(
            ProjectionSnapshot.snapshot_date >= day_start,
            ProjectionSnapshot.snapshot_date < day_end,
        )
        .all()
    )
    for snap in snapshots:
        if snap.id in seen_ids:
            continue
        fp = _to_field_player(snap, is_snapshot=True)
        player_pool.append(fp)
        seen_ids.add(fp.id)
        fallback_stats[fp.id] = (
            float(getattr(snap, 'fpts_mean') or 0.0),
            float(getattr(snap, 'fpts_std', None) or 0.0),
        )
    
    if not player_pool:
        console.print(f"[red]  ✗ No players in pool (NBAPlayer + ProjectionSnapshot empty)[/red]")
        return [], {}
    
    # === STEP 2: Map locked players to their roster slots ===
    try:
        slot_to_player = _map_players_to_slots(original_entry.player_ids, session)
    except ValueError as e:
        console.print(f"[red]✗ Failed to map players to slots for entry {original_entry.entry_id}: {e}[/red]")
        return [], {}
    
    # Build locked_assignments: {player_id: slot_name} for locked players only
    locked_assignments: Dict[int, str] = {}
    for slot_name, player_id in slot_to_player.items():
        if player_id in locked_pids:
            locked_assignments[player_id] = slot_name
    
    if debug:
        console.print(f"[dim]  → Locked assignments: {locked_assignments}[/dim]")
    
    # Diagnostic: Check that all locked players are in the pool
    missing_from_pool = set(locked_pids) - seen_ids
    if missing_from_pool:
        console.print(f"[red]  ✗ CRITICAL: Locked players {missing_from_pool} NOT in player pool! Cannot generate candidates.[/red]")
        if debug:
            console.print(f"[red]     Available player IDs (first 20): {list(seen_ids)[:20]}[/red]")
        return [], {}
    
    if debug:
        console.print(f"[dim]  → Locked assignments: {locked_assignments}[/dim]")
        console.print(f"[dim]  → Player pool: {len(player_pool)} total players[/dim]")
        console.print(f"[dim]  → All {len(locked_pids)} locked players found in pool ✓[/dim]")
    
    # === STEP 3: Generate diverse candidates using sample-based optimization ===
    candidates: List[SwapCandidate] = []
    seen_lineups: Set[Tuple[int, ...]] = set()
    
    # Generate player samples for randomness
    samples = generate_player_samples(
        session=session,
        n_samples=max(num_candidates * 3, 50),
        randomness=0.5,
        snapshot_date=lock_time.date(),
        include_active_players=True,
    )

    import random
    if samples:
        first_len = len(next(iter(samples.values())))
        sample_indices = list(range(first_len))
        random.shuffle(sample_indices)
    else:
        console.print(f"[yellow]  → No player samples available. Falling back to mean-only heuristic.[/yellow]")
        sample_indices = []
    
    console.print(f"[dim]  → Solver input: {len(player_pool)} total players, {len(locked_assignments)} locked assignments[/dim]")
    
    # Diagnostic: ensure we have base scores for all players
    if sample_indices:
        test_idx = sample_indices[0]
        sample_coverage = sum(1 for pid in seen_ids if pid in samples)
        console.print(f"[dim]  → Samples cover {sample_coverage}/{len(seen_ids)} players in pool[/dim]")
    else:
        mean_coverage = sum(1 for p in player_pool if p.fpts_mean and p.fpts_mean > 0)
        console.print(f"[dim]  → fpts_mean available for {mean_coverage}/{len(player_pool)} players[/dim]")
    
    attempts = 0
    max_attempts = num_candidates * 5
    failed_infeasible = 0
    failed_duplicate = 0
    failed_missing_locked = 0
    failed_exception = 0
    
    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1
        # Build base_scores either from sampled draws or fallback to fpts_mean
        if sample_indices:
            samp_idx = sample_indices[attempts % len(sample_indices)]
            base_scores = {pid: float(vec[samp_idx]) for pid, vec in samples.items() if pid in seen_ids}
        else:
            base_scores = {p.id: float(p.fpts_mean or 0.0) for p in player_pool}

        # Diagnostic: check locked players have scores
        if attempts == 1 and debug:
            locked_scores = {pid: base_scores.get(pid, None) for pid in locked_pids}
            console.print(f"[dim]  → Locked player scores (attempt 1): {locked_scores}[/dim]")

        # Add small jitter for diversity
        jitter = 0.05
        jittered_scores = {
            pid: score * (1.0 + random.uniform(-jitter, jitter))
            for pid, score in base_scores.items()
        }
        
        try:
            lineup_ids = generate_single_classic_lineup(
                players=player_pool,
                base_scores=jittered_scores,
                config=config,
                locked_assignments=locked_assignments,
                debug_lp_path='data/nba/logs/',
                attempt_tag=f"{original_entry.entry_id}_attempt{attempts}",
                lock_time=lock_time
            )
            
            if not lineup_ids:
                failed_infeasible += 1
                if attempts == 1:
                    console.print(f"[yellow]  → Attempt {attempts}: optimizer returned None (infeasible/non-optimal)[/yellow]")
                continue
            
            # Verify all locked players are present
            missing_locked = set(locked_pids) - set(lineup_ids)
            if missing_locked:
                failed_missing_locked += 1
                if debug and attempts == 1:
                    console.print(f"[yellow]  ⚠ Generated lineup missing locked players {missing_locked}[/yellow]")
                continue
            
            # Check uniqueness
            lineup_key = tuple(sorted(lineup_ids))
            if lineup_key in seen_lineups:
                failed_duplicate += 1
                continue
            seen_lineups.add(lineup_key)
            
            # Extract swapped player IDs (not in locked set)
            swapped_ids = [pid for pid in lineup_ids if pid not in locked_pids]
            
            candidate = SwapCandidate(
                original_entry_id=original_entry.entry_id,
                original_lineup_id=original_entry.lineup_id,
                locked_player_ids=locked_pids,
                swapped_player_ids=swapped_ids,
                candidate_lineup_id=len(candidates) + 1
            )
            candidates.append(candidate)
            
        except Exception as e:
            failed_exception += 1
            if attempts <= 3:  # Log first 3 exceptions with full details
                console.print(f"[red]  → Attempt {attempts}: Exception: {type(e).__name__}: {str(e)}[/red]")
                import traceback
                if debug:
                    console.print(f"[red]{traceback.format_exc()}[/red]")
            continue
    
    # Final diagnostic summary
    if len(candidates) == 0:
        console.print(f"[red]  ✗ Failed to generate any candidates after {attempts} attempts:[/red]")
        console.print(f"[red]    - Infeasible (None returned): {failed_infeasible}[/red]")
        console.print(f"[red]    - Duplicates: {failed_duplicate}[/red]")
        console.print(f"[red]    - Missing locked players: {failed_missing_locked}[/red]")
        console.print(f"[red]    - Exceptions: {failed_exception}[/red]")
        if debug:
            console.print(f"[red]    Locked player IDs: {locked_pids}[/red]")
            console.print(f"[red]    Locked assignments (pid→slot): {locked_assignments}[/red]")
            missing_from_pool = set(locked_pids) - seen_ids
            if missing_from_pool:
                console.print(f"[red]    ⚠ CRITICAL: Locked players {missing_from_pool} not found in player pool![/red]")
            else:
                console.print(f"[green]    All locked players found in pool ✓[/green]")
        console.print(f"[yellow]    Check data/nba/logs/ for LP files (if any were written)[/yellow]")
    else:
        console.print(f"[green]  ✓ Generated {len(candidates)} candidates[/green]")
        if debug:
            console.print(f"[dim]    (attempts={attempts}, dupes={failed_duplicate}, infeasible={failed_infeasible}, missing_locked={failed_missing_locked}, exceptions={failed_exception})[/dim]")
    
    return candidates, fallback_stats


def parse_entries_csv(
    csv_path: str,
    session: Session,
) -> Dict[int, List[OriginalEntry]]:
    """
    Parse DraftKings entries CSV in the native export format.

    Expects columns: Entry ID, Contest Name, Contest ID, Entry Fee, PG, SG, SF, PF, C, G, F, UTIL

    Entries with blank lineups (all position columns empty) are skipped.
    """
    try:
        df = pd.read_csv(csv_path, usecols=range(12))
    except Exception as e:
        Console().print(f"[red]✗ Failed to read entries CSV: {e}[/red]")
        raise

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    console = Console()
    entries_by_contest: Dict[int, List[OriginalEntry]] = {}

    # Expected position columns
    position_cols = ['pg', 'sg', 'sf', 'pf', 'c', 'g', 'f', 'util']

    # Verify all position columns exist
    available_cols = [c for c in df.columns]
    missing_cols = [col for col in position_cols if col not in df.columns]
    if missing_cols:
        console.print(
            f"[red]✗ Missing columns: {missing_cols}. Available: {', '.join(available_cols)[:100]}...[/red]"
        )
        return {}

    console.print(f"[cyan]Processing {len(df)} rows from CSV...[/cyan]")

    import re
    for _, row in df.iterrows():
        # Parse entry metadata
        entry_id_val = row.get('entry id', '')
        if pd.isna(entry_id_val):
            continue
        entry_id = str(entry_id_val).strip()

        contest_id_str = row.get('contest id', '0')
        if pd.isna(contest_id_str):
            continue
        try:
            contest_id = int(float(contest_id_str))
        except (ValueError, TypeError):
            continue

        # Check if entry has any players (all NaN means it's blank)
        has_any_player = any(
            (not pd.isna(row.get(col, '')) and str(row.get(col, '')).strip())
            for col in position_cols
        )
        if not has_any_player:
            console.print(f"[yellow]⊘ Entry {entry_id}: blank lineup, skipping[/yellow]")
            continue

        player_ids: List[int] = []
        player_names: List[str] = []
        skip_entry = False

        for pos_idx, pos_col in enumerate(position_cols):
            player_val = row.get(pos_col, '')
            player_str = '' if pd.isna(player_val) else str(player_val).strip()

            if not player_str:
                console.print(
                    f"[red]✗ Entry {entry_id}: missing player at position {pos_idx} ({pos_col.upper()})[/red]"
                )
                skip_entry = True
                break

            # Extract ID and name if available
            id_match = re.search(r'\((\d+)\)', player_str)
            name_match = re.match(r'^\s*([^\(]+?)\s*\(', player_str)

            player_id: Optional[int] = None
            if id_match:
                try:
                    player_id = int(id_match.group(1))
                except ValueError:
                    player_id = None

            if player_id is None:
                # Fallback: try pure integer ID
                try:
                    player_id = int(player_str)
                except ValueError:
                    # Fallback: fuzzy name lookup
                    name_text = player_str
                    p = session.query(NBAPlayer).filter(
                        NBAPlayer.name.ilike(f"%{name_text}%")
                    ).first()
                    if not p:
                        console.print(
                            f"[red]✗ Entry {entry_id} pos {pos_idx}: player '{name_text}' not found[/red]"
                        )
                        skip_entry = True
                        break
                    player_id = p.id
                    player_names.append(p.name)
            else:
                # We have an ID; attempt to get name from NBAPlayer; if missing, keep CSV name
                p = session.query(NBAPlayer).filter_by(id=player_id).first()
                if p:
                    player_names.append(p.name)
                elif name_match:
                    player_names.append(name_match.group(1).strip())
                else:
                    player_names.append("")

            if player_id is not None:
                player_ids.append(player_id)
            else:
                skip_entry = True
                break

        if skip_entry or len(player_ids) != 8:
            continue

        entry = OriginalEntry(
            entry_id=entry_id,
            contest_id=contest_id,
            lineup_id=-1,
            player_ids=player_ids,
            player_names=player_names,
        )

        entries_by_contest.setdefault(contest_id, []).append(entry)

    total_entries = sum(len(v) for v in entries_by_contest.values())
    if total_entries == 0:
        console.print(f"[red]✗ No valid entries parsed from {csv_path}. Failing.")
        raise RuntimeError("No valid entries parsed from entries CSV.")
    console.print(
        f"[green]✓ Parsed {total_entries} entries from {csv_path} across {len(entries_by_contest)} contests[/green]"
    )
    return entries_by_contest

# === Alternative late-swap flow (field sampling → per-entry candidates → ROI) ===

def _group_entries_by_contest(entries: List['OriginalEntry']) -> Dict[int, List['OriginalEntry']]:
    by_contest: Dict[int, List['OriginalEntry']] = {}
    for e in entries:
        by_contest.setdefault(e.contest_id, []).append(e)
    return by_contest


def _load_payouts_from_db(session: Session, contest_id: int) -> Tuple[np.ndarray, float, int]:
    """
    Returns tuple of (prize_per_rank, entry_fee, entries_max)
    prize_per_rank is 1-based index array of length entries_max+1
    """
    meta = session.query(HistoricalContest).filter_by(contest_id=contest_id).first()
    if not meta:
        raise ValueError(f"No HistoricalContest for contest_id={contest_id}")
    entry_fee = float(meta.entry_fee)
    entries_max = int(meta.multi_entry)

    prize_per_rank = np.zeros(entries_max + 1, dtype=float)
    payouts = (
        session.query(ContestPayout)
        .filter(ContestPayout.contest_id == contest_id)
        .all()
    )
    for p in payouts:
        start = int(p.min_position)
        end = int(p.max_position)
        prize_per_rank[start:end + 1] = float(p.payout_amount)
    return prize_per_rank, entry_fee, entries_max


def _sum_lineup_scores(player_samples: Dict[int, np.ndarray], lineup: List[int]) -> np.ndarray:
    # Build total score per simulation for the given lineup (sum of per-player draws)
    parts: List[np.ndarray] = []
    for pid in lineup:
        arr = player_samples.get(pid)
        if arr is None:
            continue
        parts.append(arr)
    if not parts:
        return np.zeros(1, dtype=float)
    return np.sum(np.stack(parts, axis=0), axis=0)


def _evaluate_candidates_vs_field(
    field_scores: np.ndarray,
    candidate_scores: List[np.ndarray],
    prize_per_rank: np.ndarray,
    entry_fee: float,
    entries_max: int,
) -> List[float]:
    """Compute mean ROI for each candidate against the sampled field."""
    N, T = field_scores.shape
    field_sorted_asc = np.sort(field_scores, axis=0)

    rois: List[float] = []
    for cand in candidate_scores:
        if cand.shape[0] != T:
            raise ValueError("Candidate score vector length != T")

        # For each simulation t, find how many field scores are <= candidate score
        idx_right = np.array([
            np.searchsorted(field_sorted_asc[:, t], cand[t], side="right")
            for t in range(T)
        ], dtype=int)
        # Rank in the sampled field, 1 = best
        rank_sample = 1 + (N - idx_right)

        # Scale rank to the full contest size
        rank_full = np.ceil(rank_sample * (entries_max / float(N))).astype(int)
        np.clip(rank_full, 1, entries_max, out=rank_full)

        prizes = prize_per_rank[rank_full]
        roi = (prizes - entry_fee) / entry_fee
        rois.append(float(np.mean(roi)))
    return rois


def _print_field_summary(
    session: Session,
    field_lineups: List[List[int]],
    unmatched_tracker: dict,
    contest_id: Optional[int] = None,
    snapshot_date = None,
    lineups_with_defaults: int = 0,
    total_attempted: int = 0,
    lock_time: Optional[datetime] = None,
) -> None:
    """Print comprehensive field summary after sampling.
    
    Args:
        snapshot_date: Date to use for looking up projections (uses ProjectionSnapshot if provided)
        lineups_with_defaults: Number of lineups that had to use default player values
        total_attempted: Total number of lineups attempted to parse
        lock_time: Time to determine if players are locked (game started)
    """
    from collections import Counter
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    total_lineups = len(field_lineups)
    
    # Calculate player exposures
    pid_counts = Counter(pid for lu in field_lineups for pid in lu if pid != -1)
    total_appearances = sum(pid_counts.values())
    
    # Get player info - try NBAPlayer first, then ProjectionSnapshot
    unique_pids = list(pid_counts.keys())
    players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(unique_pids)).all()
    player_map = {p.id: p for p in players}
    
    # Fill in missing players from ProjectionSnapshot if snapshot_date provided
    if snapshot_date:
        missing_pids = [pid for pid in unique_pids if pid not in player_map]
        if missing_pids:
            day_start = datetime.combine(snapshot_date, datetime.min.time())
            day_end = day_start + timedelta(days=1)
            snapshots = (
                session.query(ProjectionSnapshot)
                .filter(
                    ProjectionSnapshot.id.in_(missing_pids),
                    ProjectionSnapshot.snapshot_date >= day_start,
                    ProjectionSnapshot.snapshot_date < day_end,
                )
                .all()
            )
            # Use a simple wrapper to make ProjectionSnapshot look like NBAPlayer
            for snap in snapshots:
                if snap.id not in player_map:
                    class PlayerProxy:
                        def __init__(self, snap):
                            self.id = snap.id
                            self.name = snap.player_name or f"PID {snap.id}"
                            self.team = snap.team or ""
                            self.salary = snap.salary or 0
                            self.fpts_mean = snap.fpts_mean or 0.0
                            self.position = snap.position or ""
                            self.gametime = snap.gametime
                    player_map[snap.id] = PlayerProxy(snap)
    
    # Calculate lineup projection stats
    import numpy as np
    
    lineup_projs = []
    players_with_projs = 0
    players_without_projs = 0
    
    for lu in field_lineups:
        proj = 0.0
        for pid in lu:
            if pid in player_map and player_map[pid].fpts_mean:
                proj += player_map[pid].fpts_mean
                players_with_projs += 1
            else:
                players_without_projs += 1
        lineup_projs.append(proj)
    
    # Check if we have meaningful projection data
    proj_mean = np.mean(lineup_projs) if lineup_projs else 0.0
    has_projection_data = proj_mean > 0
    
    if lineup_projs and has_projection_data:
        proj_min = np.min(lineup_projs)
        proj_max = np.max(lineup_projs)
        proj_median = np.median(lineup_projs)
        proj_std = np.std(lineup_projs)
        
        # Create histogram buckets
        bins = np.linspace(proj_min, proj_max, 11)
        hist, _ = np.histogram(lineup_projs, bins=bins)
    
    # Create summary panel
    title = f"[bold cyan]Contest {contest_id} Field Summary[/bold cyan]" if contest_id else "[bold cyan]Field Summary[/bold cyan]"
    
    # Calculate default usage percentage
    default_pct = 100.0 * lineups_with_defaults / total_attempted if total_attempted > 0 else 0.0
    default_warning = ""
    if default_pct > 90:
        default_warning = f"\n  [red]⚠ WARNING: {default_pct:.1f}% of lineups used defaults - data quality issue![/red]"
    elif default_pct > 50:
        default_warning = f"\n  [yellow]⚠ {default_pct:.1f}% of lineups used defaults[/yellow]"
    
    summary_text = f"""[bold]Lineup Statistics:[/bold]
  • Total Lineups Attempted: {total_attempted:,}
  • Successfully Parsed: {total_lineups:,}
  • Lineups with Defaults: {lineups_with_defaults:,} ({default_pct:.1f}%){default_warning}
  • Unique Players: {len(unique_pids)}
  • Total Player Appearances: {total_appearances:,}
"""
    
    if has_projection_data:
        summary_text += f"""
[bold]Projection Distribution:[/bold]
  • Mean: {proj_mean:.2f} pts
  • Median: {proj_median:.2f} pts
  • Std Dev: {proj_std:.2f} pts
  • Range: {proj_min:.2f} - {proj_max:.2f} pts
"""
    else:
        # No projection data available
        missing_pct = 100.0 * players_without_projs / (players_with_projs + players_without_projs) if (players_with_projs + players_without_projs) > 0 else 0
        summary_text += f"""
[yellow]⚠ Projection Data:[/yellow]
  • Players with projections: {players_with_projs:,}
  • Players without projections: {players_without_projs:,} ({missing_pct:.1f}%)
  • [yellow]Unable to calculate lineup projections - missing snapshot data for {snapshot_date}[/yellow]
"""
    
    # Add histogram only if we have meaningful data
    if lineup_projs and has_projection_data:
        summary_text += "\n[bold]Lineup Projection Histogram:[/bold]\n"
        max_count = max(hist) if len(hist) > 0 else 1
        for i, count in enumerate(hist):
            bar_len = int(30 * count / max_count) if max_count > 0 else 0
            bar = "█" * bar_len
            pct = 100.0 * count / total_lineups
            summary_text += f"  {bins[i]:6.1f}-{bins[i+1]:6.1f}: {bar} {count:4d} ({pct:5.1f}%)\n"
    
    console.print(Panel(summary_text, title=title, expand=False))
    
    # Top exposures table
    top_owned = pid_counts.most_common(15)
    if top_owned:
        table = Table(title="Top 15 Player Exposures", show_header=True, header_style="bold cyan")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Player", style="white", width=22)
        table.add_column("Pos", style="cyan", width=6)
        table.add_column("Team", style="cyan", width=4)
        table.add_column("Salary", style="green", width=7, justify="right")
        table.add_column("Proj", style="magenta", width=6, justify="right")
        table.add_column("Value", style="yellow", width=6, justify="right")
        table.add_column("Exposure", style="yellow", width=10, justify="right")
        
        for rank, (pid, count) in enumerate(top_owned, 1):
            p = player_map.get(pid)
            if p:
                exposure_pct = 100.0 * count / total_lineups
                proj_display = f"{p.fpts_mean:.1f}" if p.fpts_mean and p.fpts_mean > 0 else "N/A"
                salary_display = f"${p.salary:,}" if p.salary else "N/A"
                
                # Calculate value: fpts / (salary / 1000)
                if p.salary and p.salary > 0 and p.fpts_mean and p.fpts_mean > 0:
                    value = p.fpts_mean / (p.salary / 1000)
                    value_display = f"{value:.2f}"
                else:
                    value_display = "N/A"
                
                # Get position
                position = getattr(p, 'position', '') or ''
                
                # Check if locked (game started)
                is_locked = False
                if lock_time and hasattr(p, 'gametime') and p.gametime:
                    is_locked = p.gametime <= lock_time
                
                # Color player name red if locked
                player_name = f"[red]{p.name}[/red]" if is_locked else p.name
                
                table.add_row(
                    str(rank),
                    player_name,
                    position,
                    p.team or "",
                    salary_display,
                    proj_display,
                    value_display,
                    f"{exposure_pct:.1f}%"
                )
        console.print(table)
    
    # Unmatched players summary
    if unmatched_tracker:
        total_unmatched = sum(unmatched_tracker.values())
        console.print(f"\n[yellow]⚠ Warning: {len(unmatched_tracker)} unique unmatched players ({total_unmatched} total occurrences)[/yellow]")
        if len(unmatched_tracker) <= 10:
            for name, count in sorted(unmatched_tracker.items(), key=lambda x: x[1], reverse=True):
                console.print(f"  • {name}: {count} occurrence(s)")
        else:
            top_unmatched = sorted(unmatched_tracker.items(), key=lambda x: x[1], reverse=True)[:10]
            for name, count in top_unmatched:
                console.print(f"  • {name}: {count} occurrence(s)")
            console.print(f"  • ... and {len(unmatched_tracker) - 10} more")
    else:
        console.print(f"\n[green]✓ All players matched successfully[/green]")
    
    console.print()


def _print_entry_with_candidates(
    session: Session,
    entry: 'OriginalEntry',
    candidates: List['SwapCandidate'],
    candidate_rois: List[float],
    lock_time: datetime,
    snapshot_date = None,
) -> None:
    """Print original entry and its generated candidates with ROI."""
    from rich.text import Text
    
    console = Console()
    
    # Get player info
    all_pids = set(entry.player_ids)
    for cand in candidates:
        all_pids.update(cand.full_lineup())
    
    # Try NBAPlayer first, then ProjectionSnapshot
    players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(all_pids)).all()
    player_map = {p.id: p for p in players}
    
    if snapshot_date:
        missing_pids = [pid for pid in all_pids if pid not in player_map]
        if missing_pids:
            day_start = datetime.combine(snapshot_date, datetime.min.time())
            day_end = day_start + timedelta(days=1)
            snapshots = (
                session.query(ProjectionSnapshot)
                .filter(
                    ProjectionSnapshot.id.in_(missing_pids),
                    ProjectionSnapshot.snapshot_date >= day_start,
                    ProjectionSnapshot.snapshot_date < day_end,
                )
                .all()
            )
            for snap in snapshots:
                if snap.id not in player_map:
                    class PlayerProxy:
                        def __init__(self, snap):
                            self.id = snap.id
                            self.name = snap.player_name or f"PID {snap.id}"
                            self.fpts_mean = snap.fpts_mean or 0.0
                            self.gametime = snap.gametime
                    player_map[snap.id] = PlayerProxy(snap)
    
    # Determine locked players (games already started)
    def is_locked(pid):
        p = player_map.get(pid)
        if p and hasattr(p, 'gametime') and p.gametime:
            return p.gametime <= lock_time
        return False
    
    # Print original entry
    original_proj = sum(player_map[pid].fpts_mean for pid in entry.player_ids if pid in player_map)
    
    # Create player name list with colors
    player_names = []
    for pid in entry.player_ids:
        p = player_map.get(pid)
        name = p.name if p else f"PID {pid}"
        if is_locked(pid):
            player_names.append(f"[red]{name}[/red]")
        else:
            player_names.append(name)
    
    console.print(f"\n[bold white]{entry.entry_id}:[/bold white] [{', '.join(player_names)}] | Proj: {original_proj:.1f}")
    
    # Print candidates (indented, different color)
    for idx, (cand, roi) in enumerate(zip(candidates, candidate_rois), start=1):
        cand_lineup = cand.full_lineup()
        cand_proj = sum(player_map[pid].fpts_mean for pid in cand_lineup if pid in player_map)
        
        # Create player name list with colors
        cand_names = []
        for pid in cand_lineup:
            p = player_map.get(pid)
            name = p.name if p else f"PID {pid}"
            if is_locked(pid):
                cand_names.append(f"[red]{name}[/red]")
            else:
                cand_names.append(name)
        
        console.print(f"  [cyan]→ Candidate {idx}:[/cyan] [{', '.join(cand_names)}] | Proj: {cand_proj:.1f} | ROI: {roi:.3f}")


def _sample_field_lineups(
    session: Session,
    num_field_samples: int,
    contest_id: Optional[int] = None,
    config: Optional[dict] = None,
    http_session=None,
    snapshot_date=None,
    unmatched_tracker: dict = None,
) -> tuple[List[List[int]], int, int]:
    """
    Build a list of field lineups (each a list of 8 player IDs).

    - If config[classic.field_source] == 'dk' and contest_id provided, pull from DK standings.
    - Else sample from DB-stored lineups.
    
    Args:
        unmatched_tracker: Optional dict to accumulate unmatched names (format: {name: count})
        
    Returns:
        Tuple of (lineups, lineups_with_defaults, total_attempted)
    """
    cfg = config or {}
    source = (cfg.get("classic", {}).get("field_source") or cfg.get("field_source") or "db").lower()
    if source == "dk" and contest_id is not None:
        from ivan.sports.nba.sample_field import fetch_dk_field_lineups
        # chrome profile path can be configured under paths.chrome_profile
        chrome_profile = (
            cfg.get("paths", {}).get("chrome_profile")
            or "/home/sam-pratt/ivan/chrome-selenium/dk-profile"
        )
        # Fetch as many as available from DK, we'll downsample if needed
        max_lineups = int(2_000_000)
        field, lineups_with_defaults, total_attempted = fetch_dk_field_lineups(
            session=session,
            contest_ids=[int(contest_id)],
            max_lineups=max_lineups,
            user_data_dir=chrome_profile,
            profile_directory="Default",
            sess=http_session,
            snapshot_date=snapshot_date,
            unmatched_tracker=unmatched_tracker,
        )
        # If DK has more than requested, sample down to requested size
        if isinstance(field, list) and len(field) > int(num_field_samples):
            import random as _rand
            _rand.seed(42)
            field = _rand.sample(field, int(num_field_samples))
        return (field, lineups_with_defaults, total_attempted)
    # default: DB lineups
    from ivan.sports.nba.sample_field import fetch_db_field_lineups
    db_lineups = fetch_db_field_lineups(session, num_field_samples)
    return (db_lineups, 0, len(db_lineups))


def lateswap_one_contest_field(
    session: Session,
    contest_id: int,
    entries: List['OriginalEntry'],
    lock_time: datetime,
    config: dict,
    num_candidates_per_entry: int = 30,
    num_field_samples: int = 10000,
    num_sims: int = 5000,
    seed: Optional[int] = 42,
    http_session=None,
    debug: bool = False,
) -> Tuple[List[Tuple['OriginalEntry', List[int], float]], List[Tuple[int, str, int, List[int], float]]]:
    """
    Alternative per-contest flow using sampled field and per-entry candidates.
    Returns list of (entry, best_lineup_ids, roi_mean)
    
    Args:
        debug: If True, print detailed diagnostic information
    """
    console = Console()
    console.print(f"[cyan]•[/cyan] Contest {contest_id}: sampling field (requested: {num_field_samples:,})")
    
    # Track unmatched players
    unmatched_tracker = {}
    
    field_lineups, lineups_with_defaults, total_attempted = _sample_field_lineups(
        session, 
        num_field_samples, 
        contest_id=contest_id, 
        config=config, 
        http_session=http_session,
        snapshot_date=lock_time.date(),
        unmatched_tracker=unmatched_tracker,
    )
    if not field_lineups:
        console.print(f"[red]✗[/red] No field lineups sampled for contest {contest_id}")
        raise RuntimeError(f"No field lineups sampled for contest {contest_id}")
    
    # Print comprehensive field summary
    _print_field_summary(
        session, field_lineups, unmatched_tracker, 
        contest_id=contest_id, 
        snapshot_date=lock_time.date(),
        lineups_with_defaults=lineups_with_defaults,
        total_attempted=total_attempted,
        lock_time=lock_time
    )

    # Player samples include same-day snapshots to cover locked players
    console.print(f"[cyan]•[/cyan] Building player samples (T={num_sims})")
    randomness = float(config.get("classic", {}).get("lineup_generation", {}).get("randomness", 0.05))
    player_samples = generate_player_samples(
        session,
        n_samples=num_sims,
        randomness=randomness,
        seed=seed,
        snapshot_date=lock_time.date(),
        include_active_players=True,
    )

    # Precompute field scores (N, T)
    console.print(f"[cyan]•[/cyan] Precomputing field scores")
    field_scores = np.stack([_sum_lineup_scores(player_samples, lu) for lu in field_lineups], axis=0)

    # Payouts
    prize_per_rank, entry_fee, entries_max = _load_payouts_from_db(session, contest_id)

    results: List[Tuple['OriginalEntry', List[int], float]] = []
    candidate_rows: List[Tuple[int, str, int, List[int], float]] = []  # (contest_id, entry_id, idx, lineup_ids, roi)

    # Ensure lock classification has been performed for this flow (was missing previously)
    console.print(f"[cyan]•[/cyan] Classifying locked/swappable players for {len(entries)} entries")
    classified_any = False
    for entry in entries:
        # Only classify if not already populated
        if not entry.locked_player_ids and not entry.swappable_player_ids:
            locked, swappable = _identify_locked_players(session, entry.player_ids, lock_time)
            entry.locked_player_ids = locked
            entry.swappable_player_ids = swappable
        classified_any = True
        console.print(f"  [dim]Entry {entry.entry_id}: {len(entry.locked_player_ids)} locked, {len(entry.swappable_player_ids)} swappable[/dim]")
    if not any(e.swappable_player_ids for e in entries):
        console.print(f"[yellow]⚠ All entries appear fully locked after classification; candidate generation will return originals only[/yellow]")

    console.print(f"[cyan]•[/cyan] Generating candidates and simulating ({len(entries)} entries)")


    for entry in entries:
        candidates, _ = _generate_swap_candidates(
            session=session,
            original_entry=entry,
            lock_time=lock_time,
            config=config,
            num_candidates=num_candidates_per_entry,
            debug=debug,
        )
        if not candidates:
            console.print(f"[yellow]![/yellow] No candidates for entry {entry.entry_id}; skipping")
            continue

        # Always include the original lineup as a candidate (first position)
        original_lineup = entry.player_ids
        all_candidate_objs = [SwapCandidate(
            original_entry_id=entry.entry_id,
            original_lineup_id=entry.lineup_id,
            locked_player_ids=list(entry.locked_player_ids),
            swapped_player_ids=[pid for pid in original_lineup if pid not in entry.locked_player_ids],
            candidate_lineup_id=entry.lineup_id
        )] + candidates

        candidate_lineups = [c.full_lineup() for c in all_candidate_objs]
        candidate_scores = [_sum_lineup_scores(player_samples, lu) for lu in candidate_lineups]

        # For each candidate, compute the median fpts (projected score) and select the max
        medians = [float(np.median(cand_score)) for cand_score in candidate_scores]

        # Emit per-candidate rows (median fpts)
        for idx, (lu, median_fpts) in enumerate(zip(candidate_lineups, medians), start=0):
            candidate_rows.append((contest_id, entry.entry_id, idx, lu, float(median_fpts)))

        # Pick best by median fpts
        best_idx = int(np.argmax(medians))
        results.append((entry, candidate_lineups[best_idx], float(medians[best_idx])))

        # Print original entry and all candidates
        _print_entry_with_candidates(
            session, entry, all_candidate_objs, medians, lock_time, snapshot_date=lock_time.date()
        )

    return results, candidate_rows


def lateswap_slate(
    session: Session,
    entries: List['OriginalEntry'],
    lock_time: datetime,
    config: dict,
    num_candidates_per_entry: int = 30,
    num_field_samples: int = 10000,
    num_sims: int = 5000,
    seed: Optional[int] = 42,
    debug: bool = False,
) -> Tuple[List[Tuple['OriginalEntry', List[int], float]], List[Tuple[int, str, int, List[int], float]]]:
    """Slate-level coordinator: groups by contest and runs late swap for each.
    
    Args:
        debug: If True, print detailed diagnostic information
    """
    by_contest = _group_entries_by_contest(entries)
    # Create a single authenticated DK session if using DK field source
    source = (config.get("classic", {}).get("field_source") or config.get("field_source") or "db").lower()
    http_sess = None
    if source == "dk":
        from ivan.sports.common.draftkings import build_requests_session_from_chrome
        chrome_profile = (
            config.get("paths", {}).get("chrome_profile")
            or "/home/sam-pratt/ivan/chrome-selenium/dk-profile"
        )
        http_sess = build_requests_session_from_chrome(
            user_data_dir=chrome_profile,
            profile_directory="Default",
            headless=False,
        )
    all_results: List[Tuple['OriginalEntry', List[int], float]] = []
    all_candidate_rows: List[Tuple[int, str, int, List[int], float]] = []
    for contest_id, group in by_contest.items():
        result, cand_rows = lateswap_one_contest_field(
            session=session,
            contest_id=contest_id,
            entries=group,
            lock_time=lock_time,
            config=config,
            num_candidates_per_entry=num_candidates_per_entry,
            num_field_samples=num_field_samples,
            num_sims=num_sims,
            seed=seed,
            http_session=http_sess,
            debug=debug,
        )
        all_results.extend(result)
        all_candidate_rows.extend(cand_rows)
    return all_results, all_candidate_rows


def write_lateswap_csv_simple(
    results: List[Tuple['OriginalEntry', List[int], float]],
    output_path: str,
) -> None:
    """Minimal CSV: entry_id, 8 player ids in roster order, roi_mean."""
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entry_id", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "roi_mean"])
        for entry, lineup_ids, roi_mean in results:
            row = [entry.entry_id] + lineup_ids + [roi_mean]
            w.writerow(row)
    Console().print(f"[green]✓[/green] Wrote late-swap CSV: {output_path}")


def write_lateswap_candidates_csv(
    candidate_rows: List[Tuple[int, str, int, List[int], float]],
    output_path: str,
) -> None:
    """Write all candidate lineups with ROI to CSV: contest_id, entry_id, idx, p1..p8, roi_mean."""
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contest_id", "entry_id", "candidate_index", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "roi_mean"])
        for contest_id, entry_id, idx, lineup_ids, roi_mean in candidate_rows:
            row = [contest_id, entry_id, idx] + lineup_ids + [roi_mean]
            w.writerow(row)
    Console().print(f"[green]✓[/green] Wrote candidate ROI CSV: {output_path}")




