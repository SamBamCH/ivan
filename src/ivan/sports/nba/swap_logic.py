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
from collections import Counter


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

    # Any IDs not present in NBAPlayer are assumed to be locked
    missing = set(player_ids) - found_ids
    locked.update(missing)

    return locked, swappable


def _generate_swap_candidates(
    session: Session,
    original_entry: OriginalEntry,
    lock_time: datetime,
    config: dict,
    num_candidates: int = 50,
) -> Tuple[List[SwapCandidate], Dict[int, Tuple[float, float]]]:
    """
    Generate candidate lineups by keeping locked players and optimizing swappable slots.
    
    OPTIMIZED VERSION:
    - Builds simple locked_assignments dict directly from player IDs and slot mapping
    - Uses available players (gametime > lock_time) for unlocked slots
    - Doesn't rebuild complex FieldPlayer objects for locked players
    """
    locked_pids = list(original_entry.locked_player_ids)
    n_swappable = len(original_entry.swappable_player_ids)
    
    if n_swappable == 0:
        # No swaps needed; return original lineup as the only candidate
        return [SwapCandidate(
            original_entry_id=original_entry.entry_id,
            original_lineup_id=original_entry.lineup_id,
            locked_player_ids=original_entry.player_ids,
            swapped_player_ids=[],
            candidate_lineup_id=original_entry.lineup_id
        )], {}
    
    # Map original lineup to slots to determine which slots are locked
    try:
        slot_to_player = _map_players_to_slots(original_entry.player_ids, session)
    except ValueError as e:
        console = Console()
        console.print(f"[red]✗ Failed to map players to slots for entry {original_entry.entry_id}: {e}[/red]")
        return [], {}
    
    # Build locked_assignments: Dict[player_id, slot_name] for locked players only
    locked_assignments: Dict[int, str] = {}
    for slot_name, player_id in slot_to_player.items():
        if player_id in locked_pids:
            locked_assignments[player_id] = slot_name
    
    # Query ALL active players (both locked and unlocked) from DB
    # Locked players will be forced by locked_assignments, unlocked will be optimized
    all_players_db = session.query(NBAPlayer).all()
    
    # Convert to FieldPlayer objects
    all_players: List[FieldPlayer] = []
    fallback_stats: Dict[int, Tuple[float, float]] = {}
    
    for p in all_players_db:
        pos_tuple = _coerce_positions(p.position or [])
        if isinstance(pos_tuple, str):
            pos_tuple = (pos_tuple,)
        elif not isinstance(pos_tuple, tuple):
            pos_tuple = tuple(pos_tuple) if pos_tuple else ()

        team = p.team or ""
        opponent = p.opponent or None
        if team and opponent:
            game_id = "-".join(sorted([team.upper(), opponent.upper()]))
        else:
            game_id = None

        all_players.append(FieldPlayer(
            id=p.id,
            name=p.name,
            salary=p.salary,
            team=team,
            opponent=opponent,
            position=pos_tuple,
            fpts_mean=float(p.fpts_mean) if p.fpts_mean else 0.0,
            game_id=game_id,
            gametime=p.gametime,
        ))
        
        fallback_stats[p.id] = (
            float(p.fpts_mean) if p.fpts_mean is not None else 0.0,
            float(p.fpts_std) if getattr(p, 'fpts_std', None) is not None else 0.0,
        )
    
    # Generate diverse candidates using sample-based optimization
    candidates: List[SwapCandidate] = []
    seen_lineups: Set[Tuple[int, ...]] = set()
    
    # Generate player samples for randomness
    samples = generate_player_samples(
        session=session,
        n_samples=num_candidates * 3,  # Extra samples for diversity
        randomness=0.5,
        snapshot_date=lock_time.date(),
        include_active_players=True,
    )
    
    import random
    from copy import deepcopy
    # Use sample indices in deterministic order; do not shuffle so all samples are used
    sample_indices = list(range(len(next(iter(samples.values())))))
    
    attempts = 0
    max_attempts = num_candidates * 5
    
    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1
        
        # Pick a random sample to create score variation
        samp_idx = sample_indices[attempts % len(sample_indices)]
        
        # Extract base scores for this sample
        base_scores = {pid: float(vec[samp_idx]) for pid, vec in samples.items()}
        
        # Add small jitter for additional diversity
        jitter = 0.05
        jittered_scores = {
            pid: score * (1.0 + random.uniform(-jitter, jitter))
            for pid, score in base_scores.items()
        }
        
        try:
            # Generate lineup with locked assignments
            # The LP solver will enforce lock_time exclusions automatically
            # Relax per-team cap if present to avoid infeasibility when many locks are same team
            cfg_attempt = deepcopy(config) if isinstance(config, dict) else {}
            try:
                cfg_attempt.setdefault("classic", {}).setdefault("lineup_generation", {})
                cfg_attempt["classic"]["lineup_generation"]["max_salary_per_team"] = None
            except Exception:
                pass
            lineup_ids = generate_single_classic_lineup(
                players=all_players,
                base_scores=jittered_scores,
                config=cfg_attempt,
                locked_assignments=locked_assignments,
                lock_time=lock_time,
            )
            
            if not lineup_ids:
                # If optimization failed (infeasible), we skip this attempt
                continue
            
            # Check for uniqueness
            lineup_key = tuple(sorted(lineup_ids))
            if lineup_key in seen_lineups:
                continue
            seen_lineups.add(lineup_key)
            
            # Extract swapped player IDs (those not in locked)
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
            # Log the error but continue trying to generate other candidates
            console.print(f"[red]✗ Error generating candidate for entry {original_entry.entry_id}: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            continue
    
    return candidates, fallback_stats


def _load_and_scale_payouts_from_db(
    session: Session,
    contest_id: int,
    num_field_lineups: int,
    original_field_size: int,
    entry_fee: float,
) -> Dict[int, float]:
    """Scale payouts to desired contest size."""
    from ivan.sports.nba.models import ContestPayout

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
    fallback_stats: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, int, Dict[int, Tuple[float, float]]]:
    """
    Build player FPTS samples matrix.
    
    Returns:
        - P: np.ndarray of shape (n_samples, n_players)
        - ns: actual number of samples
        - missing_players: Dict[player_id, (mean, std)] for players that used fallback
    """
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
    missing_players: Dict[int, Tuple[float, float]] = {}

    rng = np.random.default_rng(42)
    for pid, col in pid_to_idx.items():
        v = raw.get(pid)
        if isinstance(v, np.ndarray):
            vv = v.astype(np.float32, copy=False)
            if vv.shape[0] < ns:
                reps = int(np.ceil(ns / vv.shape[0]))
                vv = np.tile(vv, reps)[:ns]
            elif vv.shape[0] > ns:
                vv = vv[:ns]
            P[:, col] = vv
            continue

        # fallback: use provided (mean, std) if available, else zeros
        if fallback_stats and pid in fallback_stats:
            mu, sd = fallback_stats[pid]
            missing_players[pid] = (mu, sd)
            if sd and sd > 0:
                P[:, col] = rng.normal(loc=mu, scale=sd, size=ns).astype(np.float32)
            else:
                P[:, col] = np.full(ns, float(mu), dtype=np.float32)
        else:
            # leave zeros if nothing known
            missing_players[pid] = (0.0, 0.0)
            P[:, col] = 0.0
    
    return P, ns, missing_players


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


def _print_missing_player_report(
    session: Session,
    missing_players: Dict[int, Tuple[float, float]],
    all_candidates: List[SwapCandidate],
) -> None:
    """
    Print a concise report of missing players with default projections.
    Shows player name, mean/std used, and frequency across all candidate lineups.
    """
    if not missing_players:
        console.print("[green]✓ All players found in active projections (no fallbacks needed)[/green]")
        return
    
    # Count how many times each missing player appears in candidates
    from collections import Counter
    pid_counts = Counter()
    for cand in all_candidates:
        for pid in cand.full_lineup():
            if pid in missing_players:
                pid_counts[pid] += 1
    
    # Get player names from DB
    pids = list(missing_players.keys())
    players_db = session.query(NBAPlayer).filter(NBAPlayer.id.in_(pids)).all()
    pid_to_name = {p.id: p.name for p in players_db}
    
    # Also check ProjectionSnapshot for names of players not in NBAPlayer
    missing_names = set(pids) - set(pid_to_name.keys())
    if missing_names:
        snapshots = session.query(ProjectionSnapshot).filter(ProjectionSnapshot.id.in_(missing_names)).all()
        for snap in snapshots:
            if snap.id not in pid_to_name:
                pid_to_name[snap.id] = snap.player_name
    
    console.print(f"\n[bold yellow]⚠ Missing Player Report ({len(missing_players)} players using fallback projections)[/bold yellow]")
    console.print("─" * 90)
    console.print(f"{'Player':<30} {'ID':<10} {'Mean':<8} {'Std':<8} {'Frequency':<12}")
    console.print("─" * 90)
    
    # Sort by frequency (most common first)
    sorted_pids = sorted(missing_players.keys(), key=lambda p: pid_counts[p], reverse=True)
    
    for pid in sorted_pids:
        mean, std = missing_players[pid]
        name = pid_to_name.get(pid, f"Unknown")
        count = pid_counts[pid]
        total_candidates = len(all_candidates)
        freq_pct = 100.0 * count / total_candidates if total_candidates > 0 else 0.0
        
        console.print(f"{name:<30} {pid:<10} {mean:<8.2f} {std:<8.2f} {count:>4}/{total_candidates:<4} ({freq_pct:>5.1f}%)")
    
    console.print("─" * 90)
    console.print(f"[yellow]Note: These players were assigned projections from ProjectionSnapshot or used default values[/yellow]\n")


def _calculate_and_write_ownership(
    session: Session,
    all_candidates: List[SwapCandidate],
    lock_time: datetime,
) -> None:
    """
    Calculate ownership percentage for all players across candidates and write to database.
    - For unlocked players (gametime > lock_time): Write to NBAPlayer.ownership
    - For locked players: Write to ProjectionSnapshot.ownership (if column exists, otherwise skip)
    """
    from collections import Counter
    
    # Count player appearances across all candidates
    pid_counts = Counter()
    for cand in all_candidates:
        for pid in cand.full_lineup():
            pid_counts[pid] += 1
    
    total_lineups = len(all_candidates)
    if total_lineups == 0:
        return
    
    # Separate locked vs unlocked players
    all_pids = list(pid_counts.keys())
    unlocked_players = session.query(NBAPlayer).filter(
        NBAPlayer.id.in_(all_pids),
        NBAPlayer.gametime > lock_time
    ).all()
    
    unlocked_pids = {p.id for p in unlocked_players}
    locked_pids = set(all_pids) - unlocked_pids
    
    # Update ownership for unlocked players in NBAPlayer table
    unlocked_count = 0
    for player in unlocked_players:
        count = pid_counts.get(player.id, 0)
        ownership_pct = 100.0 * count / total_lineups
        player.ownership = ownership_pct
        unlocked_count += 1
    
    if unlocked_count > 0:
        session.commit()
        console.print(f"[green]✓ Updated ownership for {unlocked_count} unlocked players in NBAPlayer table[/green]")
    
    # Update ownership for locked players in ProjectionSnapshot
    # Check if ownership column exists using SQLAlchemy inspector
    from sqlalchemy import inspect
    inspector = inspect(session.bind)
    
    # Handle both SQLite (no schema) and PostgreSQL (with schema)
    if session.bind.dialect.name == 'sqlite':
        columns = [col['name'] for col in inspector.get_columns('projection_snapshot')]
    else:
        columns = [col['name'] for col in inspector.get_columns('projection_snapshot', schema='nba')]
    
    if 'ownership' not in columns:
        console.print(f"[yellow]⚠ ProjectionSnapshot table does not have 'ownership' column. Skipping ownership write for {len(locked_pids)} locked players.[/yellow]")
        console.print(f"[yellow]  Run: python scripts/nba/add_ownership_to_snapshot.py[/yellow]")
        return
    
    # Constrain to lock date
    day_start = datetime(lock_time.year, lock_time.month, lock_time.day)
    day_end = day_start + timedelta(days=1)
    
    locked_count = 0
    for pid in locked_pids:
        count = pid_counts.get(pid, 0)
        ownership_pct = 100.0 * count / total_lineups
        
        # Find most recent snapshot for this player
        snapshot = (
            session.query(ProjectionSnapshot)
            .filter(
                ProjectionSnapshot.id == pid,
                ProjectionSnapshot.snapshot_date >= day_start,
                ProjectionSnapshot.snapshot_date < day_end,
            )
            .order_by(ProjectionSnapshot.snapshot_date.desc())
            .first()
        )
        
        if snapshot:
            snapshot.ownership = ownership_pct
            locked_count += 1
    
    if locked_count > 0:
        session.commit()
        console.print(f"[green]✓ Updated ownership for {locked_count} locked players in ProjectionSnapshot table[/green]")


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


def parse_entries_csv(
    csv_path: str,
    session: Session,
    quiet: bool = False,
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

    if not quiet:
        console.print(f"[cyan]Parsing entries CSV ({len(df)} rows)...[/cyan]")

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
            if not quiet:
                console.print(f"[yellow]⊘ Entry {entry_id}: blank lineup, skipping[/yellow]")
            continue

        player_ids: List[int] = []
        player_names: List[str] = []
        skip_entry = False

        for pos_idx, pos_col in enumerate(position_cols):
            player_val = row.get(pos_col, '')
            player_str = '' if pd.isna(player_val) else str(player_val).strip()

            if not player_str:
                if not quiet:
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
                        if not quiet:
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
    if not quiet:
        console.print(
            f"[green]✓ Parsed {total_entries} entries across {len(entries_by_contest)} contests[/green]"
        )
    return entries_by_contest


#MAIN ENTRY POINT OF CLI SCRIPT
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
    console = Console()
    contest = session.query(HistoricalContest).filter_by(contest_id=contest_id).one()
    field_sz = int(contest.total_entries)
    entry_fee = float(contest.entry_fee)

    # Identify locked vs swappable
    console.print(f"[bold cyan]Step 1: Identifying locked players for {len(entries)} entries...")
    for entry in entries:
        locked, swappable = _identify_locked_players(session, entry.player_ids, lock_time)
        entry.locked_player_ids = locked
        entry.swappable_player_ids = swappable
        console.print(f"[cyan]  Entry {entry.entry_id}: {len(locked)} locked, {len(swappable)} swappable[/cyan]")
    if not any(e.swappable_player_ids for e in entries):
        console.print(f"[red]✗ All entries are fully locked. No swappable players. Failing.")
        raise RuntimeError("All entries are fully locked. No swappable players.")

    # Generate candidates
    console.print(f"[bold cyan]Step 2: Generating {num_swap_candidates} candidates per entry...[/bold cyan]")
    all_candidates = []
    entry_to_candidates = {}
    fallback_stats_all: Dict[int, Tuple[float, float]] = {}
    for entry in entries:
        candidates, fb_stats = _generate_swap_candidates(
            session=session,
            original_entry=entry,
            lock_time=lock_time,
            config=cfg,
            num_candidates=num_swap_candidates,
        )
        if not candidates:
            console.print(f"[red]✗ No swap candidates generated for entry {entry.entry_id}. Failing.")
            raise RuntimeError(f"No swap candidates generated for entry {entry.entry_id}.")
        entry_to_candidates[entry.entry_id] = candidates
        all_candidates.extend(candidates)
        fallback_stats_all.update(fb_stats)
    console.print(f"[green]✓ Total candidates: {len(all_candidates)}[/green]")
    if not all_candidates:
        console.print(f"[red]✗ No candidates generated for any entry. Failing.")
        raise RuntimeError("No candidates generated for any entry.")

    # Expand for simulation
    console.print(f"[bold cyan]Step 3: Expanding candidates for simulation...")
    candidates_list, lineup_ix, pid_to_idx, starts, ends = _expand_candidates_for_simulation(all_candidates)
    M = len(candidates_list)
    if M == 0:
        console.print(f"[red]✗ No candidates to simulate. Failing.")
        raise RuntimeError("No candidates to simulate.")

    # Scale payouts
    console.print(f"[bold cyan]Step 4: Scaling payouts for contest...")
    scaled = _load_and_scale_payouts_from_db(
        session=session,
        contest_id=contest_id,
        num_field_lineups=M,
        original_field_size=field_sz,
        entry_fee=entry_fee
    )

    # Generate FPTS samples
    console.print(f"[bold cyan]Step 5: Generating {num_sims} FPTS samples...")
    P, num_sims_actual, missing_players = _build_player_samples_matrix(
        session=session,
        pid_to_idx=pid_to_idx,
        n_samples=num_sims,
        fallback_stats=fallback_stats_all,
    )
    if P.shape[0] == 0 or P.shape[1] == 0:
        console.print(f"[red]✗ Failed to generate player samples. Failing.")
        raise RuntimeError("Failed to generate player samples.")
    
    # Print missing player report
    _print_missing_player_report(session, missing_players, all_candidates)
    
    # Calculate and write ownership to database
    console.print(f"[bold cyan]Step 5b: Calculating ownership and writing to database...")
    _calculate_and_write_ownership(session, all_candidates, lock_time)

    # Run simulations
    console.print(f"[bold cyan]Step 6: Running {num_sims_actual} simulations across {M} candidates...")
    import math
    chunk_size = math.ceil(num_sims_actual / min(12, num_sims_actual))
    full_scores = np.zeros((num_sims_actual, M), dtype=np.float32)
    full_payouts = np.zeros_like(full_scores)
    full_rois = np.zeros_like(full_scores)
    full_ranks = np.zeros((num_sims_actual, M), dtype=int)
    from concurrent.futures import ProcessPoolExecutor, as_completed
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
    console.print(f"[bold cyan]Step 7: Computing mean metrics per candidate...")
    for i, candidate in enumerate(candidates_list):
        candidate.mean_fpts = float(full_scores[:, i].mean())
        candidate.mean_payout = float(full_payouts[:, i].mean())
        candidate.mean_roi = float(full_rois[:, i].mean())
        candidate.mean_rank = float(full_ranks[:, i].mean())

    # Select best candidate per entry
    console.print(f"[bold cyan]Step 8: Selecting best candidate per entry (by ROI)...")
    results = []
    for entry in entries:
        entry_candidates = entry_to_candidates.get(entry.entry_id, [])
        if not entry_candidates:
            console.print(f"[red]✗ No candidates for entry {entry.entry_id} after simulation. Failing.")
            raise RuntimeError(f"No candidates for entry {entry.entry_id} after simulation.")
        best = max(entry_candidates, key=lambda c: c.mean_roi if c.mean_roi else -float('inf'))
        entry.best_candidate = best
        results.append((entry.entry_id, best))
        console.print(
            f"[green]  Entry {entry.entry_id}: best ROI={best.mean_roi:.2%}[/green]"
        )
    if not results:
        console.print(f"[red]✗ No results to write for contest {contest_id}. Failing.")
        raise RuntimeError(f"No results to write for contest {contest_id}.")
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
    
    if not rows:
        Console().print(f"[red]✗ No rows to write to {output_path}. Failing.")
        raise RuntimeError(f"No rows to write to {output_path}.")
    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)
    Console().print(f"[green]✓ Wrote {len(rows)} entries to {output_path}")


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


def _sample_field_lineups(
    session: Session,
    num_field_samples: int,
    contest_id: Optional[int] = None,
    config: Optional[dict] = None,
    http_session=None,
    snapshot_date=None,
    lock_time=None,
) -> List[List[int]]:
    """
    Build a list of field lineups (each a list of 8 player IDs).

    - If config[classic.field_source] == 'dk' and contest_id provided, pull from DK standings.
    - Else sample from DB-stored lineups.
    """
    cfg = config or {}
    source = "dk"
    if source == "dk" and contest_id is not None:
        from ivan.sports.nba.sample_field import fetch_dk_field_lineups
        # chrome profile path can be configured under paths.chrome_profile
        chrome_profile = (
            cfg.get("paths", {}).get("chrome_profile")
            or "/home/sam-pratt/ivan/chrome-selenium/dk-profile"
        )
        # Fetch as many as available from DK, we'll downsample if needed
        # IMPORTANT: Pass max_lineups as num_field_samples to sample BEFORE parsing (optimization)
        max_lineups = int(num_field_samples)
        field, _, _ = fetch_dk_field_lineups(
            session=session,
            contest_ids=[int(contest_id)],
            max_lineups=max_lineups,
            user_data_dir=chrome_profile,
            profile_directory="Default",
            sess=http_session,
            snapshot_date=snapshot_date,
            lock_time=lock_time,
            config=cfg,
        )
        # No need to downsample again since we already limited parsing
        return field
    # default: DB lineups
    from ivan.sports.nba.sample_field import fetch_db_field_lineups
    return fetch_db_field_lineups(session, num_field_samples)


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
    quiet: bool = False,
) -> Tuple[List[Tuple['OriginalEntry', List[int], float]], List[Tuple[int, str, int, List[int], float]]]:
    """
    Alternative per-contest flow using sampled field and per-entry candidates.
    Returns list of (entry, best_lineup_ids, roi_mean)
    """
    console = Console()
    if not quiet:
        console.print(f"[cyan]Contest {contest_id}: sampling field (target={num_field_samples})[/cyan]")
    field_lineups = _sample_field_lineups(
        session, 
        num_field_samples, 
        contest_id=contest_id, 
        config=config, 
        http_session=http_session,
        snapshot_date=lock_time.date(),
        lock_time=lock_time,
    )
    if not field_lineups:
        console.print(f"[red]✗[/red] No field lineups sampled for contest {contest_id}")
        raise RuntimeError(f"No field lineups sampled for contest {contest_id}")
    
    available = len(field_lineups)
    effective_target = min(int(num_field_samples), available)
    if not quiet:
        if available > int(num_field_samples):
            console.print(f"[green]✓ Field sampled: {effective_target}/{available} (capped)" )
        elif available < int(num_field_samples):
            console.print(f"[yellow]⚠ Field sampled: {available} (< requested {num_field_samples})" )
        else:
            console.print(f"[green]✓ Field sampled: {available} lineups" )

    # Print Top-10 owned players in the sampled field
    if not quiet:
        try:
            from collections import Counter
            pid_counts = Counter(pid for lu in field_lineups for pid in lu)
            total_lu = len(field_lineups)
            top5 = pid_counts.most_common(5)
            if top5:
                pids = [pid for pid, _ in top5]
                players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(pids)).all()
                pmap = {p.id: p for p in players}
                console.print("[cyan]Top owned (5) preview:[/cyan]")
                for pid, cnt in top5:
                    p = pmap.get(pid)
                    name = p.name if p else f"PID {pid}"
                    pct = 100.0 * cnt / total_lu
                    console.print(f"  {name:<22} {pct:6.2f}%")
        except Exception:
            pass

    # Player samples include same-day snapshots to cover locked players
    if not quiet:
        console.print(f"[cyan]Building player samples (T={num_sims})[/cyan]")
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
    if not quiet:
        console.print(f"[cyan]Precomputing field scores[/cyan]")
    field_scores = np.stack([_sum_lineup_scores(player_samples, lu) for lu in field_lineups], axis=0)

    # Payouts
    prize_per_rank, entry_fee, entries_max = _load_payouts_from_db(session, contest_id)

    results: List[Tuple['OriginalEntry', List[int], float]] = []
    candidate_rows: List[Tuple[int, str, int, List[int], float]] = []  # (contest_id, entry_id, idx, lineup_ids, roi)

    if not quiet:
        console.print(f"[cyan]Generating candidates & simulating ({len(entries)} entries)")
    for entry in entries:
        # Generate K candidates for this entry (reuse existing core logic)
        candidates, _ = _generate_swap_candidates(
            session=session,
            original_entry=entry,
            lock_time=lock_time,
            config=config,
            num_candidates=num_candidates_per_entry,
        )
        if not candidates:
            if not quiet:
                console.print(f"[yellow]![/yellow] No candidates for entry {entry.entry_id}; skipping")
            continue

        # Candidate lineups (complete: locked + swapped)
        candidate_lineups = [c.full_lineup() for c in candidates]
        candidate_scores = [_sum_lineup_scores(player_samples, lu) for lu in candidate_lineups]

        # ROI mean per candidate
        rois = _evaluate_candidates_vs_field(
            field_scores=field_scores,
            candidate_scores=candidate_scores,
            prize_per_rank=prize_per_rank,
            entry_fee=entry_fee,
            entries_max=entries_max,
        )

        # Emit per-candidate rows (injected-vs-field ROI)
        for idx, (lu, rmean) in enumerate(zip(candidate_lineups, rois), start=1):
            candidate_rows.append((contest_id, entry.entry_id, idx, lu, float(rmean)))

        # Pick best
        best_idx = int(np.argmax(rois))
        results.append((entry, candidate_lineups[best_idx], float(rois[best_idx])))

    if not quiet:
        console.print(f"[green]✓ Contest {contest_id} complete ({len(results)} entries processed)")
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
    quiet: bool = False,
) -> Tuple[List[Tuple['OriginalEntry', List[int], float]], List[Tuple[int, str, int, List[int], float]]]:
    """Slate-level coordinator: groups by contest and runs late swap for each."""
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
            quiet=quiet,
        )
        all_results.extend(result)
        all_candidate_rows.extend(cand_rows)
    if not quiet:
        Console().print(f"[green]✓ Slate complete: {len(all_results)} optimized entries across {len(by_contest)} contests")
    # Print player exposures across chosen candidate lineups (match scripts/nba/choose.py style)
    try:

        pid_counts = Counter(pid for _entry, lineup_ids, _roi in all_results for pid in lineup_ids)
        total_selected = len(all_results)

        if total_selected == 0:
            console.print("[yellow]⚠ No chosen lineups to compute exposures[/yellow]")
        else:
            console.print(f"\n[bold cyan]Player exposures across {total_selected} chosen lineups[/bold cyan]")
            console.print("─" * 70)
            console.print(f"{'Player':<30} {'ID':<8} {'Count':<8} {'Exposure':<10}")
            console.print("─" * 70)

            # Lookup names for pids
            pids = list(pid_counts.keys())
            players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(pids)).all()
            pid_to_name = {p.id: p.name for p in players}

            for pid, cnt in sorted(pid_counts.items(), key=lambda kv: kv[1], reverse=True):
                name = pid_to_name.get(pid, f"PID {pid}")
                pct = 100.0 * cnt / total_selected
                console.print(f"{name:<30} {pid:<8} {cnt:<8} {pct:6.2f}%")

            console.print("─" * 70)
    except Exception as e:
        console.print(f"[yellow]⚠ Could not compute exposures: {e}[/yellow]")

    return all_results, all_candidate_rows


def write_lateswap_csv_simple(
    results: List[Tuple['OriginalEntry', List[int], float]],
    output_path: str,
    quiet: bool = False,
) -> None:
    """Minimal CSV: entry_id, 8 player ids in roster order, roi_mean."""
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entry_id", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "roi_mean"])
        for entry, lineup_ids, roi_mean in results:
            row = [entry.entry_id] + lineup_ids + [roi_mean]
            w.writerow(row)
    if not quiet:
        Console().print(f"[green]✓[/green] Wrote late-swap CSV: {output_path}")


def write_lateswap_candidates_csv(
    candidate_rows: List[Tuple[int, str, int, List[int], float]],
    output_path: str,
    quiet: bool = False,
) -> None:
    """Write all candidate lineups with ROI to CSV: contest_id, entry_id, idx, p1..p8, roi_mean."""
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contest_id", "entry_id", "candidate_index", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "roi_mean"])
        for contest_id, entry_id, idx, lineup_ids, roi_mean in candidate_rows:
            row = [contest_id, entry_id, idx] + lineup_ids + [roi_mean]
            w.writerow(row)
    if not quiet:
        Console().print(f"[green]✓[/green] Wrote candidate ROI CSV: {output_path}")


def write_lateswap_csv_entries_format(
    results: List[Tuple['OriginalEntry', List[int], float]],
    output_path: str,
    session: Session,
    quiet: bool = False,
) -> None:
    """
    Write late-swap results to match the DraftKings entries CSV format.

    Columns (first 12 to match parser expectations):
    - Entry ID, Contest Name, Contest ID, Entry Fee, PG, SG, SF, PF, C, G, F, UTIL

    Values for PG..UTIL will be in "Name (ID)" format to mirror the native export.
    """
    # Header exactly as DK export
    headers = [
        "Entry ID",
        "Contest Name",
        "Contest ID",
        "Entry Fee",
        "PG",
        "SG",
        "SF",
        "PF",
        "C",
        "G",
        "F",
        "UTIL",
    ]

    # Preload contest metadata to avoid repeated queries
    from ivan.sports.nba.models import HistoricalContest, NBAPlayer
    from collections import Counter
    contest_ids = sorted({int(e.contest_id) for e, _, _ in results})
    contests = (
        session.query(HistoricalContest)
        .filter(HistoricalContest.contest_id.in_(contest_ids))
        .all()
    )
    contest_map = {int(c.contest_id): c for c in contests}

    # Preload all player names referenced
    all_pids = sorted({pid for _, lineup_ids, _ in results for pid in lineup_ids})
    players = (
        session.query(NBAPlayer)
        .filter(NBAPlayer.id.in_(all_pids))
        .all()
    )
    pid_to_name = {int(p.id): (p.name or str(p.id)) for p in players}

    # Write CSV
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for entry, lineup_ids, _roi_mean in results:
            meta = contest_map.get(int(entry.contest_id))
            contest_name = meta.name if meta and getattr(meta, "name", None) else ""
            # Format entry fee with $ and two decimals to match DK export
            try:
                fee_val = float(meta.entry_fee) if meta and getattr(meta, "entry_fee", None) is not None else 0.0
            except Exception:
                fee_val = 0.0
            entry_fee_str = f"${fee_val:.2f}"

            # Build Name (ID) strings in slot order
            pos_strings = [
                f"{pid_to_name.get(pid, str(pid))} ({pid})" for pid in lineup_ids
            ]

            row = [
                entry.entry_id,
                contest_name,
                int(entry.contest_id),
                entry_fee_str,
                *pos_strings,
            ]
            w.writerow(row)

    if not quiet:
        Console().print(f"[green]✓[/green] Wrote DK entries-format CSV: {output_path}")




