"""
Lineup generation and management for NBA Classic contests.

This module provides:
- generate_lineups: Generate multiple unique lineups and store to DB
- deduplicate_lineups: Remove duplicate lineups from database
"""

import random
from datetime import datetime
from typing import List, Optional, Callable, Dict
from multiprocessing import Process, Queue
from queue import Empty

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from sqlalchemy.orm import Session

from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import NBAPlayer, Lineup
from ivan.sports.nba.utils import _coerce_positions, _cfg
from ivan.sports.nba.pre_lock.lineup_generation import (
    FieldPlayer,
    generate_single_classic_lineup,
    SLOT_NAMES,
)
from ivan.sports.nba.pre_lock.player import generate_player_samples

BATCH_COMMIT_SIZE = 100
FPTS_MIN = 15

def _build_player_pool(session: Session) -> List[FieldPlayer]:
    """Build FieldPlayer pool from NBAPlayer table."""
    raw_players = session.query(NBAPlayer).filter(NBAPlayer.fpts_mean > FPTS_MIN).all()
    if not raw_players:
        return []
    
    players: List[FieldPlayer] = []
    for rp in raw_players:
        pos = getattr(rp, "position", None)
        pos_tuple = _coerce_positions(pos or [])

        if isinstance(pos_tuple, str):
            pos_tuple = (pos_tuple,)
        elif not isinstance(pos_tuple, tuple):
            pos_tuple = tuple(pos_tuple) if pos_tuple else ()
        
        team = getattr(rp, "team", "")
        opponent = getattr(rp, "opponent", None)
        if team and opponent:
            game_id = "-".join(sorted([team.upper(), opponent.upper()]))
        else:
            game_id = None
        
        players.append(FieldPlayer(
            id=rp.id,
            name=getattr(rp, "name"),
            salary=getattr(rp, "salary"),
            team=team,
            opponent=opponent,
            position=pos_tuple,
            fpts_mean=float(getattr(rp, "fpts_mean")),
            game_id=game_id,
            gametime=getattr(rp, "gametime", None)
        ))
    
    return players


def _generate_lineups_worker(
    worker_id: int,
    num_lineups: int,
    n_samples: int,
    randomness: float,
    jitter: float,
    min_fpts: Optional[float],
    q: Queue,
) -> None:
    """
    Worker function for parallel lineup generation with progress reporting.
    
    Each worker creates its own DB session and generates lineups independently.
    Sends progress updates via Queue.
    """
    engine = init_db('nba')
    session = get_session(engine)
    
    q.put(("start", worker_id, num_lineups, f"samples={n_samples}"))
    
    try:
        players = _build_player_pool(session)
        if not players:
            q.put(("done", worker_id, 0, "no players"))
            return
        
        samples = generate_player_samples(
            session=session,
            n_samples=n_samples,
            randomness=randomness
        )
        
        sample_indices = list(range(n_samples))
        
        generated: List[List[int]] = []
        seen_lineups: set = set()
        
        sample_cursor = worker_id * 1000  # Offset to reduce collision between workers
        max_attempts = num_lineups * 10
        attempts = 0
        last_reported = 0
        
        while len(generated) < num_lineups and attempts < max_attempts:
            attempts += 1
            samp_idx = sample_indices[sample_cursor % n_samples]
            sample_cursor += 1

            base_scores = {pid: float(vec[samp_idx]) for pid, vec in samples.items()}

            # Add random jitter
            jittered_scores = {
                pid: score * (1.0 + random.uniform(-jitter, jitter))
                for pid, score in base_scores.items()
            }

            ids_selected = generate_single_classic_lineup(
                players=players,
                base_scores=jittered_scores,
                config={},
                min_fpts=min_fpts,
            )
            
            if not ids_selected:
                continue
                
            key = tuple(sorted(ids_selected))
            if key in seen_lineups:
                continue
            seen_lineups.add(key)
            
            generated.append(ids_selected)
            
            # Report progress
            current = len(generated)
            if current > last_reported:
                q.put(("advance", worker_id, current - last_reported, f"created={current}"))
                last_reported = current
            
            # Store to DB in batches
            if len(generated) % BATCH_COMMIT_SIZE == 0:
                for lineup in generated[-BATCH_COMMIT_SIZE:]:
                    session.add(Lineup(player_ids_list=list(lineup), created_at=datetime.utcnow()))
                session.commit()
        
        # Store remaining lineups
        remaining_start = (len(generated) // BATCH_COMMIT_SIZE) * BATCH_COMMIT_SIZE
        for lineup in generated[remaining_start:]:
            session.add(Lineup(player_ids_list=list(lineup), created_at=datetime.utcnow()))
        session.commit()
        
        q.put(("done", worker_id, len(generated), f"created={len(generated)}"))
        
    except Exception as e:
        q.put(("done", worker_id, 0, f"error: {e}"))
    finally:
        session.close()


def _run_with_progress(quotas: List[int], n_samples: int, randomness: float, jitter: float, min_fpts: Optional[float]) -> int:
    """Run workers with Rich progress display."""
    q = Queue()
    procs = []
    
    for proc_no, qnty in enumerate(quotas):
        p = Process(
            target=_generate_lineups_worker,
            args=(proc_no, qnty, n_samples, randomness, jitter, min_fpts, q)
        )
        p.start()
        procs.append(p)

    n_workers = len(procs)
    total_created = 0

    console = Console()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=True,
        refresh_per_second=4,
    ) as progress:
        task_id = {}
        done_count = 0

        while done_count < n_workers:
            try:
                kind, proc_no, val, note = q.get(timeout=0.25)
            except Empty:
                continue

            if kind == "start":
                desc = f"Worker {proc_no} ({note})"
                task_id[proc_no] = progress.add_task(desc, total=int(val))
            elif kind == "advance":
                tid = task_id.get(proc_no)
                if tid is None:
                    tid = task_id[proc_no] = progress.add_task(
                        f"Worker {proc_no}", total=int(val)
                    )
                progress.advance(tid, int(val))
            elif kind == "done":
                tid = task_id.get(proc_no)
                if tid is not None:
                    total = progress.tasks[tid].total or int(val)
                    progress.update(
                        tid,
                        completed=total,
                        description=f"Worker {proc_no} ✅ {note}",
                    )
                total_created += int(val)
                done_count += 1

        progress.stop()
    
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    q.close()
    q.join_thread()
    
    return total_created


def generate_lineups(
    num_lineups: int = 25000,
    num_jobs: int = 8,
    min_fpts: Optional[float] = None,
    n_samples: int = 5000,
    randomness: float = 1.0,
    jitter: float = 0.05,
    batch_size: int = 100,
) -> int:
    """
    Generate multiple unique NBA Classic lineups and store to database.
    
    Uses parallel processing with Rich progress display.
    
    Args:
        num_lineups: Total number of lineups to generate
        num_jobs: Number of parallel workers
        min_fpts: Optional minimum total projected FPTS filter
        n_samples: Number of Monte Carlo samples per player
        randomness: Randomness factor for sampling (0-1)
        jitter: Random jitter to add to scores (0-1)
        batch_size: DB commit batch size (unused, kept for API compat)
        
    Returns:
        Number of lineups successfully created
    """
    global BATCH_COMMIT_SIZE
    BATCH_COMMIT_SIZE = batch_size
    
    # Calculate quotas per worker
    base = num_lineups // num_jobs
    remainder = num_lineups % num_jobs
    quotas = [base + (1 if i < remainder else 0) for i in range(num_jobs)]
    
    # Run with progress display
    created = _run_with_progress(quotas, n_samples, randomness, jitter, min_fpts)
    
    print(f"\n✅ Generated {created:,} lineups across {num_jobs} workers")
    return created


def deduplicate_lineups() -> int:
    """
    Remove duplicate lineups from the database.
    
    Returns:
        Number of duplicate lineups removed
    """
    engine = init_db('nba')
    session = get_session(engine)
    
    all_lineups = session.query(Lineup).all()
    
    seen: Dict[tuple, int] = {}  # key -> first lineup id
    to_delete: List[int] = []
    
    for lineup in all_lineups:
        pids = lineup.player_ids_list
        if isinstance(pids, str):
            import json
            try:
                pids = json.loads(pids)
            except:
                pids = []
        pids = [int(x) for x in (pids or []) if x is not None]
        key = tuple(sorted(pids))
        
        if key in seen:
            to_delete.append(lineup.id)
        else:
            seen[key] = lineup.id
    
    if to_delete:
        # Delete in batches to avoid SQLite limits
        BATCH_SIZE = 900
        for i in range(0, len(to_delete), BATCH_SIZE):
            batch = to_delete[i:i + BATCH_SIZE]
            session.query(Lineup).filter(Lineup.id.in_(batch)).delete(synchronize_session=False)
        session.commit()
    
    print(f"✓ Removed {len(to_delete)} duplicate lineups, {len(seen)} unique remain")
    return len(to_delete)


# Re-export SLOT_NAMES for convenience
__all__ = ['generate_lineups', 'deduplicate_lineups', 'SLOT_NAMES', 'FieldPlayer']
