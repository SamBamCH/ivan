#!/usr/bin/env python3
"""
Ownership projection using field sampling simulations.

Strategy:
1. Detect slate type (showdown vs classic) from NFLPlayer roster_slot_id
2. Generate synthetic field using sample_sd_field or sample_classic_field
3. Aggregate player exposures across the simulated field
4. Write ownership percentages back to NFLPlayer table
"""

from typing import Dict, Tuple
import logging
import numpy as np
from collections import Counter
from sqlalchemy.orm import Session

from ivan.sports.nfl.models import NFLPlayer, Lineup
from ivan.sports.nfl.sample_field import (
    sample_showdown_field_learned,
    sample_classic_field,
    build_lineup_matrix,
)

logger = logging.getLogger(__name__)


def _detect_slate_type(session: Session) -> str:
    """
    Detect slate type from roster_slot_id values.
    
    Returns:
        'showdown' if CPT/FLEX slots (511/512) present, else 'classic'
    """
    slots = set(
        session.query(NFLPlayer.roster_slot_id)
        .distinct()
        .filter(NFLPlayer.roster_slot_id.isnot(None))
        .all()
    )
    slot_ids = {s[0] for s in slots if s[0] is not None}
    
    # Showdown slots: CPT=511, FLEX=512
    if 511 in slot_ids or 512 in slot_ids:
        return 'showdown'
    return 'classic'


def _coerce_player_ids(val) -> list[int]:
    """Parse JSON player_ids_list from database."""
    import json
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


def _compute_exposures_from_counts(
    session: Session,
    counts_by_lineup: Dict[int, int],
    total_entries: int,
) -> Dict[int, float]:
    """
    Compute player ownership percentages from lineup counts.
    
    Args:
        session: Database session
        counts_by_lineup: {lineup_id: duplicate_count}
        total_entries: Total field size
    
    Returns:
        {player_id: ownership_pct} where ownership_pct is 0-100
    """
    # Get all unique lineup IDs that were selected
    selected_lineup_ids = [lid for lid, count in counts_by_lineup.items() if count > 0]
    
    if not selected_lineup_ids:
        logger.warning("No lineups selected in field simulation")
        return {}
    
    # Fetch player_ids for all selected lineups
    # Use batching to avoid SQLite variable limit (999)
    BATCH_SIZE = 900
    player_exposure_counts = Counter()
    
    for i in range(0, len(selected_lineup_ids), BATCH_SIZE):
        batch = selected_lineup_ids[i:i + BATCH_SIZE]
        rows = (
            session.query(Lineup.id, Lineup.player_ids_list)
            .filter(Lineup.id.in_(batch))
            .all()
        )
        
        for lineup_id, player_ids_json in rows:
            player_ids = _coerce_player_ids(player_ids_json)
            dupe_count = counts_by_lineup[lineup_id]
            
            # Each player in this lineup appears `dupe_count` times
            for pid in player_ids:
                player_exposure_counts[pid] += dupe_count
    
    # Convert counts to ownership percentages
    ownership_pct = {
        pid: (count / total_entries) * 100.0
        for pid, count in player_exposure_counts.items()
    }
    
    return ownership_pct


def _write_ownership_to_db(
    session: Session,
    ownership_by_pid: Dict[int, float],
) -> int:
    """
    Write ownership percentages to NFLPlayer table.
    
    Creates/updates 'ownership_proj' column (add to models.py if needed).
    
    Args:
        session: Database session
        ownership_by_pid: {player_id: ownership_pct (0-100)}
    
    Returns:
        Number of players updated
    """
    updated = 0
    
    # First, set all players to 0.0 ownership (those not in field)
    session.query(NFLPlayer).update(
        {NFLPlayer.ownership_proj: 0.0},
        synchronize_session=False
    )
    
    # Then update players that appeared in field
    for pid, own_pct in ownership_by_pid.items():
        result = (
            session.query(NFLPlayer)
            .filter(NFLPlayer.id == pid)
            .update({NFLPlayer.ownership_proj: float(own_pct)})
        )
        updated += result
    
    session.commit()
    return updated


def project_ownership(
    session: Session,
    *,
    config: dict = None,
    num_simulations: int = 1,
    aggregate_method: str = 'mean',
) -> Dict[int, float]:
    """
    Project player ownership using field simulation.
    
    Strategy:
    1. Detect slate type (showdown/classic)
    2. Check if we have generated lineups in database
    3. Run field sampling simulation(s)
    4. Aggregate ownership across simulations
    5. Write ownership back to NFLPlayer.ownership_proj
    
    Args:
        session: Database session
        field_size: Size of simulated field (default 10k)
        config: Configuration dict (optional)
        num_simulations: Number of field samples to average (default 1)
        aggregate_method: How to combine sims ('mean', 'median', 'max')
    
    Returns:
        {player_id: ownership_pct} for all players with non-zero ownership
    """
    config = config or {}
    
    # 1. Detect slate type
    slate_type = _detect_slate_type(session)
    logger.info(f"[ownership] Detected slate type: {slate_type}")
    
    # 2. Check if lineups exist
    lineup_count = session.query(Lineup).count()
    if lineup_count == 0:
        logger.warning("[ownership] No lineups found in database. Cannot project ownership.")
        logger.warning("[ownership] Run 'python scripts/nfl/generate_lineups.py' first.")
        return {}
    
    logger.info(f"[ownership] Found {lineup_count:,} lineups in database")
    
    # 3. Run field simulation(s)
    all_ownership = []
    
    for sim_idx in range(num_simulations):
        logger.info(f"[ownership] Running field simulation {sim_idx + 1}/{num_simulations}")
        
        try:
            if slate_type == 'showdown':
                field_size=50000
                counts_by_lineup, meta = sample_showdown_field_learned(
                    session=session,
                    field_size=field_size,
                    config=config,
                )
            else:  # classic
                field_size=10000
                counts_by_lineup, meta = sample_classic_field(
                    session=session,
                    field_size=field_size,
                    config=config,
                )
            
            # Compute exposures for this simulation
            ownership = _compute_exposures_from_counts(
                session=session,
                counts_by_lineup=counts_by_lineup,
                total_entries=field_size,
            )
            
            all_ownership.append(ownership)
            
            # Log summary stats
            if ownership:
                values = list(ownership.values())
                logger.info(
                    f"[ownership]   Sim {sim_idx + 1}: "
                    f"{len(ownership)} players, "
                    f"ownership range [{min(values):.2f}%, {max(values):.2f}%], "
                    f"mean={np.mean(values):.2f}%"
                )
        
        except Exception as e:
            logger.error(f"[ownership] Simulation {sim_idx + 1} failed: {e}")
            continue
    
    if not all_ownership:
        logger.error("[ownership] All simulations failed")
        return {}
    
    # 4. Aggregate across simulations
    all_player_ids = set()
    for own_dict in all_ownership:
        all_player_ids.update(own_dict.keys())
    
    aggregated_ownership = {}
    
    for pid in all_player_ids:
        values = [own.get(pid, 0.0) for own in all_ownership]
        
        if aggregate_method == 'median':
            aggregated_ownership[pid] = float(np.median(values))
        elif aggregate_method == 'max':
            aggregated_ownership[pid] = float(np.max(values))
        else:  # mean (default)
            aggregated_ownership[pid] = float(np.mean(values))
    
    logger.info(
        f"[ownership] Aggregated {num_simulations} simulation(s) using {aggregate_method}"
    )
    
    # 5. Write to database
    updated = _write_ownership_to_db(session, aggregated_ownership)
    logger.info(f"[ownership] ✓ Updated ownership for {updated} players")
    
    # Log top owned players for validation
    top_owned = sorted(aggregated_ownership.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("[ownership] Top 10 projected ownership:")
    for pid, own in top_owned:
        player = session.query(NFLPlayer).get(pid)
        if player:
            logger.info(f"[ownership]   {player.name:30s} {own:6.2f}%")
    
    return aggregated_ownership


def clear_ownership(session: Session) -> int:
    """
    Clear all ownership projections (set to 0.0).
    
    Useful for starting fresh on a new slate.
    
    Returns:
        Number of players cleared
    """
    result = session.query(NFLPlayer).update(
        {NFLPlayer.ownership_proj: 0.0},
        synchronize_session=False
    )
    session.commit()
    logger.info(f"[ownership] Cleared ownership for {result} players")
    return result