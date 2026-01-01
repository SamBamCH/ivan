from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple
import re

from ivan.sports.nba.models import HistoricalContest, ClassicContestMetadata, ProjectionSnapshot
from ivan.core.utils import normalize_name

BINS = 20
PROJECTION_MIN = 200.0  # Fixed lower bound for projection normalization

def _compute_histograms(
    projections: np.ndarray, 
    ownerships: np.ndarray,
    n_bins: int = BINS
) -> Tuple[List[float], List[float]]:
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    proj_counts, _ = np.histogram(projections, bins=bin_edges)
    own_counts, _  = np.histogram(ownerships, bins=bin_edges)

    proj_counts = proj_counts.astype(float)
    own_counts  = own_counts.astype(float)

    if proj_counts.sum() > 0:
        proj_counts = proj_counts / proj_counts.sum()
    if own_counts.sum() > 0:
        own_counts = own_counts / own_counts.sum()

    return proj_counts.tolist(), own_counts.tolist()


def _parse_classic_lineup_string(lineup_str: str) -> List[str]:
    """Parse NBA classic lineup format: 'C Nic Claxton F Kyle Filipowski G Andrew Nembhard...'"""
    if pd.isna(lineup_str) or not isinstance(lineup_str, str):
        return []
    
    # Split by NBA position labels
    parts = re.split(r'\s*(C|F|G|PF|PG|SF|SG|UTIL)\s+', lineup_str)
    
    players = []
    # After split, pattern is: [text_before, POS1, player1, POS2, player2, ...]
    # We want to extract the player names (indices 2, 4, 6, ...)
    for i in range(2, len(parts), 2):
        if i < len(parts):
            player_name = parts[i].strip().rstrip(',')  # Remove trailing comma and whitespace
            if player_name:  # Only add non-empty names
                players.append(player_name)
    
    return players


def _get_snapshot_data_for_contest(
    session: Session,
    contest_date: datetime,
    player_names: List[str],
) -> Dict[str, Dict]:
    
    # Convert contest_date (stored in UTC) to local (US/Eastern) date for matching snapshots.
    # Assumptions:
    # - ProjectionSnapshot.snapshot_date is stored as naive local (US/Eastern) timestamps.
    # - hc.date is UTC (aware or naive representing UTC).
    eastern = ZoneInfo("US/Central")
    if isinstance(contest_date, datetime):
        dt = contest_date
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone(eastern)
        target_date = local_dt.date()
    else:
        target_date = contest_date  # already a date
    
    print(f"DEBUG: Querying ProjectionSnapshot with {len(player_names)} names, date={target_date}")
    
    # Use func.date() to compare only the date portion of snapshot_date
    # Build inclusive start (00:00:00) and exclusive end (next day) datetimes
    # Build naive start/end (local) datetimes for half-open day range
    start_dt = datetime.combine(target_date, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)

    results = (
        session.query(ProjectionSnapshot)
        .filter(
            ProjectionSnapshot.snapshot_date >= start_dt,
            ProjectionSnapshot.snapshot_date < end_dt,
            ProjectionSnapshot.player_name.in_(player_names)
        )
        .all()
    )
    print(f"DEBUG: Query returned {len(results)} results")
    snapshot_by_player = {}
    for snap in results:
        if snap.player_name not in snapshot_by_player:  # Changed from normalized_name
            snapshot_by_player[snap.player_name] = {}
        
        snapshot_by_player[snap.player_name] = {  # Changed from normalized_name
            'fpts_mean': snap.fpts_mean,
            'ownership': snap.ownership
        }

    return snapshot_by_player
    

def process_classic_contest(
        session: Session,
        hc: HistoricalContest,
        df: pd.DataFrame
) -> bool:
    contest_id = hc.contest_id

    base_df = df.iloc[:, :-4].copy() #main lineup data
    own_df  = df.iloc[:, -4:].copy() #ownership table
    
    player_base_info = {}
    for _, row in own_df.iterrows():
        player_name = row.get("Player")
        if pd.isna(player_name):
            continue

        norm_name = normalize_name(player_name)
        ownership_actual = float(str(row.get("%Drafted", "0%")).replace("%", ""))
        fpts_actual = row.get("FPTS")

        player_base_info[norm_name] = {
            'name': norm_name,
            'ownership_actual': ownership_actual, 
            'fpts_actual': float(fpts_actual) if pd.notna(fpts_actual) else None,
        }

    all_player_names = [info['name'] for info in player_base_info.values()]
    snapshot_data = _get_snapshot_data_for_contest(
        session=session,
        contest_date=hc.date,
        player_names=all_player_names
    )

    # Safeguard 1: Check player coverage (require 90%+ players have snapshot data)
    players_with_data = len(snapshot_data)
    total_players = len(all_player_names)
    coverage_pct = (players_with_data / total_players * 100) if total_players > 0 else 0.0
    
    if coverage_pct < 75.0:
        print(f"[NBA][{contest_id}] SKIPPED: Player coverage {coverage_pct:.1f}% < 90% ({players_with_data}/{total_players})")
        return False
    
    print(f"[NBA][{contest_id}] Player coverage: {coverage_pct:.1f}% ({players_with_data}/{total_players})")

    lineup_projections = []
    lineup_ownerships = []

    for idx, row in base_df.iterrows():
        lineup_str = row["Lineup"] if "Lineup" in row and pd.notna(row["Lineup"]) else ""
        player_names = _parse_classic_lineup_string(lineup_str)
        if not player_names:
            continue

        proj_sum    = 0.0
        own_sum     = 0.0
        valid       = True

        for player_name in player_names:
            norm_name = normalize_name(player_name)
            snap = snapshot_data.get(norm_name)
            actuals = player_base_info.get(norm_name)
            if not snap:
                valid = False
                break
            
            proj_sum += snap.get('fpts_mean', 0.0)
            own_sum  += actuals.get('ownership_actual', 0.0)

        if valid:
            lineup_projections.append(proj_sum)
            lineup_ownerships.append(own_sum)

    if not lineup_projections:
        return False
    
    projections = np.array(lineup_projections, dtype=float)
    ownerships  = np.array(lineup_ownerships, dtype=float)

    # Filter out lineups below the fixed projection minimum
    mask = projections >= PROJECTION_MIN
    dropped = int((~mask).sum())
    total_lineups_before_filter = len(projections)
    
    # Safeguard 2: Check lineup threshold (skip if 10%+ lineups below threshold)
    dropped_pct = (dropped / total_lineups_before_filter * 100) if total_lineups_before_filter > 0 else 0.0
    
    if dropped_pct >= 10.0:
        print(f"[NBA][{contest_id}] SKIPPED: {dropped_pct:.1f}% of lineups below threshold ({dropped}/{total_lineups_before_filter})")
        return False
    
    if dropped > 0:
        print(f"[NBA][{contest_id}] Dropped {dropped} lineups ({dropped_pct:.1f}%) with proj < {PROJECTION_MIN}")
    
    projections = projections[mask]
    ownerships  = ownerships[mask]

    if projections.size == 0:
        return False

    proj_max = float(projections.max())
    own_max  = float(ownerships.max()) if ownerships.size else 0.0

    # Normalize projections to [0,1] using fixed min and observed max
    denom = max(proj_max - PROJECTION_MIN, 1e-6)
    proj_normalized = np.clip((projections - PROJECTION_MIN) / denom, 0.0, 1.0)

    # Normalize ownership sums to [0,1] based on observed max
    if own_max > 0:
        own_normalized = np.clip(ownerships / own_max, 0.0, 1.0)
    else:
        own_normalized = np.zeros_like(ownerships)

    hist_proj_counts, hist_own_counts = _compute_histograms(proj_normalized, own_normalized)

    metadata = ClassicContestMetadata(
        contest_id=contest_id,
        max_projection=proj_max,
        max_ownership=own_max,
        hist_proj_counts=hist_proj_counts,
        hist_own_counts=hist_own_counts
    )

    try:
        session.add(metadata)
        session.commit()
        return True

    except Exception as e:
        session.rollback()
        return False


    
