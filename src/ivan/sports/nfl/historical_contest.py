import re
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from ivan.database.session import init_db, get_session
from ivan.sports.common.draftkings import build_requests_session_from_chrome, download_contest_dataframe
from ivan.core.utils import normalize_name
from ivan.sports.nfl.models import (
    HistoricalContest,
    ShowdownPlayer,
    ShowdownLineup,
    ShowdownContestMetadata,
    ClassicContestMetadata,
    ProjectionSnapshot
)

log = logging.getLogger(__name__)

_SHOWDOWN_LINEUP_REGEX = re.compile(r"(CPT|FLEX)\s+([A-Za-z\s\.''-]+?)(?=\s+(?:CPT|FLEX)|$)")

# ==============================================================================
# HELPER FUNCTIONS FOR CLASSIC CONTEST PROCESSING
# ==============================================================================

BINS = 20  # Fixed-width histogram bins

def calculate_gini(values: List[float]) -> float:
    """Calculate Gini coefficient for ownership concentration."""
    if not values or len(values) < 2:
        return 0.0
    sorted_values = np.array(sorted(values))
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n


def compute_histograms(
    projections: np.ndarray,
    ownerships: np.ndarray,
    n_bins: int = BINS
) -> Tuple[List[float], List[float]]:
    """
    Compute normalized histogram distributions for projections and ownerships.

    Both should be normalized to [0, 1] BEFORE calling this function.

    Args:
        projections: Array of normalized [0-1] lineup projection sums
        ownerships: Array of normalized [0-1] lineup ownership sums
        n_bins: Number of fixed-width bins (default 20)

    Returns:
        (hist_proj_counts, hist_own_counts): Two lists of length n_bins, each summing to 1.0
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Compute histograms
    proj_counts, _ = np.histogram(projections, bins=bin_edges)
    own_counts, _ = np.histogram(ownerships, bins=bin_edges)

    # Normalize to sum to 1.0
    proj_counts = proj_counts.astype(float)
    own_counts = own_counts.astype(float)

    if proj_counts.sum() > 0:
        proj_counts = proj_counts / proj_counts.sum()
    if own_counts.sum() > 0:
        own_counts = own_counts / own_counts.sum()

    return proj_counts.tolist(), own_counts.tolist()

def get_snapshot_data_for_contest(
    session: Session,
    contest_date: datetime,
    player_names: List[str],
    player_slots: Dict[str, str] = None,  # Make optional, default None for classic
    roster_slot_filter: List[int] = None,  # NEW: allow filtering by slot IDs
) -> Dict[str, Dict]:
    """
    Get historical data for players in a contest.
    
    NOW CORRECTLY HANDLES CPT vs FLEX:
    - Queries for BOTH roster_slot_id 511 (CPT) and 512 (FLEX)
    - Returns correct salary/ownership based on actual slot
    - CPT values are already 1.5x in the snapshot (if stored correctly)
    
    Args:
        session: Database session
        contest_date: When contest locked (UTC)
        player_names: List of player display names
        player_slots: Dict mapping normalized_name -> 'CPT' or 'FLEX'
    
    Returns:
        Dict mapping normalized_name -> {projection, salary, ownership_proj, ...}
    """
    from ivan.sports.nfl.models import ProjectionSnapshot
    from ivan.core.utils import normalize_name
    from zoneinfo import ZoneInfo

    # For classic contests, only get standard (non-CPT) slots
    if roster_slot_filter is None:
        roster_slot_filter = [511, 512]  # Both CPT and FLEX by default
    
    norm_names = [normalize_name(name) for name in player_names]

    
    # Query BOTH CPT (511) and FLEX (512) snapshot data
    results = (
        session.query(ProjectionSnapshot)
        .filter(
            ProjectionSnapshot.snapshot_date == contest_date,
            ProjectionSnapshot.normalized_name.in_(norm_names),
            ProjectionSnapshot.roster_slot_id.in_(roster_slot_filter)
        )
        .all()
    )
    
    # Group by normalized_name and roster_slot_id
    snapshot_by_player = {}
    for snap in results:
        if snap.normalized_name not in snapshot_by_player:
            snapshot_by_player[snap.normalized_name] = {}
        
        slot_key = 'CPT' if snap.roster_slot_id == 511 else 'FLEX'
        snapshot_by_player[snap.normalized_name][slot_key] = {
            'projection': snap.fpts_mean or 0.0,
            'projection_std': snap.fpts_std or 0.0,
            'salary': snap.salary,
            'position': snap.position,
            'team': snap.team,
            'opponent': snap.opponent,
            'ownership_proj': snap.ownership_proj or 0.0,
        }

    # Match each player to their actual slot in this contest
    snapshot_data = {}
    for norm_name in norm_names:
        actual_slot = player_slots.get(norm_name, 'FLEX')  # Default to FLEX if unknown
        
        player_snaps = snapshot_by_player.get(norm_name, {})
        
        if actual_slot in player_snaps:
            # Use the snapshot matching their actual slot
            snapshot_data[norm_name] = player_snaps[actual_slot]
        elif 'FLEX' in player_snaps:
            # Fallback: use FLEX data and multiply by 1.5x if they're actually CPT
            flex_data = player_snaps['FLEX'].copy()
            if actual_slot == 'CPT':
                flex_data['projection'] *= 1.5
                flex_data['salary'] = int(flex_data['salary'] * 1.5)
                # Ownership should stay the same (it's the same player)
            snapshot_data[norm_name] = flex_data
        # else: player has no snapshot data (will be caught later)
    
    return snapshot_data

def parse_showdown_lineup_string(lineup_str: str) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Parse lineup string into (captain, [flex_players], position_map).
    
    Handles both formats:
    - "CPT RB Kyren Williams FLEX WR Christian McCaffrey..."
    - "CPT Kyren Williams FLEX Christian McCaffrey..."
    
    Returns:
        captain: Captain player name
        flex_players: List of flex player names
        position_map: Dict mapping player_name -> position (e.g., 'QB', 'RB', 'WR')
    """
    # Handle NaN, None, and other non-string types
    if pd.isna(lineup_str) or lineup_str is None or not isinstance(lineup_str, str):
        return None, [], {}
    pattern_no_pos = r"(CPT|FLEX)\s+([A-Za-z\s\.''-]+?)(?=\s+(?:CPT|FLEX)|$)"
    matches_no_pos = re.findall(pattern_no_pos, lineup_str)
    
    captain = None
    flex_players = []
    position_map = {}
    
    for slot, name in matches_no_pos:
        name = name.strip()
        if slot == "CPT":
            captain = name
        else:
            flex_players.append(name)
    
    return captain, flex_players, position_map


def extract_team_from_name(name: str, team_roster: Dict[str, str]) -> Optional[str]:
    """
    Try to extract team from player name using a team roster lookup.
    team_roster maps normalized_name -> team_abbrev
    """
    norm = normalize_name(name)
    return team_roster.get(norm)


def quantify_stack_pattern(
    captain: str,
    flex_players: List[str],
    player_info: Dict[str, Dict]
) -> Dict:
    """
    Quantify player correlations for the lineup.
    
    player_info structure:
    {
        normalized_name: {
            'position': 'QB',
            'team': 'KC',
            'opponent': 'PHI',
            'salary': 10000,
            ...
        }
    }
    
    Returns dict with correlation metrics.
    """
    all_players = [captain] + flex_players
    norm_names = [normalize_name(p) for p in all_players]
    
    # Get team distribution
    teams = [player_info.get(n, {}).get('team') for n in norm_names]
    teams = [t for t in teams if t]
    team_counts = Counter(teams)
    
    # Get position distribution
    positions = [player_info.get(n, {}).get('position') for n in norm_names]
    pos_counts = Counter(positions)
    
    # Calculate features
    same_team_count = max(team_counts.values()) if team_counts else 1
    
    # Game stack pattern (e.g., "5-1", "4-2")
    if len(team_counts) >= 2:
        sorted_counts = sorted(team_counts.values(), reverse=True)
        game_stack_type = f"{sorted_counts[0]}-{sorted_counts[1]}"
    else:
        game_stack_type = f"{len(all_players)}-0"
    
    # Check for specific correlations
    cpt_pos = player_info.get(normalize_name(captain), {}).get('position', '')
    cpt_team = player_info.get(normalize_name(captain), {}).get('team', '')
    
    # QB-WR stack detection
    qb_wr_stack = False
    if cpt_pos == 'QB':
        wr_teams = [
            player_info.get(normalize_name(p), {}).get('team')
            for p in flex_players
            if player_info.get(normalize_name(p), {}).get('position') in ['WR', 'TE']
        ]
        qb_wr_stack = cpt_team in wr_teams
    
    # RB vs DST negative correlation
    rb_dst_negative = False
    rb_teams = [
        player_info.get(n, {}).get('team')
        for n in norm_names
        if player_info.get(n, {}).get('position') == 'RB'
    ]
    dst_opps = [
        player_info.get(n, {}).get('opponent')
        for n in norm_names
        if player_info.get(n, {}).get('position') == 'DST'
    ]
    rb_dst_negative = any(rb_team in dst_opps for rb_team in rb_teams if rb_team)
    
    # Stack pattern string (positions)
    stack_pattern = "-".join(sorted(
        [cpt_pos if cpt_pos else 'CPT'] + 
        [player_info.get(normalize_name(p), {}).get('position', 'FLEX') for p in flex_players]
    ))
    
    # Simple correlation score (higher = more correlated)
    correlation_score = (
        same_team_count * 0.3 +  # Bonus for stacking same team
        (5 if qb_wr_stack else 0) +  # Large bonus for QB-WR
        (-3 if rb_dst_negative else 0)  # Penalty for RB-DST
    )
    
    return {
        'stack_pattern': stack_pattern,
        'same_team_count': same_team_count,
        'game_stack_type': game_stack_type,
        'qb_wr_stack': qb_wr_stack,
        'rb_dst_negative': rb_dst_negative,
        'correlation_score': correlation_score,
    }

def process_showdown_contest(
    session: Session,
    hc: HistoricalContest,
    df: pd.DataFrame
) -> bool:
    """
    FIXED: Now correctly matches CPT vs FLEX data at the LINEUP level.
    Each player can appear as both CPT and FLEX across different lineups.
    """
    contest_id = hc.contest_id
    
    base_df = df.iloc[:, :-4].copy()
    own_df = df.iloc[:, -4:].copy()
    
    # ===== STEP 1: Extract player info from ownership table =====
    # NOTE: This table shows aggregated ownership across ALL lineups,
    # so we can't rely on it for CPT vs FLEX distinction
    player_base_info = {}
    
    for _, row in own_df.iterrows():
        player_name = row.get("Player")
        roster_pos = row.get("Roster Position", "")
        pct_drafted = row.get("%Drafted", "0%")
        fpts = row.get("FPTS")
        
        if pd.isna(player_name):
            continue
        
        norm_name = normalize_name(player_name)
        ownership_actual = float(str(pct_drafted).replace("%", ""))
        
        parts = str(roster_pos).split()
        position = parts[1] if len(parts) > 1 else ""
        
        player_base_info[norm_name] = {
            'name': player_name,
            'position': position,
            'ownership_actual': ownership_actual,
            'fpts_actual': float(fpts) if pd.notna(fpts) else None,
        }
        
    norm_names = list(player_base_info.keys())
    
    # Query BOTH CPT (511) and FLEX (512) snapshot data
    results = (
        session.query(ProjectionSnapshot)
        .filter(
            ProjectionSnapshot.snapshot_date == hc.date,
            ProjectionSnapshot.normalized_name.in_(norm_names),
            ProjectionSnapshot.roster_slot_id.in_([511, 512])
        )
        .all()
    )
    
    # Group by normalized_name and roster_slot_id
    snapshot_by_player = {}
    for snap in results:
        if snap.normalized_name not in snapshot_by_player:
            snapshot_by_player[snap.normalized_name] = {}
        
        slot_key = 'CPT' if snap.roster_slot_id == 511 else 'FLEX'
        snapshot_by_player[snap.normalized_name][slot_key] = {
            'projection': snap.fpts_mean* 1.5 if slot_key =='CPT' else snap.fpts_mean or 0.0,
            'projection_std': snap.fpts_std * 1.5 if slot_key=='CPT' else snap.fpts_std or 0.0,
            'salary': snap.salary,
            'position': snap.position,
            'team': snap.team,
            'opponent': snap.opponent,
            'ownership_proj': snap.ownership_proj or 0.0,
        }
    
    # Merge base info with snapshot data
    # For the player records table, we'll use FLEX data as baseline
    player_records = []
    for norm_name, base_info in player_base_info.items():
        player_snaps = snapshot_by_player.get(norm_name, {})
        
        # Use FLEX data for player records (since they can be used in either slot)
        flex_data = player_snaps.get('FLEX', {})
        
        player_records.append(ShowdownPlayer(
            contest_id=contest_id,
            player_name=base_info['name'],
            normalized_name=norm_name,
            position=flex_data.get('position', base_info.get('position', '')),
            roster_slot='FLEX',  # Default to FLEX in player table
            salary=flex_data.get('salary', 0),
            ownership_pct=base_info['ownership_actual'],
            ownership_proj=flex_data.get('ownership_proj', 0.0),
            fpts_actual=base_info.get('fpts_actual'),
            team=flex_data.get('team', ''),
            opponent=flex_data.get('opponent', ''),
        ))
    
    # ===== STEP 2: Parse lineups and match CPT/FLEX correctly =====
    lineup_data = defaultdict(lambda: {
        'captain': None,
        'flex': [],
        'positions': {},
        'count': 0,
        'ranks': [],
        'scores': [],
    })
    
    for _, row in base_df.iterrows():
        lineup_str = row.get("Lineup", "")
        rank = row.get("Rank")
        points = row.get("Points")
        
        captain, flex_players, position_map = parse_showdown_lineup_string(lineup_str)
        
        if not captain:
            continue
        
        lineup_key = (captain, tuple(sorted(flex_players)))
        
        lineup_data[lineup_key]['captain'] = captain
        lineup_data[lineup_key]['flex'] = flex_players
        lineup_data[lineup_key]['positions'] = position_map
        lineup_data[lineup_key]['count'] += 1
        lineup_data[lineup_key]['ranks'].append(int(rank) if pd.notna(rank) else None)
        lineup_data[lineup_key]['scores'].append(float(points) if pd.notna(points) else None)
    
    # ===== STEP 3: Calculate features with correct CPT/FLEX matching =====
    lineup_records = []
    projections = []
    ownerships_proj = []
    cpt_ownerships_proj = []
    salaries = []
    skipped_lineups = 0
    
    missing_cpt_data = []
    missing_flex_data = []

    for lineup_key, lu_data in lineup_data.items():
        captain = lu_data['captain']
        flex_players = lu_data['flex']
        
        # Get captain data from CPT slot
        cpt_norm = normalize_name(captain)
        cpt_snaps = snapshot_by_player.get(cpt_norm, {})
        
        # Prefer CPT slot data, fallback to FLEX * 1.5x
        if 'CPT' in cpt_snaps:
            cpt_data = cpt_snaps['CPT']
            cpt_projection = cpt_data['projection']  # Already 1.5x in snapshot
            cpt_salary = cpt_data['salary']  # Already 1.5x in snapshot
            cpt_ownership_proj = cpt_data['ownership_proj']
        else:
            skipped_lineups += 1
            continue
        
        # Get flex players data from FLEX slot
        flex_projection = 0.0
        flex_salary = 0
        flex_ownership_proj = 0.0
        flex_ownership_actual = 0.0
        
        skip_lineup = False
        for flex_player in flex_players:
            flex_norm = normalize_name(flex_player)
            flex_snaps = snapshot_by_player.get(flex_norm, {})
            
            if 'FLEX' in flex_snaps:
                flex_data = flex_snaps['FLEX']
                flex_projection += flex_data['projection']
                flex_salary += flex_data['salary']
                flex_ownership_proj += flex_data['ownership_proj']
            else:
                skip_lineup = True
                break
            
            # Add actual ownership from base info
            flex_ownership_actual += player_base_info.get(flex_norm, {}).get('ownership_actual', 0.0)
        
        if skip_lineup:
            skipped_lineups += 1
            continue
        
        # Total lineup metrics
        projection_raw = cpt_projection + flex_projection
        total_salary = cpt_salary + flex_salary
        salary_remaining = 50000 - total_salary
        salary_remaining_pct = (salary_remaining / 50000) * 100
        
        # Ownership sums
        ownership_sum_proj = cpt_ownership_proj + flex_ownership_proj
        cpt_ownership_actual = cpt_ownership_proj #TODO: pull actual cpt ownership for contest out of DK payload eventually?
        ownership_sum_actual = cpt_ownership_actual + flex_ownership_actual
        
        # Build player_info dict for correlation analysis
        player_info_for_stack = {}
        player_info_for_stack[cpt_norm] = cpt_data.copy()
        for flex_player in flex_players:
            flex_norm = normalize_name(flex_player)
            flex_snaps = snapshot_by_player.get(flex_norm, {})
            if 'FLEX' in flex_snaps:
                player_info_for_stack[flex_norm] = flex_snaps['FLEX']
            elif 'CPT' in flex_snaps:
                # Create FLEX equivalent
                cpt_only = flex_snaps['CPT']
                player_info_for_stack[flex_norm] = {
                    'position': cpt_only['position'],
                    'team': cpt_only['team'],
                    'opponent': cpt_only['opponent'],
                }
        
        # Correlations
        stack_metrics = quantify_stack_pattern(captain, flex_players, player_info_for_stack)
        
        # Duplicates
        dup_count = lu_data['count']
        ranks = [r for r in lu_data['ranks'] if r is not None]
        scores = [s for s in lu_data['scores'] if s is not None]
        
        lineup_records.append({
            'captain': captain,
            'flex': flex_players,
            'ownership_sum_proj': ownership_sum_proj,
            'ownership_sum_actual': ownership_sum_actual,
            'cpt_ownership_proj': cpt_ownership_proj,
            'cpt_ownership_actual': cpt_ownership_actual,
            'projection_raw': projection_raw,
            'total_salary': total_salary,
            'salary_remaining': salary_remaining,
            'salary_remaining_pct': salary_remaining_pct,
            'stack_metrics': stack_metrics,
            'dup_count': dup_count,
            'best_rank': min(ranks) if ranks else None,
            'best_score': max(scores) if scores else None,
            'first_rank': ranks[0] if ranks else None,
        })
        
        projections.append(projection_raw)
        ownerships_proj.append(ownership_sum_proj)
        cpt_ownerships_proj.append(cpt_ownership_proj)
        salaries.append(total_salary)
    
    if missing_cpt_data:
        log.debug(f"Contest {contest_id}: {len(missing_cpt_data)} players used as CPT but only had FLEX snapshots")
    if missing_flex_data:
        log.debug(f"Contest {contest_id}: {len(missing_flex_data)} players used as FLEX but only had CPT snapshots")
    
    if skipped_lineups > 0:
        log.warning(f"Contest {contest_id}: Skipped {skipped_lineups} lineups due to missing snapshot data")
    
    # ===== STEP 4: Normalize features =====
    if not projections:
        log.warning(f"Contest {contest_id}: No valid lineups after processing")
        return False
    
    proj_min, proj_max = min(projections), max(projections)
    own_min, own_max = min(ownerships_proj), max(ownerships_proj)
    cpt_min, cpt_max = min(cpt_ownerships_proj), max(cpt_ownerships_proj)
    
    def normalize(val, min_val, max_val):
        if max_val == min_val:
            return 0.5
        return (val - min_val) / (max_val - min_val)
    
    # Create lineup ORM objects
    final_lineup_records = []
    for lu_rec in lineup_records:
        proj_norm = normalize(lu_rec['projection_raw'], proj_min, proj_max)
        own_norm_proj = normalize(lu_rec['ownership_sum_proj'], own_min, own_max)
        cpt_percentile_proj = normalize(lu_rec['cpt_ownership_proj'], cpt_min, cpt_max)
        
        proj_percentile = (np.searchsorted(sorted(projections), lu_rec['projection_raw']) 
                          / len(projections))
        own_percentile_proj = (np.searchsorted(sorted(ownerships_proj), lu_rec['ownership_sum_proj']) 
                              / len(ownerships_proj))
        
        final_lineup_records.append(ShowdownLineup(
            contest_id=contest_id,
            captain_name=lu_rec['captain'],
            flex_players=lu_rec['flex'],
            
            # Correlation features
            **lu_rec['stack_metrics'],
            
            # Captain ownership - PROJECTED for model
            cpt_ownership_pct=lu_rec['cpt_ownership_proj'],
            cpt_ownership_percentile=cpt_percentile_proj,
            cpt_ownership_actual=lu_rec['cpt_ownership_actual'],
            
            # Projection
            projection_raw=lu_rec['projection_raw'],
            projection_normalized=proj_norm,
            projection_percentile=proj_percentile,
            
            # Ownership - PROJECTED for model
            ownership_sum=lu_rec['ownership_sum_proj'],
            ownership_normalized=own_norm_proj,
            ownership_percentile=own_percentile_proj,
            ownership_sum_actual=lu_rec['ownership_sum_actual'],
            
            # Salary
            total_salary=lu_rec['total_salary'],
            salary_remaining=lu_rec['salary_remaining'],
            salary_remaining_pct=lu_rec['salary_remaining_pct'],
            
            # Target variables
            duplicate_count=lu_rec['dup_count'],
            best_score=lu_rec['best_score'],
            best_rank=lu_rec['best_rank'],
            first_seen_rank=lu_rec['first_rank'],
        ))
    
    # ===== STEP 5: Create contest metadata =====
    stack_pattern_counts = Counter([
        lu['stack_metrics']['stack_pattern'] 
        for lu in lineup_records
    ])
    
    metadata = ShowdownContestMetadata(
        contest_id=contest_id,
        date=hc.date,
        total_entries=hc.total_entries,
        entry_fee=hc.entry_fee,
        
        min_projection=proj_min,
        max_projection=proj_max,
        median_projection=float(np.median(projections)),
        min_ownership=own_min,
        max_ownership=own_max,
        median_ownership=float(np.median(ownerships_proj)),
        min_cpt_ownership=cpt_min,
        max_cpt_ownership=cpt_max,
        
        unique_lineup_count=len(lineup_records),
        max_duplicate_count=max([lu['dup_count'] for lu in lineup_records]),
        median_duplicate_count=int(np.median([lu['dup_count'] for lu in lineup_records])),
        
        top_stack_patterns=dict(stack_pattern_counts.most_common(10)),
        ownership_gini=calculate_gini(ownerships_proj),
        projection_std=float(np.std(projections)),
    )
    
    # ===== STEP 6: Save to database =====
    try:
        session.query(ShowdownPlayer).filter_by(contest_id=contest_id).delete()
        session.query(ShowdownLineup).filter_by(contest_id=contest_id).delete()
        session.query(ShowdownContestMetadata).filter_by(contest_id=contest_id).delete()
        
        session.bulk_save_objects(player_records)
        session.bulk_save_objects(final_lineup_records)
        session.add(metadata)
        
        session.commit()
        
        unique_with_snapshots = len(snapshot_by_player)
        total_unique = len(player_base_info)
        snapshot_match_rate = unique_with_snapshots / total_unique if total_unique else 0
        
        players_with_ownership = sum(
            1 for snaps in snapshot_by_player.values()
            if any(snap.get('ownership_proj', 0) > 0 for snap in snaps.values())
        )
        ownership_proj_rate = players_with_ownership / total_unique if total_unique else 0
        
        log.info(
            f"✓ Contest {contest_id}: {len(final_lineup_records)} lineups | "
            f"Snapshot: {snapshot_match_rate:.1%} | Ownership: {ownership_proj_rate:.1%}"
        )
        return True
        
    except Exception as e:
        session.rollback()
        log.error(f"✗ Failed to save contest {contest_id}: {e}")
        return False


def calculate_gini(values: List[float]) -> float:
    """Calculate Gini coefficient."""
    if not values or len(values) < 2:
        return 0.0
    sorted_values = np.array(sorted(values))
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

def process_contest(session: Session, hc: HistoricalContest, df: pd.DataFrame) -> bool:
    """
    Route to appropriate processor based on contest type.
    """
    contest_type = hc.contest_type or ""
    
    if "Showdown" in contest_type or "Captain Mode" in contest_type:
        return process_showdown_contest(session, hc, df)
    else:
        # Classic contest
        return process_classic_contest(session, hc, df)
    
def parse_classic_lineup_string(lineup_str: str) -> List[str]:
    """Parse classic lineup format: 'DST Falcons FLEX D'Andre Swift QB Jayden Daniels...'"""
    if pd.isna(lineup_str) or not isinstance(lineup_str, str):
        return []
    
    # Split by position labels - \s* makes leading whitespace optional (for positions at start)
    parts = re.split(r'\s*(QB|RB|WR|TE|FLEX|DST)\s+', lineup_str)
    
    players = []
    # After split, pattern is: [text_before, POS1, player1, POS2, player2, ...]
    # We want to extract the player names (indices 2, 4, 6, ...)
    for i in range(2, len(parts), 2):
        if i < len(parts):
            player_name = parts[i].strip().rstrip(',')  # Remove trailing comma and whitespace
            if player_name:  # Only add non-empty names
                players.append(player_name)
    
    return players


def process_classic_contest(
    session: Session,
    hc: HistoricalContest,
    df: pd.DataFrame
) -> bool:
    """
    Process classic contest: extract normalized distributions for projection and ownership.

    Stores 20-bin normalized histograms that will be used to train models
    predicting field distributions based on contest parameters.
    """
    contest_id = hc.contest_id

    # Parse lineup data
    base_df = df.iloc[:, :-4].copy()  # Main lineup data
    own_df = df.iloc[:, -4:].copy()   # Ownership table

    # Get player base info from ownership table
    player_base_info = {}
    for _, row in own_df.iterrows():
        player_name = row.get("Player")
        if pd.isna(player_name):
            continue

        norm_name = normalize_name(player_name)
        ownership_actual = float(str(row.get("%Drafted", "0%")).replace("%", ""))
        fpts_actual = row.get("FPTS")

        player_base_info[norm_name] = {
            'name': player_name,
            'ownership_actual': ownership_actual,
            'fpts_actual': float(fpts_actual) if pd.notna(fpts_actual) else None,
        }

    # Get snapshot data for players
    all_player_names = [info['name'] for info in player_base_info.values()]
    snapshot_data = get_snapshot_data_for_contest(
        session,
        hc.date,
        all_player_names,
        player_slots={},
        roster_slot_filter=[66, 67, 68, 69, 71]
    )

    # Parse all lineups
    lineup_projections = []
    lineup_ownerships = []
    lineup_salaries = []

    for _, row in base_df.iterrows():
        lineup_str = row.get("Lineup", "")
        player_names = parse_classic_lineup_string(lineup_str)
        if not player_names:
            continue

        proj_sum = 0.0
        own_sum = 0.0
        salary_sum = 0
        valid = True

        for player_name in player_names:
            norm_name = normalize_name(player_name)
            snap = snapshot_data.get(norm_name)
            if not snap:
                valid = False
                break

            proj_sum += snap.get('projection', 0.0)
            own_sum += snap.get('ownership_proj', 0.0)
            salary_sum += snap.get('salary', 0)

        if valid:
            lineup_projections.append(proj_sum)
            lineup_ownerships.append(own_sum)
            lineup_salaries.append(salary_sum)

    if not lineup_projections:
        log.warning(f"Contest {contest_id}: No valid lineups after processing")
        return False

    # Convert to arrays
    projections = np.array(lineup_projections)
    ownerships = np.array(lineup_ownerships)
    salaries = np.array(lineup_salaries)

    # Normalize to [0, 1] INDEPENDENTLY
    proj_min, proj_max = projections.min(), projections.max()
    own_min, own_max = ownerships.min(), ownerships.max()

    if proj_max > proj_min:
        proj_normalized = (projections - proj_min) / (proj_max - proj_min)
    else:
        proj_normalized = np.full_like(projections, 0.5)

    if own_max > own_min:
        own_normalized = (ownerships - own_min) / (own_max - own_min)
    else:
        own_normalized = np.full_like(ownerships, 0.5)

    # Compute 20-bin normalized histograms
    hist_proj_counts, hist_own_counts = compute_histograms(proj_normalized, own_normalized)

    # Calculate ownership concentration (Gini coefficient)
    ownership_gini = calculate_gini(ownerships.tolist())

    # Store metadata
    metadata = ClassicContestMetadata(
        contest_id=contest_id,
        date=hc.date,
        total_entries=hc.total_entries,
        entry_fee=hc.entry_fee,
        num_games=None,  # To be filled in manually or inferred

        # Normalization bounds
        min_projection=float(proj_min),
        max_projection=float(proj_max),
        min_ownership=float(own_min),
        max_ownership=float(own_max),

        # Histogram distributions (normalized, summing to 1.0)
        hist_proj_counts=hist_proj_counts,
        hist_own_counts=hist_own_counts,

        # Statistics
        projection_mean=float(projections.mean()),
        projection_std=float(projections.std()),
        ownership_mean=float(ownerships.mean()),
        ownership_std=float(ownerships.std()),

        salary_remaining_mean=float((50000 - salaries).mean()),

        # Ownership concentration
        ownership_gini=float(ownership_gini),
    )

    # Save to database
    try:
        session.query(ClassicContestMetadata).filter_by(contest_id=contest_id).delete()
        session.add(metadata)
        session.commit()

        log.info(
            f"✓ Contest {contest_id}: {len(projections)} lineups | "
            f"Proj: [{proj_min:.1f}, {proj_max:.1f}] | "
            f"Own: [{own_min:.1f}, {own_max:.1f}] | "
            f"Gini: {ownership_gini:.3f}"
        )
        return True

    except Exception as e:
        session.rollback()
        log.error(f"✗ Failed to save contest {contest_id}: {e}")
        return False