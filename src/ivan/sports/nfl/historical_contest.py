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
)

log = logging.getLogger(__name__)

_SHOWDOWN_LINEUP_REGEX = re.compile(r"(CPT|FLEX)\s+([A-Za-z\s\.''-]+?)(?=\s+(?:CPT|FLEX)|$)")

def get_snapshot_data_for_contest(
    session: Session,
    contest_date: datetime,
    player_names: List[str],
) -> Dict[str, Dict]:
    """
    Get ALL historical prelock data for players in a contest.
    Matches on contest date (truncated to day) + normalized player names.
    
    Args:
        session: Database session
        contest_date: When the contest locked (in UTC from DraftKings)
        player_names: List of player display names from contest
    
    Returns:
        Dict mapping normalized_name -> {projection, salary, position, team, etc.}
    """
    from ivan.sports.nfl.models import ProjectionSnapshot
    from ivan.core.utils import normalize_name
    
    # Truncate contest date to day (midnight UTC)
    contest_day = contest_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Normalize all player names
    norm_names = [normalize_name(name) for name in player_names]
    
    # Query all snapshot data for this date + these players
    results = (
        session.query(ProjectionSnapshot)
        .filter(
            ProjectionSnapshot.snapshot_date == contest_day,
            ProjectionSnapshot.normalized_name.in_(norm_names)
        )
        .all()
    )
    
    # Convert to dict keyed by normalized name
    snapshot_data = {}
    for snap in results:
        snapshot_data[snap.normalized_name] = {
            'projection': snap.fpts_mean or 0.0,
            'projection_std': snap.fpts_std or 0.0,
            'salary': snap.salary,
            'position': snap.position,
            'team': snap.team,
            'opponent': snap.opponent,
            'roster_slot_id': snap.roster_slot_id,
        }
    
    return snapshot_data

def parse_showdown_lineup_string(lineup_str: str) -> Tuple[str, List[str]]:
    """Parse lineup string into (captain, [flex_players])."""
    # Handle NaN, None, and other non-string types
    if pd.isna(lineup_str) or lineup_str is None or not isinstance(lineup_str, str):
        return None, []
    
    matches = _SHOWDOWN_LINEUP_REGEX.findall(lineup_str)
    captain = None
    flex_players = []
    for pos, name in matches:
        name = name.strip()
        if pos == "CPT":
            captain = name
        else:
            flex_players.append(name)
    return captain, flex_players


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
    Process a single showdown contest and store lineup-level data.
    Now uses ProjectionSnapshot for historical prelock data.
    
    Returns True if successful, False otherwise.
    """
    contest_id = hc.contest_id
    
    # Split dataframe into lineups and ownership sections
    base_df = df.iloc[:, :-4].copy()
    own_df = df.iloc[:, -4:].copy()
    
    # ===== STEP 1: Extract player information from DraftKings results =====
    # This gives us ACTUAL ownership (post-lock) and ACTUAL fpts
    player_info = {}
    player_records = []
    
    for _, row in own_df.iterrows():
        player_name = row.get("Player")
        roster_pos = row.get("Roster Position", "")
        pct_drafted = row.get("%Drafted", "0%")
        fpts = row.get("FPTS")
        
        if pd.isna(player_name):
            continue
        
        norm_name = normalize_name(player_name)
        ownership = float(str(pct_drafted).replace("%", ""))
        
        # Parse position from roster position (e.g., "CPT QB", "FLEX WR")
        parts = str(roster_pos).split()
        slot = parts[0] if parts else "FLEX"
        position = parts[1] if len(parts) > 1 else ""
        
        player_info[norm_name] = {
            'name': player_name,
            'position': position,
            'ownership': ownership,  # ACTUAL ownership (from DK results)
            'slot': slot,
            'fpts_actual': float(fpts) if pd.notna(fpts) else None,
        }
    
    # ===== STEP 1.5: Get HISTORICAL prelock data from snapshot =====
    all_player_names = list(player_info.keys())  # Already normalized
    snapshot_data = get_snapshot_data_for_contest(
        session,
        hc.date,
        [player_info[norm]['name'] for norm in all_player_names]
    )
    
    # Merge snapshot data into player_info
    for norm_name, snap in snapshot_data.items():
        if norm_name in player_info:
            player_info[norm_name].update({
                'projection': snap['projection'],
                'salary': snap['salary'],
                'team': snap['team'],
                'opponent': snap['opponent'],
            })
            # Update position if snapshot has it and DK doesn't
            if not player_info[norm_name]['position'] and snap['position']:
                player_info[norm_name]['position'] = snap['position']
    
    # Create player records
    for norm_name, info in player_info.items():
        player_records.append(ShowdownPlayer(
            contest_id=contest_id,
            player_name=info['name'],
            normalized_name=norm_name,
            position=info.get('position', ''),
            roster_slot=info['slot'],
            salary=info.get('salary', 0),
            ownership_pct=info['ownership'],
            fpts_actual=info.get('fpts_actual'),
            team=info.get('team'),
            opponent=info.get('opponent'),
        ))
    
    # ===== STEP 2: Parse all lineups and aggregate duplicates =====
    lineup_data = defaultdict(lambda: {
        'captain': None,
        'flex': [],
        'count': 0,
        'ranks': [],
        'scores': [],
    })
    
    for _, row in base_df.iterrows():
        lineup_str = row.get("Lineup", "")
        rank = row.get("Rank")
        points = row.get("Points")
        
        captain, flex_players = parse_showdown_lineup_string(lineup_str)
        
        if not captain:
            continue
        
        # Create a hashable lineup key
        lineup_key = (captain, tuple(sorted(flex_players)))
        
        lineup_data[lineup_key]['captain'] = captain
        lineup_data[lineup_key]['flex'] = flex_players
        lineup_data[lineup_key]['count'] += 1
        lineup_data[lineup_key]['ranks'].append(int(rank) if pd.notna(rank) else None)
        lineup_data[lineup_key]['scores'].append(float(points) if pd.notna(points) else None)
    
    # ===== STEP 3: Calculate features for each unique lineup =====
    lineup_records = []
    projections = []
    ownerships = []
    cpt_ownerships = []
    salaries = []
    
    for lineup_key, lu_data in lineup_data.items():
        captain = lu_data['captain']
        flex_players = lu_data['flex']
        all_players = [captain] + flex_players
        
        # Calculate ACTUAL ownership sum (from DK results)
        ownership_sum = sum(
            player_info.get(normalize_name(p), {}).get('ownership', 0.0)
            for p in all_players
        )
        
        # Captain ACTUAL ownership
        cpt_ownership = player_info.get(normalize_name(captain), {}).get('ownership', 0.0)
        
        # HISTORICAL projection (from snapshot at prelock)
        projection_raw = sum(
            player_info.get(normalize_name(p), {}).get('projection', 0.0)
            for p in all_players
        )
        
        # HISTORICAL salary (from snapshot at prelock)
        total_salary = sum(
            player_info.get(normalize_name(p), {}).get('salary', 0)
            for p in all_players
        )
        salary_remaining = 50000 - total_salary
        salary_remaining_pct = (salary_remaining / 50000) * 100
        
        # Quantify correlations (using snapshot team/position data)
        stack_metrics = quantify_stack_pattern(captain, flex_players, player_info)
        
        # Get duplicate count and best performance
        dup_count = lu_data['count']
        ranks = [r for r in lu_data['ranks'] if r is not None]
        scores = [s for s in lu_data['scores'] if s is not None]
        
        lineup_records.append({
            'captain': captain,
            'flex': flex_players,
            'ownership_sum': ownership_sum,
            'cpt_ownership': cpt_ownership,
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
        ownerships.append(ownership_sum)
        cpt_ownerships.append(cpt_ownership)
        salaries.append(total_salary)
    
    # ===== STEP 4: Normalize features within contest (0-1 scaling) =====
    proj_min, proj_max = min(projections), max(projections)
    own_min, own_max = min(ownerships), max(ownerships)
    cpt_min, cpt_max = min(cpt_ownerships), max(cpt_ownerships)
    
    def normalize(val, min_val, max_val):
        if max_val == min_val:
            return 0.5
        return (val - min_val) / (max_val - min_val)
    
    # Create lineup ORM objects
    final_lineup_records = []
    for i, lu_rec in enumerate(lineup_records):
        proj_norm = normalize(lu_rec['projection_raw'], proj_min, proj_max)
        own_norm = normalize(lu_rec['ownership_sum'], own_min, own_max)
        cpt_percentile = normalize(lu_rec['cpt_ownership'], cpt_min, cpt_max)
        
        # Calculate percentiles
        proj_percentile = (np.searchsorted(sorted(projections), lu_rec['projection_raw']) 
                          / len(projections))
        own_percentile = (np.searchsorted(sorted(ownerships), lu_rec['ownership_sum']) 
                         / len(ownerships))
        
        final_lineup_records.append(ShowdownLineup(
            contest_id=contest_id,
            captain_name=lu_rec['captain'],
            flex_players=lu_rec['flex'],
            
            # Correlation features (dimension 1)
            **lu_rec['stack_metrics'],
            
            # Captain ownership (dimension 2)
            cpt_ownership_pct=lu_rec['cpt_ownership'],
            cpt_ownership_percentile=cpt_percentile,
            
            # Projection (dimension 3) - NOW FROM SNAPSHOT
            projection_raw=lu_rec['projection_raw'],
            projection_normalized=proj_norm,
            projection_percentile=proj_percentile,
            
            # Ownership (dimension 4) - ACTUAL from DK results
            ownership_sum=lu_rec['ownership_sum'],
            ownership_normalized=own_norm,
            ownership_percentile=own_percentile,
            
            # Salary (dimension 5) - NOW FROM SNAPSHOT
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
        
        # Normalization bounds
        min_projection=proj_min,
        max_projection=proj_max,
        median_projection=float(np.median(projections)),
        min_ownership=own_min,
        max_ownership=own_max,
        median_ownership=float(np.median(ownerships)),
        min_cpt_ownership=cpt_min,
        max_cpt_ownership=cpt_max,
        
        # Aggregate stats
        unique_lineup_count=len(lineup_records),
        max_duplicate_count=max([lu['dup_count'] for lu in lineup_records]),
        median_duplicate_count=int(np.median([lu['dup_count'] for lu in lineup_records])),
        
        top_stack_patterns=dict(stack_pattern_counts.most_common(10)),
        ownership_gini=calculate_gini(ownerships),
        projection_std=float(np.std(projections)),
    )
    
    # ===== STEP 6: Save to database =====
    try:
        # Clear existing data for this contest
        session.query(ShowdownPlayer).filter_by(contest_id=contest_id).delete()
        session.query(ShowdownLineup).filter_by(contest_id=contest_id).delete()
        session.query(ShowdownContestMetadata).filter_by(contest_id=contest_id).delete()
        
        # Insert new data
        session.bulk_save_objects(player_records)
        session.bulk_save_objects(final_lineup_records)
        session.add(metadata)
        
        session.commit()
        
        # Log snapshot data availability
        snapshot_match_rate = len(snapshot_data) / len(player_info) if player_info else 0
        log.info(
            f"✓ Processed contest {contest_id}: {len(final_lineup_records)} unique lineups | "
            f"Snapshot match: {snapshot_match_rate:.1%} ({len(snapshot_data)}/{len(player_info)} players)"
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


def run_pipeline(
    user_data_dir: str = None,
    profile_directory: str = None,
    headless: bool = False,
    days_back: int = 7,
):
    """
    Main pipeline to process showdown contests.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    engine = init_db("nfl")
    session = get_session(engine)
    
    # Build DraftKings session
    sess = build_requests_session_from_chrome(
        user_data_dir=user_data_dir,
        profile_directory=profile_directory,
        headless=headless,
    )
    
    # Find unprocessed showdown contests
    now = datetime.utcnow()
    cutoff = now - timedelta(days=days_back)
    
    processed_contests = session.query(ShowdownContestMetadata.contest_id).subquery()
    
    to_process = (
        session.query(HistoricalContest)
        .filter(
            HistoricalContest.date >= cutoff,
            HistoricalContest.date <= now,
            HistoricalContest.contest_type == "Showdown Captain Mode",
            ~HistoricalContest.contest_id.in_(processed_contests),
        )
        .order_by(HistoricalContest.date.desc())
        .all()
    )
    
    log.info(f"Found {len(to_process)} contests to process")
    
    success_count = 0
    for hc in to_process:
        log.info(f"Processing contest {hc.contest_id} from {hc.date}")
        
        try:
            df = download_contest_dataframe(sess, hc.contest_id)
            
            if df is None or df.empty:
                log.warning(f"Skipping contest {hc.contest_id}: no data")
                continue
            
            if process_showdown_contest(session, hc, df):
                success_count += 1
                
        except Exception as e:
            log.exception(f"Error processing contest {hc.contest_id}")
            continue
    
    log.info(f"Pipeline complete: {success_count}/{len(to_process)} contests processed")
    session.close()