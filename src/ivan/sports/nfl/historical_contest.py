import re
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core.utils import calculate_gini
from ivan.core.utils import normalize_name
from ivan.sports.nfl.models import (
    HistoricalContest,
    ShowdownPlayer,
    ShowdownLineup,
    ShowdownContestMetadata,
    ClassicContestMetadata,
    ContestEntryResult,
    ContestFieldScores,
    ProjectionSnapshot
)
from ivan.sports.nfl.placement import load_payout_tiers, payout_vector, place, remove_our_copies

log = logging.getLogger(__name__)

_SHOWDOWN_LINEUP_REGEX = re.compile(r"(CPT|FLEX)\s+([A-Za-z\s\.''-]+?)(?=\s+(?:CPT|FLEX)|$)")


# ==============================================================================
# SNAPSHOT DATE + ACTUAL RESULTS (shared by both contest types)
# ==============================================================================

def resolve_snapshot_date(session: Session, contest_date: Optional[datetime]) -> Optional[datetime]:
    """
    The projection_snapshot date that belongs to a contest. Contest dates are
    UTC; snapshots were stamped in UTC through 2025 and with the local kickoff
    time since 2026, so an exact match is tried first and then the nearest
    snapshot within six hours of either reading of the contest date.
    """
    if contest_date is None:
        return None
    lo, hi = contest_date - timedelta(days=1), contest_date + timedelta(days=1)
    dates = [d for (d,) in session.query(ProjectionSnapshot.snapshot_date)
             .filter(ProjectionSnapshot.snapshot_date.between(lo, hi)).distinct().all()]
    if not dates or contest_date in dates:
        return contest_date
    local = contest_date.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    gap = lambda d: min(abs(d - contest_date), abs(d - local))
    best = min(dates, key=gap)
    return best if gap(best) <= timedelta(hours=6) else contest_date


def _standings_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Column names of the standings part of a DK export, keyed by a lowercase alias."""
    base = df.iloc[:, :-4] if df.shape[1] > 6 else df
    found = {}
    for c in base.columns:
        key = str(c).strip().lower().replace(" ", "").replace("_", "")
        found.setdefault(key, c)
    return {"rank": found.get("rank"), "entry_id": found.get("entryid"), "entry_name": found.get("entryname"),
            "points": found.get("points"), "lineup": found.get("lineup")}


def store_field_results(session: Session, hc: HistoricalContest, df: pd.DataFrame,
                        username: Optional[str] = None) -> Dict[str, int]:
    """
    Keep every entry's actual points (contest_field_scores) and, when a
    DraftKings username is given, our own entries' rank, points and payout
    (contest_entry_result). Works for showdown and classic exports alike and
    needs no snapshot, so it is stored before either processor runs.
    """
    cid = hc.contest_id
    cols = _standings_columns(df)
    if not cols["points"]:
        log.warning(f"Contest {cid}: standings export has no Points column; field scores not stored")
        return {"field": 0, "own": 0}
    pts = pd.to_numeric(df[cols["points"]], errors="coerce")
    valid = pts.notna()
    scores = np.sort(pts[valid].to_numpy(dtype=float))[::-1]

    session.query(ContestFieldScores).filter_by(contest_id=cid).delete()
    session.add(ContestFieldScores(contest_id=cid, n_entries=int(scores.size),
                                   scores=[round(float(x), 2) for x in scores]))

    n_own = 0
    if username and cols["entry_name"]:
        owner = (df[cols["entry_name"]].astype(str).str.replace(r"\s*\(\d+/\d+\)\s*$", "", regex=True)
                 .str.strip().str.lower())
        mine = df.index[valid & (owner == username.strip().lower())]
        if len(mine):
            tiers = load_payout_tiers(session, cid)
            pay = payout_vector(tiers, int(scores.size)) if tiers else np.zeros(int(scores.size))
            our = pts[mine].to_numpy(dtype=float)
            field_wo, _ = remove_our_copies(scores, our)
            ranks, pays = place(our, field_wo, pay)
            session.query(ContestEntryResult).filter_by(contest_id=cid).delete()
            for i, ix in enumerate(mine):
                row = df.loc[ix]
                try:
                    entry_id = int(row[cols["entry_id"]]) if cols["entry_id"] else None
                except (TypeError, ValueError):
                    entry_id = None
                if entry_id is None:
                    continue
                tie = int(np.sum(np.abs(np.concatenate([field_wo, our]) - our[i]) <= 1e-6))
                session.add(ContestEntryResult(
                    entry_id=entry_id, contest_id=cid, username=username, rank=int(ranks[i]), tie_size=tie,
                    points=float(our[i]), payout=float(pays[i]),
                    lineup=str(row[cols["lineup"]]) if cols["lineup"] else None))
                n_own += 1
            if not tiers:
                log.warning(f"Contest {cid}: no payout table; own entries stored with payout 0 (run get_payouts.py --all)")
    session.commit()
    log.info(f"Contest {cid}: stored {scores.size} field scores" + (f", {n_own} own entries" if n_own else ""))
    return {"field": int(scores.size), "own": n_own}

# ==============================================================================
# HELPER FUNCTIONS FOR CLASSIC CONTEST PROCESSING
# ==============================================================================

BINS = 20  # Fixed-width histogram bins

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

def parse_showdown_lineup_string(lineup_str: str) -> Tuple[str, List[str]]:
    """
    Parse lineup string into (captain, [flex_players]).
    
    Handles format: "CPT Kyren Williams FLEX Christian McCaffrey FLEX..."
    
    Returns:
        captain: Captain player name
        flex_players: List of 5 flex player names
    """
    if pd.isna(lineup_str) or lineup_str is None or not isinstance(lineup_str, str):
        return None, []
    
    pattern = r"(CPT|FLEX)\s+([A-Za-z\s\.''-]+?)(?=\s+(?:CPT|FLEX)|$)"
    matches = re.findall(pattern, lineup_str)
    
    captain = None
    flex_players = []
    
    for slot, name in matches:
        name = name.strip()
        if slot == "CPT":
            captain = name
        else:
            flex_players.append(name)
    
    return captain, flex_players


def process_showdown_contest(
    session: Session,
    hc: HistoricalContest,
    df: pd.DataFrame
) -> bool:
    """
    Streamlined showdown processing:
    1. Parse lineups, count CPT/FLEX usage
    2. Calculate actual ownership = (usage_count / total_lineups) * 100
    3. Match to ProjectionSnapshot for player IDs and metadata
    4. Store ShowdownPlayer, ShowdownLineup, ShowdownContestMetadata
    """
    contest_id = hc.contest_id
    
    base_df = df.iloc[:, :-4].copy()
    own_df = df.iloc[:, -4:].copy()
    
    # Build player lookup from ownership table
    # The export lists a player once per slot: a FLEX row with the base points
    # and a CPT row with 1.5x. Keep the base points (the FLEX row; a CPT-only
    # row is divided back), never the last row seen.
    player_lookup = {}
    for _, row in own_df.iterrows():
        player_name = row.get("Player")
        fpts = row.get("FPTS")
        
        if pd.isna(player_name):
            continue
        
        norm_name = normalize_name(player_name)
        slot = str(row.get("Roster Position") or "").strip().upper()
        base = float(fpts) if pd.notna(fpts) else None
        if slot == "CPT" and base is not None:
            base = round(base / 1.5, 2)
        prev = player_lookup.get(norm_name)
        if prev is None or (slot != "CPT" and prev.get("slot") == "CPT"):
            player_lookup[norm_name] = {'name': player_name, 'fpts_actual': base, 'slot': slot}
    
    # Get ProjectionSnapshot data for player IDs and metadata
    norm_names = list(player_lookup.keys())
    results = (
        session.query(ProjectionSnapshot)
        .filter(
            ProjectionSnapshot.snapshot_date == resolve_snapshot_date(session, hc.date),
            ProjectionSnapshot.normalized_name.in_(norm_names),
            ProjectionSnapshot.roster_slot_id.in_([511, 512])
        )
        .all()
    )
    
    snapshot_by_player = {}
    for snap in results:
        if snap.normalized_name not in snapshot_by_player:
            snapshot_by_player[snap.normalized_name] = {}
        
        slot_key = 'CPT' if snap.roster_slot_id == 511 else 'FLEX'
        
        # Use player_id if available, otherwise hash the normalized name
        player_id = getattr(snap, 'player_id', None)
        if player_id is None:
            # Generate consistent ID from normalized name + slot
            player_id = hash(f"{snap.normalized_name}_{snap.roster_slot_id}") & 0x7FFFFFFF
        
        snapshot_by_player[snap.normalized_name][slot_key] = {
            'player_id': player_id,
            'roster_slot_id': snap.roster_slot_id,
            'salary': snap.salary,
            'position': snap.position,
            'team': snap.team,
            'ownership_proj': snap.ownership_proj or 0.0,
            'fpts_proj': snap.fpts_mean or 0.0,
        }
    
    # Parse lineups and count usage
    lineup_compositions = []
    cpt_usage_count = defaultdict(int)
    flex_usage_count = defaultdict(int)
    
    total_lineups = 0
    for _, row in base_df.iterrows():
        captain, flex_players = parse_showdown_lineup_string(row.get("Lineup", ""))
        
        if not captain or len(flex_players) != 5:
            continue
        
        total_lineups += 1
        cpt_norm = normalize_name(captain)
        flex_norms = [normalize_name(p) for p in flex_players]
        
        lineup_compositions.append({
            'cpt_norm': cpt_norm,
            'flex_norms': flex_norms,
        })
        
        cpt_usage_count[cpt_norm] += 1
        for fn in flex_norms:
            flex_usage_count[fn] += 1
    
    if total_lineups == 0:
        log.warning(f"Contest {contest_id}: No valid lineups")
        return False
    
    # Create ShowdownPlayer records with actual ownership
    player_records = []
    for norm_name in player_lookup.keys():
        snaps = snapshot_by_player.get(norm_name, {})
        if not snaps:
            continue
        
        # CPT
        cpt_count = cpt_usage_count.get(norm_name, 0)
        if cpt_count > 0 and 'CPT' in snaps:
            cpt = snaps['CPT']
            player_records.append(ShowdownPlayer(
                contest_id=contest_id,
                player_id=cpt['player_id'],
                roster_slot_id=cpt['roster_slot_id'],
                salary=cpt['salary'],
                position=cpt['position'],
                team=cpt['team'],
                ownership_actual=(cpt_count / total_lineups) * 100,
                ownership_proj=cpt['ownership_proj'],
                fpts_proj=cpt['fpts_proj'],
                fpts_actual=player_lookup[norm_name]['fpts_actual'],
            ))
        
        # FLEX
        flex_count = flex_usage_count.get(norm_name, 0)
        if flex_count > 0 and 'FLEX' in snaps:
            flex = snaps['FLEX']
            player_records.append(ShowdownPlayer(
                contest_id=contest_id,
                player_id=flex['player_id'],
                roster_slot_id=flex['roster_slot_id'],
                salary=flex['salary'],
                position=flex['position'],
                team=flex['team'],
                ownership_actual=(flex_count / total_lineups) * 100,
                ownership_proj=flex['ownership_proj'],
                fpts_proj=flex['fpts_proj'],
                fpts_actual=player_lookup[norm_name]['fpts_actual'],
            ))
    
    # Group lineups by composition
    lineup_groups = defaultdict(lambda: {'cpt_norm': None, 'flex_norms': [], 'count': 0})
    for lc in lineup_compositions:
        key = (lc['cpt_norm'], tuple(sorted(lc['flex_norms'])))
        lineup_groups[key]['cpt_norm'] = lc['cpt_norm']
        lineup_groups[key]['flex_norms'] = lc['flex_norms']
        lineup_groups[key]['count'] += 1
    
    # Create ShowdownLineup records
    lineup_records = []
    for key, lg in lineup_groups.items():
        cpt_snaps = snapshot_by_player.get(lg['cpt_norm'], {})
        if 'CPT' not in cpt_snaps:
            continue
        
        cpt_player_id = cpt_snaps['CPT']['player_id']
        cpt_salary = cpt_snaps['CPT']['salary']
        
        flex_player_ids = []
        flex_salaries = []
        skip = False
        
        for fn in lg['flex_norms']:
            flex_snaps = snapshot_by_player.get(fn, {})
            if 'FLEX' not in flex_snaps:
                skip = True
                break
            flex_player_ids.append(flex_snaps['FLEX']['player_id'])
            flex_salaries.append(flex_snaps['FLEX']['salary'])
        
        if skip:
            continue
        
        lineup_records.append(ShowdownLineup(
            contest_id=contest_id,
            cpt_player_id=cpt_player_id,
            flex_player_ids=flex_player_ids,
            salary_remaining=50000 - cpt_salary - sum(flex_salaries),
            dupe_count=lg['count'],
        ))
    
    # Create ShowdownContestMetadata
    dupe_counts = [lu.dupe_count for lu in lineup_records]
    metadata = ShowdownContestMetadata(
        contest_id=contest_id,
        unique_lineup_count=len(lineup_records),
        max_dupe_count=max(dupe_counts) if dupe_counts else 0,
        median_dupe_count=int(np.median(dupe_counts)) if dupe_counts else 0,
        p10_dupe_count=int(np.percentile(dupe_counts, 10)) if dupe_counts else None,
        p25_dupe_count=int(np.percentile(dupe_counts, 25)) if dupe_counts else None,
        p75_dupe_count=int(np.percentile(dupe_counts, 75)) if dupe_counts else None,
        p90_dupe_count=int(np.percentile(dupe_counts, 90)) if dupe_counts else None,
    )
    
    # Save to database
    try:
        session.query(ShowdownPlayer).filter_by(contest_id=contest_id).delete()
        session.query(ShowdownLineup).filter_by(contest_id=contest_id).delete()
        session.query(ShowdownContestMetadata).filter_by(contest_id=contest_id).delete()
        
        session.bulk_save_objects(player_records)
        session.bulk_save_objects(lineup_records)
        session.add(metadata)
        session.commit()
        
        log.info(f"✓ Contest {contest_id}: {len(lineup_records)} unique lineups, "
                f"{len(player_records)} player records")
        return True
        
    except Exception as e:
        session.rollback()
        log.error(f"✗ Failed contest {contest_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_contest(session: Session, hc: HistoricalContest, df: pd.DataFrame,
                    username: Optional[str] = None) -> bool:
    """
    Store the field's actual scores (and our own entries, if `username` is
    given), then route to the processor for the contest type.
    """
    try:
        store_field_results(session, hc, df, username=username)
    except Exception:
        session.rollback()
        log.exception(f"Contest {hc.contest_id}: storing field results failed")
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
        resolve_snapshot_date(session, hc.date),
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