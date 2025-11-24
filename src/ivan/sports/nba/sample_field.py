import pickle
from pathlib import Path

import random
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from sqlalchemy.orm import Session

from ivan.sports.nba.models import Lineup, NBAPlayer
from ivan.core.utils import normalize_name
from ivan.sports.common.draftkings import (
    build_requests_session_from_chrome,
    download_contest_dataframe,
)
from ivan.sports.nba.utils import _coerce_positions
from ivan.sports.nba.lineup_generation import (
    FieldPlayer,
    generate_single_classic_lineup,
    SLOT_NAMES,
    SLOT_ELIGIBILITY,
)
from copy import deepcopy


def sample_field(session: Session, field_size: int, cfg: dict) -> Dict[int, int]:
    """Sample field with replacement based on projection likelihood. Returns {lineup_id: count}."""
    all_lineups = session.query(Lineup).all()
    if not all_lineups:
        raise ValueError("No lineups found in database")
    
    lineup_ids = []
    lineup_projection_sums = []
    
    for lineup in all_lineups:
        player_ids = lineup.player_ids_list
        if not player_ids:
            total_projection = 0.0
        else:
            if isinstance(player_ids, str):
                import json
                try:
                    player_ids = json.loads(player_ids)
                except:
                    player_ids = []
            if not isinstance(player_ids, list):
                player_ids = list(player_ids) if hasattr(player_ids, '__iter__') else []
            player_ids = [int(pid) for pid in player_ids if pid]
            players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(player_ids)).all()
            player_dict = {p.id: p for p in players}
            total_projection = sum(
                player_dict[pid].fpts_mean 
                for pid in player_ids 
                if pid in player_dict and player_dict[pid].fpts_mean is not None
            )
        lineup_ids.append(lineup.id)
        lineup_projection_sums.append(float(total_projection))
    
    lineup_projection_sums = np.array(lineup_projection_sums, dtype=np.float64)
    min_projection = np.min(lineup_projection_sums)
    if min_projection <= 0:
        lineup_projection_sums = lineup_projection_sums - min_projection + 1.0
    
    total_projection = np.sum(lineup_projection_sums)
    if total_projection <= 0:
        probabilities = np.ones(len(lineup_ids)) / len(lineup_ids)
    else:
        probabilities = lineup_projection_sums / total_projection
    
    exponent = cfg.get('projection_exponent', 1.0) if cfg else 1.0
    if exponent != 1.0:
        probabilities = np.power(probabilities, exponent)
        probabilities = probabilities / np.sum(probabilities)
    
    sampled_indices = np.random.choice(len(lineup_ids), size=field_size, replace=True, p=probabilities)
    counts_by_lineup = {}
    unique_indices, counts = np.unique(sampled_indices, return_counts=True)
    for idx, count in zip(unique_indices, counts):
        counts_by_lineup[lineup_ids[idx]] = int(count)
    
    return counts_by_lineup


def sample_field_with_projection_model(
        session: Session, 
        field_size: int, 
        cfg: dict, 
        model_path: str = "data/nba/models/field_distribution_model.pkl", 
        contest_params: dict = None
) -> Dict[int, int]:
    """
    Sample field using the projection-based field model.
    contest_params: dict with keys [entry_fee, total_entries, multi_entry]
    """
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler = model['scaler']
    proj_model = model['proj_model']
    bin_edges = model['bin_edges']
    projection_min = model.get('projection_min', 200.0)  # Default to 200 if not stored

    entry_fee = contest_params.get('entry_fee', 4.0)
    total_entries = contest_params.get('total_entries', 20000)
    multi_entry = contest_params.get('multi_entry', 150)

    X = np.array([[entry_fee, total_entries, multi_entry]])
    X_scaled = scaler.transform(X)
    proj_dist = proj_model.predict(X_scaled)[0]  # shape (BINS,)

    # Get all lineups and their projection sums
    all_lineups = session.query(Lineup).all()
    if not all_lineups:
        raise ValueError("No lineups found in database")
    lineup_ids = []
    lineup_projection_sums = []
    for lineup in all_lineups:
        player_ids = lineup.player_ids_list
        if not player_ids:
            total_projection = 0.0
        else:
            if isinstance(player_ids, str):
                import json
                try:
                    player_ids = json.loads(player_ids)
                except:
                    player_ids = []
            if not isinstance(player_ids, list):
                player_ids = list(player_ids) if hasattr(player_ids, '__iter__') else []
            player_ids = [int(pid) for pid in player_ids if pid]
            players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(player_ids)).all()
            player_dict = {p.id: p for p in players}
            total_projection = sum(
                player_dict[pid].fpts_mean 
                for pid in player_ids 
                if pid in player_dict and player_dict[pid].fpts_mean is not None
            )
        
        # Drop lineups below projection_min
        if total_projection >= projection_min:
            lineup_ids.append(lineup.id)
            lineup_projection_sums.append(float(total_projection))
    
    if not lineup_ids:
        raise ValueError(f"No lineups found with projection >= {projection_min}")
    
    lineup_projection_sums = np.array(lineup_projection_sums, dtype=np.float64)

    # Normalize lineup projections to [0, 1] using projection_min as min and max_proj as max
    max_proj = lineup_projection_sums.max()
    if max_proj > projection_min:
        norm_proj = (lineup_projection_sums - projection_min) / (max_proj - projection_min)
    else:
        norm_proj = np.zeros_like(lineup_projection_sums)
    
    # Clip to valid range [0, 1)
    norm_proj = np.clip(norm_proj, 0, 1 - 1e-8)
    
    # Bin each normalized projection
    bin_indices = np.floor(norm_proj * (len(bin_edges) - 1)).astype(int)
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

    # Group lineups by bin
    lineups_by_bin = {}
    for idx, bin_idx in enumerate(bin_indices):
        if bin_idx not in lineups_by_bin:
            lineups_by_bin[bin_idx] = []
        lineups_by_bin[bin_idx].append(idx)

    # Sample bins according to proj_dist, then sample lineups within each bin
    # This forces the field to match the histogram distribution
    num_bins = len(proj_dist)
    
    # Normalize proj_dist to get probabilities
    bin_probs = proj_dist / proj_dist.sum() if proj_dist.sum() > 0 else np.ones(num_bins) / num_bins
    
    # Sample which bins to draw from (with replacement)
    sampled_bins = np.random.choice(num_bins, size=field_size, replace=True, p=bin_probs)
    
    # For each sampled bin, randomly pick a lineup from that bin
    counts_by_lineup = {}
    for bin_idx in sampled_bins:
        if bin_idx in lineups_by_bin and lineups_by_bin[bin_idx]:
            # Randomly pick a lineup from this bin
            lineup_idx = np.random.choice(lineups_by_bin[bin_idx])
            lineup_id = lineup_ids[lineup_idx]
            counts_by_lineup[lineup_id] = counts_by_lineup.get(lineup_id, 0) + 1
        else:
            # Bin has no lineups - fallback to sampling from all lineups
            lineup_idx = np.random.choice(len(lineup_ids))
            lineup_id = lineup_ids[lineup_idx]
            counts_by_lineup[lineup_id] = counts_by_lineup.get(lineup_id, 0) + 1
    
    unique_lineups = len(counts_by_lineup)
    total_entries = sum(counts_by_lineup.values())
    avg_dupe = total_entries / unique_lineups if unique_lineups > 0 else 0
    
    print(f"Sampled from {len(lineup_ids)} lineups (dropped {len(all_lineups) - len(lineup_ids)} below {projection_min:.0f})")
    print(f"Field composition: {unique_lineups} unique lineups, {total_entries} total entries (avg {avg_dupe:.1f}x duplication)")
    
    return counts_by_lineup


def fetch_db_field_lineups(session: Session, num_lineups: int) -> List[List[int]]:
    """Return up to num_lineups from DB, sampled uniformly without replacement."""
    rows: List[Lineup] = session.query(Lineup).all()
    if not rows:
        return []

    lineups: List[List[int]] = []
    for r in rows:
        pids = r.player_ids_list or []
        if isinstance(pids, str):
            try:
                import json as _json
                pids = _json.loads(pids)
            except Exception:
                pids = []
        pids = [int(x) for x in (pids or []) if x is not None]
        if len(pids) == 8:
            lineups.append(pids)
    if not lineups:
        return []
    k = min(int(num_lineups), len(lineups))
    return random.sample(lineups, k)




_POS_RE = re.compile(r"\b(PG|SG|SF|PF|C|G|F|UTIL|LOCKED)\b", flags=re.IGNORECASE)
_STRICT_POS_RE = re.compile(r"\b(PG|SG|SF|PF|C|G|F|UTIL)\b", flags=re.IGNORECASE)

def _parse_lineup_string(lineup_str: str) -> List[str]:
    """Parse DK 'Lineup' text into 8 player names. 'LOCKED' positions return special marker.

    Note: This legacy parser ignores slot labels; prefer _parse_lineup_positions for slot-aware parsing.
    """
    if not isinstance(lineup_str, str) or not lineup_str.strip():
        return []
    lineup_str = re.sub(r"\s+", " ", lineup_str).strip()
    parts = [p.strip().strip(',') for p in _POS_RE.split(lineup_str) if p and p.strip()]
    names: List[str] = []
    pos_tokens = {"PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "LOCKED"}
    i = 0
    while i < len(parts) - 1 and len(names) < 8:
        tok = parts[i].upper()
        nxt = parts[i + 1]
        if tok in pos_tokens:
            if tok == "LOCKED":
                # Use special marker for locked positions - will be filtered out
                names.append("__LOCKED_PLAYER__")
            else:
                clean = _POS_RE.sub("", nxt).strip().strip(',')
                names.append(clean)
            i += 2
        else:
            i += 1
    return names[:8]


def _parse_lineup_positions(lineup_str: str) -> List[Tuple[str, Optional[str]]]:
    """Parse DK 'Lineup' text into ordered (slot, player_name|None) pairs.

    We treat tokens PG, SG, SF, PF, C, G, F, UTIL as slot labels. The chunk of text
    following each label up to the next label (or end) is considered the player name.
    If the value contains 'LOCKED' or is empty/placeholder, we return None for the name.
    """
    if not isinstance(lineup_str, str) or not lineup_str.strip():
        print("error parsing lineup")
        return []
    s = re.sub(r"\s+", " ", lineup_str).strip()

    # Find all slot tokens with their positions in the string
    matches = list(_STRICT_POS_RE.finditer(s))
    pairs: List[Tuple[str, Optional[str]]] = []
    for i, m in enumerate(matches):
        slot = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        raw_val = s[start:end].strip().strip(',')
        # Clean and detect locked/unknown
        val = _POS_RE.sub("", raw_val).strip().strip(',')
        if not val or val.upper() == "LOCKED":
            name = None
        else:
            name = val
        pairs.append((slot, name))

    # Ensure max 8 and in the official slot order if possible
    # If DK order differs, we keep encountered order; otherwise, reorder by SLOT_NAMES when lengths match
    if len(pairs) == 8:
        return pairs
    # Fallback: try to reconstruct in SLOT_NAMES order picking the first occurrence
    ordered: List[Tuple[str, Optional[str]]] = []
    used_idx = set()
    for slot in SLOT_NAMES:
        idx = next((i for i, (sl, _) in enumerate(pairs) if i not in used_idx and sl.upper() == slot), None)
        if idx is None:
            # Missing slot label; append empty
            ordered.append((slot, None))
        else:
            used_idx.add(idx)
            ordered.append(pairs[idx])
    return ordered[:8]


def _build_name_map(session: Session, snapshot_date=None) -> Dict[str, int]:
    """Return normalized_name -> player_id map from NBAPlayer and ProjectionSnapshot."""
    from ivan.sports.nba.models import ProjectionSnapshot
    from datetime import datetime, timedelta
    
    name_map: Dict[str, int] = {}
    players = session.query(NBAPlayer).all()
    for p in players:
        # Store both raw and normalized keys to improve matching robustness
        name_map[p.name] = int(p.id)
        name_map[normalize_name(p.name)] = int(p.id)
    
    query = session.query(ProjectionSnapshot)
    if snapshot_date is not None:
        if isinstance(snapshot_date, datetime):
            snapshot_date = snapshot_date.date()
        day_start = datetime.combine(snapshot_date, datetime.min.time())
        day_end = day_start + timedelta(days=1)
        query = query.filter(
            ((ProjectionSnapshot.snapshot_date >= day_start) & 
             (ProjectionSnapshot.snapshot_date < day_end)) |
            (ProjectionSnapshot.id < 0)
        )
    
    snapshots = query.all()
    for snap in snapshots:
        if snap.player_name not in name_map:
            name_map[snap.player_name] = int(snap.id)
        norm = normalize_name(snap.player_name)
        if norm not in name_map:
            name_map[norm] = int(snap.id)
    
    return name_map


def _ensure_placeholder_projection(session: Session, player_id: int = -1, 
                                    default_fpts_mean: float = 10.0, 
                                    default_fpts_std: float = 5.0) -> None:
    """Ensure ProjectionSnapshot exists for placeholder player ID."""
    from ivan.sports.nba.models import ProjectionSnapshot
    from datetime import datetime
    
    existing = session.query(ProjectionSnapshot).filter(ProjectionSnapshot.id == player_id).first()
    if not existing:
        placeholder = ProjectionSnapshot(
            id=player_id,
            player_name=f"UNKNOWN_PLAYER_{abs(player_id)}",
            snapshot_date=datetime.now(),
            team="Unknown",
            opponent="Unknown",
            position="PG/SG/SF/PF/C",  # Make eligible for all positions to avoid infeasible lineups
            roster_slot_id=0,
            gametime=datetime.now(),
            salary=0,
            status="Unknown",
            fpts_mean=default_fpts_mean,
            fpts_std=default_fpts_std
        )
        session.add(placeholder)
        session.commit()
        from rich.console import Console
        console = Console()
        console.print(f"[cyan]ℹ Created placeholder (ID {player_id}, mean={default_fpts_mean:.1f})[/cyan]")


def _map_names_to_ids(session: Session, names: List[str], snapshot_date=None, 
                       default_player_id: int = -1, unmatched_tracker: dict = None) -> Optional[List[int]]:
    """Map player names to IDs. Uses default_player_id for missing players.
    
    Args:
        unmatched_tracker: Optional dict to accumulate unmatched names (format: {name: count})
    """
    name_map = _build_name_map(session, snapshot_date=snapshot_date)
    ids: List[int] = []
    missing_names: List[str] = []
    
    for nm in names:
        # Skip LOCKED placeholder markers
        if nm == "__LOCKED_PLAYER__":
            ids.append(default_player_id)
            continue
            
        nm_norm = normalize_name(nm)
        pid = name_map.get(nm_norm)
        if pid is None:
            missing_names.append(nm_norm)
            ids.append(default_player_id)
        else:
            ids.append(pid)
    
    if missing_names:        
        _ensure_placeholder_projection(session, player_id=default_player_id)
        # Track unmatched names instead of printing immediately
        if unmatched_tracker is not None:
            for name in missing_names:
                unmatched_tracker[name] = unmatched_tracker.get(name, 0) + 1
    
    return ids


def _complete_lineups_chunk(
    slot_pairs_chunk: List[List[Tuple[str, Optional[str]]]],
    name_map: Dict[str, int],
    player_pool: List[FieldPlayer],
    base_scores: Dict[int, float],
    config: Optional[dict],
    debug_lp_path: str,
    lock_time: Optional[datetime] = None,
) -> Tuple[List[List[int]], int, Dict[str, int], int]:
    """Complete a chunk of DK lineups using optimizer, respecting locked names.

    Returns: (completed_lineups, lineups_completed_count, unmatched_counter)
    """
    from copy import deepcopy as _deepcopy
    completed: List[List[int]] = []
    lineups_completed = 0
    infeasible_count = 0
    unmatched_counter: Dict[str, int] = {}

    # Build config attempts list once per chunk
    # First attempt: ALWAYS relax team cap to avoid infeasibility from many same-team locks
    config_attempts: List[dict] = []
    primary_cfg = _deepcopy(config or {})
    node0 = primary_cfg
    node0.setdefault("classic", {}).setdefault("lineup_generation", {})
    node0["classic"]["lineup_generation"].setdefault("salary_cap", 50000)
    node0["classic"]["lineup_generation"]["min_salary"] = 0
    node0["classic"]["lineup_generation"]["max_salary_per_team"] = None
    config_attempts.append(primary_cfg)

    # Second attempt: identical fallback (kept for future toggles); currently same as primary
    fallback_cfg = _deepcopy(primary_cfg)
    config_attempts.append(fallback_cfg)

    # Quick id -> FieldPlayer map for lock checks
    id_to_fp: Dict[int, FieldPlayer] = {fp.id: fp for fp in player_pool}

    for slot_pairs in slot_pairs_chunk:
        # Build locked assignments
        locked: Dict[int, str] = {}
        known_count = 0
        for slot, maybe_name in slot_pairs[:8]:
            if not maybe_name:
                continue
            pid = name_map.get(normalize_name(maybe_name)) or name_map.get(maybe_name)
            if pid is None:
                nm = normalize_name(maybe_name)
                unmatched_counter[nm] = unmatched_counter.get(nm, 0) + 1
                continue
            slot_up = slot.upper()
            if slot_up in SLOT_NAMES:
                # ALWAYS lock known players to preserve field lineup integrity
                # We are reconstructing actual lineups, not optimizing them
                fp = id_to_fp.get(int(pid))
                
                # Verify eligibility for the given slot to avoid infeasibility
                if fp:
                    pos_tuple = fp.position if isinstance(fp.position, (tuple, list)) else (fp.position,)
                    if any(pos in SLOT_ELIGIBILITY.get(slot_up, set()) for pos in pos_tuple):
                        locked[int(pid)] = slot_up # i.e. {12345: PG}
                        known_count += 1
                    else:
                        # Player exists but not eligible for this slot?
                        # This implies a data mismatch between DK lineup and our player DB
                        print(f"[field-parser] Warning: Player {pid} ({fp.name}) not eligible for {slot_up} (pos={fp.position})")
                else:
                     # Player ID found in name_map but not in player_pool?
                     # This shouldn't happen if player_pool is built from same sources
                     pass

        # Try to complete lineup with optimizer
        completed_lineup: Optional[List[int]] = None
        for idx, cfg_attempt in enumerate(config_attempts):
            try:
                completed_lineup = generate_single_classic_lineup(
                    players=player_pool,
                    base_scores=base_scores,
                    config=cfg_attempt,
                    locked_assignments=locked,
                    attempt_tag=str(idx + 1),
                    lock_time=lock_time,
                )
            except Exception as e:
                print(f"[field-parser] Error optimizing lineup: {e}")
                import traceback
                traceback.print_exc()
                completed_lineup = None
            if completed_lineup:
                break

        if completed_lineup:
            completed.append(completed_lineup)
            if known_count < 8:
                lineups_completed += 1
        else:
            # Log why it failed (infeasible)
            # Only print first few failures to avoid spamming
            if infeasible_count < 5:
                print(f"[field-parser] Infeasible lineup (locked={locked})")
            infeasible_count += 1

    return completed, lineups_completed, unmatched_counter, infeasible_count


def fetch_dk_field_lineups(session: Session, contest_ids: List[int], max_lineups: int = 10000,
                            user_data_dir: Optional[str] = None, profile_directory: str = "Default",
                            seed: Optional[int] = 42, sess=None, snapshot_date=None,
                            unmatched_tracker: dict = None, config: Optional[dict] = None,
                            lock_time: Optional[datetime] = None) -> tuple[List[List[int]], int, int]:
    """Download DK standings CSV and extract lineup player-id lists. De-dupes and samples.
    
    Args:
        unmatched_tracker: Optional dict to accumulate unmatched names (format: {name: count})
        
    Returns:
        Tuple of (lineups, lineups_with_defaults, total_attempted)
    """
    user_data_dir = user_data_dir or "/home/sam-pratt/ivan/chrome-selenium/dk-profile"
    if sess is None:
        sess = build_requests_session_from_chrome(
            user_data_dir=user_data_dir, profile_directory=profile_directory, headless=False
        )

    all_lineups: List[List[int]] = []
    seen: set[tuple[int, ...]] = set()
    # Count how many required completion (had missing or unmatched names)
    lineups_completed = 0
    total_attempted = 0

    # Build optimization player pool and base scores once
    raw_players: List[NBAPlayer] = session.query(NBAPlayer).all()
    player_pool: List[FieldPlayer] = []
    base_scores: Dict[int, float] = {}

    def _extend_pool_from_nba_players(rows: List[NBAPlayer]):
        for rp in rows:
            pos_tuple = _coerce_positions(getattr(rp, "position", []) or [])
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
            player_pool.append(
                FieldPlayer(
                    id=int(rp.id),
                    name=getattr(rp, "name"),
                    salary=int(getattr(rp, "salary") or 0),
                    team=team,
                    opponent=opponent,
                    position=pos_tuple,
                    fpts_mean=float(getattr(rp, "fpts_mean") or 0.0),
                    game_id=game_id,
                    gametime=getattr(rp, "gametime", None),
                )
            )
            base_scores[int(rp.id)] = float(getattr(rp, "fpts_mean") or 0.0)

    def _extend_pool_from_snapshots(date_val=None):
        from datetime import datetime, timedelta
        from ivan.sports.nba.models import ProjectionSnapshot
        query = session.query(ProjectionSnapshot)
        if date_val is not None:
            if isinstance(date_val, datetime):
                day_start = datetime.combine(date_val.date(), datetime.min.time())
            else:
                # date object
                day_start = datetime.combine(date_val, datetime.min.time())
            day_end = day_start + timedelta(days=1)
            query = query.filter(
                (ProjectionSnapshot.snapshot_date >= day_start) &
                (ProjectionSnapshot.snapshot_date < day_end)
            )
        snaps = query.all()
        for s in snaps:
            pos_tuple = _coerce_positions(getattr(s, "position", []) or [])
            if isinstance(pos_tuple, str):
                pos_tuple = (pos_tuple,)
            elif not isinstance(pos_tuple, tuple):
                pos_tuple = tuple(pos_tuple) if pos_tuple else ()
            team = getattr(s, "team", "")
            opponent = getattr(s, "opponent", None)
            if team and opponent:
                game_id = "-".join(sorted([team.upper(), opponent.upper()]))
            else:
                game_id = None
            pid = int(getattr(s, "id"))
            # Avoid duplicate IDs if already present from NBAPlayer
            if any(fp.id == pid for fp in player_pool):
                continue
            player_pool.append(
                FieldPlayer(
                    id=pid,
                    name=getattr(s, "player_name"),
                    salary=int(getattr(s, "salary") or 0),
                    team=team,
                    opponent=opponent,
                    position=pos_tuple,
                    fpts_mean=float(getattr(s, "fpts_mean") or 0.0),
                    game_id=game_id,
                    gametime=getattr(s, "gametime", None),
                )
            )
            base_scores[pid] = float(getattr(s, "fpts_mean") or 0.0)

    # Always pull current NBAPlayer rows (current player set) AND same-day ProjectionSnapshot rows.
    # NBAPlayer: current active players (full-season metadata)
    # ProjectionSnapshot: locked state / point-in-time projections for the given date.
    _extend_pool_from_nba_players(raw_players)
    _extend_pool_from_snapshots(snapshot_date)

    # Build a global name_map once (uses current players and same-day snapshots)
    name_map = _build_name_map(session, snapshot_date=snapshot_date)

    # Parallel controls from config
    workers = 8
    chunk_size = 500
    if config:
        workers = int(config.get("classic", {}).get("field_parallel_workers", 8) or 8)
        chunk_size = int(config.get("classic", {}).get("field_parallel_chunk_size", 500) or 500)

    debug_lp_path = '/home/sam-pratt/ivan/data/nba/logs/'

    # Fallback: if lock_time not provided, default to current Central Time (naive) for late-swap style filtering
    if lock_time is None:
        try:
            from zoneinfo import ZoneInfo
            lock_time = datetime.now(ZoneInfo("America/Chicago")).replace(tzinfo=None)
        except Exception:
            lock_time = datetime.now().replace(tzinfo=None)

    for cid in contest_ids:
        df = download_contest_dataframe(sess=sess, contest_id=int(cid))
        if df is None or df.empty:
            print('no data returned from draftkings')
            break

        cols = [str(c).strip() for c in df.columns]
        lineup_col = next((c for c in cols if c.lower() == 'lineup'), None)
        if lineup_col:
            tmp = df[[lineup_col]].dropna().drop_duplicates()
            
            # OPTIMIZATION: Sample lineups BEFORE parsing to avoid processing entire contest
            # If we have more than max_lineups, sample down first
            if len(tmp) > int(max_lineups):
                random.seed(seed or 42)
                tmp = tmp.sample(n=int(max_lineups), random_state=seed or 42)
            
            # Prepare work: parse to slot_pairs ahead of time
            slot_pairs_list: List[List[Tuple[str, Optional[str]]]] = []
            for _, row in tmp.iterrows():
                lineup_str = str(row[lineup_col])
                slot_pairs = _parse_lineup_positions(lineup_str)
                if slot_pairs:
                    slot_pairs_list.append(slot_pairs)
            total_attempted += len(slot_pairs_list)

            if workers and workers > 1:
                # Process in parallel by chunks
                chunks: List[List[List[Tuple[str, Optional[str]]]]] = [
                    slot_pairs_list[i:i + chunk_size] for i in range(0, len(slot_pairs_list), chunk_size)
                ]
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = [
                        ex.submit(
                            _complete_lineups_chunk,
                            chunk,
                            name_map,
                            player_pool,
                            base_scores,
                            config,
                            debug_lp_path,
                            lock_time,  # Pass lock_time to parallel workers
                        )
                        for chunk in chunks
                    ]
                    infeasible_total = 0
                    for fut in as_completed(futs):
                        comp, completed_count, unmatched, infeasible_count = fut.result()
                        # Deduplicate and accumulate
                        for lu in comp:
                            key = tuple(lu)
                            if key in seen:
                                continue
                            seen.add(key)
                            all_lineups.append(lu)
                        lineups_completed += int(completed_count)
                        infeasible_total += int(infeasible_count)
                        if unmatched_tracker is not None and unmatched:
                            for k, v in unmatched.items():
                                unmatched_tracker[k] = unmatched_tracker.get(k, 0) + int(v)
                    # Simple progress diagnostic
                    print(f"[field-parser] chunks complete: attempted={total_attempted} feasible={len(all_lineups)} infeasible={infeasible_total}")
            else:
                # Sequential fallback (still faster due to single name_map)
                comp, completed_count, unmatched, infeasible_count = _complete_lineups_chunk(
                    slot_pairs_list, name_map, player_pool, base_scores, config, debug_lp_path, lock_time=lock_time
                )
                for lu in comp:
                    key = tuple(lu)
                    if key in seen:
                        continue
                    seen.add(key)
                    all_lineups.append(lu)
                lineups_completed += int(completed_count)
                if unmatched_tracker is not None and unmatched:
                    for k, v in unmatched.items():
                        unmatched_tracker[k] = unmatched_tracker.get(k, 0) + int(v)
                print(f"[field-parser] sequential complete: attempted={total_attempted} feasible={len(all_lineups)} infeasible={infeasible_count}")

    if not all_lineups:
        return ([], lineups_completed, total_attempted)
    random.seed(seed or 42)
    if len(all_lineups) > int(max_lineups):
        sampled = random.sample(all_lineups, int(max_lineups))
        return (sampled, lineups_completed, total_attempted)
    return (all_lineups, lineups_completed, total_attempted)

