import random
import re
from typing import Dict, List, Optional, Tuple
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
            position="Unknown",
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


def fetch_dk_field_lineups(session: Session, contest_ids: List[int], max_lineups: int = 10000,
                            user_data_dir: Optional[str] = None, profile_directory: str = "Default",
                            seed: Optional[int] = 42, sess=None, snapshot_date=None,
                            unmatched_tracker: dict = None, config: Optional[dict] = None) -> tuple[List[List[int]], int, int]:
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

    for cid in contest_ids:
        df = download_contest_dataframe(sess=sess, contest_id=int(cid))
        if df is None or df.empty:
            continue

        cols = [str(c).strip() for c in df.columns]
        lineup_col = next((c for c in cols if c.lower() == 'lineup'), None)
        if lineup_col:
            tmp = df[[lineup_col]].dropna().drop_duplicates()
            for _, row in tmp.iterrows():
                total_attempted += 1
                lineup_str = str(row[lineup_col])
                slot_pairs = _parse_lineup_positions(lineup_str)
                if not slot_pairs:
                    continue

                # Build locked assignments: player_id -> slot_name for known names
                locked: Dict[int, str] = {}
                known_count = 0
                name_map = _build_name_map(session, snapshot_date=snapshot_date)
                for slot, maybe_name in slot_pairs[:8]:
                    if maybe_name:
                        pid = name_map.get(normalize_name(maybe_name)) or name_map.get(maybe_name)
                        if pid is None:
                            if unmatched_tracker is not None:
                                nm = normalize_name(maybe_name)
                                unmatched_tracker[nm] = unmatched_tracker.get(nm, 0) + 1
                            # Treat as missing (not locked)
                        else:
                            # Only lock if slot label is valid
                            slot_up = slot.upper()
                            if slot_up in SLOT_NAMES:
                                locked[int(pid)] = slot_up
                                known_count += 1

                # Try to complete lineup with optimizer, respecting locked players
                # First attempt with provided config, then with relaxed constraints if needed
                config_attempts: List[dict] = []
                if config:
                    config_attempts.append(config)
                # Always include a safe fallback with min_salary relaxed
                fallback_cfg = deepcopy(config or {})
                # Navigate to min_salary in a tolerant way
                node = fallback_cfg
                node.setdefault("classic", {}).setdefault("lineup_generation", {})
                node["classic"]["lineup_generation"].setdefault("salary_cap", 50000)
                # Our core model no longer enforces min_salary; keep a low cap override just in case
                node["classic"]["lineup_generation"]["min_salary"] = 0
                config_attempts.append(fallback_cfg)

                completed_lineup: Optional[List[int]] = None
                for idx, cfg_attempt in enumerate(config_attempts):
                    try:
                        completed_lineup = generate_single_classic_lineup(
                            players=player_pool,
                            base_scores=base_scores,
                            config=cfg_attempt,
                            locked_assignments=locked,
                            attempt_tag=str(idx + 1),
                            debug_lp_path='/home/sam-pratt/ivan/data/nba/logs/'
                        )
                    except Exception:
                        completed_lineup = None
                    if completed_lineup:
                        break

                if not completed_lineup:
                    # Skip if infeasible
                    continue

                key = tuple(completed_lineup)
                if key not in seen:
                    seen.add(key)
                    if known_count < 8:
                        lineups_completed += 1
                    all_lineups.append(completed_lineup)

    if not all_lineups:
        return ([], lineups_completed, total_attempted)
    random.seed(seed or 42)
    if len(all_lineups) > int(max_lineups):
        sampled = random.sample(all_lineups, int(max_lineups))
        return (sampled, lineups_completed, total_attempted)
    return (all_lineups, lineups_completed, total_attempted)

