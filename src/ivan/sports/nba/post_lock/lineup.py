from typing import List, Tuple, Optional, Dict, Set
import re
from datetime import datetime
import pytz
import random
import logging

from ivan.sports.nba.refactor.pre_lock.lineup_generation import SLOT_NAMES, generate_single_classic_lineup, FieldPlayer
from ivan.core.utils import normalize_name
from ivan.sports.nba.refactor.post_lock.entry_parsing import OriginalEntry

# Track failed name lookups to avoid spamming logs
_failed_names: Set[str] = set()

_POS_RE = re.compile(r"\b(PG|SG|SF|PF|C|G|F|UTIL|LOCKED)\b", flags=re.IGNORECASE)
_STRICT_POS_RE = re.compile(r"\b(PG|SG|SF|PF|C|G|F|UTIL)\b", flags=re.IGNORECASE)

# Gametime is stored as naive datetime in Central Time (America/Chicago)
LOCAL_TZ = pytz.timezone('America/Chicago')

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

def finish_lineup(session, lineup_str: str, name_map: Dict[str, int], id_to_fp: Dict[int, FieldPlayer], base_scores: Dict[int, float]):
    """Finish a DK lineup string into 8 player IDs using precomputed base_scores.

    1. Parse slot/name pairs.
    2. Resolve all provided player names -> IDs.
    3. Only lock players whose games have already started (gametime <= now).
    4. Build player pool: include locked players always; include only future-start
       players for unlocked slots.
    5. Run optimizer once.
    """
    slot_pairs = _parse_lineup_positions(lineup_str)
    # Get current time in Central timezone (naive) to match gametime storage format
    now = datetime.now(LOCAL_TZ).replace(tzinfo=None)
    

    locked_assignments: Dict[int, str] = {}
    all_player_ids: List[int] = []
    
    for slot, maybe_name in slot_pairs[:8]:
        if not maybe_name:
            continue
        normalized = normalize_name(maybe_name)
        pid = name_map.get(normalized)
        if pid is None:
            # Log failed lookups (only once per unique name)
            if normalized not in _failed_names:
                _failed_names.add(normalized)
                logging.warning(f"[field parse] Name not found in DB: '{maybe_name}' (normalized: '{normalized}')")
            return None
        
        all_player_ids.append(pid)
        
        # Only lock if the player's game has already started
        fp = id_to_fp.get(pid)
        if fp and fp.gametime and fp.gametime <= now:
            locked_assignments[pid] = slot
    
    players: List[FieldPlayer] = []
    for fp in id_to_fp.values():
        if fp.id in locked_assignments:
            players.append(fp)
        else:
            if fp.gametime is None or fp.gametime > now:
                players.append(fp)

    if len(players) < 8:
        return None

    # Subset scores for selected pool (fallback to fpts_mean if missing)
    pooled_scores: Dict[int, float] = {}
    for p in players:
        pooled_scores[p.id] = base_scores.get(p.id, float(getattr(p, 'fpts_mean', 0.0) or 0.0))

    return generate_single_classic_lineup(
        players,
        base_scores=pooled_scores,
        locked_assignments=locked_assignments if locked_assignments else None,
        max_salary_per_team=50000
    )
    
def generate_candidate(session, entry: 'OriginalEntry', id_to_fp: Dict[int, FieldPlayer], base_scores: Dict[int, float], jitter: float = 0.0):
    now = datetime.now(LOCAL_TZ).replace(tzinfo=None)

    entry_ids_list = entry.player_ids()  # FIX: call method
    locked_indices = []
    for i, pid in enumerate(entry_ids_list):
        fp = id_to_fp.get(pid)
        if fp and fp.gametime and fp.gametime <= now:
            locked_indices.append(i)

    locked_assignments: Dict[int, str] = {entry_ids_list[i]: SLOT_NAMES[i] for i in locked_indices}

    players: List[FieldPlayer] = []
    for fp in id_to_fp.values():
        if fp.id in locked_assignments:
            players.append(fp)
        else:
            if fp.gametime is None or fp.gametime > now:
                players.append(fp)

    if len(players) < 8:
        return None

    pooled_scores: Dict[int, float] = {p.id: base_scores.get(p.id, float(getattr(p, 'fpts_mean', 0.0) or 0.0)) for p in players}

    # Optional: small jitter on unlocked players to help uniqueness
    if jitter and jitter > 0.0:
        for p in players:
            if p.id not in locked_assignments:
                pooled_scores[p.id] += random.gauss(0.0, jitter)

    return generate_single_classic_lineup(
        players,
        base_scores=pooled_scores,
        locked_assignments=locked_assignments if locked_assignments else None,
    )
