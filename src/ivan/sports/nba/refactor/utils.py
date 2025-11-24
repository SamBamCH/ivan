from typing import Optional, Iterable, Tuple
import re, json

def parse_position(position: str) -> Optional[str]:
    """Parse position string (PG, SG, etc.) into simplified form (G, F, C)."""
    if not position:
        return None
    position = position.strip().upper()

    guard_positions = {'PG', 'SG'}
    forward_positions = {'SF', 'PF'}
    center_positions = {'C'}

    if '/' in position:
        positions = [p.strip() for p in position.split('/')]
        for pos in positions:
            if pos in guard_positions:
                return 'G'
            elif pos in forward_positions:
                return 'F'
            elif pos in center_positions:
                return 'C'
    else:
        if position in guard_positions:
            return 'G'
        elif position in forward_positions:
            return 'F'
        elif position in center_positions:
            return 'C'

    return None

def american_to_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)
    

def _coerce_positions(pos: Iterable[str] | str) -> Tuple[str, ...]:
    """
    Normalize position(s):
      - "WR"            -> ("WR",)
      - "WR/TE"         -> ("TE", "WR")
      - ["RB","WR"]     -> ("RB","WR")
      - " DST ,  "      -> ("DST",)
    Always returns a tuple for consistency.
    """
    if isinstance(pos, str):
        tokens = [p.strip().upper() for p in re.split(r"[\\/ ,]", pos) if p.strip()]
    else:
        tokens = [str(p).strip().upper() for p in pos if str(p).strip()]
    
    uniq = sorted(set(tokens))
    
    # ALWAYS return a tuple, even for single positions
    return tuple(uniq)

def _cfg(config: dict, path: str, default):
    """Quiet dotted lookup with default (no warnings)."""
    cur = config or {}
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur

def _coerce_player_ids(val): 
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