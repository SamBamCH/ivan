from typing import Dict, List, Tuple, Optional, Iterable, Union
import re
import json


def _coerce_positions(pos: Iterable[str] | str) -> Union[str, Tuple[str, ...]]:
    """
    Normalize position(s):
      - "WR"            -> "WR"
      - "WR/TE"         -> ("TE", "WR")
      - ["RB","WR"]     -> ("RB","WR")
      - " DST ,  "      -> "DST"
    """
    if isinstance(pos, str):
        tokens = [p.strip().upper() for p in re.split(r"[\\/ ,]", pos) if p.strip()]
    else:
        tokens = [str(p).strip().upper() for p in pos if str(p).strip()]

    uniq = sorted(set(tokens))
    if not uniq:
        return tuple()
    if len(uniq) == 1:
        return uniq[0]
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