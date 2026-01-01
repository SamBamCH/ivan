from sqlalchemy.orm import Session
from typing import Dict
from ivan.sports.nba.models import NBAPlayer
from ivan.core.utils import normalize_name

def _build_name_map(session: Session) -> Dict[str, int]:
    """Return normalized_name -> player_id map from NBAPlayer and ProjectionSnapshot."""
    from ivan.sports.nba.models import ProjectionSnapshot
    from datetime import datetime, timedelta
    
    name_map: Dict[str, int] = {}
    players = session.query(NBAPlayer).all()
    for p in players:
        # Store both raw and normalized keys to improve matching robustness
        name_map[p.name] = int(p.id)
        name_map[normalize_name(p.name)] = int(p.id)
    
    return name_map