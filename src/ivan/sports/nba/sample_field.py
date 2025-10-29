import numpy as np
from sqlalchemy.orm import Session
from typing import Dict
from ivan.sports.nba.models import Lineup, NBAPlayer


def sample_field(
        session: Session,
        field_size: int,
        cfg: dict
) -> Dict[int, int]:
    """
    Sample the fantasy basketball field with replacement based on player projection likelihood.
    
    Args:
        session: SQLAlchemy session for database access
        field_size: Number of lineups to sample for the field
        cfg: Configuration dict (currently tunable for projection weighting)
    
    Returns:
        Dictionary mapping lineup_id -> count (number of times that lineup appears in the sample)
    """
    
    # 1) Pull all lineups from DB
    all_lineups = session.query(Lineup).all()
    
    if not all_lineups:
        raise ValueError("No lineups found in database")
    
    lineup_ids = []
    lineup_projection_sums = []
    
    # 2) Calculate each lineup's sum of player fpts projections
    for lineup in all_lineups:
        # Get player IDs from the lineup
        # Assuming Lineup has a player_ids_list field (JSON or similar)
        player_ids = lineup.player_ids_list
        
        if not player_ids:
            # Handle empty lineups - assign minimal projection
            total_projection = 0.0
        else:
            # Coerce to list of ints if needed
            if isinstance(player_ids, str):
                import json
                try:
                    player_ids = json.loads(player_ids)
                except:
                    player_ids = []
            
            if not isinstance(player_ids, list):
                player_ids = list(player_ids) if hasattr(player_ids, '__iter__') else []
            
            player_ids = [int(pid) for pid in player_ids if pid]
            
            # Query players and sum their fpts projections
            players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(player_ids)).all()
            
            # Create a dict for quick lookup
            player_dict = {p.id: p for p in players}
            
            # Sum projections for players in this lineup
            total_projection = sum(
                player_dict[pid].fpts_mean 
                for pid in player_ids 
                if pid in player_dict and player_dict[pid].fpts_mean is not None
            )
        
        lineup_ids.append(lineup.id)
        lineup_projection_sums.append(float(total_projection))
    
    # 3) Sample the field with replacement based on likelihood of projections
    lineup_projection_sums = np.array(lineup_projection_sums, dtype=np.float64)
    
    # Handle edge case where all projections are zero or negative
    min_projection = np.min(lineup_projection_sums)
    if min_projection <= 0:
        # Shift all projections so minimum is slightly positive
        lineup_projection_sums = lineup_projection_sums - min_projection + 1.0
    
    # Normalize to probability distribution
    total_projection = np.sum(lineup_projection_sums)
    if total_projection <= 0:
        # Fallback to uniform distribution if all projections are zero/negative
        probabilities = np.ones(len(lineup_ids)) / len(lineup_ids)
    else:
        probabilities = lineup_projection_sums / total_projection
    
    # Apply any tuning from config (e.g., smoothing parameter)
    # cfg might contain a 'projection_exponent' to make distribution flatter or sharper
    exponent = cfg.get('projection_exponent', 1.0) if cfg else 1.0
    if exponent != 1.0:
        probabilities = np.power(probabilities, exponent)
        probabilities = probabilities / np.sum(probabilities)
    
    # Sample lineup indices based on probabilities
    sampled_indices = np.random.choice(
        len(lineup_ids),
        size=field_size,
        replace=True,
        p=probabilities
    )
    
    # Count occurrences of each lineup
    counts_by_lineup = {}
    unique_indices, counts = np.unique(sampled_indices, return_counts=True)
    
    for idx, count in zip(unique_indices, counts):
        lineup_id = lineup_ids[idx]
        counts_by_lineup[lineup_id] = int(count)
    
    return counts_by_lineup