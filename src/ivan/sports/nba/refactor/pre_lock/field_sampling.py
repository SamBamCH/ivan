from sqlalchemy.orm import Session
from typing import Dict
import pickle
import numpy as np
from ivan.sports.nba.models import Lineup, NBAPlayer
import logging

def sample_field(
    session: Session,
    field_size: int,
    contest_params: dict = None
):
    model_path = "data/nba/models/field_distribution_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler = model['scaler']
    proj_model = model['proj_model']
    bin_edges = model['bin_edges']
    projection_min = 200

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

    # Assign probability to each lineup based on its bin
    probabilities = np.array([proj_dist[bin_idx] for bin_idx in bin_indices])
    if probabilities.sum() == 0:
        probabilities = np.ones_like(probabilities) / len(probabilities)
    else:
        probabilities = probabilities / probabilities.sum()

    sampled_indices = np.random.choice(len(lineup_ids), size=field_size, replace=True, p=probabilities)
    counts_by_lineup = {}
    unique_indices, counts = np.unique(sampled_indices, return_counts=True)
    for idx, count in zip(unique_indices, counts):
        counts_by_lineup[lineup_ids[idx]] = int(count)
    
    logging.info(f"Sampled from {len(lineup_ids)} lineups (dropped {len(all_lineups) - len(lineup_ids)} below {projection_min:.0f})")
    
    return counts_by_lineup
