"""
Build team-specific correlation matrices from historical data.

Instead of using a single correlation structure for all teams, this module
builds correlations from each team's actual historical game data to capture
team-specific offensive structures and tendencies.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.sports.nfl.models import NFLStat

# Standard slot labels matching the simulation structure
TEAM10_SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "TE2", "K", "DST"]

# Position to slot prefix mapping
POSITION_TO_SLOT_PREFIX = {
    'QB': 'QB',
    'RB': 'RB',
    'WR': 'WR',
    'TE': 'TE',
    'K': 'K',
    'DST': 'DST',
}


def _assign_season_slot_labels(season_df: pd.DataFrame) -> Dict[str, str]:
    """
    Assign slot labels (QB, RB1, RB2, WR1, etc.) to players for an entire season
    based on their total fantasy_points_ppr across all games.
    
    This ensures slot assignments are stable - a player is WR1 for the season
    even if they get outscored by WR3 in a specific game.
    
    Args:
        season_df: DataFrame with all games for a team in a season
        
    Returns:
        Dict mapping player_name -> slot_label
    """
    # Aggregate total fpts per player across the season
    player_totals = season_df.groupby(['player_name', 'position']).agg({
        'fantasy_points_ppr': 'sum',
        'week': 'count'  # games played
    }).reset_index()
    player_totals.columns = ['player_name', 'position', 'total_fpts', 'games_played']
    
    # Require minimum games to be considered (avoid 1-game wonders)
    min_games = 3
    player_totals = player_totals[player_totals['games_played'] >= min_games]
    
    player_to_slot = {}
    
    for position in ['QB', 'RB', 'WR', 'TE', 'K']:
        pos_players = player_totals[player_totals['position'] == position].copy()
        if pos_players.empty:
            continue
            
        # Sort by total fantasy points descending (season-long production)
        pos_players = pos_players.sort_values('total_fpts', ascending=False)
        
        # Assign slot labels based on season rank
        for i, row in enumerate(pos_players.itertuples()):
            if position == 'QB':
                slot = 'QB' if i == 0 else None
            elif position == 'RB':
                slot = f'RB{i+1}' if i < 2 else None
            elif position == 'WR':
                slot = f'WR{i+1}' if i < 3 else None
            elif position == 'TE':
                slot = f'TE{i+1}' if i < 2 else None
            elif position == 'K':
                slot = 'K' if i == 0 else None
            else:
                slot = None
                
            if slot and slot in TEAM10_SLOTS:
                player_to_slot[row.player_name] = slot
    
    return player_to_slot


def _apply_slot_labels_to_game(game_df: pd.DataFrame, player_to_slot: Dict[str, str]) -> pd.DataFrame:
    """
    Apply pre-computed season-level slot labels to a single game.
    """
    game_df = game_df.copy()
    game_df['slot_label'] = game_df['player_name'].map(player_to_slot)
    return game_df


def build_team_correlation(
    session: Session,
    team: str,
    min_games: int = 10,
) -> Tuple[np.ndarray, bool]:
    """
    Build a 10x10 within-team correlation matrix from historical data.
    
    Slot labels are assigned at the SEASON level (based on total production),
    then tracked game-by-game. This correctly captures variance - e.g., when
    the WR1 (by season role) has a dud game while WR3 booms.
    
    Args:
        session: DB session
        team: Team abbreviation (e.g., 'SEA', 'KC')
        min_games: Minimum games required to build correlation
        
    Returns:
        (correlation_matrix, is_valid): 10x10 matrix and whether it's team-specific
                                        If insufficient data, returns identity matrix
    """
    # Query historical stats for this team
    stats = session.query(NFLStat).filter(
        NFLStat.team == team,
        NFLStat.fantasy_points_ppr.isnot(None)
    ).all()
    
    if not stats:
        print(f"  No historical data for {team}, using identity correlation")
        return np.eye(10), False
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'season': s.season,
        'week': s.week,
        'player_name': s.player_display_name,
        'position': s.position,
        'fantasy_points_ppr': s.fantasy_points_ppr,
    } for s in stats])
    
    slot_idx = {s: i for i, s in enumerate(TEAM10_SLOTS)}
    all_game_slot_fpts = []
    
    # Process each season separately for slot assignment
    for season, season_df in df.groupby('season'):
        # Assign slots based on full-season production
        player_to_slot = _assign_season_slot_labels(season_df)
        
        if not player_to_slot:
            continue
        
        # Now iterate through games in this season
        for (_, week), game_df in season_df.groupby(['season', 'week']):
            game_df = _apply_slot_labels_to_game(game_df, player_to_slot)
            
            # Extract fpts for each slot
            slot_fpts = np.full(10, np.nan)
            for _, row in game_df.iterrows():
                slot = row['slot_label']
                if slot and slot in slot_idx:
                    slot_fpts[slot_idx[slot]] = row['fantasy_points_ppr']
            
            all_game_slot_fpts.append(slot_fpts)
    
    if len(all_game_slot_fpts) < min_games:
        print(f"  Only {len(all_game_slot_fpts)} valid games for {team} (need {min_games}), using identity correlation")
        return np.eye(10), False
    
    # Stack into matrix: (n_games, 10)
    X = np.array(all_game_slot_fpts)
    
    # Compute Spearman correlation, handling missing values
    # Use pairwise complete observations
    n_slots = 10
    corr = np.eye(n_slots)
    
    from scipy.stats import spearmanr
    
    for i in range(n_slots):
        for j in range(i + 1, n_slots):
            # Get pairwise complete observations
            mask = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            if mask.sum() < 5:  # Need at least 5 observations
                continue
            
            xi, xj = X[mask, i], X[mask, j]
            
            # Spearman correlation
            rho, _ = spearmanr(xi, xj)
            
            if not np.isnan(rho):
                corr[i, j] = rho
                corr[j, i] = rho
    
    print(f"  Built correlation for {team} from {len(all_game_slot_fpts)} games across {df['season'].nunique()} seasons")
    return corr, True
