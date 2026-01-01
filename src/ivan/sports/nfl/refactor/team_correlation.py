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


def build_cross_team_correlation(
    session: Session,
    team_a: str,
    team_b: str,
    min_matchups: int = 3,
) -> Tuple[np.ndarray, bool]:
    """
    Build a 10x10 cross-team correlation matrix from historical matchups.
    
    This captures how team A's performance correlates with team B's when they play.
    For example: when team A's QB does well, team B's DST usually does poorly.
    
    NOTE: Direct head-to-head matchups are rare (usually 1-2 per season), so we
    fall back to default assumptions more often. The within-team correlations
    are more important and have more data.
    
    Args:
        session: DB session
        team_a: First team abbreviation
        team_b: Second team abbreviation  
        min_matchups: Minimum matchups required (default low since h2h is rare)
        
    Returns:
        (correlation_matrix, is_valid): 10x10 matrix (team_a slots x team_b slots)
    """
    # For cross-team, we need FULL season data for slot assignment
    # Then filter to just the head-to-head matchups
    
    # Get all stats for both teams to build season-level slot assignments
    all_stats_a = session.query(NFLStat).filter(
        NFLStat.team == team_a,
        NFLStat.fantasy_points_ppr.isnot(None)
    ).all()
    
    all_stats_b = session.query(NFLStat).filter(
        NFLStat.team == team_b,
        NFLStat.fantasy_points_ppr.isnot(None)
    ).all()
    
    if not all_stats_a or not all_stats_b:
        return _default_cross_correlation(), False
    
    # Convert to DataFrames
    df_all_a = pd.DataFrame([{
        'season': s.season,
        'week': s.week,
        'player_name': s.player_display_name,
        'position': s.position,
        'opponent_team': s.opponent_team,
        'fantasy_points_ppr': s.fantasy_points_ppr,
    } for s in all_stats_a])
    
    df_all_b = pd.DataFrame([{
        'season': s.season,
        'week': s.week,
        'player_name': s.player_display_name,
        'position': s.position,
        'opponent_team': s.opponent_team,
        'fantasy_points_ppr': s.fantasy_points_ppr,
    } for s in all_stats_b])
    
    slot_idx = {s: i for i, s in enumerate(TEAM10_SLOTS)}
    X_a, X_b = [], []
    
    # Process each season
    for season in set(df_all_a['season']) & set(df_all_b['season']):
        season_a = df_all_a[df_all_a['season'] == season]
        season_b = df_all_b[df_all_b['season'] == season]
        
        # Build season-level slot assignments
        slots_a = _assign_season_slot_labels(season_a)
        slots_b = _assign_season_slot_labels(season_b)
        
        if not slots_a or not slots_b:
            continue
        
        # Find head-to-head matchups this season
        h2h_a = season_a[season_a['opponent_team'] == team_b]
        h2h_b = season_b[season_b['opponent_team'] == team_a]
        
        # Find common (season, week) for h2h games
        games_a = set(zip(h2h_a['season'], h2h_a['week']))
        games_b = set(zip(h2h_b['season'], h2h_b['week']))
        common_games = games_a & games_b
        
        for _, week in common_games:
            game_a = h2h_a[h2h_a['week'] == week]
            game_b = h2h_b[h2h_b['week'] == week]
            
            game_a = _apply_slot_labels_to_game(game_a, slots_a)
            game_b = _apply_slot_labels_to_game(game_b, slots_b)
            
            slot_fpts_a = np.full(10, np.nan)
            slot_fpts_b = np.full(10, np.nan)
            
            for _, row in game_a.iterrows():
                slot = row['slot_label']
                if slot and slot in slot_idx:
                    slot_fpts_a[slot_idx[slot]] = row['fantasy_points_ppr']
                    
            for _, row in game_b.iterrows():
                slot = row['slot_label']
                if slot and slot in slot_idx:
                    slot_fpts_b[slot_idx[slot]] = row['fantasy_points_ppr']
            
            X_a.append(slot_fpts_a)
            X_b.append(slot_fpts_b)
    
    if len(X_a) < min_matchups:
        print(f"  Only {len(X_a)} h2h matchups for {team_a} vs {team_b}, using defaults")
        return _default_cross_correlation(), False
    
    X_a = np.array(X_a)
    X_b = np.array(X_b)
    
    # Compute cross-correlation
    n_slots = 10
    cross_corr = _default_cross_correlation()  # Start with defaults, update where we have data
    
    from scipy.stats import spearmanr
    
    for i in range(n_slots):
        for j in range(n_slots):
            mask = ~np.isnan(X_a[:, i]) & ~np.isnan(X_b[:, j])
            if mask.sum() < 3:
                continue
            
            xi, xj = X_a[mask, i], X_b[mask, j]
            rho, _ = spearmanr(xi, xj)
            
            if not np.isnan(rho):
                cross_corr[i, j] = rho
    
    print(f"  Built cross-correlation for {team_a} vs {team_b} from {len(X_a)} matchups")
    return cross_corr, True


def _default_cross_correlation() -> np.ndarray:
    """
    Default cross-team correlation assumptions when insufficient data.
    
    Generally:
    - Offensive players on opposite teams have slight positive correlation (game flow)
    - Offensive players negatively correlate with opposing DST
    """
    n = 10
    corr = np.zeros((n, n))
    
    # Slot indices
    slot_idx = {s: i for i, s in enumerate(TEAM10_SLOTS)}
    
    # Small positive correlation between all offensive players (game flow/shootouts)
    offense_slots = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE1', 'TE2']
    for s1 in offense_slots:
        for s2 in offense_slots:
            corr[slot_idx[s1], slot_idx[s2]] = 0.15
    
    # Negative correlation with opposing DST
    dst_idx = slot_idx['DST']
    for s in offense_slots:
        corr[slot_idx[s], dst_idx] = -0.20
        corr[dst_idx, slot_idx[s]] = -0.20
    
    return corr


def get_team_correlations(
    session: Session,
    team_a: str,
    team_b: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all correlation matrices needed for a matchup.
    
    Returns:
        (corr_a, corr_b, corr_cross): Within-team corr for A, B, and cross-team
    """
    corr_a, _ = build_team_correlation(session, team_a)
    corr_b, _ = build_team_correlation(session, team_b)
    corr_cross, _ = build_cross_team_correlation(session, team_a, team_b)
    
    return corr_a, corr_b, corr_cross

