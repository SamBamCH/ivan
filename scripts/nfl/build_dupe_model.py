"""
Build 2-stage duplication model for NFL Showdown:

Stage 1: Contest-level model
    Input: contest features (total_entries, entry_fee, etc.)
    Output: dupe distribution quantiles (p10, p25, median, p75, p90, max)

Stage 2: Lineup-level model
    Input: lineup features (ownership_sum, salary_remaining, correlation metrics)
    Output: predicted dupe count (raw, gets scaled by stage 1)

Usage:
    python scripts/nfl/build_dupe_model.py
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import (
    ShowdownPlayer,
    ShowdownLineup,
    ShowdownContestMetadata,
    HistoricalContest
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path("models/nfl")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def calculate_lineup_features(session, lineup):
    """
    Calculate runtime features for a lineup:
    - ownership_sum (CPT + FLEX)
    - salary_remaining
    - team correlation metrics
    """
    # Get player data
    cpt_player = (
        session.query(ShowdownPlayer)
        .filter_by(contest_id=lineup.contest_id, player_id=lineup.cpt_player_id)
        .first()
    )
    
    flex_players = (
        session.query(ShowdownPlayer)
        .filter(
            ShowdownPlayer.contest_id == lineup.contest_id,
            ShowdownPlayer.player_id.in_(lineup.flex_player_ids)
        )
        .all()
    )
    
    if not cpt_player or len(flex_players) != 5:
        return None
    
    # Ownership sum
    ownership_sum = cpt_player.ownership_actual + sum(p.ownership_actual for p in flex_players)
    
    # Salary remaining (already stored)
    salary_remaining = lineup.salary_remaining
    salary_remaining_pct = salary_remaining / 50000
    
    # Team correlation
    cpt_team = cpt_player.team
    flex_teams = [p.team for p in flex_players]
    same_team_count = sum(1 for t in flex_teams if t == cpt_team)
    
    # Position correlation
    cpt_pos = cpt_player.position
    flex_positions = [p.position for p in flex_players]
    
    # QB-WR stack detection
    qb_wr_stack = 0
    if cpt_pos == 'QB':
        qb_wr_stack = sum(1 for i, pos in enumerate(flex_positions) 
                         if pos in ['WR', 'TE'] and flex_teams[i] == cpt_team)
    
    # RB-DST negative correlation
    rb_dst_negative = 0
    for i, pos in enumerate(flex_positions):
        if pos == 'RB':
            for j, other_pos in enumerate(flex_positions):
                if other_pos == 'DST' and flex_teams[i] == flex_teams[j]:
                    rb_dst_negative = 1
                    break
    
    return {
        'ownership_sum': ownership_sum,
        'salary_remaining': salary_remaining,
        'salary_remaining_pct': salary_remaining_pct,
        'same_team_count': same_team_count,
        'qb_wr_stack': qb_wr_stack,
        'rb_dst_negative': rb_dst_negative,
        'cpt_ownership': cpt_player.ownership_actual,
    }


def build_contest_level_model(session):
    """
    Stage 1: Predict contest-level dupe distribution.
    
    Input: contest features
    Output: p10, p25, median, p75, p90, max dupe counts
    """
    log.info("=" * 60)
    log.info("STAGE 1: Building contest-level model")
    log.info("=" * 60)
    
    # Load contest metadata
    contests = (
        session.query(ShowdownContestMetadata, HistoricalContest)
        .join(HistoricalContest, ShowdownContestMetadata.contest_id == HistoricalContest.contest_id)
        .all()
    )
    
    if not contests:
        log.error("No contest data found!")
        return None
    
    # Build feature matrix
    X = []
    y_median = []
    y_p75 = []
    y_p90 = []
    y_max = []
    
    for meta, hist in contests:
        X.append([
            hist.total_entries,
            hist.entry_fee,
            meta.unique_lineup_count,
        ])
        y_median.append(meta.median_dupe_count)
        y_p75.append(meta.p75_dupe_count or meta.median_dupe_count)
        y_p90.append(meta.p90_dupe_count or meta.median_dupe_count)
        y_max.append(meta.max_dupe_count)
    
    X = np.array(X)
    
    # Train separate models for each quantile
    models = {}
    
    for name, y in [('median', y_median), ('p75', y_p75), ('p90', y_p90), ('max', y_max)]:
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        models[name] = model
        
        log.info(f"  {name:6s}: MAE={mae:.2f}, R²={r2:.3f}")
    
    # Save models
    model_path = MODEL_DIR / "contest_dupe_quantiles.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    
    log.info(f"✓ Saved contest-level models to {model_path}")
    log.info(f"  Trained on {len(contests)} contests")
    
    return models


def build_lineup_level_model(session):
    """
    Stage 2: Predict lineup-level dupe counts.
    
    Input: lineup features (ownership, salary, correlations)
    Output: raw dupe count (gets scaled by stage 1 predictions)
    """
    log.info("")
    log.info("=" * 60)
    log.info("STAGE 2: Building lineup-level model")
    log.info("=" * 60)
    
    # Load all lineups
    lineups = session.query(ShowdownLineup).all()
    
    if not lineups:
        log.error("No lineup data found!")
        return None
    
    log.info(f"Processing {len(lineups)} lineups...")
    
    # Calculate features for each lineup
    X = []
    y = []
    
    for i, lineup in enumerate(lineups):
        if i % 1000 == 0:
            log.info(f"  Processed {i}/{len(lineups)} lineups...")
        
        features = calculate_lineup_features(session, lineup)
        if features is None:
            continue
        
        X.append([
            features['ownership_sum'],
            features['salary_remaining_pct'],
            features['same_team_count'],
            features['qb_wr_stack'],
            features['rb_dst_negative'],
            features['cpt_ownership'],
        ])
        y.append(lineup.dupe_count)
    
    X = np.array(X)
    y = np.array(y)
    
    log.info(f"Built feature matrix: {X.shape}")
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    log.info(f"  MAE: {mae:.2f}")
    log.info(f"  R²:  {r2:.3f}")
    
    # Feature importance
    feature_names = [
        'ownership_sum', 'salary_remaining_pct', 'same_team_count',
        'qb_wr_stack', 'rb_dst_negative', 'cpt_ownership'
    ]
    importances = model.feature_importances_
    
    log.info("\n  Feature Importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        log.info(f"    {name:25s}: {imp:.3f}")
    
    # Save model
    model_path = MODEL_DIR / "lineup_dupe_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    log.info(f"\n✓ Saved lineup-level model to {model_path}")
    log.info(f"  Trained on {len(X)} lineups")
    
    return model


def main():
    engine = init_db('nfl')
    session = get_session(engine)
    
    # Build both models
    contest_models = build_contest_level_model(session)
    lineup_model = build_lineup_level_model(session)
    
    log.info("")
    log.info("=" * 60)
    log.info("COMPLETE")
    log.info("=" * 60)
    log.info("")
    log.info("Models saved to models/nfl/:")
    log.info("  - contest_dupe_quantiles.pkl (stage 1)")
    log.info("  - lineup_dupe_model.pkl (stage 2)")
    log.info("")
    log.info("Usage:")
    log.info("  1. Predict contest quantiles from contest features")
    log.info("  2. Predict raw dupe counts for lineups")
    log.info("  3. Scale lineup predictions to match contest distribution")
    
    session.close()


if __name__ == "__main__":
    main()
