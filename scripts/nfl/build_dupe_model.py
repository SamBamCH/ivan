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

from ivan.sports.nfl.paths import MODELS_DIR
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import (
    ShowdownPlayer,
    ShowdownLineup,
    ShowdownContestMetadata,
    HistoricalContest
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR
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


def build_contest_level_model(session, cutoff=None, model_dir=None):
    """
    Stage 1: Predict contest-level dupe distribution.
    
    Input: contest features
    Output: p10, p25, median, p75, p90, max dupe counts
    """
    log.info("=" * 60)
    log.info("STAGE 1: Building contest-level model")
    log.info("=" * 60)
    
    # Load contest metadata
    q = (session.query(ShowdownContestMetadata, HistoricalContest)
         .join(HistoricalContest, ShowdownContestMetadata.contest_id == HistoricalContest.contest_id))
    if cutoff is not None:                      # as-of build: only contests that finished before the cutoff
        q = q.filter(HistoricalContest.date < cutoff)
    contests = q.all()
    
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
    model_path = Path(model_dir or MODEL_DIR) / "contest_dupe_quantiles.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    
    log.info(f"✓ Saved contest-level models to {model_path}")
    log.info(f"  Trained on {len(contests)} contests")
    
    return models


def build_lineup_level_model(session, cutoff=None, model_dir=None):
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
    q = session.query(ShowdownLineup)
    if cutoff is not None:
        before = session.query(HistoricalContest.contest_id).filter(HistoricalContest.date < cutoff)
        q = q.filter(ShowdownLineup.contest_id.in_(before))
    lineups = q.all()
    
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
    model_path = Path(model_dir or MODEL_DIR) / "lineup_dupe_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    log.info(f"\n✓ Saved lineup-level model to {model_path}")
    log.info(f"  Trained on {len(X)} lineups")
    
    return model


def main(cutoff=None, model_dir=None, db_url=None):
    """cutoff: a datetime; only contests that finished before it train the models. model_dir: where the pickles go."""
    engine = init_db('nfl', db_url)
    session = get_session(engine)
    if model_dir is not None:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Build both models
    contest_models = build_contest_level_model(session, cutoff, model_dir)
    lineup_model = build_lineup_level_model(session, cutoff, model_dir)
    
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


def main_v2(argv):
    """v2: the product rule fitted on real fields (ivan.sports.nfl.dupe_model)."""
    import argparse
    from ivan.sports.nfl import dupe_model as DM
    ap = argparse.ArgumentParser(prog="build_dupe_model.py v2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest"); b.add_argument("--db-url", default=None)
    c = sub.add_parser("build"); c.add_argument("--cutoff", default=None); c.add_argument("--model-dir", default=None); c.add_argument("--db-url", default=None)
    a = ap.parse_args(argv)
    session = get_session(init_db("nfl", a.db_url))
    if a.cmd == "backtest":
        df = DM.training_frame(session)
        print(f"{df.contest_id.nunique()} fields, {len(df)} unique lineups")
        r = DM.backtest(df)
        for fmt, v in r.items():
            print(f"{fmt}: {v['contests']} fields; fit {v['fit']}")
            for f in v["folds"]:
                print(f"   held-out {f['held_out']} ({f['contests']}): k {f['k']} beta {f['beta']}  copies/entry pred {f['copies_per_entry_pred']:6.1f} actual {f['copies_per_entry_actual']:6.1f}  "
                      f"log MAE {f['log_mae']:.2f} bias {f['log_bias']:+.2f}  top ratio {f['top_lineup_ratio_median']:.2f}")
        return 0
    df = DM.training_frame(session, cutoff=a.cutoff)
    m = DM.fit(df, cutoff=a.cutoff)
    m.cv = {k: {"contests": v["contests"], "folds": v["folds"]} for k, v in DM.backtest(df).items()} if len(df) else {}
    path = m.save(Path(a.model_dir or MODELS_DIR))
    print(f"dupe model v2 -> {path}: {m.describe()}")
    return 0


if __name__ == "__main__":
    import argparse
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "v2":
        raise SystemExit(main_v2(_sys.argv[2:]))
    from datetime import datetime as _dt
    ap = argparse.ArgumentParser(description="Build the 2-stage showdown duplication model")
    ap.add_argument("--cutoff", default=None, help="train only on contests that finished before this date (YYYY-MM-DD)")
    ap.add_argument("--model-dir", default=None, help="output directory (default the live model directory)")
    ap.add_argument("--db-url", default=None)
    a = ap.parse_args()
    main(cutoff=_dt.fromisoformat(a.cutoff) if a.cutoff else None, model_dir=a.model_dir, db_url=a.db_url)
