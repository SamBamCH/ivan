#!/usr/bin/env python3
"""
scripts/nfl/build_models.py

Build marginals + copulas and, if --plot, run diagnostics.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kstest, anderson_ksamp
from sqlalchemy.orm import Session

from ivan.sports.nfl.player_level_calculations import build_models as build_player_models
from ivan.sports.nfl.dst_scoring import build_dst_game_table
from ivan.sports.nfl.team_level_calculations   import build_team_models, _compute_fpts
from ivan.sports.nfl.diagnostics              import (
    plot_marginal_fit,
    test_marginal_ks,
    plot_copula_corr,
    compare_empirical_vs_model_scatter,
    estimate_tail_dependence
)
from ivan.sports.nfl.models import NFLStat
from ivan.database.session import init_db, get_session

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-url',    default=None)
    parser.add_argument('--model-dir', default='data/nfl/models')
    parser.add_argument('--plot',      action='store_true')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # print("→ Building player‐level models")
    build_player_models(args.db_url, model_dir, args.plot) 

    _years = [2023,2024]  # add more as needed (e.g., 2019..2023)
    nrows = build_dst_game_table(args.db_url, years=_years)
    print(f"   wrote {nrows} rows to nfl_dst_game")

    # print("→ Building team‐level copulas")
    build_team_models(args.db_url, model_dir) 

    if args.plot: #WILL NOT PLOT IF --plot arg isn't passed
        print("→ Running diagnostics…")
        engine = init_db('nfl', args.db_url)
        session = get_session(engine)
        ### 2) Team copula diagnostics ###
        # for each team, compare empirical corr vs saved copula
        # pivot your fpts exactly as in build_team_models:
        df_all = pd.read_sql(session.query(NFLStat).statement, session.bind)
        df_all['fpts'] = _compute_fpts(df_all)
        pivot = df_all.pivot_table(
            index=['season','week','recent_team'],
            columns='position_group',
            values='fpts',
            aggfunc='sum',
            fill_value=0
        )[['QB','RB','WR','TE']]

        print(pivot.head())

        for team in pivot.index.get_level_values('recent_team').unique():
            team_data = pivot.xs(team, level='recent_team')
            vine = joblib.load(model_dir / 'team_vines' / f"{team}_vine.pkl")
            if np.random.rand() > 0.9: # only plot random 10% of teams
                # 2a) empirical corr heatmap
                plot_copula_corr(team, team_data)
                # 2b) scatter of (U_i, U_j) for data vs model
                compare_empirical_vs_model_scatter(team, team_data, vine)
            # 2c) tail‐dependence estimate
            τ_data  = estimate_tail_dependence(team_data)
            τ_model = estimate_tail_dependence(vine.sample(len(team_data)))
            print(f"{team:3s} τ_data={τ_data:.2f}, τ_model={τ_model:.2f}")

if __name__ == '__main__':
    main()
