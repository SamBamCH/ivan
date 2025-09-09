#!/usr/bin/env python3
"""
scripts/nfl/build_models.py

Build marginals + copulas and, if --plot, run diagnostics.
"""
import warnings, numpy as _np
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
_np.seterr(all="ignore")
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kstest, anderson_ksamp
from sqlalchemy.orm import Session

from ivan.sports.nfl.player_level_calculations import build_models as build_player_models
from ivan.sports.nfl.build_positional_team_vines import build_positional_team_vines
from ivan.sports.nfl.dst_scoring import build_dst_game_table
from ivan.sports.nfl.team_level_calculations   import build_team_models, _compute_fpts
from ivan.sports.nfl.build_team_positional_table import build_team_positional_table
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
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("→ Building player‐level models")
    build_player_models(args.db_url, model_dir) 

    _years = [2023,2024]  # add more as needed (e.g., 2019..2023)
    nrows = build_dst_game_table(args.db_url, years=_years)
    print(f"   wrote {nrows} rows to nfl_dst_game")

    # new: fixed-width team-week positional fpts table
    n_team_rows = build_team_positional_table(args.db_url)
    print(f"   wrote {n_team_rows} rows to nfl_team_week_positional_fpts")

    print("→ Building team‐level copulas")
    build_team_models(args.db_url, model_dir)
    build_positional_team_vines(model_dir, args.db_url) 

if __name__ == '__main__':
    main()
