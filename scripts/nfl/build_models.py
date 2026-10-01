#!/usr/bin/env python3
"""
scripts/nfl/build_models.py

Build marginals + copulas for NFL player and team data.
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
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
from scipy.stats import kstest, anderson_ksamp
from sqlalchemy.orm import Session

from ivan.sports.nfl.paths import MODELS_DIR
from ivan.sports.nfl.player_level_calculations import build_models as build_player_models
from ivan.sports.nfl.build_team10_vine import build_team10_vine
from ivan.sports.nfl.calibrate_game_factors import calibrate_game_factors
from ivan.sports.nfl.dst_scoring import build_dst_game_table
from ivan.sports.nfl.refactor.dst import fit_dst_odds_model, describe_dst_odds_model
from ivan.sports.nfl.refactor.historical.data_fetch import seasons_to_fetch
from ivan.sports.nfl.team_level_calculations   import build_team_models, _compute_fpts
from ivan.sports.nfl.build_team_positional_table import build_team_positional_table
from ivan.sports.nfl.slate_models import build_slate_models
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
    parser.add_argument('--model-dir', default=MODELS_DIR)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("→ Building player‐level models")
    build_player_models(args.db_url, model_dir) 

    _years = seasons_to_fetch()   # every season in nfl_stats through the one in progress
    nrows = build_dst_game_table(args.db_url, years=_years)
    print(f"   wrote {nrows} rows to nfl_dst_game")

    # DST mean from the game lines, shape from history (writes dst_odds_params.json)
    engine = init_db("nfl", args.db_url)
    params = fit_dst_odds_model(get_session(engine), model_dir)
    print("\n".join("   " + line for line in describe_dst_odds_model(params).splitlines()))

    # new: fixed-width team-week positional fpts table
    n_team_rows = build_team_positional_table(args.db_url)
    print(f"   wrote {n_team_rows} rows to nfl_team_week_positional_fpts") 

    # === fit HGV-20 artifacts === 
    print("→ Building team‐level models")
    try: 
        import pyvinecopulib
    except ImportError:
        raise SystemExit("pyvinecopulib is required for NFL team models; please install it and try again")
    print("→ Fitting per-team 10D vine (QB..DST)")
    build_team10_vine(model_dir, args.db_url)     # writes team10_vine.pkl

    build_slate_models()


if __name__ == '__main__':
    main()
