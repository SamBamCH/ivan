from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from scipy.stats import norm
from copulas.multivariate import VineCopula
from sqlalchemy.orm import Session

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLStat

# fantasy‐points weights (unchanged)
_FPTS_WEIGHTS = {
    'QB': {'passing_yards': .04, 'passing_tds': 4, 'interceptions': -2,
           'rushing_yards': .1, 'rushing_tds': 6},
    'RB': {'rushing_yards': .1, 'rushing_tds': 6, 'receptions': 1,
           'receiving_yards': .1, 'receiving_tds': 6},
    'WR': {'receptions': 1, 'receiving_yards': .1, 'receiving_tds': 6},
    'TE': {'receptions': 1, 'receiving_yards': .1, 'receiving_tds': 6},
}

def _compute_fpts(df: pd.DataFrame) -> pd.Series:
    fpts = pd.Series(0.0, index=df.index)
    for pos, weights in _FPTS_WEIGHTS.items():
        mask = df['position_group'] == pos
        if not mask.any():
            continue
        cols = [s for s in weights if s in df.columns]
        block = df.loc[mask, cols]
        fpts.loc[mask] = block.mul(pd.Series(weights)).sum(axis=1)
    return fpts

def build_team_models(db_url: str, model_dir: Path) -> dict[str, VineCopula]:
    """
    For each team:
      1) Pivot history to (year,week,team) × position_group → fpts  
      2) U = rank_pct(axis=0).clip(1e-6,1-1e-6)  
      3) vine = VineCopula('center'); vine.fit(U)  
      4) save vine; return vine objects  
    """
    model_dir = Path(model_dir)
    vine_dir = model_dir / 'team_vines'
    vine_dir.mkdir(parents=True, exist_ok=True)

    engine = init_db('nfl', db_url)
    session = get_session(engine)
    df = pd.read_sql(session.query(NFLStat).statement, session.bind)
    df['fpts'] = _compute_fpts(df)

    # pivot by (year, week, team)
    pivot = df.pivot_table(
        index=['season','week','recent_team'],
        columns='position_group',
        values='fpts',
        aggfunc='sum',
        fill_value=0
    )

    # keep only the four positions
    pivot = pivot[['QB', 'RB', 'WR', 'TE']]

    print(pivot.head())

    vines = {}
    for team in pivot.index.get_level_values('recent_team').unique():
        mat = pivot.xs(team, level='recent_team')  # rows = (year,week), cols = QB/RB/WR/TE

        # 1) empirical uniforms
        U = mat.rank(axis=0, pct=True).clip(1e-6, 1-1e-6)

        # 2) fit vine copula
        vine = VineCopula('center')
        vine.fit(U)

        # persist
        joblib.dump(vine, vine_dir / f"{team}_vine.pkl")
        vines[team] = vine

    return vines