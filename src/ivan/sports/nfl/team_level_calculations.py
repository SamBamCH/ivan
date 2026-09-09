from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from copulas.multivariate import VineCopula
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.database.session import read_sql as _read_sql
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

def _coerce_team_col(df: pd.DataFrame) -> pd.DataFrame:
    if 'recent_team' in df.columns:
        return df
    for cand in ('team', 'team_abbrev', 'defteam', 'def_team'):
        if cand in df.columns:
            return df.rename(columns={cand: 'recent_team'})
    raise RuntimeError("Could not find a team column (recent_team/team/team_abbrev/defteam).")


def _ensure_position_group(df: pd.DataFrame) -> pd.Series:
    if 'position_group' in df.columns:
        return df['position_group']
    pos = df.get('position', pd.Series(index=df.index, dtype='object')).astype(str).str.upper()
    return pos.replace({
        'QB':'QB', 'RB':'RB', 'HB':'RB', 'FB':'RB',
        'WR':'WR',
        'TE':'TE',
        'K':'K',
        'DST':'DST','DEF':'DST','D/ST':'DST'
    })


def _compute_fpts(df: pd.DataFrame) -> pd.Series:
    fpts = pd.Series(0.0, index=df.index)
    pos_group = _ensure_position_group(df)
    for pos, weights in _FPTS_WEIGHTS.items():
        mask = (pos_group == pos)
        if not mask.any():
            continue
        cols = [c for c in weights if c in df.columns]
        if not cols:
            continue
        block = df.loc[mask, cols]
        fpts.loc[mask] = block.mul(pd.Series(weights), axis=1).sum(axis=1)
    return fpts


def build_team_models(db_url: str, model_dir: Path) -> dict[str, VineCopula]:
    """
    Fit a per-team 5D vine copula over (QB, RB, WR, TE, DST).
    Saves each vine to {model_dir}/team_vines/{TEAM}_vine.pkl
    Returns {team: VineCopula}.
    """
    model_dir = Path(model_dir)
    vine_dir = model_dir / 'team_vines'
    vine_dir.mkdir(parents=True, exist_ok=True)

    engine = init_db('nfl', db_url)
    session = get_session(engine)

    # 1) Player-level stats -> compute fpts and pivot (season, week, team)
    df_stats = _read_sql(session, session.query(NFLStat).statement)
    df_stats = _coerce_team_col(df_stats)
    df_stats['fpts'] = _compute_fpts(df_stats)

    pivot = df_stats.pivot_table(
        index=['season', 'week', 'recent_team'],
        columns='position_group',
        values='fpts',
        aggfunc='sum',
        fill_value=0.0
    )

    # 2) Join OPPONENT DST points (season, week, team/opponent)
    dst = _read_sql(session, text("SELECT season, week, team, opponent, dst_points FROM nfl_dst_game"))
    if not dst.empty:
        # For each (season, week, recent_team) we want the DST points scored by the OPPONENT
        # against this team. So align on opponent == recent_team.
        opp_dst = (
            dst.rename(columns={'opponent': 'recent_team'})
            .set_index(['season', 'week', 'recent_team'])['dst_points']
        )
        pivot = pivot.join(opp_dst)

    # 3) Reorder/select columns and rename OppDST
    want = [c for c in ['QB', 'RB', 'WR', 'TE'] if c in pivot.columns]
    want += ['dst_points']               # the joined series is still named 'dst_points'
    pivot = pivot.reindex(columns=want).rename(columns={'dst_points': 'OppDST'}).fillna(0.0)

    vines: dict[str, VineCopula] = {}
    team_index = pivot.index.get_level_values('recent_team').unique()
    for team in team_index:
        mat = pivot.xs(team, level='recent_team')  # (season, week) rows

        # If DST wasn’t present at all, add a zero column so we still fit a stable dimension
        if 'DST' not in mat.columns:
            mat['DST'] = 0.0

        # Ensure exact order (QB,RB,WR,TE,DST) for consistent column-to-dimension mapping
        ordered_cols = [c for c in ['QB', 'RB', 'WR', 'TE', 'OppDST'] if c in mat.columns]
        mat = mat[ordered_cols]

        # Skip teams with too few observations to be meaningful
        if len(mat) < 5:
            # You can relax/raise this threshold if needed
            continue

        # 4) Empirical uniforms in R^5 (rank transform per column)
        U = mat.rank(axis=0, pct=True, method='average').clip(1e-6, 1 - 1e-6)

        vine = VineCopula('center')
        vine.fit(U)

        joblib.dump(vine, vine_dir / f"{team}_vine.pkl")
        vines[team] = vine

    return vines
