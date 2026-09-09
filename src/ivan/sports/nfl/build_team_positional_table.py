from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.sports.nfl.team_level_calculations import _FPTS_WEIGHTS, _coerce_team_col, _ensure_position_group, _compute_fpts
from ivan.database.session import read_sql as _read_sql
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLStat, NFLTeamWeekPositionalFPTS, NFLDSTGame

# ---------------------------------------------------------------------
# Fantasy points weights for skill positions (not used for DST).
# Keep in sync with your team-level calculations.
# ---------------------------------------------------------------------

# ------------------------- helpers -----------------------------------

def _coerce_opp_col(df: pd.DataFrame) -> pd.DataFrame:
    for cand in ('opponent_team', 'opponent', 'opp_team', 'opp'):
        if cand in df.columns:
            return df.rename(columns={cand: 'opponent_team'})
    raise RuntimeError("Could not find opponent column (opponent_team/opponent/opp_team/opp).")

def _pick_qb(g: pd.DataFrame) -> Optional[float]:
    qb = g[g['position_group'] == 'QB'].copy()
    if qb.empty:
        return None
    for by in ('attempts', 'passing_yards', 'fpts'):
        if by not in qb.columns:
            qb[by] = 0
    qb = qb.sort_values(['attempts', 'passing_yards', 'fpts'], ascending=[False, False, False])
    return float(qb['fpts'].iloc[0])

def _pick_by_depth(g: pd.DataFrame, pos: str, key1: str, key2: str, tie_keys: List[str], n: int) -> List[Optional[float]]:
    block = g[g['position_group'] == pos].copy()
    if block.empty:
        # per your guidance: default to 0 when slot is “missing”
        return [0.0]*n
    for k in [key1, key2, *tie_keys, 'fpts']:
        if k not in block.columns:
            block[k] = 0
    sort_cols = [key1, key2] + tie_keys + ['fpts']
    block = block.sort_values(sort_cols, ascending=[False]*len(sort_cols))
    vals = [float(x) for x in block['fpts'].tolist()[:n]]
    return vals + [0.0] * (n - len(vals))

def _mode(series: pd.Series) -> Optional[str]:
    if series.empty:
        return None
    m = series.dropna().astype(str)
    if m.empty:
        return None
    return m.mode().iloc[0] if not m.mode().empty else None

# ----------------- DST ingestion from nfl_history.nfl_dst_game -----------------

def _compute_dst_fpts_if_needed(df: pd.DataFrame) -> pd.Series:
    # If some points column exists, use it
    for name in ('fpts', 'fantasy_points', 'dk_points', 'fd_points'):
        if name in df.columns:
            return pd.to_numeric(df[name], errors='coerce')

    # Otherwise compute a conventional DST total from common stat cols
    cols = {c.lower(): c for c in df.columns}
    def get(k: str):
        return pd.to_numeric(df[cols[k]], errors='coerce') if k in cols else None

    sacks = get('sacks')
    ints  = get('interceptions') or get('ints')
    fum   = get('fumbles_recovered') or get('fumbles') or get('fumbles_lost')  # rough fallback
    tds   = get('defensive_tds') or get('tds') or get('touchdowns')
    saf   = get('safeties') or get('safety')
    blk   = get('blocked_kicks') or get('blocks')
    pa    = get('points_allowed') or get('points')

    f = pd.Series(0.0, index=df.index)
    if sacks is not None: f = f + sacks * 1.0
    if ints  is not None: f = f + ints  * 2.0
    if fum   is not None: f = f + fum   * 2.0
    if tds   is not None: f = f + tds   * 6.0
    if saf   is not None: f = f + saf   * 2.0
    if blk   is not None: f = f + blk   * 2.0
    if pa is not None:
        bins = [-1, 0, 6, 13, 20, 27, 34, 1000]
        scores = [10, 7, 4, 1, 0, -1, -4]
        idx = np.digitize(pa, bins, right=True) - 1
        idx = np.clip(idx, 0, len(scores)-1)
        f = f + pd.Series([scores[i] for i in idx], index=df.index, dtype=float)
    return f

def _load_dst_history(session: Session) -> pd.DataFrame:
    """Load DST game results using the ORM mapper (no raw SQL strings)."""
    stmt = (
        session.query(
            NFLDSTGame.season.label('season'),
            NFLDSTGame.week.label('week'),
            NFLDSTGame.team.label('team'),
            NFLDSTGame.opponent.label('opponent'),
            NFLDSTGame.dst_points.label('DST'),
        ).statement
    )
    df = _read_sql(session, stmt)

    # If season_type isn't present on this table, default to REG
    if 'season_type' not in df.columns:
        df['season_type'] = 'REG'

    # Normalize & aggregate defensively
    keep = ['season', 'season_type', 'week', 'team', 'opponent', 'DST']
    df = df[keep].copy()
    df = (
        df.groupby(['season','season_type','week','team','opponent'], dropna=False, as_index=False)
          .agg({'DST': 'sum'})
    )
    return df

# ----------------- K ingestion from nflreadpy PBP -----------------
def _load_k_weekly_from_pbp(years: List[int]) -> pd.DataFrame:
    """
    Build weekly team-level K fantasy points from nflreadpy play-by-play.
    Returns columns: season, season_type, week, team, opponent, K
    """
    try:
        import nflreadpy as nfl
    except Exception as e:
        raise RuntimeError("nflreadpy is not installed. Run: pip install nflreadpy") from e

    cols = [
        'season', 'season_type', 'week', 'game_id',
        'posteam', 'defteam',
        'field_goal_result', 'kick_distance',
        'extra_point_result'
    ]
    # Pull PBP - nflreadpy returns Polars DataFrame, convert to pandas
    pbp = nfl.load_pbp(seasons=years).to_pandas()

    # Select only the columns we need
    df = pbp[cols].copy()

    # Distance buckets and points
    dist = pd.to_numeric(df['kick_distance'], errors='coerce').fillna(0.0)

    fg_made = df['field_goal_result'].astype(str).str.lower() == 'made'

    xp_good = df['extra_point_result'].astype(str).str.lower() == 'good'

    k_pts = np.zeros(len(df), dtype=np.float32)

    # Field goals made by distance
    k_pts[fg_made & (dist < 40)] += 3.0
    k_pts[fg_made & (40 <= dist) & (dist < 50)] += 4.0
    k_pts[fg_made & (dist >= 50)] += 5.0

    # Extra points
    k_pts[xp_good] += 1.0

    out = pd.DataFrame({
        'season': df['season'].astype(int),
        'season_type': df['season_type'].astype(str),
        'week': df['week'].astype(int),
        'team': df['posteam'].astype(str),
        'opponent': df['defteam'].astype(str),
        'K': k_pts
    })

    # Aggregate in case of multiple plays per game/week
    out = (out
           .groupby(['season','season_type','week','team','opponent'], as_index=False)
           .agg({'K': 'sum'}))

    return out

# ------------------------------ main ---------------------------------

def build_team_positional_table(db_url: str | None = None) -> int:
    """Build nfl_team_week_positional_fpts and merge in DST from nfl_history.nfl_dst_game."""
    engine = init_db('nfl', db_url)
    session = get_session(engine)

    # Load player-level stats and compute FPTS for skill positions
    df = _read_sql(session, session.query(NFLStat).statement)
    df = _coerce_team_col(df)
    df = _coerce_opp_col(df)

    df['position_group'] = _ensure_position_group(df)
    df['fpts'] = _compute_fpts(df)

    # group by team-week
    keys = ['season', 'season_type', 'week', 'recent_team']
    out_rows: List[Dict] = []
    for (season, season_type, week, team), g in df.groupby(keys, dropna=False):
        g = g.copy()

        opponent = _mode(g['opponent_team'])

        qb = _pick_qb(g)
        rb1, rb2 = _pick_by_depth(g, 'RB', key1='carries', key2='targets',
                                  tie_keys=['rushing_yards', 'receiving_yards'], n=2)
        wr1, wr2, wr3 = _pick_by_depth(g, 'WR', key1='targets', key2='receiving_yards',
                                       tie_keys=['receptions'], n=3)
        te1, te2 = _pick_by_depth(g, 'TE', key1='targets', key2='receiving_yards',
                                  tie_keys=['receptions'], n=2)

        row = dict(
            season=int(season), season_type=str(season_type), week=int(week),
            team=str(team), opponent=str(opponent) if opponent is not None else None,
            QB=qb if qb is not None else 0.0,
            RB1=rb1, RB2=rb2, WR1=wr1, WR2=wr2, WR3=wr3, TE1=te1, TE2=te2,
            K=0.0, DST=np.nan  # DST will be backfilled from history
        )
        out_rows.append(row)

    df_team = pd.DataFrame(out_rows)

    try:
        dst_hist = _load_dst_history(session)   # <-- pass the session here
        df_team = df_team.merge(
            dst_hist[['season','season_type','week','team','DST']],
            on=['season','season_type','week','team'],
            how='left',
            suffixes=('', '_dst')
        )
        if 'DST_dst' in df_team.columns:
            df_team['DST'] = df_team['DST_dst'].where(df_team['DST_dst'].notna(), df_team['DST'])
            df_team = df_team.drop(columns=['DST_dst'])
    except Exception as e:
        print(f"[WARN] Could not backfill DST via ORM: {e}")

    try:
        years = sorted(pd.to_numeric(df_team['season'], errors='coerce').dropna().unique().astype(int).tolist())
        if years:
            k_hist = _load_k_weekly_from_pbp(years)
            df_team = df_team.merge(
                k_hist[['season','season_type','week','team','K']],
                on=['season','season_type','week','team'],
                how='left',
                suffixes=('', '_k')
            )
            if 'K_k' in df_team.columns:
                # Prefer nflreadpy-derived K where available
                df_team['K'] = df_team['K_k'].where(df_team['K_k'].notna(), df_team['K'])
                df_team = df_team.drop(columns=['K_k'])
        else:
            print("[pos-table] No seasons found to pull K from nflreadpy.")
    except Exception as e:
        print(f"[WARN] Could not backfill K via nflreadpy: {e}")

    # Build opponent mirror and join
    idx_cols = ['season','season_type','week','team']
    mirror_cols = ['QB','RB1','RB2','WR1','WR2','WR3','TE1','TE2','K','DST']

    right = (
        df_team[idx_cols + mirror_cols]
        .rename(columns={'team': 'opponent', **{c: f'OPP{c}' for c in mirror_cols}})
    )

    df_join = df_team.merge(
        right,
        on=['season', 'season_type', 'week', 'opponent'],
        how='left'
    )

    # --- keep only rows with ALL slots present on both sides -------------
    base_slots = ['QB','RB1','RB2','WR1','WR2','WR3','TE1','TE2','K','DST']
    required_cols = [c for c in base_slots + [f'OPP{c}' for c in base_slots] if c in df_join.columns]

    # Sanity print
    print(f"[pos-table] total rows pre-filter: {len(df_join)}; requiring non-null in {len(required_cols)} cols")

    # Filter: drop any row missing any required slot (own or opponent)
    df_join_req = df_join.dropna(subset=required_cols, how='any')

    print(f"[pos-table] rows with all slots present: {len(df_join_req)} "
          f"(dropped {len(df_join) - len(df_join_req)})")

    if df_join_req.empty:
        print("[pos-table] no complete rows found; skipping persistence.")
        session.close()
        return 0

    # Resolve schema/table from the ORM model if present
    table_name = getattr(NFLTeamWeekPositionalFPTS, '__tablename__', 'nfl_team_week_positional_fpts')
    schema = None
    try:
        # If you have a declared Table, this will carry schema info
        schema = getattr(NFLTeamWeekPositionalFPTS.__table__, 'schema', None)
    except Exception:
        pass

    # Write with a real transaction; make sure we land in the right schema
    with engine.begin() as conn:
        df_join_req.to_sql(
            name=table_name,
            con=conn,
            schema=schema,          # <-- key: write to the same schema your ORM uses
            if_exists='replace',
            index=False,
            method='multi'
        )

    # Verify round-trip (counts)
    try:
        from sqlalchemy import text
        fq_name = f"{schema}.{table_name}" if schema else table_name
        with engine.connect() as conn:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {fq_name}")).scalar_one()
        print(f"[pos-table] wrote {n} rows to {fq_name}")
    except Exception as e:
        print(f"[pos-table] verify read failed: {e}")

    session.close()
    return len(df_join)
