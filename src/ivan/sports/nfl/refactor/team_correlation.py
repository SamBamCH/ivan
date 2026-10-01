"""
Team-specific correlation matrices from historical game logs.

Every team gets its own within-team Spearman matrix over the depth-chart
slots (QB, RB1-2, WR1-3, TE1-2, K, DST) plus the team's own points scored
(PTS), so a scrambling-QB offense and a pocket-passing one carry different
QB/RB and QB/WR dependence, and every offensive slot carries its own loading
on the team's score. Slot labels are assigned per team-season from
full-season production, then tracked game by game.

The DST slot is the *takeaway* part of the team's DraftKings DST line (sacks,
turnovers, blocks, defensive and return touchdowns, safeties). The
points-allowed bracket is not a slot: the sampler computes it from the
opponent's PTS draw, so the two sides of a game share one score.

Cross-team (game) correlation between A's slots and B's slots is estimated
from every game each team played, not only their head-to-head meetings: A's
slots against *its opponents'* slots, and B's opponents' slots against B's,
pooled and shrunk toward the league default by sample size. Head-to-head games
are in both pools.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.database.session import read_sql
from ivan.sports.nfl.models import NFLStat
from ivan.sports.nfl.refactor.dst import pa_bonus_vec

log = logging.getLogger(__name__)

TEAM10_SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "TE2", "K", "DST"]   # player slots
PTS_SLOT = "PTS"                                                                    # the team's points scored
TEAM_SLOTS = TEAM10_SLOTS + [PTS_SLOT]
SLOT_IDX = {s: i for i, s in enumerate(TEAM_SLOTS)}
N_SLOTS = len(TEAM_SLOTS)
SLOT_CAPS = {"QB": 1, "RB": 2, "WR": 3, "TE": 2, "K": 1}

MIN_GAMES_WITHIN = 10        # team-games before a within-team matrix is used
MIN_GAMES_CROSS = 10         # team-games before a cross matrix is used
MIN_PAIR_OBS = 5             # observations for one pair before it is estimated
CROSS_SHRINK_GAMES = 20      # pseudo-observations pulling a cross pair toward the default

# One slot table per session: {id(session.bind): DataFrame}
_SLOT_TABLE_CACHE: Dict[int, pd.DataFrame] = {}


def clear_correlation_cache() -> None:
    _SLOT_TABLE_CACHE.clear()


def _assign_season_slot_labels(season_df: pd.DataFrame) -> Dict[str, str]:
    """player_name -> slot for one team-season, ranked by season fantasy points (3+ games)."""
    totals = season_df.groupby(['player_name', 'position']).agg(
        total_fpts=('fantasy_points_ppr', 'sum'), games=('week', 'count')).reset_index()
    totals = totals[totals.games >= 3]
    out: Dict[str, str] = {}
    for position, cap in SLOT_CAPS.items():
        pos = totals[totals.position == position].sort_values('total_fpts', ascending=False)
        for i, row in enumerate(pos.itertuples()):
            if i >= cap:
                break
            out[row.player_name] = position if cap == 1 else f"{position}{i + 1}"
    return out


def league_slot_table(session: Session) -> pd.DataFrame:
    """
    Long table of every team-game slot line in history:
    columns season, week, team, opponent, slot, fpts. Built once per session.
    DST rows carry takeaway points; PTS rows carry the team's score.
    """
    key = id(session.get_bind())
    if key in _SLOT_TABLE_CACHE:
        return _SLOT_TABLE_CACHE[key]

    rows = session.query(NFLStat.season, NFLStat.week, NFLStat.team, NFLStat.opponent_team,
                         NFLStat.player_display_name, NFLStat.position, NFLStat.fantasy_points_ppr
                         ).filter(NFLStat.fantasy_points_ppr.isnot(None), NFLStat.team.isnot(None)).all()
    df = pd.DataFrame(rows, columns=['season', 'week', 'team', 'opponent', 'player_name', 'position', 'fantasy_points_ppr'])
    parts = []
    for (team, season), tdf in df.groupby(['team', 'season']):
        labels = _assign_season_slot_labels(tdf)
        if not labels:
            continue
        g = tdf[tdf.player_name.isin(labels)].copy()
        g['slot'] = g.player_name.map(labels)
        parts.append(g[['season', 'week', 'team', 'opponent', 'slot', 'fantasy_points_ppr']].rename(columns={'fantasy_points_ppr': 'fpts'}))

    dst = read_sql(session, text("SELECT season, week, team, opponent, dst_points, points_allowed FROM nfl_dst_game "
                                 "WHERE dst_points IS NOT NULL AND points_allowed IS NOT NULL"))
    if not dst.empty:
        dst['fpts'] = dst.dst_points.astype(float) - pa_bonus_vec(dst.points_allowed)
        dst['slot'] = 'DST'
        parts.append(dst[['season', 'week', 'team', 'opponent', 'slot', 'fpts']])

    sched = read_sql(session, text("SELECT season, week, home_team, away_team, home_score, away_score FROM nfl_schedule "
                                   "WHERE home_score IS NOT NULL AND away_score IS NOT NULL"))
    if not sched.empty:
        home = pd.DataFrame({'season': sched.season, 'week': sched.week, 'team': sched.home_team,
                             'opponent': sched.away_team, 'slot': PTS_SLOT, 'fpts': sched.home_score.astype(float)})
        away = pd.DataFrame({'season': sched.season, 'week': sched.week, 'team': sched.away_team,
                             'opponent': sched.home_team, 'slot': PTS_SLOT, 'fpts': sched.away_score.astype(float)})
        parts.extend([home, away])

    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['season', 'week', 'team', 'opponent', 'slot', 'fpts'])
    table = table.drop_duplicates(['season', 'week', 'team', 'slot'])
    _SLOT_TABLE_CACHE[key] = table
    return table


def _wide(table: pd.DataFrame) -> pd.DataFrame:
    """(season, week, team) x TEAM_SLOTS, with the opponent as a column."""
    if table.empty:
        return pd.DataFrame(columns=['season', 'week', 'team', 'opponent'] + TEAM_SLOTS)
    w = table.pivot_table(index=['season', 'week', 'team'], columns='slot', values='fpts', aggfunc='first')
    w = w.reindex(columns=TEAM_SLOTS)
    opp = table.groupby(['season', 'week', 'team'])['opponent'].first()
    w['opponent'] = opp
    return w.reset_index()


def _pairwise_spearman(X: np.ndarray, Y: np.ndarray, min_obs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Spearman between every column of X and every column of Y on pairwise-complete rows; also the counts."""
    R = np.full((X.shape[1], Y.shape[1]), np.nan)
    N = np.zeros_like(R, dtype=int)
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            mask = ~np.isnan(X[:, i]) & ~np.isnan(Y[:, j])
            n = int(mask.sum())
            N[i, j] = n
            if n < min_obs:
                continue
            rho, _ = spearmanr(X[mask, i], Y[mask, j])
            if np.isfinite(rho):
                R[i, j] = rho
    return R, N


def build_team_correlation(session: Session, team: str, min_games: int = MIN_GAMES_WITHIN) -> Tuple[np.ndarray, bool]:
    """
    Within-team Spearman matrix over TEAM_SLOTS for `team` from its own game
    logs, or the identity (and False) when it has fewer than `min_games` games.
    """
    w = _wide(league_slot_table(session))
    X = w[w.team == team][TEAM_SLOTS].to_numpy(float)
    if len(X) < min_games:
        log.debug(f"  {team}: {len(X)} games (need {min_games}), identity correlation")
        return np.eye(N_SLOTS), False
    R, _ = _pairwise_spearman(X, X, MIN_PAIR_OBS)
    R = np.where(np.isnan(R), 0.0, R)
    np.fill_diagonal(R, 1.0)
    return R, True


def build_cross_team_correlation(session: Session, team_a: str, team_b: str,
                                 min_games: int = MIN_GAMES_CROSS) -> Tuple[np.ndarray, bool]:
    """
    Cross matrix [A slot i, B slot j] for a game between team_a and team_b,
    pooled over every game each team played and shrunk toward the league
    default. Returns (matrix, True) or (default, False) when neither team has
    enough history.
    """
    w = _wide(league_slot_table(session))
    default = _default_cross_correlation()
    opp = w.rename(columns={s: f"opp_{s}" for s in TEAM_SLOTS})[['season', 'week', 'team'] + [f"opp_{s}" for s in TEAM_SLOTS]]

    def team_vs_opponents(team: str) -> Tuple[np.ndarray, np.ndarray]:
        g = w[w.team == team].merge(opp, left_on=['season', 'week', 'opponent'], right_on=['season', 'week', 'team'],
                                    how='inner', suffixes=('', '_o'))
        return g[TEAM_SLOTS].to_numpy(float), g[[f"opp_{s}" for s in TEAM_SLOTS]].to_numpy(float)

    Xa, Oa = team_vs_opponents(team_a)      # A's slots, A's opponents' slots
    Xb, Ob = team_vs_opponents(team_b)      # B's slots, B's opponents' slots
    if len(Xa) < min_games and len(Xb) < min_games:
        return default, False

    num = np.zeros((N_SLOTS, N_SLOTS)); den = np.zeros((N_SLOTS, N_SLOTS))
    if len(Xa) >= min_games:
        R, N = _pairwise_spearman(Xa, Oa, MIN_PAIR_OBS)           # [A slot i, opponent slot j]
        ok = ~np.isnan(R); num[ok] += (R * N)[ok]; den[ok] += N[ok]
    if len(Xb) >= min_games:
        R, N = _pairwise_spearman(Ob, Xb, MIN_PAIR_OBS)           # [opponent slot i, B slot j]
        ok = ~np.isnan(R); num[ok] += (R * N)[ok]; den[ok] += N[ok]
    cross = (num + CROSS_SHRINK_GAMES * default) / (den + CROSS_SHRINK_GAMES)
    log.debug(f"  cross {team_a} x {team_b}: {len(Xa)} + {len(Xb)} games")
    return cross, True


def _default_cross_correlation() -> np.ndarray:
    """
    League default when a team has no history: offense vs opposing offense
    slightly positive (game flow), offense vs opposing takeaways negative,
    takeaways vs opposing points negative, the two scores nearly independent.
    """
    corr = np.zeros((N_SLOTS, N_SLOTS))
    offense = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE1', 'TE2']
    for s1 in offense:
        for s2 in offense:
            corr[SLOT_IDX[s1], SLOT_IDX[s2]] = 0.15
    for s in offense:
        corr[SLOT_IDX[s], SLOT_IDX['DST']] = -0.20
        corr[SLOT_IDX['DST'], SLOT_IDX[s]] = -0.20
        corr[SLOT_IDX[s], SLOT_IDX[PTS_SLOT]] = 0.05
        corr[SLOT_IDX[PTS_SLOT], SLOT_IDX[s]] = 0.05
    corr[SLOT_IDX['DST'], SLOT_IDX[PTS_SLOT]] = -0.30
    corr[SLOT_IDX[PTS_SLOT], SLOT_IDX['DST']] = -0.30
    corr[SLOT_IDX['K'], SLOT_IDX[PTS_SLOT]] = -0.15
    corr[SLOT_IDX[PTS_SLOT], SLOT_IDX['K']] = -0.15
    corr[SLOT_IDX[PTS_SLOT], SLOT_IDX[PTS_SLOT]] = 0.05
    return corr


def get_team_correlations(session: Session, team_a: str, team_b: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(within A, within B, A x B cross) for a matchup."""
    corr_a, _ = build_team_correlation(session, team_a)
    corr_b, _ = build_team_correlation(session, team_b)
    corr_cross, _ = build_cross_team_correlation(session, team_a, team_b)
    return corr_a, corr_b, corr_cross
