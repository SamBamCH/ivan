"""
Team-specific correlation matrices from historical game logs, shrunk toward
the league.

Every team gets its own within-team Spearman matrix over the depth-chart
slots (QB, RB1-2, WR1-3, TE1-2, K, DST) plus the team's own points scored
(PTS), so a scrambling-QB offense and a pocket-passing one carry different
QB/RB and QB/WR dependence, and every offensive slot carries its own loading
on the team's score. Slot labels are assigned per team-season from
full-season production, then tracked game by game; the QB slot is the game's
actual starter.

With ~70 games per team the raw team matrices are noise-limited, so each one
is shrunk toward the league matrix (the same Spearman over every team-game
in history) by LEAGUE_SHRINK_GAMES pseudo-observations: the league structure
carries most of the weight and a team's own history tilts it. The 2025-26
backtest (scripts/nfl/backtest_correlation.py) scores this well above the
raw team matrices and above any recency weighting, which only cut the
effective sample size further.

The DST slot is the *takeaway* part of the team's DraftKings DST line (sacks,
turnovers, blocks, defensive and return touchdowns, safeties). The
points-allowed bracket is not a slot: the sampler computes it from the
opponent's PTS draw, so the two sides of a game share one score.

Cross-team (game) correlation between A's slots and B's slots is estimated
from every game each team played, not only their head-to-head meetings: A's
slots against *its opponents'* slots, and B's opponents' slots against B's,
pooled and shrunk the same way toward the league cross matrix (every team
against its opponent).
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

MIN_GAMES_WITHIN = 10        # team-games before a team's own history enters its matrix
MIN_GAMES_CROSS = 10         # team-games before a team's own history enters a cross matrix
MIN_PAIR_OBS = 5             # observations for one pair before it is estimated
MIN_LEAGUE_GAMES = 100       # team-games before the league matrices are used (else the hand-set default)
LEAGUE_SHRINK_GAMES = 300.0  # pseudo-observations pulling a team's matrices toward the league's (0 = raw team matrices)

# Caches per session bind: {id(session.bind): ...}
_SLOT_TABLE_CACHE: Dict[int, pd.DataFrame] = {}
_WIDE_CACHE: Dict[int, pd.DataFrame] = {}
_LEAGUE_CACHE: Dict[Tuple[int, Tuple[int, int] | None], Tuple[np.ndarray, np.ndarray, int]] = {}


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
    columns season, week, team, opponent, slot, fpts, player. Built once per
    session. DST rows carry takeaway points; PTS rows carry the team's score;
    only player slots carry a player name.
    """
    key = id(session.get_bind())
    if key in _SLOT_TABLE_CACHE:
        return _SLOT_TABLE_CACHE[key]

    rows = session.query(NFLStat.season, NFLStat.week, NFLStat.team, NFLStat.opponent_team,
                         NFLStat.player_display_name, NFLStat.position, NFLStat.fantasy_points_ppr
                         ).filter(NFLStat.fantasy_points_ppr.isnot(None), NFLStat.team.isnot(None)).all()
    df = pd.DataFrame(rows, columns=['season', 'week', 'team', 'opponent', 'player_name', 'position', 'fantasy_points_ppr'])
    cols = ['season', 'week', 'team', 'opponent', 'slot', 'fpts', 'player']
    parts = []
    # The QB slot is the game's actual starter (the QB with the most points that
    # week), not a season label, so every team-game has a QB line and a
    # mid-season change is attributed to the right passer. The other slots keep
    # their season labels.
    qbs = (df[df.position == 'QB'].sort_values('fantasy_points_ppr', ascending=False)
             .drop_duplicates(['season', 'week', 'team']).assign(slot='QB')
             .rename(columns={'fantasy_points_ppr': 'fpts', 'player_name': 'player'}))
    parts.append(qbs[cols])
    for (team, season), tdf in df.groupby(['team', 'season']):
        labels = {name: slot for name, slot in _assign_season_slot_labels(tdf).items() if slot != 'QB'}
        if not labels:
            continue
        g = tdf[tdf.player_name.isin(labels)].copy()
        g['slot'] = g.player_name.map(labels)
        g = g.rename(columns={'fantasy_points_ppr': 'fpts', 'player_name': 'player'})
        parts.append(g[cols])

    dst = read_sql(session, text("SELECT season, week, team, opponent, dst_points, points_allowed FROM nfl_dst_game "
                                 "WHERE dst_points IS NOT NULL AND points_allowed IS NOT NULL"))
    if not dst.empty:
        dst['fpts'] = dst.dst_points.astype(float) - pa_bonus_vec(dst.points_allowed)
        dst['slot'] = 'DST'
        dst['player'] = None
        parts.append(dst[cols])

    sched = read_sql(session, text("SELECT season, week, home_team, away_team, home_score, away_score FROM nfl_schedule "
                                   "WHERE home_score IS NOT NULL AND away_score IS NOT NULL"))
    if not sched.empty:
        home = pd.DataFrame({'season': sched.season, 'week': sched.week, 'team': sched.home_team,
                             'opponent': sched.away_team, 'slot': PTS_SLOT, 'fpts': sched.home_score.astype(float), 'player': None})
        away = pd.DataFrame({'season': sched.season, 'week': sched.week, 'team': sched.away_team,
                             'opponent': sched.home_team, 'slot': PTS_SLOT, 'fpts': sched.away_score.astype(float), 'player': None})
        parts.extend([home, away])

    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    table = table.drop_duplicates(['season', 'week', 'team', 'slot'])
    _SLOT_TABLE_CACHE[key] = table
    return table


def _wide(table: pd.DataFrame) -> pd.DataFrame:
    """
    (season, week, team) x TEAM_SLOTS, with the opponent and the starting QB's
    name as columns, sorted chronologically within each team.
    """
    cols = ['season', 'week', 'team', 'opponent', 'qb_name'] + TEAM_SLOTS
    if table.empty:
        return pd.DataFrame(columns=cols)
    w = table.pivot_table(index=['season', 'week', 'team'], columns='slot', values='fpts', aggfunc='first')
    w = w.reindex(columns=TEAM_SLOTS)
    grp = table.groupby(['season', 'week', 'team'])
    w['opponent'] = grp['opponent'].first()
    qb = table[table.slot == 'QB'].set_index(['season', 'week', 'team'])['player']
    w['qb_name'] = qb.reindex(w.index)
    return w.reset_index().sort_values(['team', 'season', 'week']).reset_index(drop=True)[cols]


def league_wide_table(session: Session) -> pd.DataFrame:
    key = id(session.get_bind())
    if key not in _WIDE_CACHE:
        _WIDE_CACHE[key] = _wide(league_slot_table(session))
    return _WIDE_CACHE[key]


def _before(df: pd.DataFrame, before: Tuple[int, int] | None) -> pd.DataFrame:
    """Rows played before (season, week); all rows when before is None."""
    if before is None:
        return df
    s, wk = before
    return df[(df.season < s) | ((df.season == s) & (df.week < wk))]


def team_games(session: Session, team: str, before: Tuple[int, int] | None = None) -> pd.DataFrame:
    """A team's rows of the wide table in chronological order, optionally only those before (season, week)."""
    w = league_wide_table(session)
    return _before(w[w.team == team], before).reset_index(drop=True)


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


def _opponent_frame(w: pd.DataFrame) -> pd.DataFrame:
    return w.rename(columns={s: f"opp_{s}" for s in TEAM_SLOTS})[['season', 'week', 'team'] + [f"opp_{s}" for s in TEAM_SLOTS]]


def _with_opponents(games: pd.DataFrame, w: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """(team slots, opponent slots) arrays for the given team-games, joined on the opponent's row of the same week."""
    g = games.merge(_opponent_frame(w), left_on=['season', 'week', 'opponent'], right_on=['season', 'week', 'team'],
                    how='inner', suffixes=('', '_o'))
    return g[TEAM_SLOTS].to_numpy(float), g[[f"opp_{s}" for s in TEAM_SLOTS]].to_numpy(float)


def league_matrices(session: Session, before: Tuple[int, int] | None = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    (within, cross, team-games) Spearman matrices pooled over every team-game
    in history (before `before`): within[i, j] between a team's own slots,
    cross[i, j] between a team's slot i and its opponent's slot j. Falls back
    to the hand-set defaults with fewer than MIN_LEAGUE_GAMES team-games.
    """
    key = (id(session.get_bind()), before)
    if key in _LEAGUE_CACHE:
        return _LEAGUE_CACHE[key]
    h = _before(league_wide_table(session), before)
    n = len(h)
    if n < MIN_LEAGUE_GAMES:
        out = (np.eye(N_SLOTS), _default_cross_correlation(), n)
    else:
        X = h[TEAM_SLOTS].to_numpy(float)
        within, _ = _pairwise_spearman(X, X, MIN_PAIR_OBS)
        within = np.where(np.isnan(within), 0.0, within)
        np.fill_diagonal(within, 1.0)
        T, O = _with_opponents(h, h)
        cross, _ = _pairwise_spearman(T, O, MIN_PAIR_OBS)
        cross = np.where(np.isnan(cross), _default_cross_correlation(), cross)
        cross = (cross + cross.T) / 2          # every game is in the pool from both sides
        out = (within, cross, n)
    _LEAGUE_CACHE[key] = out
    return out


def _shrink(team: np.ndarray, n: np.ndarray | int, league: np.ndarray, k: float) -> np.ndarray:
    """(n * team + k * league) / (n + k) elementwise; league where team is nan; team where k is 0."""
    team = np.asarray(team, float)
    n = np.broadcast_to(np.asarray(n, float), team.shape)
    if not np.isfinite(k):
        return league.copy()
    filled = np.where(np.isnan(team), league, team)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (n * filled + k * league) / (n + k)
    return np.where(np.isnan(team) | (n + k <= 0), league, out)


def build_team_correlation(session: Session, team: str, min_games: int = MIN_GAMES_WITHIN,
                           league_shrink: float = LEAGUE_SHRINK_GAMES, before: Tuple[int, int] | None = None
                           ) -> Tuple[np.ndarray, bool]:
    """
    Within-team Spearman matrix over TEAM_SLOTS for `team`: its own pairwise
    matrix shrunk toward the league matrix by `league_shrink` pseudo-games
    (per pair, by that pair's observation count). With fewer than `min_games`
    games the league matrix is returned with False.
    """
    league, _, n_league = league_matrices(session, before)
    g = team_games(session, team, before)
    X = g[TEAM_SLOTS].to_numpy(float)
    if len(X) < min_games:
        log.debug(f"  {team}: {len(X)} games (need {min_games}), league correlation")
        return league.copy(), False
    own, N = _pairwise_spearman(X, X, MIN_PAIR_OBS)
    R = _shrink(own, N, league, league_shrink)
    np.fill_diagonal(R, 1.0)
    return R, True


def team_history_summary(session: Session, team: str) -> dict:
    """Games behind a team's matrix, its latest starting QB and his games in that history (for diagnostics)."""
    g = team_games(session, team)
    if not len(g):
        return {"games": 0, "last": None, "qb": None, "qb_games": 0}
    qb = g.qb_name.iloc[-1]
    return {"games": int(len(g)), "last": (int(g.season.iloc[-1]), int(g.week.iloc[-1])),
            "qb": qb, "qb_games": int((g.qb_name == qb).sum()) if isinstance(qb, str) else 0}


def describe_team_history(summary: dict) -> str:
    """One compact token for the sampling report: '70g, QB J.Smith 12g'."""
    if not summary.get("games"):
        return "no history"
    qb = f", QB {summary['qb']} {summary['qb_games']}g" if summary.get("qb") else ""
    return f"{summary['games']}g{qb}"


def build_cross_team_correlation(session: Session, team_a: str, team_b: str, min_games: int = MIN_GAMES_CROSS,
                                 league_shrink: float = LEAGUE_SHRINK_GAMES, before: Tuple[int, int] | None = None
                                 ) -> Tuple[np.ndarray, bool]:
    """
    Cross matrix [A slot i, B slot j] for a game between team_a and team_b:
    A's slots against its opponents' and B's opponents' against B's, pooled
    over every game each team played and shrunk toward the league cross
    matrix by `league_shrink` pseudo-games. Returns (matrix, True), or the
    league matrix with False when neither team has enough history.
    """
    w = league_wide_table(session)
    _, league, _ = league_matrices(session, before)
    Xa, Oa = _with_opponents(team_games(session, team_a, before), w)      # A's slots, A's opponents' slots
    Xb, Ob = _with_opponents(team_games(session, team_b, before), w)      # B's slots, B's opponents' slots
    if len(Xa) < min_games and len(Xb) < min_games:
        return league.copy(), False

    num = np.zeros((N_SLOTS, N_SLOTS)); den = np.zeros((N_SLOTS, N_SLOTS))
    if len(Xa) >= min_games:
        R, N = _pairwise_spearman(Xa, Oa, MIN_PAIR_OBS)           # [A slot i, opponent slot j]
        ok = ~np.isnan(R); num[ok] += (R * N)[ok]; den[ok] += N[ok]
    if len(Xb) >= min_games:
        R, N = _pairwise_spearman(Ob, Xb, MIN_PAIR_OBS)           # [opponent slot i, B slot j]
        ok = ~np.isnan(R); num[ok] += (R * N)[ok]; den[ok] += N[ok]
    with np.errstate(invalid='ignore', divide='ignore'):
        own = np.where(den > 0, num / den, np.nan)
    cross = _shrink(own, den, league, league_shrink)
    log.debug(f"  cross {team_a} x {team_b}: {len(Xa)} + {len(Xb)} games")
    return cross, True


def _default_cross_correlation() -> np.ndarray:
    """
    Hand-set cross matrix used only before there is league history: offense
    vs opposing offense slightly positive (game flow), offense vs opposing
    takeaways negative, takeaways vs opposing points negative, the two scores
    nearly independent.
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


