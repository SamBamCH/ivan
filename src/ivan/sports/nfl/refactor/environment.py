# src/ivan/sports/nfl/refactor/environment.py
"""
The projected game environment: layer 1 of the game model.

For each team in a game, before anything about players, draw jointly
    points      the team's score
    plays       offensive plays (pass attempts + sacks + carries)
    pass_rate   dropbacks / plays
from the game's lines (total, spread) and the two teams' tendencies in
history, with the residual dependence between the six quantities of a game
estimated from past games (a team that scores more runs more plays and
passes less; both teams' play counts move together with pace; a trailing
team passes more).

Everything is fitted on team-games strictly before the history cutoff
(ivan.sports.nfl.actuals.history_cutoff), so a replay sees only what was
known at its lock. The training rows come from nfl_schedule (closing
spread_line / total_line and final scores) and nfl_stats (attempts, carries,
sacks_suffered per player-week, summed per team-week).

    fit = fit_environment(team_games(session, before=cutoff))
    env = predict(fit, games)          # per team-game mean and sd of each quantity
    sims = draw(fit, games, n, rng)    # joint draws: points / plays / pass_rate / dropbacks / rushes

The rolling backtest (backtest) refits week by week and reports projected
vs actual for each quantity, against two baselines: the league average and
the lines without team tendencies. Layer 2 (usage shares) and layer 3
(market marginals) build on the draws; see the project memory for the plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from ivan.database.session import read_sql

QUANTITIES = ("points", "plays", "pass_rate")
TEAM_EFFECT_K = 8.0          # pseudo-games pulling a team's tendency toward the league
TEAM_EFFECT_WINDOW = 24      # a team's tendency is estimated from its last N team-games
# which quantities get team (offense) / opponent (defense) tendencies on top of the lines; the rolling backtest
# (scripts/nfl/environment.py backtest --grid) is where these are chosen: the market already prices team strength
# into points, so tendencies only add noise there, while pass rate is a genuine team habit
# grid 2024w1-2026w5 (1234 team-games): points never gain (market already prices team strength), plays gain a little
# with heavy shrinkage (MAE 6.68 vs 6.73, corr .15 vs .11), pass rate gains clearly (corr .31 vs .19)
DEFAULT_TEAM_EFFECTS = {"points": False, "plays": {"k": 50.0, "window": 34}, "pass_rate": {"k": 8.0, "window": 17}}
# per-quantity design columns (implied_points = total/2 + spread/2, so a quantity never gets all three)
DESIGN = {"points": ("implied_points",), "plays": ("total", "spread", "abs_spread"), "pass_rate": ("spread", "total")}
PASS_RATE_BOUNDS = (0.30, 0.85)
PLAYS_MIN = 40.0


# --------------------------------------------------------------------------- data
def team_games(session, before: Optional[Tuple[int, int]] = None, since_season: int = 2022) -> pd.DataFrame:
    """
    One row per team-game with lines and actuals:
      season, week, game_id, team, opponent, is_home, spread (team margin: positive when the team is
      favored), total, implied_points, implied_opp_points, points, opp_points, pass_att, sacks, rush_att,
      dropbacks, plays, pass_rate
    Rows without a closing line or a final score are dropped. `before` keeps weeks strictly earlier.
    """
    sched = read_sql(session, text("""SELECT game_id, season, game_type, week, gameday, home_team, away_team, home_score, away_score,
                                             spread_line, total_line FROM nfl_schedule WHERE season >= :s""").bindparams(s=since_season))
    stats = read_sql(session, text("""SELECT season, week, team, opponent_team AS opponent,
                                             sum(coalesce(attempts, 0)) AS pass_att, sum(coalesce(sacks_suffered, 0)) AS sacks,
                                             sum(coalesce(carries, 0)) AS rush_att
                                      FROM nfl_stats WHERE season >= :s GROUP BY season, week, team""").bindparams(s=since_season))
    sched = sched.dropna(subset=["home_score", "away_score", "spread_line", "total_line"])
    rows = []
    for r in sched.itertuples():
        for team, opp, home in ((r.home_team, r.away_team, 1), (r.away_team, r.home_team, 0)):
            margin = float(r.spread_line) if home else -float(r.spread_line)       # nfl_schedule: positive = home favored
            total = float(r.total_line)
            pts = float(r.home_score if home else r.away_score); opp_pts = float(r.away_score if home else r.home_score)
            rows.append({"season": int(r.season), "week": int(r.week), "game_id": r.game_id, "game_type": r.game_type, "gameday": r.gameday,
                         "team": team, "opponent": opp, "is_home": home, "spread": margin, "total": total,
                         "implied_points": total / 2 + margin / 2, "implied_opp_points": total / 2 - margin / 2,
                         "points": pts, "opp_points": opp_pts})
    df = pd.DataFrame(rows)
    stats["team"] = stats.team.replace({"LA": "LAR"})
    df["team"] = df.team.replace({"LA": "LAR"}); df["opponent"] = df.opponent.replace({"LA": "LAR"})
    df = df.merge(stats[["season", "week", "team", "pass_att", "sacks", "rush_att"]], on=["season", "week", "team"], how="inner")
    df["dropbacks"] = df.pass_att + df.sacks
    df["plays"] = df.dropbacks + df.rush_att
    df = df[df.plays >= PLAYS_MIN].copy()
    df["pass_rate"] = df.dropbacks / df.plays
    if before is not None:
        s, w = before
        df = df[(df.season < s) | ((df.season == s) & (df.week < w))]
    return df.sort_values(["season", "week", "game_id", "is_home"]).reset_index(drop=True)


# --------------------------------------------------------------------------- the fit
@dataclass
class EnvironmentFit:
    coef: Dict[str, np.ndarray]                 # quantity -> [intercept] + one coefficient per DESIGN[quantity] column
    resid_sd: Dict[str, float]                  # quantity -> residual sd after lines + team effects
    team_effect: Dict[str, Dict[str, float]]    # quantity -> team -> shrunk mean residual (offense)
    opp_effect: Dict[str, Dict[str, float]]     # quantity -> opponent -> shrunk mean residual induced in opponents (defense)
    resid_corr: np.ndarray                      # 6x6 over (points, plays, pass_rate) x (team, opponent)
    league: Dict[str, float]                    # quantity -> league mean (the dumb baseline)
    n_games: int = 0
    before: Optional[Tuple[int, int]] = None
    use_team_effects: Dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_TEAM_EFFECTS))
    k: float = TEAM_EFFECT_K
    window: int = TEAM_EFFECT_WINDOW
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = []
        for q in QUANTITIES:
            terms = " ".join(f"{b:+.3g}*{c}" for b, c in zip(self.coef[q][1:], DESIGN[q]))
            parts.append(f"{q} = {self.coef[q][0]:.3g} {terms} (sd {self.resid_sd[q]:.3g}{', team effects' if self.use_team_effects.get(q) else ''})")
        return (f"environment fit on {self.n_games} team-games" + (f" before {self.before}" if self.before else "") + ": " + "; ".join(parts))


def _design(df: pd.DataFrame, q: str) -> np.ndarray:
    cols = []
    for c in DESIGN[q]:
        cols.append(np.abs(df.spread.values) if c == "abs_spread" else df[c].values)
    return np.column_stack([np.ones(len(df))] + cols).astype(float)


def _shrunk_effects(df: pd.DataFrame, resid: np.ndarray, key: str, k: float = TEAM_EFFECT_K, window: int = TEAM_EFFECT_WINDOW) -> Dict[str, float]:
    """Mean residual of the last `window` games per `key` (team or opponent), shrunk toward 0 by k pseudo-games."""
    out: Dict[str, float] = {}
    d = df.assign(_r=resid).sort_values(["season", "week"])
    for t, g in d.groupby(key):
        r = g._r.values[-window:]
        out[str(t)] = float(r.sum() / (len(r) + k))
    return out


def fit_environment(df: pd.DataFrame, before: Optional[Tuple[int, int]] = None, use_team_effects: Optional[Dict[str, bool]] = None,
                    k: float = TEAM_EFFECT_K, window: int = TEAM_EFFECT_WINDOW) -> EnvironmentFit:
    """
    Least squares of each quantity on its line columns (DESIGN), then, for the
    quantities in use_team_effects, shrunk team (offense) and opponent
    (defense) effects on the residuals, then the within-game residual
    correlation. Needs ~100+ team-games; fewer still works but the effects are
    mostly shrinkage.
    """
    df = df.reset_index(drop=True)
    use = {**DEFAULT_TEAM_EFFECTS, **(use_team_effects or {})}
    coef, resid_sd, team_eff, opp_eff, league = {}, {}, {}, {}, {}
    resids = {}
    for q in QUANTITIES:
        X = _design(df, q)
        y = df[q].values.astype(float)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        setting = use.get(q)
        if setting:
            kq = float(setting.get("k", k)) if isinstance(setting, dict) else k
            wq = int(setting.get("window", window)) if isinstance(setting, dict) else window
            te = _shrunk_effects(df, r, "team", k=kq, window=wq)
            r2 = r - df.team.map(te).fillna(0.0).values
            oe = _shrunk_effects(df, r2, "opponent", k=kq, window=wq)
            r3 = r2 - df.opponent.map(oe).fillna(0.0).values
        else:
            te, oe, r3 = {}, {}, r
        coef[q], resid_sd[q], team_eff[q], opp_eff[q], league[q] = b, float(r3.std(ddof=X.shape[1])), te, oe, float(y.mean())
        resids[q] = r3
    # within-game residual correlation: pair each team-game with its opponent's row
    R = np.eye(6)
    key = df.set_index(["game_id", "team"]).index
    idx = {k: i for i, k in enumerate(key)}
    pairs = [(i, idx[(g, o)]) for i, (g, o) in enumerate(zip(df.game_id, df.opponent)) if (g, o) in idx]
    if len(pairs) >= 50:
        a = np.array([i for i, _ in pairs]); b_ = np.array([j for _, j in pairs])
        M = np.column_stack([resids[q][a] for q in QUANTITIES] + [resids[q][b_] for q in QUANTITIES])
        R = np.corrcoef(M, rowvar=False)
        R = 0.5 * (R + R.T); np.fill_diagonal(R, 1.0)
    return EnvironmentFit(coef=coef, resid_sd=resid_sd, team_effect=team_eff, opp_effect=opp_eff, resid_corr=R, league=league,
                          n_games=int(len(df)), before=before, use_team_effects=use, k=k, window=window)


# --------------------------------------------------------------------------- prediction and draws
def predict(fit: EnvironmentFit, games: pd.DataFrame, team_effects: bool = True) -> pd.DataFrame:
    """
    Per team-game row (needs team, opponent, spread, total, implied_points): the
    mean and sd of each quantity. With team_effects=False only the lines speak
    (the second baseline of the backtest).
    """
    out = games.copy()
    for q in QUANTITIES:
        mu = _design(games, q) @ fit.coef[q]
        if team_effects and fit.use_team_effects.get(q):
            mu = mu + games.team.map(fit.team_effect[q]).fillna(0.0).values + games.opponent.map(fit.opp_effect[q]).fillna(0.0).values
        out[f"{q}_mean"] = mu
        out[f"{q}_sd"] = fit.resid_sd[q]
        out[f"{q}_league"] = fit.league[q]
    out["plays_mean"] = np.maximum(out.plays_mean, PLAYS_MIN)
    out["pass_rate_mean"] = np.clip(out.pass_rate_mean, *PASS_RATE_BOUNDS)
    return out


def draw(fit: EnvironmentFit, game: pd.DataFrame, n: int, rng: np.random.Generator) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Joint draws for one game: `game` holds the two team rows (team, opponent,
    spread, total, implied_points). Returns {team: {points, plays, pass_rate,
    dropbacks, rushes}} with n values each. Points are clipped at 0; a team's
    plays at PLAYS_MIN; pass rate inside PASS_RATE_BOUNDS.
    """
    assert len(game) == 2, "draw() takes the two team rows of one game"
    pred = predict(fit, game)
    mu = np.concatenate([pred[f"{q}_mean"].values[[0]] for q in QUANTITIES] + [pred[f"{q}_mean"].values[[1]] for q in QUANTITIES])
    sd = np.array([fit.resid_sd[q] for q in QUANTITIES] * 2)
    cov = np.outer(sd, sd) * fit.resid_corr
    z = rng.multivariate_normal(mu, cov, size=n, method="cholesky" if _is_pd(cov) else "eigh")
    out = {}
    for k, team in enumerate(game.team.values):
        pts = np.maximum(z[:, 3 * k + 0], 0.0)
        plays = np.maximum(z[:, 3 * k + 1], PLAYS_MIN)
        pr = np.clip(z[:, 3 * k + 2], *PASS_RATE_BOUNDS)
        dropbacks = plays * pr
        out[str(team)] = {"points": pts, "plays": plays, "pass_rate": pr, "dropbacks": dropbacks, "rushes": plays - dropbacks}
    return out


def _is_pd(A: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(A); return True
    except np.linalg.LinAlgError:
        return False


# --------------------------------------------------------------------------- the rolling backtest
def backtest(all_games: pd.DataFrame, test_from: Tuple[int, int], min_train: int = 200, use_team_effects: Optional[Dict[str, bool]] = None,
             k: float = TEAM_EFFECT_K, window: int = TEAM_EFFECT_WINDOW) -> pd.DataFrame:
    """
    Refit before every (season, week) from `test_from` on and predict that
    week's team-games. Returns the test rows with, per quantity, the model
    mean/sd, the lines-only mean and the league mean as of that week, plus
    the normal-PIT of the actual under the model.
    """
    weeks = sorted({(int(s), int(w)) for s, w in zip(all_games.season, all_games.week) if (int(s), int(w)) >= tuple(test_from)})
    out = []
    for s, w in weeks:
        train = all_games[(all_games.season < s) | ((all_games.season == s) & (all_games.week < w))]
        test = all_games[(all_games.season == s) & (all_games.week == w)]
        if len(train) < min_train or test.empty:
            continue
        fit = fit_environment(train, before=(s, w), use_team_effects=use_team_effects, k=k, window=window)
        p = predict(fit, test, team_effects=True)
        p0 = predict(fit, test, team_effects=False)
        for q in QUANTITIES:
            p[f"{q}_lines_only"] = p0[f"{q}_mean"].values
            from math import erf, sqrt
            zq = (p[q].values - p[f"{q}_mean"].values) / p[f"{q}_sd"].values
            p[f"{q}_pit"] = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in zq])
        out.append(p)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def backtest_summary(bt: pd.DataFrame) -> dict:
    """Projected vs actual per quantity: bias, MAE, baselines, correlation, PIT coverage, plus buckets by spread and total."""
    if bt.empty:
        return {"n": 0}
    def block(g: pd.DataFrame, q: str) -> dict:
        a = g[q].values; m = g[f"{q}_mean"].values; l = g[f"{q}_lines_only"].values; lg = g[f"{q}_league"].values; pit = g[f"{q}_pit"].values
        return {"n": int(len(g)), "actual_mean": float(a.mean()), "proj_mean": float(m.mean()), "bias": float((a - m).mean()),
                "mae": float(np.abs(a - m).mean()), "mae_lines_only": float(np.abs(a - l).mean()), "mae_league": float(np.abs(a - lg).mean()),
                "corr": float(np.corrcoef(a, m)[0, 1]) if a.std() > 0 and m.std() > 0 else None,
                "corr_lines_only": float(np.corrcoef(a, l)[0, 1]) if a.std() > 0 and l.std() > 0 else None,
                "sd_ratio": float((a - m).std() / g[f"{q}_sd"].mean()),
                "coverage_10_90": float(((pit >= 0.1) & (pit <= 0.9)).mean()), "pit_mean": float(pit.mean())}
    out = {"n": int(len(bt)), "seasons": sorted(int(s) for s in bt.season.unique()), "quantities": {}}
    for q in QUANTITIES:
        out["quantities"][q] = {"overall": block(bt, q),
                                "by_season": [{"season": int(s), **block(g, q)} for s, g in bt.groupby("season")],
                                "by_spread": [{"spread": str(k), **block(g, q)} for k, g in bt.groupby(pd.cut(bt.spread, [-30, -7, -3, 3, 7, 30]), observed=True)],
                                "by_total": [{"total": str(k), **block(g, q)} for k, g in bt.groupby(pd.cut(bt.total, [30, 40, 44, 48, 70]), observed=True)]}
    # the dependence the draws will reproduce: residual correlation on the test rows
    cols = [f"{q}_res" for q in QUANTITIES]
    r = bt.copy()
    for q in QUANTITIES:
        r[f"{q}_res"] = r[q] - r[f"{q}_mean"]
    out["residual_corr_within_team"] = r[cols].corr().round(3).to_dict()
    return out


def grid_backtest(all_games: pd.DataFrame, test_from: Tuple[int, int], ks: Sequence[float] = (4, 8, 20, 50),
                  windows: Sequence[int] = (17, 34, 68)) -> pd.DataFrame:
    """
    For every quantity, the out-of-sample MAE and correlation with team effects
    off and with each (k, window) on. One row per (quantity, setting); the
    lines-only row is the reference. This is where DEFAULT_TEAM_EFFECTS,
    TEAM_EFFECT_K and TEAM_EFFECT_WINDOW are chosen.
    """
    rows = []
    off = backtest(all_games, test_from, use_team_effects={q: False for q in QUANTITIES})
    # note: with a dict setting per quantity the grid's k/window override it, so the grid compares like with like
    for q in QUANTITIES:
        a = off[q].values; m = off[f"{q}_mean"].values
        rows.append({"quantity": q, "team_effects": False, "k": None, "window": None, "n": len(off), "mae": float(np.abs(a - m).mean()),
                     "corr": float(np.corrcoef(a, m)[0, 1]), "bias": float((a - m).mean())})
    for k in ks:
        for w in windows:
            on = backtest(all_games, test_from, use_team_effects={q: True for q in QUANTITIES}, k=k, window=w)
            for q in QUANTITIES:
                a = on[q].values; m = on[f"{q}_mean"].values
                rows.append({"quantity": q, "team_effects": True, "k": k, "window": w, "n": len(on), "mae": float(np.abs(a - m).mean()),
                             "corr": float(np.corrcoef(a, m)[0, 1]), "bias": float((a - m).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- the slate's environment (written as a run artifact)
ENVIRONMENT_COLUMNS = ["season", "week", "team", "opponent", "is_home", "total", "spread", "implied_points", "points_mean", "points_sd",
                       "plays_mean", "plays_sd", "pass_rate_mean", "pass_rate_sd", "points_league", "plays_league", "pass_rate_league",
                       "n_train", "fit_before"]


def slate_games(session) -> pd.DataFrame:
    """The games of the loaded slate (team / opponent pairs in nfl_player) with their consensus lines from game_odds, or empty."""
    from ivan.sports.nfl.models import NFLPlayer
    from ivan.sports.nfl.refactor.dst import matchup_lines
    pairs = {}
    for team, opp in session.query(NFLPlayer.team, NFLPlayer.opponent).distinct().all():
        if team and opp:
            pairs[(str(team), str(opp))] = True
    rows = []
    seen = set()
    for team, opp in sorted(pairs):
        if (opp, team) in seen:
            continue
        seen.add((team, opp))
        lines = matchup_lines(session, team, opp)
        if not lines:
            continue
        total, home_margin, is_home = lines
        for t, o, home in ((team, opp, bool(is_home)), (opp, team, not bool(is_home))):
            margin = float(home_margin) if home else -float(home_margin)
            rows.append({"team": t, "opponent": o, "is_home": int(home), "total": float(total), "spread": margin,
                         "implied_points": float(total) / 2 + margin / 2})
    return pd.DataFrame(rows)


def slate_environment(session) -> pd.DataFrame:
    """
    The projected environment of every game on the loaded slate, fitted on
    team-games strictly before the history cutoff (live: everything; replay:
    the slate's week). One row per team with the mean / sd of points, plays
    and pass rate, ready for the run artifact and for layer 2.
    """
    from ivan.sports.nfl.actuals import history_cutoff, week_for_date
    from ivan.core import clock
    games = slate_games(session)
    if games.empty:
        return pd.DataFrame(columns=ENVIRONMENT_COLUMNS)
    before = history_cutoff(session)
    hist = team_games(session, before=before)
    if len(hist) < 100:
        return pd.DataFrame(columns=ENVIRONMENT_COLUMNS)
    fit = fit_environment(hist, before=before)
    pred = predict(fit, games)
    sw = week_for_date(session, clock.now())
    pred["season"], pred["week"] = (sw if sw else (None, None))
    pred["n_train"] = fit.n_games
    pred["fit_before"] = str(before) if before else ""
    return pred[ENVIRONMENT_COLUMNS]
