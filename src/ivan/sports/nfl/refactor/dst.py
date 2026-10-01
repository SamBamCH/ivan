"""
DST projection from the game lines, built the way DraftKings scores it.

A DST line is two things: the points-allowed bracket, a step function of
what the opposing offense scores, and everything the defense takes (sacks,
turnovers, blocks, return and defensive touchdowns, safeties). The books do
not post a DST market, but they price what drives both pieces: each side's
implied total from the spread and the game total.

  * Opponent points: the opponent's implied total plus a residual bootstrapped
    from history (points allowed minus the closing implied total, sd about 9).
    The bracket is applied to that draw, so the universe where the opponent
    scores 31 is the same universe where its offense is scoring and the DST
    bracket is -1. Nothing is parametric: the residual pool keeps the real
    shape of NFL scores.
  * Takeaway points: a linear fit on the two implied totals (a trailing
    opponent throws into the pass rush) plus a residual bootstrapped from
    history, right-skewed by return touchdowns.
  * The two residuals are drawn with their historical rank correlation: fewer
    opponent points go with more takeaways.

fit_dst_odds_model runs at model-build time and writes dst_odds_params.json
with a validation table (model vs actual by opponent implied total).
project_dst_samples runs at ingest for each DST on the slate. The correlated
sampler (correlated_samples.py) draws the same pieces from the game-level
copula, so in every simulation the DST moves with the opposing offense.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from sqlalchemy import or_, and_, text

from ivan.database.session import read_sql as _read_sql
from ivan.sports.nfl.models import NFLGameOdds
from ivan.sports.nfl.paths import MODELS_DIR

PARAMS_FILE = "dst_odds_params.json"
MIN_TRAINING_GAMES = 200

# DraftKings points-allowed brackets: (points allowed up to and including, bonus); above the last: -4.
PA_BRACKETS = ((0, 10.0), (6, 7.0), (13, 4.0), (20, 1.0), (27, 0.0), (34, -1.0))
PA_FLOOR = -4.0

_MODEL_CACHE: dict[Path, dict | None] = {}


def pa_bonus_vec(points_allowed) -> np.ndarray:
    """DraftKings points-allowed bonus for an array of points allowed."""
    pa = np.asarray(points_allowed, float)
    out = np.full(pa.shape, PA_FLOOR)
    for hi, bonus in reversed(PA_BRACKETS):
        out[pa <= hi] = bonus
    return out


def implied_totals(total: float, home_margin: float) -> tuple[float, float]:
    """(home_total, away_total) from the game total and the home team's expected margin."""
    home = total / 2.0 + home_margin / 2.0
    return home, total - home


# ----------------------------------------------------------------------------- training
def training_frame(session) -> pd.DataFrame:
    """
    One row per historical team-game: DST points and their points-allowed
    component, the team's and opponent's scores, and the closing implied totals.
    """
    cols = ["season", "week", "team", "dst_points", "points_allowed", "team_score", "opp_score",
            "team_implied", "opp_implied", "total"]
    dst = _read_sql(session, text("SELECT season, week, team, opponent, dst_points, points_allowed FROM nfl_dst_game"))
    sched = _read_sql(session, text("""SELECT season, week, home_team, away_team, home_score, away_score, spread_line, total_line
                                       FROM nfl_schedule WHERE spread_line IS NOT NULL AND total_line IS NOT NULL
                                         AND home_score IS NOT NULL AND away_score IS NOT NULL"""))
    if dst.empty or sched.empty:
        return pd.DataFrame(columns=cols)
    home_tt, away_tt = implied_totals(sched.total_line.astype(float), sched.spread_line.astype(float))
    home = pd.DataFrame({"season": sched.season, "week": sched.week, "team": sched.home_team,
                         "team_score": sched.home_score, "opp_score": sched.away_score,
                         "team_implied": home_tt, "opp_implied": away_tt, "total": sched.total_line})
    away = pd.DataFrame({"season": sched.season, "week": sched.week, "team": sched.away_team,
                         "team_score": sched.away_score, "opp_score": sched.home_score,
                         "team_implied": away_tt, "opp_implied": home_tt, "total": sched.total_line})
    lines = pd.concat([home, away], ignore_index=True)
    df = dst.merge(lines, on=["season", "week", "team"], how="inner")
    df = df.dropna(subset=["dst_points", "points_allowed", "team_implied", "opp_implied"]).reset_index(drop=True)
    return df[cols]


def _design(opp_implied, team_implied) -> np.ndarray:
    opp = np.asarray(opp_implied, float)
    team = np.asarray(team_implied, float)
    return np.column_stack([np.ones_like(opp), opp, team])


def _take_mean(coef: dict, team_implied, opp_implied):
    return coef["intercept"] + coef["opp_implied"] * np.asarray(opp_implied, float) + coef["team_implied"] * np.asarray(team_implied, float)


def _expected_pa_bonus(opp_implied, pa_resid_pool: np.ndarray, chunk: int = 256, randomness: float = 1.0) -> np.ndarray:
    """E[bracket(opp_implied + randomness * residual)] over the residual pool, per opponent implied total."""
    opp = np.atleast_1d(np.asarray(opp_implied, float))
    out = np.empty(len(opp))
    for i in range(0, len(opp), chunk):
        pa = np.maximum(0.0, opp[i:i + chunk, None] + randomness * pa_resid_pool[None, :])
        out[i:i + chunk] = pa_bonus_vec(pa).mean(axis=1)
    return out


VALIDATION_BUCKETS = (0, 16, 18, 20, 22, 24, 26, 60)


def fit_dst_odds_model(session, model_dir: Path = MODELS_DIR) -> dict:
    """
    Fit the two-part DST model on every historical team-game with closing
    lines and write the params file:

      takeaway points ~ opp_implied + team_implied, residual pool
      points allowed  = opp_implied + residual pool (bracket applied at sample time)
      team score      = team_implied + residual pool (the offense's latent in the sampler)
    """
    df = training_frame(session)
    if len(df) < MIN_TRAINING_GAMES:
        raise RuntimeError(f"only {len(df)} DST games with lines and scores; need {MIN_TRAINING_GAMES} "
                           "(run fetch_schedules and build_dst_game_table first)")
    pa_pts = pa_bonus_vec(df.points_allowed)
    take = df.dst_points.to_numpy(float) - pa_pts
    X = _design(df.opp_implied, df.team_implied)
    beta, *_ = np.linalg.lstsq(X, take, rcond=None)
    coef = {"intercept": float(beta[0]), "opp_implied": float(beta[1]), "team_implied": float(beta[2])}
    take_fit = X @ beta
    take_resid = take - take_fit
    pa_resid = df.points_allowed.to_numpy(float) - df.opp_implied.to_numpy(float)
    score_resid = df.team_score.to_numpy(float) - df.team_implied.to_numpy(float)
    rho_s = float(st.spearmanr(take_resid, pa_resid).correlation)

    # validation: the model's expected DST line against what happened, by opponent implied total
    model = _expected_pa_bonus(df.opp_implied, pa_resid) + take_fit
    buckets = pd.cut(df.opp_implied, VALIDATION_BUCKETS)
    rows = []
    for b, g in df.assign(model=model, pa_pts=pa_pts, take_pts=take).groupby(buckets, observed=True):
        rows.append({"opp_implied": f"{b.left:g}-{b.right:g}", "n": int(len(g)),
                     "model": round(float(g.model.mean()), 2), "actual": round(float(g.dst_points.mean()), 2),
                     "pa_actual": round(float(g.pa_pts.mean()), 2), "take_actual": round(float(g.take_pts.mean()), 2)})
    resid = df.dst_points.to_numpy(float) - model

    params = {
        "fitted_on": datetime.now().strftime("%Y-%m-%d"),
        "n": int(len(df)),
        "seasons": sorted(int(s) for s in df.season.unique()),
        "take": {"coefficients": coef, "resid_sd": float(take_resid.std(ddof=3)),
                 "r2": float(1.0 - take_resid.var() / take.var()), "mean": float(take.mean())},
        "pa": {"mean_bonus": float(pa_pts.mean()), "resid_sd": float(pa_resid.std())},
        "score_resid_sd": float(score_resid.std()),
        "take_pa_spearman": rho_s,
        "pools": {"take_resid": np.round(take_resid, 3).tolist(),
                  "pa_resid": np.round(pa_resid, 3).tolist(),
                  "score_resid": np.round(score_resid, 3).tolist()},
        "validation": {"mean_model": round(float(model.mean()), 3), "mean_actual": round(float(df.dst_points.mean()), 3),
                       "resid_sd": round(float(resid.std()), 3),
                       "spearman": round(float(st.spearmanr(model, df.dst_points).correlation), 3),
                       "by_opp_implied": rows},
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / PARAMS_FILE).write_text(json.dumps(params))
    _MODEL_CACHE[model_dir] = _prepare(params)
    return params


def _prepare(params: dict) -> dict:
    """Sorted residual pools for quantile lookups (not written to the file)."""
    params = dict(params)
    params["_sorted"] = {k: np.sort(np.asarray(v, float)) for k, v in params["pools"].items()}
    return params


def load_dst_odds_model(model_dir: Path = MODELS_DIR) -> dict | None:
    model_dir = Path(model_dir)
    if model_dir not in _MODEL_CACHE:
        path = model_dir / PARAMS_FILE
        params = json.loads(path.read_text()) if path.exists() else None
        if params is not None and "pools" not in params:
            params = None      # file from the previous (linear) model: rebuild with build_models.py
        _MODEL_CACHE[model_dir] = _prepare(params) if params else None
    return _MODEL_CACHE[model_dir]


def describe_dst_odds_model(params: dict) -> str:
    c = params["take"]["coefficients"]
    v = params["validation"]
    lines = [f"DST = bracket(opp points) + takeaways; takeaways ~ {c['intercept']:.2f} {c['opp_implied']:+.3f}*opp_implied "
             f"{c['team_implied']:+.3f}*team_implied (resid sd {params['take']['resid_sd']:.2f}); "
             f"points allowed = opp_implied + residual (sd {params['pa']['resid_sd']:.1f}); "
             f"n={params['n']}, seasons {params['seasons'][0]}-{params['seasons'][-1]}",
             f"validation: model mean {v['mean_model']:.2f} vs actual {v['mean_actual']:.2f}, "
             f"resid sd {v['resid_sd']:.2f}, spearman {v['spearman']:+.2f}; by opponent implied total:"]
    for r in v["by_opp_implied"]:
        lines.append(f"  {r['opp_implied']:>6}  n={r['n']:4d}  model {r['model']:5.2f}  actual {r['actual']:5.2f}"
                     f"  (bracket {r['pa_actual']:5.2f} + takeaways {r['take_actual']:5.2f})")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- lines
def matchup_lines(session, team: str, opponent: str, now: datetime | None = None,
                  max_age: timedelta = timedelta(hours=12), horizon: timedelta = timedelta(days=10)):
    """
    Consensus (total, home_margin, is_home) for the upcoming team-opponent game
    from game_odds, or None. home_margin follows nfl_schedule's convention:
    positive when the home side is favored. Only a game starting within
    `horizon` (and not more than `max_age` ago) counts, so a stale line from
    a previous meeting is never used.
    """
    now = now or datetime.now()
    rows = session.query(NFLGameOdds).filter(
        NFLGameOdds.market.in_(["totals", "spreads"]),
        or_(and_(NFLGameOdds.home_abbr == team, NFLGameOdds.away_abbr == opponent),
            and_(NFLGameOdds.home_abbr == opponent, NFLGameOdds.away_abbr == team)),
    ).all()
    rows = [r for r in rows if r.point is not None and r.commence_time is not None
            and now - max_age <= r.commence_time <= now + horizon]
    if not rows:
        return None
    game_id = max(rows, key=lambda r: r.commence_time).game_id
    rows = [r for r in rows if r.game_id == game_id]
    # latest snapshot per (bookmaker, market)
    latest: dict[tuple, str] = {}
    for r in rows:
        key = (r.bookmaker, r.market)
        if r.last_update and (key not in latest or r.last_update > latest[key]):
            latest[key] = r.last_update
    rows = [r for r in rows if latest.get((r.bookmaker, r.market)) in (None, r.last_update)]
    totals = [float(r.point) for r in rows if r.market == "totals"]
    home_spreads = [float(r.point) for r in rows if r.market == "spreads" and (r.side or "").lower() == "home"]
    if not home_spreads:
        home_spreads = [-float(r.point) for r in rows if r.market == "spreads" and (r.side or "").lower() == "away"]
    if not totals or not home_spreads:
        return None
    total = float(np.median(totals))
    home_margin = -float(np.median(home_spreads))     # book spread is negative when home is favored
    is_home = rows[0].home_abbr == team
    return total, home_margin, is_home


def game_implied_totals(session, team: str, opponent: str) -> dict | None:
    """{'team_implied', 'opp_implied', 'total', 'home_margin'} for the upcoming game, or None."""
    lines = matchup_lines(session, team, opponent)
    if lines is None:
        return None
    total, home_margin, is_home = lines
    home_tt, away_tt = implied_totals(total, home_margin)
    team_tt, opp_tt = (home_tt, away_tt) if is_home else (away_tt, home_tt)
    return {"team_implied": float(team_tt), "opp_implied": float(opp_tt), "total": total, "home_margin": home_margin}


# ----------------------------------------------------------------------------- sampling
def _quantile(sorted_pool: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Linear interpolation between order statistics (a smoothed bootstrap)."""
    m = len(sorted_pool)
    return np.interp(np.clip(u, 0.0, 1.0) * (m - 1), np.arange(m), sorted_pool)


def team_points_from_latent(params: dict, implied: float, u: np.ndarray, randomness: float = 1.0) -> np.ndarray:
    """A team's points in each universe from its implied total and a uniform latent."""
    return np.maximum(0.0, implied + randomness * _quantile(params["_sorted"]["score_resid"], u))


def dst_from_latent(params: dict, team_implied: float, opp_implied: float,
                    u_opp_points: np.ndarray, u_take: np.ndarray, randomness: float = 1.0) -> np.ndarray:
    """
    DraftKings DST points per universe: the points-allowed bracket of the
    opponent's score drawn from `u_opp_points` (the same latent that drives
    the opposing offense in the sampler) plus takeaway points drawn from
    `u_take`. randomness < 1 shrinks both residuals toward zero; the bracket
    is a step function, so its mean is re-centred on the full-randomness
    expectation and the DST keeps its projected mean like every other player.
    """
    pool = params["_sorted"]["pa_resid"]
    pa = np.maximum(0.0, opp_implied + randomness * _quantile(pool, u_opp_points))
    bracket = pa_bonus_vec(pa)
    if randomness != 1.0:
        bracket = bracket - _expected_pa_bonus([opp_implied], pool, randomness=randomness)[0] \
            + _expected_pa_bonus([opp_implied], pool)[0]
    take = _take_mean(params["take"]["coefficients"], team_implied, opp_implied) \
        + randomness * _quantile(params["_sorted"]["take_resid"], u_take)
    return bracket + take


def expected_dst(params: dict, team_implied: float, opp_implied: float) -> tuple[float, float]:
    """(expected bracket points, expected takeaway points) for a matchup."""
    pa = float(_expected_pa_bonus([opp_implied], params["_sorted"]["pa_resid"])[0])
    take = float(_take_mean(params["take"]["coefficients"], team_implied, opp_implied))
    return pa, take


def project_dst_samples(session, team: str, opponent: str, rng: np.random.Generator,
                        n_samples: int = 10000, model_dir: Path = MODELS_DIR):
    """
    Samples of DraftKings DST points for `team` against `opponent` from the
    current lines, or None when the lines or the fitted model are missing.
    The second return value describes the inputs for logging.
    """
    params = load_dst_odds_model(model_dir)
    if params is None or not team or not opponent:
        return None
    g = game_implied_totals(session, team, opponent)
    if g is None:
        return None
    rho = 2.0 * np.sin(np.pi * params["take_pa_spearman"] / 6.0)      # Spearman -> Gaussian copula
    Z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n_samples)
    u = st.norm.cdf(Z)
    samples = dst_from_latent(params, g["team_implied"], g["opp_implied"], u[:, 0], u[:, 1])
    pa_mean, take_mean = expected_dst(params, g["team_implied"], g["opp_implied"])
    info = {"total": g["total"], "team_implied": round(g["team_implied"], 2), "opp_implied": round(g["opp_implied"], 2),
            "mu": round(float(samples.mean()), 2), "bracket": round(pa_mean, 2), "takeaways": round(take_mean, 2)}
    return samples, info
