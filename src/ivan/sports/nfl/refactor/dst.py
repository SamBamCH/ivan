"""
DST projection from the game lines.

The books do not post a DST fantasy-points market, but they do price what
drives it: the opponent's implied total (fewer points allowed, more punts and
desperation throws) and the team's own (a lead means the opponent passes into
a pass rush). The mean is a linear fit of historical DraftKings DST points on
the two implied totals from the closing lines in nfl_schedule. The *shape*
is not parametric: samples are the mean plus residuals bootstrapped from the
historical games whose fitted mean is closest, so the right skew (return
touchdowns) and the -4 floor survive.

fit_dst_odds_model runs at model-build time and writes dst_odds_params.json;
project_dst_samples runs at ingest for each DST on the slate.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import or_, and_, text

from ivan.database.session import read_sql as _read_sql
from ivan.sports.nfl.models import NFLGameOdds
from ivan.sports.nfl.paths import MODELS_DIR

PARAMS_FILE = "dst_odds_params.json"
DK_DST_FLOOR = -4.0       # 35+ points allowed and nothing else
POOL_SIZE = 300           # historical games lending their residuals to one projection
MIN_TRAINING_GAMES = 200

_MODEL_CACHE: dict[Path, dict | None] = {}


def implied_totals(total: float, home_margin: float) -> tuple[float, float]:
    """(home_total, away_total) from the game total and the home team's expected margin."""
    home = total / 2.0 + home_margin / 2.0
    return home, total - home


def training_frame(session) -> pd.DataFrame:
    """One row per historical team-game with DST points and the closing implied totals."""
    dst = _read_sql(session, text("SELECT season, week, team, opponent, dst_points FROM nfl_dst_game"))
    sched = _read_sql(session, text("""SELECT season, week, home_team, away_team, spread_line, total_line
                                       FROM nfl_schedule WHERE spread_line IS NOT NULL AND total_line IS NOT NULL"""))
    if dst.empty or sched.empty:
        return pd.DataFrame(columns=["season", "week", "team", "dst_points", "team_implied", "opp_implied", "total"])
    home_tt, away_tt = implied_totals(sched.total_line.astype(float), sched.spread_line.astype(float))
    home = pd.DataFrame({"season": sched.season, "week": sched.week, "team": sched.home_team,
                         "team_implied": home_tt, "opp_implied": away_tt, "total": sched.total_line})
    away = pd.DataFrame({"season": sched.season, "week": sched.week, "team": sched.away_team,
                         "team_implied": away_tt, "opp_implied": home_tt, "total": sched.total_line})
    lines = pd.concat([home, away], ignore_index=True)
    df = dst.merge(lines, on=["season", "week", "team"], how="inner")
    return df.dropna(subset=["dst_points", "team_implied", "opp_implied"]).reset_index(drop=True)


def _design(opp_implied, team_implied) -> np.ndarray:
    opp = np.asarray(opp_implied, float)
    team = np.asarray(team_implied, float)
    return np.column_stack([np.ones_like(opp), opp, team])


def fit_dst_odds_model(session, model_dir: Path = MODELS_DIR) -> dict:
    """Fit DST points ~ opponent implied total + own implied total; write the params file."""
    df = training_frame(session)
    if len(df) < MIN_TRAINING_GAMES:
        raise RuntimeError(f"only {len(df)} DST games with lines; need {MIN_TRAINING_GAMES} "
                           "(run fetch_schedules and build_dst_game_table first)")
    X = _design(df.opp_implied, df.team_implied)
    y = df.dst_points.to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    params = {
        "fitted_on": datetime.now().strftime("%Y-%m-%d"),
        "n": int(len(df)),
        "seasons": sorted(int(s) for s in df.season.unique()),
        "coefficients": {"intercept": float(beta[0]), "opp_implied": float(beta[1]), "team_implied": float(beta[2])},
        "residual_std": float(resid.std(ddof=3)),
        "r2": float(1.0 - resid.var() / y.var()),
        "mean_actual": float(y.mean()),
        "pool": {"mu": np.round(fitted, 3).tolist(), "actual": y.tolist()},
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / PARAMS_FILE).write_text(json.dumps(params))
    _MODEL_CACHE[model_dir] = params
    return params


def load_dst_odds_model(model_dir: Path = MODELS_DIR) -> dict | None:
    model_dir = Path(model_dir)
    if model_dir not in _MODEL_CACHE:
        path = model_dir / PARAMS_FILE
        _MODEL_CACHE[model_dir] = json.loads(path.read_text()) if path.exists() else None
    return _MODEL_CACHE[model_dir]


def describe_dst_odds_model(params: dict) -> str:
    c = params["coefficients"]
    return (f"DST ~ {c['intercept']:.2f} + {c['opp_implied']:.3f}*opp_implied + {c['team_implied']:.3f}*team_implied"
            f"  (n={params['n']}, r2={params['r2']:.3f}, resid sd={params['residual_std']:.2f}, "
            f"seasons {params['seasons'][0]}-{params['seasons'][-1]})")


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


def project_dst_samples(session, team: str, opponent: str, rng: np.random.Generator,
                        n_samples: int = 10000, model_dir: Path = MODELS_DIR):
    """
    Samples of DraftKings DST points for `team` against `opponent` from the
    current lines, or None when the lines or the fitted model are missing.
    Second return value describes the inputs for logging.
    """
    params = load_dst_odds_model(model_dir)
    if params is None or not team or not opponent:
        return None
    lines = matchup_lines(session, team, opponent)
    if lines is None:
        return None
    total, home_margin, is_home = lines
    home_tt, away_tt = implied_totals(total, home_margin)
    team_tt, opp_tt = (home_tt, away_tt) if is_home else (away_tt, home_tt)
    c = params["coefficients"]
    mu = c["intercept"] + c["opp_implied"] * opp_tt + c["team_implied"] * team_tt

    pool_mu = np.asarray(params["pool"]["mu"], float)
    pool_actual = np.asarray(params["pool"]["actual"], float)
    nearest = np.argsort(np.abs(pool_mu - mu))[:POOL_SIZE]
    resid = pool_actual[nearest] - pool_mu[nearest]
    samples = mu + rng.choice(resid, size=n_samples, replace=True)
    samples = np.maximum(samples, DK_DST_FLOOR)
    info = {"total": total, "team_implied": round(team_tt, 2), "opp_implied": round(opp_tt, 2), "mu": round(float(mu), 2)}
    return samples, info
