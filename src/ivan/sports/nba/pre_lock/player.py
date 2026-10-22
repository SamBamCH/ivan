"""
Player fantasy-point samples for lineup generation and contest simulation.

generate_player_samples() draws one correlated universe per game (see
projection/correlation.py): every player on the slate with a stored marginal
is sampled through his own quantiles on a Gaussian copula with the role-pair
correlation; players without quantiles are Normal(fpts_mean, fpts_std) on the
same copula; locked players from ProjectionSnapshot are independent Normals.
"""
from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Dict, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ivan.core.utils import normalize_name
from ivan.sports.nba.models import NBAPlayer, ProjectionSnapshot
from ivan.sports.nba.projection.correlation import (load_league, PairHistory, build_game_correlation,
                                                    draw_from_marginals)
from ivan.sports.nba.projection.fpts import fpts_from_minutes, load_calibration
from ivan.sports.nba.projection.team_minutes import load_params as load_team_params, sample_team_minutes


def minutes_model_from_config() -> str:
    try:
        from ivan.core.config import load_config
        return str(((load_config("nba").get("projection") or {}).get("minutes_model")) or "independent")
    except Exception:   # noqa: BLE001
        return "independent"


def sample_game_team_model(players: list, n_samples: int, rng: np.random.Generator, randomness: float,
                           spreads: Dict[str, float | None], params: dict, dispersion: dict) -> Dict[int, np.ndarray]:
    """
    One game under the team model: a shared margin and overtime draw, each team's
    minutes drawn jointly (team_minutes.sample_team_minutes with the stored roles,
    scratch probabilities and allocations), then every player's fantasy points
    from his stored rates at those minutes. Players without stored rates fall
    back to a normal on their stored mean and sd.
    """
    out: Dict[int, np.ndarray] = {}
    teams = sorted({p.team for p in players})
    if not teams:
        return out
    home = teams[0]
    margin = rng.normal(-(spreads.get(home) or 0.0), params["margin_sd"], n_samples)
    overtime = (rng.random(n_samples) < params["p_ot"]).astype(int)
    for team in teams:
        ps = [p for p in players if p.team == team]
        ok = [p for p in ps if isinstance(p.rates, dict) and p.minutes_role and (p.minutes_proj or 0) > 0]
        if ok:
            mu = [float(p.minutes_proj) for p in ok]
            with_grid = [p for p in ok if isinstance(p.rotation_cells, list) and len(p.rotation_cells) == 48]
            if len(with_grid) >= 8 and sum(sum(p.rotation_cells) for p in with_grid) >= 200:
                import pandas as pd
                from ivan.sports.nba.projection.rotation import sample_worlds
                grid = pd.DataFrame([[bool(x) for x in p.rotation_cells] for p in with_grid], index=[normalize_name(p.name) for p in with_grid], columns=list(range(48)))
                sp = {normalize_name(p.name): float(p.boundary_sd) for p in with_grid if p.boundary_sd}
                d = sample_worlds(grid, sp, n_samples, rng, spread=spreads.get(team), p_out={normalize_name(p.name): float(p.p_out or 0.0) for p in with_grid},
                                  margin=margin if team == home else -margin, overtime=overtime, margin_sd=params["margin_sd"], p_ot=params["p_ot"])
                col = {nm: j for j, nm in enumerate(d["names"])}
                M = np.zeros((n_samples, len(ok)))
                for j, p in enumerate(ok):
                    nm = normalize_name(p.name)
                    if nm in col:
                        mu = float(d["minutes"][:, col[nm]].mean())
                        M[:, j] = d["minutes"][:, col[nm]] * (float(p.minutes_proj) / mu if mu > 0 else 1.0)
                    else:
                        M[:, j] = float(p.minutes_proj)
                class _D:
                    minutes = M
                draw = _D()
            else:
                draw = sample_team_minutes(mu, [p.minutes_role for p in ok], n_samples, rng,
                                           spread=spreads.get(team), p_out=[float(p.p_out or 0.0) for p in ok], params=params,
                                           margin=margin if team == home else -margin, overtime=overtime)
            for j, p in enumerate(ok):
                fp = fpts_from_minutes(draw.minutes[:, j], p.rates, dispersion, rng)
                if randomness < 1.0:
                    fp = float(p.fpts_mean) + randomness * (fp - float(p.fpts_mean))
                out[int(p.id)] = np.clip(fp, 0.0, None)
        for p in ps:
            if int(p.id) not in out:
                x = rng.normal(float(p.fpts_mean), float(p.fpts_std) * randomness, n_samples)
                out[int(p.id)] = np.clip(x, 0.0, None)
    return out


def generate_player_samples(
    session: Session,
    n_samples: int = 1000,
    randomness: float = 1.0,
    seed: Optional[int] = None,
    snapshot_date: Optional[date] = None,
    include_active_players: bool = True,
    name_to_id: Optional[Dict[str, int]] = None,
    correlated: bool = True,
    minutes_model: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """
    Returns player_id -> np.ndarray(n_samples,) of DK fantasy points.

    randomness scales the spread around the mean (0 = deterministic, 1 = full).
    correlated=False draws every player independently (the old behaviour).
    minutes_model "team" (default: config projection.minutes_model) draws each
    game's margin and overtime once and both teams' minutes jointly
    (sample_game_team_model); "independent" is the Gaussian copula on the
    stored marginals.
    """
    rng = np.random.default_rng(seed)
    randomness = float(np.clip(randomness, 0.0, 1.0))
    samples: Dict[int, np.ndarray] = {}
    minutes_model = minutes_model or minutes_model_from_config()

    if include_active_players:
        players = session.execute(select(NBAPlayer).where(
            NBAPlayer.fpts_mean.isnot(None), NBAPlayer.fpts_std.isnot(None))).scalars().all()
        league = load_league() if correlated else {}
        pairs = PairHistory.load() if correlated else None
        # one universe per game: group by the unordered (team, opponent) pair
        games: Dict[tuple, list] = {}
        for p in players:
            key = tuple(sorted([p.team or "", p.opponent or ""])) if correlated else (p.id,)
            games.setdefault(key, []).append(p)
        if minutes_model in ("team", "rotation"):
            from ivan.sports.nba.projection.lines import GameLines
            try:
                lines = GameLines.load(session)
            except Exception:   # noqa: BLE001
                lines = GameLines(__import__("pandas").DataFrame())
            tp = load_team_params(); disp = load_calibration()["dispersion"]
            for key, ps in games.items():
                spreads = {t: lines.team_spread(t) for t in {p.team for p in ps}}
                samples.update(sample_game_team_model(ps, n_samples, rng, randomness, spreads, tp, disp))
            games = {}
        for key, ps in games.items():
            names = [normalize_name(p.name) for p in ps]
            teams = [p.team for p in ps]
            mins = [float(p.minutes_proj) if p.minutes_proj else (float(p.fpts_mean) / 0.95) for p in ps]
            means = [float(p.fpts_mean) for p in ps]
            sds = [float(p.fpts_std) for p in ps]
            quants = [p.fpts_quantiles if isinstance(p.fpts_quantiles, list) else None for p in ps]
            R = build_game_correlation(names, teams, mins, league, pairs) if correlated and len(ps) > 1 else np.eye(len(ps))
            X = draw_from_marginals(R, quants, means, sds, n_samples, rng, randomness)
            for j, p in enumerate(ps):
                samples[int(p.id)] = X[:, j]

    if snapshot_date is not None:
        day_start = datetime(snapshot_date.year, snapshot_date.month, snapshot_date.day)
        snaps = (session.query(ProjectionSnapshot)
                 .filter(ProjectionSnapshot.snapshot_date >= day_start,
                         ProjectionSnapshot.snapshot_date < day_start + timedelta(days=1),
                         ProjectionSnapshot.fpts_mean.isnot(None), ProjectionSnapshot.fpts_std.isnot(None)).all())
        for snap in snaps:
            pid = getattr(snap, "id", None)
            if pid is None or int(pid) in samples:
                continue
            x = rng.normal(float(snap.fpts_mean), float(snap.fpts_std) * randomness, n_samples)
            samples[int(pid)] = np.clip(x, 0, None)
    return samples
