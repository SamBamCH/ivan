# odds_calibration.py (or tuck these in your existing fpts_distribution_calculations.py)

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import scipy.stats as st
from sqlalchemy import select, func
from ivan.sports.nfl.models import NFLOdds, NFLPlayer
from ivan.core.utils import normalize_name

# ---------- Odds helpers ----------
def american_to_prob(american: int | float) -> float:
    """Raw implied prob from American odds (no vig removal)."""
    a = float(american)
    if a >= 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)

def devig_two_way(p_over: float, p_under: float) -> Tuple[float, float]:
    """Normalize Over/Under so they sum to 1 (simple two-way devig)."""
    total = p_over + p_under
    if total <= 0:
        return p_over, p_under
    return p_over / total, p_under / total

def normalize_market_to_stat(market: str) -> Optional[str]:
    """
    Map an odds 'market' to our model's stat key.
    Expand/adjust as needed to match your NFLOdds.market values.
    """
    m = market.strip().lower()
    if "passing" in m and "yard" in m:     return "passing_yards"
    if "rushing" in m and "yard" in m:     return "rushing_yards"
    if "receiving" in m and "yard" in m:   return "receiving_yards"
    if "receptions" in m and "yard" not in m: return "receptions"
    if "passing" in m and ("td" in m or "touchdown" in m): return "passing_tds"
    if "rushing" in m and ("td" in m or "touchdown" in m): return "rushing_tds"
    if "receiving" in m and ("td" in m or "touchdown" in m): return "receiving_tds"
    return None

# ---------- Latest odds snapshot ----------
def fetch_latest_sidepoint_by_name(session) -> list[dict]:
    """
    Latest row per (player_name, market, sportsbook, side, point) by created_at DESC.
    Returns dicts with keys: player_name, market, sportsbook, side, point, price, created_at
    """
    subq = (
        select(
            NFLOdds.player.label("player_name"),
            NFLOdds.market.label("market"),
            NFLOdds.sportsbook.label("sportsbook"),
            NFLOdds.side.label("side"),
            NFLOdds.point.label("point"),
            NFLOdds.price.label("price"),
            NFLOdds.last_update.label("last_update"),
            func.row_number().over(
                partition_by=(NFLOdds.player, NFLOdds.market, NFLOdds.sportsbook, NFLOdds.side, NFLOdds.point),
                order_by=NFLOdds.last_update.desc(),
            ).label("rn"),
        )
        .subquery()
    )
    rows = session.execute(select(subq).where(subq.c.rn == 1)).mappings().all()
    return [dict(r) for r in rows]

def _split_params(dist_fn, params: Iterable[float]) -> Tuple[Tuple[float, ...], float, float]:
    params = tuple(params)
    if len(params) < 2:
        # If someone stored only shape or nothing, fabricate loc/scale defaults
        return tuple(params), 0.0, 1.0
    shapes = params[:-2]
    loc, scale = params[-2], params[-1]
    return shapes, float(loc), float(scale)

def calibrate_loc_to_point(
    dist_name: str,
    base_params: Iterable[float],
    point: float,
    prob: float,
    side: str,  # "over" or "under"
) -> Tuple[str, Tuple[float, ...]]:
    """
    Return (dist_name, new_params) where new_params = (*shapes, new_loc, scale)
    such that P(X > point)=prob (side=='over') or P(X < point)=prob (side=='under').
    """
    prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    side_norm = side.strip().lower()
    target_cdf = 1.0 - prob if side_norm == "over" else prob

    dist_fn = getattr(st, dist_name)
    shapes, _loc, scale = _split_params(dist_fn, base_params)

    # standard-LOC quantile for the target CDF (loc=0 for translation)
    q0 = dist_fn.ppf(target_cdf, *shapes, loc=0.0, scale=scale)
    if not np.isfinite(q0):
        # fallback: use current loc as base, compute delta against that
        q0 = dist_fn.ppf(target_cdf, *shapes, loc=_loc, scale=scale) - _loc
        if not np.isfinite(q0):
            # give up; return base unchanged
            return dist_name, tuple(base_params)

    new_loc = float(point) - float(q0)
    new_params = tuple(shapes) + (new_loc, scale)
    return dist_name, new_params

def build_player_stat_overrides(session, marginals: dict) -> Dict[int, Dict[str, Tuple[str, Tuple[float, ...]]]]:
    """
    Use latest odds keyed by (player NAME, market, sportsbook, side, point) to
    adjust each player's single-stat marginal so it passes through the market line.
    Returns overrides keyed by player_id (resolved from name).
    """
    latest = fetch_latest_sidepoint_by_name(session)  # uses NFLOdds.player (string)

    # 1) Build a name -> {id, position} map (normalized names)
    name_to_player: Dict[str, dict] = {}
    dup_names: set[str] = set()
    for row in session.query(NFLPlayer.id, NFLPlayer.name, NFLPlayer.position).all():
        pid, nm, pos = row
        key = normalize_name(nm or "")
        if key in name_to_player:
            dup_names.add(key)           # ambiguous; we'll skip these safely
        else:
            name_to_player[key] = {"id": pid, "position": pos}

    # 2) Group latest odds by (name, market, sportsbook, point)
    grouped = defaultdict(dict)
    for r in latest:
        name_key = normalize_name(r["player_name"] or "")
        group_key = (name_key, r["market"], r["sportsbook"], r["point"])
        side = str(r["side"]).strip().lower()
        grouped[group_key].setdefault("meta", r)
        grouped[group_key][side] = r

    overrides: Dict[int, Dict[str, Tuple[str, Tuple[float, ...]]]] = {}
    stamp: Dict[Tuple[int, str], int] = defaultdict(lambda: -1)

    for (name_key, market, book, point), sides in grouped.items():
        # resolve name -> player
        if not name_key or name_key not in name_to_player or name_key in dup_names:
            # Unknown or ambiguous name; skip to avoid wrong matches
            continue
        pid  = name_to_player[name_key]["id"]
        pos  = name_to_player[name_key]["position"]

        stat = normalize_market_to_stat(market or "")
        if not stat or pos not in marginals or stat not in marginals[pos]:
            continue

        # choose probabilities (prefer using both sides @ same point to devig)
        p_over = p_under = None
        if "over" in sides:
            p_over = american_to_prob(sides["over"]["price"])
        if "under" in sides:
            p_under = american_to_prob(sides["under"]["price"])

        side_to_use, implied = None, None
        if p_over is not None and p_under is not None:
            p_over, p_under = devig_two_way(p_over, p_under)
            side_to_use, implied = "over", p_over
        elif p_over is not None:
            side_to_use, implied = "over", p_over
        elif p_under is not None:
            side_to_use, implied = "under", p_under
        else:
            continue

        base = marginals[pos][stat]             # {'dist': name, 'params': tuple, ...}
        dist_name = base["dist"]
        params    = base["params"]

        dist_name2, params2 = calibrate_loc_to_point(
            dist_name=dist_name,
            base_params=params,
            point=float(point),
            prob=float(implied),
            side=side_to_use,
        )

        # keep most recent per (player_id, stat)
        ts = sides.get(side_to_use, sides.get("meta")).get("created_at")
        ts_ord = int(ts.timestamp()) if hasattr(ts, "timestamp") else 0

        if (pid, stat) not in stamp or stamp[(pid, stat)] <= ts_ord:
            overrides.setdefault(pid, {})[stat] = (dist_name2, params2)
            stamp[(pid, stat)] = ts_ord

    return overrides
