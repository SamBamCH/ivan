from pathlib import Path
import joblib
import json
from typing import Optional, Dict, Tuple
from collections import defaultdict
from math import isfinite
import numpy as np
from scipy.stats import rv_continuous, rv_discrete
import scipy.stats as st
from sqlalchemy.orm import Session
from sqlalchemy import text, func, exists
from sqlalchemy.inspection import inspect

from ivan.sports.nfl.player_level_calculations import (
    position_configs,
    continuous_dists,
    fit_distribution_by_name,
    fit_best_distribution
)
from ivan.sports.nfl.stat_distribution_calculations import build_player_stat_overrides
from ivan.sports.nfl.models import NFLPlayer, NFLStat


# ------ NEW: DST helpers ------------------------------------------------------

DST_POS = {"DST", "D/ST", "DEF"}
KICKER_DEFAULT_SHAPE = 5.5
KICKER_DEFAULT_SCALE = 1.2
DEFAULT_ONE_SIDE_VIG = 0.05  # 7.5% haircut if the opposing side is missing
DEFAULT_MARKET_WEIGHTS_PATH = Path(r"D:\ivan\data\nfl\models\market_weights.json")
_MARKET_WEIGHTS_CACHE: dict | None = None

from sqlalchemy import and_, desc
from ivan.core.utils import normalize_name
from ivan.sports.nfl.models import NFLOdds

# ---- FPTS summary persistence helpers ---------------------------------------
import numpy as np
from scipy.stats import skew, kurtosis


def _load_market_weights(path: Path = DEFAULT_MARKET_WEIGHTS_PATH) -> dict:
    global _MARKET_WEIGHTS_CACHE
    if _MARKET_WEIGHTS_CACHE is None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                _MARKET_WEIGHTS_CACHE = json.load(f)
        except Exception:
            _MARKET_WEIGHTS_CACHE = {}
    return _MARKET_WEIGHTS_CACHE

def _norm_side(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"o", "over"}:  return "over"
    if s in {"u", "under"}: return "under"
    if s in {"y", "yes"}:   return "yes"
    if s in {"n", "no"}:    return "no"
    return s

def _devig_two_sided(p_a: float, p_b: float) -> tuple[float, float]:
    # Normalize so p_a + p_b = 1
    denom = max(p_a + p_b, 1e-12)
    return p_a/denom, p_b/denom

def _haircut_one_sided(p_raw: float, haircut: float = DEFAULT_ONE_SIDE_VIG) -> float:
    # Simple “fair” proxy if only one side is visible
    p = p_raw * (1.0 - haircut)
    return float(np.clip(p, 1e-6, 1 - 1e-6))

def _gather_quotes(session: Session, player_name: str, api_keys: tuple[str, ...]):
    """Get the latest odds rows for all keys; return list of NFLOdds rows."""
    return (session.query(NFLOdds)
            .filter(NFLOdds.player == player_name, NFLOdds.market.in_(api_keys))
            .order_by(desc(NFLOdds.last_update))
            .all())

def _pick_latest_per_book_point(rows):
    """
    Keep the latest row per (sportsbook, market, side, point).
    Returns dict[(book, point, side)]=(row, ts_rank)
    """
    out = {}
    rank = 0
    for r in rows:
        key = (r.sportsbook, float(r.point) if r.point is not None else 0.5, _norm_side(r.side))
        if key not in out:  # rows ordered desc already, first is latest
            out[key] = (r, rank)
            rank += 1
    return out

def _consensus_prob_for_side(
    session: Session,
    player_name: str,
    market_key: str,
    *, target_side: str,          # "over"/"under" or "yes"/"no"
    default_one_side_vig: float = DEFAULT_ONE_SIDE_VIG
) -> tuple[float | None, float | None]:
    """
    Book-weighted, de-vigged consensus for a side. Returns (consensus_point, p_fair).
    - Uses only books present; re-normalizes provided market weights.
    - For O/U: pairs Over/Under at the same book+point if possible (clean de-vig).
    - If only one side exists for that book+point, apply a haircut (7.5% default).
    - Points may differ across books; we compute a weight-average point as well.
    """
    weights_cfg = _load_market_weights()
    book_weights = (weights_cfg.get(market_key) or {}).copy()

    api_keys = (market_key,)  # allow passing an API key directly
    # If you passed an internal stat, expand to its API keys
    if market_key in _MARKET_KEYS:
        api_keys = _MARKET_KEYS[market_key]

    rows = _gather_quotes(session, player_name, api_keys)
    if not rows:
        return None, None

    latest = _pick_latest_per_book_point(rows)

    # group by book (choose the most recent point for that book)
    by_book: dict[str, dict] = defaultdict(dict)  # book -> {point: {"over":row, "under":row, "yes":row, "no":row, "rank":min}}
    for (book, point, side), (row, rnk) in latest.items():
        d = by_book[book].setdefault(point, {"rank": rnk})
        d[side] = row
        d["rank"] = min(d["rank"], rnk)

    # Per book, choose the freshest point (min rank), then compute fair p for target_side
    per_book = []
    for book, pts in by_book.items():
        # pick freshest point for that book
        point, bundle = min(pts.items(), key=lambda kv: kv[1]["rank"])
        sideA = "over" if target_side in {"over", "under"} else "yes"
        sideB = "under" if sideA == "over" else "no"
        rowA = bundle.get(sideA)
        rowB = bundle.get(sideB)

        if rowA is None and rowB is None:
            continue

        pA_raw = american_to_prob(float(rowA.price)) if rowA is not None else None
        pB_raw = american_to_prob(float(rowB.price)) if rowB is not None else None

        if pA_raw is not None and pB_raw is not None:
            pA_fair, _ = _devig_two_sided(pA_raw, pB_raw)
        elif pA_raw is not None:
            pA_fair = _haircut_one_sided(pA_raw, default_one_side_vig)
        elif pB_raw is not None:
            # want fair prob for target_side but only opposite side exists
            pB_fair = _haircut_one_sided(pB_raw, default_one_side_vig)
            pA_fair = 1.0 - pB_fair
        else:
            continue

        # weight for this book
        w = book_weights.get(book, None)
        per_book.append((book, float(point), float(pA_fair), float(w) if w is not None else None))

    if not per_book:
        return None, None

    # re-normalize weights to books we actually have; unknown books => equal small weight
    known = [(pt, p, w) for _, pt, p, w in per_book if (w is not None and isfinite(w) and w > 0)]
    unknown = [(pt, p) for _, pt, p, w in per_book if (w is None or not isfinite(w) or w <= 0)]

    if known:
        total_w = sum(w for _, _, w in known)
        known = [(pt, p, w/total_w) for pt, p, w in known]
    if unknown:
        # give equal mass among unknowns, scaled to (remaining) 0 if known else 1
        remaining = 0.0 if known else 1.0
        w_eq = remaining / len(unknown)
        known += [(pt, p, w_eq) for (pt, p) in unknown]

    # consensus
    point_cons = sum(pt * w for pt, _, w in known)
    p_cons = sum(p * w for _, p, w in known)
    return float(point_cons), float(np.clip(p_cons, 1e-6, 1 - 1e-6))

def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]

def _fpts_summary(samples: np.ndarray,
                  percentiles = (1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict:
    xs = _finite(samples)
    if xs.size == 0:
        return {"n": 0}
    s = {
        "n": int(xs.size),
        "mean": float(xs.mean()),
        "std": float(xs.std(ddof=0)),
        "min": float(xs.min()),
        "max": float(xs.max()),
        "skew": float(skew(xs, bias=True)) if xs.size > 2 else 0.0,
        "kurtosis": float(kurtosis(xs, fisher=True, bias=True)) if xs.size > 3 else 0.0,
    }
    q = np.percentile(xs, percentiles)
    for p, v in zip(percentiles, q):
        s[f"p{int(p):02d}"] = float(v)
    return s

def _save_fpts_summary_on_player(session, player, summary: dict, verbose: bool = False) -> None:
    """
    Save summary onto NFLPlayer. If a JSON column like fpts_summary or
    fpts_summary_json exists, store the whole dict there too.
    Unknown attrs are skipped silently.
    """
        # Figure out which columns exist on the mapped table
    mapper = inspect(NFLPlayer).mapper
    cols = {attr.key for attr in mapper.column_attrs}
    commit = True #can pull this out as arg if you need to. 

    # Also populate scalar columns if they exist
    mapping = {
        "fpts_n": "n",
        "fpts_mean": "mean",
        "fpts_std": "std",
        "fpts_min": "min",
        "fpts_max": "max",
        "fpts_skew": "skew",
        "fpts_kurtosis": "kurtosis",
        "fpts_p01": "p01",
        "fpts_p05": "p05",
        "fpts_p10": "p10",
        "fpts_p25": "p25",
        "fpts_p50": "p50",
        "fpts_p75": "p75",
        "fpts_p90": "p90",
        "fpts_p95": "p95",
        "fpts_p99": "p99",
    }
    # Build the update payload only for columns that actually exist
    updates = {}

    for col, key in mapping.items():
        if col in cols and key in summary:
            updates[col] = summary[key]

    if not updates:
        if verbose:
            print(f"[summary] No matching columns on NFLPlayer; nothing to persist for id={player.id}")
        return

    # Do a direct UPDATE to avoid detached/dirty state issues
    session.query(NFLPlayer).filter(NFLPlayer.id == player.id).update(
        updates, synchronize_session=False
    )
    if verbose:
        set_keys = ", ".join(f"{k}={updates[k]}" for k in sorted(updates.keys()))
        print(f"[summary] Updated NFLPlayer id={player.id}: {set_keys}")

    if commit:
        session.commit()

# --- TD odds → Poisson mean helpers ------------------------------------------
def _poisson_mu_for_threshold(point: float, side: str, p_fair: float) -> float:
    side = _norm_side(side)
    if side in {"yes", "over"}:
        k_min = int(np.floor(point)) + 1   # Over t => >= floor(t)+1
        return _invert_poisson_for_over(p_fair, k_min)
    else:
        k_max = int(np.floor(point))       # Under t => <= floor(t)
        return _invert_poisson_for_under(p_fair, k_max)


def _poisson_mu_from_prob_at_least_one(p: float) -> float:
    # P(X >= 1) = 1 - e^{-mu}  =>  mu = -ln(1-p)
    return float(-np.log(max(1.0 - p, 1e-9)))

def _invert_poisson_for_over(p_over: float, k_min: int) -> float:
    # find mu s.t. P[X >= k_min] = p_over
    lo, hi = 1e-9, 20.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        val = st.poisson.sf(k_min-1, mid)  # sf(k-1) = P[X >= k]
        if val > p_over: lo = mid
        else:            hi = mid
    return 0.5*(lo+hi)

def _invert_poisson_for_under(p_under: float, k_max: int) -> float:
    # find mu s.t. P[X <= k_max] = p_under
    lo, hi = 1e-9, 20.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        val = st.poisson.cdf(k_max, mid)
        if val < p_under: lo = mid
        else:             hi = mid
    return 0.5*(lo+hi)

def _player_has_any_odds(session: Session, player_name: str) -> bool:
    """Fast existence check: does ANY odds row exist for this player name?"""
    q = session.query(exists().where(NFLOdds.player == player_name))
    return bool(q.scalar())

def american_to_prob(american: float) -> float:
    a = float(american)
    return 100.0 / (a + 100.0) if a >= 0 else (-a) / ((-a) + 100.0)

# Replace your receiving_tds mapping (had _alternate only / anytime) with:
_MARKET_KEYS = {
    "passing_yards":     ("player_pass_yds", "player_pass_yds_alternate"),
    "rushing_yards":     ("player_rush_yds", "player_rush_yds_alternate"),
    "receiving_yards":   ("player_reception_yds", "player_reception_yds_alternate"),
    "receptions":        ("player_receptions", "player_receptions_alternate"),

    "passing_tds":       ("player_pass_tds", "player_pass_tds_alternate"),
    "rushing_tds":       ("player_rush_tds", "player_rush_tds_alternate"),
    "receiving_tds":     ("player_reception_tds", "player_reception_tds_alternate"),

    "interceptions":     ("player_pass_interceptions", "player_pass_interceptions_alternate"),
    "pass_attempts":     ("player_pass_attempts", "player_pass_attempts_alternate"),
    "pass_completions":  ("player_pass_completions", "player_pass_completions_alternate"),

    # Combined scrimmage TD markets (used as a single source of truth)
    "scrim_tds_combined": (
        "player_anytime_td",                 # Yes/No, effectively > 0.5
        "player_tds_over",                   # Over only (no Under side)
        "player_rush_reception_tds",         # O/U rush+rec TDs
        "player_pass_rush_reception_tds"     # O/U pass+rush+rec TDs (rarely used for non-QBs)
    )
}

# tolerant mapping from your internal stat keys to likely NFLStat column names
_STAT_COL_ALIASES = {
    "passing_yards":    ("passing_yards", "pass_yards", "pass_yds"),
    "passing_tds":      ("passing_tds", "pass_tds"),
    "interceptions":    ("interceptions", "pass_interceptions", "interceptions_thrown"),

    "rushing_yards":    ("rushing_yards", "rush_yards", "rush_yds"),
    "rushing_tds":      ("rushing_tds", "rush_tds"),

    "receiving_yards":  ("receiving_yards", "rec_yards", "rec_yds"),
    "receiving_tds":    ("receiving_tds", "rec_tds"),
    "receptions":       ("receptions", "rec"),
}

def _resolve_stat_column(stat: str):
    """Return the SQLA column on NFLStat that best matches `stat`, or None."""
    cand = _STAT_COL_ALIASES.get(stat, (stat,))
    for name in cand:
        col = getattr(NFLStat, name, None)
        if col is not None:
            return col
    return None

def _latest_season_for_player(session: Session, player_name: str) -> int | None:
    return session.query(func.max(NFLStat.season))\
                  .filter(NFLStat.player_display_name == player_name)\
                  .scalar()

def _season_stat_mean(session: Session, player_name: str, stat: str) -> float | None:
    """
    Per-game average of `stat` for the player's latest season.
    Returns None if the stat column doesn't exist or no rows found.
    """
    col = _resolve_stat_column(stat)
    if col is None:
        return None
    latest = _latest_season_for_player(session, player_name)
    if latest is None:
        return None
    q = session.query(func.avg(col))\
               .filter(NFLStat.player_display_name == player_name,
                       NFLStat.season == latest)
    try:
        val = q.scalar()
        if val is None:
            return None
        val = float(val)
        return val if np.isfinite(val) else None
    except Exception:
        return None
    
def _shift_loc_to_mean(dist_name: str, base_params, target_mean: float):
    """
    Adjust only `loc` so that E[X] = target_mean.
    For SciPy families, mean = loc + scale * g(shapes) (cont.) or loc + g(shapes) (disc.), so
    we can set new_loc = target_mean - current_mean + loc.
    """
    dist_fn = getattr(st, dist_name)
    shapes, loc, scale = _split_params(dist_fn, base_params)
    current = _mean_of(dist_name, base_params)
    if not np.isfinite(current):
        # if mean fails, do nothing
        return dist_name, tuple(base_params)
    delta = float(target_mean) - float(current)
    new_loc = loc + delta
    if scale is None:
        return dist_name, (shapes + (new_loc,))
    return dist_name, (shapes + (new_loc, scale))

def _latest_odds_by_name_and_stat(session: Session, player_name: str, stat: str, verbose: bool = False):
    """
    Return the most-recent NFLOdds row for (player_name, stat) using official API keys.
    - stat can be one of your internal names (e.g., 'rushing_yards') or a pass-through API key, e.g. 'player_anytime_td'.
    - Picks the latest row by NFLOdds.last_update.
    """
    keys = _MARKET_KEYS.get(stat, (stat,))  # allow direct API key calls
    if verbose:
        print(f'converted stat name to {keys} for searching odds database')
    return (
        session.query(NFLOdds)
        .filter(NFLOdds.player == player_name, NFLOdds.market.in_(keys))
        .order_by(desc(NFLOdds.last_update))
        .first()
    )

def _role_factor_from_calibrated(base_name, base_params, adj_name, adj_params, *, lo=0.2, hi=1.0) -> float:
    try:
        e_base = _mean_of(base_name, base_params)
        e_adj  = _mean_of(adj_name,  adj_params)
        if not np.isfinite(e_base) or e_base == 0:
            return 1.0
        return float(np.clip(e_adj / e_base, lo, hi))
    except Exception:
        return 1.0
    
def _is_count_dist(name: str) -> bool:
    return name in {"poisson", "nbinom"}

def _poisson_get_mu(params: tuple) -> float:
    # SciPy poisson params: (mu,) [loc implicit via _split_params]
    return float(params[0])

def _poisson_set_mu(mu: float) -> tuple:
    return (float(max(mu, 1e-9)),)

def _nbinom_get_mean(params: tuple) -> tuple[float, float, float]:
    # SciPy nbinom: (n, p), mean = n*(1-p)/p
    n, p = float(params[0]), float(params[1])
    m = n*(1.0 - p)/p
    return m, n, p

def _nbinom_set_mean_keep_p(m: float, p: float) -> tuple:
    # n = m*p/(1-p)
    n = float(max(m*p/max(1e-9, (1.0-p)), 1e-9))
    return (n, p)

def _calibrate_count_mean_from_ou(dist_name: str, base_params, point: float, side: str, p_fair: float) -> tuple[str, tuple]:
    side = _norm_side(side)
    if dist_name == "poisson":
        # solve for mu s.t. P(X > t) or P(X <= t) matches p_fair
        if side in {"over", "yes"}:
            k_min = int(np.floor(point)) + 1
            mu_new = _invert_poisson_for_over(p_fair, k_min)
        else:
            k_max = int(np.floor(point))
            mu_new = _invert_poisson_for_under(p_fair, k_max)
        return "poisson", _poisson_set_mu(mu_new)

    if dist_name == "nbinom":
        m0, _, p = _nbinom_get_mean(base_params)
        # binary search over mean m; keep p fixed, update n accordingly
        k_thr = int(np.floor(point))
        over = (side in {"over", "yes"})
        lo, hi = 1e-6, max(20.0, 5.0*m0 + 5.0)  # wide enough bracket
        for _ in range(60):
            mid = 0.5*(lo+hi)
            n_mid, p_mid = _nbinom_set_mean_keep_p(mid, p)
            if over:
                val = st.nbinom.sf(k_thr, n_mid, p_mid)  # P[X > k_thr]
                if val > p_fair: lo = mid
                else:            hi = mid
            else:
                val = st.nbinom.cdf(k_thr, n_mid, p_mid) # P[X <= k_thr]
                if val < p_fair: lo = mid
                else:            hi = mid
        m_new = 0.5*(lo+hi)
        return "nbinom", _nbinom_set_mean_keep_p(m_new, p)

    # Fallback: leave as-is (rare)
    return dist_name, tuple(base_params)
    
_MISSING_STAT_EXP = {
    # yards / receptions need strong shrink for fringe roles
    "receiving_yards": 0.9, "receptions": 0.3, "receiving_tds": 1.0,
    "rushing_yards": 0.8,  "rushing_tds": 1.0,
    "passing_yards": 0.0,  "passing_tds": 0.0,
    "interceptions": 0.0,
}

def _apply_role_to_params(dist_name: str, base_params, role: float, stat: str, verbose: bool = False):
    beta = _MISSING_STAT_EXP.get(stat, 0.9)
    rpow = float(np.clip(role, 1e-4, 1.0)) ** beta

    if _is_count_dist(dist_name):
        # scale the *mean* of count dists
        if dist_name == "poisson":
            mu = _poisson_get_mu(base_params)
            return "poisson", _poisson_set_mu(mu * rpow)
        if dist_name == "nbinom":
            m, _, p = _nbinom_get_mean(base_params)
            return "nbinom", _nbinom_set_mean_keep_p(m * rpow, p)
        return dist_name, tuple(base_params)

    # continuous dists: shrink scale ~ role, and shrink loc if nonzero
    dist_fn = getattr(st, dist_name)
    shapes, loc, scale = _split_params(dist_fn, base_params)
    new_loc   = loc * rpow
    new_scale = scale * rpow
    if verbose:
        print(f"[role] {stat}: r={role:.3f} β={beta:.2f} → loc {loc:.3g}->{new_loc:.3g}, scale {scale:.3g}->{new_scale:.3g}")
    return dist_name, (shapes + (new_loc, new_scale)) if scale is not None else (shapes + (new_loc,))

def _split_params(dist_fn, params):
    """
    Split saved params into (shapes, loc, scale_or_None) using SciPy's numargs:
      continuous: (*shapes, [loc], [scale]) -> loc default 0, scale default 1
      discrete  : (*shapes, [loc])          -> loc default 0, no scale
    """
    params = tuple(params)
    numargs = int(getattr(dist_fn, "numargs", 0))  # number of shape params
    shapes = tuple(params[:numargs])
    rest = params[numargs:]

    if isinstance(dist_fn, rv_continuous):
        # continuous allows loc + scale
        loc = float(rest[0]) if len(rest) >= 1 else 0.0
        scale = float(rest[1]) if len(rest) >= 2 else 1.0
        return shapes, loc, scale
    else:
        # discrete: ONLY loc after shapes
        loc = float(rest[0]) if len(rest) >= 1 else 0.0
        return shapes, loc, None
    
def _ppf(dist_name: str, params, u: np.ndarray) -> np.ndarray:
    """Inverse CDF with correct kwargs for cont. vs discrete."""
    dist_fn = getattr(st, dist_name)
    shapes, loc, scale = _split_params(dist_fn, params)
    if scale is None:  # discrete => no scale kw
        return dist_fn.ppf(u, *shapes, loc=loc).astype(float)
    return dist_fn.ppf(u, *shapes, loc=loc, scale=scale).astype(float)

def _mean_of(dist_name: str, params) -> float:
    """Mean via frozen RV constructed with the right kwargs."""
    dist_fn = getattr(st, dist_name)
    shapes, loc, scale = _split_params(dist_fn, params)
    rv = dist_fn(*shapes, loc=loc) if scale is None else dist_fn(*shapes, loc=loc, scale=scale)
    m = rv.mean()
    return float(m) if np.isfinite(m) else np.nan

def _role_factor_from_calibrated(base_name: str, base_params, adj_name: str, adj_params,
                                 *, lo: float = 0.02, hi: float = 0.8, verbose: bool = False) -> float:
    try:
        e_base = _mean_of(base_name, base_params)
        e_adj  = _mean_of(adj_name,  adj_params)
        if not np.isfinite(e_base) or e_base == 0:
            if verbose:
                print(f'set role_factor to 1 due to infinite/0 e_base')
            return 1.0
        role_factor = float(np.clip(e_adj / e_base, lo, hi))
        if verbose:
            print(f'role_factor: {role_factor}')
            
        return role_factor
    except Exception:
        return 1.0

def _calibrate_loc(
    dist_name: str,
    base_params,
    point: float,
    side: str,
    price: float | None = None,
    *,
    p_fair: float | None = None,
    verbose: bool = False
):
    """
    If p_fair is provided, use that directly; otherwise use 'price' to build p_fair.
    side in {'over','under','yes','no'}.
    """
    side_norm = _norm_side(side)
    if p_fair is None:
        if price is None:
            return dist_name, tuple(base_params)
        p = float(np.clip(american_to_prob(price), 1e-6, 1 - 1e-6))
        p_fair = p  # you'll usually pass de-vigged here from consensus

    target_cdf = (1.0 - p_fair) if side_norm in {"over","yes"} else p_fair
    if verbose:
        print(f"calibrate: point={point} side={side_norm} p_fair={p_fair} -> target_cdf={target_cdf}")

    dist_fn = getattr(st, dist_name)
    shapes, loc, scale = _split_params(dist_fn, base_params)

    try:
        q0 = dist_fn.ppf(target_cdf, *shapes, loc=0.0) if scale is None \
            else dist_fn.ppf(target_cdf, *shapes, loc=0.0, scale=scale)
    except TypeError:
        q0 = (dist_fn.ppf(target_cdf, *shapes, loc=loc) if scale is None
              else dist_fn.ppf(target_cdf, *shapes, loc=loc, scale=scale)) - loc

    if not np.isfinite(q0):
        return dist_name, tuple(base_params)

    new_loc = float(point) - float(q0)
    new_params = (shapes + (new_loc,)) if scale is None else (shapes + (new_loc, scale))
    return dist_name, new_params

def _fetch_dst_points(session: Session, team: str | None, opponent: str | None) -> np.ndarray:
    """
    Pull historical dst_points from nfl_dst_game.
    Preference:
      1) (team, opponent) exact matchup
      2) team-only
      3) league-wide
    """
    with session.bind.connect() as conn:
        if team and opponent:
            res = conn.execute(
                text("""SELECT dst_points FROM nfl_dst_game
                        WHERE team = :t AND opponent = :o AND dst_points IS NOT NULL"""),
                {"t": team, "o": opponent},
            ).fetchall()
            vals = np.array([r[0] for r in res], dtype=float)
            if vals.size >= 5:
                return vals

        if team:
            res = conn.execute(
                text("""SELECT dst_points FROM nfl_dst_game
                        WHERE team = :t AND dst_points IS NOT NULL"""),
                {"t": team},
            ).fetchall()
            vals = np.array([r[0] for r in res], dtype=float)
            if vals.size >= 5:
                return vals

        res = conn.execute(
            text("SELECT dst_points FROM nfl_dst_game WHERE dst_points IS NOT NULL")
        ).fetchall()
        vals = np.array([r[0] for r in res], dtype=float)
        return vals


def _sample_dst_fpts(session: Session, team: str | None, opponent: str | None, n_samples: int) -> np.ndarray:
    """
    Empirical bootstrap of DST fpts from nfl_dst_game with a safe fallback.
    """
    vals = _fetch_dst_points(session, team, opponent)
    if vals.size > 0:
        # optional tiny jitter helps continuous fitting stay well-behaved
        samples = np.random.choice(vals, size=n_samples, replace=True).astype(float)
        return samples

    # fallback if the table is empty for a new DB
    rng = np.random.default_rng()
    fpts = rng.normal(loc=5.0, scale=4.0, size=n_samples)
    return np.clip(fpts, -6.0, None)


# -----------------------------------------------------------------------------

def load_models(model_dir: Path):
    """
    Load base marginal fits and stacked position copulas from disk.
    Returns:
      - marginals: {pos: {stat: {'dist','params',...}}}
      - stacked_copulas: {pos: corr_matrix}
    """
    marginals = joblib.load(model_dir / 'marginals.pkl')
    stacked_copulas = {}
    stacked_dir = model_dir / 'stacked_player_copulas'
    if stacked_dir.exists():
        for path in stacked_dir.glob('*_stacked_copula.pkl'):
            pos = path.stem.split('_')[0]
            stacked_copulas[pos] = joblib.load(path)
    return marginals, stacked_copulas


def simulate_player_fpts(
    session: Session,
    player_id: int,
    marginals: dict,
    stacked_copulas: dict,
    n_samples: int = 10000,
    verbose: bool = False,
) -> np.ndarray:
    player = session.get(NFLPlayer, player_id)
    if verbose:
        print(f'matched {player_id} to DB player {player.name}')
    raw_pos = player.position
    pos = 'DST' if raw_pos in DST_POS else raw_pos

    if verbose:
        print(f'position: {pos}')

    # Keep K/DST behavior
    if pos == 'K':
        return st.gamma.rvs(KICKER_DEFAULT_SHAPE, scale=KICKER_DEFAULT_SCALE, size=n_samples).astype(float)
    if pos == 'DST':
        return _sample_dst_fpts(session, getattr(player, "team", None), getattr(player, "opponent", None), n_samples)

    stats = position_configs[pos]['stats']
    corr = stacked_copulas.get(pos, np.eye(len(stats)))

    Z = np.random.multivariate_normal(np.zeros(len(stats)), corr, size=n_samples)
    U = st.norm.cdf(Z)
    X = np.zeros_like(U)

    # --- 0) Build ONE combined scrimmage TD signal (rush+rec), then split; no double counting
    td_overrides: dict[str, tuple[str, tuple]] = {}
    base_mu_rush = base_mu_rec = 0.0
    if "rushing_tds" in stats:
        info = marginals[pos]["rushing_tds"]; 
        base_mu_rush = max(_mean_of(info["dist"], info["params"]), 0.0) if info["dist"] else 0.0
    if "receiving_tds" in stats:
        info = marginals[pos]["receiving_tds"]; 
        base_mu_rec  = max(_mean_of(info["dist"], info["params"]), 0.0) if info["dist"] else 0.0

    mu_total = None
    scrim_point, scrim_p = _consensus_prob_for_side(session, player.name, "scrim_tds_combined", target_side="over")
    if scrim_p is not None:
        if abs((scrim_point or 0.5) - 0.5) < 1e-6:
            mu_total = _poisson_mu_from_prob_at_least_one(scrim_p)     # P(TD>=1)
        else:
            mu_total = _poisson_mu_for_threshold(scrim_point, "over", scrim_p)

    if mu_total is not None and ("rushing_tds" in stats or "receiving_tds" in stats):
        denom = base_mu_rush + base_mu_rec
        if denom <= 1e-12:
            if pos in ("WR","TE"): alpha_rush = 0.02
            elif pos in ("RB",):   alpha_rush = 0.75
            else:                  alpha_rush = 0.80
        else:
            alpha_rush = base_mu_rush / denom

        mu_rush = mu_total * alpha_rush if "rushing_tds"    in stats else 0.0
        mu_rec  = mu_total * (1.0 - alpha_rush) if "receiving_tds" in stats else 0.0

        # If a per-stat O/U exists at 0.5, pin that mean (de-vigged)
        if "rushing_tds" in stats:
            pt_r, p_r = _consensus_prob_for_side(session, player.name, "player_rush_tds", target_side="over")
            if p_r is not None and abs((pt_r or 0.5) - 0.5) < 1e-6:
                mu_rush = _poisson_mu_from_prob_at_least_one(p_r)
        if "receiving_tds" in stats:
            pt_c, p_c = _consensus_prob_for_side(session, player.name, "player_reception_tds", target_side="over")
            if p_c is not None and abs((pt_c or 0.5) - 0.5) < 1e-6:
                mu_rec = _poisson_mu_from_prob_at_least_one(p_c)

        # Renormalize back to the combined mean
        total = mu_rush + mu_rec
        if total > 1e-12 and "rushing_tds" in stats and "receiving_tds" in stats:
            s = mu_total / total
            mu_rush *= s; mu_rec *= s
        elif "rushing_tds" in stats and "receiving_tds" not in stats:
            mu_rush = mu_total
        elif "receiving_tds" in stats and "rushing_tds" not in stats:
            mu_rec = mu_total

        if "rushing_tds" in stats:
            td_overrides["rushing_tds"] = ("poisson", (max(mu_rush, 0.0),))
        if "receiving_tds" in stats:
            td_overrides["receiving_tds"] = ("poisson", (max(mu_rec, 0.0),))

    # --- 1) Optional role factor (unchanged: probe some markets)
    role_factor: float | None = None
    for probe in ["player_anytime_td", "player_tds_over", "player_pass_rush_reception_tds",
                  "player_rush_reception_tds", "player_rush_tds", "player_reception_tds"]:
        odds_row = _latest_odds_by_name_and_stat(session, player.name, probe)
        if not odds_row:
            continue
        probe_stat = None
        if probe in {"player_anytime_td","player_tds_over","player_pass_rush_reception_tds","player_rush_reception_tds"}:
            probe_stat = "receiving_tds" if pos in ("WR","TE") else "rushing_tds"
        elif probe == "player_rush_tds": probe_stat = "rushing_tds"
        elif probe == "player_reception_tds": probe_stat = "receiving_tds"

        if probe_stat and probe_stat in position_configs.get(pos, {}).get("stats", []):
            base_info = marginals[pos][probe_stat]
            dn, bp = base_info["dist"], base_info["params"]
            pt = float(odds_row.point) if odds_row.point is not None else 0.5
            dn2, ap = _calibrate_loc(dn, bp, pt, str(odds_row.side), float(odds_row.price), verbose=verbose)
            role_factor = _role_factor_from_calibrated(dn, bp, dn2, ap, lo=0.10, hi=0.8, verbose=verbose)
            break

    # --- 2) Calibrate every required stat from de-vigged, weighted consensus
    for j, stat in enumerate(stats):
        base_info = marginals[pos][stat]
        dist_name, params = base_info["dist"], base_info["params"]

        # Use the combined TD overrides if present
        if stat in td_overrides:
            dist_name, params = td_overrides[stat]
            X[:, j] = _ppf(dist_name, params, U[:, j])
            continue

        pt_cons, p_cons = _consensus_prob_for_side(session, player.name, stat, target_side="over")
        if (pt_cons is not None) and (p_cons is not None):
            if _is_count_dist(dist_name):
                dist_name, params = _calibrate_count_mean_from_ou(dist_name, params, float(pt_cons), "over", float(p_cons))
            else:
                dist_name, params = _calibrate_loc(dist_name, params, point=float(pt_cons), side="over", p_fair=float(p_cons), verbose=verbose)
            if role_factor is None:
                role_factor = _role_factor_from_calibrated(base_info["dist"], base_info["params"], dist_name, params, lo=0.10, hi=0.8, verbose=verbose)
        else:
            # season anchor → role shrink (now stronger and hits means for counts)
            mu_season = _season_stat_mean(session, player.name, stat)
            if mu_season is not None and not _is_count_dist(dist_name):
                dist_name, params = _shift_loc_to_mean(dist_name, params, mu_season)
            if role_factor is not None:
                dist_name, params = _apply_role_to_params(dist_name, params, role_factor, stat, verbose=verbose)

        X[:, j] = _ppf(dist_name, params, U[:, j])

    # === Combine to FPTS ===
    fpts_weights = {
        'passing_yards': 0.04, 'passing_tds': 4, 'interceptions': -2,
        'rushing_yards': 0.1,  'rushing_tds': 6,
        'receptions': 1,       'receiving_yards': 0.1, 'receiving_tds': 6,
    }
    weights = np.array([fpts_weights.get(stat, 0.0) for stat in stats], dtype=float)
    return X.dot(weights).astype(float)


def fit_and_store_fpts_distribution(
    session: Session,
    player_id: int,
    fpts_samples: np.ndarray,
    *,
    save_summary: bool = True,
    verbose: bool = False
) -> None:
    """
    Fit a best marginal to the simulated fpts samples and persist it.
    Optionally store a compact summary (mean/std/percentiles) on NFLPlayer.
    """
    # Guard: ensure we have usable data
    xs = np.asarray(fpts_samples, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        # Nothing to fit; you can choose to clear fields or skip
        player = session.get(NFLPlayer, player_id)
        if player is not None and hasattr(player, "fpts_dist_name"):
            player.fpts_dist_name = None
            player.fpts_dist_params = None
            if save_summary:
                _save_fpts_summary_on_player(session, player, {"n": 0})
            session.commit()
        return

    name, params, _ = fit_best_distribution(
        xs,
        continuous_dists,
        bins=50,
        plot=False,
        plot_path=None
    )
    player = session.get(NFLPlayer, player_id)
    player.fpts_dist_name = name
    player.fpts_dist_params = params

    if save_summary:
        if verbose:
            print('saving summary...')
        summary = _fpts_summary(xs)
        if verbose:
            print(f'fpts summary: {summary}')
        _save_fpts_summary_on_player(session, player, summary, verbose=verbose)

        if verbose:
            print('-' * 200)

    session.commit()


def generate_and_fit_all(
    session: Session,
    model_dir: Path,
    n_samples: int = 10000,
    verbose: bool = False,
    delete_players_without_odds: bool = True,
    save_fpts_sample: bool = True
) -> None:
    marginals, stacked_copulas = load_models(model_dir)

    deleted = 0  # optional counter

    for player in session.query(NFLPlayer).all():
        pos_upper = (player.position or "").upper()
        keep_anyway = pos_upper in {"K", "DST", "D/ST", "DEF"}

        if delete_players_without_odds and not keep_anyway and not _player_has_any_odds(session, player.name):
            if verbose:
                print(f"[NO ODDS] Deleting NFLPlayer id={player.id} name={player.name}")
            session.delete(player)
            session.commit()
            deleted += 1
            continue
        # -------------------------------------

        fpts = simulate_player_fpts(
            session,
            player.id,
            marginals,
            stacked_copulas,
            n_samples,
            verbose=verbose,
        )
        if verbose:
            print(f'FPTS FOR ID: {player.id} :: {fpts}')
        fit_and_store_fpts_distribution(session, player.id, fpts, save_summary=save_fpts_sample, verbose=verbose)

    if verbose and delete_players_without_odds:
        print(f"[SUMMARY] Deleted {deleted} players with no odds.")
