from pathlib import Path
import json
from typing import Optional, Dict, Tuple
from collections import defaultdict
from math import isfinite
import numpy as np
from scipy.stats import rv_continuous, rv_discrete
import scipy.stats as st
from sqlalchemy.orm import Session
from sqlalchemy import text, func, exists, and_, desc
from sqlalchemy.inspection import inspect
import joblib

# ivan imports
from ivan.sports.nfl.player_level_calculations import (
    position_configs,
    continuous_dists,
    fit_distribution_by_name,
    fit_best_distribution,  # imported (we fit FPTS as normal to allow negatives)
)
from ivan.sports.nfl.stat_distribution_calculations import build_player_stat_overrides
from ivan.sports.nfl.models import NFLPlayer, NFLStat, NFLOdds, NFLGameOdds
from ivan.core.utils import normalize_name

# =====================================================================================
#                                   CONFIG / CONSTANTS
# =====================================================================================

# K / DST handling
DST_POS = {"DST", "D/ST", "DEF"}
KICKER_DEFAULT_SHAPE = 8.3
KICKER_DEFAULT_SCALE = 1.2

# Odds handling
DEFAULT_ONE_SIDE_VIG = 0.15  # 5% haircut if the opposing side is missing
DEFAULT_MARKET_WEIGHTS_PATH = Path(r"D:\ivan\data\nfl\models\market_weights.json")
_MARKET_WEIGHTS_CACHE: dict | None = None

# Simple combined scrimmage TD handling
SIMPLE_MODE = True  # simplified per-stat + scrimmage TD approach
SIMPLE_TD_ALPHA = {'RB': 0.85, 'WR': 0.05, 'TE': 0.08}
def _simple_td_alpha(pos: str) -> float:
    return float(np.clip(SIMPLE_TD_ALPHA.get(pos, 0.0), 0.0, 1.0))

# Role mapping from anytime TD
ROLE_MIN = 0.01
ROLE_MAX = 0.50
ROLE_BETA = 1.25

BUCKET_STEP = 0.5   # most props are on .5 increments; change if needed

def _bucket_line(point: float | None, step: float = BUCKET_STEP) -> float | None:
    """Snap a line to the nearest step (default 0.5) so 1.5 and 2.5 never mix."""
    if point is None:
        return None
    return round(float(point) / step) * step

def _choose_td_kernel(p_any: float, role: float) -> str:
    """
    Decide how to sample scrimmage TD count K.
    - If low role or modest anytime: 'bern' (0/1 only)
    - Else: 'pois' (allow multi-TD tails)
    """
    if (role is None) or (p_any is None):
        return "pois"  # fallback
    if (role < 0.30) or (p_any < 0.33):
        return "bern"
    return "pois"

# Mapping from internal stat keys to sportsbook API market keys
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

    # Combined scrimmage TD markets (our single source of truth for TD usage)
    "scrim_tds_combined": (
        "player_anytime_td",                 # Yes/No, effectively > 0.5
        "player_tds_over",                   # Over only (no Under side)
        "player_rush_reception_tds",         # O/U rush+rec TDs
        "player_pass_rush_reception_tds"     # O/U pass+rush+rec TDs (rare for non-QBs)
    ),
}

# tolerant mapping from internal stat keys to NFLStat columns
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

# How strongly to shrink when we lack stat-level odds (per-stat exponent)
_MISSING_STAT_EXP = {
    "receiving_yards": 0.9, "receptions": 0.3, "receiving_tds": 1.0,
    "rushing_yards": 0.8,  "rushing_tds": 1.0,
    "passing_yards": 0.0,  "passing_tds": 0.0,
    "interceptions": 0.0,
}

# FPTS weights
FPTS_WEIGHTS = {
    'passing_yards': 0.04, 'passing_tds': 4, 'interceptions': -2,
    'rushing_yards': 0.1,  'rushing_tds': 6,
    'receptions': 1,       'receiving_yards': 0.1, 'receiving_tds': 6,
}

# =====================================================================================
#                                        UTILS
# =====================================================================================

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

def american_to_prob(american: float) -> float:
    a = float(american)
    return 100.0 / (a + 100.0) if a >= 0 else (-a) / ((-a) + 100.0)

def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]

# =====================================================================================
#                             MARKET → CONSENSUS FAIR PROB
# =====================================================================================

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
    default_one_side_vig: float = DEFAULT_ONE_SIDE_VIG,
    verbose: bool = False
) -> tuple[float | None, float | None]:
    """
    Book-weighted, de-vigged consensus for a side **at a single line**.
    We DO NOT average across different lines. Steps:
      1) fetch latest quotes -> latest per (book, point, side)
      2) per book, pick freshest point (could differ per book)
      3) group by bucketed point (e.g., 1.5, 2.5, ...)
      4) within each line cluster, pair sides to de-vig and weight by book weights
      5) pick the best cluster (highest total weight; tie -> most books; tie -> freshest)
    Returns: (selected_line_point, p_fair_at_that_line)
    """
    weights_cfg = _load_market_weights()
    book_weights = (weights_cfg.get(market_key) or {}).copy()

    api_keys = (market_key,)
    if market_key in _MARKET_KEYS:
        api_keys = _MARKET_KEYS[market_key]

    rows = _gather_quotes(session, player_name, api_keys)
    if not rows:
        return None, None

    latest = _pick_latest_per_book_point(rows)  # latest per (book,point,side)

    # Build {book: {point: bundle}} where bundle holds latest side rows and a freshness rank
    by_book: dict[str, dict] = defaultdict(dict)
    for (book, point, side), (row, rnk) in latest.items():
        d = by_book[book].setdefault(point, {"rank": rnk})
        d[side] = row
        d["rank"] = min(d["rank"], rnk)

    # For each book, pick its freshest point, then cluster by bucketed point
    # cluster_map: bucket_point -> list of (book, true_point, p_fair_sideA, rank, weight_raw)
    sideA = "over" if target_side in {"over", "under"} else "yes"
    sideB = "under" if sideA == "over" else "no"
    cluster_map: dict[float, list[tuple[str, float, float, int, float | None]]] = defaultdict(list)

    for book, pts in by_book.items():
        # pick freshest point (min rank) for this book
        true_point, bundle = min(pts.items(), key=lambda kv: kv[1]["rank"])
        rowA = bundle.get(sideA)
        rowB = bundle.get(sideB)
        if rowA is None and rowB is None:
            continue

        pA_raw = american_to_prob(float(rowA.price)) if rowA is not None else None
        pB_raw = american_to_prob(float(rowB.price)) if rowB is not None else None

        if pA_raw is not None and pB_raw is not None:
            pA_fair = pA_raw / max(pA_raw + pB_raw, 1e-12)  # clean de-vig
        elif pA_raw is not None:
            pA_fair = _haircut_one_sided(pA_raw, default_one_side_vig)
        elif pB_raw is not None:
            pB_fair = _haircut_one_sided(pB_raw, default_one_side_vig)
            pA_fair = 1.0 - pB_fair
        else:
            continue

        bucket_pt = _bucket_line(true_point)
        w_raw = book_weights.get(book, None)
        cluster_map[bucket_pt].append((book, float(true_point), float(pA_fair), bundle["rank"],
                                       float(w_raw) if (w_raw is not None and isfinite(w_raw) and w_raw > 0) else None))

    if not cluster_map:
        return None, None

    # For each cluster, normalize weights among books present and compute consensus
    cluster_scores = []  # (bucket_pt, p_cons, pt_cons, total_weight, n_books, best_rank, verbose_payload)
    for bucket_pt, rows_in_cluster in cluster_map.items():
        known = [(b, pt, p, r, w) for (b, pt, p, r, w) in rows_in_cluster if w is not None]
        unknown = [(b, pt, p, r) for (b, pt, p, r, w) in rows_in_cluster if w is None]

        weighted = []
        if known:
            total_w = sum(w for _, _, _, _, w in known)
            for (b, pt, p, r, w) in known:
                weighted.append((b, pt, p, r, w / total_w))
        if unknown:
            # If some books lack explicit weights:
            #  - if known exist: give unknowns zero remaining mass (strict)
            #  - else: share equally
            if known:
                for (b, pt, p, r) in unknown:
                    weighted.append((b, pt, p, r, 0.0))
            else:
                w_eq = 1.0 / len(unknown)
                for (b, pt, p, r) in unknown:
                    weighted.append((b, pt, p, r, w_eq))

        p_cons = sum(p * w for (_, _, p, _, w) in weighted)
        # keep a representative "consensus point" for this cluster
        # (use the bucket itself; books might show 94.5 vs 95.5 inside the 95.5 bucket—rare with 0.5 step)
        pt_cons = float(bucket_pt)
        total_weight = sum(w for *_, w in weighted)  # should be 1.0 or <=1.0 if we zeroed unknowns
        n_books = len(rows_in_cluster)
        best_rank = min(r for *_, r, _ in weighted) if weighted else 999999

        cluster_scores.append((bucket_pt, p_cons, pt_cons, total_weight, n_books, best_rank, weighted))

    # Pick cluster: 1) max total_weight; 2) max n_books; 3) freshest (min best_rank)
    cluster_scores.sort(key=lambda t: (t[3], t[4], -t[5]))  # total_weight asc, n_books asc, rank desc
    best = cluster_scores[-1]
    bucket_pt, p_cons, pt_cons, total_weight, n_books, best_rank, weighted = best

    if verbose:
        print(f"[odds][clusters] {player_name} :: market={market_key} side={target_side}")
        for (bp, p_c, pt_c, tw, nb, br, rows_in) in cluster_scores:
            sig = "*" if bp == bucket_pt else " "
            print(f"  {sig} line={bp:.2f}  p_cons≈{p_c:.4f}  books={nb}  total_w={tw:.3f}  freshest_rank={br}")
            for (b, pt, p, r, w) in rows_in:
                print(f"      book={b:>12} point={pt:6.2f}  p_fair≈{p:7.4f}  w={None if w is None else f'{w:.3f}'}  rank={r}")
        print(f"    -> selected line={pt_cons:.2f}  p_fair={p_cons:.4f}")

    return pt_cons, float(np.clip(p_cons, 1e-6, 1-1e-6))

def _haircut_one_sided(p_raw: float, haircut: float = DEFAULT_ONE_SIDE_VIG) -> float:
    p = p_raw * (1.0 - haircut)
    return float(np.clip(p, 1e-6, 1 - 1e-6))

# =====================================================================================
#                              STATS + FITTING HELPERS
# =====================================================================================

def _resolve_stat_column(stat: str):
    cand = _STAT_COL_ALIASES.get(stat, (stat,))
    for name in cand:
        col = getattr(NFLStat, name, None)
        if col is not None:
            return col
    return None

def _calibrate_loc(
    dist_name: str,
    base_params,
    *,
    point: float,
    side: str,
    price: float | None = None,
    p_fair: float | None = None,
    verbose: bool = False,
) -> tuple[str, tuple]:
    """
    Adjust ONLY the 'loc' so that the marginal passes through the quoted probability:
      For side in {'over','yes'}:   CDF(point) = 1 - p_fair
      For side in {'under','no'}:   CDF(point) = p_fair

    Works for continuous families (with 'scale') and discrete-with-loc families,
    but we only call this for continuous stats; counts use _calibrate_count_mean_from_ou.
    """
    side = _norm_side(side)
    if p_fair is None:
        if price is None:
            # nothing to do
            return dist_name, tuple(base_params)
        p_fair = float(np.clip(american_to_prob(price), 1e-6, 1 - 1e-6))
    else:
        p_fair = float(np.clip(p_fair, 1e-6, 1 - 1e-6))

    # target CDF to hit at 'point'
    target_cdf = (1.0 - p_fair) if side in {"over", "yes"} else p_fair

    dist_fn = getattr(st, dist_name)
    shapes, loc, scale = _split_params(dist_fn, base_params, dist_name)

    try:
        # Find the quantile in "zero-loc" coordinates, then shift loc to make
        # that quantile sit at 'point'
        if scale is None:
            q0 = dist_fn.ppf(target_cdf, *shapes, loc=0.0)
            if not np.isfinite(q0):
                return dist_name, tuple(base_params)
            new_loc = float(point) - float(q0)
            new_params = shapes + (new_loc,)
        else:
            q0 = dist_fn.ppf(target_cdf, *shapes, loc=0.0, scale=scale)
            if not np.isfinite(q0):
                return dist_name, tuple(base_params)
            new_loc = float(point) - float(q0)
            new_params = shapes + (new_loc, scale)

        if verbose:
            try:
                m0 = _mean_of(dist_name, base_params)
                m1 = _mean_of(dist_name, new_params)
                print(f"[cal] {dist_name}: point={point:.3f} side={side} p_fair={p_fair:.4f} "
                      f"→ target_cdf={target_cdf:.4f} q0={q0:.4f} loc {loc:.4f}->{new_loc:.4f} "
                      f"mean {m0:.3f}->{m1:.3f}")
            except Exception:
                print(f"[cal] {dist_name}: point={point} side={side} p_fair={p_fair} "
                      f"→ target_cdf={target_cdf} q0={q0} loc {loc}->{new_loc}")

        return dist_name, new_params
    except Exception:
        # conservative fallback
        return dist_name, tuple(base_params)

def _latest_season_for_player(session: Session, player_name: str) -> int | None:
    return session.query(func.max(NFLStat.season))\
                  .filter(NFLStat.player_display_name == player_name)\
                  .scalar()

def _season_stat_mean(session: Session, player_name: str, stat: str) -> float | None:
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

def _split_params(dist_fn, params, dist_name: str):
    """
    Return (shapes, loc, scale_or_None) correctly for both continuous and discrete dists.
    - For Poisson: params = (mu,) or (mu, loc) -> shapes=(mu,), loc, scale=None
    - For NBinom:  params = (n, p) or (n, p, loc) -> shapes=(n,p), loc, scale=None
    - Continuous:  shapes per .numargs, then (loc, scale)
    """
    params = tuple(params)
    if dist_name == "poisson":
        if len(params) == 1:
            mu, = params
            return (float(mu),), 0.0, None
        elif len(params) >= 2:
            mu, loc = params[:2]
            return (float(mu),), float(loc), None

    if dist_name == "nbinom":
        if len(params) == 2:
            n, p = params
            return (float(n), float(p)), 0.0, None
        elif len(params) >= 3:
            n, p, loc = params[:3]
            return (float(n), float(p)), float(loc), None

    # Continuous fallback using SciPy's numargs (shape count)
    numargs = int(getattr(dist_fn, "numargs", 0))
    shapes = tuple(float(x) for x in params[:numargs])
    rest = params[numargs:]
    if len(rest) >= 2:
        loc, scale = float(rest[0]), float(rest[1])
    elif len(rest) == 1:
        loc, scale = float(rest[0]), 1.0
    else:
        loc, scale = 0.0, 1.0
    return shapes, loc, scale

def _ppf(dist_name: str, params, u: np.ndarray) -> np.ndarray:
    dist_fn = getattr(st, dist_name)
    shapes, loc, scale = _split_params(dist_fn, params, dist_name)
    if scale is None:  # discrete => no scale kw
        return dist_fn.ppf(u, *shapes, loc=loc).astype(float)
    return dist_fn.ppf(u, *shapes, loc=loc, scale=scale).astype(float)

def _mean_of(dist_name: str, params) -> float:
    dist_fn = getattr(st, dist_name)
    shapes, loc, scale = _split_params(dist_fn, params, dist_name)
    rv = dist_fn(*shapes, loc=loc) if scale is None else dist_fn(*shapes, loc=loc, scale=scale)
    m = rv.mean()
    return float(m) if np.isfinite(m) else np.nan

def _is_count_dist(name: str) -> bool:
    return name in {"poisson", "nbinom"}

# --- Count calibration (pass through quoted probability) ---------------------

def _poisson_mu_from_prob_at_least_one(p: float) -> float:
    # P(X >= 1) = 1 - e^{-mu}  =>  mu = -ln(1-p)
    return float(-np.log(max(1.0 - p, 1e-9)))

def _nbinom_get_mean(params: tuple) -> tuple[float, float, float]:
    # SciPy nbinom: (n, p), mean = n*(1-p)/p
    n, p = float(params[0]), float(params[1])
    m = n*(1.0 - p)/p
    return m, n, p

def _nbinom_set_mean_keep_p(m: float, p: float) -> tuple:
    # n = m*p/(1-p)
    n = float(max(m*p/max(1e-9, (1.0-p)), 1e-9))
    return (n, p)

def _poisson_set_mu(mu: float) -> tuple:
    return (float(max(mu, 1e-9)),)

def _invert_poisson_for_over(p_over: float, k_min: int) -> float:
    """
    Solve for mu s.t. P[X >= k_min] = p_over  (sf is MONOTONE INCREASING in mu).
    If current val > target, mu is TOO BIG -> move hi down.
    """
    lo, hi = 1e-9, 20.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        val = st.poisson.sf(k_min-1, mid)  # P[X >= k_min]
        if val > p_over:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)

def _invert_poisson_for_under(p_under: float, k_max: int) -> float:
    """
    Solve for mu s.t. P[X <= k_max] = p_under  (cdf is MONOTONE DECREASING in mu).
    If current val < target, mu is TOO BIG -> move hi down.
    """
    lo, hi = 1e-9, 20.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        val = st.poisson.cdf(k_max, mid)
        if val < p_under:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)

def _calibrate_count_mean_from_ou(dist_name: str, base_params, point: float, side: str, p_fair: float) -> tuple[str, tuple]:
    side = _norm_side(side)
    if dist_name == "poisson":
        if side in {"over", "yes"}:
            k_min = int(np.floor(point)) + 1
            mu_new = _invert_poisson_for_over(p_fair, k_min)
        else:
            k_max = int(np.floor(point))
            mu_new = _invert_poisson_for_under(p_fair, k_max)
        return "poisson", _poisson_set_mu(mu_new)

    if dist_name == "nbinom":
        m0, _, p = _nbinom_get_mean(base_params)
        k_thr = int(np.floor(point))
        over = (side in {"over", "yes"})
        lo, hi = 1e-6, max(20.0, 5.0*m0 + 5.0)
        for _ in range(60):
            mid = 0.5*(lo+hi)
            n_mid, p_mid = _nbinom_set_mean_keep_p(mid, p)
            if over:
                val = st.nbinom.sf(k_thr, n_mid, p_mid)  # MONOTONE INCREASING in mean
                if val > p_fair:
                    hi = mid
                else:
                    lo = mid
            else:
                val = st.nbinom.cdf(k_thr, n_mid, p_mid) # MONOTONE DECREASING in mean
                if val < p_fair:
                    hi = mid
                else:
                    lo = mid
        m_new = 0.5*(lo+hi)
        return "nbinom", _nbinom_set_mean_keep_p(m_new, p)

    return dist_name, tuple(base_params)

# --- Role-based shrink when stat has no odds ---------------------------------

def _apply_role_to_params(dist_name: str, base_params, role: float, stat: str, verbose: bool = False):
    beta = _MISSING_STAT_EXP.get(stat, 0.9)
    rpow = float(np.clip(role, 1e-4, 1.0)) ** beta

    if _is_count_dist(dist_name):
        if dist_name == "poisson":
            mu = float(base_params[0])
            new = ("poisson", (_poisson_set_mu(mu * rpow)))
            if verbose:
                print(f"[role] {stat}: poisson mu {mu:.3g} -> {(mu*rpow):.3g}")
            return new
        if dist_name == "nbinom":
            m, _, p = _nbinom_get_mean(base_params)
            new = ("nbinom", _nbinom_set_mean_keep_p(m * rpow, p))
            if verbose:
                print(f"[role] {stat}: nbinom mean {m:.3g} -> {(m*rpow):.3g} (p kept={p:.3g})")
            return new
        return dist_name, tuple(base_params)

    # continuous: shrink both loc and scale
    dist_fn = getattr(st, dist_name)
    shapes, loc, scale = _split_params(dist_fn, base_params, dist_name)
    new_loc   = loc * rpow
    new_scale = scale * rpow if scale is not None else None
    if verbose:
        print(f"[role] {stat}: r={role:.3f} β={beta:.2f} → loc {loc:.3g}->{new_loc:.3g}, "
              f"scale {scale if scale is not None else 'NA'}->{new_scale if new_scale is not None else 'NA'}")
    return (dist_name, (shapes + (new_loc, new_scale))) if new_scale is not None else (dist_name, (shapes + (new_loc,)))

# =====================================================================================
#                             K / DST AND FPTS SUMMARY
# =====================================================================================

from scipy.stats import skew, kurtosis

def _fetch_dst_points(session: Session, team: str | None, opponent: str | None) -> np.ndarray:
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

def _latest_matchup_total_and_home_spread_sql(conn, team: str, opponent: str) -> tuple[float | None, float | None, bool | None]:
    """
    Raw-SQL fallback for game totals and the HOME spread for a given matchup.
    Returns (total, home_spread, is_team_home) where sportsbook convention is used:
      home_spread < 0 => home favored | home_spread > 0 => home underdog
    If no data found, returns (None, None, None).
    Expects a table `game_odds` with columns like the sample you shared.
    """
    # Latest TOTAL in either orientation of the matchup
    total_row = conn.execute(text("""
        SELECT point, home_abbr, away_abbr, last_update
        FROM game_odds
        WHERE market='totals' AND (
              (home_abbr=:t AND away_abbr=:o) OR
              (home_abbr=:o AND away_abbr=:t)
        )
        ORDER BY last_update DESC
        LIMIT 1
    """), {"t": team, "o": opponent}).fetchone()

    if not total_row:
        return None, None, None

    total = float(total_row[0]) if total_row[0] is not None else None
    home_abbr = str(total_row[1])
    away_abbr = str(total_row[2])
    is_team_home = (home_abbr.upper() == (team or "").upper())

    # Latest HOME spread for this exact orientation; if only AWAY row exists, invert it
    spread_row = conn.execute(text("""
        SELECT point
        FROM game_odds
        WHERE market='spreads' AND side='home' AND home_abbr=:h AND away_abbr=:a
        ORDER BY last_update DESC
        LIMIT 1
    """), {"h": home_abbr, "a": away_abbr}).fetchone()

    if not spread_row:
        away_row = conn.execute(text("""
            SELECT point
            FROM game_odds
            WHERE market='spreads' AND side='away' AND home_abbr=:h AND away_abbr=:a
            ORDER BY last_update DESC
            LIMIT 1
        """), {"h": home_abbr, "a": away_abbr}).fetchone()
        home_spread = -float(away_row[0]) if (away_row and away_row[0] is not None) else None
    else:
        home_spread = float(spread_row[0]) if spread_row[0] is not None else None

    return total, home_spread, is_team_home


def _latest_matchup_total_and_home_spread(session: Session, team: str, opponent: str) -> tuple[float | None, float | None, bool | None]:
    """ORM-first resolver for (total, home_spread, is_team_home). Falls back to SQL if needed."""
    if NFLGameOdds is not None:
        try:
            # Latest TOTAL for either orientation
            row = (session.query(NFLGameOdds)
                          .filter(
                              NFLGameOdds.market == 'totals',
                              ((NFLGameOdds.home_abbr == team) & (NFLGameOdds.away_abbr == opponent)) |
                              ((NFLGameOdds.home_abbr == opponent) & (NFLGameOdds.away_abbr == team))
                          )
                          .order_by(desc(NFLGameOdds.last_update))
                          .first())
            if row is not None:
                total = float(row.point) if row.point is not None else None
                home_abbr = str(row.home_abbr)
                away_abbr = str(row.away_abbr)
                is_team_home = (home_abbr.upper() == (team or "").upper())

                # Latest HOME spread for this exact orientation; if missing, invert latest AWAY spread
                spr = (session.query(NFLGameOdds)
                              .filter(
                                  NFLGameOdds.market == 'spreads',
                                  NFLGameOdds.side == 'home',
                                  NFLGameOdds.home_abbr == home_abbr,
                                  NFLGameOdds.away_abbr == away_abbr
                              )
                              .order_by(desc(NFLGameOdds.last_update))
                              .first())
                if spr is None:
                    spr_away = (session.query(NFLGameOdds)
                                        .filter(
                                            NFLGameOdds.market == 'spreads',
                                            NFLGameOdds.side == 'away',
                                            NFLGameOdds.home_abbr == home_abbr,
                                            NFLGameOdds.away_abbr == away_abbr
                                        )
                                        .order_by(desc(NFLGameOdds.last_update))
                                        .first())
                    home_spread = -float(spr_away.point) if (spr_away and spr_away.point is not None) else None
                else:
                    home_spread = float(spr.point) if spr.point is not None else None

                return total, home_spread, is_team_home
        except Exception:
            pass

    # Fallback to raw SQL
    with session.bind.connect() as conn:
        return _latest_matchup_total_and_home_spread_sql(conn, team, opponent)


def _implied_team_totals(total: float, home_spread: float) -> tuple[float, float]:
    """Return (home_total, away_total) using standard conversion."""
    # Home total = T/2 - s_home/2, Away total = T/2 + s_home/2
    home_tt = float(total) * 0.5 - float(home_spread) * 0.5
    away_tt = float(total) - home_tt
    return home_tt, away_tt


def _dst_normal_params_from_lines(team: str, opponent: str, total: float, home_spread: float, is_team_home: bool) -> tuple[float, float]:
    """Heuristic mapping from implied totals/spread to a DST Normal(mean, std)."""
    home_tt, away_tt = _implied_team_totals(total, home_spread)

    if is_team_home:
        team_spread = home_spread           # negative if favored
        opp_tt = away_tt
    else:
        team_spread = -home_spread          # away spread is negative of home spread
        opp_tt = home_tt

    # just lazily estimate a mean and std based on odds. better than nothing, but still doesn't factor in interceptions or sacks/fumbles. 
    mean = 11.0 - 0.25 * float(opp_tt) + 0.15 * max(-float(team_spread), 0.0) + 0.05 * (46.0 - float(total))
    mean = float(np.clip(mean, -1.5, 14.0))
    std = 5.0
    return mean, std

def _sample_dst_fpts(session: Session, team: str | None, opponent: str | None, n_samples: int) -> np.ndarray:
    """Project DST points from NFLGameOdds when available; otherwise fall back to a generic distribution."""
    rng = np.random.default_rng()

    if team and opponent:
        total, home_spread, is_team_home = _latest_matchup_total_and_home_spread(session, team, opponent)
    else:
        total = home_spread = is_team_home = None

    if (total is not None) and (home_spread is not None) and (is_team_home is not None):
        mu, sigma = _dst_normal_params_from_lines(team, opponent, float(total), float(home_spread), bool(is_team_home))
        fpts = rng.normal(loc=mu, scale=sigma, size=n_samples)
    else:
        # Fallback: historical-ish shape
        fpts = rng.normal(loc=6.5, scale=5.0, size=n_samples)

    return np.clip(fpts, -6.0, None)

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
    mapper = inspect(NFLPlayer).mapper
    cols = {attr.key for attr in mapper.column_attrs}
    commit = True  # can pull this out as arg if you need to.

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
    updates = {}
    for col, key in mapping.items():
        if col in cols and key in summary:
            updates[col] = summary[key]

    if not updates:
        if verbose:
            print(f"[summary] No matching columns on NFLPlayer; nothing to persist for id={player.id}")
        return

    session.query(NFLPlayer).filter(NFLPlayer.id == player.id).update(
        updates, synchronize_session=False
    )
    if verbose:
        set_keys = ", ".join(f"{k}={updates[k]}" for k in sorted(updates.keys()))
        print(f"[summary] Updated NFLPlayer id={player.id}: {set_keys}")

    if commit:
        session.commit()

# =====================================================================================
#                                MODELS / MAIN FLOW
# =====================================================================================

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

def _position_baseline_scrim_td_mu(marginals: dict, pos: str, verbose: bool=False) -> float:
    """Sum baseline means for rushing_tds + receiving_tds for this position."""
    mu = 0.0
    for stat in ("rushing_tds", "receiving_tds"):
        info = marginals.get(pos, {}).get(stat, {})
        if info and info.get("dist"):
            try:
                mu += float(_mean_of(info["dist"], info["params"]))
            except Exception:
                pass
    if verbose:
        print(f"[role][base] pos={pos} baseline_scrim_mu={mu:.4f}")
    return mu

def _role_from_anytime_td(session: Session, player_name: str, pos: str, marginals: dict, verbose: bool=False) -> float | None:
    """Use player_anytime_td as the usage signal → role in [ROLE_MIN, ROLE_MAX]."""
    _, p_any = _consensus_prob_for_side(session, player_name, "player_anytime_td", target_side="yes", verbose=verbose)
    if p_any is None:
        if verbose:
            print("[role] no player_anytime_td market -> cannot compute role")
        return None
    mu_any = _poisson_mu_from_prob_at_least_one(float(p_any))   # intensity from P(TD>=1)
    mu_base = _position_baseline_scrim_td_mu(marginals, pos, verbose=verbose)
    if mu_base <= 0:
        role = ROLE_MIN
    else:
        ratio = mu_any / mu_base
        role = float(np.clip(ratio ** ROLE_BETA, ROLE_MIN, ROLE_MAX))
    if verbose:
        print(f"[role][anytime] p_any={p_any:.4f} -> mu_any={mu_any:.4f}; mu_base={mu_base:.4f} -> role≈{role:.3f}")
    return role

def _has_anytime_odds(session: Session, player_name: str) -> bool:
    q = session.query(exists().where(and_(NFLOdds.player == player_name, NFLOdds.market == "player_anytime_td")))
    return bool(q.scalar())

def simulate_player_fpts(
    session: Session,
    player_id: int,
    marginals: dict,
    stacked_copulas: dict,
    n_samples: int = 10000,
    verbose: bool = False,
) -> np.ndarray:
    """
    EXACT FLOW YOU REQUESTED:
      - For each stat in the position config:
          * If there are odds for that stat: calibrate the base marginal to pass through quoted prob
          * Else: derive role from player_anytime_td and shrink the base marginal accordingly
      - After all stat marginals are set, simulate dependent samples via the position copula
      - Convert to FPTS samples
    """
    player = session.get(NFLPlayer, player_id)
    if verbose:
        print(f"matched {player_id} to DB player {player.name}")
    raw_pos = (player.position or "").upper()
    pos = 'DST' if raw_pos in DST_POS else raw_pos

    if verbose:
        print(f"position: {pos}")
    if pos == 'K':
        return st.gamma.rvs(KICKER_DEFAULT_SHAPE, scale=KICKER_DEFAULT_SCALE, size=n_samples).astype(float)
    if pos == 'DST':
        return _sample_dst_fpts(session, player.team, player.opponent, n_samples)


    if verbose:
        print(f"[player] id={player_id} name={player.name} pos={pos}")

    stats = position_configs[pos]['stats']
    corr = stacked_copulas.get(pos, np.eye(len(stats)))
    if verbose:
        print(f"[copula] stats={stats}")
        print(f"[copula] corr shape={getattr(corr, 'shape', None)}")

    # 0) Role comes from anytime TD. If missing, we will bail (skip/delete upstream).
    role_factor = _role_from_anytime_td(session, player.name, pos, marginals, verbose=verbose)

    # 1) Build per-stat adjusted distributions (either from odds or role shrink)
    adjusted = []  # list of (dist_name, params) per stat, in order

    # TD handling (combined from anytime, then split by alpha) uses uniforms later; store base for alignment
    td_indices = {"rushing_tds": None, "receiving_tds": None}
    if "rushing_tds" in stats:   td_indices["rushing_tds"] = stats.index("rushing_tds")
    if "receiving_tds" in stats: td_indices["receiving_tds"] = stats.index("receiving_tds")

    for stat in stats:
        base_info = marginals[pos][stat]
        dist_name, params = base_info["dist"], tuple(base_info["params"])

        if verbose:
            try:
                print(f"[stat][base] {stat:16s} {dist_name} params={params} mean≈{_mean_of(dist_name, params):.3f}")
            except Exception:
                print(f"[stat][base] {stat:16s} {dist_name} params={params}")

        # TD stats are filled via anytime split during sampling; store base to keep list aligned
        if stat in {"rushing_tds", "receiving_tds"} and SIMPLE_MODE:
            adjusted.append((dist_name, params))
            continue

        # Try stat-specific odds
        pt_cons, p_cons = _consensus_prob_for_side(session, player.name, stat, target_side="over", verbose=verbose)
        if (pt_cons is not None) and (p_cons is not None):
            before = (dist_name, params)
            if _is_count_dist(dist_name):
                dist_name, params = _calibrate_count_mean_from_ou(dist_name, params, float(pt_cons), "over", float(p_cons))
            else:
                dist_name, params = _calibrate_loc(dist_name, params, point=float(pt_cons), side="over", p_fair=float(p_cons), verbose=verbose)
            if verbose:
                try:
                    print(f"[stat][odds] {stat:16s} point={pt_cons} p_fair={p_cons} {before} -> {(dist_name, params)} mean≈{_mean_of(dist_name, params):.3f}")
                except Exception:
                    print(f"[stat][odds] {stat:16s} point={pt_cons} p_fair={p_cons} {before} -> {(dist_name, params)}")
            adjusted.append((dist_name, params))
            continue

        # No odds for this stat → require role_factor and shrink from baseline
        if role_factor is None:
            raise RuntimeError(f"[role][MISSING] {player.name} has no player_anytime_td -> cannot derive role for {stat}")

        before = (dist_name, params)
        dist_name, params = _apply_role_to_params(dist_name, params, role_factor, stat, verbose=verbose)
        if verbose:
            try:
                print(f"[stat][role] {stat:16s} applied role={role_factor:.3f} {before} -> {(dist_name, params)} mean≈{_mean_of(dist_name, params):.3f}")
            except Exception:
                print(f"[stat][role] {stat:16s} applied role={role_factor:.3f} {before} -> {(dist_name, params)}")
        adjusted.append((dist_name, params))

    # 2) Simulate dependent samples via Gaussian copula
    Z = np.random.multivariate_normal(np.zeros(len(stats)), corr, size=n_samples)
    U = st.norm.cdf(Z)
    X = np.zeros_like(U)

        # TDs from anytime market: Bernoulli-or-Poisson based on role/p_any, then split
    tds_filled = False
    if SIMPLE_MODE and (td_indices["rushing_tds"] is not None or td_indices["receiving_tds"] is not None):
        _, p_any = _consensus_prob_for_side(session, player.name, 'player_anytime_td', target_side='yes', verbose=verbose)
        if p_any is not None:
            # decide kernel
            kernel = _choose_td_kernel(float(p_any), float(role_factor if role_factor is not None else 1.0))

            # choose which uniform column anchors count variability
            idx_any = td_indices["rushing_tds"] if td_indices["rushing_tds"] is not None else td_indices["receiving_tds"]
            U_any = U[:, idx_any]

            if kernel == "bern":
                # 0 or 1 TD only for backups / modest p_any
                K = (U_any < float(p_any)).astype(int)
                mu_total = float(p_any)  # for logging only
            else:
                # allow multi-TD tails for starters
                mu_total = _poisson_mu_from_prob_at_least_one(float(p_any))
                K = st.poisson.ppf(U_any, mu_total).astype(int)

            # split scrimmage TDs into rush vs rec by position prior alpha
            alpha = _simple_td_alpha(pos)
            if td_indices["rushing_tds"] is not None and td_indices["receiving_tds"] is not None:
                # use the "other" column's uniforms to keep dependence
                U_split = U[:, td_indices["receiving_tds"]] if idx_any == td_indices["rushing_tds"] else U[:, td_indices["rushing_tds"]]
            else:
                U_split = U_any

            K_rush = st.binom.ppf(U_split, K, alpha).astype(int)
            K_rec  = (K - K_rush).astype(int)

            if td_indices["rushing_tds"] is not None:
                X[:, td_indices["rushing_tds"]] = K_rush
            if td_indices["receiving_tds"] is not None:
                X[:, td_indices["receiving_tds"]] = K_rec

            tds_filled = True
            if verbose:
                print(f"[tds][simple] kernel={kernel} p_any={p_any:.4f} mu_total≈{mu_total:.4f} α_rush={alpha:.2f}  "
                      f"E[K]={K.mean():.3f}  E[rush]={K_rush.mean():.3f}  E[rec]={K_rec.mean():.3f}")
        else:
            if verbose:
                print("[tds][simple] no anytime market -> TDs fall back to base marginals")

    # For non-TD stats, transform uniforms through each calibrated/shrunk marginal
    for j, stat in enumerate(stats):
        if tds_filled and stat in {"rushing_tds", "receiving_tds"}:
            if verbose:
                print(f"[stat] {stat:16s} filled from simple scrimmage TD split")
            continue

        dist_name, params = adjusted[j]
        X[:, j] = _ppf(dist_name, params, U[:, j])

    # 3) Combine to FPTS
    weights = np.array([FPTS_WEIGHTS.get(stat, 0.0) for stat in stats], dtype=float)
    fpts = X.dot(weights).astype(float)

    if verbose:
        s = _fpts_summary(fpts)
        print(f"[fpts] n={s['n']} mean={s['mean']:.3f} std={s['std']:.3f} p05={s['p05']:.3f} p50={s['p50']:.3f} p95={s['p95']:.3f}")

    return fpts

# =====================================================================================
#                              FIT & PERSIST FPTS MARGINAL
# =====================================================================================

def fit_and_store_fpts_distribution(
    session: Session,
    player_id: int,
    fpts_samples: np.ndarray,
    *,
    save_summary: bool = True,
    verbose: bool = False
) -> None:
    """
    Fit a marginal to simulated FPTS. We intentionally allow NEGATIVE support
    to avoid inflation (so we do NOT force floc=0 or filter positives only).
    We fit a simple normal ('norm') here for robustness.
    """
    xs = np.asarray(fpts_samples, dtype=float)
    xs = xs[np.isfinite(xs)]
    player = session.get(NFLPlayer, player_id)

    if xs.size == 0:
        if player is not None and hasattr(player, "fpts_dist_name"):
            player.fpts_dist_name = None
            player.fpts_dist_params = None
            if save_summary:
                _save_fpts_summary_on_player(session, player, {"n": 0})
            session.commit()
        return

    # Fit normal (allows negative support)
    loc, scale = st.norm.fit(xs)
    name, params = "norm", (loc, scale)

    player.fpts_dist_name = name
    player.fpts_dist_params = params

    if save_summary:
        if verbose:
            print("saving summary...")
        summary = _fpts_summary(xs)
        if verbose:
            print(f"fpts summary: {summary}")
        _save_fpts_summary_on_player(session, player, summary, verbose=verbose)

    session.commit()

# =====================================================================================
#                                  DRIVER / ORCHESTRATOR
# =====================================================================================

def _player_has_any_odds(session: Session, player_name: str) -> bool:
    """Fast existence check: does ANY odds row exist for this player name?"""
    q = session.query(exists().where(NFLOdds.player == player_name))
    return bool(q.scalar())

SLOTS = {"QB":1, "RB":2, "WR":3, "TE":2, "K":1, "DST":1}  # max modeled depth per base position

def assign_slots(session: Session) -> None:
    # get all active players with a team
    players = session.query(NFLPlayer).filter(NFLPlayer.team.isnot(None)).all()

    print(f"Assigning slot_label for {len(players)} players with a team…")

    # group by (team, base_pos)
    from collections import defaultdict
    buckets = defaultdict(list)
    for p in players:
        base = (p.position or "").upper()
        if base not in ("QB","RB","WR","TE","K","DST"): 
            continue
        buckets[(p.team, base)].append(p)

    for (team, base), group in buckets.items():
        # sort by projected mean (desc); use something stable if missing
        group.sort(key=lambda p: (p.fpts_mean if p.fpts_mean is not None else -1e9), reverse=True)
        cap = SLOTS[base]
        for i, p in enumerate(group, start=1):
            depth = min(i, cap)  # pool deeper guys into the last modeled slot
            if base in ("K","DST","QB"):
                slot_label = base
            else:
                slot_label = f"{base}{depth}"  # RB1/RB2, WR1/WR2/WR3, TE1/TE2
            p.slot_label = slot_label  # add this column once to NFLPlayer
    session.commit()

def _has_anytime_odds(session: Session, player_name: str) -> bool:
    q = session.query(exists().where(and_(NFLOdds.player == player_name, NFLOdds.market == "player_anytime_td")))
    return bool(q.scalar())

def generate_and_fit_all(
    session: Session,
    model_dir: Path,
    n_samples: int = 10000,
    verbose: bool = False,
    delete_players_without_odds: bool = True,
    save_fpts_sample: bool = True,
    assign_slots_after_fit: bool = True,
) -> None:
    marginals, stacked_copulas = load_models(model_dir)

    deleted = 0

    for player in session.query(NFLPlayer).all():
        pos_upper = (player.position or "").upper()
        keep_anyway = pos_upper in {"K", "DST", "D/ST", "DEF"}

        # Require at least ANY odds; and specifically require 'anytime_td' for role
        if delete_players_without_odds and not keep_anyway:
            if not _player_has_any_odds(session, player.name):
                if verbose:
                    print(f"[NO ODDS] Deleting NFLPlayer id={player.id} name={player.name}")
                session.delete(player); session.commit(); deleted += 1
                continue
            if not _has_anytime_odds(session, player.name):
                if verbose:
                    print(f"[NO ANYTIME] Deleting NFLPlayer id={player.id} name={player.name}")
                session.delete(player); session.commit(); deleted += 1
                continue

        try:
            fpts = simulate_player_fpts(
                session,
                player.id,
                marginals,
                stacked_copulas,
                n_samples,
                verbose=verbose,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if "no player_anytime_td" in msg and delete_players_without_odds and not keep_anyway:
                if verbose:
                    print(f"[NO ANYTIME] Deleting NFLPlayer id={player.id} name={player.name} (no role)")
                session.delete(player); session.commit(); deleted += 1
                continue
            else:
                raise

        if verbose:
            print(f"FPTS FOR ID: {player.id} :: {fpts}")

        fit_and_store_fpts_distribution(session, player.id, fpts, save_summary=save_fpts_sample, verbose=verbose)

    if verbose and delete_players_without_odds:
        print(f"[SUMMARY] Deleted {deleted} players with no odds.")

    if assign_slots_after_fit:
        if verbose:
            print("[slots] assigning slot_label for all players…")
        assign_slots(session)
