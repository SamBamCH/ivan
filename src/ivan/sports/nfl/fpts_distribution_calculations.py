from pathlib import Path
import joblib
from typing import Optional, Dict, Tuple
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

from sqlalchemy import and_, desc
from ivan.core.utils import normalize_name
from ivan.sports.nfl.models import NFLOdds

# ---- FPTS summary persistence helpers ---------------------------------------
import numpy as np
from scipy.stats import skew, kurtosis



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
def _poisson_mu_for_threshold(point: float, side: str, price: float) -> float:
    """
    Convert a 0.5-style TD line (or k+0.5) and price to a Poisson rate mu.
    For 'over/yes': want P(X >= k+1) = p ; for 'under/no': P(X <= k) = p.
    Solves for mu via monotone bisection on Poisson CDF/SF.
    """
    p = float(np.clip(american_to_prob(price), 1e-6, 1 - 1e-6))
    k = int(np.floor(float(point if point is not None else 0.5)))
    side_norm = str(side).lower().strip()

    # choose target function
    greater = side_norm in {"over", "yes"}
    target = p

    # expand hi until we bracket the target
    lo, hi = 0.0, 1.0
    for _ in range(40):
        val = (st.poisson.sf(k, hi) if greater else st.poisson.cdf(k, hi))
        if (val >= target and greater) or (val <= target and not greater):
            break
        hi *= 2.0
    # bisection
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = (st.poisson.sf(k, mid) if greater else st.poisson.cdf(k, mid))
        if (val > target and greater) or (val < target and not greater):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _poisson_mu_from_prob_at_least_one(prob_at_least_one: float) -> float:
    """
    If a market implies P(X>=1) = p for a Poisson X, then mu = -ln(1-p).
    """
    p = float(np.clip(prob_at_least_one, 1e-9, 1-1e-9))
    return float(-np.log(1.0 - p))

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
    
_MISSING_STAT_EXP = {}

def _apply_role_to_params(dist_name: str, base_params, role: float, stat: str):
    dist_fn = getattr(st, dist_name)
    shapes, loc, scale = _split_params(dist_fn, base_params)
    beta = _MISSING_STAT_EXP.get(stat, 0.8)
    rpow = role ** beta
    new_loc = loc * rpow
    if scale is None:  # discrete => no scale kw
        return dist_name, (shapes + (new_loc,))
    new_scale = scale * (0.5 + 0.5 * rpow)
    return dist_name, (shapes + (new_loc, new_scale))

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
                                 *, lo: float = 0.2, hi: float = 1.0) -> float:
    try:
        e_base = _mean_of(base_name, base_params)
        e_adj  = _mean_of(adj_name,  adj_params)
        if not np.isfinite(e_base) or e_base == 0:
            return 1.0
        return float(np.clip(e_adj / e_base, lo, hi))
    except Exception:
        return 1.0

def _calibrate_loc(dist_name: str, base_params, point: float, side: str, price: float, verbose: bool = False):
    """
    Shift only 'loc' so that:
      side in {'over','yes'}  -> P(X > point) = p
      side in {'under','no'}  -> P(X < point) = p
    """
    p = float(np.clip(american_to_prob(price), 1e-6, 1 - 1e-6))
    side_norm = str(side).lower().strip()
    is_over_like = side_norm in {"over", "yes"}
    target_cdf = 1.0 - p if is_over_like else p

    if verbose:
        print(f"odds->prob: point={point} side={side} price={price} -> target_cdf={target_cdf}")

    dist_fn = getattr(st, dist_name)
    shapes, loc, scale = _split_params(dist_fn, base_params)

    # compute quantile at 'standard loc' to solve for the new loc
    try:
        if scale is None:
            q0 = dist_fn.ppf(target_cdf, *shapes, loc=0.0)
        else:
            q0 = dist_fn.ppf(target_cdf, *shapes, loc=0.0, scale=scale)
    except TypeError:
        # fallback: compute around current loc then subtract it
        if scale is None:
            q0 = dist_fn.ppf(target_cdf, *shapes, loc=loc) - loc
        else:
            q0 = dist_fn.ppf(target_cdf, *shapes, loc=loc, scale=scale) - loc

    if not np.isfinite(q0):
        return dist_name, tuple(base_params)  # give up: keep base

    new_loc = float(point) - float(q0)
    if scale is None:
        new_params = shapes + (new_loc,)
    else:
        new_params = shapes + (new_loc, scale)
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

    if pos == 'K':
        return st.gamma.rvs(KICKER_DEFAULT_SHAPE, scale=KICKER_DEFAULT_SCALE, size=n_samples).astype(float)
    if pos == 'DST':
        return _sample_dst_fpts(session, getattr(player, "team", None), getattr(player, "opponent", None), n_samples)

    stats = position_configs[pos]['stats']
    corr = stacked_copulas.get(pos, np.eye(len(stats)))

    Z = np.random.multivariate_normal(np.zeros(len(stats)), corr, size=n_samples)
    U = st.norm.cdf(Z)

    X = np.zeros_like(U)

    # 1) Try to gather a single role signal from any known market for this player.
    #    Priority: if we have a market for THIS stat, we will calibrate it directly.
    #    Otherwise, we’ll derive a role factor from any available calibrated stat (e.g., Anytime TD).
    role_factor: float | None = None

    # Probe a few “role-ish” markets up front to get role_factor early.
    # You can add others, e.g., "receptions", "rushing_yards", etc.
    role_probe_markets = [
        "player_anytime_td",                 # Yes/No
        "player_tds_over",                   # Over-only touchdowns
        "player_pass_rush_reception_tds",    # combined
        "player_rush_reception_tds",         # combined (no passing)
        # Fallback to position-specific TDs if those exist:
        "player_rush_tds",
        "player_reception_tds",
    ]
    for probe in role_probe_markets:
        odds_row = _latest_odds_by_name_and_stat(session, player.name, probe)
        if not odds_row:
            continue

        # Decide which stat to calibrate from a combined/anytime TD signal
        probe_stat = None
        if probe in {"player_anytime_td", "player_tds_over", "player_pass_rush_reception_tds", "player_rush_reception_tds"}:
            if pos in ("WR", "TE"):
                probe_stat = "receiving_tds"
            elif pos in ("RB",):
                probe_stat = "rushing_tds"
            else:
                probe_stat = "rushing_tds"  # QB: sparse rushing TDs; still usable for role
        elif probe == "player_rush_tds":
            probe_stat = "rushing_tds"
        elif probe == "player_reception_tds":
            probe_stat = "receiving_tds"

        if probe_stat and probe_stat in position_configs.get(pos, {}).get("stats", []):
            base_info = marginals[pos][probe_stat]
            dn, bp = base_info["dist"], base_info["params"]
            # when probing role markets
            pt = float(odds_row.point) if odds_row.point is not None else 0.5
            dn2, ap = _calibrate_loc(dn, bp, pt, str(odds_row.side), float(odds_row.price), verbose=verbose)
            role_factor = _role_factor_from_calibrated(dn, bp, dn2, ap, lo=0.2, hi=1.0)
            if verbose:
                print(f'role_factor: {role_factor}')
            break
    # 2) Now handle every required stat
    for j, stat in enumerate(stats):
        if verbose:
            print(f'searching for stat: {stat}')
        odds_row = _latest_odds_by_name_and_stat(session, player.name, stat, verbose=verbose)

        if verbose:
            print(f'found odds row: {odds_row}')

        base_info = marginals[pos][stat]
        dist_name, params = base_info["dist"], base_info["params"]

        if odds_row is not None:
            if verbose:
                print(f'base marginals for pos={pos} stat={stat}: {dist_name, params}')
            dist_name, params = _calibrate_loc(
                dist_name=dist_name,
                base_params=params,
                point=float(odds_row.point if odds_row.point is not None else 0.5),
                side=str(odds_row.side),   # over/under or yes/no
                price=float(odds_row.price),
                verbose=verbose,
            )
            if verbose:
                print(f'adjusted params (odds): {dist_name, params}')
            if role_factor is None:
                role_factor = _role_factor_from_calibrated(
                    base_info["dist"], base_info["params"], dist_name, params, lo=0.2, hi=1.0
                )
        else:
            # No odds → try season average anchor; then optionally apply role shrink.
            mu_season = _season_stat_mean(session, player.name, stat)
            if mu_season is not None:
                dist_name, params = _shift_loc_to_mean(dist_name, params, mu_season)
                if verbose:
                    print(f'adjusted params (season mean {stat}={mu_season}): {dist_name, params}')
            if role_factor is not None:
                dist_name, params = _apply_role_to_params(dist_name, params, role_factor, stat)
                if verbose:
                    print(f'adjusted params (role): {dist_name, params}')

        X[:, j] = _ppf(dist_name, params, U[:, j])

    # === Combine to FPTS and RETURN ===
    fpts_weights = {
        'passing_yards': 0.04, 'passing_tds': 4, 'interceptions': -2,
        'rushing_yards': 0.1,  'rushing_tds': 6,
        'receptions': 1,       'receiving_yards': 0.1, 'receiving_tds': 6,
    }
    weights = np.array([fpts_weights.get(stat, 0.0) for stat in stats], dtype=float)
    fpts = X.dot(weights).astype(float)
    return fpts


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
