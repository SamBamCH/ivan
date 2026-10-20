# src/ivan/sports/nfl/conditioned.py
"""
Conditioned contest evaluation: every lineup's expected payout in each
simulated world, in closed form, instead of one counted outcome per world.

A world is a draw of the game-level latents only: each team's points scored
when the game has lines (the sampler's team-points nodes), otherwise each
team's QB latent. Given the world, every player is Normal with a conditional
mean and covariance from the game's Gaussian copula (Gaussian conditioning,
nothing fitted), so a lineup's score is Normal(m_c(w), s_c) and the field is a
mixture of such Normals, each candidate j entered own_j times in expectation
(Poisson copies, all of them with one score). For a lineup scoring x in world
w the number of entries ahead of it is

    K = sum_j B_j N_j,   B_j ~ Bernoulli(p_j(x, w)),  N_j ~ Poisson(own_j),  p_j = Phibar((x - m_j(w)) / s_j)

a lineup blocks you once with all its copies, not once per copy. In the
Poisson limit over lineups K is compound Poisson with generating function
exp(sum_j p_j (e^{own_j (t - 1)} - 1)), which the top of the ladder gets
exactly (inverse FFT of the generating function); further down, where tiers
are wide, K is Normal with the lump variance sum_j own_j p_j + own_j^2 p_j (1 - p_j)
and the ladder's expectation is closed form. Our own entry is one of 1 + N
copies, N ~ Poisson(own_c), sharing the prizes of its rank range; the Poisson
size-biasing identity makes own_c x E[our payout] the field's expected
collection from lineup c, so the whole field is expected to collect exactly
the prize pool (the sim's conservation check). The cell is then

    E[pay | w] = int phi(z) E[share(K(m_c(w) + s_c z), 1 + N)] dz

No sampled field, no rank draw, no payout scaling: the field's top is the
tail of the mixture, not the maximum of a sample. By the law of total variance
the cell has the same expectation as the counted outcome and only the
between-world variance, so a thousand worlds carry what millions of counted
ones would (Rao-Blackwell).

What remains sampled is the world, which is where the real dependence lives:
a lineup built on the low-owned game scores high in exactly the worlds where
the field's mixture sits low, and the integral sees that in every such world.

The world must also carry what the field shares. Lineups overlap heavily (in
a showdown the ten most-owned players carry 60% of all exposure), so their
residuals are far from independent: a field whose lineups are drawn
independently given only the team scores is too wide at both ends and its
maximum far too high. Under the Gaussian model the shared residual variance
is exactly L_bar' Sigma L_bar, the variance of the field's exposure-weighted
mean lineup. A tolerance on that total is not enough, though: a player left
outside the world keeps independent residual noise in the sim's field while
our lineup is credited with his full upside, a spurious uniqueness edge the
chooser will find (a 31%-owned RB left out became an 83% captain). So every
player row the field holds at SHARED_OWN_PCT or more of projected ownership
is in the world, and the cell integrates only the fringe that really is
independent across lineups. Variance reduction is then modest for
chalk-built lineups; what remains exact is the dependence, the lumps and the
ties.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.stats as st
from scipy.signal import fftconvolve
from scipy.special import ndtr

from ivan.sports.nfl.correlated_samples import CPT_MULTIPLIER, CPT_SLOT, _mu_sigma, assemble_games
from ivan.sports.nfl.refactor.dst import _quantile, _take_mean, pa_bonus_vec, team_points_from_latent

GH_NODES = 24                      # Gauss-Hermite nodes for the lineup's own residual
SHARED_OWN_PCT = 2.0               # every player row the field holds at or above this projected ownership joins the world
GH_NODES_DST = 16
SCORE_STEP = 0.5                   # score grid for the field's tail functions
SD_CLASS_STEP = 0.5                # lineups are binned by conditional sd, split linearly between classes
OWN_CLASS_RATIO = 1.4              # expected-copies classes (geometric); lineups interpolate between neighbours
OWN_CLASS_FLOOR = 0.05
TOP_LAMBDA = 1500.0                # entries expected ahead below which the count distribution is exact
TOP_FFT = 8192                     # support of the exact count distribution
TOP_MAX_POINTS = 600               # longest exact window per world (grid points)
TOP_TAIL = 1e-5                    # the window ends where fewer lineups than this are expected ahead
SMOOTH_STRIDE = 4                  # the Normal part of the ladder is evaluated every 4th grid point
QGRID = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def _hermite(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for int f(z) phi(z) dz (probabilists' Hermite), weights summing to 1."""
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return z, w / w.sum()


# ============================================================================ the world model
@dataclass
class WorldModel:
    """
    The slate conditioned on its sampled worlds. `mean[p, w]` is player row p's
    conditional mean in world w (captain rows x1.5); `blocks` holds each game's
    conditional covariance over its player rows (independent players are 1x1
    blocks). The covariance does not depend on the world (Gaussian
    conditioning), which is what makes a lineup's conditional sd a constant.
    """
    n_worlds: int
    seed: Optional[int]
    randomness: float
    player_ids: np.ndarray                                   # (P,) NFLPlayer row ids in column order
    mean: np.ndarray                                         # (P, W) float32
    blocks: List[Tuple[np.ndarray, np.ndarray]]              # (column indexes, conditional covariance)
    teams: List[str]
    team_points: np.ndarray                                  # (W, T) drawn points per team, nan without lines
    games: List[dict] = field(default_factory=list)          # per-game diagnostics

    @property
    def n_players(self) -> int:
        return len(self.player_ids)

    def column(self, pid: int) -> Optional[int]:
        return self._col.get(int(pid))

    def __post_init__(self):
        self._col = {int(p): i for i, p in enumerate(self.player_ids)}


def _dst_take_moments(pool_sorted: np.ndarray, mu_z: np.ndarray, sd_z: float, randomness: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and variance per world of the takeaway residual Q(Phi(Z)), Z ~ N(mu_z, sd_z), by Gauss-Hermite."""
    z, w = _hermite(GH_NODES_DST)
    u = ndtr(mu_z[:, None] + sd_z * z[None, :])
    vals = randomness * _quantile(pool_sorted, u)
    mean = vals @ w
    var = np.maximum((vals ** 2) @ w - mean ** 2, 0.0)
    return mean, var


def build_world_model(session, n_worlds: int, seed: Optional[int] = None, model_dir=None,
                      randomness: float = 1.0) -> WorldModel:
    """
    Draw the slate's worlds and condition every player on them. Per game the
    world nodes are the two team-points latents (with lines) or the two QB
    latents (without); each player's latent given the world is Normal with
    mean A z_w and covariance C_pp - A C_wp, A = C_pw C_ww^-1, from the
    game's correlation matrix. Offensive players and kickers map linearly to
    points (mu + sigma Z); a DST with lines is the points-allowed bracket of
    the opponent's drawn score (a function of the world) plus takeaway points,
    whose conditional mean and variance come from the fitted residual pool.
    """
    rng = np.random.default_rng(seed)
    randomness = float(np.clip(randomness, 0.0, 1.0))
    games, independent, method, dst_params = assemble_games(session, model_dir)
    W = int(n_worlds)

    pids: List[int] = []
    means: List[np.ndarray] = []
    blocks: List[Tuple[np.ndarray, np.ndarray]] = []
    teams: List[str] = []
    points_cols: List[np.ndarray] = []
    diags: List[dict] = []

    for g in games:
        a, b = g.teams
        n_p, C = g.n_players, g.corr
        if g.implied:
            world_idx = [n_p, n_p + 1]
            world_kind = "team points"
        else:
            world_idx = []
            for t in (a, b):
                own = [i for i, p in enumerate(g.players) if p.team == t]
                qbs = [i for i in own if p_slot(g.players[i]) == "QB"]
                if qbs or own:
                    world_idx.append((qbs or own)[0])
            world_kind = "QB latents (no lines)"
        # the field's shared players: condition on them, most owned first, until what the
        # field shares of its residual variance is small (L_bar' Sigma L_bar, the mean lineup's)
        mult_g = np.array([CPT_MULTIPLIER if p.roster_slot_id == CPT_SLOT else 1.0 for p in g.players])
        lbar = np.array([max(0.0, float(getattr(p, "ownership_proj", 0.0) or 0.0)) / 100.0 for p in g.players]) * mult_g
        scale = randomness * g.sigma * mult_g
        shared_before, shared_after, n_shared = _shared_variance(C, n_p, world_idx, lbar, scale), None, 0
        order = [int(i) for i in np.argsort(-lbar) if lbar[i] > 0]
        for i in order:
            own_pct = float(getattr(g.players[i], "ownership_proj", 0.0) or 0.0)
            if own_pct < SHARED_OWN_PCT:
                break                                                  # order is by exposure, so the rest are fringe
            if i in world_idx:
                continue
            world_idx.append(i); n_shared += 1
        shared_after = _shared_variance(C, n_p, world_idx, lbar, scale)
        k = len(world_idx)
        pidx = np.arange(n_p)
        Cww = C[np.ix_(world_idx, world_idx)] + 1e-8 * np.eye(k)
        Cpw = C[np.ix_(pidx, world_idx)]
        A = Cpw @ np.linalg.pinv(Cww)                                  # (n_p, k)
        Ccond = C[:n_p, :n_p] - A @ Cpw.T
        Ccond = (Ccond + Ccond.T) / 2
        np.fill_diagonal(Ccond, np.maximum(np.diag(Ccond), 0.0))
        for i in world_idx:                                            # conditioned players are deterministic in the world
            if i < n_p:
                Ccond[i, :] = 0.0; Ccond[:, i] = 0.0
        Zw = _antithetic_normal(rng, Cww, W)
        zlat = Zw @ A.T                                                # (W, n_p) conditional latent means
        mean_g = g.mu[None, :] + randomness * g.sigma[None, :] * zlat
        kappa = randomness * g.sigma.copy()                            # points per unit latent

        pts = None
        if g.implied:
            u = {a: ndtr(Zw[:, 0]), b: ndtr(Zw[:, 1])}
            pts = {t: team_points_from_latent(dst_params, g.implied[t], u[t], randomness) for t in (a, b)}
            take_pool = dst_params["_sorted"]["take_resid"]
            pa_pool = dst_params["_sorted"]["pa_resid"]
            for j, p in enumerate(g.players):
                if p_slot(p) != "DST":
                    continue
                opp = b if p.team == a else a
                pa = np.maximum(0.0, g.implied[opp] + randomness * _quantile(pa_pool, u[opp]))
                bracket = pa_bonus_vec(pa)
                take_mean = float(_take_mean(dst_params["take"]["coefficients"], g.implied[p.team], g.implied[opp]))
                mt, vt = _dst_take_moments(take_pool, zlat[:, j], math.sqrt(max(Ccond[j, j], 0.0)), randomness)
                mean_g[:, j] = bracket + take_mean + mt
                kappa[j] = math.sqrt(vt.mean() / Ccond[j, j]) if Ccond[j, j] > 1e-9 else 0.0
        cov_g = kappa[:, None] * Ccond * kappa[None, :]
        mult = np.array([CPT_MULTIPLIER if p.roster_slot_id == CPT_SLOT else 1.0 for p in g.players])
        mean_g = mean_g * mult[None, :]
        cov_g = mult[:, None] * cov_g * mult[None, :]

        cols = np.arange(len(pids), len(pids) + n_p)
        pids.extend(int(p.id) for p in g.players)
        means.append(mean_g.T.astype(np.float32))
        blocks.append((cols, cov_g))
        for t in (a, b):
            teams.append(t)
            points_cols.append(pts[t] if pts else np.full(W, np.nan))
        explained = 1.0 - np.diag(Ccond)
        diags.append({"teams": [a, b], "world": world_kind, "players": n_p, "lines": bool(g.implied),
                      "shared_players": n_shared, "world_nodes": k,
                      "shared_variance": {"before": float(shared_before), "after": float(shared_after),
                                          "lineup": float((lbar * np.diag(C[:n_p, :n_p]) * scale ** 2).sum())},
                      "implied": g.implied, "sources": dict(g.sources),
                      "latent_variance_explained": {"mean": float(explained.mean()), "min": float(explained.min()),
                                                    "max": float(explained.max())},
                      "points": ({t: {"mean": float(pts[t].mean()), "sd": float(pts[t].std())} for t in (a, b)} if pts else None)})

    for p in independent:
        mu_i, sd_i = _mu_sigma(p, 6.0)
        mult = CPT_MULTIPLIER if p.roster_slot_id == CPT_SLOT else 1.0
        cols = np.array([len(pids)])
        pids.append(int(p.id))
        means.append(np.full((1, W), mu_i * mult, dtype=np.float32))
        blocks.append((cols, np.array([[(randomness * sd_i * mult) ** 2]])))

    mean = np.vstack(means) if means else np.zeros((0, W), dtype=np.float32)
    team_points = np.column_stack(points_cols) if points_cols else np.zeros((W, 0))
    return WorldModel(n_worlds=W, seed=seed, randomness=randomness, player_ids=np.asarray(pids, dtype=np.int64),
                      mean=mean, blocks=blocks, teams=teams, team_points=team_points,
                      games=diags + [{"independent": [int(p.id) for p in independent], "method": method}])


def _conditional_diag(C: np.ndarray, n_p: int, world_idx: list) -> np.ndarray:
    """Diagonal of the players' latent covariance given the world nodes."""
    if not world_idx:
        return np.ones(n_p)
    Cww = C[np.ix_(world_idx, world_idx)] + 1e-8 * np.eye(len(world_idx))
    Cpw = C[np.ix_(np.arange(n_p), world_idx)]
    d = 1.0 - np.einsum("ij,ij->i", Cpw @ np.linalg.pinv(Cww), Cpw)
    d[[i for i in world_idx if i < n_p]] = 0.0
    return np.maximum(d, 0.0)


def _shared_variance(C: np.ndarray, n_p: int, world_idx: list, lbar: np.ndarray, scale: np.ndarray) -> float:
    """L_bar' Sigma_cond L_bar: the residual variance of the field's mean lineup given the world."""
    pidx = np.arange(n_p)
    if world_idx:
        Cww = C[np.ix_(world_idx, world_idx)] + 1e-8 * np.eye(len(world_idx))
        Cpw = C[np.ix_(pidx, world_idx)]
        Ccond = C[:n_p, :n_p] - (Cpw @ np.linalg.pinv(Cww)) @ Cpw.T
        for i in world_idx:
            if i < n_p:
                Ccond[i, :] = 0.0; Ccond[:, i] = 0.0
    else:
        Ccond = C[:n_p, :n_p]
    v = lbar * scale
    return float(max(v @ Ccond @ v, 0.0))


def _antithetic_normal(rng: np.random.Generator, cov: np.ndarray, n: int) -> np.ndarray:
    """n draws of N(0, cov) in mirrored pairs (z, -z): the sample mean is exactly zero, so no world set leans one way."""
    k = cov.shape[0]
    if k == 0:
        return np.zeros((n, 0))
    half = rng.multivariate_normal(np.zeros(k), cov + 1e-9 * np.eye(k), size=(n + 1) // 2, method="eigh")
    return np.vstack([half, -half])[:n]


def p_slot(p) -> str:
    return str(getattr(p, "slot_label", "") or "")


def describe_world_model(wm: WorldModel) -> str:
    lines = [f"worlds: {wm.n_worlds} draws of the game latents (seed {wm.seed}), {wm.n_players} player rows"]
    for g in wm.games:
        if "teams" not in g:
            if g.get("independent"):
                lines.append(f"  {len(g['independent'])} player row(s) independent of every game")
            continue
        ev = g["latent_variance_explained"]
        pts = g.get("points")
        tail = ("; team points " + ", ".join(f"{t} {v['mean']:.1f} +- {v['sd']:.1f}" for t, v in pts.items())) if pts else ""
        sv = g.get("shared_variance") or {}
        shared = (f"; field's shared residual variance {sv['before']:.0f} -> {sv['after']:.0f} of a lineup's {sv['lineup']:.0f} "
                  f"after conditioning on {g.get('shared_players', 0)} shared players ({g.get('world_nodes')} world nodes)") if sv else ""
        lines.append(f"  {'-'.join(g['teams'])}: world = {g['world']}; {g['players']} rows; share of each player's latent "
                     f"variance carried by the world {ev['mean']:.2f} (min {ev['min']:.2f}, max {ev['max']:.2f}){tail}{shared}")
    return "\n".join(lines)


# ============================================================================ lineups
@dataclass
class LineupPool:
    """The candidate pool as an incidence matrix over the world model's player columns."""
    lineup_ids: np.ndarray                     # (N,)
    L: np.ndarray                              # (N, P) float32 incidence
    players: Dict[int, List[int]]              # lineup_id -> player ids
    missing_players: List[int]                 # player ids in lineups with no column in the world model
    m: Optional[np.ndarray] = None             # (N, W) conditional means, filled by lineup_moments
    s: Optional[np.ndarray] = None             # (N,) conditional sds


def build_lineup_pool(wm: WorldModel, lineups: Dict[int, Sequence[int]]) -> LineupPool:
    ids = np.asarray(sorted(lineups), dtype=np.int64)
    L = np.zeros((len(ids), wm.n_players), dtype=np.float32)
    missing: set = set()
    for i, lid in enumerate(ids):
        for pid in lineups[int(lid)]:
            c = wm.column(pid)
            if c is None:
                missing.add(int(pid))
            else:
                L[i, c] += 1.0
    return LineupPool(lineup_ids=ids, L=L, players={int(k): [int(x) for x in v] for k, v in lineups.items()},
                      missing_players=sorted(missing))


def lineup_moments(wm: WorldModel, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Conditional mean per world (N, W) and conditional sd (N,) of each lineup's score."""
    m = L @ wm.mean
    var = np.zeros(L.shape[0], dtype=np.float64)
    for cols, C in wm.blocks:
        Lb = L[:, cols].astype(np.float64)
        var += np.einsum("ij,ij->i", Lb @ C, Lb)
    return m.astype(np.float32), np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)


# ============================================================================ the field
def score_grid(m: np.ndarray, s: np.ndarray, step: float = SCORE_STEP) -> np.ndarray:
    lo = math.floor(float(m.min()) - 6.0 * float(s.max()))
    hi = math.ceil(float(m.max()) + 7.0 * float(s.max()))
    return np.arange(lo, hi + step, step, dtype=np.float64)


def _sd_classes(s: np.ndarray):
    s_lo, s_hi = float(s.min()), float(s.max())
    n_cls = max(2, int(math.ceil((s_hi - s_lo) / SD_CLASS_STEP)) + 1)
    s_grid = np.linspace(s_lo, max(s_hi, s_lo + SD_CLASS_STEP), n_cls)
    ds = float(s_grid[1] - s_grid[0])
    pos = np.clip((s.astype(np.float64) - s_lo) / ds, 0, n_cls - 1 - 1e-9)
    c0 = np.floor(pos).astype(int)
    return s_grid, c0, pos - c0


def mixture_tails(m: np.ndarray, s: np.ndarray, grid: np.ndarray, weights: np.ndarray,
                  kernel: str = "sf") -> np.ndarray:
    """
    sum_j weights_j * k((x - m_j(w)) / s_j) on the grid for every world, k the
    Normal tail ("sf") or its square ("sf2"). Lineups are histogrammed by
    conditional mean (split linearly between grid points) and by conditional sd
    (split linearly between sd classes); each class is then one FFT
    convolution with its kernel. Returns (W, G).
    """
    N, W = m.shape
    G = len(grid)
    step = float(grid[1] - grid[0])
    weights = np.asarray(weights, dtype=np.float64)
    keep = weights != 0
    if not keep.any():
        return np.zeros((W, G))
    m, s, weights = m[keep], s[keep], weights[keep]
    s_grid, c0, fs = _sd_classes(s)
    n_sd = len(s_grid)
    pos = (m.astype(np.float64) - grid[0]) / step
    i0 = np.clip(np.floor(pos), 0, G - 2).astype(np.int64)
    f = np.clip(pos - i0, 0.0, 1.0)
    wrow = (np.arange(W, dtype=np.int64) * n_sd)[None, :]
    H = np.zeros(W * n_sd * G)
    for cls, wc in ((c0, (1.0 - fs) * weights), (np.minimum(c0 + 1, n_sd - 1), fs * weights)):
        base = (wrow + cls[:, None]) * G + i0
        H += np.bincount(base.ravel(), weights=(wc[:, None] * (1.0 - f)).ravel(), minlength=H.size)
        H += np.bincount((base + 1).ravel(), weights=(wc[:, None] * f).ravel(), minlength=H.size)
    H = H.reshape(W, n_sd, G)
    delta = np.arange(-(G - 1), G) * step
    out = np.zeros((W, G))
    for c in range(n_sd):
        if not H[:, c, :].any():
            continue
        k = ndtr(-delta / max(float(s_grid[c]), 1e-3))        # Phibar((x_g - x_b) / s_c), indexed by g - b
        if kernel == "sf2":
            k = k * k
        out += fftconvolve(H[:, c, :], k[None, :], mode="full", axes=1)[:, G - 1:2 * G - 1]
    return np.maximum(out, 0.0)


def field_survival(m: np.ndarray, s: np.ndarray, own: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """lambda_w(x): the expected number of field entries above score x in world w, on the grid."""
    return mixture_tails(m, s, grid, own, "sf").astype(np.float32)


def field_quantiles(surv: np.ndarray, grid: np.ndarray, F: int, paid_places: int,
                    q: Sequence[float] = QGRID) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per world: the field's score percentiles, the cash line (the score with
    `paid_places` expected entries above it) and the winning score (the median
    of the maximum: ln 2 entries expected above it).
    """
    W = surv.shape[0]
    targets = np.asarray([F * (1.0 - qq / 100.0) for qq in q], dtype=np.float64)
    pct = np.zeros((W, len(q)), dtype=np.float32)
    cash = np.zeros(W, dtype=np.float32)
    win = np.zeros(W, dtype=np.float32)
    for w in range(W):
        xp = -surv[w].astype(np.float64)                   # increasing in x
        pct[w] = np.interp(-targets, xp, grid)
        cash[w] = np.interp(-float(max(paid_places, 1)), xp, grid)
        win[w] = np.interp(-math.log(2.0), xp, grid)
    return pct, cash, win


def own_classes(own: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geometric classes of expected copies: (class values, class index per lineup, fractional position in log space)."""
    own = np.asarray(own, dtype=np.float64)
    lo = max(OWN_CLASS_FLOOR, float(own[own > 0].min()) if (own > 0).any() else OWN_CLASS_FLOOR)
    hi = max(float(own.max()), lo * OWN_CLASS_RATIO)
    n = int(math.ceil(math.log(hi / lo) / math.log(OWN_CLASS_RATIO))) + 1
    vals = lo * OWN_CLASS_RATIO ** np.arange(n)
    vals[-1] = max(vals[-1], hi)
    pos = np.clip(np.log(np.maximum(own, lo) / lo) / math.log(OWN_CLASS_RATIO), 0, n - 1 - 1e-9)
    c0 = np.floor(pos).astype(int)
    return vals, c0, pos - c0


# ============================================================================ payouts
class Ladder:
    """The contest's prize ladder: tiers, cumulative prizes and the closed forms over them."""

    def __init__(self, pay_by_rank: np.ndarray, F: int):
        F = int(F)
        pay = np.zeros(F + 1)
        pr = np.asarray(pay_by_rank, dtype=np.float64)
        pay[1:min(F, len(pr)) + 1] = pr[:F]
        self.F = F
        self.pay = pay
        self.cs = np.cumsum(pay)                                     # cs[r] = prizes at ranks 1..r
        self.pool = float(self.cs[-1])
        self.top_prize = float(pay[1])
        self.paid_places = int(np.nonzero(pay > 0)[0].max()) if (pay > 0).any() else 0
        edges = np.nonzero(np.diff(pay[1:]) != 0)[0] + 2           # first rank of each new tier
        lo = np.concatenate([[1], edges])
        hi = np.concatenate([edges - 1, [F]])
        amt = pay[lo]
        keep = amt > 0
        self.tier_lo, self.tier_hi, self.tier_amt = lo[keep].astype(np.float64), hi[keep].astype(np.float64), amt[keep]

    def share(self, K, d):
        """Prize to one of d copies with K entries ahead: the mean prize over ranks K+1..K+d (d may be fractional)."""
        K = np.minimum(np.asarray(K).astype(np.int64), self.F)
        d = np.asarray(d, dtype=np.float64)
        di = np.floor(d).astype(np.int64)
        fr = d - di
        hi = np.minimum(K + di, self.F)
        extra = np.where(hi < self.F, self.pay[np.minimum(hi + 1, self.F)] * fr, 0.0)
        return (self.cs[hi] - self.cs[K] + extra) / np.maximum(d, 1e-12)

    def expected_cs_normal(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """E[cs[Y]] for Y ~ N(mu, sigma^2), cs piecewise linear: sum_t amt_t E[clip(Y - (lo_t - 1), 0, len_t)]."""
        out = np.zeros_like(mu, dtype=np.float64)
        sig = np.maximum(sigma, 1e-6)
        for lo, hi, amt in zip(self.tier_lo, self.tier_hi, self.tier_amt):
            b = lo - 1.0
            out += amt * sig * (_psi((mu - b) / sig) - _psi((mu - b - (hi - lo + 1.0)) / sig))
        return out

    def expected_share_normal(self, lam: np.ndarray, var: np.ndarray, d) -> np.ndarray:
        """E[share(K, d)] with K ~ N(lam, var) (the smooth part of the ladder)."""
        sig = np.sqrt(np.maximum(var, 1e-9))
        d = np.asarray(d, dtype=np.float64)
        return (self.expected_cs_normal(lam + d, sig) - self.expected_cs_normal(lam, sig)) / np.maximum(d, 1e-12)

    def copies_share_table(self, own_vals: np.ndarray, k_max: int) -> np.ndarray:
        """S[k, c] = E_N[share(k, 1 + N)], N ~ Poisson(own_c): our entry's prize with k entries ahead."""
        S = np.zeros((k_max, len(own_vals)))
        k = np.arange(k_max)
        for c, o in enumerate(own_vals):
            n_hi = int(o + 8 * math.sqrt(o) + 12)
            n = np.arange(0, n_hi + 1)
            pn = st.poisson.pmf(n, o)
            S[:, c] = (self.share(k[:, None], (1 + n)[None, :]) * pn[None, :]).sum(axis=1)
        return S


def _psi(u: np.ndarray) -> np.ndarray:
    """E[(Z + u)^+] for Z standard Normal: u Phi(u) + phi(u)."""
    return u * ndtr(u) + np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)


# ============================================================================ the tables
@dataclass
class ContestTables:
    """
    Per world and grid score, everything the cell reads. H[c] is E[our prize |
    score x, copies class c] (exact in the window, Normal elsewhere), Q[c] the
    first-order effect of our own expected presence in the count ahead (the
    window counts our own copies as a possible blocker and the smooth part
    counts them in lambda; both tie with us instead), and win/cash/lam the
    first-place and cash probabilities and the count ahead.
    """
    grid: np.ndarray
    lam: np.ndarray            # (W, G)
    var: np.ndarray            # (W, G)
    lam_cls: np.ndarray        # (C, W, G)
    own_vals: np.ndarray       # (C,) copies-class grid our entry interpolates on
    lump: np.ndarray           # (C,) mean copies of the class members (the lump a blocking lineup brings)
    H: np.ndarray              # (C, W, G)
    Q: np.ndarray              # (C, W, G)
    win: np.ndarray            # (W, G)
    cash: np.ndarray           # (W, G)
    dcash: np.ndarray          # (W, G) d cash / d lambda
    i_top: np.ndarray          # (W,)
    n_top: int
    S: np.ndarray              # (K, C)
    s_grid: np.ndarray         # (n_sd,) conditional-sd classes of the blur


def _exp_c64(phi: np.ndarray) -> np.ndarray:
    re = np.exp(phi.real.astype(np.float32))
    im = phi.imag.astype(np.float32)
    return re * (np.cos(im) + 1j * np.sin(im))


def contest_tables(m: np.ndarray, s: np.ndarray, own: np.ndarray, ladder: Ladder, grid: Optional[np.ndarray] = None,
                   world_chunk: int = 50, log=None) -> ContestTables:
    N, W = m.shape
    if grid is None:
        grid = score_grid(m, s)
    G = len(grid)
    step = float(grid[1] - grid[0])
    own = np.asarray(own, dtype=np.float64)
    vals, c0, _ = own_classes(own)
    C = len(vals)
    cls = np.where(own > 0, c0, -1)
    s_grid, _, _ = _sd_classes(s)

    lam = mixture_tails(m, s, grid, own, "sf")
    o2 = own * own
    var = lam + mixture_tails(m, s, grid, o2, "sf") - mixture_tails(m, s, grid, o2, "sf2")
    var = np.maximum(var, lam * (1.0 - np.minimum(lam / max(ladder.F, 1), 1.0)) + 1e-9)
    lam_cls = np.zeros((C, W, G), dtype=np.float32)
    for c in range(C):
        members = (cls == c).astype(np.float64)
        if members.any():
            lam_cls[c] = mixture_tails(m, s, grid, members, "sf")
    lam_L = lam_cls.sum(axis=0)

    # --- the smooth part: Normal count with the lump variance, our copies 1 + own_c, every SMOOTH_STRIDE-th point
    sub = np.arange(0, G, SMOOTH_STRIDE)
    if sub[-1] != G - 1:
        sub = np.append(sub, G - 1)
    lam_s, var_s = lam[:, sub], var[:, sub]
    H = np.zeros((C, W, G), dtype=np.float32)
    Q = np.zeros((C, W, G), dtype=np.float32)
    dl = np.maximum(lam_s * 1e-3, 1e-6)
    for c, o in enumerate(vals):
        h = ladder.expected_share_normal(lam_s, var_s, 1.0 + o)
        dh = (ladder.expected_share_normal(lam_s + dl, var_s, 1.0 + o) - h) / dl       # dH/dlambda
        H[c] = _interp_rows(grid[sub], h, grid)
        Q[c] = _interp_rows(grid[sub], o * dh, grid)                                  # removing own*Phibar(z) from lambda
    sig = np.sqrt(var)
    if ladder.paid_places > 0:
        u = (ladder.paid_places - 0.5 - lam) / sig
        cash = ndtr(u).astype(np.float32)
        dcash = (-np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi) / sig).astype(np.float32)
    else:
        cash = np.zeros((W, G), np.float32); dcash = np.zeros((W, G), np.float32)
    win = np.zeros((W, G), dtype=np.float32)

    # --- the exact part: compound Poisson count inverted on each world's window
    i_top = np.argmax(lam <= TOP_LAMBDA, axis=1)
    i_top[~(lam <= TOP_LAMBDA).any(axis=1)] = G - 1
    tail_end = np.argmax(lam_L < TOP_TAIL, axis=1)
    tail_end[~(lam_L < TOP_TAIL).any(axis=1)] = G - 1
    n_top = int(min(TOP_MAX_POINTS, max(2, (tail_end - i_top).max() + 2)))
    R = TOP_FFT
    S = ladder.copies_share_table(vals, R)
    S2 = np.zeros_like(S)                                            # S2[k, c] = E_N[S[k + N, c]], N ~ Poisson(own_c)
    for c, o in enumerate(vals):
        n_hi = int(o + 8 * math.sqrt(o) + 12)
        n = np.arange(0, n_hi + 1)
        pn = st.poisson.pmf(n, o)
        idx = np.minimum(np.arange(R)[:, None] + n[None, :], R - 1)
        S2[:, c] = (S[idx, c] * pn[None, :]).sum(axis=1)
    lump = np.array([own[cls == c].mean() if (cls == c).any() else vals[c] for c in range(C)])   # mean copies per class
    t = np.exp(2j * np.pi * np.arange(R) / R)
    E1 = (np.exp(lump[:, None] * (t[None, :] - 1.0)) - 1.0).astype(np.complex64)             # E[t^N] - 1, N ~ Poisson(lump_c)
    for a in range(0, W, world_chunk):
        b = min(a + world_chunk, W)
        rows = np.arange(a, b)
        idx = np.minimum(i_top[rows][:, None] + np.arange(n_top)[None, :], G - 1)
        rates = lam_cls[:, rows[:, None], idx].transpose(1, 2, 0).reshape(-1, C).astype(np.complex64)
        pmf = np.fft.fft(_exp_c64(rates @ E1), axis=1).real / R        # P(K = k) = (1/R) sum_t G(t) t^-k
        pmf = np.maximum(pmf, 0.0)
        pmf /= np.maximum(pmf.sum(axis=1, keepdims=True), 1e-12)
        Hx = (pmf @ S).reshape(b - a, n_top, C)
        Qx = (pmf @ (S2 - S)).reshape(b - a, n_top, C)              # (H_shift - H): the first-order self-block term
        w0 = pmf[:, 0].reshape(b - a, n_top)
        for r_i, w in enumerate(rows):
            j = idx[r_i]
            H[:, w, j] = Hx[r_i].T
            Q[:, w, j] = Qx[r_i].T
            win[w, j] = w0[r_i]
            if j[-1] < G - 1:                                        # beyond the window nobody is ahead
                H[:, w, j[-1] + 1:] = S[0][:, None]
                Q[:, w, j[-1] + 1:] = (S2[0] - S[0])[:, None]
                win[w, j[-1] + 1:] = 1.0
    if log:
        log(f"  tables: {C} copies classes, {len(s_grid)} sd classes, exact window {n_top} points per world, {G} grid points")
    return ContestTables(grid=grid, lam=lam.astype(np.float32), var=var.astype(np.float32), lam_cls=lam_cls, own_vals=vals,
                         lump=lump, H=H, Q=Q, win=win, cash=cash, dcash=dcash, i_top=i_top, n_top=n_top, S=S, s_grid=s_grid)


def _interp_rows(x_sub: np.ndarray, y_sub: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Row-wise linear interpolation of (W, len(x_sub)) onto x."""
    pos = np.interp(x, x_sub, np.arange(len(x_sub)))
    i0 = np.clip(np.floor(pos).astype(int), 0, len(x_sub) - 2)
    f = pos - i0
    return (y_sub[:, i0] * (1 - f) + y_sub[:, i0 + 1] * f).astype(np.float32)


# ============================================================================ the cell
def _blur(T: np.ndarray, sigma: float, step: float, kernel: str) -> np.ndarray:
    """
    Convolve each row of T (W, G) along the grid with phi(delta/sigma)/sigma
    ("phi") or phi(delta/sigma) Phibar(delta/sigma)/sigma ("phi_sf"), the
    weight of a lineup's own residual: a lineup with conditional mean m reads
    int phi(z) T(m + sigma z) dz at m.
    """
    G = T.shape[1]
    sigma = max(float(sigma), step)                   # below the grid step the residual is a delta on the grid
    delta = np.arange(-(G - 1), G) * step
    z = delta / sigma
    k = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) * (step / sigma)
    if kernel == "phi_sf":
        k = k * ndtr(-z)
    k = k * ((1.0 if kernel == "phi" else 0.5) / max(k.sum(), 1e-12))   # the discretised kernel carries the continuous weight
    # T(m + sigma z) at delta = sigma z: out[g] = sum_b T[b] k((x_b - x_g)/sigma) -> correlate with the reversed kernel
    return fftconvolve(T, k[::-1][None, :], mode="full", axes=1)[:, G - 1:2 * G - 1]


def _gather(Tbl: np.ndarray, i0: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Tbl (W, G) read at per-(lineup, world) grid positions i0, f of shape (n, W)."""
    w = np.arange(Tbl.shape[0], dtype=np.int32)[None, :]
    return Tbl[w, i0] * (1.0 - f) + Tbl[w, i0 + 1] * f


def _grid_pos(grid: np.ndarray, x: np.ndarray):
    G = len(grid)
    step = float(grid[1] - grid[0])
    pos = (x - grid[0]) / step
    i0 = np.clip(np.floor(pos), 0, G - 2).astype(np.int32)
    return i0, np.clip(pos - i0, 0.0, 1.0).astype(np.float32)


@dataclass
class ContestEvaluation:
    pay: np.ndarray        # (N, W) expected prize to one entry of the lineup, per world
    cash: np.ndarray       # (N, W) P(the entry cashes | world)
    rank: np.ndarray       # (N, W) expected rank
    win: np.ndarray        # (N,) P(first place)
    draw: np.ndarray       # (N, W) one realized prize per world (for calibration tests, not for choosing)
    tables: ContestTables

    def roi(self, fee: float) -> np.ndarray:
        return self.pay / fee - 1.0

    def roi_draw(self, fee: float) -> np.ndarray:
        return self.draw / fee - 1.0


def evaluate_contest(m: np.ndarray, s: np.ndarray, own: np.ndarray, ladder: Ladder, tables: Optional[ContestTables] = None,
                     rng: Optional[np.random.Generator] = None, chunk: int = 512, log=None) -> ContestEvaluation:
    """
    The cell for every lineup and world: the tables blurred along the score
    axis by the lineup's own residual (one FFT per copies class and sd class),
    read at the lineup's conditional mean. Also draws one realized outcome per
    world under the same law for the calibration tests.
    """
    N, W = m.shape
    rng = rng or np.random.default_rng(0)
    if tables is None:
        tables = contest_tables(m, s, own, ladder, log=log)
    T = tables
    G = len(T.grid)
    step = float(T.grid[1] - T.grid[0])
    own = np.asarray(own, dtype=np.float64)
    _, oc0, ofr = own_classes_on(T.own_vals, own)
    s_grid = T.s_grid
    n_sd = len(s_grid)
    pos_s = np.clip((s.astype(np.float64) - s_grid[0]) / (s_grid[1] - s_grid[0]), 0, n_sd - 1 - 1e-9)
    sc0 = np.floor(pos_s).astype(int)
    sfr = (pos_s - sc0).astype(np.float32)
    i0, f = _grid_pos(T.grid, m.astype(np.float64))

    pay = np.zeros((N, W), dtype=np.float32)
    cash = np.zeros((N, W), dtype=np.float32)
    rank = np.zeros((N, W), dtype=np.float32)
    win = np.zeros((N, W), dtype=np.float32)
    # per sd class: the copies-free tables, blurred
    for k in range(n_sd):
        rows = np.nonzero((sc0 == k) | (sc0 + 1 == k))[0]
        if rows.size == 0:
            continue
        wk = np.where(sc0[rows] == k, 1.0 - sfr[rows], sfr[rows])[:, None]
        sig = max(float(s_grid[k]), 1e-3)
        lam_b = _blur(T.lam, sig, step, "phi")
        cash_b = _blur(T.cash, sig, step, "phi")
        dcash_b = _blur(T.dcash, sig, step, "phi_sf")
        win_b = _blur(T.win, sig, step, "phi")
        rank[rows] += wk * (_gather(lam_b, i0[rows], f[rows]) + 1.0 - own[rows][:, None] * 0.5)
        cash[rows] += wk * (_gather(cash_b, i0[rows], f[rows]) - own[rows][:, None] * _gather(dcash_b, i0[rows], f[rows]))
        win[rows] += wk * _gather(win_b, i0[rows], f[rows])
    # per copies class and sd class: the prize tables, blurred, with the self-presence term
    for c in range(len(T.own_vals)):
        in_c = np.nonzero((oc0 == c) | (oc0 + 1 == c))[0]
        if in_c.size == 0:
            continue
        wc = np.where(oc0[in_c] == c, 1.0 - ofr[in_c], ofr[in_c])
        for k in range(n_sd):
            sel = np.nonzero((sc0[in_c] == k) | (sc0[in_c] + 1 == k))[0]
            if sel.size == 0:
                continue
            rows = in_c[sel]
            wk = (wc[sel] * np.where(sc0[rows] == k, 1.0 - sfr[rows], sfr[rows]))[:, None]
            sig = max(float(s_grid[k]), 1e-3)
            H_b = _blur(T.H[c], sig, step, "phi")
            Q_b = _blur(T.Q[c], sig, step, "phi_sf")
            pay[rows] += wk * (_gather(H_b, i0[rows], f[rows]) - _gather(Q_b, i0[rows], f[rows]))
    pay = np.maximum(pay, 0.0)
    cash = np.clip(cash, 0.0, 1.0)
    win_p = np.clip(win, 0.0, 1.0).mean(axis=1)

    # one realization per world under the same law
    draw = np.zeros((N, W), dtype=np.float32)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        mc, sc, oc = m[a:b].astype(np.float64), s[a:b].astype(np.float64), own[a:b]
        zr = rng.standard_normal((b - a, W))
        j0, g = _grid_pos(T.grid, mc + sc[:, None] * zr)
        top = j0 >= T.i_top[None, :]
        lam_r = np.maximum(_gather(T.lam, j0, g) - oc[:, None] * ndtr(-zr), 0.0)
        var_r = np.maximum(_gather(T.var, j0, g) - oc[:, None] * ndtr(-zr), lam_r * 0.5 + 1e-9)
        K = np.maximum(np.round(lam_r + np.sqrt(var_r) * rng.standard_normal((b - a, W))), 0).astype(np.int64)
        if top.any():
            ti, tw = np.nonzero(top)
            rates = T.lam_cls[:, tw, j0[ti, tw]] * (1 - g[ti, tw]) + T.lam_cls[:, tw, j0[ti, tw] + 1] * g[ti, tw]   # (C, n)
            M = rng.poisson(np.maximum(rates, 0.0))
            K[ti, tw] = rng.poisson(M * T.lump[:, None]).sum(axis=0)
        ncop = 1 + rng.poisson(oc)[:, None] * np.ones((1, W), dtype=np.int64)
        draw[a:b] = ladder.share(np.minimum(K, ladder.F), ncop)
    return ContestEvaluation(pay=pay, cash=cash, rank=rank, win=win_p, draw=draw, tables=T)


def own_classes_on(vals: np.ndarray, own: np.ndarray):
    """Class index and fractional log position of each lineup's expected copies on an existing class grid."""
    lo = float(vals[0])
    n = len(vals)
    pos = np.clip(np.log(np.maximum(np.asarray(own, dtype=np.float64), lo) / lo) / math.log(OWN_CLASS_RATIO), 0, n - 1 - 1e-9)
    c0 = np.floor(pos).astype(int)
    return vals, c0, pos - c0


def pool_conservation(ev: ContestEvaluation, own: np.ndarray, ladder: Ladder) -> dict:
    """
    What the field as a whole expects to collect in each world. With Poisson
    copies, own_c x E[our prize] is lineup c's expected collection, so the
    total should be the prize pool in every world; the gap measures the
    Poisson-limit, first-order and binning approximations.
    """
    paid = (np.asarray(own, dtype=np.float64)[:, None] * ev.pay.astype(np.float64)).sum(axis=0)
    rel = (paid - ladder.pool) / max(ladder.pool, 1e-9)
    return {"expected_pool": ladder.pool, "paid_mean": float(paid.mean()), "max_abs_error": float(np.abs(paid - ladder.pool).max()),
            "rel_error_mean": float(rel.mean()), "rel_error_max": float(np.abs(rel).max()),
            "sims_off": int((np.abs(rel) > 0.02).sum())}
