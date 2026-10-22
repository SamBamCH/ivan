"""
Team minutes as a distribution: every player's minutes drawn jointly with his
teammates', conditioned on the game.

A team plays T = 240 (+25 per overtime) minutes. In one world:
  1. the game: final margin ~ N(spread, MARGIN_SD); overtime with P_OT
  2. who plays: each player available with his own probability (status, play rate)
  3. the split: minutes_i = T * softmax_i( log mu_i + beta[role_i] * |margin| + eps_i ),
     eps_i ~ N(0, tau[role_i]); mu_i is the allocated minutes (minutes.py), the
     role is read off mu_i and the starter flag.
The softmax keeps the team at T, so one player's extra minutes are someone
else's, and tau by role makes a star's draw tight and a bench player's wide.
beta < 0 for starters: a blowout moves their minutes to the bench.

fit_team_minutes() reads tau and beta off last season's player-games (the
log of actual over allocated minutes, by role, against |final margin|); the
parameters live in data/nba/models/team_minutes.json and load_params() reads
them (defaults when the file is missing, so the sampler always runs).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ivan.sports.nba.paths import MODELS_DIR

if TYPE_CHECKING:
    from .history import HistoryState

PARAMS_JSON = MODELS_DIR / "team_minutes.json"
ROLES = ("star", "starter", "rotation", "deep")
OT_MINUTES = 25.0
REGULATION = 240.0
DEFAULTS = {
    "margin_sd": 12.5,          # final margin around the spread
    "p_ot": 0.055,              # share of games with overtime (2025-26 regular season)
    "tau": {"star": 0.14, "starter": 0.22, "rotation": 0.40, "deep": 0.70},
    "beta": {"star": -0.010, "starter": -0.006, "rotation": 0.006, "deep": 0.030},   # per point of |margin|
    "alpha": {"star": 0.0, "starter": 0.0, "rotation": 0.0, "deep": 0.0},            # fit report only: the scaled-baseline stand-in's bias by role; the sampler centres on mu
    "scratch": {"Q": 0.35, "GTD": 0.35, "D": 0.6, "P": 0.1},                        # P(out) by DK status
    # a bench player with no status who is projected to play but does not (coach's decision): by salary, measured on the
    # 2026-10-20/21 slates (109 such players: <=3.5k 38%, 3.5-4.5k 10%, 4.5-6k 9%, 6k+ none)
    "scratch_bench": {"3500": 0.35, "4500": 0.10, "6000": 0.08, "else": 0.0},
}


def role_of(mu: float, starter: Optional[bool]) -> str:
    if mu >= 28:
        return "star"
    if mu >= 18:
        return "starter" if starter else "rotation"
    if mu >= 9:
        return "rotation"
    return "deep"


def load_params(path=PARAMS_JSON) -> dict:
    p = json.loads(json.dumps(DEFAULTS))
    try:
        p.update(json.loads(path.read_text()))
    except (OSError, ValueError):
        pass
    return p


# ----------------------------------------------------------------------------- sampling
@dataclass
class TeamDraw:
    minutes: np.ndarray            # (n, k)
    available: np.ndarray          # (n, k) bool
    margin: np.ndarray             # (n,) the team's own margin (positive = won)
    overtime: np.ndarray           # (n,) int periods
    roles: List[str] = field(default_factory=list)


def sample_team_minutes(mu: Sequence[float], roles: Sequence[str], n: int, rng: np.random.Generator, spread: Optional[float] = None,
                        p_out: Optional[Sequence[float]] = None, params: Optional[dict] = None, margin: Optional[np.ndarray] = None,
                        overtime: Optional[np.ndarray] = None, pinned: Optional[Sequence[bool]] = None,
                        tau_override: Optional[Sequence[Optional[float]]] = None, budget: Optional[float] = None) -> TeamDraw:
    """
    n worlds of a team's minutes. mu: allocated minutes per player (the mean
    rotation in an average game); spread: the team's own line (negative =
    favoured), 0 when unknown; p_out: per-player scratch probability; margin /
    overtime: pass the game's draws to share them with the opponent's call.
    Pinned players keep their minutes exactly (no noise), the rest share the
    remainder.

    The share noise is centred on a reference draw at pick'em (margin ~ N(0,
    sd)), so in an average game every available player's expected minutes is
    his allocation (which already averages over game scripts), while a known
    spread or a realized blowout moves the starters' minutes by the fitted
    beta: a heavy favourite's stars project a little lower, and worlds that
    blow out move their minutes to the bench.
    """
    P = params or load_params()
    mu = np.asarray(mu, dtype=float); k = len(mu)
    tau = np.array([P["tau"][r] for r in roles]); beta = np.array([P["beta"][r] for r in roles])
    if tau_override is not None:
        tau = np.array([t if (t is not None and np.isfinite(t)) else d for t, d in zip(tau_override, tau)], dtype=float)
    if margin is None:
        margin = rng.normal(-(spread or 0.0), P["margin_sd"], n)
    if overtime is None:
        overtime = (rng.random(n) < P["p_ot"]).astype(int)
    T = (REGULATION if budget is None else float(budget)) + OT_MINUTES * overtime     # budget: the minutes assigned by hand (may not be 240)
    avail = np.ones((n, k), dtype=bool)
    if p_out is not None:
        avail = rng.random((n, k)) >= np.asarray(p_out, dtype=float)[None, :]
    pin = np.zeros(k, dtype=bool) if pinned is None else np.asarray(pinned, dtype=bool)
    free = avail & ~pin[None, :]
    eps = rng.standard_normal((n, k)) * tau[None, :]
    e0 = expected_abs_margin(0.0, P["margin_sd"])
    budget = (T - mu[pin].sum())[:, None]
    cap = (48.0 + 5.0 * overtime)[:, None] * np.ones((1, k)); cap[:, pin] = np.inf
    base = np.log(np.maximum(mu, 0.05))[None, :] + eps

    def draw(logit: np.ndarray, scale: np.ndarray) -> np.ndarray:
        w = np.where(free, np.exp(logit - logit.max(axis=1, keepdims=True)) * scale[None, :], 0.0)
        share = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        return _redistribute_over_cap(share * np.maximum(budget, 0.0), cap, free)

    # the centring scale per player, from a reference draw at pick'em with the same noise
    target = mu * (1.0 - (np.asarray(p_out, dtype=float) if p_out is not None else 0.0))
    scale = np.ones(k)
    if n >= 20 and free.any():
        ref_margin = rng.normal(0.0, P["margin_sd"], n)
        ref_logit = base + beta[None, :] * (np.abs(ref_margin)[:, None] - e0)
        for _ in range(12):
            m = np.where(free, draw(ref_logit, scale), 0.0).mean(axis=0)
            ratio = np.where(m > 1e-9, target / np.maximum(m, 1e-9), 1.0)
            scale = scale * np.clip(ratio, 0.25, 4.0)
            if np.abs(ratio[target > 0.5] - 1.0).max(initial=0.0) < 0.003:
                break
    minutes = draw(base + beta[None, :] * (np.abs(margin)[:, None] - e0), scale)
    minutes[:, pin] = mu[pin]
    minutes[~avail] = 0.0
    return TeamDraw(minutes=minutes, available=avail, margin=margin, overtime=overtime, roles=list(roles))


def _redistribute_over_cap(minutes: np.ndarray, cap: np.ndarray, free: np.ndarray, passes: int = 4) -> np.ndarray:
    """Minutes above the cap go to the other free players of the same world, pro rata; the world's total is kept."""
    for _ in range(passes):
        over = np.maximum(minutes - cap, 0.0)
        if not over.any():
            break
        minutes = minutes - over
        room = np.where((minutes < cap - 1e-9) & free, minutes, 0.0)
        tot = room.sum(axis=1, keepdims=True)
        minutes = minutes + np.where(tot > 0, over.sum(axis=1, keepdims=True) * room / np.maximum(tot, 1e-12), 0.0)
    return minutes


def tau_from_band(low: Optional[float], high: Optional[float]) -> Optional[float]:
    """The log-scale noise that puts a 10-90 band at (low, high) around their geometric middle; None when the band is not usable."""
    if low is None or high is None or low <= 0 or high <= low:
        return None
    return float((np.log(high) - np.log(low)) / (2 * 1.2816))


def expected_abs_margin(mean: float, sd: float) -> float:
    """E|X| for X ~ N(mean, sd): the folded normal mean."""
    from scipy.stats import norm
    m, sd = float(mean), float(sd)
    return sd * np.sqrt(2 / np.pi) * np.exp(-m ** 2 / (2 * sd ** 2)) + m * (1 - 2 * norm.cdf(-m / sd))


def scratch_probability(status: Optional[str], play_ewm: Optional[float] = None, mu: float = 0.0, params: Optional[dict] = None,
                        salary: Optional[int] = None, starter: Optional[bool] = None) -> float:
    """
    P(does not play) beyond what the allocation already knows: the DK injury
    status (Q, GTD, D, P), else for a bench player without a status the
    coach's-decision rate by salary (scratch_bench). The trailing DNP rate is
    not added: the allocation's baseline is already minutes x play rate.
    """
    P = params or load_params()
    st = (status or "").upper()
    if st in P.get("scratch", {}):
        return float(P["scratch"][st])
    if starter or salary is None or mu < 6:
        return 0.0
    sb = P.get("scratch_bench") or {}
    for cut in sorted(int(k) for k in sb if k != "else"):
        if int(salary) <= cut:
            return float(sb[str(cut)])
    return float(sb.get("else", 0.0))


# ----------------------------------------------------------------------------- fitting
def team_games_frame(box: pd.DataFrame, hist_rows: pd.DataFrame) -> pd.DataFrame:
    """
    One row per player-game with what the fit needs: minutes, the pre-game
    baseline (the state after his previous game: min_ewm x play_ewm), starter
    flag, the team's final margin and overtime periods.
    """
    played = box[box["played"]].copy()
    pts = played.groupby(["game_id", "team"])["points"].sum().reset_index()
    opp = pts.merge(pts, on="game_id", suffixes=("", "_opp"))
    opp = opp[opp.team != opp.team_opp]
    opp["margin"] = opp.points - opp.points_opp
    mins = played.groupby(["game_id", "team"])["minutes"].sum().reset_index().rename(columns={"minutes": "team_minutes"})
    opp = opp.merge(mins, on=["game_id", "team"])
    opp["overtime"] = np.clip(np.round((opp.team_minutes - REGULATION) / OT_MINUTES), 0, 4).astype(int)
    h = hist_rows.sort_values(["player_name", "game_date", "game_id"]).copy()
    h["prev_min_ewm"] = h.groupby("player_name")["min_ewm"].shift(1)
    h["prev_play_ewm"] = h.groupby("player_name")["play_ewm"].shift(1)
    h["prev_games"] = h.groupby("player_name")["games"].shift(1)
    df = played.merge(h[["player_name", "game_id", "prev_min_ewm", "prev_play_ewm", "prev_games"]], on=["player_name", "game_id"], how="left")
    df = df.merge(opp[["game_id", "team", "margin", "overtime", "team_minutes"]], on=["game_id", "team"], how="left")
    return df


def allocated_frame(box: pd.DataFrame, hist: "HistoryState", start, end) -> pd.DataFrame:
    """
    The real allocator's minutes per player-game (projection.walkforward: oracle
    availability and starters), joined to the actuals and the game's margin and
    overtime: what the fit should measure the share noise against.
    """
    from .engine import ProjectionParams
    from .walkforward import build_cases, run
    cases = build_cases(box, start, end)
    res = run(cases, hist, ProjectionParams(n_sims=0), simulate=False)
    played = box[box["played"]]
    pts = played.groupby(["game_id", "team"])["points"].sum().reset_index()
    opp = pts.merge(pts, on="game_id", suffixes=("", "_opp")); opp = opp[opp.team != opp.team_opp]
    opp["margin"] = opp.points - opp.points_opp
    mins = played.groupby(["game_id", "team"])["minutes"].sum().reset_index().rename(columns={"minutes": "team_minutes"})
    opp = opp.merge(mins, on=["game_id", "team"])
    opp["overtime"] = np.clip(np.round((opp.team_minutes - REGULATION) / OT_MINUTES), 0, 4).astype(int)
    df = res.merge(opp[["game_id", "team", "margin", "overtime", "team_minutes"]], on=["game_id", "team"], how="left")
    df = df.rename(columns={"minutes": "alloc", "minutes_act": "minutes"})
    df["game_date"] = df["date"]
    return df


def fit_team_minutes(box: pd.DataFrame, hist_rows: pd.DataFrame, min_games: int = 5, alloc: Optional[pd.DataFrame] = None) -> dict:
    """
    tau and beta per role from last season: the log of actual over allocated
    minutes by role, against |final margin|. `alloc` (allocated_frame) uses the
    real allocator; without it the stand-in is the pre-game baseline scaled so
    the team's baselines sum to its minutes.
    """
    if alloc is not None:
        df = alloc[(alloc.games >= min_games) & alloc.margin.notna() & (alloc.alloc > 0)].copy()
        df["role"] = [role_of(float(b), bool(s)) for b, s in zip(df.baseline, df.starter_flag)]
    else:
        df = team_games_frame(box, hist_rows)
        df = df[(df.prev_games >= min_games) & df.prev_min_ewm.notna() & df.margin.notna()].copy()
        df["base"] = df.prev_min_ewm * df.prev_play_ewm.clip(0.3, 1.0)
        tot = df.groupby(["game_id", "team"])["base"].transform("sum")
        df["alloc"] = df.base / tot * df.team_minutes
        df["role"] = [role_of(a, s > 0) for a, s in zip(df.alloc, df.starter)]
    df["lr"] = np.log(np.maximum(df.minutes, 0.5)) - np.log(np.maximum(df.alloc, 0.5))
    df["abs_margin"] = df.margin.abs()
    out = {"tau": {}, "beta": {}, "alpha": {}, "n": {}}
    for r in ROLES:
        g = df[df.role == r]
        if len(g) < 50:
            out["tau"][r] = DEFAULTS["tau"][r]; out["beta"][r] = DEFAULTS["beta"][r]; out["alpha"][r] = 0.0; out["n"][r] = int(len(g)); continue
        X = np.column_stack([np.ones(len(g)), g.abs_margin.values])
        coef, *_ = np.linalg.lstsq(X, g.lr.values, rcond=None)
        resid = g.lr.values - X @ coef
        # tau as the scale that puts 80% of the residuals inside the 10-90 band (the raw sd is inflated by the
        # baseline's own error, which the mean carries, and by the fat tails of garbage-time minutes)
        tau = float(np.percentile(np.abs(resid), 80) / 1.2816)
        out["alpha"][r] = float(coef[0]); out["beta"][r] = float(coef[1]); out["tau"][r] = tau; out["tau_sd"] = out.get("tau_sd", {})
        out["tau_sd"][r] = float(resid.std()); out["n"][r] = int(len(g))
    tg = df.drop_duplicates(["game_id", "team"])
    out["p_ot"] = float((tg.overtime > 0).mean())
    out["margin_sd_unconditional"] = float(tg.margin.std())
    out["margin_sd"] = DEFAULTS["margin_sd"]
    out["scratch"] = DEFAULTS["scratch"]
    out["fit_rows"] = int(len(df))
    return out


def save_params(params: dict, path=PARAMS_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=1))


# ----------------------------------------------------------------------------- evaluation
def load_historical_lines(path) -> Dict[tuple, float]:
    """(date, team) -> the team's own spread, from scripts/nba/fetch_historical_lines.py's CSV."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    out: Dict[tuple, float] = {}
    for r in df.itertuples():
        if pd.isna(r.spread_home):
            continue
        out[(r.date, r.home_abbr)] = float(r.spread_home); out[(r.date, r.away_abbr)] = -float(r.spread_home)
    return out


def evaluate(box: pd.DataFrame, hist_rows: pd.DataFrame, params: dict, start, end, n: int = 400, seed: int = 0,
             independent_sd=None, alloc: Optional[pd.DataFrame] = None, spreads: Optional[Dict[tuple, float]] = None) -> pd.DataFrame:
    """
    Walk-forward over team-games in [start, end]: for every player who played,
    the team draw's mean, 10th and 90th percentile of his minutes against the
    actual, with oracle availability and starters and the real margin's sign
    unknown (spread 0 unless given). independent_sd(mu) gives the current
    sampler's sd for the same player for a side-by-side coverage check.
    Returns one row per player-game.
    """
    rng = np.random.default_rng(seed)
    if alloc is not None:
        df = alloc[(alloc.game_date >= start) & (alloc.game_date <= end) & alloc.margin.notna()].copy()
    else:
        df = team_games_frame(box, hist_rows)
        df = df[(df.game_date >= start) & (df.game_date <= end) & df.prev_min_ewm.notna() & df.margin.notna()].copy()
        df["base"] = df.prev_min_ewm * df.prev_play_ewm.clip(0.3, 1.0)
    rows = []
    for (gid, team), g in df.groupby(["game_id", "team"]):
        if alloc is not None:
            alloc_m = g.alloc; roles = [role_of(float(b), bool(s)) for b, s in zip(g.baseline, g.starter_flag)]
        else:
            alloc_m = g.base / g.base.sum() * REGULATION; roles = [role_of(a, s > 0) for a, s in zip(alloc_m, g.starter)]
        alloc = alloc_m
        day = g.game_date.iloc[0]
        spread = (spreads or {}).get((day, team))
        draw = sample_team_minutes(alloc.values, roles, n, rng, spread=spread if spread is not None else 0.0, params=params)
        q10, q50, q90 = np.percentile(draw.minutes, [10, 50, 90], axis=0)
        mean = draw.minutes.mean(axis=0)
        for i, r in enumerate(g.itertuples()):
            rec = dict(game_id=gid, team=team, player_name=r.player_name, role=roles[i], alloc=float(alloc.iloc[i]), actual=float(r.minutes),
                       mean=float(mean[i]), q10=float(q10[i]), q50=float(q50[i]), q90=float(q90[i]), margin=float(r.margin),
                       spread=spread)
            if independent_sd is not None:
                sd = float(independent_sd(alloc.iloc[i]))
                rec["ind_q10"] = max(alloc.iloc[i] - 1.2816 * sd, 0.0); rec["ind_q90"] = alloc.iloc[i] + 1.2816 * sd
            rows.append(rec)
    return pd.DataFrame(rows)


def coverage_table(ev: pd.DataFrame) -> pd.DataFrame:
    t = []
    for r in ROLES + ("all",):
        g = ev if r == "all" else ev[ev.role == r]
        if not len(g):
            continue
        rec = dict(role=r, n=len(g), mae_alloc=(g.alloc - g.actual).abs().mean(), mae_mean=(g["mean"] - g.actual).abs().mean(),
                   bias_mean=(g["mean"] - g.actual).mean(), cover_team=((g.actual >= g.q10) & (g.actual <= g.q90)).mean(),
                   width_team=(g.q90 - g.q10).mean())
        if "ind_q10" in g:
            rec["cover_ind"] = ((g.actual >= g.ind_q10) & (g.actual <= g.ind_q90)).mean(); rec["width_ind"] = (g.ind_q90 - g.ind_q10).mean()
        t.append(rec)
    return pd.DataFrame(t).set_index("role")
