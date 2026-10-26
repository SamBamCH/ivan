"""
project_team(): roster + availability + history -> per-player projection.

Inputs are plain records so the same code serves the live slate (NBAPlayer
rows, Rotowire starters, a manual minutes CSV) and the walk-forward backtest
(actual availability and starters from the box score).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .fpts import simulate_fpts, summarize, minutes_sd
from .history import HistoryState, STATS
from .minutes import MinutesInputs, allocate_team_minutes, baseline_minutes, salary_weight
from .rates import shrunk_rates, fpts_per_minute, apply_usage_bump, USAGE_RECOVERY
from .market import blend_rates
from .team_minutes import load_params as load_team_params, role_of, sample_team_minutes, scratch_probability, tau_from_band

MINUTES_MODELS = ("independent", "team", "rotation")


@dataclass
class RosterPlayer:
    name: str
    available: bool
    starter: bool | None = None       # confirmed starter; None = unknown
    salary: int | None = None
    minutes_override: float | None = None
    market_rates: dict[str, float] | None = None   # stat -> per-minute rate from the props
    rate_mult: float = 1.0                         # e.g. conditional "without X" usage
    status: str | None = None                      # DK status (Q, GTD, ...): the team model's scratch probability
    fpm_override: float | None = None              # hand-set fantasy points per minute (every stat rate scaled to match)
    new_team: bool = False                         # his history is on another team
    games_on_team: int = 0                         # games for this team this season
    games_this_season: int = 0
    minutes_low: float | None = None               # hand-set 10-90 minutes band (team model: the player's own noise scale)
    minutes_high: float | None = None
    position: str | None = None                    # DK eligibility ("PF/C"): the minutes solve's position group
    market_means: dict[str, float] | None = None   # stat -> the books' consensus mean: the solve's book-implied minutes anchor


@dataclass
class ProjectionParams:
    start_shift: float = 6.0
    alloc: str = "headroom"
    cut_tau: float | None = 8.0
    use_play_prob: bool = True
    recovery: float = USAGE_RECOVERY
    prior_minutes: float = 60.0
    n_sims: int = 4000
    dispersion: dict | None = None
    market_weight: float = 1.0
    quantiles: int = 101          # points of the marginal kept for the correlated sampler
    minutes_model: str = "independent"   # "independent": each player's minutes a clipped normal; "team": the joint team draw
    rebalance: bool = False              # True: a pin re-allocates the rest of the team to 241; False: the others keep the model's
                                         # minutes and the difference is left for a person to place (the desk shows it)
    rotation: dict | None = None         # minutes_model "rotation": {"grid": DataFrame, "spreads": {name: sd}, "curve": DataFrame|None}
    team_profile: dict | None = None     # the coach's pattern (team_profile.profile): starter / bench minutes for the role prior, opening spans for the grid
    salary_blend: bool = True            # blend the minutes baseline toward the salary curve (minutes.salary_weight)
    spread: float | None = None          # the team's own line for the team draw (negative = favoured); None = pick'em
    team_params: dict | None = None      # team_minutes.load_params() when None
    group_targets: dict | None = None    # alloc "solve": position group -> (target minutes, sd) for this team (groups.group_targets)

    @classmethod
    def from_config(cls, config: dict | None = None, **overrides) -> "ProjectionParams":
        """projection.* from config/nba.json (minutes_model, n_sims), then keyword overrides."""
        if config is None:
            from ivan.core.config import load_config
            try:
                config = load_config("nba")
            except Exception:   # noqa: BLE001 - no config file is a plain default
                config = {}
        block = (config or {}).get("projection") or {}
        kw = {k: block[k] for k in ("minutes_model", "n_sims", "market_weight", "recovery", "rebalance", "salary_blend", "alloc") if k in block}
        kw.update(overrides)
        return cls(**kw)


MAX_MINUTES = 48.0          # a world's minutes for one player (overtime is added by the game draw downstream)
GRID_SHAPE_MIN = 0.5        # the grid's own mean must be at least this share of the allocation for its shape to be used


def fit_draw_to_mean(draw: np.ndarray, a_min: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    """
    World-level minutes for one player: the grid's draw rescaled to the allocated mean when the grid describes him
    (its mean is at least GRID_SHAPE_MIN of the allocation); otherwise an independent draw around the allocation.
    Rescaling a grid that barely holds him (2026-10-23: a player pinned at 15 minutes but absent from a drawn grid
    came out 0 to 61 minutes, 65% of worlds at zero, and the sim loved the tail) is never allowed.
    """
    draw = np.asarray(draw, dtype=float); n = draw.shape[0]
    if a_min <= 0:
        return np.zeros(n)
    mu = float(draw.mean())
    if mu >= GRID_SHAPE_MIN * a_min:
        return np.clip(draw * (a_min / mu), 0.0, MAX_MINUTES)
    return np.clip(rng.normal(a_min, max(sd, 1.0), n), 0.0, MAX_MINUTES)


def solve_allocation(avail, recs, state, history, params):
    """The solve's anchors from the roster: (alloc frame like allocate_team_minutes', {name: book minutes})."""
    from .book_minutes import book_minutes
    from .groups import group_of
    from .solve import Anchor, BOOK_SD, DEEP_SD, EDIT_SD, HIST_SD, NOHIST_SD, solve
    anchors, book = [], {}
    for p in avail:
        rec = recs[p.name]; base = float(rec["baseline"]); grp = group_of(p.position or "")
        bm = book_minutes(p.market_means or {}, state.loc[p.name] if p.name in state.index else None, history.league_rates) if p.market_means else None
        if p.minutes_override is not None:
            anchors.append(Anchor(p.name, float(p.minutes_override), EDIT_SD, grp, source="edit"))
        elif bm is not None and base >= 8:
            anchors.append(Anchor(p.name, float(bm["minutes"]), BOOK_SD, grp, source="book")); book[p.name] = float(bm["minutes"])
        else:
            sd = NOHIST_SD if rec["games"] == 0 else (HIST_SD if base >= 12 else DEEP_SD)
            anchors.append(Anchor(p.name, max(0.0, base), sd, grp, source="history" if base >= 8 else "deep"))
    sol = solve(anchors, params.group_targets or {})
    names = [p.name for p in avail]
    return (pd.DataFrame({"baseline": [recs[n]["baseline"] for n in names], "minutes": [sol.minutes[n] for n in names],
                          "pinned": [next(p for p in avail if p.name == n).minutes_override is not None for n in names]},
                         index=pd.Index(names, name="player_name")), book)


def project_team(
    roster: list[RosterPlayer],
    state: pd.DataFrame,              # HistoryState.before(date), indexed by player_name
    history: HistoryState,
    team_fpts: float | None = None,   # trailing team production; league average when None
    params: ProjectionParams | None = None,
    rng: np.random.Generator | None = None,
    simulate: bool = True,
) -> pd.DataFrame:
    """
    Returns one row per *available* player: minutes, per-stat means, fpts_mean,
    fpts_std, plus baseline minutes and the usage multiplier for inspection.
    Absent players are listed only so the caller can see who was excluded;
    the usage bump works from the team's trailing production, not from them.
    """
    params = params or ProjectionParams()
    rng = rng or np.random.default_rng(0)

    def srow(name):
        return state.loc[name] if name in state.index else None

    # per-player rates and baseline minutes for everyone on the roster
    recs = {}
    for p in roster:
        s = srow(p.name)
        rates = shrunk_rates(s, history.league_rates, params.prior_minutes)
        rates = blend_rates(rates, p.market_rates, params.market_weight)
        if p.rate_mult != 1.0:
            rates = {k: v * p.rate_mult for k, v in rates.items()}
        mi = MinutesInputs(
            name=p.name,
            min_ewm=None if s is None else float(s["min_ewm"]),
            start_ewm=0.0 if s is None else float(s["start_ewm"]),
            starter=p.starter,
            play_prob=1.0 if (s is None or not params.use_play_prob) else float(s["play_ewm"]),
            salary=p.salary, minutes_override=p.minutes_override,
            salary_w=0.0 if not params.salary_blend else salary_weight(0 if s is None else int(s["games"]), p.new_team, p.games_on_team, p.games_this_season),
        )
        recs[p.name] = dict(inputs=mi, rates=rates, fpm=fpts_per_minute(rates),
                            priced=bool(p.market_rates),
                            baseline=baseline_minutes(mi, params.start_shift,
                                                      team_starter_min=(params.team_profile or {}).get("starter_min"), team_bench_min=(params.team_profile or {}).get("bench_min")),
                            hist_sd=None if s is None else float(s["min_ewm_sd"]),
                            games=0 if s is None else int(s["games"]),
                            available=p.available)

    avail = [p for p in roster if p.available]
    absent = [p for p in roster if not p.available]
    budget = None
    if params.alloc == "solve":
        # one solve per team: the books' implied minutes anchor priced players, history the rest, a hand pin is held almost
        # exactly, the team sums to its budget within the position groups (projection/solve.py; docs/nba_minutes_model_plan.md)
        alloc, book = solve_allocation(avail, recs, state, history, params)
        for n, bm in book.items():
            recs[n]["book_minutes"] = bm
    elif not params.rebalance and any(p.minutes_override is not None for p in avail):
        # hand-set minutes do not move anyone else: the others keep the model's own allocation, every player is then
        # fixed, and the team total is whatever that adds to (the desk shows the minutes still to place)
        base = allocate_team_minutes([replace(recs[p.name]["inputs"], minutes_override=None) for p in avail],
                                     start_shift=params.start_shift, alloc=params.alloc, cut_tau=params.cut_tau)
        for p in avail:
            if p.minutes_override is None:
                recs[p.name]["inputs"] = replace(recs[p.name]["inputs"], minutes_override=float(base.loc[p.name, "minutes"]))
                recs[p.name]["soft_pin"] = True
        budget = float(sum(recs[p.name]["inputs"].minutes_override for p in avail))
    if params.alloc != "solve":
        alloc = allocate_team_minutes([recs[p.name]["inputs"] for p in avail],
                                      start_shift=params.start_shift, alloc=params.alloc,
                                      cut_tau=params.cut_tau)
    zero_minutes: list[str] = []
    if params.minutes_model == "rotation" and params.rotation is not None and params.rotation.get("skeleton"):
        # the grid IS the projection: the coach's slots (the roster's stint curves) filled with tonight's players at the
        # solve's minutes, locked rows held (projection/skeleton.py). Each player's minutes are his row; a player the
        # schedule leaves off the floor has none and leaves the projection (the desk adds him back by painting a row).
        from .skeleton import apply_depth, build_grid, choose_rotation, minutes_of
        from .groups import group_of
        rot = params.rotation
        names = list(alloc.index)
        locks = rot.get("locks")
        targets = {n: float(alloc.loc[n, "minutes"]) for n in names}
        if rot.get("depth") or rot.get("pool") is not None:
            # the coach plays about `depth` players: the deep bench's small expected minutes (honest on average, not on a
            # night) fold into the rotation. Locked rows, hand pins and named starters always stay. The kept players are
            # re-solved through their anchors (books, history, position budgets), so the freed minutes land where the
            # anchors are loosest rather than pro rata on the stars.
            keep = [p.name for p in avail if p.minutes_override is not None or p.starter] + ([n for n in locks.index] if locks is not None else [])
            # the pool rule (rot["pool"]: the last ordinary game's rotation minus the outs plus the returners, live.rotation_for)
            # names the rotation outright; the depth cut picks it by the targets
            chosen = ((set(rot["pool"]) & set(names)) | set(keep)) if rot.get("pool") is not None else choose_rotation(targets, int(rot["depth"]), keep)
            if params.alloc == "solve" and len(chosen) < len(names):
                kept_alloc, _ = solve_allocation([p for p in avail if p.name in chosen], recs, state, history, params)
                targets = {n: (float(kept_alloc.loc[n, "minutes"]) if n in kept_alloc.index else 0.0) for n in names}
            else:
                targets = {n: (t if n in chosen else 0.0) for n, t in targets.items()} if rot.get("pool") is not None else apply_depth(targets, int(rot["depth"]), keep)
            for n in names:
                alloc.loc[n, "minutes"] = targets[n]
        grid = build_grid(rot.get("curve"), targets, names,
                          starters=[p.name for p in avail if p.starter], groups={p.name: group_of(p.position or "") for p in roster},
                          locks=locks, vacated=rot.get("vacated"), shares=rot.get("shares"), profile=params.team_profile)
        rot["grid"] = grid; rot["targets"] = dict(targets)
        gm = minutes_of(grid)
        alloc["target"] = alloc["minutes"]
        alloc["minutes"] = [gm.get(n, 0.0) for n in alloc.index]
        if locks is not None and len(locks):
            alloc["pinned"] = [bool(alloc.loc[n, "pinned"]) or n in locks.index for n in alloc.index]
        zero_minutes = [n for n in alloc.index if alloc.loc[n, "minutes"] <= 0]
        alloc = alloc[alloc["minutes"] > 0]
        avail = [p for p in avail if p.name in alloc.index]
        rot["zero_minutes"] = zero_minutes
    if params.alloc == "solve":
        # a priced player's total is the books': his per-minute rate is the market mean over his solved minutes
        for p in avail:
            if p.market_means and recs[p.name]["priced"]:
                mins = float(alloc.loc[p.name, "minutes"])
                if mins > 0:
                    r = dict(recs[p.name]["rates"])
                    for st, mean in p.market_means.items():
                        if st in r and mean is not None and not (isinstance(mean, float) and np.isnan(mean)):
                            r[st] = float(mean) / mins
                    recs[p.name]["rates"] = r; recs[p.name]["fpm"] = fpts_per_minute(r)
    if budget is not None:
        alloc["pinned"] = [not recs[n].get("soft_pin") for n in alloc.index]        # only the hand pins count as pins
    alloc["fpm"] = [recs[n]["fpm"] for n in alloc.index]
    alloc["priced"] = [recs[n]["priced"] for n in alloc.index]
    team_total = history.league_team_fpts if team_fpts is None else float(team_fpts)
    bump = apply_usage_bump(alloc, team_total, params.recovery)

    # the team draw: every available player's minutes in every world, jointly (team model only)
    if params.minutes_model not in MINUTES_MODELS:
        raise ValueError(f"unknown minutes_model {params.minutes_model!r}; expected one of {MINUTES_MODELS}")
    by_name = {p.name: p for p in avail}
    team_params = params.team_params or (load_team_params() if params.minutes_model in ("team", "rotation") else None)
    roles, p_out = {}, {}
    for name in alloc.index:
        p = by_name[name]
        st = recs[name]["inputs"]
        mu = float(alloc.loc[name, "minutes"])
        roles[name] = role_of(float(recs[name]["baseline"]), p.starter)
        p_out[name] = 0.0 if (alloc.loc[name, "pinned"] or p.starter) else scratch_probability(p.status, st.play_prob if params.use_play_prob else None, mu, team_params or {"scratch": {}},
                                                                                                   salary=p.salary, starter=p.starter)
    draw = None
    if params.minutes_model == "rotation" and params.rotation is not None and simulate and params.n_sims > 0 and len(alloc):
        from .rotation import SPREAD_MULT, default_spread, sample_worlds
        rot = params.rotation; grid = rot["grid"]
        names = list(alloc.index)
        taus = {n: tau_from_band(by_name[n].minutes_low, by_name[n].minutes_high) for n in names}
        spreads = dict(rot.get("spreads") or {})
        for n in names:                                                   # wider where the history is weaker
            cls = "no_history" if recs[n]["games"] == 0 else ("new_team" if by_name[n].new_team else "same_team")
            lvl = "star" if float(alloc.loc[n, "minutes"]) >= 28 else "starter" if float(alloc.loc[n, "minutes"]) >= 18 else "rotation" if float(alloc.loc[n, "minutes"]) >= 9 else "deep"
            spreads[n] = spreads.get(n, default_spread(float(alloc.loc[n, "minutes"]))) * SPREAD_MULT.get(cls, 1.0) * SPREAD_MULT.get(lvl, 1.0)
        for n, t in taus.items():                                        # a hand band sets the player's own spread
            if t is not None:
                spreads[n] = float(t) * float(alloc.loc[n, "minutes"]) / 1.2816 / np.sqrt(2.0)
        d = sample_worlds(grid, spreads, params.n_sims, rng, spread=params.spread, p_out={n: p_out[n] for n in names if n in grid.index},
                          curve=rot.get("curve"), margin_sd=float((team_params or {}).get("margin_sd", 14.75)), p_ot=float((team_params or {}).get("p_ot", 0.052)))
        M = np.zeros((params.n_sims, len(names)))
        col = {nm: j for j, nm in enumerate(d["names"])}
        for j, n in enumerate(names):
            a_min = float(alloc.loc[n, "minutes"]) * (1.0 - float(p_out.get(n, 0.0)))   # the mean includes the scratch worlds
            if n in col:
                # the allocation is the mean (it already averages over blowouts); the grid gives the shape and the
                # world-to-world redistribution. A drawn grid is its own mean (the pins hold it).
                M[:, j] = fit_draw_to_mean(d["minutes"][:, col[n]], a_min, minutes_sd(a_min, recs[n]["hist_sd"]), rng)
            else:
                M[:, j] = np.clip(rng.normal(a_min, minutes_sd(a_min, recs[n]["hist_sd"]), params.n_sims), 0.0, MAX_MINUTES) if a_min > 0 else 0.0
        class _D:  # the TeamDraw shape the loop below reads
            minutes = M
        draw = _D()
    elif params.minutes_model == "team" and simulate and params.n_sims > 0 and len(alloc):
        names = list(alloc.index)
        # a pin sets the mean (the allocation already holds it); a band sets the player's own spread; nobody is frozen
        taus = [tau_from_band(by_name[n].minutes_low, by_name[n].minutes_high) for n in names]
        draw = sample_team_minutes(alloc["minutes"].values, [roles[n] for n in names], params.n_sims, rng, spread=params.spread,
                                   p_out=[p_out[n] for n in names], params=team_params, tau_override=taus, budget=budget)

    rows = []
    for j, name in enumerate(alloc.index):
        r = recs[name]
        mins = float(alloc.loc[name, "minutes"])
        rates = {s: r["rates"][s] * float(bump[name]) for s in STATS}
        fpm = r["fpm"] * float(bump[name])
        fpm_override = by_name[name].fpm_override
        if fpm_override is not None and fpm > 0:
            rates = {s: v * float(fpm_override) / fpm for s, v in rates.items()}; fpm = float(fpm_override)
        rec = {"player_name": name, "minutes": mins, "baseline": r["baseline"],
               "pinned": bool(alloc.loc[name, "pinned"]), "usage_mult": float(bump[name]),
               "games": r["games"], "fpm": fpm, "priced": r["priced"], "fpm_pinned": fpm_override is not None,
               "band_pinned": tau_from_band(by_name[name].minutes_low, by_name[name].minutes_high) is not None,
               "role": roles[name], "p_out": p_out[name], "rates": rates}
        for s in STATS:
            rec[f"proj_{s}"] = mins * rates[s]
        if simulate and mins > 0 and params.n_sims > 0:
            if draw is not None:
                samples = simulate_fpts(mins, rates, n=params.n_sims, dispersion=params.dispersion, rng=rng, minutes=draw.minutes[:, j])
                rec["minutes_p10"], rec["minutes_p90"] = (float(x) for x in np.percentile(draw.minutes[:, j], [10, 90]))
            else:
                samples = simulate_fpts(mins, rates, n=params.n_sims,
                                        min_sd=minutes_sd(mins, r["hist_sd"]),
                                        dispersion=params.dispersion, rng=rng)
            rec["fpts_mean"], rec["fpts_std"] = summarize(samples)
            rec["quantiles"] = np.quantile(samples, np.linspace(0, 1, params.quantiles)).round(2).tolist()
        else:
            rec["fpts_mean"] = mins * rec["fpm"]
            rec["fpts_std"] = float("nan")
            rec["quantiles"] = None
        rows.append(rec)
    return pd.DataFrame(rows).set_index("player_name")
