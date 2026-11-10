"""
project_team(): roster + availability + history -> the mapping, per player (layer 1 of the projection).

The mapping is what the game simulation reads: each player's minutes (the solve within position groups, then the
rotation skeleton: his row of the 48-minute grid), his per-minute rates (the books' means over his minutes where priced,
history elsewhere, the usage bump, the desk's pins), his scratch probability and role. The frame's fpts_mean is the
plan's total (minutes x fantasy points per minute): a yardstick, not the projection. The projection, every player's
fantasy points, minutes and stats as a distribution, is layer 2: the World drawn on this mapping (projection.worlds).

Inputs are plain records so the same code serves the live slate (NBAPlayer rows, Rotowire starters, a manual minutes
CSV) and the backtest's replay (actual availability and starters from the box score).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .history import HistoryState, STATS
from .minutes import MinutesInputs, allocate_team_minutes, baseline_minutes, salary_weight
from .rates import shrunk_rates, fpts_per_minute, apply_usage_bump, USAGE_RECOVERY
from .market import blend_rates
from .team_minutes import load_params as load_team_params, role_of, scratch_probability, tau_from_band
from . import cuts as CUTS

MINUTES_MODELS = ("rotation",)


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
    stat_overrides: dict[str, float] | None = None  # hand-set totals for the game, stat -> total: that stat's rate becomes total / minutes
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
    market_weight: float = 1.0
    minutes_model: str = "rotation"      # the rotation grid's worlds give the minutes band; a player the grid does not hold gets a clipped normal
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
        """projection.* from config/nba.json (minutes_model, market_weight, ...), then keyword overrides."""
        if config is None:
            from ivan.core.config import load_config
            try:
                config = load_config("nba")
            except Exception:   # noqa: BLE001 - no config file is a plain default
                config = {}
        block = (config or {}).get("projection") or {}
        kw = {k: block[k] for k in ("minutes_model", "market_weight", "recovery", "rebalance", "salary_blend", "alloc") if k in block}
        kw.update(overrides)
        return cls(**kw)


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
) -> pd.DataFrame:
    """
    Returns one row per *available* player: minutes (the grid row's), the per-minute rates after the pins and the
    usage bump, proj_<stat> (minutes x rate: the plan's total per stat), fpts_mean (the plan's fantasy total), the
    scratch probability and role, plus baseline minutes and the usage multiplier for inspection. Absent players are
    listed only so the caller can see who was excluded; the usage bump works from the team's trailing production, not
    from them. `rng` is unused since the draw moved to the World (kept for the callers' signature).
    """
    params = params or ProjectionParams()

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
        # exactly, the team sums to its budget within the position groups (projection/solve.py; docs/nba/notes/minutes_model_plan.md)
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
        # the soft pool's fringe: left off the floor by the plan (the depth cut) or dropped by hand (a 0-minute pin makes him unavailable
        # to the plan; his status says he is not hurt), not the players ruled out
        from .live import OUT_STATUSES
        fringe = [p for p in avail if p.name in zero_minutes] + [p for p in roster if not p.available and (p.status or "").upper() not in OUT_STATUSES and p.name in recs]
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

    # who plays: every projected player's role and scratch probability (the World reads p_out; a pin or a named starter plays)
    if params.minutes_model not in MINUTES_MODELS:
        raise ValueError(f"unknown minutes_model {params.minutes_model!r}; expected one of {MINUTES_MODELS}")
    by_name = {p.name: p for p in avail}
    team_params = params.team_params or load_team_params()
    cut_table = CUTS.load_table(); cut_mode = CUTS.cuts_mode()
    roles, p_out = {}, {}
    for name in alloc.index:
        p = by_name[name]
        st = recs[name]["inputs"]
        mu = float(alloc.loc[name, "minutes"])
        roles[name] = role_of(float(recs[name]["baseline"]), p.starter)
        p_out[name] = 0.0 if (alloc.loc[name, "pinned"] or p.starter) else scratch_probability(p.status, st.play_prob if params.use_play_prob else None, mu, team_params or {"scratch": {}},
                                                                                                   salary=p.salary, starter=p.starter)

    rows = []
    for name in alloc.index:
        r = recs[name]
        mins = float(alloc.loc[name, "minutes"])
        rates = {s: r["rates"][s] * float(bump[name]) for s in STATS}
        fpm = r["fpm"] * float(bump[name])
        fpm_override = by_name[name].fpm_override
        if fpm_override is not None and fpm > 0:
            rates = {s: v * float(fpm_override) / fpm for s, v in rates.items()}; fpm = float(fpm_override)
        stat_pins = {s: float(v) for s, v in (by_name[name].stat_overrides or {}).items() if s in rates and v is not None}
        if stat_pins and mins > 0:
            # a typed total is his for the game whatever his minutes: that stat's rate is total / minutes, the others stay
            rates = {**rates, **{s: v / mins for s, v in stat_pins.items()}}; fpm = fpts_per_minute(rates)
        s_row = srow(name)
        own = None if s_row is None or "cut10" not in s_row.index else float(s_row["cut10"])
        held = by_name[name].minutes_override is not None               # a typed minutes pin; a locked row holds his minutes when he plays, not whether he does
        rec = {"player_name": name, "minutes": mins, "baseline": r["baseline"],
               "pinned": bool(alloc.loc[name, "pinned"]), "usage_mult": float(bump[name]),
               "games": r["games"], "fpm": fpm, "priced": r["priced"], "fpm_pinned": fpm_override is not None,
               "band_pinned": tau_from_band(by_name[name].minutes_low, by_name[name].minutes_high) is not None,
               "stat_pinned": sorted(stat_pins),
               "role": roles[name], "p_out": p_out[name], "rates": rates, "fringe": False,
               "p_cut": CUTS.cut_probability(cut_table, mins, own, starter=bool(by_name[name].starter), held=held, status=by_name[name].status, mode=cut_mode),
               "cut10": own, "fringe_p": 0.0, "fringe_minutes": 0.0,
               "minutes_sd": None if s_row is None or "min_sd20" not in s_row.index or pd.isna(s_row["min_sd20"]) else float(s_row["min_sd20"])}
        for s in STATS:
            rec[f"proj_{s}"] = mins * rates[s]
        rec["fpts_mean"] = mins * fpm                 # the plan's total; the distribution is the World's (projection.worlds)
        rows.append(rec)
    # the fringe: available players the plan left at zero. They stay in the frame at 0 minutes with their history rates and the
    # chance the night uses them anyway (projection.cuts.fringe_profile), so the World can hand them the minutes real nights do
    for p in (fringe if params.minutes_model == "rotation" and params.rotation is not None and params.rotation.get("skeleton") else []):
        if cut_mode == "off":
            break
        r = recs[p.name]; s_row = srow(p.name)
        own = None if s_row is None or "cut10" not in s_row.index else float(s_row["cut10"])
        fp, fm, _ = CUTS.fringe_profile(cut_table, own)
        rates = {s: r["rates"][s] for s in STATS}
        rec = {"player_name": p.name, "minutes": 0.0, "baseline": r["baseline"], "pinned": False, "usage_mult": 1.0, "games": r["games"], "fpm": r["fpm"],
               "priced": r["priced"], "fpm_pinned": False, "band_pinned": False, "stat_pinned": [], "role": "deep", "p_out": 0.0, "rates": rates, "fringe": True,
               "p_cut": 0.0, "cut10": own, "fringe_p": fp, "fringe_minutes": fm, "minutes_sd": None, "fpts_mean": 0.0}
        for s in STATS:
            rec[f"proj_{s}"] = 0.0
        rows.append(rec)
    return pd.DataFrame(rows).set_index("player_name")
