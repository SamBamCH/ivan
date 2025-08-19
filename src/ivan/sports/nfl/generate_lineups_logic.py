from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Union
from datetime import datetime
import random

import numpy as np
import pulp as plp
from sqlalchemy import func
from sqlalchemy.orm import Session

from ivan.core.utils import get_config_value
from ivan.sports.nfl.models import NFLPlayer, Lineup
import re

# ----- Roster & position config (DK Classic) -----
# final roster = 9 slots: QB, RB, RB, WR, WR, WR, TE, DST, FLEX (RB/WR/TE)
_DK_SLOT_COUNTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1, "FLEX": 1}
_FLEX_ELIGIBLE  = {"RB", "WR", "TE"}

PositionType = Union[str, Tuple[str, ...]]

def _safe_var_name(name: str) -> str:
    """Make a solver-safe var name: letters/digits/underscore, and never start with a digit."""
    s = re.sub(r'[^A-Za-z0-9_]+', '_', str(name).strip())
    if not s or s[0].isdigit():
        s = 'x_' + s
    # optional: keep names short to avoid super long lines
    return s[:80]




@dataclass(frozen=True)
class FieldPlayer:
    id: int
    name: str
    salary: int
    team: str
    opponent: Optional[str]
    position: PositionType  # was Tuple[str, ...]
    fpts_projection: Optional[float]  # used only for optional min-projection constraint


class NFLConstraints:
    """
    Mirror of your MLB Constraints class, adapted for NFL + FLEX.
    Builds constraints over binary variables keyed by (player, slot).
    """
    def __init__(
        self,
        problem: plp.LpProblem,
        players: List[FieldPlayer],
        lp_variables: Dict[Tuple[FieldPlayer, str], plp.LpVariable],
        config: dict,
        verbose: bool = False,
    ) -> None:
        self.problem = problem
        self.players = players
        self.vars    = lp_variables
        self.config  = config
        self.verbose = verbose

    # ---- helpers to collect var lists ----
    def _vars_for_slot(self, slot: str) -> List[plp.LpVariable]:
        return [self.vars[(p, slot)]
                for p in self.players
                if (p, slot) in self.vars]

    def _vars_for_team(self, team: str) -> List[plp.LpVariable]:
        return [v for (p, slot), v in self.vars.items() if p.team == team]

    def _receiver_vars_for_team(self, team: str) -> List[plp.LpVariable]:
        out = []
        for p in self.players:
            if p.team != team:
                continue
            bases = _as_bases(p.position)
            if not any(bp in {"WR", "TE"} for bp in bases):
                continue
            for slot in ("WR", "TE", "FLEX"):
                if (p, slot) in self.vars:
                    out.append(self.vars[(p, slot)])
        return out

    # ---- constraints ----
    def _add_salary_cap(self) -> None:
        max_sal = 50000
        min_sal = get_config_value(self.config, "lineup.min_salary", 0)
        expr = plp.lpSum(
            p.salary * self.vars[(p, slot)]
            for p in self.players
            for slot in _slot_list_for_player(p)
            if (p, slot) in self.vars
        )
        self.problem += (expr <= max_sal), "Max_Salary"
        self.problem += (expr >= min_sal), "Min_Salary"

    def _add_roster_counts(self) -> None:
        # exact count per required slot type
        for slot, need in _DK_SLOT_COUNTS.items():
            self.problem += (plp.lpSum(self._vars_for_slot(slot)) == need, f"Slot_{slot}")

        # total roster size
        total_slots = sum(_DK_SLOT_COUNTS.values())
        self.problem += (
            plp.lpSum(v for v in self.vars.values()) == total_slots,
            "TotalRosterSize"
        )

    def _add_single_use(self) -> None:
        # Each player can be used in at most one slot (QB/RB/WR/TE/DST/FLEX).
        by_player: Dict[int, List[plp.LpVariable]] = {}
        for (p, slot), v in self.vars.items():
            by_player.setdefault(p.id, []).append(v)
        for pid, vs in by_player.items():
            self.problem += (plp.lpSum(vs) <= 1, f"SingleUse_{pid}")

    def _add_max_players_per_team(self) -> None:
        limit = get_config_value(self.config, "lineup.max_players_per_team", None)
        if not limit:
            return
        teams = {p.team for p in self.players if p.team}
        for t in teams:
            self.problem += (
                plp.lpSum(self._vars_for_team(t)) <= int(limit),
                f"MaxPlayers_{t}"
            )

    def _add_qb_stack_constraint(self) -> None:
        """
        Require at least N WR/TE (including FLEX) from the QB's team.
        Enabled when lineup.qb_stack_min >= 1.
        """
        need = int(get_config_value(self.config, "lineup.qb_stack_min", 0) or 0)
        if need <= 0:
            return

        teams = {p.team for p in self.players if p.team}
        # y_t = 1 iff the QB from team t is selected.
        y = plp.LpVariable.dicts("QBTeam", list(teams), lowBound=0, upBound=1, cat=plp.LpBinary)

        # Link y_t to all QB vars from team t: y_t >= x_qb for each QB x_qb in t
        for t in teams:
            qb_vars_t = [self.vars[(p, "QB")] for p in self.players if (p, "QB") in self.vars and p.team == t]
            if not qb_vars_t:
                self.problem += (y[t] == 0, f"NoQB_{t}")
                continue
            for x in qb_vars_t:
                self.problem += (y[t] >= x, f"LinkQB_{t}_{x.name}")

        # Exactly one QB overall → exactly one y_t will be 1
        self.problem += (plp.lpSum(y[t] for t in teams) == 1, "ExactlyOneQBTeam")

        # Stack: receivers on the QB's team >= need * y_t
        for t in teams:
            rec_vars_t = self._receiver_vars_for_team(t)
            if rec_vars_t:
                self.problem += (
                    plp.lpSum(rec_vars_t) >= need * y[t],
                    f"QBStack_{t}"
                )

    def add_all(self) -> None:
        self._add_salary_cap()
        self._add_roster_counts()
        self._add_single_use()
        # self._add_max_players_per_team()
        # self._add_qb_stack_constraint()


# ---- helpers ----

def _coerce_positions(pos: Iterable[str] | str) -> PositionType:
    """
    Normalize position(s):
      - "WR"            -> "WR"
      - "WR/TE"         -> ("TE", "WR")
      - ["RB","WR"]     -> ("RB","WR")
      - " DST ,  "      -> "DST"
    """
    if isinstance(pos, str):
        tokens = [p.strip().upper() for p in re.split(r"[\/,]", pos) if p.strip()]
    else:
        tokens = [str(p).strip().upper() for p in pos if str(p).strip()]

    uniq = sorted(set(tokens))
    if not uniq:
        return tuple()  # or return "" if you prefer a sentinel
    if len(uniq) == 1:
        return uniq[0]  # <-- single position returns string
    return tuple(uniq)

def _as_bases(position: PositionType) -> Tuple[str, ...]:
    return position if isinstance(position, tuple) else (position,)

def _slot_list_for_player(p: FieldPlayer) -> List[str]:
    bases = _as_bases(p.position)
    allowed = set(_DK_SLOT_COUNTS.keys())
    slots = [s for s in bases if s in allowed]
    if any(bp in _FLEX_ELIGIBLE for bp in bases) and "FLEX" in allowed:
        slots.append("FLEX")
    return slots



# ---- public API ----

def generate_single_lineup_from_samples(
    players: List[FieldPlayer],
    sample_values: Dict[int, float],
    config: dict | None = None,
    debug_lp_path: Optional[str] = None,
) -> Optional[List[int]]:
    """
    Build & solve a single DK Classic lineup using per-player sampled fpts.

    Returns list of selected player IDs (length 9) or None if infeasible.
    """
    config  = config or {}
    verbose = bool(get_config_value(config, "verbose", False))
    prob    = plp.LpProblem("nfl_field_lineup", plp.LpMaximize)

    # Create binary decision variables for each (player, eligible_slot).
    lp_vars: Dict[Tuple[FieldPlayer, str], plp.LpVariable] = {}
    for p in players:
        base = _safe_var_name(p.name)
        for slot in _slot_list_for_player(p):
            var_name = f"{base}_{slot}_{p.id}"
            lp_vars[(p, slot)] = plp.LpVariable(var_name, cat=plp.LpBinary)

    # Objective: maximize sum of sampled fpts
    prob += plp.lpSum(
        sample_values.get(p.id, 0.0) * lp_vars[(p, slot)]
        for p in players
        for slot in _slot_list_for_player(p)
    ), "TotalSampledFPTS"

    # Optional: minimum projection guard (uses p.fpts_projection, not sample)
    min_proj = get_config_value(config, "lineup.min_fpts_threshold", None)
    if min_proj is not None:
        proj_expr = plp.lpSum(
            (p.fpts_projection or 0.0) * lp_vars[(p, slot)]
            for p in players
            for slot in _slot_list_for_player(p)
        )
        prob += (proj_expr >= float(min_proj)), "MinProjectionSum"

    # Constraints
    NFLConstraints(prob, players, lp_vars, config, verbose).add_all()

    # Solve with GLPK
    solver = plp.GLPK_CMD(
    path=r"D:\ivan\glpk-4.65\w64\glpsol.exe",  # adjust if your exe lives elsewhere
    keepFiles=True,                             # keep the .lp/.sol temp files
    options=["--log", r"data\nfl/logs/glpsol.log"]  # write a GLPK log file
    )
    status_code = prob.solve(solver)
    status = plp.LpStatus[status_code]

    if status != "Optimal":
        if debug_lp_path:
            prob.writeLP(debug_lp_path)
        return None

    chosen_ids = [
        p.id for (p, slot), var in lp_vars.items()
        if var.varValue == 1
    ]
    # De-duplicate in case multiple slots could theoretically turn on (shouldn't happen with SingleUse)
    chosen_ids = list(dict.fromkeys(chosen_ids))
    return chosen_ids if len(chosen_ids) == sum(_DK_SLOT_COUNTS.values()) else None

def generate_single_lineup_from_scores(
    players: List[FieldPlayer],
    scores: Dict[int, float],
    config: dict | None = None,
    debug_lp_path: Optional[str] = None,
) -> Optional[List[int]]:
    """
    Classic lineup with a deterministic score vector (e.g., medians).
    """
    config  = config or {}
    verbose = bool(get_config_value(config, "verbose", False))
    prob    = plp.LpProblem("nfl_field_lineup_median", plp.LpMaximize)

    lp_vars: Dict[Tuple[FieldPlayer, str], plp.LpVariable] = {}
    for p in players:
        base = _safe_var_name(p.name)
        for slot in _slot_list_for_player(p):
            lp_vars[(p, slot)] = plp.LpVariable(f"{base}_{slot}_{p.id}", cat=plp.LpBinary)

    # Objective: maximize sum of medians + tiny salary tiebreak (keeps builds near cap)
    salary_expr = plp.lpSum(
        p.salary * lp_vars[(p, slot)]
        for p in players for slot in _slot_list_for_player(p)
    )
    score_expr = plp.lpSum(
        (scores.get(p.id, 0.0)) * lp_vars[(p, slot)]
        for p in players for slot in _slot_list_for_player(p)
    )
    prob += score_expr + 1e-4 * (salary_expr / 50000.0), "MedianPlusSalaryTiebreak"

    # Optional: min projection guard
    min_proj = get_config_value(config, "lineup.min_fpts_threshold", None)
    if min_proj is not None:
        proj_expr = plp.lpSum(
            (p.fpts_projection or 0.0) * lp_vars[(p, slot)]
            for p in players for slot in _slot_list_for_player(p)
        )
        prob += (proj_expr >= float(min_proj)), "MinProjectionSum"

    NFLConstraints(prob, players, lp_vars, config, verbose).add_all()

    solver = plp.GLPK_CMD(path=r"D:\ivan\glpk-4.65\w64\glpsol.exe", keepFiles=True,
                          options=["--log", r"data\nfl/logs/glpsol.log"])
    status_code = prob.solve(solver)
    if plp.LpStatus[status_code] != "Optimal":
        if debug_lp_path:
            prob.writeLP(debug_lp_path)
        return None
    chosen_ids = [p.id for (p, slot), var in lp_vars.items() if var.varValue == 1]
    chosen_ids = list(dict.fromkeys(chosen_ids))
    return chosen_ids if len(chosen_ids) == sum(_DK_SLOT_COUNTS.values()) else None

def generate_single_showdown_lineup(
    players: List[FieldPlayer],
    roster_slot_by_id: Dict[int, int],
    base_scores: Dict[int, float],
    *,
    jitter: float = 0.10,           # ±10% randomization to "un-optimize"
    cap: int = 50000,
    min_cap_drop: int = 0,          # random salary floor distance from cap (inclusive)
    max_cap_drop: int = 1200,       # (e.g., uniform 0..1200)
    debug_lp_path: Optional[str] = None,
) -> Optional[List[int]]:
    """
    Build one DK Showdown lineup with 1 CPT (511) + 5 FLEX (512).
    """
    prob  = plp.LpProblem("dk_showdown", plp.LpMaximize)

    # Partition pool by slot id (from DK payload)
    CPT, FLEX = 511, 512
    cpt_ids  = [p.id for p in players if roster_slot_by_id.get(p.id) == CPT]
    flex_ids = [p.id for p in players if roster_slot_by_id.get(p.id) == FLEX]
    if not cpt_ids or len(flex_ids) < 5:
        return None

    xC = plp.LpVariable.dicts("xC", cpt_ids, 0, 1, plp.LpBinary)
    xF = plp.LpVariable.dicts("xF", flex_ids, 0, 1, plp.LpBinary)

    # Exactly one CPT, exactly five FLEX
    prob += plp.lpSum(xC[i] for i in cpt_ids) == 1, "ExactlyOneCaptain"
    prob += plp.lpSum(xF[i] for i in flex_ids) == 5, "ExactlyFiveFlex"

    # Mutually exclusive per real player: same name+team cannot appear twice
    by_key: Dict[tuple, Dict[str, int]] = {}
    for p in players:
        key = (p.name, p.team)
        rs  = roster_slot_by_id.get(p.id)
        if rs == CPT:
            by_key.setdefault(key, {})["cpt"] = p.id
        elif rs == FLEX:
            by_key.setdefault(key, {})["flex"] = p.id
    for key, ids in by_key.items():
        if "cpt" in ids and "flex" in ids:
            prob += xC[ids["cpt"]] + xF[ids["flex"]] <= 1, f"NoDup_{_safe_var_name(key[0])}_{key[1]}"

    # Salary: captain rows already have the DK-boosted salary in NFLPlayer
    salary = {p.id: p.salary for p in players}
    spend  = plp.lpSum(salary[i] * xC[i] for i in cpt_ids) + plp.lpSum(salary[i] * xF[i] for i in flex_ids)
    # Near-cap floor randomized per lineup
    import random as _rnd
    min_spend = cap - _rnd.randint(min_cap_drop, max_cap_drop)
    prob += spend <= cap,        "Cap"
    prob += spend >= min_spend,  "NearCapFloor"

    # Objective: lightly randomized points to avoid concentration
    # (CPT rows already include the 1.5x DK CPT projection effect via their salary; points use same base)
    # If you prefer, multiply base_scores for CPT by 1.5 here; or keep them equal and let jitter explore.
    obj = (
        plp.lpSum((base_scores.get(i, 0.0) * (1.0 + _rnd.uniform(-jitter, jitter))) * xC[i] for i in cpt_ids) +
        plp.lpSum((base_scores.get(i, 0.0) * (1.0 + _rnd.uniform(-jitter, jitter))) * xF[i] for i in flex_ids)
    )
    prob += obj, "NoisyPoints"

    solver = plp.GLPK_CMD(path=r"D:\ivan\glpk-4.65\w64\glpsol.exe", keepFiles=True,
                          options=["--log", r"data\nfl/logs/glpsol.log"])
    status = prob.solve(solver)
    if plp.LpStatus[status] != "Optimal":
        if debug_lp_path:
            prob.writeLP(debug_lp_path)
        return None

    lineup = [i for i in cpt_ids if plp.value(xC[i]) > 0.5]
    lineup += [i for i in flex_ids if plp.value(xF[i]) > 0.5]
    return lineup if len(lineup) == 6 else None

def _infer_slate_type(roster_slot_by_id: Dict[int, int]) -> str:
    """Return 'showdown' if any DK Showdown roster slots (511/512) exist; else 'classic'."""
    vals = set(roster_slot_by_id.values())
    return "showdown" if (511 in vals or 512 in vals) else "classic"

def generate_and_store_field_lineups(
    session: Session,
    samples: Dict[int, np.ndarray],
    config: dict | None = None,
    num_lineups: int = 1000,
    workers: int = 1,
) -> int:
    """
    Build up to `num_lineups` lineups by sweeping sample indices.
    - `workers` only influences commit batching: batch_size = max(1, num_lineups // workers)
    - Slate type is inferred from roster_slot_id to choose Classic vs Showdown builder.
    Saves ivan.sports.nfl.models.Lineup(player_ids_list, created_at) rows.
    Returns the count created.
    """
    config = config or {}
    # compute batch size from workers (guard for small num_lineups/workers)
    batch_size = max(1, num_lineups // max(1, workers))

    # 1) Load just the players referenced by samples
    ids = list(samples.keys())
    if not ids:
        return 0

    raw_players = (
        session.query(NFLPlayer)
               .filter(NFLPlayer.id.in_(ids))
               .all()
    )
    if not raw_players:
        return 0

    # 2) Wrap into FieldPlayer and capture roster_slot_id by id (for Showdown inference)
    players: List[FieldPlayer] = []
    roster_slot_by_id: Dict[int, int] = {}
    for rp in raw_players:
        pos = getattr(rp, "position", None)
        pos_tuple = _coerce_positions(pos or [])
        players.append(FieldPlayer(
            id=rp.id,
            name=getattr(rp, "name", str(rp.id)),
            salary=int(getattr(rp, "salary", 0) or 0),
            team=getattr(rp, "team", None) or "",
            opponent=getattr(rp, "opponent", None),
            position=pos_tuple,
            fpts_projection=float(getattr(rp, "fpts_projection", 0.0) or 0.0),
        ))
        roster_slot_by_id[rp.id] = int(getattr(rp, "roster_slot_id", 0) or 0)

    # 2b) Infer slate type
    slate_type = _infer_slate_type(roster_slot_by_id)

    # 3) Prepare to iterate over sample indices
    n_samples = len(next(iter(samples.values())))
    idx_order = list(range(n_samples))
    random.shuffle(idx_order)

    created = 0
    batch   = 0
    seen_lineups: set[tuple[int, ...]] = set()  # prevent duplicates

    # showdown config knobs (used only if slate_type == 'showdown')
    jitter       = float(get_config_value(config, "showdown.jitter", 0.10))
    cap          = int(get_config_value(config, "lineup.salary_cap", 50000))
    min_cap_drop = int(get_config_value(config, "showdown.min_cap_drop", 0))
    max_cap_drop = int(get_config_value(config, "showdown.max_cap_drop", 1200))

    # 4) Sweep sample indices, build & save lineups
    attempts = 0
    max_attempts = max(num_lineups * 10, num_lineups + 50)

    while created < num_lineups and attempts < max_attempts:
        sample_idx = idx_order[attempts % n_samples]
        attempts += 1

        # materialize this round's sampled fpts
        per_player_sample = {pid: float(vec[sample_idx]) for pid, vec in samples.items()}

        if slate_type == "showdown":
            ids_selected = generate_single_showdown_lineup(
                players=players,
                roster_slot_by_id=roster_slot_by_id,
                base_scores=per_player_sample,  # use this sample as the base score signal
                jitter=jitter,
                cap=cap,
                min_cap_drop=min_cap_drop,
                max_cap_drop=max_cap_drop,
                debug_lp_path=f"data/nfl/logs/debug_showdown_{attempts}.lp",
            )
        else:
            ids_selected = generate_single_lineup_from_samples(
                players=players,
                sample_values=per_player_sample,
                config=config,
                debug_lp_path=f"data/nfl/logs/debug_classic_{attempts}.lp",
            )

        if not ids_selected:
            continue

        key = tuple(sorted(ids_selected))
        if key in seen_lineups:
            continue
        seen_lineups.add(key)

        session.add(Lineup(
            player_ids_list=list(ids_selected),
            created_at=datetime.utcnow()
        ))
        created += 1
        batch   += 1

        if batch >= batch_size:
            session.commit()
            batch = 0

    if batch > 0:
        session.commit()

    return created