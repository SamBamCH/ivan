from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Union
from datetime import datetime
import random
from math import ceil

import numpy as np
import pulp as plp
from sqlalchemy import func
from sqlalchemy.orm import Session

from ivan.core.utils import get_config_value
from ivan.sports.nfl.models import NFLPlayer, Lineup
import re

# ----- slot configs for both formats -----
_CLASSIC_SLOT_COUNTS  = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1, "FLEX": 1}
_SHOWDOWN_SLOT_COUNTS = {"CPT": 1, "FLEX": 5}
_FLEX_ELIGIBLE        = {"RB", "WR", "TE"}  # classic flex eligibility

PositionType = Union[str, Tuple[str, ...]]

def _safe_var_name(name: str) -> str:
    """Make a solver-safe var name: letters/digits/underscore, and never start with a digit."""
    s = re.sub(r'[^A-Za-z0-9_]+', '_', str(name).strip())
    if not s or s[0].isdigit():
        s = 'x_' + s
    # optional: keep names short to avoid super long lines
    return s[:80]

# --- helpers ---
def _as_bases(position: PositionType) -> Tuple[str, ...]:
    return position if isinstance(position, tuple) else (position,)

def _slot_list_for_player_classic(p: FieldPlayer) -> List[str]:
    bases = _as_bases(p.position)
    allowed = set(_CLASSIC_SLOT_COUNTS.keys())
    slots = [s for s in bases if s in allowed]
    if any(bp in _FLEX_ELIGIBLE for bp in bases) and "FLEX" in allowed:
        slots.append("FLEX")
    return slots

def _eligible_slots_for_player(
    p: FieldPlayer,
    fmt: str,
    roster_slot_by_id: Optional[Dict[int, int]] = None
) -> List[str]:
    """
    Classic → derive from positions (+FLEX if eligible).
    Showdown → use DK roster_slot_id: 511 → CPT, 512 → FLEX.
    """
    if fmt == "classic":
        return _slot_list_for_player_classic(p)
    if fmt == "showdown":
        if not roster_slot_by_id:
            return []
        rs = int(roster_slot_by_id.get(p.id, 0) or 0)
        if rs == 511:
            return ["CPT"]
        if rs == 512:
            return ["FLEX"]
        return []
    return []




@dataclass(frozen=True)
class FieldPlayer:
    id: int
    name: str
    salary: int
    team: str
    opponent: Optional[str]
    position: PositionType  # was Tuple[str, ...]
    fpts_projection: Optional[float]  # used only for optional min-projection constraint



class Constraints:
    """
    Shared constraints for Classic & Showdown.
    Build vars as (player, slot) → x_{p,slot} in your solver code, then call:
        Constraints(...).add_constraints("classic", ...)
    or:
        Constraints(...).add_constraints("showdown", ...)
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

    # ---- common helpers ----
    def _vars_for_slot(self, slot: str) -> List[plp.LpVariable]:
        return [self.vars[(p, slot)] for p in self.players if (p, slot) in self.vars]

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

    # ---- atomic constraints (format-agnostic) ----
    def _add_salary_cap(self, *, cap: int, min_salary: int = 0, min_spend: Optional[int] = None) -> None:
        expr = plp.lpSum(
            p.salary * self.vars[(p, slot)]
            for p in self.players
            for slot in list({s for (_, s) in self.vars.keys()})
            if (p, slot) in self.vars
        )
        self.problem += (expr <= cap), "Max_Salary"
        floor = min_spend if (min_spend is not None) else min_salary
        if floor and floor > 0:
            self.problem += (expr >= floor), "Min_Salary"

    def _add_exact_slot_counts(self, slot_counts: Dict[str, int]) -> None:
        for slot, need in slot_counts.items():
            self.problem += (plp.lpSum(self._vars_for_slot(slot)) == need, f"Slot_{slot}")
        total_slots = sum(slot_counts.values())
        self.problem += (plp.lpSum(v for v in self.vars.values()) == total_slots, "TotalRosterSize")

    def _add_single_use_by_player_id(self) -> None:
        # Each (real) classic player used in at most one slot (QB/RB/WR/TE/DST/FLEX).
        by_player: Dict[int, List[plp.LpVariable]] = {}
        for (p, slot), v in self.vars.items():
            by_player.setdefault(p.id, []).append(v)
        for pid, vs in by_player.items():
            self.problem += (plp.lpSum(vs) <= 1, f"SingleUse_{pid}")

    def _add_no_duplicate_real_player(self) -> None:
        """
        For Showdown: prevent picking both (Captain row) and (Flex row) for same real person.
        Group by (name, team).
        """
        groups: Dict[Tuple[str, str], List[plp.LpVariable]] = {}
        for (p, slot), v in self.vars.items():
            groups.setdefault((p.name, p.team or ""), []).append(v)
        for key, vs in groups.items():
            if len(vs) > 1:
                tag = _safe_var_name(f"{key[0]}_{key[1]}")
                self.problem += (plp.lpSum(vs) <= 1, f"NoDupRealPlayer_{tag}")

    def _add_max_players_per_team(self, limit: Optional[int]) -> None:
        if not limit:
            return
        teams = {p.team for p in self.players if p.team}
        for t in teams:
            self.problem += (plp.lpSum(self._vars_for_team(t)) <= int(limit), f"MaxPlayers_{t}")

    def _add_qb_stack_constraint(self, need: int) -> None:
        """
        Classic only: at least `need` WR/TE (incl. FLEX) from the QB’s team.
        """
        if need <= 0:
            return
        teams = {p.team for p in self.players if p.team}
        y = plp.LpVariable.dicts("QBTeam", list(teams), lowBound=0, upBound=1, cat=plp.LpBinary)
        # Link: y_t >= any QB from t
        for t in teams:
            qb_vars_t = [self.vars[(p, "QB")] for p in self.players if (p, "QB") in self.vars and p.team == t]
            if not qb_vars_t:
                self.problem += (y[t] == 0, f"NoQB_{t}")
                continue
            for x in qb_vars_t:
                self.problem += (y[t] >= x, f"LinkQB_{t}_{x.name}")
        # Exactly one QB team
        self.problem += (plp.lpSum(y[t] for t in teams) == 1, "ExactlyOneQBTeam")
        # Receivers on that team >= need * y_t
        for t in teams:
            rec_vars_t = self._receiver_vars_for_team(t)
            if rec_vars_t:
                self.problem += (plp.lpSum(rec_vars_t) >= need * y[t], f"QBStack_{t}")

    # ---- public entrypoint ----
    def add_constraints(
        self,
        fmt: str,
        *,
        slot_counts: Dict[str, int],
        cap: int = 50000,
        min_salary: int = 0,
        min_spend: Optional[int] = None,
        max_players_per_team: Optional[int] = None,
        qb_stack_min: int = 0,
        dedupe_real_players: bool = True,
    ) -> None:
        """Dispatch + compose shared constraints."""
        # Shared
        self._add_salary_cap(cap=cap, min_salary=min_salary, min_spend=min_spend)
        self._add_exact_slot_counts(slot_counts)
        self._add_max_players_per_team(max_players_per_team)

        if fmt == "classic":
            self._add_single_use_by_player_id()
            self._add_qb_stack_constraint(qb_stack_min)
        elif fmt == "showdown":
            # Showdown rows are separate CPT/FLEX IDs → guard duplicates of same real player.
            if dedupe_real_players:
                self._add_no_duplicate_real_player()
        else:
            raise ValueError(f"Unknown format: {fmt}")
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

def generate_single_lineup_from_samples(
    players: List[FieldPlayer],
    sample_values: Dict[int, float],
    config: dict | None = None,
    debug_lp_path: Optional[str] = None,
) -> Optional[List[int]]:
    config  = config or {}
    verbose = bool(get_config_value(config, "verbose", False))
    prob    = plp.LpProblem("nfl_field_lineup", plp.LpMaximize)

    # (player, slot) vars
    lp_vars: Dict[Tuple[FieldPlayer, str], plp.LpVariable] = {}
    for p in players:
        base = _safe_var_name(p.name)
        for slot in _eligible_slots_for_player(p, "classic"):
            lp_vars[(p, slot)] = plp.LpVariable(f"{base}_{slot}_{p.id}", cat=plp.LpBinary)

    # objective = sum of sampled fpts
    prob += plp.lpSum(
        sample_values.get(p.id, 0.0) * lp_vars[(p, slot)]
        for p in players
        for slot in _eligible_slots_for_player(p, "classic")
        if (p, slot) in lp_vars
    ), "TotalSampledFPTS"

    # optional min-projection guard (uses p.fpts_projection)
    min_proj = get_config_value(config, "lineup.min_fpts_threshold", None)
    if min_proj is not None:
        prob += plp.lpSum(
            (p.fpts_projection or 0.0) * lp_vars[(p, slot)]
            for p in players
            for slot in _eligible_slots_for_player(p, "classic")
            if (p, slot) in lp_vars
        ) >= float(min_proj), "MinProjectionSum"

    # constraints (shared)
    Constraints(prob, players, lp_vars, config, verbose).add_constraints(
        "classic",
        slot_counts=_CLASSIC_SLOT_COUNTS,
        cap=int(get_config_value(config, "lineup.salary_cap", 50000)),
        min_salary=int(get_config_value(config, "lineup.min_salary", 0)),
        max_players_per_team=get_config_value(config, "lineup.max_players_per_team", None),
        qb_stack_min=int(get_config_value(config, "lineup.qb_stack_min", 0) or 0),
    )

    solver = plp.GLPK_CMD(
        path=r"D:\ivan\glpk-4.65\w64\glpsol.exe",
        keepFiles=True,
        options=["--log", r"data\nfl/logs/glpsol.log"]
    )
    status_code = prob.solve(solver)
    if plp.LpStatus[status_code] != "Optimal":
        if debug_lp_path:
            prob.writeLP(debug_lp_path)
        return None

    chosen_ids = [p.id for (p, slot), var in lp_vars.items() if var.varValue == 1]
    chosen_ids = list(dict.fromkeys(chosen_ids))
    return chosen_ids if len(chosen_ids) == sum(_CLASSIC_SLOT_COUNTS.values()) else None


def generate_single_showdown_lineup(
    players: List[FieldPlayer],
    roster_slot_by_id: Dict[int, int],
    base_scores: Dict[int, float],
    *,
    jitter: float = 0.10,
    cap: int = 50000,
    min_cap_drop: int = 0,
    max_cap_drop: int = 1200,
    debug_lp_path: Optional[str] = None,
) -> Optional[List[int]]:
    prob  = plp.LpProblem("dk_showdown", plp.LpMaximize)

    # sanity: make sure we have at least 1 CPT id and 5 FLEX ids
    cpt_ids  = [p.id for p in players if roster_slot_by_id.get(p.id) == 511]
    flex_ids = [p.id for p in players if roster_slot_by_id.get(p.id) == 512]
    if not cpt_ids or len(flex_ids) < 5:
        return None

    # (player, slot) vars (CPT or FLEX per row id)
    lp_vars: Dict[Tuple[FieldPlayer, str], plp.LpVariable] = {}
    for p in players:
        for slot in _eligible_slots_for_player(p, "showdown", roster_slot_by_id):
            lp_vars[(p, slot)] = plp.LpVariable(f"{_safe_var_name(p.name)}_{slot}_{p.id}", cat=plp.LpBinary)

    # salaries already DK-boosted for CPT rows in your DB
    # Objective: lightly jittered base scores
    import random as _rnd
    prob += (
        plp.lpSum((base_scores.get(p.id, 0.0) * (1.0 + _rnd.uniform(-jitter, jitter))) * lp_vars[(p, slot)]
                  for p in players
                  for slot in _eligible_slots_for_player(p, "showdown", roster_slot_by_id)
                  if (p, slot) in lp_vars)
    ), "NoisyPoints"

    # randomized near-cap floor for THIS lineup
    min_spend = cap - _rnd.randint(min_cap_drop, max_cap_drop)

    # constraints (shared)
    Constraints(prob, players, lp_vars, config={}, verbose=False).add_constraints(
        "showdown",
        slot_counts=_SHOWDOWN_SLOT_COUNTS,  # Exactly one CPT & five FLEX
        cap=cap,
        min_spend=min_spend,                # randomized floor
        dedupe_real_players=True,           # no dup of same (name, team)
    )

    solver = plp.GLPK_CMD(
        path=r"D:\ivan\glpk-4.65\w64\glpsol.exe",
        keepFiles=True,
        options=["--log", r"data\nfl/logs/glpsol.log"]
    )
    status = prob.solve(solver)
    if plp.LpStatus[status] != "Optimal":
        if debug_lp_path:
            prob.writeLP(debug_lp_path)
        return None

    chosen_ids = [p.id for (p, slot), var in lp_vars.items() if var.varValue == 1]
    return chosen_ids if len(chosen_ids) == 6 else None


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
    max_attempts = max(num_lineups * 1.2, num_lineups + 50)

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