"""
Core lineup generation logic for NBA Classic contests.

This module provides:
- FieldPlayer dataclass for representing players in optimization
- ROSTER_STRUCTURE, SLOT_NAMES, SLOT_ELIGIBILITY for DK Classic format
- generate_single_classic_lineup for ILP-based lineup optimization
"""

from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import os
import time
import re

import pulp as plp
from pulp import PulpSolverError
from ivan.sports.nba.paths import LOGS_DIR


# === EXPLICIT ROSTER STRUCTURE DEFINITION ===
ROSTER_STRUCTURE = [
    ("PG", "Point Guard", {"PG"}),
    ("SG", "Shooting Guard", {"SG"}),
    ("SF", "Small Forward", {"SF"}),
    ("PF", "Power Forward", {"PF"}),
    ("C", "Center", {"C"}),
    ("G", "Guard", {"PG", "SG"}),
    ("F", "Forward", {"SF", "PF"}),
    ("UTIL", "Utility", {"PG", "SG", "SF", "PF", "C"}),
]

SLOT_NAMES = [slot[0] for slot in ROSTER_STRUCTURE]
SLOT_ELIGIBILITY = {slot[0]: slot[2] for slot in ROSTER_STRUCTURE}


@dataclass(frozen=True)
class FieldPlayer:
    """Immutable representation of a player for lineup optimization."""
    id: int
    name: str
    salary: int
    team: str
    opponent: Optional[str]
    position: Union[str, Tuple[str, ...]]
    fpts_mean: Optional[float]
    game_id: Optional[str] = None
    gametime: Optional[datetime] = None


def _safe_var_name(name: str) -> str:
    """Make a solver-safe var name: letters/digits/underscore, never starting with a digit."""
    s = re.sub(r'[^A-Za-z0-9_]+', '_', str(name).strip())
    if not s or s[0].isdigit():
        s = 'x_' + s
    return s[:80]


def _unique_problem_name(base: str = "nba_classic_lineup") -> str:
    """Unique name to prevent PuLP temp/solution file collisions across processes."""
    return f"{base}_{os.getpid()}_{int(time.time()*1e6)}_{uuid4().hex[:6]}"


def _cfg(config: dict, path: str, default):
    """Quiet dotted lookup with default (no warnings)."""
    cur = config or {}
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _make_glpk_solver(config: dict):
    """Prefer GLPK if available; fall back to CBC."""
    import shutil
    keep_files = bool(_cfg(config, "solver.keep_files", False))
    msg = bool(_cfg(config, "solver.msg", False))

    path = (_cfg(config, "solver.glpk_path", None)
            or os.environ.get("GLPK_PATH")
            or shutil.which("glpsol"))

    if path and os.path.exists(path):
        return plp.GLPK_CMD(path=path, keepFiles=keep_files, msg=msg)

    if shutil.which("glpsol"):
        return plp.GLPK_CMD(keepFiles=keep_files, msg=msg)

    return plp.PULP_CBC_CMD(msg=msg)


def _resolve_lp_path(path: str, status: str) -> str:
    """Convert directory path to unique file path, or return file path as-is."""
    p = Path(path)
    
    if path.endswith('/') or (p.exists() and p.is_dir()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lineup_{status}_{timestamp}_{uuid4().hex[:8]}.lp"
        return str(p / filename)
    
    return path


def generate_single_classic_lineup(
    players: List[FieldPlayer],
    base_scores: Dict[int, float],
    config: dict | None = None,
    debug_lp_path: Optional[str] = None,
    locked_assignments: Optional[Dict[int, str]] = None,
    attempt_tag: Optional[str] = None,
    lock_time: Optional[datetime] = None,
    min_fpts: Optional[float] = None,
    max_salary_per_team: Optional[int] = 25000
) -> Optional[List[int]]:
    """
    Generate a single optimal NBA classic lineup given player pool and projected scores.

    Roster: PG, SG, SF, PF, C, G, F, UTIL (8 players total)
    Constraints:
      - Slot fill: exactly one eligible player per roster slot
      - Uniqueness: each player can appear at most once
      - Max total salary (config: max_salary or salary_cap)
      - Optional per-team max salary (config: max_salary_per_team)
      - Optional locked player-to-slot assignments
      - Lock time enforcement: players whose games started before lock_time are excluded
      - Optional minimum total projected FPTS (min_fpts)
      - Objective: maximize total projected fantasy points

    Args:
        players: List of FieldPlayer objects to choose from
        base_scores: Dict mapping player_id -> projected fantasy points
        config: Configuration dict for solver and constraints
        debug_lp_path: Optional path or directory to write LP problem snapshots
        locked_assignments: Optional Dict[player_id: int, slot_name: str] to lock players to specific slots
        lock_time: Optional datetime; players with gametime <= lock_time are excluded unless locked
        min_fpts: Optional minimum total projected FPTS for the lineup

    Returns:
        List of 8 player IDs if optimal solution found, None otherwise
    """
    config = config or {}
    
    if not debug_lp_path:
        debug_lp_path = str(LOGS_DIR) + "/"
    
    try:
        _p = Path(debug_lp_path)
        if str(debug_lp_path).endswith('/') or (_p.exists() and _p.is_dir()):
            _p.mkdir(parents=True, exist_ok=True)
        else:
            _p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    
    def _cfg_val(key: str, default=None):
        """Lookup config with fallback locations."""
        for path in [
            ("classic", "lineup_generation", key),
            ("lineup_generation", key),
            ("classic", key),
            (key,),
        ]:
            cur = config
            for k in path[:-1]:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    cur = None
                    break
            if cur is not None and isinstance(cur, dict) and path[-1] in cur:
                return cur[path[-1]]
        return default
    
    salary_cap = int(_cfg_val("max_salary", _cfg_val("salary_cap", 50000)))
    max_salary_per_team = max_salary_per_team
    try:
        max_salary_per_team_val = int(max_salary_per_team) if max_salary_per_team is not None else None
    except (TypeError, ValueError):
        max_salary_per_team_val = None
    
    # Create binary decision variables for each (player, slot) pair
    x: Dict[int, Dict[str, plp.LpVariable]] = {}
    
    locked_player_ids = set(locked_assignments.keys()) if locked_assignments else set()
    
    for player in players:
        x[player.id] = {}
        
        # Lock time check
        if lock_time is not None and player.gametime is not None:
            if player.gametime <= lock_time and player.id not in locked_player_ids:
                continue
        
        player_positions = player.position
        if isinstance(player_positions, str):
            player_positions = (player_positions,)
        elif not isinstance(player_positions, (tuple, list)):
            player_positions = (player_positions,) if player_positions else ()
        
        for slot_name in SLOT_NAMES:
            if not any(pos in SLOT_ELIGIBILITY[slot_name] for pos in player_positions):
                continue
            
            var_name = f"x_{_safe_var_name(player.name)}_slot{slot_name}_{player.id}"
            x[player.id][slot_name] = plp.LpVariable(var_name, 0, 1, plp.LpBinary)
    
    prob = plp.LpProblem(_unique_problem_name("nba_classic_lineup"), plp.LpMaximize)

    min_distinct_games = 2
    
    # Each slot must have exactly one player
    for slot_name in SLOT_NAMES:
        locked_player_id = None
        if locked_assignments:
            for pid, slot in locked_assignments.items():
                if slot == slot_name:
                    locked_player_id = pid
                    break
        
        if locked_player_id is not None:
            if locked_player_id in x and slot_name in x[locked_player_id]:
                prob += x[locked_player_id][slot_name] == 1, f"LockedSlot_{slot_name}_{locked_player_id}"
            else:
                raise ValueError(
                    f"Locked player {locked_player_id} is not eligible for slot {slot_name}."
                )
        else:
            eligible_vars = [
                x[p.id][slot_name]
                for p in players
                if slot_name in x[p.id]
            ]
            if eligible_vars:
                prob += plp.lpSum(eligible_vars) == 1, f"Slot_{slot_name}"
            else:
                raise ValueError(
                    f"CRITICAL: No variables created for slot '{slot_name}'. "
                    f"Required positions: {SLOT_ELIGIBILITY[slot_name]}."
                )
    
    # Each player can only be selected once
    for player in players:
        slot_vars = [x[player.id][slot] for slot in x[player.id]]
        if slot_vars:
            prob += plp.lpSum(slot_vars) <= 1, f"PlayerOnce_{player.id}"
    
    # Exactly 8 unique players
    y: Dict[int, plp.LpVariable] = {}
    for player in players:
        if player.id in x and x[player.id]:
            y[player.id] = plp.LpVariable(f"y_selected_{player.id}", 0, 1, plp.LpBinary)
            slot_sum = plp.lpSum([x[player.id][slot] for slot in x[player.id]])
            prob += y[player.id] == slot_sum, f"LinkSelected_{player.id}"
    
    if y:
        prob += plp.lpSum(y.values()) == 8, "ExactlyEightPlayers"
    
    # Salary constraint
    salary_expr = plp.lpSum(
        player.salary * x[player.id][slot]
        for player in players
        if player.id in x
        for slot in x[player.id]
    )
    prob += salary_expr <= salary_cap, "MaxSalary"

    # Optional minimum FPTS constraint (uses player fpts_mean, not jittered scores)
    if min_fpts is not None:
        fpts_expr = plp.lpSum(
            (player.fpts_mean or 0.0) * x[player.id][slot]
            for player in players
            if player.id in x
            for slot in x[player.id]
        )
        prob += fpts_expr >= min_fpts, "MinFPTS"

    # Optional per-team salary cap
    if max_salary_per_team_val is not None:
        team_to_player_ids: Dict[str, List[int]] = {}
        for p in players:
            team_key = (p.team or "").upper().strip()
            if not team_key:
                continue
            team_to_player_ids.setdefault(team_key, []).append(p.id)

        for team_key, pids in team_to_player_ids.items():
            team_salary = plp.lpSum(
                next((pp.salary for pp in players if pp.id == pid), 0) * x[pid][slot]
                for pid in pids
                if pid in x
                for slot in x[pid]
            )
            prob += team_salary <= max_salary_per_team_val, f"TeamMaxSalary_{_safe_var_name(team_key)}"
    
    # Objective - maximize total projected fantasy points
    score = plp.lpSum(
        base_scores.get(player.id, 0.0) * x[player.id][slot]
        for player in players
        if player.id in x
        for slot in x[player.id]
    )
    prob += score

    # Optional distinct games constraint
    try:
        game_ids = {}
        for p in players:
            if p.id in x and x[p.id] and p.game_id:
                game_ids.setdefault(p.game_id, []).append(p.id)
        if game_ids and min_distinct_games and min_distinct_games > 1:
            z: Dict[str, plp.LpVariable] = {g: plp.LpVariable(f"game_used_{_safe_var_name(g)}", 0, 1, plp.LpBinary) for g in game_ids}
            for g, pids in game_ids.items():
                prob += z[g] <= plp.lpSum(y[pid] for pid in pids if pid in y), f"GameLink_{_safe_var_name(g)}"
            prob += plp.lpSum(z.values()) >= min_distinct_games, "MinDistinctGames"
    except Exception:
        pass
    
    # Solve
    solver = _make_glpk_solver(config)
    try:
        status_code = prob.solve(solver)
    except PulpSolverError:
        if debug_lp_path:
            lp_file = _resolve_lp_path(debug_lp_path, "solver_error")
            Path(lp_file).parent.mkdir(parents=True, exist_ok=True)
            prob.writeLP(lp_file)
        raise
    
    if plp.LpStatus[status_code] != "Optimal":
        if debug_lp_path:
            status_name = plp.LpStatus.get(status_code, "unknown")
            tag = attempt_tag or "na"
            lp_file = _resolve_lp_path(debug_lp_path, f"attempt{tag}_{status_name.lower()}")
            Path(lp_file).parent.mkdir(parents=True, exist_ok=True)
            prob.writeLP(lp_file)
        return None
    
    # Extract selected player IDs
    selected = [
        player.id
        for player in players
        if player.id in x
        for slot in x[player.id]
        if x[player.id][slot].varValue == 1
    ]

    # Validate locked assignments
    if locked_assignments:
        missing_locked = set(locked_assignments.keys()) - set(selected)
        if missing_locked:
            return None

    return selected if len(selected) == 8 else None

