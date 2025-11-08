from sqlalchemy.orm import Session
from typing import Dict, List, Tuple, Optional, Iterable, Union, Callable
from dataclasses import dataclass
import random
from pathlib import Path
import os, time, re
from uuid import uuid4
from datetime import datetime
import pulp as plp
from pulp import PulpSolverError

from ivan.sports.nba.models import NBAPlayer, Lineup
from ivan.sports.nba.utils import _coerce_positions, _cfg
from ivan.sports.nba.generate_samples import generate_player_samples


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
    id: int
    name: str
    salary: int
    team: str
    opponent: Optional[str]
    position: Union[str, Tuple[str, ...]]
    fpts_mean: Optional[float]
    game_id: Optional[str] = None  # Unique identifier for the game (e.g., "LAL-GSW")
    gametime: Optional[datetime] = None  # Game start time for lock checking


def _safe_var_name(name: str) -> str:
    """Make a solver-safe var name: letters/digits/underscore, never starting with a digit."""
    s = re.sub(r'[^A-Za-z0-9_]+', '_', str(name).strip())
    if not s or s[0].isdigit():
        s = 'x_' + s
    return s[:80]


def _unique_problem_name(base: str = "nba_classic_lineup") -> str:
    """Unique name to prevent PuLP temp/solution file collisions across processes."""
    return f"{base}_{os.getpid()}_{int(time.time()*1e6)}_{uuid4().hex[:6]}"


def _make_glpk_solver(config: dict):
    """
    Prefer GLPK if available; fall back to CBC.
    """
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
    """
    Convert directory path to unique file path, or return file path as-is.
    
    Args:
        path: Either a directory path (ends with /) or full file path
        status: Status string to include in filename (e.g., "infeasible", "solver_error")
    
    Returns:
        Full path to LP file
    """
    p = Path(path)
    
    # If it's a directory or ends with /, generate unique filename
    if path.endswith('/') or (p.exists() and p.is_dir()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lineup_{status}_{timestamp}_{uuid4().hex[:8]}.lp"
        return str(p / filename)
    
    # Otherwise assume it's already a full file path
    return path


def generate_single_classic_lineup(
    players: List[FieldPlayer],
    base_scores: Dict[int, float],
    config: dict | None = None,
    debug_lp_path: Optional[str] = None,
    locked_assignments: Optional[Dict[int, str]] = None,
    attempt_tag: Optional[str] = None,
    lock_time: Optional[datetime] = None,
) -> Optional[List[int]]:
    """
    Generate a single optimal NBA classic lineup given player pool and projected scores.

    Roster: PG, SG, SF, PF, C, G, F, UTIL (8 players total)
    Constraints (stripped-down core):
      - Slot fill: exactly one eligible player per roster slot
      - Uniqueness: each player can appear at most once
      - Max total salary (config: max_salary or salary_cap)
      - Optional per-team max salary (config: max_salary_per_team)
      - Optional locked player-to-slot assignments
      - Lock time enforcement: players whose games started before lock_time are excluded
        (unless explicitly locked via locked_assignments)
      - Objective: maximize total projected fantasy points

    Note: Multi-game and min-salary constraints intentionally omitted for flexibility
    (pre-lock and post-lock scenarios).

    Args:
        players: List of FieldPlayer objects to choose from
        base_scores: Dict mapping player_id -> projected fantasy points
        config: Configuration dict for solver and constraints
        debug_lp_path: Optional path or directory to write LP problem snapshots
        locked_assignments: Optional Dict[player_id: int, slot_name: str] to lock players to specific slots.
                           These players will be forced into the specified slots regardless of optimization.
        lock_time: Optional datetime; players with gametime <= lock_time are excluded unless locked
    """
    config = config or {}
    # Default logs directory if none provided
    if not debug_lp_path:
        debug_lp_path = "data/nba/logs/"
    # Ensure logs directory (or parent) exists early
    try:
        _p = Path(debug_lp_path)
        if str(debug_lp_path).endswith('/') or (_p.exists() and _p.is_dir()):
            _p.mkdir(parents=True, exist_ok=True)
        else:
            _p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    
    # Config helper
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
    
    # Prefer max_salary, fall back to salary_cap
    salary_cap = int(_cfg_val("max_salary", _cfg_val("salary_cap", 50000)))
    # Optional per-team salary cap; if not set, skip team cap constraint
    max_salary_per_team = _cfg_val("max_salary_per_team", None)
    try:
        max_salary_per_team_val = int(max_salary_per_team) if max_salary_per_team is not None else None
    except (TypeError, ValueError):
        max_salary_per_team_val = None
    
    # --- STEP 1: Create binary decision variables for each (player, slot) pair ---
    # x[player_id][slot_name] = 1 if player assigned to slot, 0 otherwise
    x: Dict[int, Dict[str, plp.LpVariable]] = {}
    
    # Determine locked player IDs (those explicitly assigned to slots)
    locked_player_ids = set(locked_assignments.keys()) if locked_assignments else set()
    
    for player in players:
        x[player.id] = {}
        
        # Lock time check: skip players whose games have started UNLESS they're locked
        if lock_time is not None and player.gametime is not None:
            if player.gametime <= lock_time and player.id not in locked_player_ids:
                # Game has started and player is not locked - exclude from optimization
                continue
        
        # FIX: Ensure position is always a tuple/iterable
        player_positions = player.position
        if isinstance(player_positions, str):
            player_positions = (player_positions,)
        elif not isinstance(player_positions, (tuple, list)):
            player_positions = (player_positions,) if player_positions else ()
        
        for slot_name in SLOT_NAMES:
            # Check if player is eligible for this slot
            if not any(pos in SLOT_ELIGIBILITY[slot_name] for pos in player_positions):
                continue  # Player not eligible for this slot
            
            var_name = f"x_{_safe_var_name(player.name)}_slot{slot_name}_{player.id}"
            x[player.id][slot_name] = plp.LpVariable(var_name, 0, 1, plp.LpBinary)
    
    # (Intentionally removing multi-game grouping/constraints for a minimal core model)
    
    # --- STEP 2 & 3: Set up LP problem and slot-fill constraints ---
    prob = plp.LpProblem(_unique_problem_name("nba_classic_lineup"), plp.LpMaximize)
    
    # Each slot must have exactly one player (or be locked to a specific player)
    for slot_name in SLOT_NAMES:
        # Check if this slot is locked to a specific player
        locked_player_id = None
        if locked_assignments:
            for pid, slot in locked_assignments.items():
                if slot == slot_name:
                    locked_player_id = pid
                    break
        
        if locked_player_id is not None:
            # Slot is locked - force this player into this slot
            if locked_player_id in x and slot_name in x[locked_player_id]:
                prob += x[locked_player_id][slot_name] == 1, f"LockedSlot_{slot_name}_{locked_player_id}"
            else:
                # ERROR: locked player not eligible for this slot
                if debug_lp_path:
                    lp_file = _resolve_lp_path(debug_lp_path, f"locked_ineligible_{_safe_var_name(slot_name)}")
                    diag_file = _resolve_lp_path(debug_lp_path, f"diag_locked_ineligible_{_safe_var_name(slot_name)}")
                    if diag_file.endswith('.lp'):
                        diag_file = diag_file[:-3] + 'txt'
                    Path(lp_file).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        # LP snapshot (may be minimal if error occurred before constraints)
                        prob.writeLP(lp_file)
                        # Human-readable diagnostics
                        slot_counts = {s: sum(1 for p in players if p.id in x and s in x[p.id]) for s in SLOT_NAMES}
                        required = ','.join(sorted(SLOT_ELIGIBILITY.get(slot_name, set())))
                        sample_players = '\n'.join(
                            f"id={p.id} name={p.name} pos={p.position} team={p.team}"
                            for p in players[:25]
                        )
                        with open(diag_file, 'w') as f:
                            f.write(f"Locked ineligible slot: {slot_name} (requires: {required})\n")
                            f.write(f"Locked player id: {locked_player_id}\n\n")
                            f.write("Eligible var counts by slot:\n")
                            for s, c in slot_counts.items():
                                f.write(f"  {s}: {c}\n")
                            f.write("\nSample players (first 25):\n")
                            f.write(sample_players + "\n")
                        print(f"[DEBUG] LP written to: {lp_file}; diag: {diag_file}")
                    except Exception:
                        pass
                raise ValueError(
                    f"Locked player {locked_player_id} is not eligible for slot {slot_name}. "
                    f"Check position compatibility."
                )
        else:
            # Slot is not locked - normal constraint: exactly one player
            eligible_vars = [
                x[p.id][slot_name]
                for p in players
                if slot_name in x[p.id]
            ]
            if eligible_vars:
                prob += plp.lpSum(eligible_vars) == 1, f"Slot_{slot_name}"
            else:
                # CRITICAL ERROR: No eligible players for this slot!
                if debug_lp_path:
                    lp_file = _resolve_lp_path(debug_lp_path, f"no_vars_{_safe_var_name(slot_name)}")
                    diag_file = _resolve_lp_path(debug_lp_path, f"diag_no_vars_{_safe_var_name(slot_name)}")
                    if diag_file.endswith('.lp'):
                        diag_file = diag_file[:-3] + 'txt'
                    Path(lp_file).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        prob.writeLP(lp_file)
                        # Write human-readable diagnostics
                        slot_counts = {s: sum(1 for p in players if p.id in x and s in x[p.id]) for s in SLOT_NAMES}
                        required = ','.join(sorted(SLOT_ELIGIBILITY.get(slot_name, set())))
                        # List a few players that should be eligible by position but aren't (if any)
                        sample_players = '\n'.join(
                            f"id={p.id} name={p.name} pos={p.position} team={p.team}"
                            for p in players[:50]
                        )
                        with open(diag_file, 'w') as f:
                            f.write(f"No variables for slot: {slot_name} (requires: {required})\n\n")
                            f.write("Eligible var counts by slot:\n")
                            for s, c in slot_counts.items():
                                f.write(f"  {s}: {c}\n")
                            f.write("\nSample players (first 50):\n")
                            f.write(sample_players + "\n")
                        print(f"[DEBUG] LP written to: {lp_file}; diag: {diag_file}")
                    except Exception:
                        pass
                raise ValueError(
                    f"CRITICAL: No variables created for slot '{slot_name}'. "
                    f"Required positions: {SLOT_ELIGIBILITY[slot_name]}. "
                    f"This should never happen - check position data."
                )
    
    # --- STEP 4: Each player can only be selected once (across all slots) ---
    for player in players:
        slot_vars = [x[player.id][slot] for slot in x[player.id]]
        if slot_vars:
            prob += plp.lpSum(slot_vars) <= 1, f"PlayerOnce_{player.id}"
    
    # --- STEP 4b: Exactly 8 unique players must be selected (total roster size) ---
    # Create binary indicator variables: y[player_id] = 1 if player is selected in ANY slot
    y: Dict[int, plp.LpVariable] = {}
    for player in players:
        if player.id in x and x[player.id]:
            y[player.id] = plp.LpVariable(f"y_selected_{player.id}", 0, 1, plp.LpBinary)
            # Link y to x: if any slot has this player, y must be 1
            # y[pid] >= sum(x[pid][slot] for all slots) would work, but we can use == since player can only be in 1 slot
            slot_sum = plp.lpSum([x[player.id][slot] for slot in x[player.id]])
            prob += y[player.id] == slot_sum, f"LinkSelected_{player.id}"
    
    # Exactly 8 unique players must be selected
    if y:
        prob += plp.lpSum(y.values()) == 8, "ExactlyEightPlayers"
    
    # (Removed multi-game constraints)
    
    # --- Salary constraint ---
    salary_expr = plp.lpSum(
        player.salary * x[player.id][slot]
        for player in players
        if player.id in x
        for slot in x[player.id]
    )
    prob += salary_expr <= salary_cap, "MaxSalary"

    # --- Optional per-team salary cap ---
    if max_salary_per_team_val is not None:
        # Group player IDs by team (uppercase), skip missing/blank teams
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
    
    # --- STEP 5: Objective - maximize total projected fantasy points ---
    score = plp.lpSum(
        base_scores.get(player.id, 0.0) * x[player.id][slot]
        for player in players
        if player.id in x
        for slot in x[player.id]
    )
    prob += score
    
    # --- STEP 6: Solve ---
    solver = _make_glpk_solver(config)
    try:
        status_code = prob.solve(solver)
    except PulpSolverError:
        if debug_lp_path:
            # Generate unique filename if directory path provided
            lp_file = _resolve_lp_path(debug_lp_path, "solver_error")
            Path(lp_file).parent.mkdir(parents=True, exist_ok=True)
            prob.writeLP(lp_file)
            print(f"[DEBUG] LP written to: {lp_file}")
        raise
    
    if plp.LpStatus[status_code] != "Optimal":
        if debug_lp_path:
            status_name = plp.LpStatus.get(status_code, "unknown")
            tag = attempt_tag or "na"
            lp_file = _resolve_lp_path(debug_lp_path, f"attempt{tag}_{status_name.lower()}")
            Path(lp_file).parent.mkdir(parents=True, exist_ok=True)
            prob.writeLP(lp_file)
            print(f"[DEBUG] Non-optimal status: {status_name} (code={status_code}) -> {lp_file}")
        return None
    
    # --- STEP 7: Extract and return selected player IDs ---
    selected = [
        player.id
        for player in players
        if player.id in x
        for slot in x[player.id]
        if x[player.id][slot].varValue == 1
    ]

    # --- STEP 8: Validate locked assignments (if provided) ---
    if locked_assignments:
        # All locked player IDs must appear in selection
        missing_locked = set(locked_assignments.keys()) - set(selected)
        if missing_locked:
            if debug_lp_path:
                tag = attempt_tag or "na"
                lp_file = _resolve_lp_path(debug_lp_path, f"attempt{tag}_optimal_missing_locked")
                Path(lp_file).parent.mkdir(parents=True, exist_ok=True)
                prob.writeLP(lp_file)
                print(f"[DEBUG] Optimal solve missing locked players {missing_locked} -> {lp_file}")
            return None

    # Optional: write successful LP snapshot for inspection if explicitly requested via env toggle
    if debug_lp_path and os.environ.get("WRITE_SUCCESSFUL_LP", "0") == "1":
        tag = attempt_tag or "na"
        lp_file = _resolve_lp_path(debug_lp_path, f"attempt{tag}_optimal")
        Path(lp_file).parent.mkdir(parents=True, exist_ok=True)
        prob.writeLP(lp_file)
        print(f"[DEBUG] Wrote optimal LP snapshot -> {lp_file}")
    
    # Validate we got exactly 8 players (one per slot)
    return selected if len(selected) == 8 else None


def generate_and_store_field_lineups(
    session: Session,
    config: dict | None = None,
    num_lineups: int = 1000,
    batch_size: int = 100,
    on_progress: Callable | None = None,
):
    
    config = config or {}

    raw_players = (
        session.query(NBAPlayer)
                .all()
    )
    if not raw_players:
        print("no players found in the NBAPlayer table")
        return 0
    
    players: List[FieldPlayer] = []

    for rp in raw_players:
        pos = getattr(rp, "position", None)
        pos_tuple = _coerce_positions(pos or [])

        # ENSURE it's always a tuple
        if isinstance(pos_tuple, str):
            pos_tuple = (pos_tuple,)
        elif not isinstance(pos_tuple, tuple):
            pos_tuple = tuple(pos_tuple) if pos_tuple else ()
        
        # Create a unique game ID from team matchups (canonical form: alphabetically sorted)
        team = getattr(rp, "team", "")
        opponent = getattr(rp, "opponent", None)
        if team and opponent:
            game_id = "-".join(sorted([team.upper(), opponent.upper()]))
        else:
            game_id = None
        
        players.append(FieldPlayer(
            id=rp.id,
            name=getattr(rp, "name"),
            salary=getattr(rp, "salary"),
            team=team,
            opponent=opponent,
            position=pos_tuple,
            fpts_mean=float(getattr(rp, "fpts_mean")),
            game_id=game_id,
            gametime=getattr(rp, "gametime", None)
        ))

    randomness = float(_cfg(config, "randomness", 1.0))
    n_samples = int(_cfg(config, "n_samples", 5000))
    jitter = float(_cfg(config, "jitter", 0.05))
    
    # Generate all sample sets: samples[player_id] = [score_sample_0, score_sample_1, ...]
    samples = generate_player_samples(
        session=session,
        n_samples=n_samples, 
        randomness=randomness
    )
    
    # Shuffle the sample indices so we cycle through them randomly
    shuffled_sample_indices = list(range(n_samples))
    random.shuffle(shuffled_sample_indices)

    created = 0
    batch = 0
    seen_lineups: set[tuple[int, ...]] = set()  # per-worker dedupe

    sample_cursor = 0
    while created < num_lineups:
        # Pick next sample set from shuffled order
        samp_idx = shuffled_sample_indices[sample_cursor % n_samples]
        sample_cursor += 1

        # Extract base scores for this sample set from all players
        base_scores = {pid: float(vec[samp_idx]) for pid, vec in samples.items()}

        # Add random jitter to create lineup variation
        jittered_scores = {
            pid: score * (1.0 + random.uniform(-jitter, jitter))
            for pid, score in base_scores.items()
        }

        ids_selected = generate_single_classic_lineup(
            players=players,
            base_scores=jittered_scores,
            config=config
        )
        if not ids_selected:
            continue
        key = tuple(sorted(ids_selected))
        if key in seen_lineups:
            continue
        seen_lineups.add(key)

        session.add(Lineup(player_ids_list=list(ids_selected), created_at=datetime.utcnow()))
        created += 1
        batch   += 1

        if on_progress is not None:
            on_progress(1, f"created={created}")

        if batch >= batch_size:
            session.commit()
            batch = 0

    if batch > 0:
        session.commit()

    return created