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


def generate_single_classic_lineup(
        players: List[FieldPlayer],
        base_scores: Dict[int, float],
        config: dict | None = None,
        debug_lp_path: Optional[str] = None,
) -> Optional[List[int]]:
    """
    Generate a single optimal NBA classic lineup given player pool and projected scores.
    
    Roster: PG, SG, SF, PF, C, G, F, UTIL (8 players total)
    Constraints:
      - Each slot must be filled by exactly one eligible player
      - Each player can only be selected once across all slots
      - Lineup must respect salary cap (~$50,000)
      - Lineup must have players from at least 2 different games
      - Maximize total projected fantasy points
    """
    config = config or {}
    
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
    
    salary_cap = int(_cfg_val("salary_cap", 50000))
    min_salary = int(_cfg_val("min_salary", 49000))
    
    # --- STEP 1: Create binary decision variables for each (player, slot) pair ---
    # x[player_id][slot_name] = 1 if player assigned to slot, 0 otherwise
    x: Dict[int, Dict[str, plp.LpVariable]] = {}
    
    for player in players:
        x[player.id] = {}
        
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
    
    # --- Group players by game ---
    games: Dict[Optional[str], List[int]] = {}
    for player in players:
        if player.game_id not in games:
            games[player.game_id] = []
        games[player.game_id].append(player.id)
    
    # --- STEP 2 & 3: Set up LP problem and slot-fill constraints ---
    prob = plp.LpProblem(_unique_problem_name("nba_classic_lineup"), plp.LpMaximize)
    
    # Each slot must have exactly one player
    for slot_name in SLOT_NAMES:
        eligible_vars = [
            x[p.id][slot_name]
            for p in players
            if slot_name in x[p.id]
        ]
        if eligible_vars:
            prob += plp.lpSum(eligible_vars) == 1, f"Slot_{slot_name}"
        else:
            # CRITICAL ERROR: No eligible players for this slot!
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
    
    # --- Multi-game constraint: lineup must have players from at least 2 different games ---
    # For each game, create a binary variable indicating if any player from that game is selected
    game_vars: Dict[Optional[str], plp.LpVariable] = {}
    for game_id, player_ids in games.items():
        if player_ids:
            game_var = plp.LpVariable(f"game_{_safe_var_name(str(game_id))}", 0, 1, plp.LpBinary)
            game_vars[game_id] = game_var
            
            # game_var = 1 if ANY player from this game is selected
            # Big-M constraint: if any player from game is selected, game_var must be 1
            num_players_in_game = len(player_ids)
            player_selection_sum = plp.lpSum(
                x[pid][slot]
                for pid in player_ids
                if pid in x  # Only include players that have variables
                for slot in x[pid]
            )
            prob += player_selection_sum <= num_players_in_game * game_var, f"GameActive_{_safe_var_name(str(game_id))}"
    
    # At least 2 different games must be represented
    if len(game_vars) > 1:
        prob += plp.lpSum(game_vars.values()) >= 2, "AtLeastTwoGames"
    
    # --- Salary constraint ---
    salary_expr = plp.lpSum(
        player.salary * x[player.id][slot]
        for player in players
        if player.id in x
        for slot in x[player.id]
    )
    prob += salary_expr <= salary_cap, "MaxSalary"
    prob += salary_expr >= min_salary, "MinSalary"
    
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
            Path(debug_lp_path).parent.mkdir(parents=True, exist_ok=True)
            prob.writeLP(debug_lp_path)
        raise
    
    if plp.LpStatus[status_code] != "Optimal":
        if debug_lp_path:
            Path(debug_lp_path).parent.mkdir(parents=True, exist_ok=True)
            prob.writeLP(debug_lp_path)
        return None
    
    # --- STEP 7: Extract and return selected player IDs ---
    selected = [
        player.id
        for player in players
        if player.id in x
        for slot in x[player.id]
        if x[player.id][slot].varValue == 1
    ]
    
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
            game_id=game_id
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