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


@dataclass(frozen=True)
class FieldPlayer:
    id: int
    name: str
    salary: int
    team: str
    opponent: Optional[str]
    position: Union[str, Tuple[str, ...]]
    fpts_mean: Optional[float]


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
        config: dict | None = None
) -> Optional[List[int]]:
    """
    NBA classic lineup generator using PuLP.

    Basic constraints:
    - 8 total players
    - Salary cap (default $50,000)
    - Min salary (default $49,000)
    - Max salary per team (optional, no limit if not set)
    - Min FPTS based on mean player FPTS (optional, no limit if not set)
    - Maximize projected points

    Args:
        players: List of FieldPlayer objects
        base_scores: Dict mapping player id -> projected points
        config: Optional config dict for tweaking parameters
            - salary_cap: Total salary cap (default 50000)
            - min_salary: Minimum salary (default 49000)
            - max_salary_per_team: Max salary allowed for a single team (optional)
            - min_fpts: Minimum FPTS sum based on player mean FPTS (optional)

    Returns:
        List of selected player ids, or None if infeasible
    """
    config = config or {}

    # Extract config parameters
    cap = int(_cfg(config, "salary_cap", 50000))
    min_salary = int(_cfg(config, "min_salary", 49000))
    max_salary_per_team = _cfg(config, "max_salary_per_team", None)
    if max_salary_per_team is not None:
        max_salary_per_team = int(max_salary_per_team)

    players_by_id: Dict[int, FieldPlayer] = {p.id: p for p in players}

    if not players_by_id:
        return None

    # Create the LP problem
    prob = plp.LpProblem(_unique_problem_name("nba_classic_lineup"), plp.LpMaximize)

    # Decision variables: x[i] = 1 if player i is selected, 0 otherwise
    x: Dict[int, plp.LpVariable] = {
        pid: plp.LpVariable(f"x_{_safe_var_name(players_by_id[pid].name)}_{pid}", 0, 1, plp.LpBinary)
        for pid in players_by_id
    }

    # ---- Roster constraints ----
    # Exactly 8 players
    prob += plp.lpSum(x[i] for i in players_by_id) == 8, "ExactlyEightPlayers"

    # Salary constraints
    salary_expr = plp.lpSum(players_by_id[i].salary * x[i] for i in players_by_id)
    prob += salary_expr <= int(cap), "MaxSalary"
    prob += salary_expr >= int(min_salary), "MinSalary"

    # Max salary per team constraint (if configured)
    if max_salary_per_team is not None:
        # Group player IDs by team
        teams: Dict[str, List[int]] = {}
        for pid in players_by_id:
            team = players_by_id[pid].team
            if team not in teams:
                teams[team] = []
            teams[team].append(pid)

        # Add constraint for each team
        for team, team_pids in teams.items():
            team_salary = plp.lpSum(players_by_id[pid].salary * x[pid] for pid in team_pids)
            prob += team_salary <= max_salary_per_team, f"MaxSalary_{_safe_var_name(team)}"

    # Minimum FPTS constraint (if configured)
    min_fpts = _cfg(config, "min_fpts", None)
    if min_fpts is not None:
        min_fpts = float(min_fpts)
        fpts_expr = plp.lpSum(float(players_by_id[i].fpts_mean or 0.0) * x[i] for i in players_by_id)
        prob += fpts_expr >= min_fpts, "MinFpts"

    # Objective: maximize projected points
    score = plp.lpSum(float(base_scores.get(i, 0.0)) * x[i] for i in players_by_id)
    prob += score

    # Solve
    solver = _make_glpk_solver(config)
    try:
        status_code = prob.solve(solver)
    except PulpSolverError:
        return None

    if plp.LpStatus[status_code] != "Optimal":
        return None

    # Extract selected player ids
    sel = [i for i, var in x.items() if var.varValue == 1]
    return sel if len(sel) == 8 else None
    

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
        players.append(FieldPlayer(
            id=rp.id,
            name=getattr(rp, "name"),
            salary=getattr(rp, "salary"),
            team=getattr(rp, "team"),
            opponent=getattr(rp, "opponent", None),
            position=pos_tuple,
            fpts_mean=float(getattr(rp, "fpts_mean"))
        ))

    randomness = float(_cfg(config, "randomness", 1.0))
    n_samples = 2500 #TODO: could make this a tunable parameter, though probably don't need to.
    samples = generate_player_samples(
        session=session,
        n_samples=n_samples, 
        randomness=randomness
    )
    idx_order = list(range(n_samples))
    random.shuffle(idx_order)

    created = 0
    batch = 0
    seen_lineups: set[tuple[int, ...]] = set() #per-worker dedupe

    jitter = 0.05 #TODO: pull out to config

    sample_cursor = 0
    while created < num_lineups:
        samp_idx = idx_order[sample_cursor % n_samples]
        sample_cursor += 1

        base_scores = {pid: float(vec[samp_idx]) for pid, vec in samples.items()}

        jittered = {pid: s * (1.0 + random.uniform(-jitter, jitter)) for pid, s in base_scores.items()}

        ids_selected = generate_single_classic_lineup(
            players=players,
            base_scores=jittered,
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
    