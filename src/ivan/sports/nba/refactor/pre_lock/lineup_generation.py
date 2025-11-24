from multiprocessing import Process, Queue
from sqlalchemy.orm import Session
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import math
from dataclasses import dataclass
import random
import pulp as plp
import re, os, time, shutil
from uuid import uuid4
from queue import Empty
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn,MofNCompleteColumn


from ivan.database.session import get_session, init_db
from ivan.sports.nba.models import NBAPlayer, Lineup
from ivan.sports.nba.utils import _coerce_player_ids, _coerce_positions
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nba.refactor.pre_lock.fpts_samples import generate_player_samples

BATCH_COMMIT_SIZE = 100
JITTER = 0.05
MAX_SALARY_PER_TEAM = 50000
SALARY_CAP = 50000


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

def _make_solver():
    """Return a PuLP solver instance.

    Prefer the system `glpsol` if available. If not found, look for a
    `bin/glpsol` under the project root. If neither is available fall back
    to the default CBC solver.
    """
    # Prefer system-installed glpsol
    glpsol_path = shutil.which("glpsol")
    if glpsol_path:
        return plp.GLPK_CMD(path=glpsol_path, msg=False)

    # Fall back to a glpsol under the project tree: <PROJECT_ROOT>/bin/glpsol
    project_glpsol = os.path.join(str(PROJECT_ROOT), "bin", "glpsol")
    if os.path.exists(project_glpsol):
        return plp.GLPK_CMD(path=project_glpsol, msg=False)

    logging.error(f'no GLPK (glpsol) found on PATH or at {project_glpsol}, defaulting to CBC_CMD')
    return plp.PULP_CBC_CMD()

def generate_single_classic_lineup(
        players: List[FieldPlayer],
        base_scores: Dict[int, float],
        locked_assignments: Optional[Dict[int, str]] = None,
):

    #1: create binary decision variables for each (player, slot) pair
    # x[player_id][slot_name] = 1 if player assigned to slot, 0 otherwise
    x: Dict[int, Dict[str, plp.LpVariable]] = {}
    # Determine locked player IDs (those explicitly assigned to slots)
    for player in players:
        x[player.id] = {}

        for slot_name in SLOT_NAMES:
            if not any(pos in SLOT_ELIGIBILITY[slot_name] for pos in player.position):
                continue #player not eligible for this slot

            var_name = f"x_{_safe_var_name(player.name)}_slot{slot_name}_{player.id}"
            x[player.id][slot_name] = plp.LpVariable(var_name, 0, 1, plp.LpBinary)

    # DEBUG: Print locked assignments info
    if locked_assignments:
        print(f"\n[DEBUG] Locked assignments provided: {len(locked_assignments)} players")
        for pid, slot in locked_assignments.items():
            player_obj = next((p for p in players if p.id == pid), None)
            if player_obj:
                print(f"  - Player {pid} ({player_obj.name}) -> {slot} | Positions: {player_obj.position} | Salary: ${player_obj.salary}")
            else:
                print(f"  - Player {pid} -> {slot} | WARNING: NOT IN PLAYER POOL")

    #2: setup LP problem
    prob = plp.LpProblem(_unique_problem_name("nba_classic_lineup"), plp.LpMaximize)

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
                print(f"[DEBUG] Locked slot {slot_name} to player {locked_player_id}")
            else:
                player_obj = next((p for p in players if p.id == locked_player_id), None)
                if player_obj:
                    print(f"[DEBUG ERROR] Player {locked_player_id} ({player_obj.name}) positions {player_obj.position} NOT ELIGIBLE for slot {slot_name}")
                    print(f"  Required positions for {slot_name}: {SLOT_ELIGIBILITY[slot_name]}")
                else:
                    print(f"[DEBUG ERROR] Player {locked_player_id} not found in player pool for slot {slot_name}")
                raise ValueError(f"locked player {locked_player_id} is not eligible for slot {slot_name}")

        else:
            #slot is not locked, normal optimization constraint for slot
            eligible_vars = [
                x[p.id][slot_name]
                for p in players
                if slot_name in x[p.id]
            ]
            if eligible_vars:
                prob += plp.lpSum(eligible_vars) == 1, f"Slot_{slot_name}"
                print(f"[DEBUG] Slot {slot_name} has {len(eligible_vars)} eligible players")
            else:
                print(f"[DEBUG ERROR] No eligible players for slot {slot_name}")
                raise ValueError(f"no variables passed for slot {slot_name}")

    #3: every player can only be selected once: 
    for player in players:
        slot_vars = [x[player.id][slot] for slot in x[player.id]]
        if slot_vars:
            prob += plp.lpSum(slot_vars) <= 1, f"PlayerOnce_{player.id}"

    #4: exactly 8 players
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

        # --- Salary constraint ---
    salary_expr = plp.lpSum(
        player.salary * x[player.id][slot]
        for player in players
        if player.id in x
        for slot in x[player.id]
    )
    prob += salary_expr <= SALARY_CAP, "MaxSalary"

    # --- Optional per-team salary cap ---
    if MAX_SALARY_PER_TEAM is not None:
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
            prob += team_salary <= MAX_SALARY_PER_TEAM, f"TeamMaxSalary_{_safe_var_name(team_key)}"

    game_ids = {}
    for p in players:
        if p.id in x and x[p.id] and p.game_id:
            game_ids.setdefault(p.game_id, []).append(p.id)
    # Binary z_g indicating if any player from game g is selected
    z: Dict[str, plp.LpVariable] = {g: plp.LpVariable(f"game_used_{_safe_var_name(g)}", 0, 1, plp.LpBinary) for g in game_ids}
    for g, pids in game_ids.items():
        prob += z[g] <= plp.lpSum(y[pid] for pid in pids if pid in y), f"GameLink_{_safe_var_name(g)}"
        # If any pid selected then z[g] can be 1; allow slack by not forcing equality
    prob += plp.lpSum(z.values()) >= 2, "MinDistinctGames"
    
    # --- STEP 5: Objective - maximize total projected fantasy points ---
    score = plp.lpSum(
        base_scores.get(player.id, 0.0) * x[player.id][slot]
        for player in players
        if player.id in x
        for slot in x[player.id]
    )
    prob += score

    solver = _make_solver()
    
    # DEBUG: Print constraint summary before solving
    if locked_assignments:
        print(f"\n[DEBUG] Problem setup summary:")
        print(f"  Total players in pool: {len(players)}")
        print(f"  Total locked players: {len(locked_assignments)}")
        
        # Calculate salary of locked players
        locked_salary = sum(p.salary for p in players if p.id in locked_assignments)
        remaining_cap = SALARY_CAP - locked_salary
        print(f"  Locked salary: ${locked_salary}")
        print(f"  Remaining salary cap: ${remaining_cap}")
        
        # Count unlocked players
        unlocked_players = [p for p in players if p.id not in locked_assignments]
        print(f"  Unlocked players available: {len(unlocked_players)}")
        
        # Check if locked players are from too many games
        locked_game_ids = set(p.game_id for p in players if p.id in locked_assignments and p.game_id)
        print(f"  Games represented in locked players: {len(locked_game_ids)}")
        
        # Check per-team salary constraint
        if MAX_SALARY_PER_TEAM:
            team_salaries = {}
            for p in players:
                if p.id in locked_assignments and p.team:
                    team_key = p.team.upper().strip()
                    team_salaries[team_key] = team_salaries.get(team_key, 0) + p.salary
            for team, salary in team_salaries.items():
                print(f"  Locked salary for team {team}: ${salary} (max: ${MAX_SALARY_PER_TEAM})")
                if salary > MAX_SALARY_PER_TEAM:
                    print(f"    [WARNING] Team {team} already exceeds max salary cap!")
    
    status_code = prob.solve(solver)
    
    if plp.LpStatus[status_code] != "Optimal":
        print(f"\n[DEBUG] Solver returned status: {plp.LpStatus[status_code]}")
        
        if locked_assignments:
            print("[DEBUG] Infeasibility diagnostics:")
            print(f"  - Check if locked players can form valid roster")
            print(f"  - Check if remaining salary cap is sufficient")
            print(f"  - Check if per-team salary constraints are violated")
            print(f"  - Check if min games constraint can be met")
            
            # Show which slots are locked
            print("\n[DEBUG] Locked slot assignments:")
            for slot in SLOT_NAMES:
                locked_pid = None
                if locked_assignments:
                    for pid, s in locked_assignments.items():
                        if s == slot:
                            locked_pid = pid
                            break
                if locked_pid:
                    player_obj = next((p for p in players if p.id == locked_pid), None)
                    if player_obj:
                        print(f"  {slot}: {player_obj.name} (${player_obj.salary})")
                else:
                    print(f"  {slot}: [UNLOCKED]")
        
        logging.error('infeasibility reached for lineup, returning none...')
        return None
    
    selected = [
        player.id for player in players if player.id in x for slot in x[player.id] if x[player.id][slot].varValue == 1
    ]

    return selected if len(selected) == 8 else None


def _lineup_worker_loop_logic(session: Session, num_lineups: int, on_progress, seed: int = 42):
    raw_players = session.query(NBAPlayer).all()
    if not raw_players:
        logging.error("no players found in the NBAPlayer table")

    players: List[FieldPlayer] = []

    for rp in raw_players:
        pos = getattr(rp, "position", None)
        if pos is None:
            logging.error(f'no position found for {rp.name}')
            continue
        pos_tuple = _coerce_positions(pos)

        if rp.team and rp.opponent:
            game_id = "-".join(sorted([rp.team.upper(), rp.opponent.upper()]))
        else:
            game_id = None

        players.append(FieldPlayer(
            id=rp.id,
            name=rp.name,
            salary=rp.salary,
            team=rp.team,
            opponent=rp.opponent,
            position=pos_tuple,
            fpts_mean=rp.fpts_mean,
            game_id=game_id,
            gametime=rp.gametime
        ))

    # Sweep randomness values to generate diverse samples
    samples = {}
    randomness_values = [0.05, 0.05, 0.10, 0.10, 0.25, 0.25, 0.50, 0.50, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    # Ensure we generate enough samples across randomness sweeps. Use ceil so
    # total samples >= requested num_lineups and avoid zero-sized sample arrays.
    samples_per_randomness = max(1, math.ceil(num_lineups / len(randomness_values)))
    # Use a distinct seed per randomness sweep to diversify samples across workers
    for idx, rand in enumerate(randomness_values):
        local_seed = int(seed) + idx
        rand_samples = generate_player_samples(session, n_samples=samples_per_randomness, seed=local_seed, randomness=rand)
        samples.update(rand_samples)

    # Determine actual number of samples available from generated vectors
    if not samples:
        logging.error("no samples generated for lineup creation")
        return 0

    first_vec = next(iter(samples.values()))
    sample_count = len(first_vec)
    # Use sample indices in deterministic order; do not shuffle so all samples are used
    sample_indices = list(range(sample_count))

    created = 0
    batch = 0
    seen_lineups: set[tuple[int, ...]] = set()

    sample_cursor = 0
    while created < num_lineups:
        samp_idx = sample_indices[sample_cursor % sample_count]
        sample_cursor += 1

        base_scores = {pid: float(vec[samp_idx]) for pid, vec in samples.items()}
        jittered_scores = {pid: score * (1.0 + random.uniform(-JITTER, JITTER)) for pid, score in base_scores.items()}

        ids_selected = generate_single_classic_lineup(
            players=players,
            base_scores=jittered_scores
        )
        if not ids_selected:
            continue
        key = tuple(sorted(ids_selected))
        seen_lineups.add(key)

        session.add(Lineup(player_ids_list=list(ids_selected), created_at=datetime.utcnow()))
        created += 1
        batch   += 1

        if on_progress is not None:
            on_progress(1, f"created={created}")

        if batch >= BATCH_COMMIT_SIZE:
            session.commit()
            batch = 0

    if batch > 0:
        session.commit()

    return created
    

def _worker(worker_id: int, num_lineups: int, total_workers: int, q: Queue):

    engine = init_db('nba')
    session = get_session(engine)

    # Create a per-worker seed so each process generates different samples/jitter
    base_seed = 42
    seed = base_seed + worker_id

    q.put(("start", worker_id, num_lineups, f"seed={seed}"))

    # Seed Python's RNG in this process for jitter diversity
    random.seed(seed + os.getpid())

    def on_progress(delta: int, note: str | None = None):
        q.put(("advance", worker_id, delta, note or ""))

    created = _lineup_worker_loop_logic(
        session,
        num_lineups=num_lineups,
        on_progress=on_progress,
        seed=seed,
    )

    q.put(("done", worker_id, created, f"created={created}"))


def generate_lineups(num_lineups, num_jobs,):
    print(f'generating {num_lineups} lineups in {num_jobs} jobs')
    q = Queue()
    procs = []
    base = num_lineups // num_jobs
    remainder = num_lineups % num_jobs
    quotas = [base + (1 if i < remainder else 0) for i in range(num_jobs)]

    for proc_no, qnty in enumerate(quotas):
        p = Process(
            target=_worker,
            args=(proc_no, qnty, num_jobs, q)
        )
        p.start()
        procs.append(p)

    n_workers = len(procs)

    console = Console()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=True,
        refresh_per_second=4,
    ) as progress:
        task_id = {}
        done_count = 0

        while done_count < n_workers:
            try:
                kind, proc_no, val, note = q.get(timeout=0.25)
            except Empty:
                continue
            if kind == "start":
                # Show worker config in initial description
                desc = f"Worker {proc_no} ({note})"
                task_id[proc_no] = progress.add_task(desc, total=int(val))
            elif kind == "advance":
                tid = task_id.get(proc_no)
                if tid is None:
                    tid = task_id[proc_no] = progress.add_task(
                        f"Worker {proc_no}", total=int(val)
                    )
                progress.advance(tid, int(val))
                # Don't update description on every advance - causes flicker
            elif kind == "done":
                tid = task_id.get(proc_no)
                if tid is not None:
                    total = progress.tasks[tid].total or int(val)
                    progress.update(
                        tid,
                        completed=total,
                        description=f"Worker {proc_no} ✅ {note}",
                    )
                done_count += 1

        progress.stop()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    q.close(); q.join_thread()
    print(f'generated {num_lineups} lineups across {num_jobs} jobs')

def deduplicate_lineups():
    engine = init_db('nba')
    session = get_session(engine)

    seen_keys = set()
    duplicates = []

    for lu in session.query(Lineup).all():
        ids = _coerce_player_ids(lu.player_ids_list)
        key = tuple(sorted(ids))
        if key in seen_keys:
            duplicates.append(lu)
        else:
            seen_keys.add(key)

    if duplicates:
        for dup in duplicates:
            session.delete(dup)
        session.commit()
    
    print(f"Deduplicator: removed {len(duplicates)} duplicate lineups, kept {len(seen_keys)} unique.")
    