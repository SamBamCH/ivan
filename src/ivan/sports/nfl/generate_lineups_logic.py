from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Union
from datetime import datetime
import random
from pathlib import Path
import os, shutil, re, time
from uuid import uuid4

import numpy as np
from pulp import PulpSolverError
import pulp as plp
from sqlalchemy.orm import Session

from ivan.sports.nfl.models import NFLPlayer, Lineup
from ivan.sports.nfl.correlated_samples import generate_correlated_samples_gaussian

# ------- Logging directory (used only if you enable solver logging) -------
LOG_DIR = Path(__file__).resolve().parents[3] / "data" / "nfl" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =======================
# Helpers & Data Classes
# =======================

def _safe_var_name(name: str) -> str:
    """Make a solver-safe var name: letters/digits/underscore, never starting with a digit."""
    s = re.sub(r'[^A-Za-z0-9_]+', '_', str(name).strip())
    if not s or s[0].isdigit():
        s = 'x_' + s
    return s[:80]


def _cfg(config: dict, path: str, default):
    """Quiet dotted lookup with default (no warnings)."""
    cur = config or {}
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


@dataclass(frozen=True)
class FieldPlayer:
    id: int
    name: str
    salary: int
    team: str
    opponent: Optional[str]
    position: Union[str, Tuple[str, ...]]
    fpts_projection: Optional[float]


def _coerce_positions(pos: Iterable[str] | str) -> Union[str, Tuple[str, ...]]:
    """
    Normalize position(s):
      - "WR"            -> "WR"
      - "WR/TE"         -> ("TE", "WR")
      - ["RB","WR"]     -> ("RB","WR")
      - " DST ,  "      -> "DST"
    """
    if isinstance(pos, str):
        tokens = [p.strip().upper() for p in re.split(r"[\\/ ,]", pos) if p.strip()]
    else:
        tokens = [str(p).strip().upper() for p in pos if str(p).strip()]

    uniq = sorted(set(tokens))
    if not uniq:
        return tuple()
    if len(uniq) == 1:
        return uniq[0]
    return tuple(uniq)


def _as_bases(position: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    return position if isinstance(position, tuple) else (position,)


def _is_kicker(p: FieldPlayer) -> bool:
    return "K" in _as_bases(p.position)


def _is_dst(p: FieldPlayer) -> bool:
    return "DST" in _as_bases(p.position)


def _make_glpk_solver(config: dict):
    """
    Prefer GLPK if available; generate per-process log file; fall back to CBC.

    MP-safe changes:
      - keepFiles defaults to False to avoid temp file races.
      - If logging is enabled (solver.log=True), write to a per-process log:
        glpsol_{pid}_{millis}.log to avoid shared file locks.
    """
    keep_files = bool(_cfg(config, "solver.keep_files", False))
    msg = bool(_cfg(config, "solver.msg", False))
    log_enabled = bool(_cfg(config, "solver.log", False))

    path = (_cfg(config, "solver.glpk_path", None)
            or os.environ.get("GLPK_PATH")
            or shutil.which("glpsol"))

    options = []
    if log_enabled:
        log_path = str(LOG_DIR / f"glpsol_{os.getpid()}_{int(time.time()*1000)}.log")
        options = ["--log", log_path]

    if path and os.path.exists(path):
        return plp.GLPK_CMD(path=path, keepFiles=keep_files, msg=msg, options=options)

    if shutil.which("glpsol"):
        return plp.GLPK_CMD(keepFiles=keep_files, msg=msg, options=options)

    return plp.PULP_CBC_CMD(msg=msg)


# =======================
# Showdown ILP (single)
# =======================

def _unique_problem_name(base: str = "nfl_showdown_lineup") -> str:
    # Unique name to prevent GLPK/PuLP temp/solution file collisions across processes
    return f"{base}_{os.getpid()}_{int(time.time()*1e6)}_{uuid4().hex[:6]}"

def generate_single_showdown_lineup_ilp(
    players: List[FieldPlayer],
    roster_slot_by_id: Dict[int, int],
    base_scores: Dict[int, float],
    *,
    cap: int = 50000,
    min_cap_drop: int = 0,
    max_cap_drop: int = 2000,
    cap_drop_exponent: float = 2.0,  # skew toward using cap (larger -> closer to cap)
    min_k: int = 0,            # optional per-lineup minimum Kickers (kept but default off)
    min_dst: int = 0,          # optional per-lineup minimum DST (kept but default off)
    min_k_or_dst: int = 0,     # optional per-lineup minimum from {K,DST} combined (kept but default off)
    config: dict | None = None,
    debug_lp_path: Optional[str] = None,
) -> Optional[List[int]]:
    """
    ILP for DK Showdown:
      - exactly 1 CPT (slot 511) and 5 FLEX (slot 512)
      - no player appears twice (CPT+FLEX)
      - salary within [cap - drop, cap], where drop is drawn with bias toward small drops
      - optional per-lineup K/DST requirements (defaults OFF)
      - objective: maximize 1.5x score at CPT + score at FLEX (+ tiny salary tiebreak)
    """
    config = config or {}
    CPT, FLEX = 511, 512

    # Partition ids by slot
    players_by_id: Dict[int, FieldPlayer] = {p.id: p for p in players}
    cpt_ids  = [p.id for p in players if roster_slot_by_id.get(p.id) == CPT]
    flex_ids = [p.id for p in players if roster_slot_by_id.get(p.id) == FLEX]
    if not cpt_ids or len(flex_ids) < 5:
        return None

    # Map "identity" to ids for no-dup constraint
    ident_to_ids: Dict[Tuple[str,str], List[int]] = {}
    for p in players:
        ident_to_ids.setdefault((p.name, p.team), []).append(p.id)

    # Candidate sets for K / DST (constraints only added if mins > 0)
    k_ids     = [p.id for p in players if _is_kicker(p)]
    dst_ids   = [p.id for p in players if _is_dst(p)]
    k_dst_ids = list(set(k_ids) | set(dst_ids))

    # Draw a cap drop with bias toward small drops (skews toward using more salary)
    drop_range = max(0, int(max_cap_drop) - int(min_cap_drop))
    u = random.random()  # uniform [0,1)
    scaled = u ** float(max(1e-9, cap_drop_exponent))
    cap_drop = int(int(min_cap_drop) + scaled * drop_range)
    near_cap_floor = int(cap) - cap_drop

    # Build ILP with a unique name (prevents temp/sol file collisions across processes)
    prob = plp.LpProblem(_unique_problem_name("nfl_showdown_lineup"), plp.LpMaximize)
    x: Dict[int, plp.LpVariable] = {
        pid: plp.LpVariable(f"x_{_safe_var_name(players_by_id[pid].name)}_{pid}", 0, 1, plp.LpBinary)
        for pid in players_by_id
    }

    team_counts = Counter(p.team for p in players if getattr(p, "team", None))
    if len(team_counts) < 2:
        # With fewer than 2 teams in the pool, this constraint would be infeasible
        return None
    # Take the two most common teams (robust to any stray/FA rows)
    showdown_teams = [t for t, _ in team_counts.most_common(2)]

    for t in showdown_teams:
        team_ids = [p.id for p in players if p.team == t]
        if not team_ids:
            return None  # defensive: shouldn't happen, but avoids infeasible model
        prob += plp.lpSum(x[i] for i in team_ids) >= 1, f"AtLeastOneFrom_{_safe_var_name(t)}" #TODO: this probably works, but if not: change logic to needing a min of 2 unique teams. that'll be more robust for sure. 

    # Counts per slot
    prob += plp.lpSum(x[i] for i in cpt_ids)  == 1,  "ExactlyOneCPT"
    prob += plp.lpSum(x[i] for i in flex_ids) == 5,  "ExactlyFiveFLEX"

    # No CPT+FLEX for same real player
    for (name, team), ids in ident_to_ids.items():
        prob += plp.lpSum(x[i] for i in ids) <= 1, f"NoDup_{_safe_var_name(name)}_{_safe_var_name(team)}"

    # Salary window
    salary_expr = plp.lpSum(players_by_id[i].salary * x[i] for i in players_by_id)
    prob += salary_expr <= int(cap),            "MaxSalary"
    prob += salary_expr >= int(near_cap_floor), "MinSalaryWindow"

    # Optional per-lineup K/DST requirements (only added if mins > 0)
    if int(min_k) > 0 and k_ids:
        prob += plp.lpSum(x[i] for i in k_ids) >= int(min_k), "MinKicker"
    if int(min_dst) > 0 and dst_ids:
        prob += plp.lpSum(x[i] for i in dst_ids) >= int(min_dst), "MinDST"
    if int(min_k_or_dst) > 0 and k_dst_ids:
        prob += plp.lpSum(x[i] for i in k_dst_ids) >= int(min_k_or_dst), "MinK_or_DST"

    # Objective: 1.5x for CPT rows; add tiny salary tiebreak to stay near cap
    score = plp.lpSum(
        (1.5 if roster_slot_by_id.get(i) == CPT else 1.0) * float(base_scores.get(i, 0.0)) * x[i]
        for i in players_by_id
    )
    prob += score + 1e-4 * (salary_expr / float(max(1, int(cap)))), "ScorePlusSalaryTiebreak"

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

    sel = [i for i, var in x.items() if var.varValue == 1]
    return sel if len(sel) == 6 else None

def generate_single_classic_lineup(
        players: List[FieldPlayer],
        roster_slot_by_id: Dict[int, int],
        base_scores: Dict[int, float],
        cap: int = 50000,
        min_salary: int = 49000,
        config: dict | None = None,
        debug_lp_path: Optional[str] = None
) -> Optional[List[int]]:
    """
    Reads constraints from a nested config like:
      config["classic"]["lineup_generation"][...]
    with graceful fallbacks to flatter locations.
    """
    config = config or {}

    # -------------------------
    # Config lookup helper(s)
    # -------------------------
    def _get_nested(d: dict, path: Tuple[str, ...], default=None):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _cfg_val(key: str, default=None):
        # Primary: classic -> lineup_generation
        v = _get_nested(config, ("classic", "lineup_generation", key), None)
        if v is not None:
            return v
        # Secondary fallbacks
        for path in [
            ("lineup_generation", key),
            ("classic", key),
            (key,),
        ]:
            v = _get_nested(config, path, None)
            if v is not None:
                return v
        return default

    def _as_int(x, default=None):
        try:
            return int(x) if x is not None else default
        except (TypeError, ValueError):
            return default

    def _as_opt_int(x):
        try:
            return int(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    # --- Keys you can pass in config ---
    max_off_vs_dst    = _as_opt_int(_cfg_val("max_off_vs_dst", None))               # int or None
    # legacy: min_qb_stack = min RB/WR/TE from QB team (kept for backward compat)
    min_qb_stack_legacy = _as_int(_cfg_val("min_qb_stack", _cfg_val("min_same_team_stack", 0)), 0)
    min_qb_bringback    = _as_int(_cfg_val("min_qb_bringback", _cfg_val("min_bringback", 0)), 0)
    max_team_players    = _as_opt_int(_cfg_val("max_team_players", None))           # int or None
    # NEW:
    min_qb_wrte_stack   = _as_int(_cfg_val("min_qb_wrte_stack", 0), 0)              # WR/TE from QB team
    min_qb_skill_stack  = _as_int(_cfg_val("min_qb_skill_stack", 0), 0)             # RB/WR/TE from QB team (total)

    # Allow config to override salary band too
    cap_cfg        = _as_int(_cfg_val("salary_cap", None))
    min_salary_cfg = _as_int(_cfg_val("min_salary", None))
    if cap_cfg is not None:
        cap = cap_cfg
    if min_salary_cfg is not None:
        min_salary = min_salary_cfg

    QB, RB, WR, TE, DST = 66, 67, 68, 69, 71
    OFF_POS        = {QB, RB, WR, TE}
    OFF_NO_QB_POS  = {RB, WR, TE}
    WRTE_POS       = {WR, TE}

    players_by_id: Dict[int, FieldPlayer] = {p.id: p for p in players}
    qb_ids  = [p.id for p in players if roster_slot_by_id.get(p.id) == QB]
    rb_ids  = [p.id for p in players if roster_slot_by_id.get(p.id) == RB]
    wr_ids  = [p.id for p in players if roster_slot_by_id.get(p.id) == WR]
    te_ids  = [p.id for p in players if roster_slot_by_id.get(p.id) == TE]
    wrte_ids = list(set(wr_ids) | set(te_ids))
    dst_ids = [p.id for p in players if roster_slot_by_id.get(p.id) == DST]
    skill_ids = list(set(rb_ids) | set(wr_ids) | set(te_ids))

    if not qb_ids:
        raise ValueError("No QB ids matched. Check filtering conditions")

    # Helper pools
    offense_ids    = [pid for pid in players_by_id if roster_slot_by_id.get(pid) in OFF_POS]
    offense_no_qb  = [pid for pid in players_by_id if roster_slot_by_id.get(pid) in OFF_NO_QB_POS]
    wrte_no_qb     = [pid for pid in players_by_id if roster_slot_by_id.get(pid) in WRTE_POS]

    # Team maps (all positions)
    teams = sorted({players_by_id[pid].team for pid in players_by_id if getattr(players_by_id[pid], "team", None)})
    team_to_all_ids = {t: [pid for pid in players_by_id if getattr(players_by_id[pid], "team", None) == t] for t in teams}

    # Group duplicate name+team
    ident_to_ids: Dict[Tuple[str, str], List[int]] = {}
    for p in players:
        ident_to_ids.setdefault((p.name, p.team), []).append(p.id)

    # --- LP model ---
    prob = plp.LpProblem(_unique_problem_name("nfl_classic_lineup"), plp.LpMaximize)

    # Decision vars
    x: Dict[int, plp.LpVariable] = {
        pid: plp.LpVariable(f"x_{_safe_var_name(players_by_id[pid].name)}_{pid}", 0, 1, plp.LpBinary)
        for pid in players_by_id
    }

    # --- Roster constraints ---
    prob += plp.lpSum(x[i] for i in qb_ids)  == 1, "ExactlyOneQB"
    prob += plp.lpSum(x[i] for i in dst_ids) == 1, "ExactlyOneDST"

    prob += plp.lpSum(x[i] for i in rb_ids) >= 2, "AtLeastTwoRB"
    prob += plp.lpSum(x[i] for i in wr_ids) >= 3, "AtLeastThreeWR"
    prob += plp.lpSum(x[i] for i in te_ids) >= 1, "AtLeastOneTE"

    # 7 total among RB/WR/TE (2 RB + 3 WR + 1 TE + 1 FLEX)
    prob += plp.lpSum(x[i] for i in skill_ids) == 7, "ExactlySevenSkill"
    prob += plp.lpSum(x[i] for i in players_by_id) == 9, "ExactlyNinePlayers"

    # No duplicate same-name same-team
    for (name, team), ids in ident_to_ids.items():
        prob += plp.lpSum(x[i] for i in ids) <= 1, f"NoDup_{_safe_var_name(name)}_{_safe_var_name(team)}"

    # Salary band
    salary_expr = plp.lpSum(players_by_id[i].salary * x[i] for i in players_by_id)
    prob += salary_expr <= int(cap),        "MaxSalary"
    prob += salary_expr >= int(min_salary), "MinSalary"

    # --- Max players per single team (all positions count) ---
    if isinstance(max_team_players, int):
        for t in teams:
            prob += plp.lpSum(x[i] for i in team_to_all_ids[t]) <= max_team_players, f"MaxPlayers_{_safe_var_name(t)}"

    # --- QB same-team stacks ---
    # Effective total skill requirement (RB/WR/TE). Keep legacy key for back-compat.
    eff_min_skill = max(int(min_qb_skill_stack), int(min_qb_stack_legacy))
    if eff_min_skill > 0 or int(min_qb_wrte_stack) > 0:
        for q in qb_ids:
            team_q = getattr(players_by_id[q], "team", None)
            if not team_q:
                continue
            # Pools from QB's team
            skill_pool = [i for i in offense_no_qb if getattr(players_by_id[i], "team", None) == team_q]
            wrte_pool  = [i for i in wrte_no_qb    if getattr(players_by_id[i], "team", None) == team_q]

            # (1) WR/TE-only requirement (e.g., force at least one WR/TE with QB)
            if int(min_qb_wrte_stack) > 0 and len(wrte_pool) > 0:
                req_wrte = min(int(min_qb_wrte_stack), len(wrte_pool))
                prob += plp.lpSum(x[i] for i in wrte_pool) >= req_wrte * x[q], f"QB_WRTE_Stack_{q}"

            # (2) Total teammates from RB/WR/TE (e.g., QB + 2 total, second can be RB)
            if eff_min_skill > 0 and len(skill_pool) > 0:
                req_skill = min(eff_min_skill, len(skill_pool))
                prob += plp.lpSum(x[i] for i in skill_pool) >= req_skill * x[q], f"QB_Skill_Stack_{q}"

    # --- QB bring-backs (min RB/WR/TE from QB's opposing team) ---
    if int(min_qb_bringback) > 0:
        for q in qb_ids:
            opp_q = getattr(players_by_id[q], "opponent", None)
            if not opp_q:
                continue
            bb_pool = [i for i in offense_no_qb if getattr(players_by_id[i], "team", None) == opp_q]
            req = min(int(min_qb_bringback), len(bb_pool))
            if req > 0 and len(bb_pool) > 0:
                prob += plp.lpSum(x[i] for i in bb_pool) >= req * x[q], f"QBBRingback_{q}"

    # --- Limit offensive players vs the chosen DST ---
    if isinstance(max_off_vs_dst, int):
        BIG_M_OFF = 8  # max possible offense (QB + 7 skill)
        for d in dst_ids:
            opp_d = getattr(players_by_id[d], "opponent", None)
            if not opp_d:
                continue
            off_vs_dst = [i for i in offense_ids if getattr(players_by_id[i], "team", None) == opp_d]
            if not off_vs_dst:
                continue
            prob += (
                plp.lpSum(x[i] for i in off_vs_dst)
                <= int(max_off_vs_dst) + BIG_M_OFF * (1 - x[d])
            ), f"MaxOffVsDST_{d}"

    # Objective
    score = plp.lpSum(float(base_scores.get(i, 0.0)) * x[i] for i in players_by_id)
    prob += score

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

    sel = [i for i, var in x.items() if var.varValue == 1]
    return sel if len(sel) == 9 else None




def _infer_slate_type(roster_slot_by_id: Dict[int, int]) -> str:
    """Return 'showdown' if any DK Showdown roster slots (511/512) exist; else 'classic'."""
    vals = set(roster_slot_by_id.values())
    return "showdown" if (511 in vals or 512 in vals) else "classic"


# =======================
# Generation loop
# =======================

def generate_and_store_field_lineups(
    session: Session,
    config: dict | None = None,
    num_lineups: int = 1000,
    batch_size: int = 100,
    on_progress: callable | None = None,
) -> int:
    """
    Single-process generator for DK Showdown only (classic removed for now):
      - Loads/creates samples internally (no internal ProcessPool).
      - Fast ILP with salary window + jitter for variety.
      - Writes Lineup rows in batches of `batch_size`.

    Returns: number of lineups created.
    """
    config = config or {}

    # ---- 0) Generate per-player samples (pid -> np.ndarray) ----
    # Try to pass optional sampling config (if supported) so caller can control num samples, etc.

    raw_players = (
        session.query(NFLPlayer)
               .all()
    )
    if not raw_players:
        return 0

    # Wrap into FieldPlayer + roster slots
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
            fpts_projection=float(getattr(rp, "fpts_p50", 0.0) or 0.0),
        ))
        roster_slot_by_id[rp.id] = int(getattr(rp, "roster_slot_id", 0) or 0)


    samples = generate_correlated_samples_gaussian(session, n_samples=max(5000, num_lineups*2))
    slate_type = _infer_slate_type(roster_slot_by_id)

    # ---- 2) infer slate type, and choose branch ----
    if slate_type == "classic":
        n_samples = len(next(iter(samples.values())))
        idx_order = list(range(n_samples))
        random.shuffle(idx_order)

        created = 0
        batch   = 0
        seen_lineups: set[tuple[int, ...]] = set() #per-worker dedupe

        #---------- config knobs ----------------------------------------
        jitter            = float(_cfg(config, "classic.lineup_generation.jitter", 0.10))
        salary_cap        = int(_cfg(config, "classic.lineup_generation.salary_cap", 50000))
        min_salary        = int(_cfg(config, "classic.lineup_generation.min_salary", 49000))
        debug_lp_dir      = _cfg(config, "lineup_generation.debug_lp_dir", None)

        sample_cursor = 0
        while created < num_lineups:
            samp_idx = idx_order[sample_cursor % n_samples]
            sample_cursor += 1

            base_scores = {pid: float(vec[samp_idx]) for pid, vec in samples.items()}

            jittered = {pid: s * (1.0 + random.uniform(-jitter, jitter)) for pid, s in base_scores.items()}

            debug_path = None
            if debug_lp_dir:
                debug_path = str(Path(debug_lp_dir) / f"debug_classic_{os.getpid()}_{created+1}.lp")

            ids_selected = generate_single_classic_lineup(
                players=players,
                roster_slot_by_id=roster_slot_by_id,
                base_scores=jittered,
                cap=salary_cap,
                min_salary=min_salary,
                config=config,
                debug_lp_path=debug_path
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
    

    elif slate_type == "showdown":
        # ---- 3) Precompute iteration order over sample indices ----
        n_samples = len(next(iter(samples.values())))
        idx_order = list(range(n_samples))
        random.shuffle(idx_order)

        created = 0
        batch   = 0
        seen_lineups: set[tuple[int, ...]] = set()  # per-worker dedupe

        # ---- 4) Config knobs (showdown ONLY; defined once) ----
        jitter            = float(_cfg(config, "showdown.lineup_generation.jitter", 0.10))
        salary_cap        = int(_cfg(config, "showdown.lineup_generation.salary_cap", 50000))
        min_cap_drop      = int(_cfg(config, "showdown.lineup_generation.min_cap_drop", 0))
        max_cap_drop      = int(_cfg(config, "showdown.lineup_generation.max_cap_drop", 2000))
        cap_drop_exponent = float(_cfg(config, "showdown.lineup_generation.cap_drop_exponent", 2.0))
        debug_lp_dir      = _cfg(config, "showdown.lineup_generation.debug_lp_dir", None)

        # ---- 5) Generate lineups (no exposure forcing; no max_attempts cap) ----
        sample_cursor = 0
        while created < num_lineups:
            # Rotate through samples for variety
            samp_idx = idx_order[sample_cursor % n_samples]
            sample_cursor += 1

            base_scores = {pid: float(vec[samp_idx]) for pid, vec in samples.items()}

            # Jitter scores (Uniform; keep this simple & fast)
            jittered = {pid: s * (1.0 + random.uniform(-jitter, jitter)) for pid, s in base_scores.items()}

            # Optional LP dump; include PID to avoid MP collisions
            debug_path = None
            if debug_lp_dir:
                debug_path = str(Path(debug_lp_dir) / f"debug_showdown_{os.getpid()}_{created+1}.lp")

            ids_selected = generate_single_showdown_lineup_ilp(
                players=players,
                roster_slot_by_id=roster_slot_by_id,
                base_scores=jittered,
                cap=salary_cap,
                min_cap_drop=min_cap_drop,
                max_cap_drop=max_cap_drop,
                cap_drop_exponent=cap_drop_exponent,
                # No forced position mins; rely on projections/upstream instead.
                min_k=0,
                min_dst=0,
                min_k_or_dst=0,
                config=config,
                debug_lp_path=debug_path,
            )

            if not ids_selected:
                # Nothing stored; continue trying.
                continue

            key = tuple(sorted(ids_selected))
            if key in seen_lineups:
                continue
            seen_lineups.add(key)

            session.add(Lineup(player_ids_list=list(ids_selected), created_at=datetime.utcnow()))
            created += 1
            batch   += 1

            if batch >= batch_size:
                session.commit()
                batch = 0

        if batch > 0:
            session.commit()

        return created
    
    else:
        raise ValueError("couldn't infer slate type based on payload from draftkings")
