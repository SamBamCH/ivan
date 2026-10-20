# src/ivan/sports/mma/lineup_generation.py
"""
Candidate lineups: what the field builds. Each candidate is the optimum of
a jittered objective (win probability, boosted for main-event fighters,
plus noise) under DraftKings' rules: six fighters, the salary cap, a salary
floor, and at most one fighter per fight. Solved with GLPK when glpsol is
available, else PuLP's bundled CBC. Unique candidates go to the lineup table.
"""
from __future__ import annotations

import os
import shutil
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pulp as plp
from sqlalchemy.orm import Session

from ivan.core.seed import seed_for
from ivan.sports.mma.models import Lineup, LineupSimulationResult, MMAFight, MMAPlayer

DEFAULTS = {"n_candidates": 6000, "jitter_sd": 0.15, "min_salary": 48500, "salary_cap": 50000,
            "roster_size": 6, "main_event_boost": 1.15, "projection_share": 0.0}
# projection_share: the share of solves whose objective is the fighter's projected points (scaled like win
# probability) instead of win probability, so the pool also holds the field's points-chasing builds (underdog
# finishers) that a pure win-probability objective never produces.


def _solver():
    path = os.environ.get("GLPK_PATH") or shutil.which("glpsol")
    if path:
        return plp.GLPK_CMD(path=path, msg=False)
    return plp.PULP_CBC_CMD(msg=False)


def solve_lineup(players: list[dict], fights: list[tuple[int, int]], scores: dict[int, float], cfg: dict) -> Optional[tuple[int, ...]]:
    """One MIP solve; the chosen fighter ids sorted, or None when infeasible."""
    prob = plp.LpProblem("mma_lineup", plp.LpMaximize)
    x = {p["id"]: plp.LpVariable(f"p_{p['id']}", cat="Binary") for p in players}
    prob += plp.lpSum(x[p["id"]] * scores[p["id"]] for p in players)
    prob += plp.lpSum(x.values()) == int(cfg["roster_size"])
    prob += plp.lpSum(x[p["id"]] * p["salary"] for p in players) <= int(cfg["salary_cap"])
    prob += plp.lpSum(x[p["id"]] * p["salary"] for p in players) >= int(cfg["min_salary"])
    for a, b in fights:
        if a in x and b in x:
            prob += x[a] + x[b] <= 1
    prob.solve(_solver())
    if prob.status != plp.LpStatusOptimal:
        return None
    return tuple(sorted(pid for pid, v in x.items() if v.varValue is not None and v.varValue > 0.5))


def jittered_scores(players: list[dict], main_event: set[int], cfg: dict, rng: np.random.Generator) -> dict[int, float]:
    """
    Win probability on a 0..100 scale (or, for projection_share of the solves,
    projected points rescaled to the same mean), main-event boost, Gaussian
    noise, floored at zero.
    """
    out = {}
    by_projection = rng.random() < float(cfg.get("projection_share", 0.0))
    if by_projection:
        projs = [float(p.get("projection") or 0.0) for p in players]
        wins = [float(p["win_prob"]) for p in players]
        scale = (100.0 * sum(wins) / len(wins)) / max(1e-9, sum(projs) / len(projs))
    for p in players:
        base = float(p.get("projection") or 0.0) * scale if by_projection else float(p["win_prob"]) * 100.0
        if p["id"] in main_event:
            base *= float(cfg["main_event_boost"])
        out[p["id"]] = max(0.0, base + rng.normal(0.0, float(cfg["jitter_sd"]) * 100.0))
    return out


def _worker(args) -> list[tuple[int, ...]]:
    players, fights, main_event, cfg, n, seed = args
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(n):
        lu = solve_lineup(players, fights, jittered_scores(players, main_event, cfg, rng), cfg)
        if lu:
            found.append(lu)
    return found


def generate_candidates(session: Session, cfg: dict, n_candidates: Optional[int] = None, n_jobs: int = 1,
                        fresh: bool = True) -> dict:
    """
    Solve n_candidates jittered lineups over n_jobs processes, dedupe, and
    store the new ones. fresh=True first clears the lineup pool and every
    simulation result (a new slate). Returns counts.
    """
    gen = {**DEFAULTS, **cfg.get("lineup_generation", {})}
    n_candidates = int(n_candidates or gen["n_candidates"])
    players = [{"id": p.id, "salary": p.salary, "win_prob": p.win_prob if p.win_prob is not None else 0.5, "projection": p.projection}
               for p in session.query(MMAPlayer).all()]
    if not players:
        raise SystemExit("no fighters in the slate; run scripts/mma/ingest.py first")
    fights = [(f.fighter1_id, f.fighter2_id) for f in session.query(MMAFight).all()]
    main_event = {pid for f in session.query(MMAFight).filter(MMAFight.rounds == 5).all() for pid in (f.fighter1_id, f.fighter2_id)}

    if fresh:
        session.query(LineupSimulationResult).delete(synchronize_session=False)
        session.query(Lineup).delete(synchronize_session=False)
        session.commit()
    existing = {tuple(sorted(lu.player_ids_list)) for lu in session.query(Lineup).all()}

    base_seed = seed_for("mma.generate")
    n_jobs = max(1, int(n_jobs))
    per = [n_candidates // n_jobs + (1 if i < n_candidates % n_jobs else 0) for i in range(n_jobs)]
    seeds = [None if base_seed is None else (base_seed + i) % (2 ** 32) for i in range(n_jobs)]
    tasks = [(players, fights, main_event, gen, per[i], seeds[i]) for i in range(n_jobs) if per[i] > 0]
    if n_jobs == 1:
        results = [_worker(tasks[0])]
    else:
        with Pool(len(tasks)) as pool:
            results = pool.map(_worker, tasks)

    solved = sum(len(r) for r in results)
    new = []
    for r in results:
        for lu in r:
            if lu not in existing:
                existing.add(lu)
                new.append(lu)
    session.bulk_save_objects([Lineup(player_ids_list=list(lu)) for lu in new])
    session.commit()
    return {"requested": n_candidates, "solved": solved, "new_unique": len(new), "pool": len(existing)}


def generate_universe(session: Session, cfg: dict, fresh: bool = True) -> dict:
    """
    Make the pool every valid lineup of the slate (ivan.sports.mma.universe):
    six fighters, one per fight at most, under the cap and at or above
    lineup_generation.universe_min_salary (default 0). The field model then
    decides each lineup's frequency; nothing is left to an optimiser's guess.
    """
    from ivan.sports.mma.universe import enumerate_lineups
    gen = {**DEFAULTS, **cfg.get("lineup_generation", {})}
    players = session.query(MMAPlayer).all()
    if not players:
        raise SystemExit("no fighters in the slate; run scripts/mma/ingest.py first")
    pool = [{"id": p.id, "salary": p.salary, "fight_id": p.fight_id} for p in players]
    win = {p.id: (p.win_prob if p.win_prob is not None else 0.5) for p in players}
    U = enumerate_lineups(pool, win, salary_cap=int(gen["salary_cap"]), min_salary=int(gen.get("universe_min_salary", 0)))
    if fresh:
        session.query(LineupSimulationResult).delete(synchronize_session=False)
        session.query(Lineup).delete(synchronize_session=False)
        session.commit()
    existing = {tuple(sorted(lu.player_ids_list)) for lu in session.query(Lineup).all()}
    new = [k for k in U.key if k not in existing]
    session.bulk_save_objects([Lineup(player_ids_list=list(k)) for k in new])
    session.commit()
    return {"requested": len(U), "solved": len(U), "new_unique": len(new), "pool": len(existing) + len(new)}
