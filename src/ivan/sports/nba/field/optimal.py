# src/ivan/sports/nba/field/optimal.py
"""
The optimal lineup in each world: who the slate's best lineup holds, and how often.

Every team's projection store record carries the World's per-world fantasy points (projection.worlds: samples). World i of
every game taken together is one night (the games are independent in the World, so any pairing of their worlds is a world);
the salary-capped optimal lineup is solved in each (field.solver.generate_single_classic_lineup, the generator's solver) and
each player's share of those optimals is what the generator's exposures converge to before a generate runs: the desk's
**opt** column, read beside the field's ownership.

    optimal_shares(session, n_worlds, jobs, config, log)   -> the result dict: at, stamp (the store's when solved), n_worlds, players
                                                             {pid: {name, team, share (% of worlds), in_mean}}, mean_optimal {ids, total}
    write(result) / read()                                 slates/<dg>/projection/optimal.json for the current slate
    is_stale(result)                                       a team was projected since (the store's stamp moved)
    solve_worlds(players, scores_by_world, config, jobs)   the solves themselves (worker processes), [ids per world]

scripts/nba/optimal.py is the command; the desk's Solve button runs it as a job.
"""
from __future__ import annotations

import json
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sqlalchemy.orm import Session

from ivan.core.config import cfg
from ivan.sports.nba import slates as S
from ivan.sports.nba.field.generate import player_pool
from ivan.sports.nba.field.solver import FieldPlayer, generate_single_classic_lineup
from ivan.sports.nba.projection import store as PSTORE

FILE = "optimal.json"
DEFAULT_WORLDS = 100

_W: dict = {}      # a worker's players and config


def path(root: Optional[Path] = None) -> Path:
    return S.store_dir() / FILE if root is None else S.paths(S.current(), root).root / "projection" / FILE


def read(root: Optional[Path] = None) -> Optional[dict]:
    p = path(root)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:   # noqa: BLE001 - a broken file is no file
        return None


def write(result: dict, root: Optional[Path] = None) -> Path:
    p = path(root); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(result, indent=0)); tmp.replace(p)
    return p


def is_stale(result: Optional[dict]) -> bool:
    """True when a team was projected since the file was solved (the projection store's stamp moved), or there is no file."""
    if not result:
        return True
    st = PSTORE.stamp()
    return [st[0], st[1]] != list(result.get("stamp") or [])


def world_scores(session: Session, players: Sequence[FieldPlayer], n_worlds: int) -> tuple:
    """
    (scores, worlds, missing): scores is (worlds, players) fantasy points, world i of every team's samples side by side; a
    player with no samples (his team drawn before the store held them, or below the pool floor) sits at his mean in every
    world. worlds = min(n_worlds, the fewest samples any team holds). missing = teams with no samples.
    """
    recs = PSTORE.load_all(next((p for p in [S.current()]), None))
    by_pid: Dict[int, np.ndarray] = {}
    missing = []
    for team, rec in recs.items():
        s = rec.get("samples")
        if not s or "fpts" not in s:
            missing.append(team); continue
        for j, pid in enumerate(s["pids"]):
            by_pid[int(pid)] = np.asarray(s["fpts"][:, j], dtype=float)
    avail = min([len(v) for v in by_pid.values()] + [int(n_worlds)]) if by_pid else 0
    scores = np.zeros((avail, len(players)))
    for k, p in enumerate(players):
        v = by_pid.get(int(p.id))
        scores[:, k] = v[:avail] if v is not None else float(p.fpts_mean or 0.0)
    return scores, avail, sorted(missing)


def _init(players, config):
    _W["players"] = players; _W["config"] = config
    _W["max_team"] = cfg(config, "max_salary_per_team", 25000)


def _solve(scores: List[float]) -> Optional[List[int]]:
    players = _W["players"]
    base = {int(p.id): float(x) for p, x in zip(players, scores)}
    try:
        return generate_single_classic_lineup(players=players, base_scores=base, config=_W["config"], max_salary_per_team=_W["max_team"])
    except Exception:   # noqa: BLE001 - one world's solve failing is one world lost
        return None


def solve_worlds(players: Sequence[FieldPlayer], scores: np.ndarray, config: dict, jobs: int = 4, log=print) -> List[Optional[List[int]]]:
    """The optimal lineup per row of scores (a world each), solved in `jobs` worker processes; None where no lineup fits."""
    rows = [list(map(float, r)) for r in np.asarray(scores)]
    if not rows:
        return []
    jobs = max(1, min(int(jobs), len(rows)))
    if jobs == 1:
        _init(list(players), config)
        return [_solve(r) for r in rows]
    with get_context("fork").Pool(jobs, initializer=_init, initargs=(list(players), config)) as pool:
        out = []
        for i, ids in enumerate(pool.imap(_solve, rows, chunksize=max(1, len(rows) // (jobs * 4)))):
            out.append(ids)
            if (i + 1) % 25 == 0:
                log(f"  {i + 1} of {len(rows)} worlds solved")
    return out


def optimal_shares(session: Session, n_worlds: int = DEFAULT_WORLDS, jobs: int = 4, config: Optional[dict] = None, log=print,
                   player_min_fpts: Optional[float] = None) -> dict:
    """The result dict (see the module docstring); nothing written: the caller does (write)."""
    if config is None:
        from ivan.core.config import load_config
        config = load_config("nba")
    floor = float(player_min_fpts if player_min_fpts is not None else cfg(config, "generate.player_min_fpts", cfg(config, "player_min_fpts", 15.0)))
    players = player_pool(session, floor)
    stamp = PSTORE.stamp()
    if not players:
        return dict(at=datetime.now().isoformat(timespec="seconds"), stamp=list(stamp), n_worlds=0, players={}, mean_optimal=None, missing_teams=[], note="no projected players")
    scores, avail, missing = world_scores(session, players, n_worlds)
    log(f"{len(players)} players above {floor:g} fantasy points, {avail} worlds" + (f"; no samples for {', '.join(missing)}" if missing else ""))
    means = np.array([float(p.fpts_mean or 0.0) for p in players])
    rows = np.vstack([means[None, :], scores]) if avail else means[None, :]
    solved = solve_worlds(players, rows, config, jobs=jobs, log=log)
    mean_ids, world_ids = solved[0], [s for s in solved[1:] if s]
    by_pid = {int(p.id): p for p in players}
    counts: Dict[int, int] = {}
    for ids in world_ids:
        for i in ids:
            counts[int(i)] = counts.get(int(i), 0) + 1
    n_ok = len(world_ids)
    out_players = {}
    for pid, c in counts.items():
        p = by_pid.get(pid)
        out_players[str(pid)] = dict(name=p.name if p else str(pid), team=p.team if p else None, share=round(100.0 * c / max(1, n_ok), 1),
                                     in_mean=bool(mean_ids and pid in mean_ids))
    for i in (mean_ids or []):
        out_players.setdefault(str(int(i)), dict(name=by_pid[int(i)].name, team=by_pid[int(i)].team, share=0.0, in_mean=True))
    mean_opt = None if not mean_ids else dict(ids=[int(i) for i in mean_ids], total=round(float(sum(means[[k for k, p in enumerate(players) if int(p.id) in set(mean_ids)]])), 2),
                                              names=[by_pid[int(i)].name for i in mean_ids])
    return dict(at=datetime.now().isoformat(timespec="seconds"), stamp=list(stamp), n_worlds=int(n_ok), worlds_asked=int(n_worlds), players=out_players,
                mean_optimal=mean_opt, missing_teams=missing, floor=floor, draft_group=S.current())
