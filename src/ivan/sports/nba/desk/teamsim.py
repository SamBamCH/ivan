# src/ivan/sports/nba/desk/teamsim.py
"""
A team's game simulated on demand, watched while it runs (the team page's Simulation view, scripts/nba/team_sim.py).

The Projections view sets the inputs (minutes, stat totals, starters); every edit writes the team into the pool. This
draws the team's game from the pool as it stands, in batches, and keeps a running summary a page can poll: per player
his drawn minutes (median, 10-90, how often he plays, how often under 8), the share of worlds he is on the floor each
minute, his fantasy points (mean, sd, 10-50-90, how often he reaches five times his salary) and the points, rebounds
and assists behind them; for the game the drawn total and margin against the line. A finished run is kept as the
baseline of the next, so each number carries its change since the inputs were last simulated.

start(dg, team, n)   begin (or restart) the run for the team's game in a background thread; one run per game, both
                     teams' pages read the same one.
view(dg, team, session)   what the page shows now: {"state": none | running | done | error, "n", "target", "rows",
                     "game", "stale", ...}.
run(session, team, n, batch, on_batch)   the same draw in the foreground (the script).
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.sports.nba.gamesim import world as W
from ivan.sports.nba.projection import worlds as WD

BATCH = 250
DEFAULT_N = 2000
MAX_N = 10000
VALUE_X = 5.0                # the 'reaches value' line: this many fantasy points per $1,000 of salary
UNDER_MIN = 8.0
BLOWOUT = 20.0

_RUNS: Dict[Tuple[int, frozenset], "Run"] = {}
_LOCK = threading.Lock()


@dataclass
class Run:
    home: str
    away: str
    target: int
    fingerprint: str
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    n: int = 0
    error: Optional[str] = None
    cancel: bool = False
    summary: Dict[str, dict] = field(default_factory=dict)          # team -> {"rows": [...], "game": {...}}
    baseline: Dict[str, dict] = field(default_factory=dict)         # the run before this one, the same shape

    @property
    def state(self) -> str:
        return "error" if self.error else ("done" if self.finished else "running")


class _Acc:
    """One team's worlds as they arrive."""
    def __init__(self, setup: W.TeamSetup):
        self.names = list(setup.names); self.pids = [int(x) for x in setup.pids]; self.base = np.asarray(setup.base, dtype=float)
        self.minutes: List[np.ndarray] = []; self.fpts: List[np.ndarray] = []
        self.stats = {s: [] for s in ("pts", "reb", "ast")}
        self.cover = np.zeros((len(self.names), W.MINUTES)); self.n = 0

    def add(self, setup: W.TeamSetup, acc) -> None:
        stats = {"pts": acc.pts(), "fg3m": acc.fg3m, "reb": acc.oreb + acc.dreb, "ast": acc.ast, "stl": acc.stl, "blk": acc.blk, "tov": acc.tov}
        self.minutes.append(np.asarray(acc.minutes, dtype=np.float32)); self.fpts.append(W.dk_points(stats).astype(np.float32))
        for s in self.stats:
            self.stats[s].append(np.asarray(stats[s], dtype=np.float32))
        self.cover += setup.F.sum(axis=0); self.n += setup.F.shape[0]

    def rows(self, salary: Dict[int, float], baseline: Optional[dict]) -> List[dict]:
        m = np.concatenate(self.minutes); f = np.concatenate(self.fpts)
        st = {s: np.concatenate(v).mean(axis=0) for s, v in self.stats.items()}
        before = {r["pid"]: r for r in (baseline or {}).get("rows", [])}
        out = []
        for j, nm in enumerate(self.names):
            x, y = m[:, j], f[:, j]
            sal = float(salary.get(self.pids[j]) or 0.0)
            r = dict(name=nm, pid=self.pids[j], target=float(self.base[j]), min_med=float(np.median(x)), min_mean=float(x.mean()), min_p10=float(np.percentile(x, 10)),
                     min_p90=float(np.percentile(x, 90)), plays=float((x > 0).mean()), under=float((x < UNDER_MIN).mean()),
                     fpts=float(y.mean()), sd=float(y.std()), p10=float(np.percentile(y, 10)), p50=float(np.percentile(y, 50)), p90=float(np.percentile(y, 90)),
                     pts=float(st["pts"][j]), reb=float(st["reb"][j]), ast=float(st["ast"][j]), salary=sal,
                     value=float((y >= VALUE_X * sal / 1000.0).mean()) if sal > 0 else None, cover=[round(float(c), 3) for c in self.cover[j] / max(self.n, 1)])
            b = before.get(self.pids[j])
            r["d_min"] = r["min_med"] - b["min_med"] if b else None
            r["d_fpts"] = r["fpts"] - b["fpts"] if b else None
            r["d_p90"] = r["p90"] - b["p90"] if b else None
            out.append(r)
        return sorted(out, key=lambda r: (-r["target"], -r["min_mean"]))


def fingerprint(game: W.GameIn) -> str:
    """What the draw depends on, in a few characters: each player's minutes, rates, scratch chance and whether the grid starts him, and the line."""
    parts = [f"{game.home}|{game.away}|{game.spread_home}|{game.total}"]
    for t in sorted(game.players):
        for p in sorted(game.players[t], key=lambda q: q.pid):
            parts.append(f"{p.pid}:{p.minutes:.2f}:{p.p_out:.2f}:{int(bool(p.cells and p.cells[0]))}:" + ",".join(f"{k}={v:.4f}" for k, v in sorted(p.rates.items())))
    return hashlib.md5("\n".join(parts).encode()).hexdigest()[:12]


def game_of(session: Session, ctx: W.WorldContext, team: str) -> Optional[W.GameIn]:
    for g in W.slate_inputs(session, ctx.teams):
        if team in (g.home, g.away):
            return g
    return None


def _salaries(session: Session) -> Dict[int, float]:
    return {int(i): float(s or 0.0) for i, s in session.execute(text("select id, salary from nba_player")).all()}


def _summarise(accs: Dict[str, _Acc], game: W.GameIn, margins: List[np.ndarray], totals: List[np.ndarray], ot: List[np.ndarray], salary: Dict[int, float],
               baseline: Dict[str, dict]) -> Dict[str, dict]:
    mg = np.concatenate(margins); tt = np.concatenate(totals); o = np.concatenate(ot)
    out = {}
    for t, a in accs.items():
        side = mg if t == game.home else -mg
        line = None if game.spread_home is None else (-game.spread_home if t == game.home else game.spread_home)
        rows = a.rows(salary, baseline.get(t))
        out[t] = dict(rows=rows, game=dict(total=float(tt.mean()), total_line=game.total, margin=float(side.mean()), margin_line=line, sd_margin=float(side.std()),
                                           p_win=float((side > 0).mean()), p_ot=float((o > 0).mean()), blowout=float((np.abs(side) >= BLOWOUT).mean()),
                                           team_fpts=float(sum(r["fpts"] for r in rows)), minutes=float(sum(r["min_mean"] for r in rows))))
    return out


def run(session: Session, team: str, n: int = DEFAULT_N, batch: int = BATCH, on_batch: Optional[Callable[[Run], None]] = None, ctx: Optional[W.WorldContext] = None,
        state: Optional[Run] = None, seed: Optional[int] = None) -> Run:
    """Draw the team's game n worlds in batches from the pool as it stands; `state` is filled as the batches arrive (a new Run when absent)."""
    ctx = ctx or WD.context(session)
    game = game_of(session, ctx, team)
    if game is None:
        raise ValueError(f"{team} has no game in this slate's pool")
    r = state or Run(home=game.home, away=game.away, target=int(n), fingerprint=fingerprint(game))
    salary = _salaries(session)
    accs: Dict[str, _Acc] = {}; margins, totals, ot = [], [], []
    base = WD.game_seed(game.home, game.away) if seed is None else int(seed)
    b = 0
    while r.n < r.target and not r.cancel:
        k = min(batch, r.target - r.n)
        d = ctx.draw_game(game, k, np.random.default_rng([base, b]))
        for t in (game.home, game.away):
            accs.setdefault(t, _Acc(d["setups"][t])).add(d["setups"][t], d["accs"][t])
        margins.append(np.asarray(d["margin"], dtype=float)); totals.append(np.asarray(d["total"], dtype=float)); ot.append(np.asarray(d["overtime"]))
        summary = _summarise(accs, game, margins, totals, ot, salary, r.baseline)
        r.summary = summary; r.n += k; b += 1
        if on_batch:
            on_batch(r)
    r.finished = time.time()
    return r


def start(dg: int, team: str, n: int = DEFAULT_N) -> Run:
    """Begin the run for the team's game in the background (a run already going for the game is stopped; a finished one becomes the baseline)."""
    from ivan.sports.nba import slates as S
    n = int(max(BATCH, min(int(n or DEFAULT_N), MAX_N)))
    session = S.session(dg)
    try:
        ctx = WD.context(session)
        game = game_of(session, ctx, team)
        if game is None:
            raise ValueError(f"{team} has no game in this slate's pool")
        key = (int(dg), frozenset((game.home, game.away)))
        with _LOCK:
            old = _RUNS.get(key)
            if old is not None and old.state == "running":
                old.cancel = True
            baseline = old.summary if old is not None and old.state == "done" else (old.baseline if old is not None else {})
            r = Run(home=game.home, away=game.away, target=n, fingerprint=fingerprint(game), baseline=baseline)
            _RUNS[key] = r
    except Exception:
        session.close(); raise

    def work():
        try:
            run(session, team, n, ctx=ctx, state=r)
        except Exception as e:   # noqa: BLE001 - the page shows it
            r.error = f"{type(e).__name__}: {e}"; r.finished = time.time()
        finally:
            session.close()
    threading.Thread(target=work, name=f"teamsim-{dg}-{team}", daemon=True).start()
    return r


def view(dg: int, team: str, session: Optional[Session] = None) -> dict:
    """The Simulation view's state for the team. With a session and a finished run, `stale` says whether the pool's inputs for the game changed since."""
    with _LOCK:
        r = next((v for (d, pair), v in _RUNS.items() if d == int(dg) and team in pair), None)
    if r is None:
        return dict(state="none", team=team, n=0, target=DEFAULT_N, rows=[], game={}, stale=False, has_baseline=False, value_x=VALUE_X, under=UNDER_MIN)
    s = r.summary.get(team) or {}
    stale = False
    if session is not None and r.state == "done":
        try:
            g = game_of(session, WD.context(session), team)
            stale = g is not None and fingerprint(g) != r.fingerprint
        except Exception:   # noqa: BLE001 - the flag is a convenience
            stale = False
    end = r.finished or time.time()
    return dict(state=r.state, team=team, opp=r.away if team == r.home else r.home, n=r.n, target=r.target, rows=s.get("rows", []), game=s.get("game", {}), stale=stale,
                error=r.error, seconds=end - r.started, has_baseline=bool(r.baseline.get(team)), value_x=VALUE_X, under=UNDER_MIN)
