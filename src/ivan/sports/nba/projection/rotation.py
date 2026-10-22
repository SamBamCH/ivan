"""
Rotations as a grid: who is on the floor in each minute of the game.

curves(session, team)      one row per player, 48 columns: the recency-weighted
                           share of recent games he was on the floor during
                           that minute (from the stint table, pbp.py)
typical_stints(row)        the contiguous minutes where that share is above a half
propose(curves, available, starters)
                           a full rotation for tonight: exactly five players in
                           every minute, chosen minute by minute by their curve
                           with hysteresis so stints hold together, starters
                           forced on at the tip, scratched players never on,
                           and a player's minutes capped near his own history
minutes_of(grid)           minutes per player (the row sums)
save_drawn / load_drawn    a hand-drawn grid per team (data/nba/input/rotations/<team>.json):
                           what the desk writes when a rotation is applied and
                           what the projection reads as pinned minutes

The grid is boolean players x 48; the sampler reads stints and closers off it.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.sports.nba.paths import INPUT_DIR

MINUTES = 48
ROTATIONS_DIR = INPUT_DIR / "rotations"
ON_COURT = 5
HYSTERESIS = 0.15          # a player on the floor stays until someone's curve beats his by this
MIN_STINT = 2              # minutes: shorter stints are merged away in the proposal
DEFICIT = 0.5              # weight of the minutes a player is still owed against his curve


def curves(session: Session, team: str, before: Optional[date] = None, n_games: int = 10, halflife: float = 4.0) -> pd.DataFrame:
    """index player, columns 0..47 (minute of regulation), values 0..1; plus games, minutes (expected), stints."""
    q = text("select game_id, game_date, player_name, start_sec, end_sec, minutes, to_buzzer, margin_out from nba_stint where team = :t"
             + (" and game_date < :d" if before else "") + " order by game_date, game_id")
    df = pd.read_sql(q.bindparams(t=team, **({"d": before.isoformat()} if before else {})), session.get_bind())
    if df.empty:
        return pd.DataFrame()
    games = list(dict.fromkeys(df.game_id))[-n_games:]
    df = df[df.game_id.isin(games)]
    w = {g: 0.5 ** ((len(games) - 1 - i) / halflife) for i, g in enumerate(games)}
    wsum = sum(w.values())
    rows = {}
    for name, g in df.groupby("player_name"):
        on = np.zeros(MINUTES); n_st = []
        for gid, s in g.groupby("game_id"):
            cur = np.zeros(MINUTES)
            for a, b in zip(s.start_sec, s.end_sec):
                lo, hi = int(a // 60), int(min(b, 2880) // 60 + (1 if min(b, 2880) % 60 > 0 else 0))
                for m in range(max(0, lo), min(MINUTES, hi)):
                    cur[m] = max(cur[m], min(1.0, (min(b, (m + 1) * 60) - max(a, m * 60)) / 60.0))
            on += w[gid] * cur; n_st.append(len(s))
        row = {m: on[m] / wsum for m in range(MINUTES)}
        row.update(games=len(n_st), minutes=float(on.sum() / wsum), stints=float(np.mean(n_st)), played_share=float(sum(w[g] for g in g.game_id.unique()) / wsum))
        rows[name] = row
    out = pd.DataFrame(rows).T
    return out.sort_values("minutes", ascending=False)


def typical_stints(row: Sequence[float], threshold: float = 0.5) -> List[Tuple[int, int]]:
    """[(start minute, end minute)) runs where the on-court share is above threshold."""
    on = np.asarray(row[:MINUTES], dtype=float) > threshold
    out, start = [], None
    for m, v in enumerate(on):
        if v and start is None:
            start = m
        if not v and start is not None:
            out.append((start, m)); start = None
    if start is not None:
        out.append((start, MINUTES))
    return out


def propose(curve: pd.DataFrame, available: Iterable[str], starters: Optional[Iterable[str]] = None, salary_prior: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    A boolean grid (index player, columns 0..47) with exactly five on the floor
    every minute. Players without a curve (no history on this team) get a flat
    prior from salary_prior minutes (spread evenly) so they can be drawn in.
    """
    avail = [n for n in available]
    starters = set(starters or [])
    P = np.zeros((len(avail), MINUTES))
    for i, n in enumerate(avail):
        if n in curve.index:
            P[i] = curve.loc[n, list(range(MINUTES))].values.astype(float)
        elif salary_prior and salary_prior.get(n):
            P[i] = min(0.95, float(salary_prior[n]) / MINUTES)
    # every player's target is his expected minutes, scaled so the available roster adds to 240; the minute-by-minute
    # pick follows his curve (when he plays) and his remaining target (how much he still has to play)
    hist = np.array([float(curve.loc[n, "minutes"]) if n in curve.index else float((salary_prior or {}).get(n, 4.0)) for n in avail])
    target = hist * (ON_COURT * MINUTES / max(1e-9, hist.sum()))
    cap = np.minimum(MINUTES, target * 1.1 + 2)
    grid = np.zeros((len(avail), MINUTES), dtype=bool)
    on = set()
    if starters:
        on = {i for i, n in enumerate(avail) if n in starters}
        if len(on) > ON_COURT:
            on = set(sorted(on, key=lambda i: -P[i, 0])[:ON_COURT])
    used = np.zeros(len(avail))
    for m in range(MINUTES):
        left = MINUTES - m
        need = np.maximum(target - used, 0.0)                      # minutes still owed
        score = P[:, m] + DEFICIT * (need / np.maximum(target, 1.0)) + 0.6 * np.clip(need - left * 0.8, 0, None) / max(1.0, left)
        score[used >= cap] = -1.0                                  # out of minutes
        if m == 0 and on:
            cur = set(on)
            while len(cur) < ON_COURT:
                rest = [i for i in np.argsort(-score) if i not in cur]
                cur.add(int(rest[0]))
        else:
            cur = set(on)
            # keep the floor unless a bench player's curve beats a floor player's by the hysteresis
            for _ in range(ON_COURT):
                if not cur:
                    break
                worst = min(cur, key=lambda i: score[i])
                best_off = max((i for i in range(len(avail)) if i not in cur), key=lambda i: score[i], default=None)
                if best_off is not None and score[best_off] > score[worst] + HYSTERESIS:
                    cur.remove(worst); cur.add(int(best_off))
                else:
                    break
            while len(cur) < ON_COURT:
                rest = [int(i) for i in np.argsort(-score) if i not in cur]
                if not rest:
                    break
                cur.add(rest[0])
            while len(cur) > ON_COURT:
                cur.remove(min(cur, key=lambda i: score[i]))
        for i in cur:
            grid[i, m] = True; used[i] += 1
        on = cur
    grid = _merge_short_stints(grid, P)
    return pd.DataFrame(grid, index=avail, columns=list(range(MINUTES)))


def _merge_short_stints(grid: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Remove stints shorter than MIN_STINT by handing their minutes to the best off-floor player at that time (keeps five on)."""
    g = grid.copy()
    for i in range(g.shape[0]):
        for a, b in typical_stints(g[i].astype(float)):
            if b - a < MIN_STINT:
                for m in range(a, b):
                    g[i, m] = False
                    off = [j for j in range(g.shape[0]) if not g[j, m] and j != i]
                    if off:
                        g[max(off, key=lambda j: P[j, m]), m] = True
    return g


def minutes_of(grid: pd.DataFrame) -> Dict[str, float]:
    return {str(n): float(v) for n, v in grid.sum(axis=1).items()}


def stints_of(grid: pd.DataFrame) -> Dict[str, List[Tuple[int, int]]]:
    return {str(n): typical_stints(grid.loc[n].values.astype(float)) for n in grid.index}


def column_counts(grid: pd.DataFrame) -> List[int]:
    return [int(x) for x in grid.sum(axis=0).values]


# ----------------------------------------------------------------------------- drawn rotations
def drawn_path(team: str) -> Path:
    return ROTATIONS_DIR / f"{team}.json"


def save_drawn(team: str, grid: pd.DataFrame, note: str = "") -> Path:
    p = drawn_path(team); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"team": team, "saved_at": datetime.now().isoformat(timespec="seconds"), "note": note,
                             "players": [str(n) for n in grid.index], "grid": [[int(x) for x in row] for row in grid.values]}, indent=0))
    return p


def load_drawn(team: str) -> Optional[pd.DataFrame]:
    p = drawn_path(team)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return pd.DataFrame(np.array(d["grid"], dtype=bool), index=d["players"], columns=list(range(MINUTES)))


def clear_drawn(team: str) -> bool:
    p = drawn_path(team)
    if p.exists():
        p.unlink(); return True
    return False


def drawn_minutes(team: str) -> Dict[str, float]:
    g = load_drawn(team)
    return minutes_of(g) if g is not None else {}


def format_grid(grid: pd.DataFrame, curve: Optional[pd.DataFrame] = None) -> str:
    L = [f"  {'player':22s} {'min':>4s}  " + "".join("|" if m % 12 == 0 else ("." if m % 6 == 0 else " ") for m in range(MINUTES))]
    for n in grid.index:
        row = "".join("#" if v else " " for v in grid.loc[n].values)
        L.append(f"  {n[:22]:22s} {grid.loc[n].sum():4.0f}  {row}" + (f"  hist {float(curve.loc[n, 'minutes']):.1f}" if curve is not None and n in curve.index else ""))
    L.append(f"  {'on floor':22s}       " + "".join(str(c) if c != ON_COURT else "5" for c in column_counts(grid)))
    return "\n".join(L)
