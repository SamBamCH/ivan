"""
ivan.sports.nba.projection.team_profile

The coach's pattern with his starters, per team, from the team's own ordinary games this season (final margin under 15,
five named starters) with the league's pattern underneath it: how long the starters hold the floor at the tip and at the
third-quarter tip before the first sub, how many starters are on the floor as the half closes, how many minutes a starter
and a bench player get. Measured 2026-10-26 on 898 ordinary team-games: first sub 5.3 minutes into Q1 (Portland 4.5, the
Clippers 6.4), 4.2 into Q3 (Memphis 1.2, the Clippers 5.7), 3.7 starters on at minute 22 (Golden State 3.1, Denver 4.4),
starters 31.1 minutes (Golden State 28.0, Houston 34.8), bench 16.1. A team's opening span and starter minutes repeat
across its own games (r = 0.72 and 0.68 between halves of a team's games); who closes does not (r = -0.03), so the closers
stay a league rule in the scheduler (the most-played five).

    profile(session, team, before=None)  -> {open_q1, open_q3, half_close, starter_min, bench_min, n, league: {...}}

Blended toward the league by n / (n + SHRINK_GAMES) so a team with four games is not its own law. Cached per process by
(team, before, since).
"""
from __future__ import annotations

from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

SHRINK_GAMES = 6
ORDINARY_MARGIN = 15
LEAGUE_FALLBACK = dict(open_q1=5.3, open_q3=4.2, half_close=3.7, starter_min=31.1, bench_min=16.1)
_CACHE: dict = {}


def _game_rows(session: Session, team: Optional[str], before: Optional[date], since: Optional[date]) -> pd.DataFrame:
    q = "select game_id, game_date, team, player_name, start_sec, end_sec, margin_out, starter, minutes from nba_stint where 1 = 1"
    kw = {}
    if team:
        q += " and team = :t"; kw["t"] = team
    if before:
        q += " and game_date < :b"; kw["b"] = before.isoformat()
    if since:
        q += " and game_date >= :s"; kw["s"] = since.isoformat()
    return pd.read_sql(text(q).bindparams(**kw), session.get_bind())


def _measure(st: pd.DataFrame) -> Optional[dict]:
    """The pattern over a set of team-games: per game the first-sub minutes, the starters on at 20-23, starter and bench minutes."""
    out = []
    for (gid, team), g in st.groupby(["game_id", "team"]):
        starters = set(g[g.starter == 1].player_name)
        if len(starters) != 5:
            continue
        last = g.sort_values("end_sec").iloc[-1]
        margin = abs(int(last.margin_out)) if pd.notna(last.margin_out) else 99
        if margin >= ORDINARY_MARGIN:
            continue
        row = np.zeros(48)
        for r in g.itertuples():
            if r.player_name in starters:
                a = int(np.floor(r.start_sec / 60 + 0.5)); b = int(np.floor(min(r.end_sec, 2880) / 60 + 0.5))
                row[max(0, a):min(48, b)] += 1
        first = next((m for m in range(48) if row[m] < 5), 48)
        q3 = next((m for m in range(24, 48) if row[m] < 5), 48) - 24
        tot = g.groupby("player_name").minutes.sum()
        smin = [float(tot[n]) for n in starters if n in tot]; bmin = [float(v) for n, v in tot.items() if n not in starters and v >= 1]
        out.append(dict(open_q1=first, open_q3=q3, half_close=float(row[20:24].mean()), starter_min=float(np.mean(smin)) if smin else np.nan,
                        bench_min=float(np.mean(bmin)) if bmin else np.nan))
    if not out:
        return None
    df = pd.DataFrame(out)
    return dict(open_q1=float(df.open_q1.mean()), open_q3=float(df.open_q3.mean()), half_close=float(df.half_close.mean()),
                starter_min=float(df.starter_min.mean()), bench_min=float(df.bench_min.mean()), n=int(len(df)))


def league(session: Session, before: Optional[date] = None) -> dict:
    key = ("__league__", before)
    if key not in _CACHE:
        m = _measure(_game_rows(session, None, before, None))
        _CACHE[key] = {**LEAGUE_FALLBACK, **(m or {})}
    return _CACHE[key]


def profile(session: Session, team: str, before: Optional[date] = None, since: Optional[date] = None) -> dict:
    """The team's pattern, shrunk toward the league by its number of ordinary games; the league's when it has none."""
    if since is None:
        d = before or date.today()
        since = date(d.year if d.month >= 10 else d.year - 1, 10, 1)
    key = (team, before, since)
    if key in _CACHE:
        return _CACHE[key]
    lg = league(session, before)
    m = _measure(_game_rows(session, team, before, since))
    if not m:
        out = {**{k: lg[k] for k in LEAGUE_FALLBACK}, "n": 0, "league": lg}
    else:
        w = m["n"] / (m["n"] + SHRINK_GAMES)
        out = {k: (w * m[k] + (1 - w) * lg[k]) if pd.notna(m[k]) else lg[k] for k in LEAGUE_FALLBACK}
        out.update(n=m["n"], league=lg)
    _CACHE[key] = out
    return out


def clear_cache() -> None:
    _CACHE.clear()
