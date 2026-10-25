# src/ivan/sports/nba/projection/groups.py
"""
Position groups and each team's minutes budget by group: the structure the minutes solve respects.

group_of(position)     big (centre-eligible) | guard (point-guard-eligible) | wing (the rest), from DK eligibility
position_map(session)  player -> DK positions, from every pool this season holds (archived captures, the live pool) and
                       last season's projection snapshots, so past box-score rows can be grouped
group_targets(box, team, before, posmap)
                       the team's trailing minutes per group (half-life GROUP_HALFLIFE games) and a spread: a soft target
                       for the solve, once the team has GROUP_MIN_GAMES games. Measured 2026-10-23 over 1,236 team-games: a trailing average misses the next game's
                       group minutes by 8 / 11 / 13 (big / guard / wing) against 13.5 / 15 / 19 for the league mean, and
                       group totals swing 10-16 minutes game to game, so the spread floor is GROUP_SD_FLOOR.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ivan.core.utils import normalize_name

GROUP_HALFLIFE = 10.0
GROUP_SD_FLOOR = 8.0
GROUP_MIN_GAMES = 8        # a team's group budgets act only with this much history: on 3-6 games (2026-10-24 backtest, this
                           # season) they cost 0.1-0.2 minutes of accuracy; on last season's full history they gained it back
GROUPS = ("big", "guard", "wing")


def group_of(position) -> str:
    e = {str(p).strip().upper() for p in (position if isinstance(position, (list, tuple, set)) else str(position or "").split("/"))}
    return "big" if "C" in e else ("guard" if "PG" in e else "wing")


def position_map(session) -> Dict[str, str]:
    """normalized player name -> the DK position string seen most often for him."""
    from sqlalchemy import text
    seen: Dict[str, Counter] = {}
    for name, pos in session.execute(text("select name, position from nba_player")).all():
        seen.setdefault(normalize_name(name), Counter())[str(pos)] += 3           # the live pool outweighs old labels
    try:
        for name, pos in session.execute(text("select player_name, position from projection_snapshot where position is not null")).all():
            seen.setdefault(normalize_name(name), Counter())[str(pos)] += 1
    except Exception:   # noqa: BLE001 - last season's table may be absent on a fresh install
        pass
    try:
        from ivan.core import archive
        for cap in archive.captures(session, "nba")[-60:]:
            for p in archive.load_blobs(session, cap.id).get("players", []):
                if p.get("position"):
                    seen.setdefault(normalize_name(str(p.get("name"))), Counter())[str(p["position"])] += 1
    except Exception:   # noqa: BLE001
        pass
    return {n: c.most_common(1)[0][0] for n, c in seen.items()}


def group_targets(box: pd.DataFrame, team: str, before: date, posmap: Dict[str, str], halflife: float = GROUP_HALFLIFE) -> Dict[str, tuple]:
    """group -> (target minutes, sd) for `team` from its games before `before`; {} with fewer than GROUP_MIN_GAMES games."""
    g = box[(box.team == team) & (box.game_date < before)]
    if g.empty:
        return {}
    g = g.assign(grp=g.player_name.map(lambda n: group_of(posmap.get(n, ""))))
    per = g.groupby(["game_date", "grp"]).minutes.sum().unstack(fill_value=0.0).sort_index()
    if len(per) < GROUP_MIN_GAMES:
        return {}
    out = {}
    for k in GROUPS:
        if k not in per:
            continue
        x = per[k].to_numpy(dtype=float); w = np.power(0.5, (len(x) - 1 - np.arange(len(x))) / halflife)
        sd = float(np.std(x[-15:])) if len(x) >= 5 else 12.0
        out[k] = (float((w * x).sum() / w.sum()), max(GROUP_SD_FLOOR, sd))
    return out


DEPTH_MIN_MINUTES = 5.0    # a player with this many minutes counts toward the depth (the 5-minute garbage-time man does not)
DEPTH_GAMES = 10
DEPTH_HALFLIFE = 4.0
DEPTH_FLOOR, DEPTH_CEIL = 8, 12


def rotation_depth(box: pd.DataFrame, team: str, before: Optional[date] = None, default: int = 10) -> int:
    """
    How many players the coach uses: the recency-weighted mean count of players with DEPTH_MIN_MINUTES+ minutes over the
    team's last DEPTH_GAMES games (league 2025-26: mean 10.0, p10 9, p90 12), rounded, within DEPTH_FLOOR..DEPTH_CEIL.
    `default` when the team has no games yet.
    """
    g = box[(box.team == team) & box.played]
    if before is not None:
        g = g[g.game_date < before]
    if g.empty:
        return default
    per_game = g[g.minutes >= DEPTH_MIN_MINUTES].groupby(["game_date", "game_id"]).size().sort_index()
    if per_game.empty:
        return default
    per_game = per_game.iloc[-DEPTH_GAMES:]
    w = np.array([0.5 ** ((len(per_game) - 1 - i) / DEPTH_HALFLIFE) for i in range(len(per_game))])
    return int(np.clip(round(float((per_game.values * w).sum() / w.sum())), DEPTH_FLOOR, DEPTH_CEIL))
