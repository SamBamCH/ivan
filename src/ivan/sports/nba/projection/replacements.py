# src/ivan/sports/nba/projection/replacements.py
"""
Who comes in for whom: substitution pairs from the stints (pbp.py).

A substitution is a stint ending mid-period at the second another teammate's stint begins: "in for out". When several
players change at the same second, each out-in pairing gets an equal share (k outs and k ins: 1/k each), so the credit
is never invented. Period breaks are left out: who sits at a break is not a substitution decision of the same kind.

pair_events(stints)            one row per (game, team, out, in, weight)
replacement_shares(events)     per team and outgoing player: the share of his exits each teammate covered,
                               recency weighted (half-life in games)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

PERIOD_SEC = 720.0


def _period_edges(t: float) -> bool:
    """True at a period boundary (0, 720, ..., 2880, then overtime every 300 s)."""
    if t <= 2880.0 + 1e-6:
        return abs(t / PERIOD_SEC - round(t / PERIOD_SEC)) < 1e-6
    return abs((t - 2880.0) / 300.0 - round((t - 2880.0) / 300.0)) < 1e-6


def pair_events(stints: pd.DataFrame) -> pd.DataFrame:
    """stints: game_id, game_date, team, player_name, start_sec, end_sec. Returns game_id, game_date, team, out, inn, weight, at."""
    rows = []
    for (gid, team), d in stints.groupby(["game_id", "team"], sort=False):
        date = d.game_date.iloc[0] if "game_date" in d else None
        ends = d.groupby(d.end_sec.round(1)).player_name.apply(list)
        starts = d.groupby(d.start_sec.round(1)).player_name.apply(list)
        for t, outs in ends.items():
            if _period_edges(float(t)) or t not in starts.index:
                continue
            ins = [p for p in starts.loc[t] if p not in outs]
            outs = [p for p in outs if p not in ins]
            if not outs or not ins:
                continue
            w = 1.0 / max(len(outs), len(ins))
            for o in outs:
                for i in ins:
                    rows.append((gid, date, team, o, i, w, float(t)))
    return pd.DataFrame(rows, columns=["game_id", "game_date", "team", "out", "inn", "weight", "at"])


def replacement_shares(events: pd.DataFrame, before=None, halflife_games: float = 15.0, team: Optional[str] = None) -> pd.DataFrame:
    """team, out, inn, share (of the outgoing player's weighted exits), exits (weighted count): from games before `before`."""
    ev = events if before is None else events[pd.to_datetime(events.game_date) < pd.Timestamp(before)]
    if team is not None:
        ev = ev[ev.team == team]
    if ev.empty:
        return pd.DataFrame(columns=["team", "out", "inn", "share", "exits"])
    # recency: games numbered back from the newest per team
    order = ev[["team", "game_id", "game_date"]].drop_duplicates().sort_values(["team", "game_date", "game_id"])
    order["age"] = order.groupby("team").cumcount(ascending=False)
    ev = ev.merge(order[["team", "game_id", "age"]], on=["team", "game_id"])
    ev["w"] = ev.weight * np.power(0.5, ev.age / float(halflife_games))
    g = ev.groupby(["team", "out", "inn"]).w.sum().reset_index()
    tot = g.groupby(["team", "out"]).w.transform("sum")
    g["share"] = g.w / tot; g["exits"] = tot
    return g.rename(columns={"w": "weight"}).sort_values(["team", "out", "share"], ascending=[True, True, False]).reset_index(drop=True)
