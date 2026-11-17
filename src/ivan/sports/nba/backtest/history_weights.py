# src/ivan/sports/nba/backtest/history_weights.py
"""
How a player's trailing history should be weighted (projection.history.build_history): the half-lives of his minutes and
of his per-minute rates, and what his games before a change of team or of season stay worth.

Walk-forward on the box scores alone: for every game a player played from `start`, the state strictly before it predicts
  minutes   min_ewm against the minutes he played
  fpts      his shrunk per-minute rates times the minutes he really played, against his fantasy points (the rates alone)
scored as MAE / bias over all player-games, over a player's first NEW_GAMES games after a change of team, and over his
first NEW_GAMES games of a season. `compare` runs a list of settings; scripts/nba/history_weights.py prints it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ivan.sports.nba.projection import history as H
from ivan.sports.nba.projection.rates import fpts_per_minute, shrunk_rates

NEW_GAMES = 15
MIN_GAMES = 5
OLD = dict(min_halflife=5, rate_halflife=20, min_team_keep=1.0, min_season_keep=1.0, rate_team_keep=1.0)     # the weighting until 2026-11-16
SETTINGS: List[Tuple[str, Dict]] = [
    ("no discount, 5 / 20", OLD),
    ("live", {}),
    ("live, minutes half-life 5", dict(min_halflife=5)),
    ("live, rate half-life 20", dict(rate_halflife=20)),
    ("live, nothing kept of another team", dict(min_team_keep=0.0, rate_team_keep=0.0)),
    ("live, team change only", dict(min_season_keep=1.0)),
    ("half-lives only (3 / 10)", dict(min_team_keep=1.0, min_season_keep=1.0, rate_team_keep=1.0)),
]


def frame(box: pd.DataFrame, start: Optional[str] = None, **kw) -> pd.DataFrame:
    """One row per game played from `start`: the actuals beside what the state before the game says."""
    hist = H.build_history(box, **kw)
    rows = hist.rows.copy()
    rows["fpm"] = [fpts_per_minute(shrunk_rates(r, hist.league_rates)) for _, r in rows.iterrows()]
    rows["t"] = pd.to_datetime(rows.game_date)
    p = box[box.played].copy(); p["t"] = pd.to_datetime(p.game_date)
    p = p.sort_values(["player_name", "t", "game_id"])
    g = p.groupby("player_name")
    p["new_team"] = (g.team.shift().notna() & (p.team != g.team.shift())).astype(int)
    stretch = g.new_team.cumsum()
    p["since_team"] = p.groupby(["player_name", stretch]).cumcount(); p["moved"] = stretch > 0
    p["season"] = p.game_date.map(H.season_of); p["season_n"] = p.groupby(["player_name", "season"]).cumcount()
    d = pd.merge_asof(p.sort_values("t"), rows[["player_name", "t", "min_ewm", "fpm", "games"]].sort_values("t"), on="t", by="player_name", allow_exact_matches=False)
    d = d[d.games >= MIN_GAMES]
    if start:
        d = d[d.t >= pd.Timestamp(start)]
    d["e_min"] = d.min_ewm - d.minutes; d["e_fpts"] = d.fpm * d.minutes - d.dk_fpts
    return d


def score(d: pd.DataFrame) -> Dict[str, float]:
    out = {"n": len(d)}
    sub = {"all": d, "new_team": d[d.moved & (d.since_team < NEW_GAMES)], "new_season": d[(d.season_n < NEW_GAMES) & (d.season > d.season.min())]}
    for k, x in sub.items():
        out[f"n_{k}"] = len(x)
        for col in ("min", "fpts"):
            out[f"{col}_mae_{k}"] = float(x[f"e_{col}"].abs().mean()); out[f"{col}_bias_{k}"] = float(x[f"e_{col}"].mean())
    return out


def compare(box: pd.DataFrame, start: Optional[str] = "2025-11-15", settings: Optional[List[Tuple[str, Dict]]] = None) -> pd.DataFrame:
    return pd.DataFrame([{"setting": name, **score(frame(box, start, **kw))} for name, kw in (settings or SETTINGS)]).set_index("setting")
