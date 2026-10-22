"""
Walk-forward cases for the projection: one case per team-game with the roster,
availability and starters taken from the box score (oracle), so the numbers
measure the minutes / rate model and not injury-news timing. Shared by
scripts/nba/backtest_projection.py and the team minutes fit.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .engine import ProjectionParams, RosterPlayer, project_team
from .history import STATS, HistoryState

ROSTER_WINDOW_GAMES = 10   # a player seen in the team's last N games is "on the roster"
STAR_BASELINE = 28.0       # a missing player projected above this makes it a star-out game


def team_games(box: pd.DataFrame, start: date, end: date) -> list[tuple[str, str, date]]:
    tg = box[["team", "game_id", "game_date"]].drop_duplicates()
    tg = tg[(tg.game_date >= start) & (tg.game_date <= end)]
    return list(tg.sort_values(["game_date", "game_id"]).itertuples(index=False, name=None))


def build_cases(box: pd.DataFrame, start: date, end: date, dnp_available: bool = False) -> list[dict]:
    """One case per team-game: roster with availability/starter flags and actuals."""
    played = box[box.played]
    recent = {}  # team -> list of (game_date, game_id, set(players))
    for (team, gid, d), g in played.groupby(["team", "game_id", "game_date"], sort=False):
        recent.setdefault(team, []).append((d, gid, set(g.player_name)))
    for t in recent:
        recent[t].sort()
    cases = []
    for team, gid, d in team_games(box, start, end):
        today = box[(box.team == team) & (box.game_id == gid)]
        prior = [p for p in recent[team] if p[0] < d][-ROSTER_WINDOW_GAMES:]
        roster_names = set().union(*[p[2] for p in prior]) if prior else set()
        roster_names |= set(today.player_name)
        act = today.set_index("player_name")
        roster = []
        for n in sorted(roster_names):
            if n in act.index and act.loc[n, "minutes"] > 0:
                roster.append(RosterPlayer(n, True, bool(act.loc[n, "starter"] > 0)))
            elif dnp_available and n in act.index:
                # dressed but did not play: live we cannot tell him from the rotation
                roster.append(RosterPlayer(n, True, False))
            else:
                roster.append(RosterPlayer(n, False, None))
        cases.append(dict(team=team, game_id=gid, date=d, roster=roster,
                          actual=act[act.minutes > 0][["minutes", "dk_fpts"] + STATS]))
    return cases


def run(cases, hist: HistoryState, params: ProjectionParams, simulate=False) -> pd.DataFrame:
    rows = []
    cache = {}
    for c in cases:
        d = c["date"]
        if d not in cache:
            cache = {d: hist.before(d)}  # cases are date-sorted; keep one
        state = cache[d]
        team_fpts = hist.team_before(d).get(c["team"])
        proj = project_team(c["roster"], state, hist, team_fpts, params, simulate=simulate)
        star_out = any((not p.available) and (p.name in state.index) and state.loc[p.name, "min_ewm"] >= STAR_BASELINE
                       for p in c["roster"])
        j = proj.join(c["actual"], how="inner", rsuffix="_act")
        j["starter_flag"] = [bool(next((p.starter for p in c["roster"] if p.name == n), False)) for n in j.index]
        j["team"], j["game_id"], j["date"], j["star_out"] = c["team"], c["game_id"], d, star_out
        rows.append(j.reset_index())
    return pd.concat(rows, ignore_index=True)


def metrics(df: pd.DataFrame, label: str) -> dict:
    d = df[df.games >= 5]  # players with some history; rookies' first games are noise for every method
    em = d.minutes - d.minutes_act
    ef = d.fpts_mean - d.dk_fpts
    return dict(label=label, n=len(d),
                min_mae=em.abs().mean(), min_bias=em.mean(),
                fpts_mae=ef.abs().mean(), fpts_rmse=np.sqrt((ef ** 2).mean()), fpts_bias=ef.mean(),
                corr=np.corrcoef(d.fpts_mean, d.dk_fpts)[0, 1])


def fmt(ms: list[dict]) -> str:
    t = pd.DataFrame(ms).set_index("label")
    return t.to_string(float_format=lambda v: f"{v:7.3f}")


