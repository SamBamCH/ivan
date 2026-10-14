# src/ivan/sports/nfl/actuals.py
"""
What the players actually scored, in DraftKings points, from the history
tables the overnight job keeps current (nfl_stats, nfl_dst_game, nfl_schedule).

    week_for_date(session, when)              -> (season, week) a slate date falls in
    actual_points(session, season, week)      -> {(normalized name, team): fpts} for offense/K, {("DST", team): fpts}

Team codes follow DraftKings (LAR, not nflverse's LA). Names are normalized
with ivan.core.utils.normalize_name on both sides.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from ivan.core import clock
from ivan.core.utils import normalize_name
from ivan.database.session import read_sql

# nflverse -> DraftKings team codes (everything else is identical)
TEAM_TO_DK = {"LA": "LAR"}
DK_TO_NFLVERSE = {v: k for k, v in TEAM_TO_DK.items()}


def dk_team(code: Optional[str]) -> Optional[str]:
    return TEAM_TO_DK.get(code, code) if code else code


def dk_offense_points(st: pd.DataFrame) -> pd.Series:
    """DraftKings classic scoring for QB/RB/WR/TE rows of nfl_stats, bonuses included."""
    n = lambda c: pd.to_numeric(st[c], errors="coerce").fillna(0.0)
    pass_yds, rush_yds, rec_yds = n("passing_yards"), n("rushing_yards"), n("receiving_yards")
    return (0.04 * pass_yds + 4 * n("passing_tds") - 1 * n("passing_interceptions")
            + 0.1 * rush_yds + 6 * n("rushing_tds") + n("receptions") + 0.1 * rec_yds + 6 * n("receiving_tds")
            - (n("rushing_fumbles_lost") + n("receiving_fumbles_lost") + n("sack_fumbles_lost"))
            + 2 * (n("passing_2pt_conversions") + n("rushing_2pt_conversions") + n("receiving_2pt_conversions"))
            + 6 * (n("special_teams_tds") + n("fumble_recovery_tds"))
            + 3 * (pass_yds >= 300) + 3 * (rush_yds >= 100) + 3 * (rec_yds >= 100))


def dk_kicker_points(st: pd.DataFrame) -> pd.Series:
    n = lambda c: pd.to_numeric(st[c], errors="coerce").fillna(0.0)
    return (3 * (n("fg_made_0_19") + n("fg_made_20_29") + n("fg_made_30_39")) + 4 * n("fg_made_40_49")
            + 5 * (n("fg_made_50_59") + n("fg_made_60_")) + n("pat_made"))


def week_lookup(session) -> pd.DataFrame:
    """(season, week, first, last) gameday per schedule week, sorted."""
    s = read_sql(session, text("SELECT season, week, min(gameday) first, max(gameday) last FROM nfl_schedule GROUP BY 1,2"))
    return s.sort_values(["season", "week"]).reset_index(drop=True)


def week_for_date(session, when: datetime | date, weeks: Optional[pd.DataFrame] = None) -> Optional[Tuple[int, int]]:
    """The first schedule week whose last game is on or after `when` (a date, or a datetime whose date is used)."""
    weeks = week_lookup(session) if weeks is None else weeks
    d = when.date().isoformat() if isinstance(when, datetime) else when.isoformat()
    cand = weeks[weeks["last"] >= d]
    if cand.empty:
        return None
    row = cand.iloc[0]
    return int(row.season), int(row.week)


def history_cutoff(session) -> Optional[Tuple[int, int]]:
    """
    (season, week) of the slate a replay is pinned to, or None when live.
    History is every game in a week strictly before it: the stats the
    overnight job would have had when that slate was played.
    """
    when = clock.as_of()
    if when is None:
        return None
    return week_for_date(session, when)


def before_cutoff(df, before: Optional[Tuple[int, int]]):
    """Rows of a frame with season/week columns played before (season, week); all rows when before is None."""
    if before is None:
        return df
    s, w = before
    return df[(df.season < s) | ((df.season == s) & (df.week < w))]


def actual_points(session, season: int, week: int) -> Dict[Tuple[str, str], float]:
    """
    DraftKings points for every player-week and defense of one schedule week,
    keyed (normalized name, DK team code); defenses as ("DST", team). Missing
    entirely for a week the overnight job has not fetched yet.
    """
    st = _stats_week(session, season, week)
    out: Dict[Tuple[str, str], float] = {}
    if not st.empty:
        st = st.copy()
        st["pts"] = np.where(st.position == "K", dk_kicker_points(st), dk_offense_points(st))
        for name, team, pts in zip(st.player_display_name, st.team, st.pts):
            out[(normalize_name(str(name)), dk_team(team))] = float(pts)
    dst = read_sql(session, text("SELECT team, dst_points FROM nfl_dst_game WHERE season = :s AND week = :w")
                   .bindparams(s=season, w=week))
    for team, pts in zip(dst.team, dst.dst_points):
        if pts is not None and not (isinstance(pts, float) and np.isnan(pts)):
            out[("DST", dk_team(team))] = float(pts)
    return out


def _stats_week(session, season: int, week: int) -> pd.DataFrame:
    return read_sql(session, text("SELECT * FROM nfl_stats WHERE season = :s AND week = :w").bindparams(s=season, w=week))


SUFFIXES = (" jr", " sr", " ii", " iii", " iv", " v")


def name_variants(name: str) -> list[str]:
    """The normalized name and, when it carries a generational suffix, the name without it (nflverse drops them)."""
    nm = normalize_name(name)
    out = [nm]
    for s in SUFFIXES:
        if nm.endswith(s):
            out.append(nm[: -len(s)].strip())
    return out


def lookup_points(table: Dict[Tuple[str, str], float], name: str, team: str, position: str) -> Optional[float]:
    """A player's actual points, by (name, team) first, then by name alone (trades, code drift)."""
    if position.upper() in ("DST", "D/ST", "DEF"):
        return table.get(("DST", team))
    for nm in name_variants(name):
        if (nm, team) in table:
            return table[(nm, team)]
    for nm in name_variants(name):
        hits = [v for (n, t), v in table.items() if n == nm and t != "DST"]
        if len(hits) == 1:
            return hits[0]
    return None


def team_played(table: Dict[Tuple[str, str], float], team: str) -> bool:
    """Whether the week's table holds any stat line for this team (so a missing player scored 0, not 'unknown')."""
    return any(t == team for (_, t) in table)
