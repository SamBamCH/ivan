# src/ivan/sports/nba/gamesim/profiles.py
"""
Each player's shooting profile from the box scores, as of any date: how he scores his points.

The pool's per-minute rates (history or the books) say how many points, rebounds, assists, steals, blocks, turnovers and
threes a player is expected to produce; they do not say how. Layer D needs the attempts behind the points and the
percentages that turn attempts into makes:

    fga_pm, fg3a_pm, fta_pm   attempts, three-point attempts and free throws per minute
    fg2_pct, fg3_pct, ft_pct  makes per attempt, by kind
    oreb_frac                 the share of his rebounds that are offensive

Trailing (half-life HALFLIFE games), shrunk toward the league with PRIOR_MINUTES pseudo-minutes for the rates and
PRIOR_ATTEMPTS pseudo-attempts for the percentages. ShootingProfiles.build(box) walks nba_stats once; before(date) is the
state as of a date; profile(state, player) the shrunk numbers for one player.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

HALFLIFE = 20.0
PRIOR_MINUTES = 300.0
PRIOR_ATTEMPTS = 100.0
COUNTS = ["minutes", "fga", "fg3a", "fta", "fgm", "fg3m", "ftm", "rebounds_off", "rebounds_def"]
KEYS = ("fga_pm", "fg3a_pm", "fta_pm", "fg2_pct", "fg3_pct", "ft_pct", "oreb_frac")


def load_box(session: Session, before: Optional[date] = None) -> pd.DataFrame:
    """nba_stats rows with the shooting columns, names normalized, real games only."""
    from ivan.sports.nba.names import normalize_name
    q = "select game_id, game_date, player_name, team_abbreviation team, " + ", ".join(COUNTS) + " from nba_stats where substr(game_id,1,3) in ('002','004')"
    if before is not None:
        q += " and game_date < :d"
    df = pd.read_sql(text(q).bindparams(d=before.isoformat()) if before is not None else text(q), session.get_bind())
    df["player"] = df.player_name.map(normalize_name)
    df["game_date"] = pd.to_datetime(df.game_date).dt.date
    for c in COUNTS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df[df.minutes > 0]


def _ratio(num: float, den: float, prior: float, pseudo: float) -> float:
    return (num + pseudo * prior) / (den + pseudo) if den + pseudo > 0 else prior


@dataclass
class ShootingProfiles:
    rows: pd.DataFrame        # player, game_date, game_id, the decayed COUNTS
    league: Dict[str, float]  # KEYS

    @classmethod
    def build(cls, box: pd.DataFrame, halflife: float = HALFLIFE) -> "ShootingProfiles":
        a = 0.5 ** (1.0 / halflife)
        out = []
        for name, g in box.sort_values(["game_date", "game_id"]).groupby("player", sort=False):
            acc = {c: 0.0 for c in COUNTS}
            for r in g.itertuples(index=False):
                for c in COUNTS:
                    acc[c] = a * acc[c] + float(getattr(r, c))
                out.append(dict(player=name, game_date=r.game_date, game_id=r.game_id, **acc))
        rows = pd.DataFrame(out, columns=["player", "game_date", "game_id"] + COUNTS)
        tot = {c: float(box[c].sum()) for c in COUNTS}
        league = cls._keys(tot)
        return cls(rows=rows, league=league)

    @staticmethod
    def _keys(t: Dict[str, float]) -> Dict[str, float]:
        m = max(t["minutes"], 1e-9)
        fg2a = t["fga"] - t["fg3a"]; fg2m = t["fgm"] - t["fg3m"]; reb = t["rebounds_off"] + t["rebounds_def"]
        return dict(fga_pm=t["fga"] / m, fg3a_pm=t["fg3a"] / m, fta_pm=t["fta"] / m,
                    fg2_pct=fg2m / fg2a if fg2a > 0 else 0.5, fg3_pct=t["fg3m"] / t["fg3a"] if t["fg3a"] > 0 else 0.35,
                    ft_pct=t["ftm"] / t["fta"] if t["fta"] > 0 else 0.77, oreb_frac=t["rebounds_off"] / reb if reb > 0 else 0.25)

    def before(self, d: date) -> pd.DataFrame:
        sub = self.rows[self.rows.game_date < d]
        return sub.groupby("player").tail(1).set_index("player")

    def latest(self) -> pd.DataFrame:
        return self.rows.groupby("player").tail(1).set_index("player")

    def profile(self, state: pd.DataFrame, player: str) -> Dict[str, float]:
        L = self.league
        if player not in state.index:
            return dict(L)
        r = state.loc[player]
        m = float(r.minutes)
        fg2a = float(r.fga - r.fg3a); fg2m = float(r.fgm - r.fg3m); reb = float(r.rebounds_off + r.rebounds_def)
        return dict(fga_pm=_ratio(float(r.fga), m, L["fga_pm"], PRIOR_MINUTES), fg3a_pm=_ratio(float(r.fg3a), m, L["fg3a_pm"], PRIOR_MINUTES),
                    fta_pm=_ratio(float(r.fta), m, L["fta_pm"], PRIOR_MINUTES),
                    fg2_pct=_ratio(fg2m, fg2a, L["fg2_pct"], PRIOR_ATTEMPTS), fg3_pct=_ratio(float(r.fg3m), float(r.fg3a), L["fg3_pct"], PRIOR_ATTEMPTS),
                    ft_pct=_ratio(float(r.ftm), float(r.fta), L["ft_pct"], PRIOR_ATTEMPTS), oreb_frac=_ratio(float(r.rebounds_off), reb, L["oreb_frac"], 30.0))


def points_per_minute(p: Dict[str, float]) -> float:
    """What the profile's attempts and percentages score per minute."""
    return (p["fga_pm"] - p["fg3a_pm"]) * p["fg2_pct"] * 2.0 + p["fg3a_pm"] * p["fg3_pct"] * 3.0 + p["fta_pm"] * p["ft_pct"]


def fit_to_rates(p: Dict[str, float], points_pm: Optional[float], fg3m_pm: Optional[float], lo: float = 0.5, hi: float = 2.0) -> Dict[str, float]:
    """
    The profile's attempts rescaled so that it scores the pool's points per minute and makes its threes per minute (the books'
    numbers for a priced player): threes first (fg3a from fg3m / fg3_pct), then the two-point attempts and free throws
    together by one factor for the remaining points. Factors are kept within [lo, hi] of the profile.
    """
    q = dict(p)
    if fg3m_pm is not None and q["fg3_pct"] > 0:
        q["fg3a_pm"] = float(np.clip(fg3m_pm / q["fg3_pct"], lo * p["fg3a_pm"], hi * max(p["fg3a_pm"], 1e-6)))
    if points_pm is not None:
        from3 = q["fg3a_pm"] * q["fg3_pct"] * 3.0
        rest = (p["fga_pm"] - p["fg3a_pm"]) * q["fg2_pct"] * 2.0 + p["fta_pm"] * q["ft_pct"]
        f = float(np.clip((points_pm - from3) / rest, lo, hi)) if rest > 1e-9 else 1.0
        q["fga_pm"] = q["fg3a_pm"] + (p["fga_pm"] - p["fg3a_pm"]) * f
        q["fta_pm"] = p["fta_pm"] * f
    return q
