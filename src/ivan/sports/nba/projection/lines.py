"""
Game lines reduced to one row per game: the consensus spread, total and
win probability, from the per-book rows in game_odds (ingest.store_game_lines).

spread_home   median across books of the home side's spread (negative = home favoured)
total         median across books of the total
p_home        de-vigged moneyline probability for the home team (median across books)
team_spread(team) is what the minutes model wants: the team's own spread
(negative when favoured), None when the game is not priced.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.sports.nba.models import NBAGameOdds


def american_to_prob(price: float) -> float:
    price = float(price)
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def consensus(rows: pd.DataFrame) -> pd.DataFrame:
    """parse_game_odds rows -> one row per game: game_id, commence_time, home_abbr, away_abbr, spread_home, total, p_home, books."""
    out = []
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["game_id", "commence_time", "home_abbr", "away_abbr", "spread_home", "total", "p_home", "books"])
    for gid, g in rows.groupby("game_id"):
        sp = g[(g.market == "spreads") & (g.side == "home")]["point"].dropna()
        tot = g[(g.market == "totals") & (g.side == "over")]["point"].dropna()
        ph = []
        for book, b in g[g.market == "h2h"].groupby("bookmaker"):
            h = b[b.side == "home"]["price"]; a = b[b.side == "away"]["price"]
            if len(h) and len(a):
                p_h, p_a = american_to_prob(h.iloc[0]), american_to_prob(a.iloc[0])
                ph.append(p_h / (p_h + p_a))
        out.append(dict(game_id=gid, commence_time=g.commence_time.iloc[0], home_abbr=g.home_abbr.iloc[0], away_abbr=g.away_abbr.iloc[0],
                        spread_home=float(sp.median()) if len(sp) else np.nan, total=float(tot.median()) if len(tot) else np.nan,
                        p_home=float(np.median(ph)) if ph else np.nan, books=int(g.bookmaker.nunique())))
    return pd.DataFrame(out)


def load_consensus(session: Session) -> pd.DataFrame:
    df = pd.read_sql(session.query(NBAGameOdds).statement, session.get_bind())
    return consensus(df)


class GameLines:
    """Per-team lookup over the consensus: spread (own, negative = favoured), total, win probability."""

    def __init__(self, cons: pd.DataFrame):
        self.by_team: Dict[str, dict] = {}
        for r in cons.itertuples():
            sh = None if pd.isna(r.spread_home) else float(r.spread_home)
            tot = None if pd.isna(r.total) else float(r.total)
            ph = None if pd.isna(r.p_home) else float(r.p_home)
            if r.home_abbr:
                self.by_team[r.home_abbr] = dict(spread=sh, total=tot, p_win=ph, opponent=r.away_abbr, home=True, commence=r.commence_time)
            if r.away_abbr:
                self.by_team[r.away_abbr] = dict(spread=None if sh is None else -sh, total=tot, p_win=None if ph is None else 1 - ph,
                                                 opponent=r.home_abbr, home=False, commence=r.commence_time)

    @classmethod
    def load(cls, session: Session) -> "GameLines":
        return cls(load_consensus(session))

    def team_spread(self, team: str) -> Optional[float]:
        return (self.by_team.get(team) or {}).get("spread")

    def team_total(self, team: str) -> Optional[float]:
        return (self.by_team.get(team) or {}).get("total")
