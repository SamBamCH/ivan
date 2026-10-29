"""
Game lines reduced to one row per game: the consensus spread, total and
win probability, from the per-book rows in game_odds (ingest.store_game_lines).

spread_home   median across books of the home side's spread (negative = home favoured)
total         median across books of the total
p_home        de-vigged moneyline probability for the home team (median across books)
team_spread(team) is what the minutes model wants: the team's own spread
(negative when favoured), None when the game is not priced.

game_odds keeps every game ever fetched (past nights, games weeks ahead), so a team has many rows: each team's
line is the game nearest its tip on the slate (`tips`, team -> tip time, within TIP_WINDOW), or without tips the
next game from now. A slate whose lines were not fetched gets None, never another night's game (until
2026-10-29 the last game read won, so the desk and the minutes model could use a game days away).
"""
from __future__ import annotations

from datetime import datetime, timedelta
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


TIP_WINDOW = timedelta(hours=12)     # a game's commence time this close to the team's tip is that night's game
UPCOMING_GRACE = timedelta(hours=4)  # without tips: a game that started this long ago still counts as the next one


class GameLines:
    """Per-team lookup over the consensus: spread (own, negative = favoured), total, win probability, for one game per team."""

    def __init__(self, cons: pd.DataFrame, tips: Optional[Dict[str, datetime]] = None, now: Optional[datetime] = None):
        now = now or datetime.now()
        cand: Dict[str, list] = {}
        for r in cons.itertuples():
            sh = None if pd.isna(r.spread_home) else float(r.spread_home)
            tot = None if pd.isna(r.total) else float(r.total)
            ph = None if pd.isna(r.p_home) else float(r.p_home)
            when = pd.Timestamp(r.commence_time).to_pydatetime() if r.commence_time is not None and not pd.isna(r.commence_time) else None
            if when is not None and when.tzinfo is not None:
                when = when.astimezone().replace(tzinfo=None)
            if r.home_abbr:
                cand.setdefault(r.home_abbr, []).append(dict(spread=sh, total=tot, p_win=ph, opponent=r.away_abbr, home=True, commence=when))
            if r.away_abbr:
                cand.setdefault(r.away_abbr, []).append(dict(spread=None if sh is None else -sh, total=tot, p_win=None if ph is None else 1 - ph,
                                                             opponent=r.home_abbr, home=False, commence=when))
        self.by_team: Dict[str, dict] = {}
        for team, games in cand.items():
            games = [g for g in games if g["commence"] is not None]
            if tips is not None:
                tip = tips.get(team)
                near = [g for g in games if tip is not None and abs(g["commence"] - tip) <= TIP_WINDOW]
                pick = min(near, key=lambda g: abs(g["commence"] - tip)) if near else None
            else:
                ahead = [g for g in games if g["commence"] >= now - UPCOMING_GRACE]
                pick = min(ahead, key=lambda g: g["commence"]) if ahead else None
            if pick is not None:
                self.by_team[team] = pick

    @classmethod
    def load(cls, session: Session, tips: Optional[Dict[str, datetime]] = None) -> "GameLines":
        return cls(load_consensus(session), tips=tips)

    def favourite(self, team: str) -> Optional[tuple]:
        """(the favoured team, by how many points) for the team's game; None when unpriced, (None, 0) for a pick'em."""
        g = self.by_team.get(team)
        if not g or g.get("spread") is None:
            return None
        sp = float(g["spread"])
        if sp == 0:
            return (None, 0.0)
        return (team, -sp) if sp < 0 else (g["opponent"], sp)

    def team_spread(self, team: str) -> Optional[float]:
        return (self.by_team.get(team) or {}).get("spread")

    def team_total(self, team: str) -> Optional[float]:
        return (self.by_team.get(team) or {}).get("total")
