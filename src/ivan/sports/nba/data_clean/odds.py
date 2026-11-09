# src/ivan/sports/nba/data_clean/odds.py
"""
The odds API's rows into the tables: the player props (odds) and the game lines (game_odds).

    store_props        replace the odds rows of the games a fetch priced (other games' rows stay; old games are dropped)
    odds_stamp         (rows, max id) of the odds table: moves with every fetch, so the market rates know when to rebuild
    store_game_lines   replace the stored lines of every game in the frame
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.sports.nba.models import (NBAGameOdds, NBAOdds)

Log = Callable[[str], None]


GAME_LINE_COLS = ["game_id", "sport_key", "commence_time", "home_team", "away_team", "home_abbr", "away_abbr", "bookmaker", "market",
                  "last_update", "outcome_name", "side", "point", "price"]


def store_game_lines(session: Session, lines: pd.DataFrame, fetched_at: Optional[datetime] = None) -> int:
    """Replace the stored lines of every game in `lines`. Returns rows stored."""
    if lines is None or lines.empty:
        return 0
    games = lines["game_id"].unique().tolist()
    session.query(NBAGameOdds).filter(NBAGameOdds.game_id.in_(games)).delete(synchronize_session=False)
    rows = lines[GAME_LINE_COLS].to_dict("records")
    now = fetched_at or datetime.now()
    for r in rows:
        r["fetched_at"] = now
        r["price"] = None if pd.isna(r.get("price")) else int(r["price"])
        r["point"] = None if pd.isna(r.get("point")) else float(r["point"])
        r["commence_time"] = None if pd.isna(r.get("commence_time")) else pd.Timestamp(r["commence_time"]).to_pydatetime()
    session.bulk_insert_mappings(NBAGameOdds, rows)
    session.commit()
    return len(rows)


PROPS_KEEP_DAYS = 3        # odds rows of a game that tipped longer ago than this are dropped at the next props store (the captures keep them)


def store_props(session: Session, odds: pd.DataFrame) -> int:
    """
    Replace the odds table's rows for the games a fetch priced. The rows of every game in `odds` (game_id) go, then the fetched rows
    come in; a row without a game (stored before 2026-11-07, or a capture made then) is replaced by player, as the table used to be.
    Other games' rows stay: a fetch for one slate's teams must not strip another slate's games, and a back-to-back's two games
    sit side by side, each read by its own slate (market.fresh_rows). Games that tipped more than PROPS_KEEP_DAYS ago are dropped
    (legacy rows by last_update). Returns the rows stored.
    """
    if odds is None or len(odds) == 0:
        return 0
    odds = odds.copy()
    for c in ("game_id", "commence_time", "home_abbr", "away_abbr"):
        if c not in odds.columns:
            odds[c] = None
    odds = odds.astype(object).where(odds.notna(), None)              # NaN -> NULL
    games = [str(g) for g in odds.game_id.dropna().unique()]
    for i in range(0, len(games), 500):
        session.query(NBAOdds).filter(NBAOdds.game_id.in_(games[i:i + 500])).delete(synchronize_session=False)
    players = [str(x) for x in odds[odds.game_id.isna()].player.dropna().unique()]
    for i in range(0, len(players), 500):
        session.query(NBAOdds).filter(NBAOdds.player.in_(players[i:i + 500])).delete(synchronize_session=False)
    session.bulk_insert_mappings(NBAOdds, odds.to_dict("records"))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=PROPS_KEEP_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    session.query(NBAOdds).filter(NBAOdds.commence_time.isnot(None), NBAOdds.commence_time < cutoff).delete(synchronize_session=False)
    session.query(NBAOdds).filter(NBAOdds.commence_time.is_(None), NBAOdds.last_update.isnot(None),
                                  NBAOdds.last_update < cutoff).delete(synchronize_session=False)
    session.commit()
    return int(len(odds))


def odds_stamp(session: Session) -> tuple:
    """(rows, max id) of the odds table: a props fetch replaces the rows of the games it priced, so the stamp moves with every fetch
    (the books' own last_update does not: a fetch stores the books' timestamps, which predate it). (0, None) without the table."""
    try:
        n, m = session.execute(text("select count(*), max(id) from odds")).one()
        return (int(n or 0), None if m is None else int(m))
    except Exception:   # noqa: BLE001
        return (0, None)
