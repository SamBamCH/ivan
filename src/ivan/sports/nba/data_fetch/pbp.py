# src/ivan/sports/nba/data_fetch/pbp.py
"""
The play-by-play fetched and processed into the tables, one game at a time (the parsing is data_clean.pbp).

    fetch_game                     nba_api PlayByPlayV3 for one game, kept raw under nba_api/pbp/<game id>
    process_game                   fetch (or the raw copy), parse stints, fouls and substitutions, store them
    process_fouls / process_subs   the backfills from the raw copy, for games whose stints predate the foul or sub tables
    games_without_stints / fouls / subs   what the nightly still has to process
"""
from __future__ import annotations

from datetime import date
from typing import Callable, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.core import raw
from ivan.sports.nba.data_clean.pbp import fouls_from_pbp, load_raw, stints_from_pbp, store_fouls, store_stints, store_subs

Log = Callable[[str], None]


def fetch_game(game_id: str, timeout: int = 60) -> pd.DataFrame:
    from nba_api.stats.endpoints import playbyplayv3
    df = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=timeout).get_data_frames()[0]
    raw.store("nba", "nba_api/pbp", game_id, df.to_dict("records"))
    return df

def games_without_subs(session: Session) -> List[str]:
    have = {g for (g,) in session.execute(text("select distinct game_id from nba_sub")).all()}
    return sorted(g for (g,) in session.execute(text("select distinct game_id from nba_stint")).all() if g not in have)

def process_subs(session: Session, game_id: str, log: Log = print) -> int:
    """Substitution pairs only, from the stored raw play-by-play (the backfill)."""
    pbp = load_raw(game_id)
    if pbp is None:
        pbp = fetch_game(game_id)
    box = box_for_game(session, game_id)
    subs: list = []
    stints_from_pbp(pbp, box, game_id, subs_out=subs)
    gd = pd.to_datetime(box.game_date.iloc[0]).date() if len(box) else None
    n = store_subs(session, game_id, gd, subs)
    log(f"{game_id} {gd}: {n} substitutions")
    return n

def games_without_fouls(session: Session) -> List[str]:
    """Games with stints whose fouls were never parsed (the raw play-by-play is on disk: no fetch needed)."""
    have = {g for (g,) in session.execute(text("select distinct game_id from nba_foul")).all()}
    return sorted(g for (g,) in session.execute(text("select distinct game_id from nba_stint")).all() if g not in have)

def box_for_game(session: Session, game_id: str) -> pd.DataFrame:
    return pd.read_sql(text("select player_name, team_abbreviation, minutes, position, game_date from nba_stats where game_id = :g").bindparams(g=str(game_id)), session.get_bind())

def process_game(session: Session, game_id: str, log: Log = print, refetch: bool = False) -> int:
    pbp = None if refetch else load_raw(game_id)
    if pbp is None:
        pbp = fetch_game(game_id)
    box = box_for_game(session, game_id)
    subs: list = []
    df = stints_from_pbp(pbp, box, game_id, subs_out=subs)
    gd = pd.to_datetime(box.game_date.iloc[0]).date() if len(box) else None
    n = store_stints(session, game_id, gd, df)
    nf = store_fouls(session, game_id, gd, fouls_from_pbp(pbp, box, game_id))
    ns = store_subs(session, game_id, gd, subs)
    log(f"{game_id} {gd}: {n} stints, {df.player_name.nunique() if len(df) else 0} players, {nf} fouls, {ns} substitutions")
    return n

def process_fouls(session: Session, game_id: str, log: Log = print) -> int:
    """Fouls only, from the stored raw play-by-play (the backfill for games whose stints predate the foul table)."""
    pbp = load_raw(game_id)
    if pbp is None:
        pbp = fetch_game(game_id)
    box = box_for_game(session, game_id)
    gd = pd.to_datetime(box.game_date.iloc[0]).date() if len(box) else None
    n = store_fouls(session, game_id, gd, fouls_from_pbp(pbp, box, game_id))
    log(f"{game_id} {gd}: {n} fouls")
    return n

def games_without_stints(session: Session, since: Optional[date] = None, prefixes=("002", "004")) -> List[str]:
    have = {g for (g,) in session.execute(text("select distinct game_id from nba_stint")).all()}
    q = "select distinct game_id from nba_stats where substr(game_id, 1, 3) in :p"
    rows = session.execute(text("select distinct game_id, min(game_date) d from nba_stats group by game_id")).all()
    return sorted(g for g, d in rows if g[:3] in prefixes and g not in have and (since is None or pd.to_datetime(d).date() >= since))
