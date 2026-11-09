# src/ivan/sports/nba/data_fetch/slate.py
"""
The slate's inputs, fetched: DraftKings' lobby (classic_slates) and draftables (the pool), Rotowire's lineups page, the odds
API's player props and game lines. Every payload is kept raw (ivan.core.raw) before anything parses it.

    classic_slates     today's classic draft groups from the lobby: id, start, name, contests, games, lock, title
    fetch_slate        one SlateFetch: the pool rows (pool_from_draftables), the lock, the starters, the props, the game lines
    fetch_game_lines   moneyline, spread and total for every upcoming game (~3 credits)

IngestError is what a fetch or a store raises when nothing can be done with what came back (nothing changed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
from sqlalchemy.orm import Session

from ivan.core import raw
from ivan.core.config import require_env
from ivan.sports.nba.names import normalize_name
from ivan.sports.common.draftkings import (get_players_for_slate, _derive_opponent_for_player,
                                           _parse_gametime_to_local, list_slates)
from ivan.sports.common.odds import get_game_odds, get_odds, parse_game_odds, parse_odds
from ivan.sports.nba.data_fetch.boxscores import fetch_rotowire_html, parse_rotowire
from ivan.sports.nba.projection.market import NBA_MARKETS

Log = Callable[[str], None]


ODDS_SPORT = "basketball_nba"


class IngestError(RuntimeError):
    pass


def slate_title(suffix) -> Optional[str]:
    """DraftKings' slate title from the lobby's ContestStartTimeSuffix: " (Early Only)" -> "Early Only"; None when the slate has none
    (the main slate of the night carries no suffix)."""
    t = re.sub(r"^[\s(]+|[\s)]+$", "", str(suffix or "")).strip()
    return t or None


def classic_slates(day: Optional[datetime] = None) -> list[dict]:
    """Today's (or `day`'s) classic NBA draft groups from the DK lobby: id, start, name, contests, games, lock, title (DK's slate
    title when it has one, e.g. "Early Only")."""
    out = []
    for s in list_slates("NBA"):
        if str(s.get("game_type", "")).lower() != "classic":
            continue
        # DK's start is a display string ("Tue 2:00PM"); the lock comes from the draftables' game times
        lock = None
        if s.get("start_utc"):
            try:                                              # "2026-10-25T23:00:00.0000000Z" -> local naive datetime
                from datetime import timezone
                lock = datetime.fromisoformat(str(s["start_utc"])[:19]).replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
            except ValueError:
                lock = None
        out.append({"id": int(s["id"]), "start": s.get("start"), "name": s.get("name"), "contests": s.get("contests"),
                    "games": s.get("games"), "lock": lock, "title": slate_title(s.get("suffix"))})
    return out


@dataclass
class SlateFetch:
    draft_group_id: int
    players: list[dict] = field(default_factory=list)      # rows for nba_player
    lock: Optional[datetime] = None
    starters: Optional[pd.DataFrame] = None
    odds: Optional[pd.DataFrame] = None
    game_lines: Optional[pd.DataFrame] = None               # parse_game_odds rows for the slate's games
    teams: set = field(default_factory=set)
    fetched_at: datetime = field(default_factory=datetime.now)


def _fppg(d: dict) -> Optional[float]:
    for a in d.get("draftStatAttributes") or []:
        try:
            if int(a.get("id", -1)) == 90:
                return float(a.get("value"))
        except (TypeError, ValueError):
            continue
    for a in d.get("draftStatAttributes") or []:
        try:
            return float(a.get("value"))
        except (TypeError, ValueError):
            continue
    return None


def pool_from_draftables(draftables: list[dict], draft_group_id: int) -> tuple[list[dict], set]:
    """One row per player (first roster slot), DK status and FPPG kept."""
    rows, seen, teams = [], set(), set()
    for d in draftables:
        name = normalize_name(d.get("displayName") or "")
        if not name or name in seen:
            continue
        if d.get("draftableId") is None or d.get("salary") is None or d.get("position") is None:
            continue
        seen.add(name)
        team = d.get("teamAbbreviation"); comp = d.get("competition") or {}
        opp = _derive_opponent_for_player(team, comp)
        teams.update(t for t in (team, opp) if t)
        gt = _parse_gametime_to_local(comp.get("startTime"))
        if gt is not None and getattr(gt, "tzinfo", None) is not None:
            gt = gt.replace(tzinfo=None)          # naive local time, like every other datetime in the DB
        rows.append(dict(id=int(d["draftableId"]), name=name, team=team, opponent=opp, position=d["position"],
                         roster_slot_id=d.get("rosterSlotId"), gametime=gt,
                         salary=int(d["salary"]), status=d.get("status"), dk_avg_fpts=_fppg(d),
                         draft_group_id=int(draft_group_id)))
    return rows, teams


def fetch_slate(session: Session, draft_group_id: int, *, update_odds: bool, starters: bool = True,
                api_key: Optional[str] = None, log: Log = print, lines: Optional[bool] = None) -> SlateFetch:
    """DK draftables, Rotowire's lineups, the props (update_odds, ~12 credits a game) and the game lines (`lines`, ~3
    credits; None = with the props). The lines are the spread and total the desk and the minutes model read: until
    2026-10-29 only the autopilot tick fetched them, so a day worked by hand had no spreads."""
    out = SlateFetch(draft_group_id=int(draft_group_id))
    log(f"fetching DK draftables for draft group {draft_group_id}")
    draftables = get_players_for_slate(int(draft_group_id), raw_sport="nba")
    out.players, out.teams = pool_from_draftables(draftables, draft_group_id)
    if not out.players:
        raise IngestError("DraftKings returned no players for that draft group; nothing changed.")
    times = [p["gametime"] for p in out.players if p.get("gametime")]
    out.lock = min(times) if times else None
    if starters:
        try:
            html = fetch_rotowire_html()
            raw.store("nba", "rotowire/lineups", datetime.now().strftime("%Y-%m-%d"), html, ext="html")
            out.starters = parse_rotowire(html)
            log(f"rotowire: {len(out.starters)} starters listed, "
                f"{int(out.starters.confirmed.sum()) if len(out.starters) else 0} confirmed")
        except Exception as e:   # noqa: BLE001 - starters are optional
            log(f"WARNING: rotowire fetch failed: {type(e).__name__}: {e}")
    if update_odds:
        key = api_key or require_env("ODDS_API_KEY")
        odds = get_odds(ODDS_SPORT, key, sorted(out.teams), markets=NBA_MARKETS, raw_sport="nba")
        df = parse_odds(odds, games=True)                               # each row carries its game: a slate reads only its own games' rows
        if len(df) and "player" in df.columns:
            df["player"] = df["player"].map(normalize_name)        # the NBA key (names.py): the shared parser keeps a suffix the pool drops
        if df is None or len(df) == 0:
            raise IngestError("the-odds-api returned no NBA props; check ODDS_API_KEY and the plan. Nothing changed.")
        out.odds = df
        log(f"props: {len(df)} rows, {df.player.nunique()} players, {df.sportsbook.nunique()} books")
    if update_odds if lines is None else lines:
        try:
            out.game_lines = fetch_game_lines(api_key=api_key, log=log)
        except Exception as e:   # noqa: BLE001 - the lines are a convenience on top of the pool and the props
            log(f"WARNING: game lines fetch failed: {type(e).__name__}: {e}")
    return out


def fetch_game_lines(api_key: Optional[str] = None, log: Log = print) -> Optional[pd.DataFrame]:
    """Moneyline, spread and total for every upcoming NBA game (raw odds/lines; ~3 credits). None when the API returns nothing."""
    key = api_key or require_env("ODDS_API_KEY")
    payload, _ = get_game_odds(ODDS_SPORT, key, markets="h2h,spreads,totals", raw_sport="nba")
    df = parse_game_odds(payload) if payload else None
    if df is None or df.empty:
        log("WARNING: the-odds-api returned no NBA game lines")
        return None
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True).dt.tz_convert("America/Chicago").dt.tz_localize(None)
    log(f"game lines: {df.game_id.nunique()} games, {len(df)} rows, {df.bookmaker.nunique()} books")
    return df
