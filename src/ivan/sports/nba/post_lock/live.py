# src/ivan/sports/nba/post_lock/live.py
"""
Live game state for the late swap: NBA's public live-data feed (no key).

scoreboard()        today's games: game id, teams, status, period, clock, score
boxscore(game_id)   every player's minutes and counting stats so far
live_state(...)     one row per player of the slate's started games: minutes so far,
                    fantasy points so far (DK scoring, bonuses included), the team's
                    live margin and the elapsed share of regulation; plus a per-game
                    summary. Raw payloads are kept under raw nba_live/{scoreboard,boxscore}.

The in-game update (ingame.py) turns this into each player's remaining-game draw.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests

from ivan.core import raw
from ivan.core.utils import normalize_name
from ivan.sports.nba.projection.rates import DK_WEIGHTS

Log = Callable[[str], None]
SPORT = "nba"
SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/", "Accept": "application/json"}
REGULATION_SEC = 2880.0
_CLOCK = re.compile(r"PT(?:(\d+)M)?(?:([\d.]+)S)?")


def _get(url: str, kind: str, key: str, timeout: int = 15) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    try:
        raw.store(SPORT, kind, key, d)
    except Exception:   # noqa: BLE001 - keeping the raw copy must never break a swap
        pass
    return d


def clock_seconds(clock: str) -> float:
    m = _CLOCK.match(str(clock or ""))
    if not m:
        return 0.0
    return float(m.group(1) or 0) * 60.0 + float(m.group(2) or 0.0)


def elapsed_seconds(period: int, clock: str) -> float:
    """Elapsed game seconds (regulation 4 x 720, overtimes 300) at period / clock remaining."""
    period = int(period or 0)
    if period <= 0:
        return 0.0
    if period <= 4:
        return (period - 1) * 720.0 + (720.0 - clock_seconds(clock))
    return REGULATION_SEC + (period - 5) * 300.0 + (300.0 - clock_seconds(clock))


def minutes_of(s: str) -> float:
    """'PT24M30.00S' -> 24.5."""
    return clock_seconds(s) / 60.0


def dk_points(st: dict) -> float:
    """DraftKings classic points from a live statistics block, double- and triple-double bonuses included."""
    pts, reb, ast = float(st.get("points", 0)), float(st.get("reboundsTotal", 0)), float(st.get("assists", 0))
    stl, blk, tov, fg3 = float(st.get("steals", 0)), float(st.get("blocks", 0)), float(st.get("turnovers", 0)), float(st.get("threePointersMade", 0))
    lin = (DK_WEIGHTS["points"] * pts + DK_WEIGHTS["fg3m"] * fg3 + DK_WEIGHTS["rebounds"] * reb + DK_WEIGHTS["assists"] * ast
           + DK_WEIGHTS["steals"] * stl + DK_WEIGHTS["blocks"] * blk + DK_WEIGHTS["turnovers"] * tov)
    tens = sum(1 for x in (pts, reb, ast, stl, blk) if x >= 10)
    return lin + (3.0 if tens >= 3 else 0.0) + (1.5 if tens >= 2 else 0.0)


def scoreboard(log: Log = print) -> List[dict]:
    d = _get(SCOREBOARD_URL, "nba_live/scoreboard", "today")
    out = []
    for g in (d.get("scoreboard") or {}).get("games", []):
        h, a = g.get("homeTeam") or {}, g.get("awayTeam") or {}
        out.append(dict(game_id=str(g.get("gameId")), status=int(g.get("gameStatus") or 0), period=int(g.get("period") or 0),
                        clock=g.get("gameClock") or "", home=h.get("teamTricode"), away=a.get("teamTricode"),
                        home_score=int(h.get("score") or 0), away_score=int(a.get("score") or 0)))
    return out


def boxscore(game_id: str) -> dict:
    return _get(BOXSCORE_URL.format(game_id=game_id), "nba_live/boxscore", game_id).get("game") or {}


def parse_boxscore(game: dict) -> pd.DataFrame:
    """One row per player: name (normalized), team, minutes, fpts so far, fouls, on court, starter, plus the game's period/clock/score."""
    period, clock = int(game.get("period") or 0), game.get("gameClock") or ""
    el = elapsed_seconds(period, clock)
    status = int(game.get("gameStatus") or 0)
    rows = []
    teams = [(game.get("homeTeam") or {}, game.get("awayTeam") or {})]
    for side, other in ((teams[0][0], teams[0][1]), (teams[0][1], teams[0][0])):
        margin = int(side.get("score") or 0) - int(other.get("score") or 0)
        for p in side.get("players") or []:
            st = p.get("statistics") or {}
            rows.append(dict(player_name=normalize_name(p.get("name") or f"{p.get('firstName', '')} {p.get('familyName', '')}"),
                             team=side.get("teamTricode"), minutes=minutes_of(st.get("minutes") or st.get("minutesCalculated") or ""),
                             fpts=dk_points(st), fouls=int(st.get("foulsPersonal") or 0), on_court=str(p.get("oncourt")) == "1",
                             starter=str(p.get("starter")) == "1", played=str(p.get("played")) == "1",
                             margin=margin, elapsed=el, period=period, final=status == 3,
                             points=float(st.get("points", 0)), rebounds=float(st.get("reboundsTotal", 0)), assists=float(st.get("assists", 0)),
                             steals=float(st.get("steals", 0)), blocks=float(st.get("blocks", 0))))
    return pd.DataFrame(rows)


def live_state(teams: Optional[set] = None, log: Log = print) -> pd.DataFrame:
    """The parsed box score of every started game today (optionally only games involving `teams`)."""
    frames = []
    for g in scoreboard(log=log):
        if g["status"] < 2:
            continue                                               # not started
        if teams and not ({g["home"], g["away"]} & set(teams)):
            continue
        try:
            frames.append(parse_boxscore(boxscore(g["game_id"])).assign(game_id=g["game_id"]))
        except Exception as e:   # noqa: BLE001
            log(f"live box score {g['game_id']} ({g['away']} @ {g['home']}): {type(e).__name__}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
