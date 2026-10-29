"""
ivan.sports.nba.considerations

What is unusual about a team's game tonight, before the minutes are mapped: the things a rotation built from history
does not know. One list per team, loudest first (warn, then info), each a sentence with the evidence in it:

  rest        back to back (played yesterday), 3 games in 4 nights, or 3+ days off; the opponent's back to back
  starters    a regular starter (started most games, 20+ usual minutes) OUT or questionable tonight
  lineup      tonight's five (Rotowire, with hand calls) against the five that started the team's last game:
              who is new in it, who dropped out of it; a rookie in the five
  returning   a 15+ minute player available tonight who sat out the team's last game(s): minutes restriction?
  new team    a player whose last game on file was for another team (a trade or signing)
  spread      a big line (10+): the starters' minutes at risk in a blowout; or no line yet
  opponent    the opponent's regular starters out

Everything comes from what is on file: the box scores (the nightly brings last night's game, so before it runs a
back to back reads as one day of rest more), the pool's statuses, the starters, the game line. The desk shows it on
the team card above tonight's grid (Desk.considerations); scripts/nba/considerations.py prints it per team.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from ivan.core.utils import normalize_name

OUT_STATUSES = {"OUT", "O", "IR", "SUSP"}
QUESTION_STATUSES = {"Q", "GTD", "D", "P"}
REGULAR_START = 0.6      # start_ewm at or above: a regular starter
REGULAR_MIN = 20.0       # and this many usual minutes
RETURN_MIN = 15.0        # a player this big sitting out the last game is news when he is back
BIG_SPREAD = 10.0
REST_DAYS = 3            # this many days off or more: rested


def usual_minutes(state: pd.DataFrame, name: str) -> Optional[float]:
    if name not in state.index:
        return None
    s = state.loc[name]
    return float(s["min_ewm"]) * float(s.get("play_ewm", 1.0))


def _team_dates(box: pd.DataFrame, team: str, before: date) -> List[date]:
    g = box[(box.team == team) & (box.game_date < before)]
    return sorted(set(g.game_date))


def rest_notes(box: pd.DataFrame, team: str, tip_day: date, who: str = "") -> List[tuple]:
    days = _team_dates(box, team, tip_day)
    if not days:
        return []
    last = days[-1]
    off = (tip_day - last).days - 1
    lab = f"{who}{team}" if who else team
    out = []
    if off == 0:
        out.append(("warn", f"{lab} on a back to back: played {last:%a %m-%d}"))
    elif sum(1 for d in days if (tip_day - d).days <= 3) >= 2:
        out.append(("warn", f"{lab}: 3 games in 4 nights"))
    elif off >= REST_DAYS:
        out.append(("info", f"{lab} rested: {off} days off since {last:%a %m-%d}"))
    return out


def last_game(box: pd.DataFrame, team: str, before: date) -> pd.DataFrame:
    days = _team_dates(box, team, before)
    if not days:
        return box.iloc[0:0]
    g = box[(box.team == team) & (box.game_date == days[-1])]
    return g[g.game_id == g.game_id.iloc[-1]] if len(g) else g


def missed_streak(box: pd.DataFrame, team: str, name: str, before: date) -> int:
    """How many of the team's latest games in a row he did not play (0 when he played the last one)."""
    tg = box[(box.team == team) & (box.game_date < before)].sort_values(["game_date", "game_id"])
    n = 0
    for gid in reversed(list(dict.fromkeys(tg.game_id))):
        row = tg[(tg.game_id == gid) & (tg.player_name == name)]
        if len(row) and float(row.minutes.iloc[0]) > 0:
            break
        n += 1
    return n


def team_considerations(team: str, players: Iterable, box: pd.DataFrame, state: pd.DataFrame, starters: Optional[set],
                        confirmed: bool = False, spread: Optional[float] = None, opponent: Optional[str] = None,
                        opp_players: Iterable = (), is_rookie: Optional[Callable[[str], bool]] = None,
                        tip: Optional[datetime] = None) -> List[dict]:
    """The list for one team (see the module doc): [{level: warn|info, kind, text}], warn first."""
    players = list(players)
    tip = tip or min((p.gametime for p in players if p.gametime), default=datetime.now())
    day = tip.date()
    out: List[tuple] = []
    out += [(lv, "rest", t) for lv, t in rest_notes(box, team, day)]
    if opponent:
        out += [(lv, "rest", t) for lv, t in rest_notes(box, opponent, day, who="the opponent ") if lv == "warn"]

    status = {normalize_name(p.name): (p.status or "").upper() for p in players}
    raw = {normalize_name(p.name): p.name for p in players}

    def regular(n):
        if n not in state.index:
            return False
        um = usual_minutes(state, n) or 0.0
        return float(state.loc[n].get("start_ewm", 0.0)) >= REGULAR_START and um >= REGULAR_MIN

    for n, st in status.items():
        if regular(n) and st in OUT_STATUSES:
            out.append(("warn", "starters", f"regular starter {raw[n]} OUT ({usual_minutes(state, n):.0f} usual min): his minutes and his spot move"))
        elif regular(n) and st in QUESTION_STATUSES:
            out.append(("warn", "starters", f"regular starter {raw[n]} {st} ({usual_minutes(state, n):.0f} usual min)"))

    lg = last_game(box, team, day)
    if starters and len(lg):
        prev = set(lg[lg.starter > 0].player_name)
        new_in = sorted(starters - prev); dropped = sorted(prev - starters)
        how = "confirmed" if confirmed else "expected"
        if new_in or dropped:
            bits = []
            if new_in:
                bits.append("in: " + ", ".join(raw.get(n, n) for n in new_in))
            if dropped:
                bits.append("out of it: " + ", ".join(raw.get(n, n) + (f" ({status[n]})" if status.get(n) else "") for n in dropped))
            out.append(("warn", "lineup", f"new starting five vs the last game ({lg.game_date.iloc[0]:%m-%d}, {how}): " + "; ".join(bits)))
        if is_rookie:
            rk = [raw.get(n, n) for n in sorted(starters) if is_rookie(n)]
            if rk:
                out.append(("info", "lineup", "rookie starting: " + ", ".join(rk)))
    elif not starters:
        out.append(("info", "lineup", "no starting five yet (Rotowire has none, no hand call)"))

    for n, st in status.items():
        if st in OUT_STATUSES:
            continue
        um = usual_minutes(state, n)
        if um is None or um < RETURN_MIN:
            continue
        k = missed_streak(box, team, n, day)
        if k >= 1:
            out.append(("warn", "returning", f"{raw[n]} back after missing the last {k} game{'s' if k > 1 else ''} ({um:.0f} usual min): a minutes limit?"
                        if k > 1 else f"{raw[n]} sat out the last game ({um:.0f} usual min): back tonight, any limit?"))

    for n in status:
        g = box[(box.player_name == n) & (box.game_date < day) & (box.minutes > 0)]
        if len(g) and str(g.sort_values("game_date").team.iloc[-1]) != team:
            out.append(("info", "new team", f"{raw[n]} new to {team}: his last game was for {g.sort_values('game_date').team.iloc[-1]}"))

    if spread is None:
        out.append(("info", "spread", "no line for tonight's game yet: the blowout model is off until the props refresh"))
    elif abs(spread) >= BIG_SPREAD:
        fav = team if spread < 0 else (opponent or "the opponent")
        out.append(("warn", "spread", f"{fav} favoured by {abs(spread):g}: blowout risk, the starters' minutes can be cut"))

    if opponent:
        for p in opp_players:
            n = normalize_name(p.name); st = (p.status or "").upper()
            if st in OUT_STATUSES and regular(n):
                out.append(("info", "opponent", f"{opponent} without regular starter {p.name} ({usual_minutes(state, n):.0f} usual min)"))

    order = {"warn": 0, "info": 1}
    return [dict(level=lv, kind=k, text=t) for lv, k, t in sorted(out, key=lambda x: order[x[0]])]
