# src/ivan/sports/nba/gamesim/possessions.py
"""
Events, possessions and five-man states from the stored play-by-play.

The raw PlayByPlayV3 feed (pbp.load_raw, one file per game) carries every action with its clock, team, player,
shot value and result, the free-throw trip, the rebound, the turnover, the running score. This module turns one
game into one row per event that matters to the simulation, with the possession bookkeeping and who was on the floor:

game_events(pbp, box, stints)   one row per event: period, elapsed, segment, team, player, kind (fg | ft | reb | tov |
                                stl | blk | foul | sub), made, value, assist, trip (i of n), reb_kind (off | def),
                                margin (the event team's lead before it), poss_end (this event ends the offence's
                                possession) and poss_team, on_off / on_def (the five on the floor for the offence and
                                the defence, '|'-joined normalized names) and on_ok (the acting player is among his
                                team's five and both fives are whole).
team_segments(events)           per game, team and six-minute segment (overtime segments 8, 9, ...): possessions,
                                points, FGA, 3PA, FTA, TOV, OREB, DREB.
player_games(events)            per game and player: the box-score counts the feed implies plus the possessions he was
                                on the floor for (his team's, off_poss, and the opponent's, def_poss) and what happened
                                around him while on (ON_FLOOR: teammates' attempts, makes and misses, the opponent's
                                attempts and misses): the usage fit's denominators.
reconcile(events, box)          the feed's counts against nba_stats, per player and per team: how far the parse is
                                from the box score (the check of this layer).
build(session, ...)             every game with stints and a stored feed -> data/nba/derived/gamesim/events.parquet,
                                incrementally (games already in the file are kept unless rebuild).

Possession ends (the offence's team is charged one possession): a made field goal with no and-one free throw at the same
clock, the last free throw of an ordinary trip when made (a miss leaves it to the rebound; technical, flagrant and
clear-path trips keep the ball), a defensive rebound (the offence's possession ends), a turnover. A period's end is not
counted (less than one possession per period, both teams alike). The check is the classic estimate
FGA - OREB + TOV + 0.44 FTA, which the counted possessions should sit within a few per cent of, and the two teams'
counts should differ by at most a few.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.sports.nba.data_clean import pbp as P
from ivan.sports.nba.names import normalize_name
from ivan.sports.nba.paths import NBA_DATA_DIR

Log = Callable[[str], None]
DERIVED_DIR = NBA_DATA_DIR / "derived" / "gamesim"
EVENTS_PARQUET = DERIVED_DIR / "events.parquet"
SEG_SEC = 360.0
REG_SEC = 2880.0
OT_SEC = 300.0
_AST = re.compile(r"\(([^()]+?) (\d+) AST\)")
_STL = re.compile(r"^(.+?) STEAL \(")
_BLK = re.compile(r"^(.+?) BLOCK \(")
_TRIP = re.compile(r"(\d+) of (\d+)")
EVENT_COLS = ["game_id", "game_date", "period", "elapsed", "segment", "team", "opp", "player", "kind", "made", "value", "pts",
              "assist", "trip_i", "trip_n", "reb_kind", "margin", "home", "poss_end", "poss_team", "on_off", "on_def", "on_ok", "action"]


def segment_of(elapsed: float) -> int:
    """Six-minute segments of regulation 0..7; overtime periods are segments 8, 9, ... (one per period)."""
    if elapsed < REG_SEC:
        return min(int(elapsed // SEG_SEC), 7)
    return 8 + int((elapsed - REG_SEC) // OT_SEC)


def segment_seconds(segment: int) -> float:
    return SEG_SEC if segment < 8 else OT_SEC


class _Floor:
    """Who is on the floor for a team at any elapsed second, from its stints."""
    def __init__(self, stints: pd.DataFrame):
        self.names = stints.player_name.to_numpy()
        self.start = stints.start_sec.to_numpy(dtype=float)
        self.end = stints.end_sec.to_numpy(dtype=float)

    def at(self, t: float, after: bool = True) -> List[str]:
        """The players on at `t`: a stint that starts at `t` counts (after a substitution at that second), one that ends at `t` does not."""
        if after:
            m = (self.start <= t) & (self.end > t)
        else:
            m = (self.start < t) & (self.end >= t)
        return sorted(set(self.names[m].tolist()))


def _resolve(names: Dict[str, Dict[str, List[str]]], team: str, label: Optional[str], initial: Optional[str] = None) -> Optional[str]:
    if not label or team not in names:
        return None
    return P._resolve_label(names[team], str(label), initial=initial)


def game_events(pbp: pd.DataFrame, box: pd.DataFrame, stints: Optional[pd.DataFrame], game_id: str, game_date=None) -> pd.DataFrame:
    """
    One row per event of one game (EVENT_COLS). box: the game's nba_stats rows (player_name, team_abbreviation, position);
    stints: the game's nba_stint rows (team, player_name, start_sec, end_sec) or None (on_off / on_def empty, on_ok False).
    """
    if pbp is None or pbp.empty or box is None or box.empty:
        return pd.DataFrame(columns=EVENT_COLS)
    pbp = pbp.copy()
    pbp["elapsed"] = [P.elapsed_at(int(p), c) for p, c in zip(pbp.period, pbp.clock)]
    pbp = pbp.sort_values(["period", "elapsed", "actionNumber"]).reset_index(drop=True)
    teams = [str(t) for t in box.team_abbreviation.unique()]
    if len(teams) != 2:
        return pd.DataFrame(columns=EVENT_COLS)
    home = P.home_of(pbp, teams); away = next(t for t in teams if t != home)
    opp = {home: away, away: home}
    tri_of_id = {int(i): str(t) for i, t in pbp[["teamId", "teamTricode"]].drop_duplicates().itertuples(index=False) if t and int(i or 0)}
    names = P._last_name_map(box)
    sh = pd.to_numeric(pbp.scoreHome, errors="coerce").ffill().fillna(0).astype(int).to_numpy()
    sa = pd.to_numeric(pbp.scoreAway, errors="coerce").ffill().fillna(0).astype(int).to_numpy()
    floors = {}
    if stints is not None and not stints.empty:
        for t in teams:
            floors[t] = _Floor(stints[stints.team == t])
    kinds = pbp.actionType.astype(str).str.lower().to_numpy()
    descs = pbp.description.astype(str).fillna("").to_numpy()
    subtypes = pbp.subType.astype(str).fillna("").to_numpy() if "subType" in pbp else np.full(len(pbp), "")
    tricodes = pbp.teamTricode.astype(str).fillna("").to_numpy()
    person = pd.to_numeric(pbp.personId, errors="coerce").fillna(0).astype(int).to_numpy() if "personId" in pbp else np.zeros(len(pbp), int)
    labels = pbp.playerName.astype(str).fillna("").to_numpy() if "playerName" in pbp else np.full(len(pbp), "")
    initials = pbp.playerNameI.astype(str).fillna("").to_numpy() if "playerNameI" in pbp else np.full(len(pbp), "")
    values = pd.to_numeric(pbp.shotValue, errors="coerce").fillna(0).astype(int).to_numpy() if "shotValue" in pbp else np.zeros(len(pbp), int)
    is_fg = pd.to_numeric(pbp.isFieldGoal, errors="coerce").fillna(0).astype(int).to_numpy() if "isFieldGoal" in pbp else np.zeros(len(pbp), int)
    elapsed = pbp.elapsed.to_numpy(dtype=float); periods = pbp.period.to_numpy(dtype=int)
    actions = pbp.actionNumber.to_numpy() if "actionNumber" in pbp else np.arange(len(pbp))
    n = len(pbp)

    def team_of(i: int) -> Optional[str]:
        t = tricodes[i]
        if t in opp:
            return t
        return tri_of_id.get(int(person[i]))            # a team rebound / turnover names the team by its id

    rows = []
    last_miss_team: Optional[str] = None                 # the team whose missed shot or final free throw is waiting for a live rebound
    dead_ball = False                                    # a missed free throw with more to come: the next 'rebound' is a dead-ball team rebound (not a rebound, ends nothing)
    for i in range(n):
        k = kinds[i]; d = descs[i]; st = subtypes[i]
        team = team_of(i)
        if team is None or k in ("period", "timeout", "instant replay", "jump ball", "violation", "heave", "game"):
            continue
        t = float(elapsed[i]); per = int(periods[i])
        mh = int(sh[i - 1] - sa[i - 1]) if i > 0 else 0                    # the margin before this event, home side
        margin = mh if team == home else -mh
        base = dict(game_id=str(game_id), game_date=game_date, period=per, elapsed=t, segment=segment_of(t), team=team, opp=opp[team],
                    player=None, kind=None, made=False, value=0, pts=0, assist=None, trip_i=0, trip_n=0, reb_kind=None, margin=margin,
                    home=team == home, poss_end=False, poss_team=None, on_off="", on_def="", on_ok=False, action=int(actions[i]))
        player = _resolve(names, team, labels[i], initials[i]) if labels[i] else None
        if k == "made shot" or k == "missed shot":
            if not is_fg[i]:
                continue
            made = k == "made shot"
            r = dict(base, player=player, kind="fg", made=made, value=int(values[i]) or 2, pts=(int(values[i]) or 2) if made else 0)
            if made:
                m = _AST.search(d)
                r["assist"] = _resolve(names, team, m.group(1)) if m else None
                # an and-one: a 'Free Throw 1 of 1' by the same team at the same clock keeps the ball alive
                and_one = False
                j = i + 1
                while j < n and abs(elapsed[j] - t) < 1e-6:
                    if kinds[j] == "free throw" and team_of(j) == team and "1 of 1" in subtypes[j]:
                        and_one = True; break
                    j += 1
                if not and_one:
                    r["poss_end"] = True; r["poss_team"] = team
                last_miss_team = None
            else:
                last_miss_team = team
            dead_ball = False
            rows.append(r)
        elif k == "free throw":
            m = _TRIP.search(st) or _TRIP.search(d)
            ti, tn = (int(m.group(1)), int(m.group(2))) if m else (1, 1)
            made = "MISS" not in d.upper()
            low = (st + " " + d).lower()
            keeps_ball = "technical" in low or "flagrant" in low or "clear path" in low
            r = dict(base, player=player, kind="ft", made=made, value=1, pts=1 if made else 0, trip_i=ti, trip_n=tn)
            dead_ball = False
            if ti == tn and not keeps_ball:
                if made:
                    r["poss_end"] = True; r["poss_team"] = team; last_miss_team = None
                else:
                    last_miss_team = team
            elif not made:
                dead_ball = True
            rows.append(r)
        elif k == "rebound":
            if dead_ball:                                # the feed's team rebound after a missed 1 of 2: no rebound, no change of possession
                dead_ball = False
                continue
            if last_miss_team is None:                   # nothing waiting (a heave at the buzzer, a feed gap): not a rebound that moves the ball
                continue
            kind = "off" if last_miss_team == team else "def"
            r = dict(base, player=player, kind="reb", reb_kind=kind)
            if kind == "def":
                r["poss_end"] = True; r["poss_team"] = opp[team]
            last_miss_team = None
            rows.append(r)
        elif k == "turnover":
            rows.append(dict(base, player=player, kind="tov", poss_end=True, poss_team=team))
            last_miss_team = None
        elif k == "foul":
            rows.append(dict(base, player=player, kind="foul"))
        elif k == "substitution":
            rows.append(dict(base, player=player, kind="sub"))
        elif k == "":
            m = _STL.match(d)
            if m:
                rows.append(dict(base, player=_resolve(names, team, m.group(1), initials[i]), kind="stl")); continue
            m = _BLK.match(d)
            if m:
                rows.append(dict(base, player=_resolve(names, team, m.group(1), initials[i]), kind="blk")); continue
    ev = pd.DataFrame(rows, columns=EVENT_COLS)
    if ev.empty:
        return ev
    if floors:
        on_off, on_def, on_ok = [], [], []
        for r in ev.itertuples(index=False):
            fo = floors.get(r.team); fd = floors.get(r.opp)
            # a substitution is read before its second; everything else after the substitutions of that second
            a = fo.at(r.elapsed, after=r.kind != "sub") if fo else []
            b = fd.at(r.elapsed, after=True) if fd else []
            on_off.append("|".join(a)); on_def.append("|".join(b))
            on_ok.append(len(a) == 5 and len(b) == 5 and (r.player is None or r.player in a or r.kind in ("stl", "blk", "sub")))
        ev["on_off"] = on_off; ev["on_def"] = on_def; ev["on_ok"] = on_ok
    return ev


# ----------------------------------------------------------------------------- tables
def team_segments(events: pd.DataFrame) -> pd.DataFrame:
    """Per game, team, segment: poss, pts, fga, fg3a, fgm, fg3m, fta, ftm, tov, oreb, dreb, seconds. Every team gets every segment of its game."""
    cols = ["game_id", "game_date", "team", "opp", "home", "segment", "poss", "pts", "fga", "fg3a", "fgm", "fg3m", "fta", "ftm", "tov", "oreb", "dreb", "seconds"]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for gid, g in events.groupby("game_id", sort=False):
        segs = range(int(g.segment.max()) + 1)
        date = g.game_date.iloc[0]
        for team in g.team.unique():
            o = g[g.team == team]; opp = o.opp.iloc[0]; home = bool(o.home.iloc[0])
            pe = g[g.poss_team == team]
            for s in segs:
                e = o[o.segment == s]; fg = e[e.kind == "fg"]; ft = e[e.kind == "ft"]; rb = e[e.kind == "reb"]
                out.append(dict(game_id=gid, game_date=date, team=team, opp=opp, home=home, segment=s, poss=int((pe.segment == s).sum()),
                                pts=int(e.pts.sum()), fga=int(len(fg)), fg3a=int((fg.value == 3).sum()), fgm=int(fg.made.sum()), fg3m=int(((fg.value == 3) & fg.made).sum()),
                                fta=int(len(ft)), ftm=int(ft.made.sum()), tov=int((e.kind == "tov").sum()),
                                oreb=int((rb.reb_kind == "off").sum()), dreb=int((rb.reb_kind == "def").sum()), seconds=segment_seconds(s)))
    return pd.DataFrame(out, columns=cols)


def estimated_possessions(seg: pd.DataFrame) -> pd.Series:
    """The classic estimate per row: FGA - OREB + TOV + 0.44 FTA."""
    return seg.fga - seg.oreb + seg.tov + 0.44 * seg.fta


PLAYER_COUNTS = ["pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"]
# what happened around a player while he was on the floor (the usage fit's denominators): his team's possessions and the
# opponent's, his teammates' field-goal attempts, makes and misses (assist and offensive-rebound chances), the opponent's
# attempts and misses (block and defensive-rebound chances); a missed final free throw counts as a miss on both sides
ON_FLOOR = ["off_poss", "def_poss", "team_fga_on", "team_fgm_on", "team_miss_on", "opp_fga_on", "opp_miss_on"]


def player_games(events: pd.DataFrame) -> pd.DataFrame:
    """Per game, team, player: PLAYER_COUNTS from the feed plus the ON_FLOOR counts (from the events with whole fives)."""
    cols = ["game_id", "game_date", "team", "player"] + PLAYER_COUNTS + ON_FLOOR
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)
    acc: Dict[Tuple[str, str, str], dict] = {}

    def row(gid, date, team, player):
        key = (gid, team, player)
        if key not in acc:
            acc[key] = dict(game_id=gid, game_date=date, team=team, player=player, **{c: 0 for c in PLAYER_COUNTS}, **{c: 0 for c in ON_FLOOR})
        return acc[key]

    def bump(gid, date, team, five: str, col: str, skip: Optional[str] = None):
        for nm in five.split("|"):
            if nm != skip:
                row(gid, date, team, nm)[col] += 1

    for r in events.itertuples(index=False):
        whole = bool(r.on_off) and bool(r.on_def)
        if whole and r.kind == "fg":
            bump(r.game_id, r.game_date, r.team, r.on_off, "team_fga_on", skip=r.player)
            bump(r.game_id, r.game_date, r.team, r.on_off, "team_fgm_on" if r.made else "team_miss_on", skip=r.player)
            bump(r.game_id, r.game_date, r.opp, r.on_def, "opp_fga_on")
            if not r.made:
                bump(r.game_id, r.game_date, r.opp, r.on_def, "opp_miss_on")
        elif whole and r.kind == "ft" and r.trip_i == r.trip_n and not r.made:
            bump(r.game_id, r.game_date, r.team, r.on_off, "team_miss_on", skip=r.player)
            bump(r.game_id, r.game_date, r.opp, r.on_def, "opp_miss_on")
        if r.player:
            x = row(r.game_id, r.game_date, r.team, r.player)
            if r.kind == "fg":
                x["fga"] += 1; x["fgm"] += int(r.made); x["pts"] += int(r.pts)
                if r.value == 3:
                    x["fg3a"] += 1; x["fg3m"] += int(r.made)
                if r.assist:
                    row(r.game_id, r.game_date, r.team, r.assist)["ast"] += 1
            elif r.kind == "ft":
                x["fta"] += 1; x["ftm"] += int(r.made); x["pts"] += int(r.pts)
            elif r.kind == "reb":
                x["oreb" if r.reb_kind == "off" else "dreb"] += 1
            elif r.kind == "tov":
                x["tov"] += 1
            elif r.kind == "stl":
                x["stl"] += 1
            elif r.kind == "blk":
                x["blk"] += 1
            elif r.kind == "foul":
                x["pf"] += 1
        if r.poss_end and r.on_off and r.on_def and r.poss_team:
            off_team = r.poss_team; def_team = r.opp if r.team == off_team else r.team
            off_five, def_five = (r.on_off, r.on_def) if r.team == off_team else (r.on_def, r.on_off)
            for nm in off_five.split("|"):
                row(r.game_id, r.game_date, off_team, nm)["off_poss"] += 1
            for nm in def_five.split("|"):
                row(r.game_id, r.game_date, def_team, nm)["def_poss"] += 1
    return pd.DataFrame(list(acc.values()), columns=cols)


BOX_MAP = {"pts": "points", "ast": "assists", "stl": "steals", "blk": "blocks", "tov": "turnovers", "fga": "fga", "fgm": "fgm", "fg3m": "fg3m", "fta": "fta", "ftm": "ftm"}


def reconcile(events: pd.DataFrame, box: pd.DataFrame) -> dict:
    """
    How far the parse sits from the box score. box: nba_stats rows of the same games (player_name, team_abbreviation,
    game_id, points, rebounds, assists, steals, blocks, turnovers, fga, fgm, fg3m, fta, ftm).
    Returns {"players": frame per stat (n, exact share, mean abs gap, total gap), "teams": frame per game-team
    (counted vs estimated possessions, both teams' counts), "summary": dict}.
    """
    pg = player_games(events)
    b = box.copy(); b["player"] = b.player_name.map(normalize_name); b = b.rename(columns={"team_abbreviation": "team"})
    b["reb_box"] = b["rebounds"]
    m = pg.merge(b, on=["game_id", "team", "player"], how="outer", suffixes=("", "_box"))
    stats = []
    for ours, theirs in BOX_MAP.items():
        if theirs not in m:
            continue
        x = m[ours].fillna(0).astype(float); y = m[theirs].fillna(0).astype(float)
        stats.append(dict(stat=ours, n=int(len(m)), exact=float((x == y).mean()), mae=float((x - y).abs().mean()), total_gap=float((x - y).sum()), total=float(y.sum())))
    x = (m.oreb.fillna(0) + m.dreb.fillna(0)).astype(float); y = m.reb_box.fillna(0).astype(float)
    stats.append(dict(stat="reb", n=int(len(m)), exact=float((x == y).mean()), mae=float((x - y).abs().mean()), total_gap=float((x - y).sum()), total=float(y.sum())))
    players = pd.DataFrame(stats)
    seg = team_segments(events)
    tg = seg.groupby(["game_id", "team", "opp"], as_index=False)[["poss", "fga", "fta", "tov", "oreb", "pts"]].sum()
    tg["poss_est"] = estimated_possessions(tg)
    other = tg[["game_id", "team", "poss"]].rename(columns={"team": "opp", "poss": "poss_opp"})
    tg = tg.merge(other, on=["game_id", "opp"], how="left")
    summary = dict(games=int(tg.game_id.nunique()), poss_mean=float(tg.poss.mean()) if len(tg) else float("nan"),
                   poss_vs_estimate=float((tg.poss - tg.poss_est).mean()) if len(tg) else float("nan"),
                   poss_vs_estimate_mae=float((tg.poss - tg.poss_est).abs().mean()) if len(tg) else float("nan"),
                   teams_gap_mae=float((tg.poss - tg.poss_opp).abs().mean()) if len(tg) else float("nan"),
                   on_ok_share=float(events.on_ok.mean()) if len(events) else float("nan"),
                   unresolved_share=float(events[events.kind.isin(["fg", "ft", "tov"])].player.isna().mean()) if len(events) else float("nan"))
    return {"players": players, "teams": tg, "summary": summary}


# ----------------------------------------------------------------------------- the store
def load_game_inputs(session: Session, game_id: str) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """(raw play-by-play or None, the game's box rows, the game's stints)."""
    raw = P.load_raw(str(game_id))
    box = pd.read_sql(text("select game_id, game_date, player_name, team_abbreviation, position, minutes, points, rebounds, assists, steals, blocks, "
                           "turnovers, fgm, fga, fg3m, fg3a, ftm, fta from nba_stats where game_id = :g"), session.get_bind(), params={"g": str(game_id)})
    st = pd.read_sql(text("select game_id, team, player_name, start_sec, end_sec, minutes from nba_stint where game_id = :g"), session.get_bind(), params={"g": str(game_id)})
    return raw, box, st


def games_with_stints(session: Session, since=None) -> pd.DataFrame:
    """game_id, game_date of every game that has stints (and so a stored feed), oldest first."""
    q = "select game_id, min(game_date) as game_date from nba_stint group by game_id order by min(game_date), game_id"
    df = pd.read_sql(text(q), session.get_bind())
    df["game_date"] = pd.to_datetime(df.game_date).dt.date
    if since is not None:
        df = df[df.game_date >= pd.Timestamp(since).date()]
    return df


def load_events(path: Path = EVENTS_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path) if Path(path).exists() else pd.DataFrame(columns=EVENT_COLS)


def build(session: Session, since=None, rebuild: bool = False, max_games: Optional[int] = None, path: Path = EVENTS_PARQUET,
          log: Log = print) -> pd.DataFrame:
    """Parse every game with stints into the events store; returns the whole store. Games already stored are skipped unless rebuild."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    have = pd.DataFrame(columns=EVENT_COLS) if rebuild else load_events(path)
    done = set(have.game_id.unique()) if len(have) else set()
    todo = games_with_stints(session, since)
    todo = todo[~todo.game_id.isin(done)]
    if max_games:
        todo = todo.head(int(max_games))
    frames = [have] if len(have) else []
    n_ok = 0
    for i, g in enumerate(todo.itertuples(index=False), 1):
        raw, box, st = load_game_inputs(session, g.game_id)
        if raw is None or box.empty:
            continue
        ev = game_events(raw, box, st, g.game_id, game_date=pd.Timestamp(g.game_date).date())
        if not ev.empty:
            frames.append(ev); n_ok += 1
        if i % 50 == 0:
            log(f"  {i}/{len(todo)} games parsed")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EVENT_COLS)
    if n_ok or rebuild:
        store = out.copy()
        store["game_date"] = pd.to_datetime(store.game_date)
        store.to_parquet(path, index=False)
    log(f"events: {n_ok} games parsed, {out.game_id.nunique()} in the store ({path})")
    return out
