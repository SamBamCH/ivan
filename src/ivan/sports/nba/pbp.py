# src/ivan/sports/nba/pbp.py
"""
Stints from the NBA play-by-play: when each player was on the floor, for
how long at a stretch, and what the score was when he came off.

fetch_game        nba_api PlayByPlayV3 for one game, raw nba_api/pbp/<game id>
stints_from_pbp   the substitution log + the box-score starters -> one row per
                  stint: elapsed seconds in and out, minutes, the margin (the
                  player's team minus the opponent) at entry and exit, whether
                  the stint ran to the final buzzer. Each player's stint minutes
                  are scaled so they add to his box-score minutes (a period
                  boundary without a substitution event is inferred from who
                  acts first).
store_stints      the nba_stint table, replaced per game
stints_from_pbp(..., subs_out=[])
                  also collects the explicit substitution pairs the feed records ("SUB: Eason FOR Smith Jr."):
                  (team, out, in, elapsed, period), resolved with the floor state. store_subs -> nba_sub.
fouls_from_pbp    the foul events -> one row per personal foul: who, when (elapsed
                  seconds, period), his personal-foul count after it, the kind
                  (shooting, loose ball, offensive...). Technicals are kept with
                  personal = 0 and no count. The nba_foul table, replaced per game.
rotation_map      a team's trailing rotation as a grid: minutes per player per
                  game segment (eight 6-minute segments; overtime folded into
                  the last), weighted by recency, plus stint statistics
                  (stints per game, longest stint, share of minutes in the last
                  ten minutes of close games, share in garbage time).

Garbage time: the score is out of reach for the time left (margin >= 20 with
8 minutes left, 15 with 5, 10 with 2, interpolated).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import Column, Date, Float, Integer, String, UniqueConstraint, text
from sqlalchemy.orm import Session

from ivan.core import raw
from ivan.core.utils import normalize_name
from ivan.sports.nba.models import NBABase

Log = Callable[[str], None]
REG_PERIOD = 720.0
OT_PERIOD = 300.0
SEGMENTS = 8                      # six-minute segments of regulation
SEG_LEN = 2880.0 / SEGMENTS
_CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")
SETTLE_IN = True           # a player whose first substitution of a period is an 'in' sat the break
SETTLE_OUT = False         # (the 'out' inference was measured to hurt: 386 vs 64 mismatches on 110 games; off)


class NBAStint(NBABase):
    """One stretch on the floor for one player in one game."""
    __tablename__ = "nba_stint"
    __table_args__ = (UniqueConstraint("game_id", "player_name", "stint_no", name="uq_nba_stint"),)

    id          = Column(Integer, primary_key=True, autoincrement=True)
    game_id     = Column(String, nullable=False, index=True)
    game_date   = Column(Date, nullable=True, index=True)
    team        = Column(String, nullable=False, index=True)
    player_name = Column(String, nullable=False, index=True)       # normalized
    stint_no    = Column(Integer, nullable=False)
    start_sec   = Column(Float, nullable=False)                     # elapsed game seconds at entry
    end_sec     = Column(Float, nullable=False)
    minutes     = Column(Float, nullable=False)                     # scaled to the box score
    period_in   = Column(Integer, nullable=True)
    period_out  = Column(Integer, nullable=True)
    margin_in   = Column(Integer, nullable=True)                    # team minus opponent at entry
    margin_out  = Column(Integer, nullable=True)
    to_buzzer   = Column(Integer, nullable=True)                    # 1 when the stint ended with the game
    starter     = Column(Integer, nullable=True)
    raw_ratio   = Column(Float, nullable=True)                      # raw stint total / box minutes (far from 1 = a missed substitution)


class NBAFoul(NBABase):
    """One foul called on one player in one game."""
    __tablename__ = "nba_foul"
    __table_args__ = (UniqueConstraint("game_id", "player_name", "elapsed", "kind", name="uq_nba_foul"),)

    id          = Column(Integer, primary_key=True, autoincrement=True)
    game_id     = Column(String, nullable=False, index=True)
    game_date   = Column(Date, nullable=True, index=True)
    team        = Column(String, nullable=False, index=True)
    player_name = Column(String, nullable=False, index=True)       # normalized
    period      = Column(Integer, nullable=False)
    elapsed     = Column(Float, nullable=False)                     # elapsed game seconds
    foul_no     = Column(Integer, nullable=True)                    # his personal-foul count after this one (None for a technical)
    kind        = Column(String, nullable=True)                     # the play-by-play subType: Shooting, Personal, Loose Ball, Offensive, Technical...
    personal    = Column(Integer, nullable=False, default=1)        # 1 = counts toward six


class NBASub(NBABase):
    """One recorded substitution: player_in came on for player_out at `elapsed` game seconds."""
    __tablename__ = "nba_sub"
    __table_args__ = (UniqueConstraint("game_id", "team", "player_out", "player_in", "elapsed", name="uq_nba_sub"),)

    id          = Column(Integer, primary_key=True, autoincrement=True)
    game_id     = Column(String, nullable=False, index=True)
    game_date   = Column(Date, nullable=True, index=True)
    team        = Column(String, nullable=False, index=True)
    player_out  = Column(String, nullable=False, index=True)       # normalized
    player_in   = Column(String, nullable=False, index=True)
    period      = Column(Integer, nullable=False)
    elapsed     = Column(Float, nullable=False)


def ensure_tables(engine) -> None:
    NBABase.metadata.create_all(engine, tables=[NBAStint.__table__, NBAFoul.__table__, NBASub.__table__])


# ----------------------------------------------------------------------------- fetch
def fetch_game(game_id: str, timeout: int = 60) -> pd.DataFrame:
    from nba_api.stats.endpoints import playbyplayv3
    df = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=timeout).get_data_frames()[0]
    raw.store("nba", "nba_api/pbp", game_id, df.to_dict("records"))
    return df


def load_raw(game_id: str) -> Optional[pd.DataFrame]:
    art = raw.latest("nba", "nba_api/pbp", game_id)
    if art is None:
        return None
    return pd.DataFrame(raw.load(art.path, "nba"))


# ----------------------------------------------------------------------------- reconstruction
def clock_seconds(clock: str) -> Optional[float]:
    m = _CLOCK.match(str(clock or ""))
    return None if not m else int(m.group(1)) * 60 + float(m.group(2))


def period_length(period: int) -> float:
    return REG_PERIOD if period <= 4 else OT_PERIOD


def period_start_elapsed(period: int) -> float:
    return sum(period_length(p) for p in range(1, period))


def elapsed_at(period: int, clock: str) -> float:
    rem = clock_seconds(clock)
    return period_start_elapsed(period) + (period_length(period) - (rem if rem is not None else period_length(period)))


def game_length(max_period: int) -> float:
    return period_start_elapsed(max_period + 1)


def garbage_threshold(seconds_left: float) -> float:
    """Margin at which coaches pull the closers for the time left: 12 up with 90 seconds, 15 with 3 minutes, 20 with 6, 25 with 8; never before."""
    pts = [(90.0, 12.0), (180.0, 15.0), (360.0, 20.0), (480.0, 25.0)]
    if seconds_left <= pts[0][0]:
        return pts[0][1]
    if seconds_left >= pts[-1][0]:
        return 99.0
    return float(np.interp(seconds_left, [p[0] for p in pts], [p[1] for p in pts]))


def _sub_names(desc: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.match(r"\s*SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$", str(desc or ""))
    return (m.group(1), m.group(2)) if m else (None, None)


SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _strip_suffix(tokens: List[str]) -> List[str]:
    """['paul', 'reed', 'jr'] -> ['paul', 'reed']: the feed writes 'Reed' for Paul Reed Jr. and 'Holland II' for Ron Holland."""
    core = [t for t in tokens if t not in SUFFIXES]
    return core or tokens


def _last_name_map(box: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """
    team -> {key: [normalized full names]} with every spelling the feed uses for a player: the surname with and without
    his suffix ('reed', 'reedjr'), the full name squeezed ('paulreedjr', 'paulreed'), the name after the first token
    ('gilgeousalexander' for Shai Gilgeous-Alexander).
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for r in box.itertuples():
        n = normalize_name(r.player_name); toks = n.split(); core = _strip_suffix(toks)
        d = out.setdefault(r.team_abbreviation, {})
        keys = {toks[-1], core[-1], n.replace(" ", ""), "".join(core), "".join(toks[1:]) if len(toks) > 1 else toks[-1], "".join(core[1:]) if len(core) > 1 else core[-1]}
        for key in keys:
            d.setdefault(key, []).append(n)
    return out


def _label_tokens(label: str) -> List[str]:
    """A feed label without its suffix tokens: 'Holland II' -> ['holland'], 'Jal. Williams' -> ['jal', 'williams']."""
    return _strip_suffix(normalize_name(str(label or "")).split())


def _resolve_label(d: Dict[str, List[str]], label: str, initial: Optional[str] = None) -> Optional[str]:
    """A play-by-play label ('Smith Jr.', 'J. Smith Jr.') -> one normalized name from a team's last-name map, or None when ambiguous."""
    toks = _label_tokens(label)
    if not toks:
        return None
    lab = " ".join(toks)
    cands = d.get("".join(toks), []) or d.get(toks[-1], [])
    cands = list(dict.fromkeys(cands))
    if len(cands) == 1:
        return cands[0]
    if not cands:
        return None
    full = [c for c in cands if c == lab or c.endswith(" " + lab) or " ".join(_strip_suffix(c.split())) == lab]
    if len(full) == 1:
        return full[0]
    if len(toks) > 1:                                        # 'jal williams' (an abbreviated first name) -> jalen, not jaylin
        by_first = [c for c in cands if c.split()[0].startswith(toks[0])]
        if len(by_first) == 1:
            return by_first[0]
    ini = normalize_name(str(initial or "")).split()[0][:1] if initial else ""
    if ini:
        by_ini = [c for c in cands if c[:1] == ini]
        if len(by_ini) == 1:
            return by_ini[0]
    return None


_PF_NO = re.compile(r"\(P(\d+)[.)]")


def fouls_from_pbp(pbp: pd.DataFrame, box: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """
    One row per foul event with a player on one of the box's teams: player_name, team, period, elapsed, kind (subType),
    personal (0 for a technical), foul_no (the personal count after this foul: the (Pn) tag when the feed carries it,
    else the running count). Rows are ordered by time.
    """
    if pbp is None or pbp.empty or box.empty:
        return pd.DataFrame(columns=["game_id", "team", "player_name", "period", "elapsed", "kind", "personal", "foul_no"])
    f = pbp[pbp.actionType.astype(str).str.lower() == "foul"].copy()
    if f.empty:
        return pd.DataFrame(columns=["game_id", "team", "player_name", "period", "elapsed", "kind", "personal", "foul_no"])
    names = _last_name_map(box)
    f["elapsed"] = [elapsed_at(int(p), c) for p, c in zip(f.period, f.clock)]
    f = f.sort_values(["period", "elapsed", "actionNumber"])
    rows = []; count: Dict[str, int] = {}
    for r in f.itertuples():
        team = str(r.teamTricode or "")
        if team not in names:
            continue
        kind = str(getattr(r, "subType", "") or "").strip()
        desc = str(getattr(r, "description", "") or "")
        technical = kind.lower().startswith("technical") or "T.FOUL" in desc.upper().replace(" ", "")
        label = str(getattr(r, "playerName", "") or "") or desc.split(" ")[0]
        nm = _resolve_label(names[team], label, initial=getattr(r, "playerNameI", None))
        if nm is None:
            continue
        no = None
        if not technical:
            m = _PF_NO.search(desc)
            no = int(m.group(1)) if m else count.get(nm, 0) + 1
            count[nm] = no
        rows.append(dict(game_id=str(game_id), team=team, player_name=nm, period=int(r.period), elapsed=float(r.elapsed),
                         kind=kind or None, personal=0 if technical else 1, foul_no=no))
    df = pd.DataFrame(rows, columns=["game_id", "team", "player_name", "period", "elapsed", "kind", "personal", "foul_no"])
    df["foul_no"] = df.foul_no.astype(object).where(df.foul_no.notna(), None)      # a technical has no count: None, not NaN
    return df


def stints_from_pbp(pbp: pd.DataFrame, box: pd.DataFrame, game_id: str, subs_out: Optional[list] = None) -> pd.DataFrame:
    """
    box: this game's nba_stats rows (player_name normalized, team_abbreviation, minutes, position).
    Returns one row per stint with the NBAStint columns (minus id). With `subs_out` (a list), every resolved substitution
    pair is appended to it as dict(team, player_out, player_in, period, elapsed).
    """
    if pbp.empty or box.empty:
        return pd.DataFrame()
    pbp = pbp.copy()
    pbp["elapsed"] = [elapsed_at(int(p), c) for p, c in zip(pbp.period, pbp.clock)]
    pbp = pbp.sort_values(["period", "elapsed", "actionNumber"]).reset_index(drop=True)
    max_period = int(pbp.period.max()); end = game_length(max_period)
    # home team: the first scoring action whose team raises scoreHome
    home = None; last = (0, 0)
    for r in pbp.itertuples():
        try:
            sh, sa = int(r.scoreHome), int(r.scoreAway)
        except (TypeError, ValueError):
            continue
        if (sh, sa) != last and r.teamTricode:
            home = r.teamTricode if sh > last[0] else (None if sa <= last[1] else r.teamTricode)
            if sa > last[1] and sh == last[0]:
                home = None; away = r.teamTricode
                home = next((t for t in box.team_abbreviation.unique() if t != away), None)
            last = (sh, sa); break
    teams = list(box.team_abbreviation.unique())
    if home is None:
        home = teams[0]
    # running margin from the home side; filled forward
    sh = pd.to_numeric(pbp.scoreHome, errors="coerce").ffill().fillna(0); sa = pd.to_numeric(pbp.scoreAway, errors="coerce").ffill().fillna(0)
    pbp["margin_home"] = (sh - sa).astype(int)

    # PBP uses last names; map through the box: last name -> normalized full name within the team
    by_team_last = _last_name_map(box)

    on: Dict[str, set] = {}; entry: Dict[str, Tuple[float, int, int]] = {}

    def resolve(team: str, label: str, initial: Optional[str] = None, prefer_off: bool = False, strict: bool = False,
                description: Optional[str] = None) -> Optional[str]:
        """A play-by-play label ('Towns', 'K. Towns', 'Jones') -> the box score's normalized name on that team.
        Two players sharing a last name are told apart by the initial (the action's playerNameI), by the abbreviated
        first name the feed puts in the description when the initial is shared ('Jal. Williams' / 'Jay. Williams'),
        or, for a player coming in, by who is off the floor. With `strict` a tie is never guessed."""
        toks = _label_tokens(label)
        if not toks:
            return None
        lab = " ".join(toks)
        d = by_team_last.get(team, {})
        cands = d.get("".join(toks), []) or d.get(toks[-1], [])     # a hyphenated surname first ('gilgeous alexander' -> 'gilgeousalexander')
        cands = list(dict.fromkeys(cands))
        if len(cands) == 1:
            return cands[0]
        if not cands:
            return None
        full = [c for c in cands if c == lab or c.endswith(" " + lab) or " ".join(_strip_suffix(c.split())) == lab]
        if len(full) == 1:
            return full[0]
        if len(toks) > 1:                                       # 'jal williams' (an abbreviated first name) -> jalen, not jaylin
            by_first = [c for c in cands if c.split()[0].startswith(toks[0])]
            if len(by_first) == 1:
                return by_first[0]
        ini = normalize_name(str(initial or "")).split()[0][:1] if initial else ""
        if ini:
            by_ini = [c for c in cands if c[:1] == ini]
            if len(by_ini) == 1:
                return by_ini[0]
            if len(by_ini) > 1:
                cands = by_ini
        if description:                                          # 'MISS Jal. Williams 25' 3PT' names him by his first three letters
            desc = normalize_name(str(description))
            by_desc = [c for c in cands if f"{c.split()[0][:3]} {_strip_suffix(c.split())[-1]}" in desc]
            if len(by_desc) == 1:
                return by_desc[0]
        if strict:
            return None                                          # ambiguous: the caller would rather not guess
        if prefer_off:
            off = [c for c in cands if c not in on.get(team, set())]
            if len(off) == 1:
                return off[0]
        else:
            on_now = [c for c in cands if c in on.get(team, set())]
            if len(on_now) == 1:
                return on_now[0]
        return cands[0]

    rows = []
    starters = {t: {normalize_name(r.player_name) for r in box[box.team_abbreviation == t].itertuples() if str(r.position).upper() != "BENCH" and float(r.minutes or 0) > 0} for t in teams}
    for t in teams:
        on[t] = set(starters[t])
        for n in on[t]:
            entry[n] = (0.0, 1, 0)

    def margin_for(team: str, mh: int) -> int:
        return mh if team == home else -mh

    def close(name: str, team: str, at: float, period: int, mh: int, buzzer: bool):
        st, p_in, m_in = entry.pop(name)
        if at > st:
            rows.append(dict(game_id=game_id, team=team, player_name=name, start_sec=st, end_sec=at, period_in=p_in, period_out=period,
                             margin_in=m_in, margin_out=margin_for(team, mh), to_buzzer=int(buzzer), starter=int(name in starters[team])))

    def open_(name: str, team: str, at: float, period: int, mh: int):
        if name in entry:
            return
        entry[name] = (at, period, margin_for(team, mh)); on[team].add(name)

    # the substitution log misses the period breaks: a player whose first sub event of a period is an "in" was off at the
    # break, one whose first event is an "out" was on. Resolve them up front per period so the break can be settled at once.
    first_event: Dict[Tuple[int, str], str] = {}
    for r in pbp.itertuples():
        if str(r.actionType).lower() != "substitution" or r.teamTricode not in on:
            continue
        sub_in, sub_out = _sub_names(r.description); period = int(r.period)
        for lab, kind in ((sub_out, "out"), (sub_in, "in")):
            nm = resolve(r.teamTricode, lab, initial=getattr(r, "playerNameI", None) if kind == "out" else None, strict=True)
            if nm and (period, nm) not in first_event:
                first_event[(period, nm)] = kind

    def settle_break(period: int, mh: int):
        start = period_start_elapsed(period)
        for t in teams:
            for nm in list(on[t]):
                if SETTLE_IN and first_event.get((period, nm)) == "in":          # recorded coming in later: he sat the break
                    close(nm, t, start, period, mh, False); on[t].discard(nm)
            for (pp, nm), kind in first_event.items():
                if SETTLE_OUT and pp == period and kind == "out" and nm not in entry and nm in {normalize_name(x) for x in box[box.team_abbreviation == t].player_name}:
                    open_(nm, t, start, period, mh)              # recorded going out later without coming in: he started the period

    period_seen: Dict[str, set] = {}
    cur_period = 1
    for r in pbp.itertuples():
        period = int(r.period); at = float(r.elapsed); mh = int(r.margin_home)
        team = r.teamTricode if r.teamTricode in on else None
        if period != cur_period:
            cur_period = period; period_seen = {t: set() for t in teams}
            settle_break(period, mh)
        def open_at_break(name: str):
            """A player on the floor without a substitution event: he came in at the period break. Open him at the period
            start and, when that makes six, close whoever sat the break: a player who crossed the break (entered before the
            period started) and has done nothing this period; the most recent entrant comes off first."""
            start = period_start_elapsed(period)
            open_(name, team, start, period, mh)
            if len(on[team]) > 5:
                idle = [n for n in on[team] if n != name and n in entry and entry[n][0] < start and n not in period_seen.get(team, set())]
                if idle:
                    victim = max(idle, key=lambda n: entry[n][0])
                    close(victim, team, start, period, mh, False); on[team].discard(victim)

        if str(r.actionType).lower() == "substitution":
            if team is None:
                continue
            sub_in, sub_out = _sub_names(r.description)
            n_out = resolve(team, sub_out, initial=getattr(r, "playerNameI", None), description=r.description)      # the action's player is the one going out
            n_in = resolve(team, sub_in, prefer_off=True, description=r.description)
            if subs_out is not None and n_out and n_in:
                subs_out.append(dict(team=team, player_out=n_out, player_in=n_in, period=period, elapsed=at))
            if n_out and n_out not in entry and period > 1:
                open_at_break(n_out)                             # going out without ever coming in: he started the period
            if n_out and n_out in entry:
                close(n_out, team, at, period, mh, False); on[team].discard(n_out)
            if n_in:
                open_(n_in, team, at, period, mh)
            continue
        # any other action by a player not on the floor: he came in at the period break without a substitution event
        # (a shared surname that the initial and the description cannot settle is not guessed: a guess opened phantom stints)
        name = resolve(team, r.playerName, initial=getattr(r, "playerNameI", None), strict=True, description=r.description) if team and r.playerName else None
        if name and name not in entry and period > 1 and name not in period_seen.get(team, set()) and not any(
                row["player_name"] == name and row["period_in"] == period for row in rows):
            open_at_break(name)                                  # only a player who has not been on at all this period
        if name and team:
            period_seen.setdefault(team, set()).add(name)
    final_mh = int(pbp.margin_home.iloc[-1])
    for t in teams:
        for n in list(on[t]):
            if n in entry:
                close(n, t, end, max_period, final_mh, True)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["minutes"] = (df.end_sec - df.start_sec) / 60.0
    # scale every player's stints to his box minutes; a raw total far from the box means a substitution was missed
    box_min = {normalize_name(r.player_name): float(r.minutes or 0) for r in box.itertuples()}
    tot = df.groupby("player_name").minutes.transform("sum")
    df["raw_ratio"] = tot / df.player_name.map(box_min).replace(0, np.nan)
    df["minutes"] = df.minutes * df.player_name.map(box_min).fillna(0) / tot.replace(0, np.nan)
    df["minutes"] = df.minutes.fillna(0.0)
    df = df.sort_values(["team", "player_name", "start_sec"]).reset_index(drop=True)
    df["stint_no"] = df.groupby("player_name").cumcount() + 1
    return df


def store_stints(session: Session, game_id: str, game_date, df: pd.DataFrame) -> int:
    session.query(NBAStint).filter(NBAStint.game_id == str(game_id)).delete(synchronize_session=False)
    if df is None or df.empty:
        session.commit(); return 0
    recs = df.to_dict("records")
    for r in recs:
        r["game_date"] = game_date
    session.bulk_insert_mappings(NBAStint, recs); session.commit()
    return len(recs)


def store_fouls(session: Session, game_id: str, game_date, df: pd.DataFrame) -> int:
    session.query(NBAFoul).filter(NBAFoul.game_id == str(game_id)).delete(synchronize_session=False)
    if df is None or df.empty:
        session.commit(); return 0
    recs = df.drop_duplicates(["player_name", "elapsed", "kind"]).to_dict("records")
    for r in recs:
        r["game_date"] = game_date
        r["foul_no"] = None if r.get("foul_no") is None or (isinstance(r["foul_no"], float) and np.isnan(r["foul_no"])) else int(r["foul_no"])
    session.bulk_insert_mappings(NBAFoul, recs); session.commit()
    return len(recs)


def store_subs(session: Session, game_id: str, game_date, subs: list) -> int:
    session.query(NBASub).filter(NBASub.game_id == str(game_id)).delete(synchronize_session=False)
    if not subs:
        session.commit(); return 0
    seen = set(); recs = []
    for r in subs:
        key = (r["team"], r["player_out"], r["player_in"], round(float(r["elapsed"]), 1))
        if key in seen:
            continue
        seen.add(key); recs.append(dict(game_id=str(game_id), game_date=game_date, **r))
    session.bulk_insert_mappings(NBASub, recs); session.commit()
    return len(recs)


def game_subs(session: Session, game_id: str, team: Optional[str] = None) -> pd.DataFrame:
    """One game's recorded substitutions (one team when given): team, player_out, player_in, period, elapsed, ordered by time."""
    q = "select team, player_out, player_in, period, elapsed from nba_sub where game_id = :g" + (" and team = :t" if team else "") + " order by elapsed"
    return pd.read_sql(text(q).bindparams(g=str(game_id), **({"t": team} if team else {})), session.get_bind())


def sub_shares(session: Session, team: Optional[str] = None, players: Optional[List[str]] = None, before: Optional[date] = None,
               n_games: int = 15, halflife: float = 6.0) -> Dict[str, Dict[str, float]]:
    """
    out player -> {in player: share of his recorded single-swap exits that player covered}, recency weighted over the last
    n_games (by team, or by the given players' own games on any team). The coach's habit of who comes in for whom.
    """
    if players is not None:
        names = [str(n) for n in players]
        if not names:
            return {}
        marks = ", ".join(f":n{i}" for i in range(len(names)))
        params = {f"n{i}": n for i, n in enumerate(names)}
        q = (f"select s.game_id, s.game_date, s.team, s.player_out, s.player_in, (select count(*) from nba_sub o where o.game_id = s.game_id and o.team = s.team and o.elapsed = s.elapsed) n "
             f"from nba_sub s where s.player_out in ({marks})" + (" and s.game_date < :d" if before else ""))
    else:
        params = {"t": team}
        q = ("select s.game_id, s.game_date, s.team, s.player_out, s.player_in, (select count(*) from nba_sub o where o.game_id = s.game_id and o.team = s.team and o.elapsed = s.elapsed) n "
             "from nba_sub s where s.team = :t" + (" and s.game_date < :d" if before else ""))
    if before:
        params["d"] = before.isoformat()
    df = pd.read_sql(text(q + " order by s.game_date, s.game_id").bindparams(**params), session.get_bind())
    if df.empty:
        return {}
    # single swaps only: when several players change at once the feed's pairing is the scorer's typing order, not the
    # roles (2026-10-24: 40% same position group at multi-swaps against 61% at single swaps)
    df = df[df.n == 1]
    if df.empty:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for name, g in df.groupby("player_out"):
        games = list(dict.fromkeys(g.game_id))[-n_games:]
        g = g[g.game_id.isin(games)]
        w = {gid: 0.5 ** ((len(games) - 1 - i) / halflife) for i, gid in enumerate(games)}
        tot: Dict[str, float] = {}
        for gid, inn in zip(g.game_id, g.player_in):
            tot[inn] = tot.get(inn, 0.0) + w[gid]
        z = sum(tot.values())
        if z > 0:
            out[str(name)] = {k: v / z for k, v in sorted(tot.items(), key=lambda kv: -kv[1])}
    return out


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


def game_fouls(session: Session, game_id: str, team: Optional[str] = None) -> pd.DataFrame:
    """The personal fouls of one game (one team when given): player_name, period, elapsed, minute (0-47, overtime folded into 47), foul_no, kind."""
    q = "select player_name, team, period, elapsed, foul_no, kind, personal from nba_foul where game_id = :g" + (" and team = :t" if team else "") + " order by elapsed"
    df = pd.read_sql(text(q).bindparams(g=str(game_id), **({"t": team} if team else {})), session.get_bind())
    if df.empty:
        return df
    df["minute"] = np.minimum((df.elapsed // 60).astype(int), 47)
    return df


def fouls_per_game(session: Session, since: Optional[date] = None) -> Dict[Tuple[str, str], int]:
    """(game_id, player_name) -> personal fouls, for the box-score views."""
    q = "select game_id, player_name, sum(personal) n from nba_foul" + (" where game_date >= :d" if since else "") + " group by game_id, player_name"
    rows = session.execute(text(q).bindparams(**({"d": since.isoformat()} if since else {}))).all()
    return {(str(g), str(n)): int(c or 0) for g, n, c in rows}


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


# ----------------------------------------------------------------------------- the rotation map
def segment_minutes(start: float, end: float) -> np.ndarray:
    """Minutes of a stint falling in each of the eight regulation segments (overtime into the last)."""
    out = np.zeros(SEGMENTS)
    for k in range(SEGMENTS):
        a, b = k * SEG_LEN, (k + 1) * SEG_LEN
        if k == SEGMENTS - 1:
            b = 1e9
        out[k] = max(0.0, min(end, b) - max(start, a)) / 60.0
    return out


@dataclass
class RotationMap:
    team: str
    games: List[str]
    grid: pd.DataFrame          # index player, columns seg1..seg8 (expected minutes), total, plus stint stats


def rotation_map(session: Session, team: str, before: Optional[date] = None, n_games: int = 10, halflife: float = 4.0) -> Optional[RotationMap]:
    q = text("select * from nba_stint where team = :t" + (" and game_date < :d" if before else "") + " order by game_date, game_id")
    df = pd.read_sql(q.bindparams(t=team, **({"d": before.isoformat()} if before else {})), session.get_bind())
    if df.empty:
        return None
    games = list(dict.fromkeys(df.game_id))[-n_games:]
    df = df[df.game_id.isin(games)].copy()
    w_game = {g: 0.5 ** ((len(games) - 1 - i) / halflife) for i, g in enumerate(games)}
    rows: Dict[str, dict] = {}
    for name, g in df.groupby("player_name"):
        seg = np.zeros(SEGMENTS); wsum = 0.0; late_close = 0.0; late_total = 0.0; garbage = 0.0; total = 0.0
        n_st = []; longest = []
        for gid, s in g.groupby("game_id"):
            w = w_game[gid]; wsum += w
            segs = sum((segment_minutes(a, b) * (m / max(1e-9, (b - a) / 60.0)) for a, b, m in zip(s.start_sec, s.end_sec, s.minutes)), np.zeros(SEGMENTS))
            seg += w * segs; total += w * s.minutes.sum()
            n_st.append(len(s)); longest.append(s.minutes.max())
            for a, b, m, mo, buzz in zip(s.start_sec, s.end_sec, s.minutes, s.margin_out, s.to_buzzer):
                late = max(0.0, min(b, 2880.0) - max(a, 2280.0)) / 60.0           # the last ten minutes of regulation
                if late > 0 and abs(int(mo)) < 15:
                    late_close += w * late
                left = max(0.0, 2880.0 - b)
                if abs(int(mo)) >= garbage_threshold(left):
                    garbage += w * m
        if wsum == 0:
            continue
        rec = {f"seg{k + 1}": seg[k] / wsum for k in range(SEGMENTS)}
        rec.update(total=total / wsum, games=len(n_st), stints=float(np.mean(n_st)), longest=float(np.mean(longest)),
                   late_close=late_close / wsum, garbage_share=(garbage / max(1e-9, total)) if total else 0.0,
                   started=float(g.groupby("game_id").starter.max().mean()))
        rows[name] = rec
    grid = pd.DataFrame(rows).T.sort_values("total", ascending=False)
    return RotationMap(team=team, games=games, grid=grid)


def format_rotation(rm: RotationMap, min_total: float = 3.0) -> str:
    g = rm.grid[rm.grid.total >= min_total]
    L = [f"{rm.team}: trailing {len(rm.games)} games, minutes per 6-minute segment (Q1a Q1b Q2a Q2b Q3a Q3b Q4a Q4b), recency-weighted",
         f"  {'player':22s} {'tot':>5s} " + " ".join(f"{s:>4s}" for s in ("Q1a", "Q1b", "Q2a", "Q2b", "Q3a", "Q3b", "Q4a", "Q4b")) + f"  {'stints':>6s} {'long':>5s} {'close10':>7s} {'garb%':>5s} {'start':>5s}"]
    for name, r in g.iterrows():
        L.append(f"  {name[:22]:22s} {r.total:5.1f} " + " ".join(f"{r[f'seg{k + 1}']:4.1f}" for k in range(SEGMENTS))
                 + f"  {r.stints:6.1f} {r.longest:5.1f} {r.late_close:7.1f} {100 * r.garbage_share:5.0f} {100 * r.started:5.0f}")
    seg_sum = g[[f"seg{k + 1}" for k in range(SEGMENTS)]].sum()
    L.append(f"  {'team':22s} {g.total.sum():5.1f} " + " ".join(f"{v:4.1f}" for v in seg_sum) + "   (each segment should hold 30)")
    return "\n".join(L)
