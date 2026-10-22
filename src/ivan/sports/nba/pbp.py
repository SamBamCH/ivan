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


def ensure_tables(engine) -> None:
    NBABase.metadata.create_all(engine, tables=[NBAStint.__table__])


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
    """Margin beyond which the game is decided for the time left: 20 at 8 min, 15 at 5, 10 at 2, flat outside."""
    pts = [(120.0, 10.0), (300.0, 15.0), (480.0, 20.0)]
    if seconds_left <= pts[0][0]:
        return pts[0][1]
    if seconds_left >= pts[-1][0]:
        return 99.0
    return float(np.interp(seconds_left, [p[0] for p in pts], [p[1] for p in pts]))


def _sub_names(desc: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.match(r"\s*SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$", str(desc or ""))
    return (m.group(1), m.group(2)) if m else (None, None)


def stints_from_pbp(pbp: pd.DataFrame, box: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """
    box: this game's nba_stats rows (player_name normalized, team_abbreviation, minutes, position).
    Returns one row per stint with the NBAStint columns (minus id).
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
    by_team_last: Dict[str, Dict[str, List[str]]] = {}
    for r in box.itertuples():
        n = normalize_name(r.player_name)
        by_team_last.setdefault(r.team_abbreviation, {}).setdefault(n.split()[-1], []).append(n)
        by_team_last[r.team_abbreviation].setdefault(n.replace(" ", ""), []).append(n)

    def resolve(team: str, label: str) -> Optional[str]:
        lab = normalize_name(str(label or ""))
        if not lab:
            return None
        cands = by_team_last.get(team, {}).get(lab.split()[-1], []) or by_team_last.get(team, {}).get(lab.replace(" ", ""), [])
        if len(cands) == 1:
            return cands[0]
        full = [c for c in cands if c == lab or c.endswith(lab)]
        return full[0] if len(full) == 1 else (cands[0] if cands else None)

    on: Dict[str, set] = {}; entry: Dict[str, Tuple[float, int, int]] = {}
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

    period_seen: Dict[str, set] = {}
    cur_period = 1
    for r in pbp.itertuples():
        period = int(r.period); at = float(r.elapsed); mh = int(r.margin_home)
        team = r.teamTricode if r.teamTricode in on else None
        if period != cur_period:
            cur_period = period; period_seen = {t: set() for t in teams}
        if str(r.actionType).lower() == "substitution":
            if team is None:
                continue
            sub_in, sub_out = _sub_names(r.description)
            n_in, n_out = resolve(team, sub_in), resolve(team, sub_out)
            if n_out and n_out in entry:
                close(n_out, team, at, period, mh, False); on[team].discard(n_out)
            if n_in:
                open_(n_in, team, at, period, mh)
            continue
        # any other action by a player not on the floor: he came in at the period break without a substitution event
        name = resolve(team, r.playerName) if team and r.playerName else None
        if name and name not in entry and period > 1:
            start = period_start_elapsed(period)
            open_(name, team, start, period, mh)
            if len(on[team]) > 5:
                # someone without an action this period and no substitution so far must have sat at the break
                idle = [n for n in on[team] if n != name and n not in period_seen.get(team, set())]
                if idle:
                    victim = max(idle, key=lambda n: entry[n][0] if n in entry else 0)
                    close(victim, team, start, period, mh, False); on[team].discard(victim)
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
    # scale every player's stints to his box minutes
    box_min = {normalize_name(r.player_name): float(r.minutes or 0) for r in box.itertuples()}
    tot = df.groupby("player_name").minutes.transform("sum")
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


def box_for_game(session: Session, game_id: str) -> pd.DataFrame:
    return pd.read_sql(text("select player_name, team_abbreviation, minutes, position, game_date from nba_stats where game_id = :g").bindparams(g=str(game_id)), session.get_bind())


def process_game(session: Session, game_id: str, log: Log = print, refetch: bool = False) -> int:
    pbp = None if refetch else load_raw(game_id)
    if pbp is None:
        pbp = fetch_game(game_id)
    box = box_for_game(session, game_id)
    df = stints_from_pbp(pbp, box, game_id)
    gd = pd.to_datetime(box.game_date.iloc[0]).date() if len(box) else None
    n = store_stints(session, game_id, gd, df)
    log(f"{game_id} {gd}: {n} stints, {df.player_name.nunique() if len(df) else 0} players")
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
