"""
Rotations as a grid: who is on the floor in each minute of the game.

curves(session, team)      one row per player, 48 columns: the recency-weighted
                           share of recent games he was on the floor during
                           that minute (from the stint table, pbp.py)
typical_stints(row)        the contiguous minutes where that share is above a half
propose(curves, available, starters)
                           a full rotation for tonight: exactly five players in
                           every minute, chosen minute by minute by their curve
                           with hysteresis so stints hold together, starters
                           forced on at the tip, scratched players never on,
                           and a player's minutes capped near his own history
minutes_of(grid)           minutes per player (the row sums)
save_drawn / load_drawn    a hand-drawn grid per team (data/nba/input/rotations/<team>.json):
                           what the desk writes when a rotation is applied and
                           what the projection reads as pinned minutes

The grid is boolean players x 48; the sampler reads stints and closers off it.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.core.utils import normalize_name
from ivan.sports.nba.paths import INPUT_DIR

MINUTES = 48
ROTATIONS_DIR = INPUT_DIR / "rotations"
ON_COURT = 5
HYSTERESIS = 0.15          # a player on the floor stays until someone's curve beats his by this
MIN_STINT = 2              # minutes: shorter stints are merged away in the proposal
GARBAGE_FROM = 34          # garbage time is only looked for from this minute on
DEFICIT = 0.5              # weight of the minutes a player is still owed against his curve
MAX_MINUTES = 40.0         # ceiling in a proposal (the top of the league plays 36-38)
MAX_STINT = 15             # minutes on the floor before a breather is forced
OPENING = 3                # minutes the named starters hold the floor before any swap


def curves(session: Session, team: Optional[str] = None, before: Optional[date] = None, n_games: int = 10, halflife: float = 4.0,
           players: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    index player, columns 0..47 (minute of regulation), values 0..1; plus games,
    minutes (expected), stints, played_share. With `players`, each player's own
    last n games on any team (his pattern travels with him to a new roster);
    with `team`, the team's last n games.
    """
    if players is not None:
        names = [str(n) for n in players]
        if not names:
            return pd.DataFrame()
        marks = ", ".join(f":n{i}" for i in range(len(names)))
        params = {f"n{i}": n for i, n in enumerate(names)}
        if before:
            params["d"] = before.isoformat()
        q = text(f"select game_id, game_date, player_name, start_sec, end_sec, minutes, to_buzzer, margin_out from nba_stint where player_name in ({marks})"
                 + (" and game_date < :d" if before else "") + " order by game_date, game_id").bindparams(**params)
    else:
        q = text("select game_id, game_date, player_name, start_sec, end_sec, minutes, to_buzzer, margin_out from nba_stint where team = :t"
                 + (" and game_date < :d" if before else "") + " order by game_date, game_id").bindparams(t=team, **({"d": before.isoformat()} if before else {}))
    df = pd.read_sql(q, session.get_bind())
    if df.empty:
        return pd.DataFrame()
    # the plan for a close game: in a game that went to garbage time, the closers (that game's five most-used players)
    # count as on the floor from the onset and the garbage players as off, so the curve is "who plays when it matters"
    # and the sampler applies garbage time per world without counting it twice
    onset = garbage_onsets(df)
    rows = {}
    for name, g in df.groupby("player_name"):
        games = list(dict.fromkeys(g.game_id))[-n_games:]
        g = g[g.game_id.isin(games)]
        w = {gid: 0.5 ** ((len(games) - 1 - i) / halflife) for i, gid in enumerate(games)}
        wsum = sum(w.values())
        on = np.zeros(MINUTES); n_st = []
        for gid, s in g.groupby("game_id"):
            cur = np.zeros(MINUTES)
            for a, b in zip(s.start_sec, s.end_sec):
                lo, hi = int(a // 60), int(min(b, 2880) // 60 + (1 if min(b, 2880) % 60 > 0 else 0))
                for m in range(max(0, lo), min(MINUTES, hi)):
                    cur[m] = max(cur[m], min(1.0, (min(b, (m + 1) * 60) - max(a, m * 60)) / 60.0))
            g_on, closers = onset.get(gid, (None, set()))
            if g_on is not None:
                cur[g_on:] = 1.0 if name in closers else 0.0
            on += w[gid] * cur; n_st.append(len(s))
        row = {m: on[m] / wsum for m in range(MINUTES)}
        row.update(games=len(n_st), minutes=float(on.sum() / wsum), stints=float(np.mean(n_st)), played_share=1.0,
                   last_game=str(g.game_date.max()))
        rows[name] = row
    out = pd.DataFrame(rows).T
    return out.sort_values("minutes", ascending=False)


def garbage_onsets(stints: pd.DataFrame) -> Dict[str, Tuple[Optional[int], set]]:
    """game_id -> (minute garbage time began or None, that game's closers for the stints' team): the earliest exit in the last
    fourteen minutes at a margin beyond the pull threshold for the time left."""
    from ivan.sports.nba.pbp import garbage_threshold
    out = {}
    for gid, g in stints.groupby("game_id"):
        tot = g.groupby("player_name").minutes.sum().sort_values(ascending=False)
        closers = set(tot.index[:5])
        on = None
        late = g[(g.end_sec >= GARBAGE_FROM * 60) & (g.end_sec < 2880) & g.margin_out.notna()]
        for e, mo in zip(late.end_sec, late.margin_out):
            if abs(int(mo)) >= garbage_threshold(2880 - float(e)):
                on = int(e // 60) if on is None else min(on, int(e // 60))
        out[gid] = (on, closers)
    return out


def typical_stints(row: Sequence[float], threshold: float = 0.5) -> List[Tuple[int, int]]:
    """[(start minute, end minute)) runs where the on-court share is above threshold."""
    on = np.asarray(row[:MINUTES], dtype=float) > threshold
    out, start = [], None
    for m, v in enumerate(on):
        if v and start is None:
            start = m
        if not v and start is not None:
            out.append((start, m)); start = None
    if start is not None:
        out.append((start, MINUTES))
    return out


def propose(curve: pd.DataFrame, available: Iterable[str], starters: Optional[Iterable[str]] = None, salary_prior: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    A boolean grid (index player, columns 0..47) with exactly five on the floor
    every minute. Players without a curve (no history on this team) get a flat
    prior from salary_prior minutes (spread evenly) so they can be drawn in.
    """
    avail = [n for n in available]
    starters = set(starters or [])
    P = np.zeros((len(avail), MINUTES))
    for i, n in enumerate(avail):
        if n in curve.index:
            P[i] = curve.loc[n, list(range(MINUTES))].values.astype(float)
        elif salary_prior and salary_prior.get(n):
            P[i] = min(0.95, float(salary_prior[n]) / MINUTES)
    # every player's target is his expected minutes, scaled so the available roster adds to 240; the minute-by-minute
    # pick follows his curve (when he plays) and his remaining target (how much he still has to play)
    hist = np.array([float(curve.loc[n, "minutes"]) if n in curve.index else float((salary_prior or {}).get(n, 4.0)) for n in avail])
    target = scale_targets(hist)
    cap = np.minimum(MAX_MINUTES, target * 1.08 + 1.5)
    grid = np.zeros((len(avail), MINUTES), dtype=bool)
    streak = np.zeros(len(avail))                                  # consecutive minutes on the floor
    on = set()
    if starters:
        on = {i for i, n in enumerate(avail) if n in starters}
        if len(on) > ON_COURT:
            on = set(sorted(on, key=lambda i: -P[i, 0])[:ON_COURT])
    used = np.zeros(len(avail))
    for m in range(MINUTES):
        left = MINUTES - m
        need = np.maximum(target - used, 0.0)                      # minutes still owed
        score = P[:, m] + DEFICIT * (need / np.maximum(target, 1.0)) + 0.6 * np.clip(need - left * 0.8, 0, None) / max(1.0, left)
        score[used >= cap] = -1.0                                  # out of minutes
        score[streak >= MAX_STINT] -= 5.0                          # a breather after a long stretch (nobody plays 20 straight)
        if m < OPENING and starters and m == 0:
            cur = set(on)
            while len(cur) < ON_COURT:
                rest = [i for i in np.argsort(-score) if i not in cur]
                cur.add(int(rest[0]))
        elif m < OPENING and starters:
            cur = set(on)                                          # the opening unit plays the first minutes
        else:
            cur = set(on)
            # keep the floor unless a bench player's curve beats a floor player's by the hysteresis
            for _ in range(ON_COURT):
                if not cur:
                    break
                worst = min(cur, key=lambda i: score[i])
                best_off = max((i for i in range(len(avail)) if i not in cur), key=lambda i: score[i], default=None)
                if best_off is not None and score[best_off] > score[worst] + HYSTERESIS:
                    cur.remove(worst); cur.add(int(best_off))
                else:
                    break
            while len(cur) < ON_COURT:
                rest = [int(i) for i in np.argsort(-score) if i not in cur]
                if not rest:
                    break
                cur.add(rest[0])
            while len(cur) > ON_COURT:
                cur.remove(min(cur, key=lambda i: score[i]))
        for i in cur:
            grid[i, m] = True; used[i] += 1
        streak = np.where(np.isin(np.arange(len(avail)), list(cur)), streak + 1, 0.0)
        on = cur
    grid = _merge_short_stints(grid, P)
    return pd.DataFrame(grid, index=avail, columns=list(range(MINUTES)))


def scale_targets(hist: np.ndarray, budget: float = ON_COURT * MINUTES, cap: float = MAX_MINUTES, cut_tau: float = 8.0, headroom_cap: float = 38.0) -> np.ndarray:
    """
    Each player's target from his when-played minutes so the roster adds to the
    budget, the way the allocator does it: a surplus (a full roster listed, a
    coach who plays nine) comes off the bottom of the rotation (weights
    m x exp(-m / cut_tau)), a shortfall (someone out) goes to the players with
    headroom (weights m x (headroom_cap - m)), nobody above the cap.
    """
    t = np.minimum(np.asarray(hist, dtype=float), cap)
    for _ in range(12):
        gap = budget - t.sum()
        if abs(gap) < 0.05:
            break
        if gap < 0:
            w = t * np.exp(-t / cut_tau); w[t <= 0] = 0.0
            if w.sum() <= 0:
                break
            t = np.maximum(t + gap * w / w.sum(), 0.0)
        else:
            w = t * np.maximum(headroom_cap - t, 1.0); w[t < 4.0] = 0.0; w[t >= cap] = 0.0
            if w.sum() <= 0:
                w = np.where(t < cap, 1.0, 0.0)
                if w.sum() <= 0:
                    break
            t = np.minimum(t + gap * w / w.sum(), cap)
    return t


def _merge_short_stints(grid: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Remove stints shorter than MIN_STINT by handing their minutes to the best off-floor player at that time (keeps five on)."""
    g = grid.copy()
    for i in range(g.shape[0]):
        for a, b in typical_stints(g[i].astype(float)):
            if b - a < MIN_STINT:
                for m in range(a, b):
                    g[i, m] = False
                    off = [j for j in range(g.shape[0]) if not g[j, m] and j != i]
                    if off:
                        g[max(off, key=lambda j: P[j, m]), m] = True
    return g


def minutes_of(grid: pd.DataFrame) -> Dict[str, float]:
    return {str(n): float(v) for n, v in grid.sum(axis=1).items()}


def stints_of(grid: pd.DataFrame) -> Dict[str, List[Tuple[int, int]]]:
    return {str(n): typical_stints(grid.loc[n].values.astype(float)) for n in grid.index}


def column_counts(grid: pd.DataFrame) -> List[int]:
    return [int(x) for x in grid.sum(axis=0).values]


# ----------------------------------------------------------------------------- the five spots
SPOT_ORDER = {"PG": 0, "SG": 1, "G": 1, "SF": 2, "F": 2, "SF/PF": 2, "PF": 3, "C": 4, "PF/C": 4, "F/C": 4}


def _spot_rank(position: Optional[str], group: Optional[str]) -> float:
    """Starters sort into the five spots guard -> wing -> big, by their DK positions (PG first, C last)."""
    pos = [x.strip().upper() for x in str(position or "").split("/") if x.strip()]
    ranks = [SPOT_ORDER[x] for x in pos if x in SPOT_ORDER]
    if ranks:
        return float(np.mean(ranks))
    return {"guard": 0.5, "wing": 2.0, "big": 4.0}.get(group or "", 2.0)


def spots_of(grid: pd.DataFrame, starters: Optional[Iterable[str]] = None, groups: Optional[Dict[str, str]] = None,
             positions: Optional[Dict[str, str]] = None, pairs: Optional[Iterable[dict]] = None,
             shares: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Which of the five lineup spots each player fills in each minute: the five on the floor at the tip own spots 0-4
    (guards first, bigs last), and when a player goes off, the player who comes on in his place inherits his spot.
    Who came on for whom is, in order of trust:
      1. the recorded substitution when one player came in for one (`pairs`: dicts with player_out, player_in, elapsed),
      2. a returning player's own spot when it is freed in the same minute,
      3. the likest player going off: the same position group first, within it the coach's habit (`shares`: out -> {in:
         share of his single-swap exits}, pbp.sub_shares), then the nearest position order.
    Returns (spots: index as grid, columns 0..47, -1 off the floor / unassigned, and the spot owners' names in order).
    A minute with more than five on holds the extras at -1; a spot nobody fills stays free for the next player in.
    """
    groups = groups or {}; positions = positions or {}; shares = shares or {}
    names = list(grid.index); G = grid[list(range(MINUTES))].values.astype(bool)
    idx = {n: i for i, n in enumerate(names)}
    S = np.full(G.shape, -1, dtype=int)
    tip = [n for n in names if G[names.index(n), 0]]
    st = [n for n in (starters or []) if n in tip] or tip
    st = sorted(st, key=lambda n: (_spot_rank(positions.get(n), groups.get(n)), n))[:ON_COURT]
    for k, n in enumerate(st):
        S[names.index(n), 0] = k
    free = [k for k in range(ON_COURT) if k not in set(range(len(st)))]
    rank = {n: _spot_rank(positions.get(n), groups.get(n)) for n in names}
    last_spot = {names.index(n): k for k, n in enumerate(st)}        # the spot each player held last (a returning player reclaims his when it is free)
    # recorded pairs count only when one player came in for one: at a multi-swap the feed pairs them in the scorer's
    # typing order (40% same position group against 61% at single swaps, 2026-10-24), so those moments are re-paired below
    by_moment: Dict[float, List[dict]] = {}
    for pr in pairs or []:
        try:
            by_moment.setdefault(round(float(pr["elapsed"]), 1), []).append(pr)
        except (KeyError, TypeError, ValueError):
            continue
    by_minute: Dict[int, List[Tuple[str, str]]] = {}
    for t, prs in by_moment.items():
        if len(prs) != 1:
            continue
        m0 = int(t // 60)
        for mm in (m0, m0 + 1):                                       # the grid's cells turn at the minute of the sub or the next
            by_minute.setdefault(mm, []).append((str(prs[0]["player_out"]), str(prs[0]["player_in"])))
    for m in range(1, MINUTES):
        S[:, m] = np.where(G[:, m] & G[:, m - 1], S[:, m - 1], -1)
        outs = [i for i in range(len(names)) if G[i, m - 1] and not G[i, m] and S[i, m - 1] >= 0]
        ins = [i for i in range(len(names)) if G[i, m] and not G[i, m - 1]]
        free_spots = [S[i, m - 1] for i in outs] + free
        free = []
        # 1. the recorded pairs at this minute
        for o_name, i_name in by_minute.get(m, []):
            o, i = idx.get(o_name), idx.get(i_name)
            if o is None or i is None or o not in outs or i not in ins:
                continue
            sp = S[o, m - 1]
            if sp in free_spots:
                S[i, m] = sp; last_spot[i] = sp; free_spots.remove(sp); outs.remove(o); ins.remove(i)
        # 2-4. the rest: own spot first, then the coach's habit, then the likest player going off
        def habit(i, sp):
            o = next((x for x in outs if S[x, m - 1] == sp), None)
            return float(shares.get(names[o], {}).get(names[i], 0.0)) if o is not None else 0.0
        ins_sorted = sorted(ins, key=lambda i: (0 if last_spot.get(i, -1) in free_spots else 1, rank[names[i]]))
        for i in ins_sorted:
            if not free_spots:
                break
            if last_spot.get(i, -1) in free_spots:
                best = last_spot[i]
            else:
                best = None; best_key = None
                for sp in free_spots:
                    owner = next((o for o in outs if S[o, m - 1] == sp), None)
                    h = habit(i, sp)
                    if owner is not None:      # the same line first; within it the coach's habit, then the nearest position
                        key = (0 if groups.get(names[owner], "wing") == groups.get(names[i], "wing") else 1, -h, abs(rank[names[owner]] - rank[names[i]]))
                    else:
                        key = (2, 0.0, 0.0)
                    if best_key is None or key < best_key:
                        best, best_key = sp, key
            S[i, m] = best; last_spot[i] = best; free_spots.remove(best)
        free = free_spots                                   # a spot nobody took stays free for the next entrant
    return pd.DataFrame(S, index=grid.index, columns=list(range(MINUTES))), st


# ----------------------------------------------------------------------------- the last game's rotation
def team_games(session: Session, team: str, before: Optional[date] = None, since: Optional[date] = None, limit: int = 20) -> List[dict]:
    """The team's games with stints, newest first: game_id, game_date, opp (the other team in the same game, when stored)."""
    q = text("select s.game_id, s.game_date, (select min(o.team) from nba_stint o where o.game_id = s.game_id and o.team != s.team) as opp"
             " from nba_stint s where s.team = :t" + (" and s.game_date < :d" if before else "") + (" and s.game_date >= :s" if since else "")
             + " group by s.game_id, s.game_date order by s.game_date desc, s.game_id desc limit :n")
    rows = session.execute(q.bindparams(t=team, n=int(limit), **({"d": before.isoformat()} if before else {}), **({"s": since.isoformat()} if since else {}))).all()
    return [dict(game_id=str(g), game_date=str(d), opp=o or "") for g, d, o in rows]


def played_sets(session: Session, team: str, game_ids: Iterable[str], min_minutes: float = 5.0) -> Dict[str, Dict[str, float]]:
    """{game_id: {player: minutes}} for the players who played at least `min_minutes` in each game (garbage-time cameos left out)."""
    ids = [str(g) for g in game_ids]
    if not ids:
        return {}
    q = text("select game_id, player_name, sum(minutes) as m from nba_stint where team = :t and game_id in (" + ",".join(f"'{g}'" for g in ids) + ")"
             " group by game_id, player_name having sum(minutes) >= :mm")
    out: Dict[str, Dict[str, float]] = {g: {} for g in ids}
    for g, n, m in session.execute(q.bindparams(t=team, mm=float(min_minutes))).all():
        out[str(g)][str(n)] = float(m)
    return out


def lineup_matches(games: List[dict], played: Dict[str, Dict[str, float]], tonight: Dict[str, float]) -> List[dict]:
    """
    How close each played game's group is to tonight's: `tonight` = {player: usual minutes} for the players available
    tonight who matter (the rotation), `played` from played_sets. Each game gets missing (tonight's players who did not
    play that night, with their usual minutes), extra (players who played that night and are not available tonight,
    with the minutes they played) and a score = the minutes on both lists; 0 = the same group. Games newest first, so
    the best score's first holder is the last time this group played together.
    """
    out = []
    for g in games:
        pl = played.get(str(g["game_id"]), {})
        missing = {n: m for n, m in tonight.items() if n not in pl}
        extra = {n: m for n, m in pl.items() if n not in tonight}
        out.append(dict(g, missing=missing, extra=extra, score=float(sum(missing.values()) + sum(extra.values())), exact=not missing and not extra))
    return out


BLOWOUT_MARGIN = 20          # a final margin this wide is a blowout even when the stints never crossed the garbage threshold
LEFT_EARLY_SEC = 12 * 60     # a rotation player whose last exit came with this much time left, before garbage time, left early (injury, ejection)
LEFT_EARLY_SHARE = 0.6       # ... and played under this share of his usual minutes
FOUL_TROUBLE = 5


def game_oddities(session: Session, team: str, game_ids: Iterable[str], usual: Optional[Dict[str, float]] = None) -> Dict[str, dict]:
    """
    Was each game ordinary, or not a picture of the coach's plan? {game_id: odd, reasons, margin, ot, garbage_from, fouls,
    left_early}. Odd when: a blowout (garbage time began, or the final margin is BLOWOUT_MARGIN+), overtime, a player in
    foul trouble (FOUL_TROUBLE+ personal fouls), or a rotation player who left early (last exit LEFT_EARLY_SEC+ before the
    end and before garbage time, under LEFT_EARLY_SHARE of his usual minutes). The desk prefers an ordinary game as the
    played-game default and dims the garbage-time columns.
    """
    ids = [str(g) for g in game_ids]
    out: Dict[str, dict] = {g: dict(odd=False, reasons=[], margin=None, ot=False, garbage_from=None, fouls={}, left_early=[]) for g in ids}
    if not ids:
        return out
    inl = ",".join(f"'{g}'" for g in ids)
    st = pd.read_sql(text(f"select game_id, player_name, start_sec, end_sec, minutes, margin_out, to_buzzer from nba_stint where team = :t and game_id in ({inl})").bindparams(t=team),
                     session.get_bind())
    if st.empty:
        return out
    st["game_id"] = st["game_id"].astype(str)
    onsets = garbage_onsets(st)
    fouls = pd.read_sql(text(f"select game_id, player_name, count(*) as n from nba_foul where team = :t and personal = 1 and game_id in ({inl}) group by game_id, player_name").bindparams(t=team),
                        session.get_bind())
    usual = usual or {}
    for gid, g in st.groupby("game_id"):
        o = out[str(gid)]
        last = g.sort_values("end_sec").iloc[-1]
        o["margin"] = None if pd.isna(last.margin_out) else int(last.margin_out)
        o["ot"] = bool(g.end_sec.max() > 2880)
        on, _ = onsets.get(gid, (None, set()))
        o["garbage_from"] = on
        if on is not None:
            o["reasons"].append(f"blowout {o['margin']:+d}, garbage time from minute {on}" if o["margin"] is not None else f"garbage time from minute {on}")
        elif o["margin"] is not None and abs(o["margin"]) >= BLOWOUT_MARGIN:
            o["reasons"].append(f"blowout {o['margin']:+d}")
        if o["ot"]:
            o["reasons"].append("overtime")
        f = fouls[fouls.game_id.astype(str) == str(gid)]
        o["fouls"] = {str(r.player_name): int(r.n) for r in f.itertuples()}
        for n, k in o["fouls"].items():
            if k >= FOUL_TROUBLE:
                o["reasons"].append(f"{n.split()[-1]} {k} fouls")
        cutoff = (on * 60) if on is not None else 2880
        for n, pg in g.groupby("player_name"):
            u = float(usual.get(str(n), 0.0))
            end = float(pg.end_sec.max()); mins = float(pg.minutes.sum())
            if u >= 15 and end <= cutoff - LEFT_EARLY_SEC and mins < LEFT_EARLY_SHARE * u:
                o["left_early"].append(str(n)); o["reasons"].append(f"{n.split()[-1]} left early ({mins:.0f} of {u:.0f} min)")
        o["odd"] = bool(o["reasons"])
    return out


def best_match(matches: List[dict]) -> Optional[dict]:
    """The lowest score among the ordinary games, else the lowest overall; the newest game on ties (the list is newest first)."""
    if not matches:
        return None
    return min(matches, key=lambda m: (bool(m.get("odd")), m["score"]))


def last_game_grid(session: Session, team: str, before: Optional[date] = None, since: Optional[date] = None, game_id: Optional[str] = None) -> Optional[dict]:
    """
    The team's most recent game (or `game_id`) as a grid (index player, columns
    0..47; each stint runs from its start rounded to the nearest minute to its
    end rounded the same way; overtime folded into the last minute), with the
    game id and date. None
    when the team has no stints (since `since`, when given: the desk asks for
    this season only).
    """
    if game_id:
        row = session.execute(text("select game_id, game_date from nba_stint where team = :t and game_id = :g limit 1").bindparams(t=team, g=str(game_id))).first()
    else:
        q = text("select game_id, game_date from nba_stint where team = :t" + (" and game_date < :d" if before else "") + (" and game_date >= :s" if since else "")
                 + " order by game_date desc, game_id desc limit 1")
        row = session.execute(q.bindparams(t=team, **({"d": before.isoformat()} if before else {}), **({"s": since.isoformat()} if since else {}))).first()
    if row is None:
        return None
    gid, gdate = row
    st = pd.read_sql(text("select player_name, start_sec, end_sec, minutes from nba_stint where game_id = :g and team = :t").bindparams(g=gid, t=team), session.get_bind())
    names = list(dict.fromkeys(st.player_name))
    grid = np.zeros((len(names), MINUTES), dtype=bool)
    # every boundary is cut at its nearest minute, so one substitution time moves the player going off and the player
    # coming on in the same column (the older "on if 30 seconds of the minute" rule put both on at an exact half minute
    # and let a swap straddle two columns: 102 columns without five on the floor over 50 team-games, 2026-10-24)
    for i, nm in enumerate(names):
        for a, b in zip(st[st.player_name == nm].start_sec, st[st.player_name == nm].end_sec):
            a_m = int(np.floor(float(a) / 60.0 + 0.5)); b_m = int(np.floor(min(float(b), 2880.0) / 60.0 + 0.5))
            for m in range(max(0, a_m), min(MINUTES, b_m)):          # a stint that rounds to nothing shows nothing (a forced cell made six on the floor)
                grid[i, m] = True
    g = pd.DataFrame(grid, index=names, columns=list(range(MINUTES)))
    g = g.loc[sorted(g.index, key=lambda n: -int(g.loc[n].sum()))]
    return {"grid": g, "game_id": str(gid), "game_date": str(gdate)}


# ----------------------------------------------------------------------------- drawn rotations (the desk's locked rows)
def drawn_path(team: str) -> Path:
    return ROTATIONS_DIR / f"{team}.json"


def save_drawn(team: str, grid: pd.DataFrame, note: str = "", locked: Optional[Iterable[str]] = None) -> Path:
    """
    The team's painted rotation. `locked` names the rows the desk holds exactly (the skeleton fills the others around
    them, projection/skeleton.py); None locks every row (the older meaning of a drawn grid).
    """
    p = drawn_path(team); p.parent.mkdir(parents=True, exist_ok=True)
    names = [str(n) for n in grid.index]
    p.write_text(json.dumps({"team": team, "saved_at": datetime.now().isoformat(timespec="seconds"), "note": note,
                             "players": names, "grid": [[int(x) for x in row] for row in grid[list(range(MINUTES))].values],
                             "locked": names if locked is None else [str(n) for n in locked if str(n) in names]}, indent=0))
    return p


def load_drawn(team: str) -> Optional[pd.DataFrame]:
    """The whole saved grid (locked rows and the fill that was saved with them); .attrs["locked"] names the locked rows."""
    p = drawn_path(team)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    g = pd.DataFrame(np.array(d["grid"], dtype=bool).reshape(len(d["players"]), MINUTES), index=d["players"], columns=list(range(MINUTES)))
    g.attrs["locked"] = list(d.get("locked", d["players"]))
    return g


def load_locks(team: str) -> Optional[pd.DataFrame]:
    """Only the locked rows of the saved grid (what the skeleton must hold); None when nothing is locked."""
    g = load_drawn(team)
    if g is None:
        return None
    locked = [n for n in g.attrs.get("locked", list(g.index)) if n in g.index]
    return g.loc[locked] if locked else None


def clear_drawn(team: str) -> bool:
    p = drawn_path(team)
    if p.exists():
        p.unlink(); return True
    return False


def drawn_minutes(team: str) -> Dict[str, float]:
    g = load_drawn(team)
    return minutes_of(g) if g is not None else {}


def format_grid(grid: pd.DataFrame, curve: Optional[pd.DataFrame] = None) -> str:
    L = [f"  {'player':22s} {'min':>4s}  " + "".join("|" if m % 12 == 0 else ("." if m % 6 == 0 else " ") for m in range(MINUTES))]
    for n in grid.index:
        row = "".join("#" if v else " " for v in grid.loc[n].values)
        L.append(f"  {n[:22]:22s} {grid.loc[n].sum():4.0f}  {row}" + (f"  hist {float(curve.loc[n, 'minutes']):.1f}" if curve is not None and n in curve.index else ""))
    L.append(f"  {'on floor':22s}       " + "".join(str(c) if c != ON_COURT else "5" for c in column_counts(grid)))
    return "\n".join(L)


# ----------------------------------------------------------------------------- the stint-level sampler
DEFAULT_BOUNDARY_SD = {"star": 1.6, "starter": 2.2, "rotation": 3.0, "deep": 4.5}      # minutes, by minutes level
MARGIN_SD = 14.75               # final margin around the spread (team_minutes.json has the fitted value)
P_OT = 0.052
JITTER_SCALE = 1.6              # boundary jitter multiplier that puts the 10-90 band at ~80% coverage (January 2026 walk-forward)
# spread multipliers: January coverage by level (star 88%, rotation 74%) and the weaker history classes seen on 2026-10-21
SPREAD_MULT = {"star": 0.85, "starter": 1.0, "rotation": 1.15, "deep": 1.0, "same_team": 1.0, "new_team": 1.3, "no_history": 1.5}


def boundary_spreads(session: Session, players: Iterable[str], before: Optional[date] = None, n_games: int = 12) -> Dict[str, float]:
    """
    player -> the typical spread (sd, minutes) of his stint boundaries: for each
    stint number, the sd across his recent games of that stint's entry and exit
    minute, averaged over the stints he usually has. Players with under four
    games fall back to the level default by minutes.
    """
    names = [str(n) for n in players]
    if not names:
        return {}
    marks = ", ".join(f":n{i}" for i in range(len(names)))
    params = {f"n{i}": n for i, n in enumerate(names)}
    if before:
        params["d"] = before.isoformat()
    q = text(f"select game_id, game_date, player_name, stint_no, start_sec, end_sec, minutes from nba_stint where player_name in ({marks})"
             + (" and game_date < :d" if before else "") + " order by game_date, game_id").bindparams(**params)
    df = pd.read_sql(q, session.get_bind())
    out: Dict[str, float] = {}
    for name, g in df.groupby("player_name"):
        games = list(dict.fromkeys(g.game_id))[-n_games:]
        g = g[g.game_id.isin(games)]
        if len(games) < 4:
            continue
        # the spread per boundary that reproduces his game-to-game minutes sd: sd_total^2 = 2 x stints x sd_boundary^2
        per_game = g.groupby("game_id").agg(minutes=("minutes", "sum"), stints=("stint_no", "max"))
        sd_total = float(per_game.minutes.std(ddof=1)); n_st = float(per_game.stints.mean())
        if np.isfinite(sd_total) and n_st > 0:
            out[name] = float(np.clip(sd_total / np.sqrt(2.0 * n_st), 0.5, 6.0))
    return out


def default_spread(minutes: float) -> float:
    return DEFAULT_BOUNDARY_SD["star" if minutes >= 28 else "starter" if minutes >= 18 else "rotation" if minutes >= 9 else "deep"]


def sub_events(grid: pd.DataFrame) -> List[dict]:
    """The grid's substitutions: {minute, outs, ins} at every minute where the floor changes."""
    g = grid.values.astype(bool); names = list(grid.index)
    events = []
    for m in range(1, MINUTES):
        outs = [names[i] for i in range(len(names)) if g[i, m - 1] and not g[i, m]]
        ins = [names[i] for i in range(len(names)) if g[i, m] and not g[i, m - 1]]
        if outs or ins:
            events.append(dict(minute=m, outs=outs, ins=ins))
    return events


def garbage_threshold_minutes(minutes_left: float) -> float:
    from ivan.sports.nba.pbp import garbage_threshold
    return garbage_threshold(minutes_left * 60.0)


def sample_worlds(grid: pd.DataFrame, spreads: Dict[str, float], n: int, rng: np.random.Generator, spread: Optional[float] = None,
                  p_out: Optional[Dict[str, float]] = None, margin: Optional[np.ndarray] = None, overtime: Optional[np.ndarray] = None,
                  margin_sd: float = MARGIN_SD, p_ot: float = P_OT, jitter_scale: Optional[float] = None, curve: Optional[pd.DataFrame] = None) -> dict:
    """
    n worlds of minutes from a rotation grid. Every substitution's time is jittered
    once per world (so the floor stays at five), a scratched player's stints go to
    the best substitute off the floor at those minutes, garbage time (the world's
    margin path crossing the threshold for the time left) sends the closers off for
    good and the garbage minutes to the end of the bench, and overtime adds five
    minutes to whoever is on the floor at the buzzer.
    Returns {"minutes": (n, k) in the grid's row order, "margin", "overtime", "garbage_at": (n,) minute or -1}.
    """
    jitter_scale = JITTER_SCALE if jitter_scale is None else float(jitter_scale)
    names = list(grid.index); k = len(names); idx = {nm: i for i, nm in enumerate(names)}
    G = grid.values.astype(bool)
    base = G.sum(axis=1).astype(float)
    if margin is None:
        margin = rng.normal(-(spread or 0.0), margin_sd, n)
    if overtime is None:
        overtime = (rng.random(n) < p_ot).astype(int)
    events = sub_events(grid)
    # each boundary is an event; a player's stint = (entry event or tip) .. (exit event or buzzer)
    bounds: List[Tuple[int, Optional[int], Optional[int]]] = []     # (player, entry event index or None, exit event index or None)
    entry_ev: Dict[int, Optional[int]] = {}
    for i in range(k):
        if G[i, 0]:
            entry_ev[i] = None
    for e_i, e in enumerate(events):
        for nm in e["outs"]:
            i = idx[nm]
            if i in entry_ev:
                bounds.append((i, entry_ev.pop(i), e_i))
        for nm in e["ins"]:
            entry_ev[idx[nm]] = e_i
    for i, e_i in entry_ev.items():
        bounds.append((i, e_i, None))
    sd_ev = np.array([jitter_scale * np.mean([spreads.get(nm, default_spread(base[idx[nm]])) for nm in e["outs"] + e["ins"]] or [2.0]) for e in events])
    t_ev = np.array([e["minute"] for e in events], dtype=float)
    T = t_ev[None, :] + rng.standard_normal((n, len(events))) * sd_ev[None, :] if events else np.zeros((n, 0))
    T = np.clip(T, 0.5, MINUTES - 0.5)
    minutes = np.zeros((n, k))
    # each player's boundaries in his own order (a jitter cannot put an exit before its entry)
    per_player: Dict[int, List[Tuple[Optional[int], Optional[int]]]] = {}
    for i, a, b in bounds:
        per_player.setdefault(i, []).append((a, b))
    for i, stints in per_player.items():
        cols = []
        for a, b in stints:
            cols.append(np.zeros(n) if a is None else T[:, a]); cols.append(np.full(n, float(MINUTES)) if b is None else T[:, b])
        B = np.sort(np.column_stack(cols), axis=1)
        minutes[:, i] = (B[:, 1::2] - B[:, 0::2]).sum(axis=1)
    # the sort can move a minute between players; every world goes back to 240 (overtime is added later)
    tot = minutes.sum(axis=1, keepdims=True)
    minutes = minutes * (ON_COURT * MINUTES) / np.maximum(tot, 1e-9)
    # scratches: a world where he is out hands each of his stints to the best substitute off the floor then
    avail = np.ones((n, k), dtype=bool)
    if p_out:
        for nm, p in p_out.items():
            if nm not in idx or not p:
                continue
            i = idx[nm]; outw = rng.random(n) < float(p)
            if not outw.any():
                continue
            avail[outw, i] = False
            lost = minutes[outw, i].copy(); minutes[outw, i] = 0.0
            for a, b in typical_stints(G[i].astype(float)):
                share = (b - a) / max(1.0, base[i])
                off = [j for j in range(k) if j != i and not G[j, a:b].any()]
                if not off:
                    off = [j for j in range(k) if j != i]            # a thin roster: whoever has the fewest minutes
                    pick = min(off, key=lambda j: base[j]) if off else None
                else:
                    pick = max(off, key=lambda j: (float(curve.loc[names[j], list(range(a, b))].mean()) if curve is not None and names[j] in curve.index else base[j] / MINUTES))
                if pick is None:
                    continue
                minutes[outw, pick] += lost * share
    # garbage time: the margin path is a bridge to the final margin; once it crosses the threshold for the time left,
    # the closers (starters / the five with the most minutes) are done and the end of the bench plays out the game
    closers = set(np.argsort(-base)[:5].tolist())
    garbage_at = np.full(n, -1)
    ms = np.arange(GARBAGE_FROM, MINUTES)
    frac = ms / MINUTES
    path_sd = margin_sd * np.sqrt(frac * (1 - frac))
    paths = margin[:, None] * frac[None, :] + rng.standard_normal((n, len(ms))) * path_sd[None, :]
    thr = np.array([garbage_threshold_minutes(MINUTES - m) for m in ms])
    hit = np.abs(paths) >= thr[None, :]
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    bench = [j for j in np.argsort(base) if j not in closers][:5]
    for w in np.flatnonzero(first >= 0):
        m0 = int(ms[first[w]]); garbage_at[w] = m0
        freed = 0.0
        for i in closers:
            if not avail[w, i]:
                continue
            on_after = float(G[i, m0:].sum())
            take = min(on_after, minutes[w, i]); minutes[w, i] -= take; freed += take
        takers = [j for j in bench if avail[w, j]]
        if takers and freed > 0:
            for j in takers:
                minutes[w, j] += freed / len(takers)
    # overtime: five extra minutes for whoever closes
    closers_at_end = [i for i in range(k) if G[i, MINUTES - 1]]
    if closers_at_end:
        for w in np.flatnonzero(overtime > 0):
            if garbage_at[w] < 0:
                for i in closers_at_end:
                    if avail[w, i]:
                        minutes[w, i] += 5.0 * overtime[w]
    minutes = np.clip(minutes, 0.0, 53.0)
    return {"minutes": minutes, "margin": margin, "overtime": overtime, "garbage_at": garbage_at, "names": names}


def evaluate_rotation(session: Session, start: date, end: date, n: int = 300, seed: int = 0, spreads_by_game: Optional[Dict[tuple, float]] = None,
                      jitter_scale: Optional[float] = None, alloc: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Walk-forward: for every team-game in the window, propose the rotation from the
    players' stints before that date (oracle availability and starters), draw n
    worlds, and compare each player's draw (mean, p10, p90) to his actual minutes.
    With `alloc` (team_minutes.allocated_frame: the real allocator's minutes per
    player-game) each player's worlds are scaled to the allocation, which is the
    live configuration (the allocator's mean, the grid's shape).
    """
    alloc_map = {}
    if alloc is not None:
        alloc_map = {(str(r.game_id), str(r.team), str(r.player_name)): float(r.alloc) for r in alloc.itertuples()}
    rng = np.random.default_rng(seed)
    box = pd.read_sql(text("select game_id, game_date, player_name, team_abbreviation team, minutes, position from nba_stats "
                           "where substr(game_id,1,3) in ('002','004') and game_date >= :a and game_date <= :b").bindparams(a=start.isoformat(), b=end.isoformat()), session.get_bind())
    box["game_date"] = pd.to_datetime(box.game_date).dt.date
    rows = []
    for (gid, team, day), g in box.groupby(["game_id", "team", "game_date"]):
        played = g[g.minutes > 0]
        if len(played) < 8:
            continue
        names = [normalize_name(n) for n in played.player_name]
        starters = [normalize_name(r.player_name) for r in played.itertuples() if str(r.position).upper() != "BENCH"]
        curve = curves(session, players=names, before=day, n_games=10)
        if curve.empty:
            continue
        grid = propose(curve, names, starters)
        spreads = boundary_spreads(session, names, before=day)
        sp = (spreads_by_game or {}).get((day, team), 0.0)
        d = sample_worlds(grid, spreads, n, rng, spread=sp, jitter_scale=jitter_scale, curve=curve)
        if alloc_map:
            for i, nm in enumerate(d["names"]):
                a = alloc_map.get((str(gid), str(team), nm)); mu = float(d["minutes"][:, i].mean())
                if a is not None and mu > 0:
                    d["minutes"][:, i] *= a / mu
        q10, q50, q90 = np.percentile(d["minutes"], [10, 50, 90], axis=0); mean = d["minutes"].mean(axis=0)
        act = dict(zip(names, played.minutes.astype(float)))
        for i, nm in enumerate(d["names"]):
            prop = float(grid.loc[nm].sum())
            rows.append(dict(game_id=gid, team=team, date=day, player=nm, proposed=prop, mean=float(mean[i]), q10=float(q10[i]), q90=float(q90[i]),
                             actual=act.get(nm, 0.0), hist=float(curve.loc[nm, "minutes"]) if nm in curve.index else np.nan,
                             starter=nm in starters, role="star" if prop >= 28 else "starter" if prop >= 18 else "rotation" if prop >= 9 else "deep",
                             garbage_share=float((d["garbage_at"] >= 0).mean())))
    return pd.DataFrame(rows)


def coverage_by_role(ev: pd.DataFrame) -> pd.DataFrame:
    t = []
    for r in ("star", "starter", "rotation", "deep", "all"):
        g = ev if r == "all" else ev[ev.role == r]
        if not len(g):
            continue
        t.append(dict(role=r, n=len(g), mae_proposed=(g.proposed - g.actual).abs().mean(), mae_hist=(g["hist"] - g.actual).abs().mean(),
                      bias=(g["mean"] - g.actual).mean(), cover=((g.actual >= g.q10) & (g.actual <= g.q90)).mean(), width=(g.q90 - g.q10).mean()))
    return pd.DataFrame(t).set_index("role")
