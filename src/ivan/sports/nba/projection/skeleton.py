# src/ivan/sports/nba/projection/skeleton.py
"""
The coach's rotation as slots, filled with tonight's players.

A team's history (rotation.curves: for every player the share of recent games he was on the floor in each minute) is
the skeleton: each regular's curve is a slot, "this stretch of the game belongs to this role". Tonight the slots are
filled from the solve (projection/solve.py: how many minutes each available player gets, anchored on the books'
implied minutes, history and the desk's edits) and the injury report:

  - an available player with a curve keeps his own slot;
  - an absent owner's slot (out, or gone from the roster) is inherited by the available players the solve says are
    owed minutes beyond their own slot, same position group first (a big's slot goes to bigs: measured 2026-10-23,
    bigs' minutes flow only to bigs, guards ~4:1, wings proportionally);
  - a player with no history takes what he inherits, else a flat prior over the game.

build_grid() then schedules the game minute by minute: five on the floor, each player's chance of being on at minute m
from his (inherited) curve, pulled toward the solve's target by how many minutes he is still owed, stints kept whole by
hysteresis, the named starters at the tip, breathers after long stretches. Locked rows (the desk's painted players)
are fixed and the rest fill around them. A balance pass then moves single boundary minutes between players until each
row is within BALANCE_TOL of its target, so the grid's minutes ARE the solve's minutes: the grid is the projection.

Players the schedule gives no minutes have none (they are off the pool; the desk adds them back by painting a row).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MINUTES = 48
ON_COURT = 5
HYSTERESIS = 0.25          # a player on the floor stays until someone's score beats his by this
DEFICIT = 0.5              # weight of the minutes a player is still owed against his curve
LATE = 0.6                 # weight of "he cannot make his target in the time left" (a shortfall becomes urgent late)
MIN_STINT = 3              # minutes: a player who comes on stays at least this long, one who goes off rests at least this long;
                           # shorter stints are merged away (never in a locked row)
MAX_STINT = 15             # minutes on the floor before a breather is forced
OPENING = 3                # minutes the named starters hold the floor before any swap
Q3_OPEN = 24               # the named starters open the third quarter too and hold it OPENING minutes (1,348 team-games 2025-26: all five on at the
                           # Q3 tip 86% of ordinary games, four or more 99%; the skeleton managed all five 29% of the time before this, 2026-10-26)
CLOSE_FROM = 44            # from here the five with the largest targets close the game (close games: 4.4 of the top five by minutes on at minute 46,
                           # four or more 93%; the named starters only 3.85, so the closing five is the most-played five, not the starting five)
CAP_EXTRA = 2.0            # a player may run this far over his target in the schedule (the balance pass pulls him back)
HARD_CAP = 44.0
BALANCE_TOL = 0.5          # rows end within this of their target (integer grid: a target of 31.4 lands on 31 or 32)
BALANCE_PASSES = 400
RELAXED_TOL = 1.5          # a second pass allows 2-minute stints for rows still this far from their target
SAME_GROUP_WEIGHT = 3.0    # a guard's or wing's vacated slot: same-group candidates weigh this much more than others
BIG_EXCLUSIVE = True       # a big's vacated slot goes to bigs alone when any big is owed minutes
OWED_MIN = 0.5             # minutes owed below which a player does not inherit
HABIT_W = 4.0              # a vacated slot's heirs: weight x (1 + HABIT_W x share of the owner's exits that heir covered)
ALIGN_AFTER_BALANCE = False   # a second alignment after the balance trades totals for lines: backtest fpts MAE 7.756 vs 7.703 without (2026-10-24)
HABIT_BONUS = 0.0          # the schedule: habit as a swap bonus made the lines worse and the backtest no better (2026-10-24); kept at 0
SCALE_LOW, SCALE_HIGH = 0.6, 1.3   # a player's own curve is scaled by target / usual minutes within these bounds
MOVE_SHARE = 0.6           # owed at least this share of a vacated slot (and no bigger a slot of his own): the heir moves into it


def _curve_matrix(curve: Optional[pd.DataFrame], names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """(P: len(names) x 48 on-floor shares, own: expected minutes from the curve; zeros for players without one)."""
    P = np.zeros((len(names), MINUTES)); own = np.zeros(len(names))
    if curve is None or curve.empty:
        return P, own
    cols = [c for c in range(MINUTES) if c in curve.columns]
    for i, n in enumerate(names):
        if n in curve.index:
            P[i, cols] = np.clip(curve.loc[n, cols].values.astype(float), 0.0, 1.0)
            own[i] = float(curve.loc[n, "minutes"]) if "minutes" in curve.columns else P[i].sum()
    return P, own


def inherit(P: np.ndarray, owed: np.ndarray, own: np.ndarray, names: Sequence[str], groups: Dict[str, str], curve: pd.DataFrame,
            vacated: Iterable[str], locked: Sequence[bool], shares: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Fill each vacated slot (largest first) from the available players owed minutes, same position group first (a big's
    slot only to bigs when any big is owed). When one candidate is owed most of the slot and his own slot is no bigger,
    he MOVES INTO it (his curve becomes the slot's) and his own slot is vacated in turn, so a backup big who starts for
    the injured starter plays the starter's stretches and the third big takes the backup's old ones; otherwise the slot
    is shared, weighted by minutes owed. Returns the raised P and {vacated: {heir: share}}.
    """
    P = P.copy(); owed = owed.copy(); habits = shares or {}; shares = {}
    cols = [c for c in range(MINUTES) if c in curve.columns]

    def curve_row(n):
        cv = np.zeros(MINUTES); cv[cols] = np.clip(curve.loc[n, cols].values.astype(float), 0.0, 1.0); return cv

    queue = [(float(curve_row(v).sum()), v, curve_row(v), groups.get(v, "wing")) for v in vacated if v in curve.index]
    queue = [q for q in queue if q[0] > 0]
    queue.sort(key=lambda q: -q[0])
    moved_own: Dict[str, np.ndarray] = {}
    guard = 0
    while queue and guard < 40:
        guard += 1
        mv, v, cv, gv = queue.pop(0)
        cands = [i for i in range(len(names)) if owed[i] >= OWED_MIN and not locked[i]]
        if not cands:
            continue
        same = [i for i in cands if groups.get(names[i], "wing") == gv]
        if gv == "big" and BIG_EXCLUSIVE and same:
            cands = same
        pref = same if same else cands
        top = max(pref, key=lambda i: owed[i])
        if owed[top] >= MOVE_SHARE * mv and own[top] <= mv + 2.0 and names[top] not in moved_own:
            old = P[top].copy()
            P[top] = cv; shares[v] = {names[top]: 1.0}
            owed[top] = max(0.0, owed[top] + own[top] - mv)          # his target against the slot he now holds
            if old.sum() > 0:
                moved_own[names[top]] = old
                queue.append((float(old.sum()), f"{names[top]} (old slot)", old, groups.get(names[top], "wing")))
                queue.sort(key=lambda q: -q[0])
            own[top] = mv
            continue
        hv = habits.get(v.replace(" (old slot)", ""), {})
        w = np.array([owed[i] * (SAME_GROUP_WEIGHT if groups.get(names[i], "wing") == gv else 1.0) * (1.0 + HABIT_W * float(hv.get(names[i], 0.0))) for i in cands])
        w = w / w.sum()
        shares[v] = {}
        for i, wi in zip(cands, w):
            P[i] = np.minimum(1.0, P[i] + wi * cv)
            owed[i] = max(0.0, owed[i] - wi * mv)
            shares[v][names[i]] = float(wi)
    return P, shares


COVER_GROUPS = ("big", "guard")     # the floor keeps at least one of each whenever the roster can (C-eligible on 97% of minutes, PG 93%: measured 2026-10-23)
COVER_MIN_TARGET = 46.0             # ...which it can only when the group's targets add to a whole game (two bigs at 36 + 12; one big alone cannot)


def _covered(on_idx, groups: Sequence[str], want: Sequence[str]) -> bool:
    have = {groups[i] for i in on_idx}
    return all(g in have for g in want)


def cover_groups(groups: Sequence[str], target: np.ndarray) -> list:
    """The lines the schedule must keep on the floor: groups with two or more players whose targets add to COVER_MIN_TARGET."""
    out = []
    for g in COVER_GROUPS:
        idx = [i for i in range(len(groups)) if groups[i] == g and target[i] > 0]
        if len(idx) >= 2 and float(sum(target[i] for i in idx)) >= COVER_MIN_TARGET:
            out.append(g)
    return out


def schedule(P: np.ndarray, target: np.ndarray, starters: Sequence[bool], locked: Optional[np.ndarray] = None,
             is_locked: Optional[np.ndarray] = None, groups: Optional[Sequence[str]] = None, habit: Optional[np.ndarray] = None) -> np.ndarray:
    """
    The minute-by-minute pass: a boolean grid with five on the floor. `locked` (rows x 48) with `is_locked` (rows) fixes
    those rows exactly (a locked row may be all False: a player painted off); the free rows fill the remaining places.
    With `groups` (big / guard / wing per player) the floor keeps a big and a guard on at every minute the roster allows,
    so the centre line and the point guard line stay intact through the rotation.
    """
    k = len(target)
    groups = list(groups) if groups is not None else ["wing"] * k
    want = cover_groups(groups, target)
    habit = np.zeros((k, k)) if habit is None else np.asarray(habit, dtype=float)      # habit[out, in]: share of out's exits that in covered
    grid = np.zeros((k, MINUTES), dtype=bool)
    is_locked = np.zeros(k, dtype=bool) if is_locked is None else np.asarray(is_locked, dtype=bool)
    locked = np.zeros((k, MINUTES), dtype=bool) if locked is None else np.asarray(locked, dtype=bool)
    cap = np.minimum(HARD_CAP, target + CAP_EXTRA)
    used = np.zeros(k); streak = np.zeros(k); rest = np.zeros(k)
    free = [i for i in range(k) if not is_locked[i]]
    on: set = set()
    if any(starters):
        on = {i for i in free if starters[i]}
        if len(on) > ON_COURT:
            on = set(sorted(on, key=lambda i: -P[i, 0])[:ON_COURT])
    for m in range(MINUTES):
        left = MINUTES - m
        need = np.maximum(target - used, 0.0)
        score = P[:, m] + DEFICIT * (need / np.maximum(target, 1.0)) + LATE * np.clip(need - left * 0.8, 0, None) / max(1.0, left)
        score[used >= cap] = -1.0
        score[streak >= MAX_STINT] -= 5.0
        score[target <= 0] = -2.0
        score[(rest > 0) & (rest < MIN_STINT)] -= 5.0                 # just went off: he rests a while
        fresh = (streak > 0) & (streak < MIN_STINT)                  # just came on: he stays a while
        fixed_on = {i for i in range(k) if is_locked[i] and locked[i, m]}
        slots = max(0, ON_COURT - len(fixed_on))
        cur = {i for i in on if i in free}
        if m == 0 and any(starters):
            cur = {i for i in cur if starters[i]}
        elif m < OPENING and any(starters):
            pass                                                   # the opening unit plays the first minutes
        elif m == Q3_OPEN and any(starters):
            # the third quarter opens with the starters again (those with minutes left to give)
            opener = [i for i in free if starters[i] and target[i] > 0 and used[i] < cap[i]]
            cur = set(sorted(opener, key=lambda i: -need[i])[:slots])
        elif Q3_OPEN < m < Q3_OPEN + OPENING and any(starters):
            pass                                                   # ...and holds the floor as at the tip
        elif m == CLOSE_FROM:
            # the closing unit: the five with the largest targets who still have minutes, on until the end
            closers = sorted([i for i in free if target[i] > 0 and used[i] < cap[i]], key=lambda i: -target[i])[:slots]
            cur = set(closers)
        elif m > CLOSE_FROM:
            pass                                                   # the closers stay
        else:
            for _ in range(ON_COURT):
                if not cur:
                    break
                movable = sorted([i for i in cur if not fresh[i]] or list(cur), key=lambda i: score[i])
                swapped = False
                for worst in movable:                                    # the weakest on the floor goes first...
                    # ...for the strongest off it, the coach's usual replacement for him favoured, if the floor keeps its big and guard
                    offs = sorted((i for i in free if i not in cur), key=lambda i: -(score[i] + HABIT_BONUS * habit[worst, i]))
                    for best_off in offs[:4]:
                        if score[best_off] + HABIT_BONUS * habit[worst, best_off] <= score[worst] + HYSTERESIS:
                            break
                        after = (cur | fixed_on) - {worst} | {best_off}
                        if _covered(after, groups, want):
                            cur.remove(worst); cur.add(int(best_off)); swapped = True
                            break
                    if swapped:
                        break
                if not swapped:
                    break
        while len(cur) < slots:
            pool = [int(i) for i in np.argsort(-score) if i in free and i not in cur and target[i] > 0]
            pool = pool or [int(i) for i in np.argsort(-score) if i in free and i not in cur]        # only when nobody with a target is left
            if not pool:
                break
            missing = [g for g in want if not _covered(cur | fixed_on, groups, [g])]
            if missing and len(cur) + 1 >= slots - (len(missing) - 1):                              # the last places go to the missing lines
                grp_pool = [i for i in pool if groups[i] in missing]
                pool = grp_pool or pool
            cur.add(pool[0])
        while len(cur) > slots:
            drop = sorted(cur, key=lambda i: score[i])
            victim = next((i for i in drop if _covered((cur | fixed_on) - {i}, groups, want)), drop[0])
            cur.remove(victim)
        for i in cur | fixed_on:
            grid[i, m] = True; used[i] += 1
        active = np.zeros(k, dtype=bool); active[list(cur | fixed_on)] = True
        streak = np.where(active, streak + 1, 0.0)
        rest = np.where(active, 0.0, rest + 1)
        on = cur
    return grid


def _stints(row: np.ndarray) -> List[Tuple[int, int]]:
    out, start = [], None
    for m, v in enumerate(row):
        if v and start is None:
            start = m
        if not v and start is not None:
            out.append((start, m)); start = None
    if start is not None:
        out.append((start, MINUTES))
    return out


def balance(grid: np.ndarray, target: np.ndarray, free: Sequence[bool], P: np.ndarray, tol: float = BALANCE_TOL, passes: int = BALANCE_PASSES,
            starters: Optional[Sequence[bool]] = None, min_stint: int = MIN_STINT, groups: Optional[Sequence[str]] = None,
            protected: Optional[set] = None) -> np.ndarray:
    """
    Move boundary minutes from players over their target to players under it, one minute at a time, at a minute where
    the giver is on and the taker is off and at least one of them sits at a stint edge there (so stints stretch and
    shrink rather than fragment; a taker extending his own stint is preferred). Five on the floor is preserved: every
    move is a swap within one minute. The opening minutes stay with the named starters.
    """
    g = grid.copy()
    protect_opening = starters is not None and any(starters)
    groups = list(groups) if groups is not None else ["wing"] * len(target)
    want = cover_groups(groups, target)
    for _ in range(passes):
        mins = g.sum(axis=1).astype(float)
        diff = mins - target
        givers = [i for i in np.argsort(-diff) if free[i] and diff[i] > tol]
        takers = [i for i in np.argsort(diff) if free[i] and diff[i] < -tol]
        if not givers or not takers:
            break
        moved = False
        for a in givers:
            for b in takers:
                best, best_s = None, -1e9
                for m in range(OPENING if protect_opening else 0, MINUTES):
                    if protected and m in protected:               # the Q3 opening and the close are the scheduler's
                        continue
                    if not g[a, m] or g[b, m]:
                        continue
                    a_edge = (m == 0 or not g[a, m - 1]) or (m == MINUTES - 1 or not g[a, m + 1])
                    b_edge = (m > 0 and g[b, m - 1]) or (m < MINUTES - 1 and g[b, m + 1])
                    if not (a_edge or b_edge):
                        continue
                    if not _coherent_after(g, a, b, m, min_stint):
                        continue
                    if groups[a] != groups[b] and groups[a] in want and not _covered([i for i in range(len(target)) if g[i, m] and i != a] + [b], groups, [groups[a]]):
                        continue                                          # he was the only big (or guard) on the floor then
                    s = (3.0 if b_edge else 0.0) + (1.0 if a_edge else 0.0) + P[b, m] - P[a, m]
                    if s > best_s:
                        best, best_s = m, s
                if best is not None:
                    g[a, best] = False; g[b, best] = True; moved = True
                    break
            if moved:
                break
        if not moved:
            break
    return g


def _coherent_after(g: np.ndarray, a: int, b: int, m: int, min_stint: int = MIN_STINT) -> bool:
    """Would moving minute m from a to b leave both rows with stints of min_stint or more (an emptied stint is fine)?"""
    ra = g[a].copy(); ra[m] = False
    rb = g[b].copy(); rb[m] = True
    ok_a = all(e - s0 >= min_stint for s0, e in _stints(ra) if e == m or s0 == m + 1)   # what is left of a's stint either side of m
    ok_b = all(e - s0 >= min_stint for s0, e in _stints(rb) if s0 <= m < e)
    return ok_a and ok_b


def align_lines(grid: np.ndarray, groups: Sequence[str], free: Sequence[bool], target: Optional[np.ndarray] = None, min_stint: int = MIN_STINT) -> np.ndarray:
    """
    A same-line swap that the schedule put a minute apart (the backup big enters at minute 8, the starting big leaves at
    9, so for one minute the lanes hand the centre's spot to a wing) is snapped into one minute: the two entrants of the
    adjacent minutes trade entry minutes when that pairs the big with the big. The floor count is unchanged in every
    minute, no line is left uncovered, no stint drops under min_stint, and each player trades at most one minute (the
    balance pass that follows restores the totals).
    """
    g = grid.copy(); k, M = g.shape
    want = cover_groups(list(groups), target) if target is not None else []
    traded = set()
    def ins(m): return [i for i in range(k) if g[i, m] and not g[i, m - 1] and free[i] and i not in traded]
    def outs(m): return [i for i in range(k) if g[i, m - 1] and not g[i, m] and free[i]]
    def stint_len(row, m):
        a = m
        while a > 0 and row[a - 1]:
            a -= 1
        b = m
        while b < M and row[b]:
            b += 1
        return b - a
    def ok_after(m):
        on = [i for i in range(k) if g[i, m]]
        return _covered(on, groups, want)
    def trade(early, late, m):
        """`early` entered at m and now enters at m+1; `late` entered at m+1 and now enters at m."""
        if stint_len(g[early], m) - 1 < min_stint:
            return False
        g[early, m] = False; g[late, m] = True
        if not ok_after(m):
            g[early, m] = True; g[late, m] = False
            return False
        traded.update({early, late})
        return True
    for m in range(1, M - 1):
        done = False
        # an entrant A at m whose line-mate B leaves at m+1 while C (another line) enters at m+1: A and C trade entry minutes
        for a_ in ins(m):
            if any(groups[o] == groups[a_] for o in outs(m)):
                continue                                           # A already answers an exit of his own line
            if not any(groups[b_] == groups[a_] for b_ in outs(m + 1)):
                continue
            for c_ in ins(m + 1):
                if groups[c_] != groups[a_] and trade(a_, c_, m):
                    done = True
                    break
            if done:
                break
        if done:
            continue
        # the mirror: B of a line leaves at m while C (another line) enters; A of B's line enters at m+1: C and A trade entry minutes
        for b_ in outs(m):
            if any(groups[i] == groups[b_] for i in ins(m)):
                continue
            for a_ in ins(m + 1):
                if groups[a_] != groups[b_]:
                    continue
                for c_ in ins(m):
                    if groups[c_] != groups[b_] and trade(c_, a_, m):
                        done = True
                        break
                if done:
                    break
            if done:
                break
    return g


def merge_short_stints(grid: np.ndarray, P: np.ndarray, free: Sequence[bool], groups: Optional[Sequence[str]] = None) -> np.ndarray:
    """Stints shorter than MIN_STINT (in free rows) hand their minutes to the best free player off the floor then, same line first."""
    g = grid.copy(); k = g.shape[0]
    groups = list(groups) if groups is not None else ["wing"] * k
    for i in range(k):
        if not free[i]:
            continue
        for a, b in _stints(g[i]):
            if b - a < MIN_STINT:
                for m in range(a, b):
                    g[i, m] = False
                    off = [j for j in range(k) if free[j] and not g[j, m] and j != i]
                    if off:
                        adjacent = [j for j in off if (m > 0 and g[j, m - 1]) or (m < MINUTES - 1 and g[j, m + 1])]
                        same = [j for j in (adjacent or off) if groups[j] == groups[i]]
                        pool = same or adjacent or off
                        g[max(pool, key=lambda j: P[j, m]), m] = True
    return g


def choose_rotation(targets: Dict[str, float], depth: int, keep: Iterable[str] = ()) -> set:
    """
    Who plays tonight: a coach uses about `depth` players (groups.rotation_depth). Everyone in `keep` (locked rows, hand
    pins, named starters) stays and counts toward the depth; the rest of the places go to the highest targets.
    """
    keep = set(keep)
    order = sorted(targets, key=lambda n: (-float(targets[n]), n))
    chosen = set(n for n in order if n in keep)
    for n in order:
        if len(chosen) >= max(int(depth), len(chosen)):
            break
        chosen.add(n)
    return chosen


def apply_depth(targets: Dict[str, float], depth: int, keep: Iterable[str] = ()) -> Dict[str, float]:
    """
    choose_rotation, then the minutes of the players left out spread over the kept players in proportion to their
    targets (the fallback when the solve cannot be rerun; the engine re-solves the kept players through their anchors).
    """
    keep = set(keep)
    chosen = choose_rotation(targets, depth, keep)
    freed = sum(float(v) for n, v in targets.items() if n not in chosen)
    base = sum(float(targets[n]) for n in chosen if n not in keep)
    out = {}
    for n, v in targets.items():
        if n not in chosen:
            out[n] = 0.0
        elif n in keep or base <= 0:
            out[n] = float(v)
        else:
            out[n] = float(v) * (1.0 + freed / base)
    return out


def build_grid(curve: Optional[pd.DataFrame], targets: Dict[str, float], available: Sequence[str], starters: Optional[Iterable[str]] = None,
               groups: Optional[Dict[str, str]] = None, locks: Optional[pd.DataFrame] = None, vacated: Optional[Iterable[str]] = None,
               shares: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """
    Tonight's grid: index = the available players (targets descending), columns 0..47, exactly five on in every minute
    (locked rows permitting). `targets` are the solve's minutes; `locks` a boolean frame of painted rows (held exactly);
    `vacated` the absent owners whose slots are inherited (default: every curve row not available); `shares` the coach's
    habits (pbp.sub_shares: out -> {in: share of his exits}) which steer who inherits a vacated slot and who comes on for whom.
    The frame carries .attrs["inherited"] = {vacated owner: {heir: share}} and .attrs["deviation"] = {player: grid - target}.
    """
    names = [n for n in available]
    groups = groups or {}
    k = len(names)
    if k == 0:
        return pd.DataFrame(columns=list(range(MINUTES)))
    tgt = np.array([max(0.0, float(targets.get(n, 0.0))) for n in names])
    P, own = _curve_matrix(curve, names)
    locked = np.zeros((k, MINUTES), dtype=bool); is_locked = np.zeros(k, dtype=bool)
    if locks is not None and len(locks):
        for i, n in enumerate(names):
            if n in locks.index:
                locked[i] = locks.loc[n, list(range(MINUTES))].values.astype(bool); is_locked[i] = True
                tgt[i] = float(locked[i].sum())                       # a painted row is its own target
    # the grid holds exactly 240 player-minutes: the free rows' targets scale to what the locked rows leave (the solve's
    # budget is 241 with the overtime share, so this is a 0.4% shave; a hand pin elsewhere makes it larger and visible)
    free_total = ON_COURT * MINUTES - float(tgt[is_locked].sum())
    free_sum = float(tgt[~is_locked].sum())
    if free_sum > 0 and free_total > 0:
        tgt = np.where(is_locked, tgt, tgt * free_total / free_sum)
    owed = np.maximum(tgt - own, 0.0)
    # a player the solve wants well below his usual minutes keeps his strong stretches and loses his weak ones: his
    # whole curve is scaled by target / usual, so the weak stretches fall under the others' in the schedule
    for i in range(k):
        if own[i] > 0 and not is_locked[i]:
            P[i] = np.clip(P[i] * float(np.clip(tgt[i] / own[i], SCALE_LOW, SCALE_HIGH)), 0.0, 1.0)
    if curve is not None and not curve.empty:
        vac = list(vacated) if vacated is not None else [n for n in curve.index if n not in set(names)]
        P, inherited = inherit(P, owed, own.copy(), names, groups, curve, vac, is_locked, shares)
    else:
        inherited = {}
    habit = np.zeros((k, k))
    for o, hv in (shares or {}).items():
        if o in names:
            for i_name, sh in hv.items():
                if i_name in names:
                    habit[names.index(o), names.index(i_name)] = float(sh)
    for i in range(k):
        if P[i].sum() <= 0 and tgt[i] > 0:
            P[i] = min(0.95, tgt[i] / MINUTES)                        # no history, nothing inherited: flat over the game
    st = set(starters or [])
    starter_flags = [n in st for n in names]
    glist = [groups.get(n, "wing") for n in names]
    grid = schedule(P, tgt, starter_flags, locked if is_locked.any() else None, is_locked if is_locked.any() else None, groups=glist, habit=habit)
    free = [not x for x in is_locked]
    grid = merge_short_stints(grid, P, free, groups=glist)
    grid = align_lines(grid, glist, free, tgt)                                                                   # same-line swaps a minute apart snap together...
    prot = (set(range(Q3_OPEN, Q3_OPEN + OPENING)) if st else set()) | set(range(CLOSE_FROM, MINUTES))
    grid = balance(grid, tgt, free, P, starters=starter_flags, groups=glist, protected=prot)                      # ...the balance restores the totals they moved
    grid = balance(grid, tgt, free, P, tol=RELAXED_TOL, starters=starter_flags, min_stint=2, groups=glist, protected=prot)   # a row still far off may make 2-minute stints
    if ALIGN_AFTER_BALANCE:
        grid = align_lines(grid, glist, free, tgt)
    out = pd.DataFrame(grid, index=pd.Index(names, name="player"), columns=list(range(MINUTES)))
    order = sorted(names, key=lambda n: (-float(tgt[names.index(n)]), n))
    out = out.loc[order]
    out.attrs["inherited"] = inherited
    out.attrs["deviation"] = {n: float(out.loc[n].sum() - tgt[names.index(n)]) for n in names}
    out.attrs["locked"] = [n for i, n in enumerate(names) if is_locked[i]]
    return out


def column_counts(grid: pd.DataFrame) -> List[int]:
    return [int(x) for x in grid[list(range(MINUTES))].sum(axis=0).values]


def minutes_of(grid: pd.DataFrame) -> Dict[str, float]:
    return {str(n): float(v) for n, v in grid[list(range(MINUTES))].sum(axis=1).items()}
