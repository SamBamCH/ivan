# src/ivan/sports/nba/gamesim/rotation.py
"""
Layer B: the rotation under game events. Who is on the floor, minute by minute, in every world.

The skeleton grid (projection/skeleton.py: the coach's slots filled with tonight's players) is the plan. A world jitters
every substitution's time once (so the floor stays at five), then the game happens to the plan:

  scratches     a world where a player is out (p_out) hands his minutes to his heirs: the owed minutes flow across position
                groups by the measured mid-game matrix (fouls.DEFAULT_FLOW: a big's go 40% to bigs, 30% to guards, 8% to
                wings, a fifth to nobody in particular) and inside a group to the players with room in those minutes
                (docs/nba/notes/midgame_exits_2026-11-07.md: room, not curve height, decides who can take minutes);
  fouls         each player fouls at his own rate per minute on the floor (FoulRates: trailing, shrunk toward the level
                prior; a player's rate is his, split-half r = 0.66); the coach's reaction is the measured share of the
                remaining time a player keeps after his fourth and fifth foul (SIT: with 12+ minutes left a fourth foul costs
                about a quarter of it, a fifth about half; under four minutes nothing), the sixth ends his night and his
                heirs take the rest the same way;
  the path      the game's margin path (layer A's, or the real one in the check) decides garbage time: the first minute from
                GARBAGE_FROM where the lead beats the threshold for the time left sends the closers (the five with the most
                planned minutes) off for good and the end of the bench on;
  overtime      the five on the floor at the buzzer play each extra period.

draw(grid, spreads, n, rng, margin_path, overtime, ...) returns the on-floor coverage per world, player and minute
(n, k, 48: the fraction of each minute on the floor, five per minute), the minutes (with overtime), the foul counts and
foul-outs, garbage onset. Layers C and D read the coverage: the five on the floor at each minute share that minute's
possessions.

check(session, start, end, n): the layer with the truth plugged in above it. For every team-game of the window the
grid is proposed from the stints before that date (projection/rotation.propose, the shape from history), the margin path
is the real one (pbp.game_score) and the draw's minutes are scored against the box score: MAE, bias and 10-90 coverage
by role, by final-margin bucket, and for the players who really reached five fouls; next to the current sampler
(rotation.sample_worlds) on the same grids with the same final margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.sports.nba.gamesim.fouls import DEFAULT_FLOW
from ivan.sports.nba.data_fetch.pbp import garbage_threshold
from ivan.sports.nba.projection.rotation import JITTER_SCALE, default_spread, sub_events

MINUTES = 48
ON_COURT = 5
GARBAGE_FROM = 34
REENTRY = 0.6                 # the closers stay off only while the lead holds this share of the pull threshold for the time left
OT_MIN = 5.0
FOUL_HALFLIFE = 20.0          # games
FOUL_PRIOR_MINUTES = 300.0    # pseudo-minutes toward the level prior
LEVEL_FOULS_36 = [(30.0, 2.6), (20.0, 3.15), (10.0, 3.9), (0.0, 4.1)]   # (planned minutes at least, fouls per 36): measured 2026-11-07
# the share of the remaining regulation time a player keeps after his k-th foul, by minutes left (measured 2026-11-07 against the
# share a player on the floor at that minute normally plays: 4th foul 0.40-0.43 vs ~0.57 with 12+ left, 5th 0.22-0.26 vs 0.57;
# under eight minutes a fourth foul changes little, a fifth costs a third; under four, nothing)
SIT = {4: [(12.0, 0.72), (8.0, 0.90), (0.0, 1.0)], 5: [(12.0, 0.42), (8.0, 0.60), (4.0, 0.75), (0.0, 1.0)]}


def sit_share(k: int, minutes_left: float) -> float:
    for lo, s in SIT.get(k, []):
        if minutes_left >= lo:
            return s
    return 1.0


def level_rate(planned_minutes: float) -> float:
    """Fouls per minute on the floor for a player of this level."""
    for lo, r36 in LEVEL_FOULS_36:
        if planned_minutes >= lo:
            return r36 / 36.0
    return LEVEL_FOULS_36[-1][1] / 36.0


# ----------------------------------------------------------------------------- foul rates
@dataclass
class FoulRates:
    """Trailing fouls and minutes per player after each game (EWM, half-life FOUL_HALFLIFE games)."""
    rows: pd.DataFrame      # player, game_date, game_id, f_num, m_den

    @classmethod
    def load(cls, session: Session) -> "FoulRates":
        f = pd.read_sql(text("select game_id, player_name, count(*) pf from nba_foul where personal = 1 group by game_id, player_name"), session.get_bind())
        m = pd.read_sql(text("select game_id, min(game_date) game_date, player_name, sum(minutes) minutes from nba_stint group by game_id, player_name"), session.get_bind())
        d = m.merge(f, on=["game_id", "player_name"], how="left").fillna({"pf": 0.0})
        d["game_date"] = pd.to_datetime(d.game_date).dt.date
        a = 0.5 ** (1.0 / FOUL_HALFLIFE)
        out = []
        for name, g in d.sort_values(["game_date", "game_id"]).groupby("player_name", sort=False):
            fn = mn = 0.0
            for r in g.itertuples(index=False):
                fn = a * fn + float(r.pf); mn = a * mn + float(r.minutes)
                out.append(dict(player=name, game_date=r.game_date, game_id=r.game_id, f_num=fn, m_den=mn))
        return cls(pd.DataFrame(out, columns=["player", "game_date", "game_id", "f_num", "m_den"]))

    def before(self, d: date) -> pd.DataFrame:
        sub = self.rows[self.rows.game_date < d]
        return sub.groupby("player").tail(1).set_index("player")

    def rate(self, state: pd.DataFrame, player: str, planned_minutes: float) -> float:
        prior = level_rate(planned_minutes)
        if player not in state.index:
            return prior
        r = state.loc[player]
        return (float(r.f_num) + FOUL_PRIOR_MINUTES * prior) / (float(r.m_den) + FOUL_PRIOR_MINUTES)


# ----------------------------------------------------------------------------- the draw
def _raster(entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """(n,) stint entry and exit minutes -> (n, 48) fraction of each minute on the floor."""
    m = np.arange(MINUTES, dtype=float)
    return np.clip(np.minimum(exit_[:, None], m[None, :] + 1.0) - np.maximum(entry[:, None], m[None, :]), 0.0, 1.0)


def _hand_over(F: np.ndarray, w: np.ndarray, i: int, freed: np.ndarray, from_minute: np.ndarray, groups: Sequence[str], flow: Dict[str, Dict[str, float]],
               avail: np.ndarray) -> None:
    """
    Worlds `w`: player i's freed coverage (len(w), 48) from each world's `from_minute` goes to his heirs: across groups by the
    flow matrix, inside a group by the room each available player has in that minute. What no heir can take is left for
    the per-minute renormalisation (the rotation plays short: everyone else a little longer).
    """
    k = F.shape[1]
    g_out = groups[i]
    for hg, share in flow.get(g_out, {}).items():
        if share <= 0:
            continue
        members = [j for j in range(k) if j != i and groups[j] == hg]
        if not members:
            continue
        room = np.clip(1.0 - F[w][:, members, :], 0.0, 1.0) * avail[w][:, members, None]            # (len(w), |members|, 48)
        tot = room.sum(axis=1, keepdims=True)
        take = np.where(tot > 0, room / np.maximum(tot, 1e-9), 0.0) * (freed * share)[:, None, :]
        F[np.ix_(w, members, np.arange(MINUTES))] += take


def _renormalise(F: np.ndarray, weights: Optional[np.ndarray] = None, passes: int = 8) -> None:
    """
    Five on the floor in every minute of every world. Water-filling: the players on the floor in a minute are scaled toward
    the five, nobody past a whole minute per pass, until the minute holds five; a minute with too few players holding any
    coverage brings in the others by `weights` (n, k) (their planned minutes among the available), then fills again.
    """
    for _ in range(passes):
        tot = F.sum(axis=1, keepdims=True)
        if np.allclose(tot, ON_COURT, atol=1e-6):
            return
        full = F >= 1.0 - 1e-9
        part = F * ~full
        room_tot = part.sum(axis=1, keepdims=True)
        need = ON_COURT - full.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            top = np.where(part > 0, part, 0.0).max(axis=1, keepdims=True)
            scale = np.where((room_tot > 1e-9) & (need > 0), need / np.maximum(room_tot, 1e-9), 1.0)
            scale = np.minimum(scale, np.where(top > 0, 1.0 / np.maximum(top, 1e-9), 1.0))        # nobody past a whole minute this pass
        scale = np.where(need > 0, scale, 1.0)
        F[...] = np.where(full, F, F * scale)
        np.clip(F, 0.0, 1.0, out=F)
        # over five (a hand-over landed on a full minute): scale everyone down
        tot = F.sum(axis=1, keepdims=True)
        over = tot > ON_COURT + 1e-9
        if over.any():
            F[...] = np.where(over, F * ON_COURT / np.maximum(tot, 1e-9), F)
        # under five with nobody left to scale: bring in the players at zero by weight
        tot = F.sum(axis=1, keepdims=True)
        short = (tot < ON_COURT - 1e-6) & ((F > 0) & (F < 1.0 - 1e-9)).sum(axis=1, keepdims=True).__eq__(0)
        if short.any() and weights is not None:
            zero = (F <= 0) & short
            wts = np.where(zero, weights[:, :, None], 0.0)
            wsum = wts.sum(axis=1, keepdims=True)
            F[...] += np.where(wsum > 0, wts / np.maximum(wsum, 1e-9) * (ON_COURT - tot), 0.0)
            np.clip(F, 0.0, 1.0, out=F)


def draw(grid: pd.DataFrame, spreads: Dict[str, float], n: int, rng: np.random.Generator, margin_path: np.ndarray, overtime: np.ndarray,
         *, p_out: Optional[Dict[str, float]] = None, foul_rate: Optional[Dict[str, float]] = None, groups: Optional[Dict[str, str]] = None,
         flow: Optional[Dict[str, Dict[str, float]]] = None, jitter_scale: Optional[float] = None, garbage: bool = True) -> dict:
    """
    grid: rows players, 48 columns of 0/1 (the plan). spreads: player -> sd of his substitution times (minutes). margin_path:
    (n, 48) the team's lead at the end of each minute in each world. overtime: (n,) extra periods. p_out: scratch probability
    per player. foul_rate: fouls per minute on the floor per player (None: no fouls drawn). groups: player -> big | guard | wing.
    Returns {"onfloor": (n, k, 48), "minutes": (n, k), "ot_minutes": (n, k), "garbage_at": (n,), "fouls": (n, k), "fouled_out": (n, k), "names"}.
    """
    jitter_scale = JITTER_SCALE if jitter_scale is None else float(jitter_scale)
    flow = flow or DEFAULT_FLOW
    names = list(grid.index); k = len(names); idx = {nm: i for i, nm in enumerate(names)}
    G = grid.values.astype(bool); base = G.sum(axis=1).astype(float)
    grp = [(groups or {}).get(nm, "wing") for nm in names]
    margin_path = np.asarray(margin_path, dtype=float).reshape(n, MINUTES)
    overtime = np.asarray(overtime).reshape(n)
    # the plan's substitutions, jittered once per world
    events = sub_events(grid)
    t_ev = np.array([e["minute"] for e in events], dtype=float)
    sd_ev = np.array([jitter_scale * np.mean([spreads.get(nm, default_spread(base[idx[nm]])) for nm in e["outs"] + e["ins"]] or [2.0]) for e in events])
    T = np.clip(t_ev[None, :] + rng.standard_normal((n, len(events))) * sd_ev[None, :], 0.5, MINUTES - 0.5) if events else np.zeros((n, 0))
    # each player's stints as (entry event, exit event), rasterised to minute coverage
    F = np.zeros((n, k, MINUTES))
    entry_ev: Dict[int, Optional[int]] = {i: None for i in range(k) if G[i, 0]}
    stints: List[Tuple[int, Optional[int], Optional[int]]] = []
    for e_i, e in enumerate(events):
        for nm in e["outs"]:
            i = idx[nm]
            if i in entry_ev:
                stints.append((i, entry_ev.pop(i), e_i))
        for nm in e["ins"]:
            entry_ev[idx[nm]] = e_i
    for i, e_i in entry_ev.items():
        stints.append((i, e_i, None))
    per_player: Dict[int, List[Tuple[Optional[int], Optional[int]]]] = {}
    for i, a, b in stints:
        per_player.setdefault(i, []).append((a, b))
    for i, st in per_player.items():
        cols = []
        for a, b in st:
            cols.append(np.zeros(n) if a is None else T[:, a]); cols.append(np.full(n, float(MINUTES)) if b is None else T[:, b])
        B = np.sort(np.column_stack(cols), axis=1)            # a jitter cannot put an exit before its entry
        for s in range(0, B.shape[1], 2):
            F[:, i, :] += _raster(B[:, s], B[:, s + 1])
    np.clip(F, 0.0, 1.0, out=F)
    avail = np.ones((n, k), dtype=bool)
    _renormalise(F, base[None, :] * avail)
    all_minutes = np.zeros(n, dtype=int)
    # scratches
    for nm, p in (p_out or {}).items():
        if nm not in idx or not p:
            continue
        i = idx[nm]; outw = np.flatnonzero(rng.random(n) < float(p))
        if not len(outw):
            continue
        avail[outw, i] = False
        freed = F[outw, i, :].copy(); F[outw, i, :] = 0.0
        _hand_over(F, outw, i, freed, all_minutes[outw], grp, flow, avail)
    _renormalise(F, base[None, :] * avail)
    # fouls as a process along the on-floor minutes: exponential gaps at the player's rate (the hazard is flat in the foul
    # count, measured 2026-11-07: 2.9-3.3 per 36 holding 0..5 fouls); each sit after the 4th and 5th shrinks the exposure the
    # next foul can land in, which is what keeps real foul-outs rare (1.1% of player-games) when a flat Poisson count says 3%
    fouls = np.zeros((n, k), dtype=int); fouled_out = np.zeros((n, k), dtype=bool)
    if foul_rate:
        for i, nm in enumerate(names):
            h = float(foul_rate.get(nm, level_rate(base[i])))
            if h <= 0:
                continue
            gaps = rng.exponential(1.0 / h, size=(n, 6))
            arrivals = np.cumsum(gaps, axis=1)                     # on-floor minutes at which fouls 1..6 would come, before any sit
            C = np.cumsum(F[:, i, :], axis=1)                      # (n, 48)
            total = C[:, -1]
            fouls[:, i] = (arrivals[:, :4] <= total[:, None]).sum(axis=1)           # up to four fouls need no sit
            trouble = np.flatnonzero((arrivals[:, 3] <= total) & avail[:, i])
            for w in trouble:
                row = C[w]; t = arrivals[w, 3]
                for kth in (4, 5, 6):
                    m = min(int(np.searchsorted(row, t, side="left")), MINUTES - 1)
                    if kth == 6:
                        freed = np.zeros((1, MINUTES)); freed[0, m:] = F[w, i, m:]; F[w, i, m:] = 0.0
                        fouled_out[w, i] = True; avail[w, i] = False; fouls[w, i] = 6
                        _hand_over(F, np.array([w]), i, freed, np.array([m]), grp, flow, avail)
                        break
                    s = sit_share(kth, float(MINUTES - m))
                    if s < 1.0:
                        freed = np.zeros((1, MINUTES)); freed[0, m:] = F[w, i, m:] * (1.0 - s); F[w, i, m:] *= s
                        _hand_over(F, np.array([w]), i, freed, np.array([m]), grp, flow, avail)
                        row = np.cumsum(F[w, i, :])                                  # the exposure after the sit
                    t = t + gaps[w, kth]                                              # the next foul's on-floor minute
                    if t > row[-1]:
                        break
                    fouls[w, i] = kth + 1
        _renormalise(F, base[None, :] * avail)
    # garbage time off the path: the closers are done once the lead beats the threshold for the time left AND stays beyond
    # REENTRY of it to the buzzer (a lead that comes back brings the closers back: a real path crosses and returns, and so
    # does a drawn one; the old bridge to the final never did), the end of the bench plays it out
    garbage_at = np.full(n, -1)
    ms = np.arange(GARBAGE_FROM, MINUTES)
    if garbage and len(ms):
        thr = np.array([garbage_threshold((MINUTES - m) * 60.0) for m in ms])
        lead = np.abs(margin_path[:, ms])
        stays = np.flip(np.cumprod(np.flip(lead >= REENTRY * thr[None, :], axis=1), axis=1), axis=1).astype(bool)   # beyond the re-entry line from here to the end
        hit = (lead >= thr[None, :]) & stays
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        closers = np.argsort(-base)[:ON_COURT].tolist()
        bench = [j for j in np.argsort(base) if j not in closers][:ON_COURT]
        for w in np.flatnonzero(first >= 0):
            m0 = int(ms[first[w]]); garbage_at[w] = m0
            takers = [j for j in bench if avail[w, j]]
            if not takers:
                continue
            room = np.clip(1.0 - F[w, takers, m0:], 0.0, 1.0); tot = room.sum(axis=0)
            on_closers = F[w, closers, m0:]
            # the coach pulls as many closers as he has bodies for: the pull per minute is capped by the bench's room
            pull = np.minimum(on_closers.sum(axis=0), tot)
            share = np.where(on_closers.sum(axis=0) > 0, pull / np.maximum(on_closers.sum(axis=0), 1e-9), 0.0)
            F[w][np.ix_(closers, np.arange(m0, MINUTES))] = on_closers * (1.0 - share[None, :])
            F[w][np.ix_(takers, np.arange(m0, MINUTES))] += np.where(tot > 0, room / np.maximum(tot, 1e-9), 0.0) * pull[None, :]
        _renormalise(F, base[None, :] * avail)
    # overtime for whoever is on at the buzzer (not after a garbage-time pull: the closers would come back; those worlds are rare and ignored)
    ot_minutes = np.zeros((n, k))
    ot_w = np.flatnonzero((overtime > 0) & (garbage_at < 0))
    if len(ot_w):
        last = F[ot_w, :, MINUTES - 1]
        ot_minutes[ot_w] = last * (OT_MIN * overtime[ot_w])[:, None]
    minutes = F.sum(axis=2) + ot_minutes
    return dict(onfloor=F, minutes=minutes, ot_minutes=ot_minutes, garbage_at=garbage_at, fouls=fouls, fouled_out=fouled_out, names=names)


# ----------------------------------------------------------------------------- the check
def real_margin_path(session: Session, game_id: str, team: str) -> Optional[Tuple[np.ndarray, int]]:
    """(the team's lead at the end of each of the 48 minutes, overtime periods) from the stored feed; None without one."""
    from ivan.sports.nba.data_fetch import pbp as P
    sc = P.game_score(str(game_id))
    if not sc or team not in sc:
        return None
    opp = next((t for t in sc if t not in (team, "home", "final", "overtime")), None)
    if opp is None:
        return None
    path = np.asarray(sc[team], dtype=float) - np.asarray(sc[opp], dtype=float)
    raw = P.load_raw(str(game_id))
    periods = int(raw.period.max()) if raw is not None and len(raw) else 4
    return path, max(0, periods - 4)


def check(session: Session, start: date, end: date, n: int = 300, seed: int = 0, groups: Optional[Dict[str, str]] = None, jitter_scale: Optional[float] = None,
          log=print) -> dict:
    """
    Walk-forward over the team-games of the window with the real margin path and the real overtime plugged in: the new draw
    (fouls, heirs by flow, the path's garbage time) and the current sampler (its own bridge to the same final margin) on the
    same proposed grid, both against the box score. Returns {"rows", "by_role", "by_margin", "five_fouls", "summary"}.
    """
    from ivan.sports.nba.names import normalize_name
    from ivan.sports.nba.projection.rotation import boundary_spreads, curves, propose, sample_worlds
    rng = np.random.default_rng(seed)
    box = pd.read_sql(text("select game_id, game_date, player_name, team_abbreviation team, minutes, position from nba_stats "
                           "where substr(game_id,1,3) in ('002','004') and game_date >= :a and game_date <= :b").bindparams(a=start.isoformat(), b=end.isoformat()), session.get_bind())
    box["game_date"] = pd.to_datetime(box.game_date).dt.date
    fouls = pd.read_sql(text("select game_id, player_name, max(foul_no) fouls from nba_foul where personal = 1 group by game_id, player_name"), session.get_bind())
    real_fouls = {(r.game_id, r.player_name): int(r.fouls) for r in fouls.itertuples(index=False)}
    fr = FoulRates.load(session)
    rows = []
    done = 0
    for (gid, team, day), g in box.groupby(["game_id", "team", "game_date"]):
        played = g[g.minutes > 0]
        if len(played) < 8:
            continue
        rp = real_margin_path(session, gid, team)
        if rp is None:
            continue
        path, ot = rp
        names = [normalize_name(x) for x in played.player_name]
        starters = [normalize_name(r.player_name) for r in played.itertuples() if str(r.position).upper() != "BENCH"]
        curve = curves(session, players=names, before=day, n_games=10)
        if curve.empty:
            continue
        grid = propose(curve, names, starters)
        spreads = boundary_spreads(session, names, before=day)
        state = fr.before(day)
        rates = {nm: fr.rate(state, nm, float(grid.loc[nm].sum())) for nm in grid.index}
        d_new = draw(grid, spreads, n, rng, np.repeat(path[None, :], n, axis=0), np.full(n, ot), foul_rate=rates, groups=groups, jitter_scale=jitter_scale)
        d_old = sample_worlds(grid, spreads, n, rng, margin=np.full(n, float(path[-1])), overtime=np.full(n, ot), curve=curve)
        act = dict(zip(names, played.minutes.astype(float)))
        for i, nm in enumerate(d_new["names"]):
            j = d_old["names"].index(nm)
            prop = float(grid.loc[nm].sum())
            mn, mo = d_new["minutes"][:, i], d_old["minutes"][:, j]
            rows.append(dict(game_id=gid, team=team, date=day, player=nm, proposed=prop, actual=act.get(nm, 0.0), final_margin=float(path[-1]),
                             real_fouls=real_fouls.get((gid, nm), 0), starter=nm in starters,
                             role="star" if prop >= 28 else "starter" if prop >= 18 else "rotation" if prop >= 9 else "deep",
                             new_mean=float(mn.mean()), new_q10=float(np.percentile(mn, 10)), new_q90=float(np.percentile(mn, 90)),
                             old_mean=float(mo.mean()), old_q10=float(np.percentile(mo, 10)), old_q90=float(np.percentile(mo, 90)),
                             new_pfo=float(d_new["fouled_out"][:, i].mean()), new_fouls=float(d_new["fouls"][:, i].mean())))
        done += 1
        if done % 50 == 0:
            log(f"  {done} team-games")
    ev = pd.DataFrame(rows)
    if ev.empty:
        return {"rows": ev, "by_role": ev, "by_margin": ev, "five_fouls": ev, "summary": dict(team_games=0)}

    def table(frame: pd.DataFrame, key: str, cuts: Iterable[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        t = []
        for label, d in cuts:
            if not len(d):
                continue
            rec = {key: label, "n": int(len(d))}
            for tag in ("new", "old"):
                rec[f"mae_{tag}"] = float((d[f"{tag}_mean"] - d.actual).abs().mean()); rec[f"bias_{tag}"] = float((d[f"{tag}_mean"] - d.actual).mean())
                rec[f"cover_{tag}"] = float(((d.actual >= d[f"{tag}_q10"]) & (d.actual <= d[f"{tag}_q90"])).mean()); rec[f"width_{tag}"] = float((d[f"{tag}_q90"] - d[f"{tag}_q10"]).mean())
            t.append(rec)
        return pd.DataFrame(t)

    by_role = table(ev, "role", [(r, ev if r == "all" else ev[ev.role == r]) for r in ("star", "starter", "rotation", "deep", "all")])
    am = ev.final_margin.abs()
    by_margin = table(ev, "margin", [("close (<6)", ev[am < 6]), ("mid (6-15)", ev[(am >= 6) & (am < 16)]), ("blowout (16+)", ev[am >= 16])])
    five = table(ev, "fouls", [("5+ fouls", ev[ev.real_fouls >= 5]), ("6 fouls", ev[ev.real_fouls >= 6]), ("<=3 fouls", ev[ev.real_fouls <= 3])])
    summary = dict(team_games=int(done), player_games=int(len(ev)), mae_new=float((ev.new_mean - ev.actual).abs().mean()), mae_old=float((ev.old_mean - ev.actual).abs().mean()),
                   cover_new=float(((ev.actual >= ev.new_q10) & (ev.actual <= ev.new_q90)).mean()), cover_old=float(((ev.actual >= ev.old_q10) & (ev.actual <= ev.old_q90)).mean()),
                   fouls_drawn=float(ev.new_fouls.mean()), fouls_real=float(ev.real_fouls.mean()), foul_out_rate_drawn=float(ev.new_pfo.mean()), foul_out_rate_real=float((ev.real_fouls >= 6).mean()))
    return {"rows": ev, "by_role": by_role, "by_margin": by_margin, "five_fouls": five, "summary": summary}
