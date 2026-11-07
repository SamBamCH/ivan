# src/ivan/sports/nba/gamesim/world.py
"""
The World: every game of the slate simulated minute by minute, n worlds at once.

Inputs per game: the pool rows of both teams (ivan.sports.nba.models.NBAPlayer: minutes_proj, the per-minute rates the
projection stored for points, rebounds, assists, steals, blocks, turnovers and threes, the rotation grid row, the
substitution spread, the scratch probability), the books' spread and total (projection.lines.GameLines), each team's
trailing pace and the league constants (data/nba/models/gamesim/teams.json, written by `gamesim.py fit-script`), each
player's shooting profile (gamesim.profiles) and foul rate (gamesim.rotation.FoulRates), the position groups.

For each game:

  A  the script (gamesim.script.draw): the pace of the night and the possessions of every six-minute segment;
  B  the rotation (gamesim.rotation.draw): each team's on-floor coverage per player per minute, with scratches, fouls and
     the heirs; garbage time is applied here, minute by minute, off the emergent score;
  C  the usage: each minute the team's possessions (the segment's, spread over its six minutes) become two-point attempts,
     three-point attempts, free throws and turnovers at the team's pie (Poisson), split among the five on the floor by
     their per-minute rates times their coverage (multinomial); the pie is the roster's own expected totals per
     possession, with the attempts scaled by one team factor so the expected points are what the lines say the team
     scores (World.reconcile reports the factor);
  D  the outcomes: makes by each player's percentages; the misses become offensive rebounds at the team's share and
     defensive rebounds for the other five; the turnovers become the other team's steals; the missed field goals the
     other team's blocks; the makes carry assists at the team's assisted share. The score accumulates, the lead decides
     garbage time (the closers out while it holds, back if it does not), a tie at the buzzer plays overtime with the five
     on the floor.

The result is every player's stat line in every world (World.stats: pts, fg3m, reb, ast, stl, blk, tov, minutes, fouls),
his DraftKings points (with the double-double and triple-double bonuses out of the joint draw), and every game's score,
possessions and overtime. to_player_samples() is the {player id: fpts[n]} the field, the contest sim and the chooser
already consume. The dependence between players is the game: the same possessions, the same five, the same misses and
turnovers on both sides, the same blowout.

`state` (the late swap): a dict per game with the clock and each player's line so far is accepted by draw_game and the
remaining minutes are simulated from it (phase 2 of the plan; the parameter is in place, the live box-score reader is not).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ivan.sports.nba.gamesim import profiles as PR
from ivan.sports.nba.gamesim import rotation as RO
from ivan.sports.nba.gamesim import script as SC
from ivan.sports.nba.paths import MODELS_DIR

TEAMS_JSON = MODELS_DIR / "gamesim" / "teams.json"
MINUTES = 48
SEG_MIN = 6
OT_MIN = 5
STATS = ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov")
DK = {"pts": 1.0, "fg3m": 0.5, "reb": 1.25, "ast": 1.5, "stl": 2.0, "blk": 2.0, "tov": -0.5}
LEAGUE_DEFAULTS = dict(league_pace=98.5, oreb_share=0.26, ast_share=0.60, p_stl=0.55, p_blk=0.11, ft_miss_live=0.5, paces={})
FACTOR_LO, FACTOR_HI = 0.7, 1.4         # the team reconciliation factor on attempts


# ----------------------------------------------------------------------------- inputs
@dataclass
class PlayerIn:
    pid: int
    name: str
    team: str
    minutes: float
    rates: Dict[str, float]            # per minute: points, rebounds, assists, steals, blocks, turnovers, fg3m
    cells: Optional[List[int]]
    boundary_sd: Optional[float]
    p_out: float
    fpts_mean: float
    fpts_std: float


@dataclass
class GameIn:
    home: str
    away: str
    spread_home: Optional[float]
    total: Optional[float]
    pace_home: float
    pace_away: float
    players: Dict[str, List[PlayerIn]] = field(default_factory=dict)      # team -> players


def load_teams(path: Path = TEAMS_JSON) -> dict:
    t = dict(LEAGUE_DEFAULTS)
    if Path(path).exists():
        try:
            t.update(json.loads(Path(path).read_text()))
        except Exception:   # noqa: BLE001
            pass
    return t


def _rates_of(p, league_rates: Dict[str, float]) -> Dict[str, float]:
    """The pool's per-minute rates; a player without them gets the league's shape scaled to his fantasy mean."""
    if isinstance(p.rates, dict) and p.rates:
        return {k: float(v) for k, v in p.rates.items()}
    mins = float(p.minutes_proj or 0.0)
    if mins <= 0 or not p.fpts_mean:
        return dict(league_rates)
    fpm_league = sum(DK[k if k != "points" else "pts"] * v for k, v in league_rates.items() if k in ("points",)) or 1.0
    scale = float(p.fpts_mean) / mins / max(fpm_league, 1e-6)
    return {k: v * scale for k, v in league_rates.items()}


LEAGUE_RATES = {"points": 0.47, "rebounds": 0.185, "assists": 0.105, "steals": 0.031, "blocks": 0.021, "turnovers": 0.055, "fg3m": 0.055}


def slate_inputs(session: Session, teams: Optional[dict] = None) -> List[GameIn]:
    """Every game of the slate with its players, lines and paces, from the pool."""
    from ivan.sports.nba.projection.lines import GameLines
    teams = teams or load_teams()
    players = session.execute(select(__import__("ivan.sports.nba.models", fromlist=["NBAPlayer"]).NBAPlayer)).scalars().all()
    players = [p for p in players if p.team and p.opponent and p.fpts_mean is not None]
    tips: dict = {}
    for p in players:
        if p.gametime and (p.team not in tips or p.gametime < tips[p.team]):
            tips[p.team] = p.gametime
    try:
        lines = GameLines.load(session, tips=tips)
    except Exception:   # noqa: BLE001
        lines = GameLines(pd.DataFrame())
    games: Dict[tuple, GameIn] = {}
    from ivan.sports.nba.names import normalize_name
    for p in players:
        key = tuple(sorted([p.team, p.opponent]))
        if key not in games:
            g = lines.by_team.get(p.team) or {}
            home = p.team if g.get("home") else (p.opponent if g else key[0])
            away = key[1] if home == key[0] else key[0]
            gl = lines.by_team.get(home) or {}
            sp = gl.get("spread"); tot = gl.get("total")
            paces = teams.get("paces") or {}
            games[key] = GameIn(home=home, away=away, spread_home=None if sp is None else float(sp), total=None if tot is None else float(tot),
                                pace_home=float(paces.get(home, teams["league_pace"])), pace_away=float(paces.get(away, teams["league_pace"])))
        games[key].players.setdefault(p.team, []).append(PlayerIn(
            pid=int(p.id), name=normalize_name(p.name), team=p.team, minutes=float(p.minutes_proj or 0.0), rates=_rates_of(p, LEAGUE_RATES),
            cells=[int(x) for x in p.rotation_cells] if isinstance(p.rotation_cells, list) and len(p.rotation_cells) == MINUTES else None,
            boundary_sd=float(p.boundary_sd) if p.boundary_sd else None, p_out=float(p.p_out or 0.0), fpts_mean=float(p.fpts_mean), fpts_std=float(p.fpts_std or 0.0)))
    return list(games.values())


# ----------------------------------------------------------------------------- one team's setup
@dataclass
class TeamSetup:
    names: List[str]
    pids: List[int]
    F: np.ndarray                 # (n, k, 48) coverage from the rotation draw (before garbage time)
    fouls: np.ndarray             # (n, k)
    base: np.ndarray              # (k,) planned minutes
    avail: np.ndarray             # (n, k) not scratched / fouled out at the end
    w_fg2: np.ndarray; w_fg3: np.ndarray; w_ft: np.ndarray; w_tov: np.ndarray; w_oreb: np.ndarray; w_dreb: np.ndarray
    w_ast: np.ndarray; w_stl: np.ndarray; w_blk: np.ndarray                                    # (k,) per-minute rates: the multinomial weights
    pct2: np.ndarray; pct3: np.ndarray; pctft: np.ndarray                                         # (k,)
    pie: Dict[str, float]         # per possession: fg2a, fg3a, fta, tov; shares: oreb_share, ast_share, p_stl, p_blk
    factor: float                 # the reconciliation factor on attempts
    exp_points: float             # the roster's expected points before reconciliation
    target_points: Optional[float]


def _grid_from(players: Sequence[PlayerIn]) -> pd.DataFrame:
    """The team's grid: the stored rows; a player without a row gets a flat row of his minutes share (the fill normalises)."""
    rows = {}
    for p in players:
        if p.cells is not None:
            rows[p.name] = [bool(x) for x in p.cells]
        else:
            share = min(1.0, p.minutes / MINUTES)
            rows[p.name] = [share > 0.5] * MINUTES if share > 0.5 else [False] * MINUTES
    return pd.DataFrame(rows, index=list(range(MINUTES))).T


def setup_team(players: Sequence[PlayerIn], opp_players: Sequence[PlayerIn], n: int, rng: np.random.Generator, P: float, target_points: Optional[float],
               prof_state: pd.DataFrame, profiles: PR.ShootingProfiles, foul_state: pd.DataFrame, foul_rates: RO.FoulRates, groups: Dict[str, str], teams: dict) -> TeamSetup:
    players = [p for p in players if p.minutes > 0 or p.cells]
    names = [p.name for p in players]; k = len(names)
    grid = _grid_from(players)
    spreads = {p.name: (p.boundary_sd if p.boundary_sd else RO.default_spread(p.minutes)) for p in players}
    rates = {p.name: foul_rates.rate(foul_state, p.name, p.minutes) for p in players}
    d = RO.draw(grid, spreads, n, rng, np.zeros((n, MINUTES)), np.zeros(n, dtype=int), p_out={p.name: p.p_out for p in players if p.p_out},
                foul_rate=rates, groups=groups, garbage=False)
    order = [d["names"].index(nm) for nm in names]
    F = d["onfloor"][:, order, :]; fouls = d["fouls"][:, order]
    avail = ~d["fouled_out"][:, order] & (F.sum(axis=2) > 0)
    base = np.array([p.minutes for p in players], dtype=float)
    # each player's attempts and percentages, scaled to the pool's points and threes
    prof = [PR.fit_to_rates(profiles.profile(prof_state, p.name), p.rates.get("points"), p.rates.get("fg3m")) for p in players]
    w_fg3 = np.array([q["fg3a_pm"] for q in prof]); w_fg2 = np.array([q["fga_pm"] - q["fg3a_pm"] for q in prof]); w_ft = np.array([q["fta_pm"] for q in prof])
    pct2 = np.array([q["fg2_pct"] for q in prof]); pct3 = np.array([q["fg3_pct"] for q in prof]); pctft = np.array([q["ft_pct"] for q in prof])
    ofrac = np.array([q["oreb_frac"] for q in prof])
    r = lambda key: np.array([p.rates.get(key, 0.0) for p in players], dtype=float)            # noqa: E731
    w_tov, w_reb, w_ast, w_stl, w_blk = r("turnovers"), r("rebounds"), r("assists"), r("steals"), r("blocks")
    w_oreb = w_reb * ofrac; w_dreb = w_reb * (1.0 - ofrac)
    mins = base
    exp = lambda w: float((w * mins).sum())                                                      # noqa: E731
    exp_points = float(((w_fg2 * pct2 * 2.0 + w_fg3 * pct3 * 3.0 + w_ft * pctft) * mins).sum())
    factor = float(np.clip(target_points / exp_points, FACTOR_LO, FACTOR_HI)) if target_points and exp_points > 0 else 1.0
    # the opponent's expected events, for the coupling shares
    oprof = [PR.fit_to_rates(profiles.profile(prof_state, p.name), p.rates.get("points"), p.rates.get("fg3m")) for p in opp_players if p.minutes > 0]
    omins = np.array([p.minutes for p in opp_players if p.minutes > 0], dtype=float)
    o_fga = float(sum(q["fga_pm"] * m for q, m in zip(oprof, omins))); o_fgm = float(sum(((q["fga_pm"] - q["fg3a_pm"]) * q["fg2_pct"] + q["fg3a_pm"] * q["fg3_pct"]) * m for q, m in zip(oprof, omins)))
    o_tov = float(sum(p.rates.get("turnovers", 0.0) * p.minutes for p in opp_players if p.minutes > 0))
    own_miss = exp(w_fg2 * (1 - pct2) + w_fg3 * (1 - pct3)) * factor + exp(w_ft * (1 - pctft)) * factor * teams.get("ft_miss_live", 0.5)
    own_fgm = exp(w_fg2 * pct2 + w_fg3 * pct3) * factor
    pie = dict(fg2a=exp(w_fg2) * factor / P, fg3a=exp(w_fg3) * factor / P, fta=exp(w_ft) * factor / P, tov=exp(w_tov) / P,
               oreb_share=float(np.clip(exp(w_oreb) / own_miss, 0.05, 0.6)) if own_miss > 0 else teams["oreb_share"],
               ast_share=float(np.clip(exp(w_ast) / own_fgm, 0.3, 0.85)) if own_fgm > 0 else teams["ast_share"],
               p_stl=float(np.clip(exp(w_stl) / o_tov, 0.2, 0.9)) if o_tov > 0 else teams["p_stl"],
               p_blk=float(np.clip(exp(w_blk) / max(o_fga - o_fgm, 1e-6), 0.02, 0.3)) if o_fga > 0 else teams["p_blk"])
    return TeamSetup(names=names, pids=[p.pid for p in players], F=F, fouls=fouls, base=base, avail=avail, w_fg2=w_fg2, w_fg3=w_fg3, w_ft=w_ft, w_tov=w_tov,
                     w_oreb=w_oreb, w_dreb=w_dreb, w_ast=w_ast, w_stl=w_stl, w_blk=w_blk, pct2=pct2, pct3=pct3, pctft=pctft, pie=pie, factor=factor,
                     exp_points=exp_points, target_points=target_points)


# ----------------------------------------------------------------------------- the minute loop
def _split(rng: np.random.Generator, count: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(n,) counts over (n, k) weights -> (n, k) multinomial draws; a world with no weight spreads evenly."""
    n, k = weights.shape
    w = np.clip(weights, 0.0, None)
    tot = w.sum(axis=1, keepdims=True)
    p = np.where(tot > 0, w / np.maximum(tot, 1e-12), 1.0 / k)
    p = p / p.sum(axis=1, keepdims=True)
    out = np.zeros((n, k), dtype=np.int64)
    has = count > 0
    if has.any():
        out[has] = rng.multinomial(count[has].astype(np.int64), p[has])
    return out


class _Acc:
    def __init__(self, n: int, k: int):
        self.fg2a = np.zeros((n, k), np.int64); self.fg3a = np.zeros((n, k), np.int64); self.fta = np.zeros((n, k), np.int64)
        self.fg2m = np.zeros((n, k), np.int64); self.fg3m = np.zeros((n, k), np.int64); self.ftm = np.zeros((n, k), np.int64)
        self.oreb = np.zeros((n, k), np.int64); self.dreb = np.zeros((n, k), np.int64); self.ast = np.zeros((n, k), np.int64)
        self.stl = np.zeros((n, k), np.int64); self.blk = np.zeros((n, k), np.int64); self.tov = np.zeros((n, k), np.int64)
        self.minutes = np.zeros((n, k))
        self.carry = np.random.default_rng(n).random(n)          # the endings' running fraction, started at random so the minutes do not all round the same way

    def pts(self) -> np.ndarray:
        return 2 * self.fg2m + 3 * self.fg3m + self.ftm


def _minute(rng, N: np.ndarray, cov: np.ndarray, T: TeamSetup, acc: _Acc, teams: dict, make_factor: Optional[np.ndarray] = None) -> dict:
    """
    One team's offence for one minute: N (n,) expected possessions, cov (n, k) coverage, make_factor (n,) the night's and the
    lead's pull on the make percentages. Every possession ends once: the minute's endings are the possessions times the
    team's endings per possession, carried as a running fraction so the count has no noise beyond the clock's (error
    diffusion: the minute gets the whole endings the cumulative expectation has crossed), split into two-point attempts,
    three-point attempts, free-throw trips and turnovers by the pie (multinomial), then among the five on the floor by their
    rates times coverage; makes are binomial at each player's percentage. Returns what the defence needs.
    """
    kinds = np.array([T.pie["fg2a"], T.pie["fg3a"], T.pie["fta"] / 2.0, T.pie["tov"]])
    acc.carry += N * kinds.sum()
    endings = np.floor(acc.carry).astype(np.int64)
    acc.carry -= endings
    by_kind = rng.multinomial(endings, kinds / kinds.sum())                       # (n, 4)
    fg2a = _split(rng, by_kind[:, 0], T.w_fg2[None, :] * cov)
    fg3a = _split(rng, by_kind[:, 1], T.w_fg3[None, :] * cov)
    fta = 2 * _split(rng, by_kind[:, 2], T.w_ft[None, :] * cov)
    tov = _split(rng, by_kind[:, 3], T.w_tov[None, :] * cov)
    f = np.ones(len(N)) if make_factor is None else make_factor
    p2 = np.clip(T.pct2[None, :] * f[:, None], 0.0, 1.0); p3 = np.clip(T.pct3[None, :] * f[:, None], 0.0, 1.0)
    fg2m = rng.binomial(fg2a, p2); fg3m = rng.binomial(fg3a, p3); ftm = rng.binomial(fta, T.pctft[None, :])
    acc.fg2a += fg2a; acc.fg3a += fg3a; acc.fta += fta; acc.tov += tov; acc.fg2m += fg2m; acc.fg3m += fg3m; acc.ftm += ftm
    acc.minutes += cov
    fgm = (fg2m + fg3m).sum(axis=1); fg_miss = (fg2a + fg3a).sum(axis=1) - fgm
    ft_miss_live = rng.binomial((fta - ftm).sum(axis=1), teams.get("ft_miss_live", 0.5))
    ast = _split(rng, rng.binomial(fgm, T.pie["ast_share"]), T.w_ast[None, :] * cov)
    acc.ast += ast
    return dict(fg_miss=fg_miss, miss=fg_miss + ft_miss_live, tov=tov.sum(axis=1), pts=(2 * fg2m + 3 * fg3m + ftm).sum(axis=1))


def _defence(rng, off: dict, T_off: TeamSetup, cov_off: np.ndarray, acc_off: _Acc, T_def: TeamSetup, cov_def: np.ndarray, acc_def: _Acc) -> None:
    """The offence's misses and turnovers land on both sides: offensive rebounds and the defence's rebounds, steals and blocks."""
    oreb = rng.binomial(off["miss"], T_off.pie["oreb_share"])
    acc_off.oreb += _split(rng, oreb, T_off.w_oreb[None, :] * cov_off)
    acc_def.dreb += _split(rng, off["miss"] - oreb, T_def.w_dreb[None, :] * cov_def)
    acc_def.stl += _split(rng, rng.binomial(off["tov"], T_def.pie["p_stl"]), T_def.w_stl[None, :] * cov_def)
    acc_def.blk += _split(rng, rng.binomial(off["fg_miss"], T_def.pie["p_blk"]), T_def.w_blk[None, :] * cov_def)


def draw_game(game: GameIn, n: int, rng: np.random.Generator, params: dict, teams: dict, prof_state: pd.DataFrame, profiles: PR.ShootingProfiles,
              foul_state: pd.DataFrame, foul_rates: RO.FoulRates, groups: Dict[str, str], state: Optional[dict] = None) -> dict:
    """One game, n worlds. Returns per team the TeamSetup and accumulators, plus the game's margin, totals, possessions, overtime, garbage onsets."""
    P = SC.expected_possessions(game.pace_home, game.pace_away, params)
    if game.total is not None and game.spread_home is not None:
        e_h, e_a = SC.efficiencies(game.total, game.spread_home, P)
        targets = {game.home: e_h * P, game.away: e_a * P}
    else:
        targets = {game.home: None, game.away: None}
        e_h = e_a = 1.13
    script = SC.draw(n, P, e_h, e_a, params, rng)
    poss_min = np.repeat(script["poss"], SEG_MIN, axis=1) / SEG_MIN                      # (n, 48) expected possessions per minute
    home_players = game.players.get(game.home, []); away_players = game.players.get(game.away, [])
    setups = {game.home: setup_team(home_players, away_players, n, rng, P, targets[game.home], prof_state, profiles, foul_state, foul_rates, groups, teams),
              game.away: setup_team(away_players, home_players, n, rng, P, targets[game.away], prof_state, profiles, foul_state, foul_rates, groups, teams)}
    accs = {t: _Acc(n, len(s.names)) for t, s in setups.items()}
    cov_plan = {t: s.F for t, s in setups.items()}
    garbage = {t: None for t in setups}                 # team -> (n, k, 48) coverage once the closers are pulled, or None
    in_garbage = np.zeros(n, dtype=bool); garbage_at = np.full(n, -1)
    margin = np.zeros(n)                                 # home minus away
    teams_order = [game.home, game.away]
    # the script's night: a fast night is a proportionally less efficient one (lambda on the pace deviation), and the lead pulls
    # the efficiencies back (kappa points per possession per point of lead), both as factors on the make percentages
    # the World's own game-level knobs (world_* in script.json, calibrated on the World by `gamesim.py calibrate-world`; the
    # script's standalone values otherwise): the pace coupling, the lead pull-back and each team's pull toward its own
    # expected scoring (gamma: real team points vary less than independent possessions say), all as factors on the makes
    lam = float(params.get("world_lambda", params.get("lambda_pace", 0.0))); kappa = float(params.get("world_kappa", params.get("kappa", 0.0)))
    gamma = float(params.get("world_gamma", 0.0))
    zp = np.log(np.maximum(script["pace"], 1e-6) / P) + 0.5 * float(params.get("sigma_pace", 0.0)) ** 2
    night = 1.0 - lam * zp
    e_of = {game.home: e_h, game.away: e_a}
    pts_so_far = {t: np.zeros(n) for t in teams_order}; poss_so_far = np.zeros(n)
    for m in range(MINUTES):
        covs = {}
        for t in teams_order:
            s = setups[t]
            c = cov_plan[t][:, :, m].copy()
            if in_garbage.any() and garbage[t] is not None:
                c[in_garbage] = garbage[t][in_garbage, :, m]
            covs[t] = c
        lead = {game.home: margin, game.away: -margin}
        factors = {}
        for t in teams_order:
            own_dev = (pts_so_far[t] - poss_so_far * e_of[t]) / P                   # the team's points over its own expectation so far, per expected possession
            factors[t] = np.clip(night * (1.0 - kappa * lead[t] / max(e_of[t], 1e-6)) * (1.0 - gamma * own_dev), 0.6, 1.4)
        offs = {t: _minute(rng, poss_min[:, m], covs[t], setups[t], accs[t], teams, factors[t]) for t in teams_order}
        for t in teams_order:
            pts_so_far[t] += offs[t]["pts"]
        poss_so_far += poss_min[:, m]
        for t, o in zip(teams_order, [game.away, game.home]):
            _defence(rng, offs[t], setups[t], covs[t], accs[t], setups[o], covs[o], accs[o])
        margin += offs[game.home]["pts"] - offs[game.away]["pts"]
        # garbage time off the emergent score: pull while the lead beats the threshold, return when it falls back
        if params.get("garbage", True) and m + 1 >= RO.GARBAGE_FROM and m + 1 < MINUTES:
            left = (MINUTES - (m + 1)) * 60.0
            thr = RO.garbage_threshold(left)
            enter = (~in_garbage) & (np.abs(margin) >= thr)
            leave = in_garbage & (np.abs(margin) < RO.REENTRY * thr)
            if enter.any():
                for t in teams_order:
                    if garbage[t] is None:
                        garbage[t] = setups[t].F.copy()
                    _pull_closers(garbage[t], np.flatnonzero(enter), m + 1, setups[t])
                garbage_at[enter & (garbage_at < 0)] = m + 1
                in_garbage |= enter
            if leave.any():
                in_garbage &= ~leave
    # the endgame tie bump and overtime
    q_tie = float(params.get("world_q_tie", params.get("q_tie", 0.0)))
    if q_tie > 0:
        close = (np.abs(margin) <= SC.TIE_WINDOW) & (margin != 0) & (rng.random(n) < q_tie)
        if close.any():
            # the trailing side scores the difference: the points go to its five on the floor at the buzzer
            for sign, t in ((1, game.away), (-1, game.home)):
                w = close & (np.sign(margin) == sign)
                if w.any():
                    acc = accs[t]; cov = cov_plan[t][:, :, MINUTES - 1]
                    extra = np.abs(margin).astype(np.int64) * w
                    acc.fg2m += _split(rng, extra // 2, setups[t].w_fg2[None, :] * cov); acc.ftm += _split(rng, extra % 2, setups[t].w_ft[None, :] * cov)
            margin[close] = 0.0
    overtime = np.zeros(n, dtype=int)
    tied = margin == 0
    guard = 0
    while tied.any() and guard < 3:
        idx = np.flatnonzero(tied)
        for _ in range(OT_MIN):
            covs = {t: cov_plan[t][:, :, MINUTES - 1] * tied[:, None] for t in teams_order}
            N = script["pace"] / MINUTES * tied
            offs = {t: _minute(rng, N, covs[t], setups[t], accs[t], teams) for t in teams_order}
            for t, o in zip(teams_order, [game.away, game.home]):
                _defence(rng, offs[t], setups[t], covs[t], accs[t], setups[o], covs[o], accs[o])
            margin += offs[game.home]["pts"] - offs[game.away]["pts"]
        overtime[idx] += 1
        tied = margin == 0; guard += 1
    pts = {t: accs[t].pts().sum(axis=1) for t in teams_order}
    return dict(game=game, setups=setups, accs=accs, margin=margin, total=pts[game.home] + pts[game.away], points=pts, poss=script["poss"].sum(axis=1),
                overtime=overtime, garbage_at=garbage_at, P=P)


def _pull_closers(Fg: np.ndarray, worlds: np.ndarray, m0: int, s: TeamSetup) -> None:
    """The garbage coverage from m0 for these worlds: the five with the most planned minutes out as far as the bench can fill, the end of the bench in."""
    k = len(s.names)
    closers = np.argsort(-s.base)[:RO.ON_COURT].tolist()
    bench = [j for j in np.argsort(s.base) if j not in closers][:RO.ON_COURT]
    if not bench:
        return
    for w in worlds:
        takers = [j for j in bench if s.avail[w, j]]
        if not takers:
            continue
        room = np.clip(1.0 - Fg[w, takers, m0:], 0.0, 1.0); tot = room.sum(axis=0)
        on_closers = Fg[w, closers, m0:]
        pull = np.minimum(on_closers.sum(axis=0), tot)
        share = np.where(on_closers.sum(axis=0) > 0, pull / np.maximum(on_closers.sum(axis=0), 1e-9), 0.0)
        Fg[w][np.ix_(closers, np.arange(m0, MINUTES))] = on_closers * (1.0 - share[None, :])
        Fg[w][np.ix_(takers, np.arange(m0, MINUTES))] += np.where(tot > 0, room / np.maximum(tot, 1e-9), 0.0) * pull[None, :]


# ----------------------------------------------------------------------------- the World
@dataclass
class World:
    n: int
    seed: Optional[int]
    names: List[str]
    pids: List[int]
    team_of: List[str]
    stats: Dict[str, np.ndarray]              # stat -> (n, players): pts, fg3m, reb, ast, stl, blk, tov, minutes, fouls, fga, fta
    fpts: np.ndarray                          # (n, players)
    games: List[dict]                         # per game: home, away, P, margin, total, poss, overtime, garbage_at, factors, targets
    drawn_at: datetime

    @classmethod
    def draw(cls, session: Session, n: int = 1000, seed: Optional[int] = None, as_of: Optional[date] = None, games: Optional[List[GameIn]] = None,
             params: Optional[dict] = None, teams: Optional[dict] = None, log=None) -> "World":
        """
        Draw the slate. `games` (slate_inputs) may be given (the as-of runner builds them from a tip record); `as_of` cuts the
        shooting profiles and foul rates at a date (default: everything on file).
        """
        rng = np.random.default_rng(seed)
        params = params or SC.load_params(); teams = teams or load_teams()
        games = games if games is not None else slate_inputs(session, teams)
        from ivan.sports.nba.projection.groups import group_of, position_map
        groups = {nm: group_of(p) for nm, p in position_map(session).items()}
        box = PR.load_box(session, before=as_of)
        profiles = PR.ShootingProfiles.build(box)
        prof_state = profiles.before(as_of) if as_of else profiles.latest()
        fr = RO.FoulRates.load(session)
        foul_state = fr.before(as_of) if as_of else fr.rows.groupby("player").tail(1).set_index("player")
        names: List[str] = []; pids: List[int] = []; team_of: List[str] = []
        cols = {s: [] for s in ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov", "minutes", "fouls", "fga", "fta")}
        out_games = []
        for g in games:
            d = draw_game(g, n, rng, params, teams, prof_state, profiles, foul_state, fr, groups)
            for t in (g.home, g.away):
                s, a = d["setups"][t], d["accs"][t]
                names += s.names; pids += s.pids; team_of += [t] * len(s.names)
                cols["pts"].append(a.pts()); cols["fg3m"].append(a.fg3m); cols["reb"].append(a.oreb + a.dreb); cols["ast"].append(a.ast)
                cols["stl"].append(a.stl); cols["blk"].append(a.blk); cols["tov"].append(a.tov); cols["minutes"].append(a.minutes); cols["fouls"].append(s.fouls)
                cols["fga"].append(a.fg2a + a.fg3a); cols["fta"].append(a.fta)
            out_games.append(dict(home=g.home, away=g.away, P=d["P"], spread_home=g.spread_home, total=g.total, margin=d["margin"], total_pts=d["total"],
                                  poss=d["poss"], overtime=d["overtime"], garbage_at=d["garbage_at"],
                                  factor={t: d["setups"][t].factor for t in (g.home, g.away)}, exp_points={t: d["setups"][t].exp_points for t in (g.home, g.away)},
                                  target_points={t: d["setups"][t].target_points for t in (g.home, g.away)}))
            if log:
                log(f"  {g.away} @ {g.home}: P {d['P']:.1f}, factors {d['setups'][g.home].factor:.2f} / {d['setups'][g.away].factor:.2f}, "
                    f"mean total {d['total'].mean():.1f} (line {g.total}), margin {d['margin'].mean():+.1f} (spread {g.spread_home})")
        stats = {s: (np.concatenate(v, axis=1) if v else np.zeros((n, 0))) for s, v in cols.items()}
        fpts = dk_points(stats)
        return cls(n=n, seed=seed, names=names, pids=pids, team_of=team_of, stats=stats, fpts=fpts, games=out_games, drawn_at=datetime.now())

    def to_player_samples(self) -> Dict[int, np.ndarray]:
        """{player id: fantasy points (n,)}: the shape pre_lock.player.generate_player_samples returns."""
        return {int(pid): self.fpts[:, j].astype(float) for j, pid in enumerate(self.pids)}

    def frame(self) -> pd.DataFrame:
        """Per player: team, mean and sd of fantasy points and of each stat."""
        rows = []
        for j, nm in enumerate(self.names):
            rec = dict(player=nm, pid=self.pids[j], team=self.team_of[j], fpts_mean=float(self.fpts[:, j].mean()), fpts_std=float(self.fpts[:, j].std()))
            for s, arr in self.stats.items():
                rec[s] = float(arr[:, j].mean())
            rows.append(rec)
        return pd.DataFrame(rows)


def moments(w: "World") -> dict:
    """The World's game-level moments over its games: sd of the total around the line, of the margin around the spread, of a team's points, the overtime rate."""
    lined = [g for g in w.games if g["total"] is not None and g["spread_home"] is not None]
    if not lined:
        return {}
    tot = np.concatenate([g["total_pts"] - g["total"] for g in lined]); mar = np.concatenate([g["margin"] + g["spread_home"] for g in lined])
    team = np.concatenate([(g["total_pts"] + g["margin"]) / 2 - (g["total"] - g["spread_home"]) / 2 for g in lined] + [(g["total_pts"] - g["margin"]) / 2 - (g["total"] + g["spread_home"]) / 2 for g in lined])
    return dict(sd_total=float(tot.std()), sd_margin=float(mar.std()), sd_team=float(team.std()), p_ot=float(np.mean([(g["overtime"] > 0).mean() for g in lined])),
                bias_total=float(tot.mean()), bias_margin=float(mar.mean()), garbage=float(np.mean([(g["garbage_at"] >= 0).mean() for g in lined])))


def calibrate(session: Session, games: List[GameIn], params: dict, real: dict, n: int = 400, rounds: int = 2, log=print) -> dict:
    """
    The World's four game-level knobs against four real moments (script.json's real_sd_total / real_sd_margin / p_ot, and the
    real team sd implied by them: var_team = (var_total + var_margin) / 4): gamma from the team sd, kappa from the margin sd,
    lambda from the total sd, q_tie from the overtime rate, by coordinate search on the World itself (a slate's games, n worlds).
    Returns the params with world_gamma, world_kappa, world_lambda, world_q_tie set.
    """
    p = dict(params)
    p.setdefault("world_gamma", 0.0); p.setdefault("world_kappa", p.get("kappa", 0.0)); p.setdefault("world_lambda", p.get("lambda_pace", 0.0)); p.setdefault("world_q_tie", p.get("q_tie", 0.0))
    target_team = float(np.sqrt((real["sd_total"] ** 2 + real["sd_margin"] ** 2) / 4.0))
    targets = [("world_gamma", "sd_team", target_team, (0.0, 0.6)), ("world_kappa", "sd_margin", real["sd_margin"], (0.0, 0.012)),
               ("world_lambda", "sd_total", real["sd_total"], (0.0, 2.0)), ("world_q_tie", "p_ot", real["p_ot"], (0.0, 0.6))]

    def moment(key, value, which):
        q = dict(p); q[key] = float(value)
        return moments(World.draw(session, n=n, seed=11, games=games, params=q))[which]

    for r in range(rounds):
        for key, which, target, (lo, hi) in targets:
            grid = np.linspace(lo, hi, 7)
            vals = [moment(key, g, which) for g in grid]
            # the knob lowers its moment (gamma, kappa, lambda) or raises it (q_tie): take the grid point nearest the target, then refine between its neighbours
            i = int(np.argmin(np.abs(np.asarray(vals) - target)))
            lo2, hi2 = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
            grid2 = np.linspace(lo2, hi2, 5); vals2 = [moment(key, g, which) for g in grid2]
            j = int(np.argmin(np.abs(np.asarray(vals2) - target)))
            p[key] = float(grid2[j])
            log(f"  round {r + 1} {key} = {p[key]:.4f} ({which} {vals2[j]:.2f} vs {target:.2f})")
    return p


def dk_points(stats: Dict[str, np.ndarray]) -> np.ndarray:
    fp = sum(DK[s] * stats[s] for s in STATS).astype(float)
    cats = np.stack([stats["pts"], stats["reb"], stats["ast"], stats["stl"], stats["blk"]]) >= 10
    n10 = cats.sum(axis=0)
    return fp + 1.5 * (n10 >= 2) + 3.0 * (n10 >= 3)
