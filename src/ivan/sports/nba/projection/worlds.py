# src/ivan/sports/nba/projection/worlds.py
"""
Layer 2 of the projection: the game simulation drawn on the mapping, read back per player.

The mapping (layer 1, projection.engine.project_team through live.LiveSlate: each player's minutes and grid row, his
per-minute rates, his scratch probability) is written to the pool first; the World (ivan.sports.nba.gamesim.world) reads
it from there and draws the game both teams are in, n worlds at once; this module turns that draw into what the pool,
the projection store and the desk hold for every player:

    simulate_game(session, ctx, home, away, n, seed)   GameSim: per team a frame indexed by player (fpts_mean, fpts_std, quantiles:
                                                       101 points of the marginal, minutes_sim, minutes_p10, minutes_p90, sim_<stat> for
                                                       points, rebounds, assists, fg3m, steals, blocks, turnovers), per team the samples
                                                       (names, pids, fpts (n, k) and minutes (n, k) as float32: the optimal-lineup share
                                                       (field.optimal) is solved on them), and the game's draw (the factor on each side's
                                                       attempts, the lines, the drawn total and margin, the overtime rate)
    game_pairs(slate_or_players)                       the slate's games as (team, opponent) pairs from the pool
    game_seed(home, away)                              the draw's seed, from the pair: a game's draw depends only on its inputs
    context(session)                                   the WorldContext (profiles, foul rates, groups) cached per process on the history key
    n_worlds(config=None)                              config projection.n_worlds (default 1000): the one number every path draws at
    apply_fpts_pin(frame, samples, name, value)        a hand fpts pin: the mean becomes the pin, the sd, the quantiles and the samples scale

The pool's fpts_mean, fpts_std, fpts_quantiles, minutes_p10 and minutes_p90 are these (live.LiveSlate.write_sim). The
generator draws its own World for the field (gamesim.draw.generate_player_samples): the same inputs, the same distribution.
"""
from __future__ import annotations

import threading
import zlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.sports.nba.gamesim import world as W

QUANTILES = 101
STAT_COLS = {"pts": "points", "fg3m": "fg3m", "reb": "rebounds", "ast": "assists", "stl": "steals", "blk": "blocks", "tov": "turnovers"}

_CTX: dict = {}
_CTX_LOCK = threading.Lock()


@dataclass
class GameSim:
    home: str
    away: str
    n: int
    seed: int
    frames: Dict[str, pd.DataFrame]              # team -> per player (index player_name)
    samples: Dict[str, dict]                     # team -> {names, pids, fpts (n, k) float32, minutes (n, k) float32}
    game: dict = field(default_factory=dict)     # factor per team, lines, drawn total / margin, p_ot, garbage share

    def teams(self) -> Tuple[str, str]:
        return (self.home, self.away)


def n_worlds(config: Optional[dict] = None) -> int:
    """config/nba.json projection.n_worlds, default 1000."""
    try:
        from ivan.core.config import cfg, load_config
        return max(50, int(cfg(config if config is not None else load_config("nba"), "projection.n_worlds", 1000)))
    except Exception:   # noqa: BLE001 - no config: the default
        return 1000


def game_seed(home: str, away: str) -> int:
    """The same pair always draws with the same seed, so a game's draw depends only on its inputs (what the desk and a publish must agree on)."""
    key = "|".join(sorted([str(home).upper(), str(away).upper()]))
    return int(zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF)


def _history_key(session: Session) -> tuple:
    n = session.execute(text("select count(*), max(game_date) from nba_stats")).one()
    return (int(n[0]), str(n[1]))


def context(session: Session, refresh: bool = False) -> W.WorldContext:
    """The WorldContext for the live draw (everything on file), one per process, rebuilt when the box scores grew (the nightly)."""
    key = _history_key(session)
    with _CTX_LOCK:
        if refresh or _CTX.get("key") != key:
            _CTX["value"] = W.WorldContext.build(session)
            _CTX["key"] = key
        return _CTX["value"]


def game_pairs(players: Iterable) -> List[Tuple[str, str]]:
    """The slate's games as (team, opponent) pairs, each once, from pool rows (NBAPlayer or anything with team / opponent)."""
    seen = set(); out = []
    for p in players:
        t, o = getattr(p, "team", None), getattr(p, "opponent", None)
        if not t or not o:
            continue
        key = frozenset((str(t), str(o)))
        if key in seen:
            continue
        seen.add(key); out.append((str(t), str(o)))
    return out


def game_inputs(session: Session, ctx: W.WorldContext) -> Dict[frozenset, W.GameIn]:
    """Every game of the slate as the World's inputs, keyed by its pair (W.slate_inputs: the pool as it stands, so the mapping must be written first)."""
    return {frozenset((g.home, g.away)): g for g in W.slate_inputs(session, ctx.teams)}


def stat_shares(values) -> list:
    """A drawn stat as the share of the worlds at each whole value: [P(0), P(1), ..., P(max)], rounded to 4 places. What the
    desk's stat hover draws as our distribution; a record of the draw, the draw itself is untouched."""
    v = np.rint(np.asarray(values, dtype=float)).astype(int).clip(min=0)
    if not len(v):
        return []
    return [round(float(x), 4) for x in np.bincount(v) / float(len(v))]


def _frame(setup: W.TeamSetup, acc, n: int) -> Tuple[pd.DataFrame, dict]:
    stats = {"pts": acc.pts(), "fg3m": acc.fg3m, "reb": acc.oreb + acc.dreb, "ast": acc.ast, "stl": acc.stl, "blk": acc.blk, "tov": acc.tov}
    fpts = W.dk_points(stats)
    minutes = np.asarray(acc.minutes, dtype=float)
    qs = np.linspace(0.0, 1.0, QUANTILES)
    rows = []
    for j, nm in enumerate(setup.names):
        x = fpts[:, j]; m = minutes[:, j]
        rec = dict(player_name=nm, pid=int(setup.pids[j]), fpts_mean=float(x.mean()), fpts_std=float(x.std()),
                   quantiles=np.quantile(x, qs).round(2).tolist(), minutes_plan=float(setup.base[j]), minutes_sim=float(m.mean()),
                   minutes_p10=float(np.percentile(m, 10)), minutes_p90=float(np.percentile(m, 90)), p_zero=float((m <= 0).mean()))
        dist = {}
        for short, long in STAT_COLS.items():
            v = stats[short][:, j]
            rec[f"sim_{long}"] = float(v.mean())
            dist[long] = stat_shares(v)
        rec["stat_dist"] = dist
        rows.append(rec)
    frame = pd.DataFrame(rows).set_index("player_name") if rows else pd.DataFrame(columns=["pid", "fpts_mean", "fpts_std", "quantiles"]).rename_axis("player_name")
    samples = dict(names=list(setup.names), pids=[int(x) for x in setup.pids], fpts=fpts.astype(np.float32), minutes=minutes.astype(np.float32))
    return frame, samples


def simulate_game(session: Session, ctx: W.WorldContext, game: W.GameIn, n: Optional[int] = None, seed: Optional[int] = None) -> GameSim:
    """One game of the slate drawn n worlds on the mapping as written to the pool, read back per player (GameSim)."""
    n = int(n or n_worlds())
    seed = game_seed(game.home, game.away) if seed is None else int(seed)
    d = ctx.draw_game(game, n, np.random.default_rng(seed))
    frames, samples = {}, {}
    for t in (game.home, game.away):
        frames[t], samples[t] = _frame(d["setups"][t], d["accs"][t], n)
    info = dict(home=game.home, away=game.away, P=float(d["P"]), spread_home=game.spread_home, total=game.total,
                factor={t: float(d["setups"][t].factor) for t in (game.home, game.away)},
                exp_points={t: float(d["setups"][t].exp_points) for t in (game.home, game.away)},
                target_points={t: d["setups"][t].target_points for t in (game.home, game.away)},
                mean_total=float(np.mean(d["total"])), mean_margin=float(np.mean(d["margin"])), sd_total=float(np.std(d["total"])),
                p_ot=float(np.mean(d["overtime"] > 0)), garbage=float(np.mean(d["garbage_at"] >= 0)), garbage_on=bool(ctx.params.get("garbage", True)))
    return GameSim(home=game.home, away=game.away, n=n, seed=seed, frames=frames, samples=samples, game=info)


def apply_fpts_pin(frame: pd.DataFrame, samples: Optional[dict], name: str, value: float) -> None:
    """
    A hand fpts pin on a player: his fantasy mean becomes the pin and his sd, quantiles and samples scale by the same ratio
    (the shape of his night stays, the level moves); the World's own mean is kept in fpts_model. In place.
    """
    if name not in frame.index:
        return
    if "fpts_model" not in frame.columns:
        frame["fpts_model"] = frame["fpts_mean"].astype(float)
    old = float(frame.loc[name, "fpts_mean"]); ratio = (float(value) / old) if old > 0 else 1.0
    frame.loc[name, "fpts_mean"] = float(value)
    frame.loc[name, "fpts_std"] = float(frame.loc[name, "fpts_std"]) * ratio
    q = frame.loc[name, "quantiles"]
    if isinstance(q, list):
        frame.at[name, "quantiles"] = [round(float(x) * ratio, 2) for x in q]
    if samples and name in samples.get("names", []):
        j = samples["names"].index(name)
        samples["fpts"][:, j] = (samples["fpts"][:, j] * ratio).astype(np.float32)
