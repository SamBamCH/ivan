# src/ivan/sports/nba/gamesim/rerun.py
"""
The as-of rerun: what the current code says about a past slate, from its tip record, next to what happened.

A past slate's backtest screen has three columns per layer: what we published at the time (the tip record's
projection), what the code on this checkout says as of that tip, and the actuals. This module makes the middle one:

inputs_from_tip(session, dg, tip)   the World's inputs from the tip record (tips.read_tip): the pool with its minutes, grid
                                    rows and scratch probabilities as the desk had them; the per-minute rates rebuilt the way the
                                    live projection built them (history before the day, shrunk, the books' means over the
                                    projected minutes where priced: the record's odds.parquet); the lines from the record's
                                    game_lines rows; each team's trailing pace from the events store before the day
rerun(session, dg, n, seed, tip)    World.draw with the history cut at the day (shooting profiles, foul rates), the actuals
                                    (nba_stats, the day's events), the layered scorecard, and the record written under
                                    data/nba/slates/<dg>/rerun/<stamp>-<commit>/ (meta.json, players.json, arrays.npz)
scorecard(...)                      one row per layer with the truth plugged in above it: possessions (vs the real count), points
                                    and margin (vs the real finals), minutes, attempts, makes and outcomes, fantasy points (per
                                    player: MAE, bias, the share of actuals inside the drawn 10-90 band, the mean PIT), with the
                                    published projection's error beside it where it has one
load_latest(dg) / load(dg, name)    the records on file for the screen; player_view(rec, stat) the per-player rows with each
                                    player's drawn histogram of a stat and his actual inside it

Nothing here reads anything dated after the tip except the actuals, which are the answer.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.sports.nba import slates as S
from ivan.sports.nba import tips as T
from ivan.sports.nba.gamesim import possessions as X
from ivan.sports.nba.gamesim import script as SC
from ivan.sports.nba.gamesim import world as W
from ivan.sports.nba.names import normalize_name

STATS_OUT = ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov", "fga", "fta", "minutes", "fpts")
BOX_COLS = {"pts": "points", "fg3m": "fg3m", "reb": "rebounds", "ast": "assists", "stl": "steals", "blk": "blocks", "tov": "turnovers", "fga": "fga", "fta": "fta",
            "minutes": "minutes", "fpts": "dk_fpts"}
LAYERS = [("possessions", "game", ["poss"]), ("points and margin", "game", ["total", "margin"]), ("minutes", "player", ["minutes"]),
          ("attempts", "player", ["fga", "fta"]), ("makes and outcomes", "player", ["pts", "fg3m", "reb", "ast", "stl", "blk", "tov"]), ("fantasy points", "player", ["fpts"])]
HIST_BINS = 24


def rerun_dir(dg: int, root: Optional[Path] = None) -> Path:
    return S.paths(int(dg), root).root / "rerun"


# ----------------------------------------------------------------------------- inputs
def lines_from_record(rows: list, players: list) -> Dict[str, dict]:
    """team -> {spread, total, home, opponent} from the record's game_lines rows (per-book rows or a consensus frame)."""
    from ivan.sports.nba.projection.lines import GameLines, consensus
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    cons = consensus(df) if "bookmaker" in df.columns else df
    tips: Dict[str, datetime] = {}
    for p in players:
        gt = p.get("gametime")
        if p.get("team") and gt:
            t = pd.Timestamp(gt).to_pydatetime()
            if p["team"] not in tips or t < tips[p["team"]]:
                tips[p["team"]] = t
    return GameLines(cons, tips=tips).by_team


def rates_as_of(session: Session, day: date, players: list, odds: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per player the per-minute rates the live projection would have used at the tip: history before the day (shrunk) with
    the books' means over the projected minutes where priced."""
    from ivan.sports.nba.projection.history import STATS, build_history, load_box_scores
    from ivan.sports.nba.projection.market import blend_rates, market_means, market_rates
    from ivan.sports.nba.projection.rates import shrunk_rates
    box = load_box_scores(session)
    box = box[box.game_date < day]
    hist = build_history(box)
    state = hist.before(day)
    basis = {str(p["name"]): float(p.get("minutes_proj") or 0.0) for p in players}
    market = {}
    if odds is not None and len(odds):
        o = odds.copy(); o["player"] = o.player.map(normalize_name)
        try:
            market = market_rates(market_means(o), basis).rates
        except Exception:   # noqa: BLE001 - odd odds rows: history alone
            market = {}
    out = {}
    for p in players:
        nm = str(p["name"])
        hr = shrunk_rates(state.loc[nm] if nm in state.index else None, hist.league_rates)
        out[nm] = blend_rates(hr, market.get(nm))
    return out


def inputs_from_tip(session: Session, dg: int, tip: datetime, root: Optional[Path] = None) -> Tuple[List[W.GameIn], date, dict]:
    rec = T.read_tip(dg, tip, root)
    if rec is None:
        raise FileNotFoundError(f"slate {dg} has no tip record at {tip:%Y-%m-%d %H:%M}")
    players = [p for p in rec["players"] if p.get("team") and p.get("opponent") and p.get("fpts_mean") is not None]
    day = tip.date()
    by_team = lines_from_record(rec["game_lines"], players)
    rates = rates_as_of(session, day, players, rec["odds"])
    ev = X.load_events(); ev["game_date"] = pd.to_datetime(ev.game_date).dt.date
    seg = X.team_segments(ev[ev.game_date < day])
    paces = SC.team_paces(seg) if len(seg) else {}
    teams = W.load_teams(); league_pace = float(SC.game_pace(seg).poss.mean()) if len(seg) else teams["league_pace"]
    from ivan.sports.nba.projection.rotation import default_spread
    games: Dict[tuple, W.GameIn] = {}
    for p in players:
        key = tuple(sorted([p["team"], p["opponent"]]))
        if key not in games:
            g = by_team.get(p["team"]) or {}
            home = p["team"] if g.get("home") else (p["opponent"] if g else key[0]); away = key[1] if home == key[0] else key[0]
            gl = by_team.get(home) or {}
            games[key] = W.GameIn(home=home, away=away, spread_home=gl.get("spread"), total=gl.get("total"),
                                  pace_home=float(paces.get(home, league_pace)), pace_away=float(paces.get(away, league_pace)))
        cells = p.get("rotation_cells")
        games[key].players.setdefault(p["team"], []).append(W.PlayerIn(
            pid=int(p["id"]), name=normalize_name(str(p["name"])), team=p["team"], minutes=float(p.get("minutes_proj") or 0.0), rates=rates[str(p["name"])],
            cells=[int(x) for x in cells] if isinstance(cells, list) and len(cells) == W.MINUTES else None,
            boundary_sd=default_spread(float(p.get("minutes_proj") or 0.0)), p_out=float(p.get("p_out") or 0.0),
            fpts_mean=float(p["fpts_mean"]), fpts_std=float(p.get("fpts_std") or 0.0)))
    meta = dict(tip=tip.isoformat(timespec="minutes"), players=len(players), lined=sum(1 for g in games.values() if g.total is not None), games=len(games),
                record=rec["meta"])
    return list(games.values()), day, meta


# ----------------------------------------------------------------------------- actuals
def actuals(session: Session, day: date) -> pd.DataFrame:
    cols = ", ".join(sorted(set(BOX_COLS.values())))
    df = pd.read_sql(text(f"select game_id, player_name, team_abbreviation team, {cols} from nba_stats where game_date = :d").bindparams(d=day.isoformat()), session.get_bind())
    df["player"] = df.player_name.map(normalize_name)
    return df


def game_actuals(day: date, home: str, away: str) -> Optional[dict]:
    """The real possessions (both teams' mean) and points of one game from the events store; None without a stored feed."""
    ev = X.load_events()
    ev = ev[(pd.to_datetime(ev.game_date).dt.date == day) & ev.team.isin([home, away]) & ev.opp.isin([home, away])]
    if ev.empty:
        return None
    seg = X.team_segments(ev)
    gp = SC.game_pace(seg)
    h = gp[gp.team == home]; a = gp[gp.team == away]
    if h.empty or a.empty:
        return None
    return dict(poss=float((h.poss.iloc[0] + a.poss.iloc[0]) / 2), pts_home=float(h.pts.iloc[0]), pts_away=float(a.pts.iloc[0]), overtime=int(h.overtime.iloc[0]))


# ----------------------------------------------------------------------------- the scorecard
def _pit(sample: np.ndarray, x: float) -> float:
    s = np.sort(sample)
    return float((np.searchsorted(s, x, side="left") + np.searchsorted(s, x, side="right")) / (2.0 * len(s)))


def _band(sample: np.ndarray, x: float) -> bool:
    lo, hi = np.percentile(sample, [10, 90])
    return bool(lo <= x <= hi)


def scorecard(world: W.World, act: pd.DataFrame, games_act: Dict[tuple, Optional[dict]], published: Dict[int, dict]) -> List[dict]:
    """One row per layer and metric. Player rows use the players who played (minutes > 0); game rows the games with a stored feed."""
    rows = []
    # games
    for layer, _, metrics in LAYERS[:2]:
        for m in metrics:
            vals, pits, inb, pub = [], [], [], []
            for g in world.games:
                ga = games_act.get((g["home"], g["away"]))
                if not ga:
                    continue
                if m == "poss":
                    draw, truth = g["poss"], ga["poss"]
                elif m == "total":
                    draw, truth = g["total_pts"], ga["pts_home"] + ga["pts_away"]
                else:
                    draw, truth = g["margin"], ga["pts_home"] - ga["pts_away"]
                vals.append(abs(float(draw.mean()) - truth)); pits.append(_pit(draw, truth)); inb.append(_band(draw, truth))
                if m == "total" and g["total"] is not None:
                    pub.append(abs(g["total"] - truth))
                if m == "margin" and g["spread_home"] is not None:
                    pub.append(abs(-g["spread_home"] - truth))
            if vals:
                rows.append(dict(layer=layer, metric=m, n=len(vals), mae=float(np.mean(vals)), bias=None, in_band=float(np.mean(inb)), pit=float(np.mean(pits)),
                                 mae_published=float(np.mean(pub)) if pub else None, note="the line" if pub else None))
    # players
    a = act[act.minutes > 0].set_index("player")
    idx = {nm: j for j, nm in enumerate(world.names)}
    for layer, _, metrics in LAYERS[2:]:
        for m in metrics:
            col = BOX_COLS[m]
            errs, pits, inb, pubs = [], [], [], []
            for nm, j in idx.items():
                if nm not in a.index:
                    continue
                truth = float(a.loc[nm, col]) if not isinstance(a.loc[nm], pd.DataFrame) else float(a.loc[nm, col].iloc[0])
                draw = world.fpts[:, j] if m == "fpts" else world.stats[m][:, j]
                errs.append(float(draw.mean()) - truth); pits.append(_pit(draw, truth)); inb.append(_band(draw, truth))
                pb = published.get(world.pids[j])
                if pb and m in ("minutes", "fpts") and pb.get(m) is not None:
                    pubs.append(abs(float(pb[m]) - truth))
            if errs:
                e = np.asarray(errs)
                rows.append(dict(layer=layer, metric=m, n=len(errs), mae=float(np.abs(e).mean()), bias=float(e.mean()), in_band=float(np.mean(inb)), pit=float(np.mean(pits)),
                                 mae_published=float(np.mean(pubs)) if pubs else None, note="the projection at the tip" if pubs else None))
    rows.append(dict(layer="ownership and the field", metric="ownership", n=0, mae=None, bias=None, in_band=None, pit=None, mae_published=None, note="not rerun: the field model is a separate component"))
    return rows


# ----------------------------------------------------------------------------- run and store
def rerun(session: Session, dg: int, n: int = 1000, seed: Optional[int] = 0, tip: Optional[datetime] = None, root: Optional[Path] = None, log=print) -> dict:
    tips = T.tips_of(dg, root)
    if not tips:
        raise FileNotFoundError(f"slate {dg} has no tip records (data/nba/slates/{dg}/tips/)")
    tip = tip or tips[0]
    games, day, meta = inputs_from_tip(session, dg, tip, root)
    log(f"slate {dg}: tip {tip:%m-%d %H:%M}, {meta['players']} players, {meta['games']} games ({meta['lined']} lined); drawing {n} worlds as of {day}")
    world = W.World.draw(session, n=n, seed=seed, as_of=day, games=games, log=log)
    act = actuals(session, day)
    games_act = {(g["home"], g["away"]): game_actuals(day, g["home"], g["away"]) for g in world.games}
    rec = T.read_tip(dg, tip, root)
    published = {int(p["id"]): dict(minutes=p.get("minutes_proj"), fpts=p.get("fpts_mean"), fpts_std=p.get("fpts_std"), own=p.get("ownership")) for p in rec["players"]}
    card = scorecard(world, act, games_act, published)
    from ivan.core.gitinfo import git_state
    gs = git_state()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{gs.get('commit') or 'nocommit'}"
    d = rerun_dir(dg, root) / name
    d.mkdir(parents=True, exist_ok=True)
    a = act.set_index("player")
    players = []
    for j, nm in enumerate(world.names):
        row = dict(name=nm, pid=world.pids[j], team=world.team_of[j], published=published.get(world.pids[j]), actual=None)
        if nm in a.index:
            r = a.loc[nm] if not isinstance(a.loc[nm], pd.DataFrame) else a.loc[nm].iloc[0]
            row["actual"] = {m: float(r[c]) for m, c in BOX_COLS.items()}
        players.append(row)
    arrays = {m: (world.fpts if m == "fpts" else world.stats[m]).astype(np.float32) for m in STATS_OUT}
    np.savez_compressed(d / "arrays.npz", **arrays)
    (d / "players.json").write_text(json.dumps(players, default=str))
    gm = [dict(home=g["home"], away=g["away"], P=g["P"], spread_home=g["spread_home"], total=g["total"], factor=g["factor"], target_points=g["target_points"],
               exp_points=g["exp_points"], drawn_total=float(g["total_pts"].mean()), drawn_margin=float(g["margin"].mean()), drawn_poss=float(g["poss"].mean()),
               p_ot=float((g["overtime"] > 0).mean()), garbage=float((g["garbage_at"] >= 0).mean()), actual=games_act.get((g["home"], g["away"])),
               hist_total=np.histogram(g["total_pts"], bins=HIST_BINS)[0].tolist(), hist_margin=np.histogram(g["margin"], bins=HIST_BINS)[0].tolist(),
               edges_total=np.histogram(g["total_pts"], bins=HIST_BINS)[1].tolist(), edges_margin=np.histogram(g["margin"], bins=HIST_BINS)[1].tolist()) for g in world.games]
    meta_out = dict(draft_group=int(dg), tip=tip.isoformat(timespec="minutes"), day=day.isoformat(), n=int(n), seed=seed, commit=gs.get("commit"), dirty=gs.get("dirty"),
                    subject=gs.get("subject"), written_at=datetime.now().isoformat(timespec="seconds"), inputs=meta, moments=W.moments(world), scorecard=card, games=gm)
    (d / "meta.json").write_text(json.dumps(meta_out, indent=1, default=str))
    log(f"written {d}")
    return dict(meta=meta_out, players=players, arrays=arrays, dir=d)


def list_reruns(dg: int, root: Optional[Path] = None) -> List[str]:
    d = rerun_dir(dg, root)
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "meta.json").exists()) if d.is_dir() else []


def load(dg: int, name: str, root: Optional[Path] = None) -> Optional[dict]:
    d = rerun_dir(dg, root) / name
    if not (d / "meta.json").exists():
        return None
    arrays = dict(np.load(d / "arrays.npz")) if (d / "arrays.npz").exists() else {}
    return dict(meta=json.loads((d / "meta.json").read_text()), players=json.loads((d / "players.json").read_text()), arrays=arrays, dir=d)


def load_latest(dg: int, root: Optional[Path] = None) -> Optional[dict]:
    names = list_reruns(dg, root)
    return load(dg, names[-1], root) if names else None


def histogram(sample: np.ndarray, actual: Optional[float], bins: int = HIST_BINS) -> dict:
    """Bar heights 0..1, the bin edges, which bin holds the actual (or None), the 10-90 band bins: what the sparkline draws."""
    sample = np.asarray(sample, dtype=float)
    lo, hi = float(sample.min()), float(sample.max())
    if actual is not None:
        lo, hi = min(lo, actual), max(hi, actual)
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(sample, bins=bins, range=(lo, hi))
    top = counts.max() if counts.max() > 0 else 1
    q10, q90 = np.percentile(sample, [10, 90])
    at = None if actual is None else int(min(bins - 1, max(0, np.searchsorted(edges, actual, side="right") - 1)))
    return dict(bars=[float(c) / float(top) for c in counts], lo=lo, hi=hi, at=at, pit=None if actual is None else _pit(sample, actual),
                band=[int(min(bins - 1, max(0, np.searchsorted(edges, q, side="right") - 1))) for q in (q10, q90)], mean=float(sample.mean()))


def player_view(rec: dict, stat: str = "fpts") -> Dict[str, List[dict]]:
    """team -> rows (sorted by the published fantasy mean): published / drawn mean / actual for minutes and fpts, the stat's histogram with the actual."""
    arr = rec["arrays"].get(stat if stat in rec["arrays"] else "fpts")
    fp = rec["arrays"].get("fpts"); mn = rec["arrays"].get("minutes")
    out: Dict[str, List[dict]] = {}
    for j, p in enumerate(rec["players"]):
        a = p.get("actual") or {}
        pub = p.get("published") or {}
        row = dict(name=p["name"], pid=p["pid"], played=bool(a) and float(a.get("minutes") or 0) > 0,
                   min_pub=pub.get("minutes"), min_sim=float(mn[:, j].mean()) if mn is not None else None, min_act=a.get("minutes"),
                   fpts_pub=pub.get("fpts"), fpts_sim=float(fp[:, j].mean()) if fp is not None else None, fpts_sd_sim=float(fp[:, j].std()) if fp is not None else None,
                   fpts_act=a.get("fpts"), stat_sim=float(arr[:, j].mean()) if arr is not None else None, stat_act=a.get(stat),
                   hist=histogram(arr[:, j], a.get(stat) if a else None) if arr is not None else None)
        out.setdefault(p["team"], []).append(row)
    for t in out:
        out[t].sort(key=lambda r: -(r["fpts_pub"] or 0))
    return out
