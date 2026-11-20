"""
ivan.sports.nba.backtest.game

One game of a past slate, at three depths: the slate's games with how each was projected, one game drawn by the game
simulation from its tip record with the inputs open to change, one player's drawn distributions against what he did.
The desk's backtest view (ivan.web.nba.backtest) and scripts/nba/backtest_game.py show the same.

    slate_games(dg)                      the slate's games from its tip records: key (AWAY-HOME), the tip, the tip record the game
                                         is judged on (the newest at or before its tip), the line in that record
    overview(session, dg)                one row per game: the line against the final, each team's projected and scored fantasy
                                         points, the minutes and fantasy point errors of what was published at the game's own tip,
                                         the largest misses, whether a draw is on file
    simulate(session, dg, key, n, edits) the game drawn n worlds by the code on this checkout from the tip record (history,
                                         shooting profiles and foul rates cut at the day), written to the game's directory:
                                         base.json without edits, what_if.json with them
    view(session, dg, key)               everything the game's page shows: per team each pool player's published / set / drawn /
                                         actual minutes, fantasy points and stats, the share of worlds he is on the floor each
                                         minute over the minutes he played
    player_view(session, dg, key, name)  one player: the drawn distribution of his fantasy points, minutes and each stat with the
                                         actual marked (and the unedited draw beside it when there are edits)
    actual_edits(session, dg, key)       the edits that set every player to the minutes he played and the five who started

The edits (data/nba/slates/<dg>/backtest/<AWAY-HOME>/edits.json): {"minutes": {player: minutes}, "starters": {player: bool},
"spread_home": x, "total": x}. A typed minutes number is what the player's rotation is steered to, with no scratch and no cut
chance (the backtest says he played that); 0 takes him out. The team's players without a typed number share what is left of
240 in proportion to their minutes. The per-minute rates are the tip record's (the books' means over the minutes projected
at the tip, history elsewhere), so a change of minutes moves a player's totals in proportion. A starter call decides who is on
the floor at the tip (the stint rotation reads it; the plan rotation draws the stored grid row and ignores it). Nothing here
reads anything dated after the tip except the actuals and what the user types.
"""
from __future__ import annotations

import json
import shutil
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.sports.nba import slates as S
from ivan.sports.nba.backtest import rerun as RR
from ivan.sports.nba.backtest import tips as T
from ivan.sports.nba.gamesim import world as W
from ivan.sports.nba.names import normalize_name

STATS = (("points", "pts"), ("rebounds", "reb"), ("assists", "ast"), ("fg3m", "3pm"), ("steals", "stl"), ("blocks", "blk"), ("turnovers", "tov"))
CHART_STATS = (("fpts", "fantasy points"), ("minutes", "minutes")) + tuple((s, s if s != "fg3m" else "threes") for s, _ in STATS)
TEAM_MINUTES = 240.0
MINUTES_CAP = 44.0
CHART_BINS = 16
N_MIN, N_MAX = 250, 20000
BIG_MISS = 8.0

_LOCK = threading.Lock()
_DAY: Dict[tuple, dict] = {}        # (day, box scores on file) -> the history state, the paces and the WorldContext cut at the day
_TIP: Dict[tuple, dict] = {}        # (dg, tip, root) -> the tip record's players, lines and rates
_GAMES: Dict[tuple, list] = {}      # (dg, root, the tips on file) -> slate_games (a tip record is written once, so the list holds while the tips do)


# ----------------------------------------------------------------------------- the slate's games
def game_key(away: str, home: str) -> str:
    return f"{away}-{home}"


def game_dir(dg: int, key: str, root: Optional[Path] = None) -> Path:
    return S.paths(int(dg), root).root / "backtest" / str(key).upper()


def _when(x) -> Optional[datetime]:
    if x is None or (isinstance(x, float) and x != x):
        return None
    try:
        return pd.Timestamp(x).to_pydatetime().replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def slate_games(dg: int, root: Optional[Path] = None) -> List[dict]:
    """The slate's games from its tip records, by tip then key; [] for a slate without one (not held here, or archived before them)."""
    tips = T.tips_of(dg, root)
    if not tips:
        return []
    ck = (int(dg), str(root or ""), tuple(tips))
    with _LOCK:
        if ck in _GAMES:
            return [dict(g) for g in _GAMES[ck]]
    players = json.loads((T.tip_dir(dg, tips[0], root) / "players.json").read_text())
    pairs: Dict[frozenset, dict] = {}
    for p in players:
        t, o = p.get("team"), p.get("opponent")
        if not t or not o:
            continue
        g = pairs.setdefault(frozenset((str(t), str(o))), dict(teams=sorted([str(t), str(o)]), tip=None))
        gt = _when(p.get("gametime"))
        if gt is not None and (g["tip"] is None or gt < g["tip"]):
            g["tip"] = gt
    lines: Dict[datetime, dict] = {}
    out = []
    for g in pairs.values():
        rec = (T.tip_for(dg, g["tip"], root) if g["tip"] is not None else None) or tips[0]
        if rec not in lines:
            f = T.tip_dir(dg, rec, root) / "game_lines.json"
            try:
                lines[rec] = RR.lines_from_record(json.loads(f.read_text()) if f.exists() else [], players)
            except Exception:   # noqa: BLE001 - odd line rows: the games without a line
                lines[rec] = {}
        a, b = g["teams"]
        la = lines[rec].get(a) or {}
        home = a if la.get("home") else (b if la else a)
        away = b if home == a else a
        lh = lines[rec].get(home) or {}
        out.append(dict(key=game_key(away, home), home=home, away=away, tip=g["tip"], record=rec, sided=bool(la),
                        spread_home=None if lh.get("spread") is None else float(lh["spread"]), total=None if lh.get("total") is None else float(lh["total"])))
    out.sort(key=lambda g: (g["tip"] or datetime.max, g["key"]))
    with _LOCK:
        if len(_GAMES) > 8:
            _GAMES.clear()
        _GAMES[ck] = [dict(g) for g in out]
    return out


def find_game(dg: int, key: str, root: Optional[Path] = None) -> Optional[dict]:
    """The game by its key, either way round, or by one of its teams."""
    key = str(key).upper()
    parts = set(key.split("-"))
    for g in slate_games(dg, root):
        if g["key"] == key or parts == {g["home"], g["away"]} or key in (g["home"], g["away"]):
            return g
    return None


def finals(session: Session, day: date) -> Dict[str, dict]:
    """{team: points, minutes, game_id} from the day's box scores."""
    rows = session.execute(text("select team_abbreviation, sum(points), sum(minutes), max(game_id) from nba_stats where game_date = :d group by 1").bindparams(d=day.isoformat())).all()
    return {str(t): dict(points=float(p or 0), minutes=float(m or 0), game_id=str(g)) for t, p, m, g in rows}


def _final_of(fin: Dict[str, dict], home: str, away: str) -> Optional[dict]:
    h, a = fin.get(home), fin.get(away)
    if not h or not a:
        return None
    return dict(home=h["points"], away=a["points"], total=h["points"] + a["points"], margin=h["points"] - a["points"], overtime=h["minutes"] > TEAM_MINUTES + 12.0)


def _errors(rows: List[dict]) -> dict:
    """The projection at lock against the box score over the players projected to play or who played."""
    use = [r for r in rows if (r.get("min_proj") or 0) > 0 or (r.get("min_act") or 0) > 0]
    if not use or all(r.get("fpts_act") is None for r in use):
        return dict(n=len(use), min_mae=None, fpts_mae=None, fpts_bias=None)
    dm = [(r.get("min_proj") or 0.0) - (r.get("min_act") or 0.0) for r in use]
    df = [(r.get("fpts_proj") or 0.0) - (r.get("fpts_act") or 0.0) for r in use]
    return dict(n=len(use), min_mae=float(np.mean(np.abs(dm))), fpts_mae=float(np.mean(np.abs(df))), fpts_bias=float(np.mean(df)))


def published_rows(dg: int, g: dict, box_rows: Dict[str, dict], root: Optional[Path] = None, cache: Optional[dict] = None) -> Dict[str, list]:
    """{team: rows} for one game from its own tip record against the box score: every pool player projected at that tip or who played
    (name, min_proj, fpts_proj, min_act, fpts_act; a projected player without a box line sat: 0 and 0 once the game has a box score)."""
    cache = cache if cache is not None else {}
    if g["record"] not in cache:
        cache[g["record"]] = json.loads((T.tip_dir(dg, g["record"], root) / "players.json").read_text())
    out: Dict[str, list] = {g["away"]: [], g["home"]: []}
    has_box = {t: any(v["team"] == t for v in box_rows.values()) for t in out}
    for p in cache[g["record"]]:
        t = p.get("team")
        if t not in out:
            continue
        nm = normalize_name(str(p.get("name") or ""))
        a = box_rows.get(nm) if (box_rows.get(nm) or {}).get("team") == t else None
        proj = p.get("fpts_mean") is not None and _num(p.get("minutes_proj")) > 0
        if not proj and not (a and a["minutes"] > 0):
            continue
        out[t].append(dict(name=nm, min_proj=_num(p.get("minutes_proj")) if proj else None, fpts_proj=_num(p.get("fpts_mean")) if proj else None,
                           min_act=a["minutes"] if a else (0.0 if has_box[t] else None), fpts_act=a["fpts"] if a else (0.0 if has_box[t] else None)))
    return out


def overview(session: Session, dg: int, players: Optional[Dict[str, list]] = None, day: Optional[date] = None, rotations: Optional[dict] = None,
             root: Optional[Path] = None) -> List[dict]:
    """
    One row per game of the slate: each game judged on its own tip record (what its page shows as published), so a player ruled
    out between the slate's lock and his game's tip is not a miss. `rotations` = slate_backtest's grid overlays (the agreement).
    A slate without tip records gets its games and rows from `players` (slate_backtest's {team: rows}, the projection at lock):
    the sides are not known then and the game cannot be opened (`can_open` False).
    """
    games = slate_games(dg, root)
    by_game: Dict[str, Dict[str, list]] = {}
    if games:
        day = games[0]["record"].date()
        b = box(session, day, {t for g in games for t in (g["home"], g["away"])})
        cache: dict = {}
        by_game = {g["key"]: published_rows(dg, g, b, root, cache) for g in games}
    else:
        players = players or {}
        seen = {}
        for t, rows in players.items():
            o = next((r.get("opp") for r in rows if r.get("opp")), None)
            if o:
                a, b_ = sorted([t, str(o)])
                seen.setdefault((a, b_), dict(key=game_key(a, b_), home=b_, away=a, tip=_when(next((r.get("gametime") for r in rows if r.get("gametime")), None)),
                                              record=None, sided=False, spread_home=None, total=None))
        games = sorted(seen.values(), key=lambda g: (g["tip"] or datetime.max, g["key"]))
        by_game = {g["key"]: {t: players.get(t) or [] for t in (g["away"], g["home"])} for g in games}
    fin = finals(session, day) if day is not None else {}
    out = []
    for g in games:
        teams = []
        for t in (g["away"], g["home"]):
            rows = by_game[g["key"]].get(t) or []
            ov = (rotations or {}).get(t) or {}
            teams.append(dict(team=t, fpts_proj=sum(r.get("fpts_proj") or 0.0 for r in rows), fpts_act=sum(r.get("fpts_act") or 0.0 for r in rows) if any(r.get("fpts_act") is not None for r in rows) else None,
                              agreement=ov.get("agreement"), **_errors(rows)))
        both = [r for t in (g["away"], g["home"]) for r in by_game[g["key"]].get(t) or []]
        miss = sorted([dict(name=r["name"], delta=r["fpts_act"] - r["fpts_proj"]) for r in both if r.get("fpts_act") is not None and r.get("fpts_proj") is not None
                       and ((r.get("min_proj") or 0) > 0 or (r.get("min_act") or 0) > 0)], key=lambda m: -abs(m["delta"]))
        d = game_dir(dg, g["key"], root)
        out.append(dict(g, final=_final_of(fin, g["home"], g["away"]), teams=teams, misses=[m for m in miss if abs(m["delta"]) >= BIG_MISS][:3],
                        can_open=g["record"] is not None, drawn=(d / "base.json").exists(), what_if=(d / "what_if.json").exists(), **_errors(both)))
    return out


# ----------------------------------------------------------------------------- the edits
def clean_edits(edits: Optional[dict]) -> dict:
    """The edits in their stored shape, the empty parts left out: {} = the tip record as it was."""
    e = edits or {}
    out: dict = {}
    mins = {}
    for k, v in (e.get("minutes") or {}).items():
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x == x and x >= 0:
            mins[normalize_name(str(k))] = round(min(x, 48.0 + 25.0), 1)
    if mins:
        out["minutes"] = mins
    st = {normalize_name(str(k)): bool(v) for k, v in (e.get("starters") or {}).items() if v is not None}
    if st:
        out["starters"] = st
    for k in ("spread_home", "total"):
        try:
            if e.get(k) is not None and str(e.get(k)).strip() != "":
                out[k] = float(e[k])
        except (TypeError, ValueError):
            pass
    return out


def load_edits(dg: int, key: str, root: Optional[Path] = None) -> dict:
    f = game_dir(dg, key, root) / "edits.json"
    try:
        return clean_edits(json.loads(f.read_text())) if f.exists() else {}
    except (OSError, ValueError):
        return {}


def save_edits(dg: int, key: str, edits: Optional[dict], root: Optional[Path] = None) -> dict:
    """Write the game's edits; none = the files of the what-if go (the unedited draw stays)."""
    d = game_dir(dg, key, root)
    e = clean_edits(edits)
    if not e:
        for f in ("edits.json", "what_if.json"):
            (d / f).unlink(missing_ok=True)
        return {}
    d.mkdir(parents=True, exist_ok=True)
    (d / "edits.json").write_text(json.dumps(e, indent=1, sort_keys=True))
    return e


def reset(dg: int, key: str, root: Optional[Path] = None, everything: bool = False) -> None:
    """Drop the game's edits and its what-if; `everything` also drops the unedited draw."""
    if everything:
        shutil.rmtree(game_dir(dg, key, root), ignore_errors=True)
    else:
        save_edits(dg, key, None, root)


# ----------------------------------------------------------------------------- the inputs
def _day_state(session: Session, day: date) -> dict:
    """What every game of the day shares (about 13 s to build, kept per process): the history before the day, each team's pace, the WorldContext cut at the day."""
    from ivan.sports.nba.gamesim import possessions as X
    from ivan.sports.nba.gamesim import script as SC
    from ivan.sports.nba.projection import worlds as PW
    key = (day, PW._history_key(session))
    with _LOCK:
        hit = _DAY.get(key)
    if hit is not None:
        return hit
    hist, state = RR.history_as_of(session, day)
    ev = X.load_events()
    if len(ev):
        ev["game_date"] = pd.to_datetime(ev.game_date).dt.date
    seg = X.team_segments(ev[ev.game_date < day]) if len(ev) else pd.DataFrame()
    teams = W.load_teams()
    out = dict(hist=hist, state=state, paces=SC.team_paces(seg) if len(seg) else {}, league_pace=float(SC.game_pace(seg).poss.mean()) if len(seg) else float(teams["league_pace"]),
               ctx=W.WorldContext.build(session, as_of=day))
    with _LOCK:
        for k in [k for k in _DAY if k != key][:-1]:               # two days at most
            _DAY.pop(k, None)
        _DAY[key] = out
    return out


def _tip_state(session: Session, dg: int, tip: datetime, root: Optional[Path] = None) -> dict:
    """The tip record read once per process: every pool player with a team (the soft pool filled for a record without it), the lines, the rates."""
    key = (int(dg), tip, str(root or ""))
    with _LOCK:
        hit = _TIP.get(key)
    if hit is not None:
        return hit
    rec = T.read_tip(dg, tip, root)
    if rec is None:
        raise FileNotFoundError(f"slate {dg} has no tip record at {tip:%Y-%m-%d %H:%M}")
    ds = _day_state(session, tip.date())
    RR.soft_pool(rec["players"], ds["state"])
    players = [p for p in rec["players"] if p.get("team") and p.get("opponent")]
    out = dict(players=players, lines=RR.lines_from_record(rec["game_lines"], players),
               rates=RR.rates_as_of(session, tip.date(), players, rec["odds"], history=(ds["hist"], ds["state"])), meta=rec["meta"])
    with _LOCK:
        for k in [k for k in _TIP if k[0] != int(dg)]:              # one slate's tips at a time
            _TIP.pop(k, None)
        _TIP[key] = out
    return out


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if x != x else x


def in_world(p: dict) -> bool:
    """Whether the tip record's row is a player the World draws without an edit: projected, or a fringe player."""
    return p.get("fpts_mean") is not None or _num(p.get("fringe_p")) > 0


def build_game(g: dict, players: List[dict], rates: Dict[str, dict], paces: Dict[str, float], league_pace: float, edits: Optional[dict] = None,
               starter_calls: bool = True) -> W.GameIn:
    """
    The World's inputs for one game from the tip record's rows, with the edits applied (see the module doc). `g` is a
    slate_games row, `players` the record's rows, `rates` {record name: per-minute rates}. Pure: no session, no files.
    """
    from ivan.sports.nba.projection.rotation import default_spread
    e = clean_edits(edits)
    typed = e.get("minutes") or {}
    calls = (e.get("starters") or {}) if starter_calls else {}
    game = W.GameIn(home=g["home"], away=g["away"], spread_home=e.get("spread_home", g.get("spread_home")), total=e.get("total", g.get("total")),
                    pace_home=float(paces.get(g["home"], league_pace)), pace_away=float(paces.get(g["away"], league_pace)))
    for team in (g["home"], g["away"]):
        rows = [p for p in players if p.get("team") == team]
        names = {id(p): normalize_name(str(p.get("name") or "")) for p in rows}
        team_typed = {names[id(p)]: typed[names[id(p)]] for p in rows if names[id(p)] in typed}
        free = [p for p in rows if names[id(p)] not in typed and in_world(p) and _num(p.get("minutes_proj")) > 0]
        scale = 1.0
        if team_typed and free:
            scale = max(0.0, TEAM_MINUTES - sum(team_typed.values())) / sum(_num(p.get("minutes_proj")) for p in free)
        out = []
        for p in rows:
            nm = names[id(p)]
            cells = p.get("rotation_cells")
            cells = [int(x) for x in cells] if isinstance(cells, list) and len(cells) == W.MINUTES else None
            if nm in calls:
                cells = list(cells) if cells is not None else [0] * W.MINUTES
                cells[0] = 1 if calls[nm] else 0
            if nm in typed:
                mins = float(typed[nm])
                if mins <= 0:
                    continue                                            # typed out of the game
                extra = dict(p_out=0.0, p_cut=0.0, fringe_p=0.0, fringe_minutes=0.0)
            elif in_world(p):
                mins = min(MINUTES_CAP, _num(p.get("minutes_proj")) * scale) if team_typed else _num(p.get("minutes_proj"))
                extra = dict(p_out=_num(p.get("p_out")), p_cut=_num(p.get("p_cut")), fringe_p=_num(p.get("fringe_p")), fringe_minutes=_num(p.get("fringe_minutes")))
            else:
                continue
            out.append(W.PlayerIn(pid=int(p["id"]), name=nm, team=team, minutes=mins, rates=dict(rates.get(str(p["name"])) or rates.get(nm) or W.LEAGUE_RATES), cells=cells,
                                  boundary_sd=float(p["boundary_sd"]) if p.get("boundary_sd") else default_spread(mins), fpts_mean=_num(p.get("fpts_mean")),
                                  fpts_std=_num(p.get("fpts_std")), minutes_sd=float(p["minutes_sd"]) if p.get("minutes_sd") else None, **extra))
        game.players[team] = out
    return game


def _record(sim, game: W.GameIn, meta: dict) -> dict:
    players = {}
    by_name = {p.name: p for t in game.players.values() for p in t}
    for team, fr in sim.frames.items():
        for nm, r in fr.iterrows():
            d = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
            d.update(team=team, minutes_in=float(by_name[nm].minutes) if nm in by_name else None)
            players[str(nm)] = d
    return dict(meta=dict(meta, game=sim.game, n=int(sim.n), seed=int(sim.seed)), players=players)


def simulate(session: Session, dg: int, key: str, n: Optional[int] = None, edits: Optional[dict] = None, root: Optional[Path] = None, write: bool = True,
             log=lambda m: None) -> dict:
    """
    Draw the game. `edits` None reads the game's stored edits; {} draws the tip record as it was. Without edits the draw is
    the game's base.json; with them it is what_if.json, the edits are stored, and the base is drawn first when none is on file
    or the one on file is another commit's or another number of worlds (so the two differ by the edits alone). Returns
    {base, what_if (None without edits)}.
    """
    from ivan.core.gitinfo import git_state
    from ivan.sports.nba.projection import worlds as PW
    g = find_game(dg, key, root)
    if g is None:
        raise KeyError(f"slate {dg} has no game {key} on its tip records")
    key = g["key"]
    e = load_edits(dg, key, root) if edits is None else clean_edits(edits)
    n = max(N_MIN, min(N_MAX, int(n or PW.n_worlds())))
    ts = _tip_state(session, dg, g["record"], root)
    ds = _day_state(session, g["record"].date())
    ctx = ds["ctx"]
    gs = git_state()
    d = game_dir(dg, key, root)
    meta = dict(draft_group=int(dg), key=key, home=g["home"], away=g["away"], tip=g["tip"].isoformat(timespec="minutes") if g["tip"] else None,
                record=g["record"].isoformat(timespec="minutes"), day=g["record"].date().isoformat(), commit=gs.get("commit"), dirty=gs.get("dirty"),
                rotation=str(ctx.params.get("rotation")), written_at=datetime.now().isoformat(timespec="seconds"))
    stints = str(ctx.params.get("rotation")) == "stints"

    def draw(ed: dict) -> dict:
        game = build_game(g, ts["players"], ts["rates"], ds["paces"], ds["league_pace"], ed, starter_calls=stints)
        log(f"{g['away']} @ {g['home']}: drawing {n} worlds as of {meta['day']}" + (" with the edits" if ed else ""))
        return _record(PW.simulate_game(session, ctx, game, n), game, dict(meta, edits=ed))

    base = load_draw(dg, key, "base", root)
    if base is None or not e or (base["meta"].get("commit"), base["meta"].get("dirty"), base["meta"].get("n")) != (meta["commit"], meta["dirty"], n):
        base = draw({})
        if write:
            d.mkdir(parents=True, exist_ok=True)
            (d / "base.json").write_text(json.dumps(base, default=str))
    what_if = None
    if e:
        what_if = draw(e)
        if write:
            save_edits(dg, key, e, root)
            (d / "what_if.json").write_text(json.dumps(what_if, default=str))
    elif write:
        save_edits(dg, key, None, root)
    return dict(base=base, what_if=what_if, game=g, edits=e)


def load_draw(dg: int, key: str, which: str = "base", root: Optional[Path] = None) -> Optional[dict]:
    f = game_dir(dg, key, root) / f"{which}.json"
    try:
        return json.loads(f.read_text()) if f.exists() else None
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- the actuals
def box(session: Session, day: date, teams) -> Dict[str, dict]:
    """{player: his box score line} for the day's games of `teams`: team, minutes, fpts, the seven stats."""
    a = RR.actuals(session, day)
    a = a[a.team.isin(list(teams))]
    out = {}
    for r in a.itertuples():
        out[str(r.player)] = dict(team=str(r.team), minutes=_num(r.minutes), fpts=_num(r.dk_fpts), **{s: _num(getattr(r, s)) for s, _ in STATS})
    return out


def floor(session: Session, team: str, day: date) -> dict:
    """The team's game as played: {played: {player: 48 bools}, starters: set, minutes: {player: minutes with overtime}}; empty without stints."""
    from ivan.sports.nba.projection import rotation as ROT
    gid = session.execute(text("select game_id from nba_stint where team = :t and game_date = :d limit 1").bindparams(t=team, d=day.isoformat())).scalar()
    if gid is None:
        return dict(played={}, starters=set(), minutes={})
    last = ROT.last_game_grid(session, team, game_id=str(gid))
    if last is None:
        return dict(played={}, starters=set(), minutes={})
    st = session.execute(text("select distinct player_name from nba_stint where game_id = :g and team = :t and starter = 1").bindparams(g=str(gid), t=team)).all()
    grid = last["grid"]
    return dict(played={normalize_name(str(n)): [bool(x) for x in grid.loc[n].values] for n in grid.index}, starters={normalize_name(str(r[0])) for r in st},
                minutes={normalize_name(str(k)): float(v) for k, v in last["minutes"].items()})


def actual_edits(session: Session, dg: int, key: str, root: Optional[Path] = None) -> dict:
    """The edits that set the game to what happened: every pool player of both teams at the minutes he played (0 for one who sat),
    the five who started. The line is left alone."""
    g = find_game(dg, key, root)
    if g is None:
        raise KeyError(f"slate {dg} has no game {key} on its tip records")
    day = g["record"].date()
    b = box(session, day, (g["home"], g["away"]))
    players = [p for p in json.loads((T.tip_dir(dg, g["record"], root) / "players.json").read_text()) if p.get("team") in (g["home"], g["away"])]
    mins, starters = {}, {}
    for team in (g["home"], g["away"]):
        fl = floor(session, team, day)
        for p in (q for q in players if q.get("team") == team):
            nm = normalize_name(str(p.get("name") or ""))
            played = _num((b.get(nm) or {}).get("minutes")) if (b.get(nm) or {}).get("team") == team else 0.0
            if played > 0 or _num(p.get("minutes_proj")) > 0:
                mins[nm] = round(played, 1)
            if fl["starters"] and played > 0:
                was = str(p.get("starter") or "") in ("confirmed", "expected")
                if (nm in fl["starters"]) != was:
                    starters[nm] = nm in fl["starters"]
    return clean_edits(dict(minutes=mins, starters=starters))


# ----------------------------------------------------------------------------- the page
def _pct_shares(shares: Optional[list], x: Optional[float]) -> Optional[float]:
    """Where a whole-valued actual falls in a drawn distribution given as the share of worlds at each whole value: P(below) + half of P(at)."""
    if not shares or x is None:
        return None
    k = int(round(x))
    below = float(sum(shares[:max(0, k)]))
    at = float(shares[k]) if 0 <= k < len(shares) else 0.0
    return min(1.0, below + 0.5 * at)


def _pct_quantiles(q: Optional[list], x: Optional[float]) -> Optional[float]:
    if not isinstance(q, list) or len(q) < 2 or x is None:
        return None
    a = np.asarray(q, dtype=float)
    return float((np.searchsorted(a, x, side="left") + np.searchsorted(a, x, side="right")) / (2.0 * len(a)))


def view(session: Session, dg: int, key: str, root: Optional[Path] = None) -> dict:
    """What the game's page shows (the module doc). `shown` names the draw the grid and the percentiles are read from: the what-if when
    there are edits with a draw, else the base, else None (not drawn yet: the published numbers and the actuals only)."""
    g = find_game(dg, key, root)
    if g is None:
        raise KeyError(f"slate {dg} has no game {key} on its tip records")
    key = g["key"]
    day = g["record"].date()
    edits = load_edits(dg, key, root)
    base = load_draw(dg, key, "base", root)
    what_if = load_draw(dg, key, "what_if", root) if edits else None
    shown = what_if or base
    b = box(session, day, (g["home"], g["away"]))
    fin = _final_of(finals(session, day), g["home"], g["away"])
    players = [p for p in json.loads((T.tip_dir(dg, g["record"], root) / "players.json").read_text()) if p.get("team") in (g["home"], g["away"])]
    typed = edits.get("minutes") or {}
    calls = edits.get("starters") or {}
    teams = []
    for team in (g["away"], g["home"]):
        fl = floor(session, team, day)
        rows, hidden = [], 0
        pool = set()
        for p in (q for q in players if q.get("team") == team):
            nm = normalize_name(str(p.get("name") or ""))
            pool.add(nm)
            a = b.get(nm) if (b.get(nm) or {}).get("team") == team else None
            sb = (base or {}).get("players", {}).get(nm)
            sw = (what_if or {}).get("players", {}).get(nm)
            ss = sw if what_if else sb
            min_pub = _num(p.get("minutes_proj")) if p.get("fpts_mean") is not None else None
            if not ((min_pub or 0) > 0 or (a and a["minutes"] > 0) or nm in typed or ss):
                hidden += 1
                continue
            pub_start = str(p.get("starter") or "") in ("confirmed", "expected")
            stats = [dict(stat=s, label=lab, drawn=None if ss is None else ss.get(f"sim_{s}"), base=None if sb is None else sb.get(f"sim_{s}"),
                          act=None if a is None else a[s], pct=_pct_shares(((ss or {}).get("stat_dist") or {}).get(s), None if a is None else a[s])) for s, lab in STATS]
            fa = None if a is None else a["fpts"]
            fd = None if ss is None else ss.get("fpts_mean")
            rows.append(dict(
                name=nm, pos=str(p.get("position") or ""), salary=p.get("salary"), status=str(p.get("status") or "").upper().replace("NONE", ""),
                starter_pub=pub_start, starter=calls.get(nm, pub_start), starter_edit=nm in calls, started=nm in fl["starters"] if fl["starters"] else None,
                min_pub=min_pub, min_set=typed.get(nm), min_in=None if ss is None else ss.get("minutes_in"), min_base=None if sb is None else sb.get("minutes_sim"),
                min_drawn=None if ss is None else ss.get("minutes_sim"), min_p10=None if ss is None else ss.get("minutes_p10"), min_p90=None if ss is None else ss.get("minutes_p90"),
                plays=None if ss is None else 1.0 - float(ss.get("p_zero") or 0.0), min_act=None if a is None else a["minutes"],
                min_pct=_pct_shares((ss or {}).get("minutes_dist"), None if a is None else a["minutes"]),
                fpts_pub=p.get("fpts_mean"), fpts_base=None if sb is None else sb.get("fpts_mean"), fpts_what_if=None if sw is None else sw.get("fpts_mean"),
                fpts_drawn=fd, fpts_sd=None if ss is None else ss.get("fpts_std"), fpts_act=fa, delta=None if fa is None or fd is None else fa - fd,
                delta_pub=None if fa is None or p.get("fpts_mean") is None else fa - float(p["fpts_mean"]),
                pct=_pct_quantiles((ss or {}).get("quantiles"), fa) if a is not None and a["minutes"] > 0 else None,
                played=bool(a and a["minutes"] > 0), stats=stats, cover=(ss or {}).get("cover"), floor=fl["played"].get(nm), drawn=ss is not None))
        rows.sort(key=lambda r: (-(r["min_pub"] or 0), -(r["min_act"] or 0)))
        outside = sorted([dict(name=n, minutes=v["minutes"], fpts=v["fpts"]) for n, v in b.items() if v["team"] == team and n not in pool and v["minutes"] > 0], key=lambda r: -r["minutes"])
        tot = lambda k: sum(r[k] or 0.0 for r in rows)                                          # noqa: E731
        drawn = [r for r in rows if r["drawn"] and ((r["min_act"] or 0) > 0 or (r["min_drawn"] or 0) > 0)] if b else []
        pub = _errors([dict(min_proj=r["min_pub"], min_act=r["min_act"], fpts_proj=r["fpts_pub"], fpts_act=r["fpts_act"]) for r in rows]) if b else {}
        on5 = None
        if any(r["cover"] for r in rows):
            on5 = [round(sum((r["cover"] or [0.0] * W.MINUTES)[m] for r in rows), 2) for m in range(W.MINUTES)]
        teams.append(dict(team=team, home=team == g["home"], rows=rows, hidden=hidden, outside=outside, on5=on5, has_floor=bool(fl["played"]),
                          totals=dict(min_pub=tot("min_pub"), min_in=tot("min_in"), min_drawn=tot("min_drawn"), min_act=tot("min_act"), fpts_pub=tot("fpts_pub"),
                                      fpts_base=tot("fpts_base"), fpts_what_if=tot("fpts_what_if"), fpts_drawn=tot("fpts_drawn"), fpts_act=tot("fpts_act")),
                          min_mae=float(np.mean([abs((r["min_drawn"] or 0.0) - (r["min_act"] or 0.0)) for r in drawn])) if drawn else None,
                          fpts_mae=float(np.mean([abs((r["fpts_drawn"] or 0.0) - (r["fpts_act"] or 0.0)) for r in drawn])) if drawn else None,
                          min_mae_pub=pub.get("min_mae"), fpts_mae_pub=pub.get("fpts_mae")))
    games = slate_games(dg, root)
    i = next((k for k, x in enumerate(games) if x["key"] == key), 0)
    return dict(draft_group=int(dg), game=g, key=key, day=day, final=fin, edits=edits, n_edits=len(typed) + len(calls) + sum(1 for k in ("spread_home", "total") if k in edits),
                base=None if base is None else base["meta"], what_if=None if what_if is None else what_if["meta"], shown=None if shown is None else shown["meta"],
                which="what_if" if what_if else ("base" if base else None), teams=teams, has_box=bool(b), games=games,
                prev=games[i - 1]["key"] if i > 0 else None, next=games[i + 1]["key"] if i + 1 < len(games) else None)


# ----------------------------------------------------------------------------- one player
def _trim(p: list) -> list:
    p = [float(x) for x in p]
    tail = 0.0
    while len(p) > 1 and tail + p[-1] < 0.005:
        tail += p.pop()
    return p


def chart(shares: Optional[list], other: Optional[list] = None, actual: Optional[float] = None, mark: Optional[float] = None) -> Optional[dict]:
    """
    One distribution as bars: `shares` the share of worlds at each whole value (the draw shown), `other` the unedited draw's when
    the shown one carries edits, `actual` what happened, `mark` the published number. Whole values, binned in equal steps past
    CHART_BINS. {bins: [{lo, hi, label, a, b, hit}], ymax, mean, p10, p50, p90, pct, actual_at, mark_at (positions in bins from the left)}.
    """
    if not shares:
        return None
    a = _trim(shares); b = _trim(other) if other else []
    top = max(len(a) - 1, len(b) - 1, int(np.ceil(actual)) if actual is not None else 0, int(np.ceil(mark)) if mark is not None else 0, 1)
    step = int(np.ceil((top + 1) / CHART_BINS))
    bins = []
    k = None if actual is None else int(round(actual))
    for i in range(int(np.ceil((top + 1) / step))):
        lo, hi = i * step, (i + 1) * step - 1
        bins.append(dict(lo=lo, hi=hi, label=str(lo) if step == 1 else f"{lo}-{hi}", a=float(sum(a[lo:hi + 1])), b=float(sum(b[lo:hi + 1])) if b else None,
                         hit=k is not None and lo <= k <= hi))
    full = [float(x) for x in shares]
    cum = np.cumsum(full); tot = float(cum[-1]) or 1.0
    q = lambda t: int(np.searchsorted(cum / tot, t, side="left"))                               # noqa: E731
    ymax = max(0.1, float(np.ceil(10 * max([max(x["a"], x["b"] or 0.0) for x in bins]) - 1e-9)) / 10)
    return dict(bins=bins, step=step, ymax=ymax, mean=float(sum(i * x for i, x in enumerate(full)) / tot), p10=q(0.1), p50=q(0.5), p90=q(0.9),
                pct=_pct_shares(full, actual), actual_at=None if actual is None else (float(actual) + 0.5) / step, mark_at=None if mark is None else (float(mark) + 0.5) / step,
                other_mean=float(sum(i * x for i, x in enumerate(other)) / (sum(other) or 1.0)) if other else None)


def player_view(session: Session, dg: int, key: str, name: str, root: Optional[Path] = None) -> dict:
    """One player of the game: for his fantasy points, minutes and each stat, the drawn distribution with the actual marked; the
    published minutes and fantasy points marked where there is one; the unedited draw beside the what-if when there are edits."""
    from ivan.sports.nba.desk.rows import quantile_shares
    g = find_game(dg, key, root)
    if g is None:
        raise KeyError(f"slate {dg} has no game {key} on its tip records")
    key = g["key"]; name = normalize_name(name)
    edits = load_edits(dg, key, root)
    base = load_draw(dg, key, "base", root)
    what_if = load_draw(dg, key, "what_if", root) if edits else None
    sb = (base or {}).get("players", {}).get(name)
    sw = (what_if or {}).get("players", {}).get(name)
    ss = sw if what_if else sb
    rec = next((p for p in json.loads((T.tip_dir(dg, g["record"], root) / "players.json").read_text()) if normalize_name(str(p.get("name") or "")) == name), None)
    a = box(session, g["record"].date(), (g["home"], g["away"])).get(name)

    def dist(s: Optional[dict], stat: str) -> Optional[list]:
        if s is None:
            return None
        if stat == "fpts":
            return quantile_shares(s.get("quantiles"))
        return s.get("minutes_dist") if stat == "minutes" else (s.get("stat_dist") or {}).get(stat)

    charts = []
    for stat, label in CHART_STATS:
        act = None if a is None else a[stat]
        pub = None if rec is None else (rec.get("fpts_mean") if stat == "fpts" else rec.get("minutes_proj") if stat == "minutes" and rec.get("fpts_mean") is not None else None)
        c = chart(dist(ss, stat), dist(sb, stat) if what_if else None, act, pub)
        charts.append(dict(stat=stat, label=label, chart=c, actual=act, published=pub))
    return dict(name=name, team=(rec or {}).get("team"), pos=(rec or {}).get("position"), salary=(rec or {}).get("salary"), key=key, charts=charts,
                drawn=ss is not None, what_if=what_if is not None, played=bool(a and a["minutes"] > 0), in_box=a is not None,
                minutes_in=None if ss is None else ss.get("minutes_in"), n=None if ss is None else ((what_if or base)["meta"]).get("n"))
