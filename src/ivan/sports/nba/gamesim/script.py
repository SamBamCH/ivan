# src/ivan/sports/nba/gamesim/script.py
"""
Layer A: the game script. Possessions and points per six-minute segment for both teams, around the books' lines.

A game's expected possessions come from the two teams' trailing pace (possessions per 48 minutes, shrunk toward the
league); the books' total and spread then fix each team's expected points per possession:

    e_home = (total - spread_home) / (2 P),   e_away = (total + spread_home) / (2 P)      (spread_home < 0: home favoured)

A world draws the game's pace (one log-normal factor on P, sd SIGMA_PACE), each team's efficiency for the night (a shared
shift on e, sd SIGMA_EFF: the 'shooting night' every player on the team has in common), and then segment by segment the
possessions (the league's segment shape of regulation, Poisson around the share) and the points (Normal around
possessions x efficiency with the per-possession scoring variance, SIGMA_PTS per possession). The margin path decides
garbage time (pbp.garbage_threshold on the time left, read at the end of segment 6 and inside segment 7 through a
Brownian bridge, the way rotation.sample_worlds does at minute level) and overtime (a tie after regulation plays a
five-minute segment, again if tied).

In the full World the players' makes are the points (layer D) and this module supplies the latents: P, the segment
shape, each team's efficiency target and shooting-night shift. On its own it is the layer A check: with every lined
game's P and e plugged in, the drawn finals should cover the real finals (PIT of the total and the margin flat, the
overtime rate and the garbage-time rate right) and the trailing pace should predict the game's possessions within its
stated spread.

fit(seg, lines)       SIGMA_PACE, SIGMA_EFF, SIGMA_PTS, the pace regression and the segment shape from the lined games
                      with events, written to data/nba/models/gamesim/script.json with the window and the commit
draw(n, P, e_h, e_a)  the worlds
check(seg, lines)     the coverage table
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.sports.nba.paths import INPUT_DIR, MODELS_DIR
from ivan.sports.nba.pbp import garbage_threshold

MODEL_JSON = MODELS_DIR / "gamesim" / "script.json"
HIST_LINES_CSV = INPUT_DIR / "historical_lines.csv"
SEGMENTS = 8
SEG_MIN = 6.0
REG_MIN = 48.0
OT_MIN = 5.0
PACE_GAMES = 10            # trailing games in a team's pace
PACE_HALFLIFE = 5.0
PACE_PRIOR_GAMES = 6.0     # pseudo-games toward the league pace
TIE_WINDOW = 3             # a buzzer margin this close can be tied by the endgame
DEFAULTS = dict(league_pace=99.0, pace_a=0.0, pace_b=1.0, sigma_pace=0.045, sigma_seg=1.6, sigma_eff=0.0, sigma_pts=1.14, kappa=0.004, lambda_pace=0.0,
                q_tie=0.0, segment_shape=[0.126, 0.129, 0.122, 0.13, 0.123, 0.128, 0.117, 0.126], p_ot=0.055)


# ----------------------------------------------------------------------------- lines
def load_lines(session: Optional[Session], seg: pd.DataFrame) -> pd.DataFrame:
    """
    game_id, spread_home, total for every game in `seg` (team_segments) that has a line: the historical CSV (one row per
    game day and home/away pair) and this season's game_odds consensus, matched to the game by its date and teams.
    """
    games = seg[["game_id", "game_date", "team", "opp", "home"]].drop_duplicates()
    home = games[games.home.astype(bool)].rename(columns={"team": "home_abbr", "opp": "away_abbr"})[["game_id", "game_date", "home_abbr", "away_abbr"]]
    home = home.assign(game_date=pd.to_datetime(home.game_date).dt.date)
    frames = []
    if Path(HIST_LINES_CSV).exists():
        h = pd.read_csv(HIST_LINES_CSV)
        h["game_date"] = pd.to_datetime(h.date).dt.date
        frames.append(h[["game_date", "home_abbr", "away_abbr", "spread_home", "total"]])
    if session is not None:
        try:
            from ivan.sports.nba.projection.lines import load_consensus
            c = load_consensus(session)
            if len(c):
                when = pd.to_datetime(c.commence_time, utc=True, errors="coerce").dt.tz_convert("America/New_York")
                c = c.assign(game_date=when.dt.date)
                frames.append(c[["game_date", "home_abbr", "away_abbr", "spread_home", "total"]])
        except Exception:   # noqa: BLE001 - a session without the table
            pass
    if not frames:
        return pd.DataFrame(columns=["game_id", "spread_home", "total"])
    lines = pd.concat(frames, ignore_index=True).dropna(subset=["spread_home", "total"])
    lines = lines.groupby(["game_date", "home_abbr", "away_abbr"], as_index=False)[["spread_home", "total"]].median()
    out = home.merge(lines, on=["game_date", "home_abbr", "away_abbr"], how="inner")
    return out[["game_id", "spread_home", "total"]]


# ----------------------------------------------------------------------------- pace
def game_pace(seg: pd.DataFrame) -> pd.DataFrame:
    """Per game and team: regulation possessions, points, the final (with overtime), overtime flag, home, opp, date."""
    reg = seg[seg.segment < SEGMENTS]
    g = reg.groupby(["game_id", "game_date", "team", "opp", "home"], as_index=False).agg(poss=("poss", "sum"), pts_reg=("pts", "sum"))
    fin = seg.groupby(["game_id", "team"], as_index=False).agg(pts=("pts", "sum"), overtime=("segment", lambda s: int(s.max() >= SEGMENTS)))
    g = g.merge(fin, on=["game_id", "team"])
    g["game_date"] = pd.to_datetime(g.game_date).dt.date
    return g.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def trailing_pace(gp: pd.DataFrame, league: float, n_games: int = PACE_GAMES, halflife: float = PACE_HALFLIFE,
                  prior_games: float = PACE_PRIOR_GAMES) -> pd.Series:
    """For every row of game_pace: the team's pace before that game (its last n games, recency weighted, shrunk toward the league); NaN with no history."""
    out = pd.Series(np.nan, index=gp.index)
    for team, d in gp.groupby("team"):
        d = d.sort_values(["game_date", "game_id"])
        vals = d.poss.to_numpy(dtype=float)
        for k in range(len(d)):
            prev = vals[max(0, k - n_games):k]
            if len(prev) == 0:
                out.loc[d.index[k]] = league
                continue
            w = 0.5 ** ((len(prev) - 1 - np.arange(len(prev))) / halflife)
            m = float((w * prev).sum() / w.sum()); n_eff = float(w.sum())
            out.loc[d.index[k]] = (n_eff * m + prior_games * league) / (n_eff + prior_games)
    return out


def expected_possessions(pace_home: float, pace_away: float, params: dict) -> float:
    return float(params["pace_a"] + params["pace_b"] * 0.5 * (pace_home + pace_away))


def efficiencies(total: float, spread_home: float, P: float) -> Tuple[float, float]:
    """(home, away) expected points per possession from the total and the home spread (negative = home favoured)."""
    return (total - spread_home) / (2.0 * P), (total + spread_home) / (2.0 * P)


# ----------------------------------------------------------------------------- the draw
def draw(n: int, P: float, e_home: float, e_away: float, params: dict, rng: np.random.Generator) -> dict:
    """
    n worlds of one game. Returns poss (n, 8) possessions per segment (both teams share the pace), pts_home / pts_away (n, 8),
    eff_home / eff_away (n,) the night's efficiency, margin (n,) home minus away after overtime, total (n,), overtime (n,)
    periods played, garbage_at (n,) the minute garbage time began or -1, path (n, 8) home margin at each segment's end.
    """
    shape = np.asarray(params["segment_shape"], dtype=float); shape = shape / shape.sum()
    kappa = float(params.get("kappa", 0.0)); lam = float(params.get("lambda_pace", 0.0)); q_tie = float(params.get("q_tie", 0.0))
    zp = rng.normal(0.0, params["sigma_pace"], n)
    pace = P * np.exp(zp - 0.5 * params["sigma_pace"] ** 2)
    # a fast night is a less efficient night (lambda_pace: the share of the pace deviation taken back out of points per possession)
    eff_h = e_home * (1.0 + rng.normal(0.0, params["sigma_eff"], n) - lam * zp)
    eff_a = e_away * (1.0 + rng.normal(0.0, params["sigma_eff"], n) - lam * zp)
    # possessions per segment: the clock fixes them within a possession or two of pace x shape (SIGMA_SEG, measured), not a Poisson count
    poss = np.maximum(1.0, np.rint(pace[:, None] * shape[None, :] + rng.standard_normal((n, SEGMENTS)) * params["sigma_seg"]))
    sd = params["sigma_pts"] * np.sqrt(np.maximum(poss, 1.0))
    pts_h = np.zeros((n, SEGMENTS)); pts_a = np.zeros((n, SEGMENTS)); path = np.zeros((n, SEGMENTS))
    lead = np.zeros(n)
    for s in range(SEGMENTS):
        # the lead pulls the efficiencies back (the leader eases off, the trailer presses): kappa points per possession per point of lead
        eh = eff_h - kappa * lead; ea = eff_a + kappa * lead
        pts_h[:, s] = np.maximum(0.0, np.rint(poss[:, s] * eh + rng.standard_normal(n) * sd[:, s]))
        pts_a[:, s] = np.maximum(0.0, np.rint(poss[:, s] * ea + rng.standard_normal(n) * sd[:, s]))
        lead = lead + pts_h[:, s] - pts_a[:, s]; path[:, s] = lead
    margin = path[:, -1].copy(); total = pts_h.sum(axis=1) + pts_a.sum(axis=1)
    # garbage time: the margin at the end of segment 6 (minute 42) against the threshold for six minutes left, then a bridge
    # from there to the final through the last segment, minute by minute, against the threshold for the time left
    garbage_at = np.full(n, -1)
    m42 = path[:, 6]
    hit42 = np.abs(m42) >= garbage_threshold(6 * 60.0)
    garbage_at[hit42] = 42
    rest = np.flatnonzero(~hit42)
    if len(rest):
        steps = np.arange(43, 48)
        frac = (steps - 42) / 6.0
        bridge_sd = params["sigma_pts"] * np.sqrt(2.0 * poss[rest, 7][:, None] * frac[None, :] * (1 - frac)[None, :])
        mid = m42[rest, None] + (margin[rest] - m42[rest])[:, None] * frac[None, :] + rng.standard_normal((len(rest), len(steps))) * bridge_sd
        thr = np.array([garbage_threshold((48 - m) * 60.0) for m in steps])
        hit = np.abs(mid) >= thr[None, :]
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        ok = first >= 0
        garbage_at[rest[ok]] = steps[first[ok]]
    # the endgame ties a close game more often than possession noise says (the trailer fouls, the leader runs the clock):
    # a world within TIE_WINDOW points at the buzzer is tied with probability q_tie (fitted to the real overtime rate)
    if q_tie > 0:
        close = (np.abs(margin) <= TIE_WINDOW) & (margin != 0) & (rng.random(n) < q_tie)
        total[close] -= np.abs(margin[close]); margin[close] = 0.0
    # overtime: tied after regulation, five more minutes, again while tied
    overtime = np.zeros(n, dtype=int)
    tied = margin == 0
    guard = 0
    while tied.any() and guard < 4:
        idx = np.flatnonzero(tied)
        p_ot = pace[idx] * OT_MIN / REG_MIN
        po = rng.poisson(p_ot).astype(float); sdo = params["sigma_pts"] * np.sqrt(np.maximum(po, 1.0))
        h = np.maximum(0.0, np.rint(po * eff_h[idx] + rng.normal(0.0, 1.0, len(idx)) * sdo))
        a = np.maximum(0.0, np.rint(po * eff_a[idx] + rng.normal(0.0, 1.0, len(idx)) * sdo))
        margin[idx] += h - a; total[idx] += h + a; overtime[idx] += 1
        tied = margin == 0; guard += 1
    return dict(poss=poss, pts_home=pts_h, pts_away=pts_a, eff_home=eff_h, eff_away=eff_a, margin=margin, total=total,
                overtime=overtime, garbage_at=garbage_at, path=path, pace=pace)


# ----------------------------------------------------------------------------- fit and check
def _git_commit() -> Optional[str]:
    try:
        from ivan.core.gitinfo import git_state
        return git_state().get("commit") or None
    except Exception:   # noqa: BLE001
        return None


def lined_games(seg: pd.DataFrame, lines: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """One row per lined game: date, home, away, poss (the game's mean of the two teams' counts), pts_home, pts_away, overtime,
    pace_home, pace_away (trailing), P (expected possessions), e_home, e_away, garbage (did the stints' threshold trip)."""
    params = params or DEFAULTS
    gp = game_pace(seg)
    league = float(gp.poss.mean()) if len(gp) else params["league_pace"]
    gp["pace_prior"] = trailing_pace(gp, league)
    h = gp[gp.home.astype(bool)].rename(columns={"team": "home_abbr", "opp": "away_abbr", "poss": "poss_home", "pts": "pts_home", "pace_prior": "pace_home"})
    a = gp[~gp.home.astype(bool)].rename(columns={"poss": "poss_away", "pts": "pts_away", "pace_prior": "pace_away"})[["game_id", "poss_away", "pts_away", "pace_away"]]
    g = h.merge(a, on="game_id").merge(lines, on="game_id", how="inner")
    g["poss"] = 0.5 * (g.poss_home + g.poss_away)
    g["P"] = [expected_possessions(ph, pa, params) for ph, pa in zip(g.pace_home, g.pace_away)]
    eff = [efficiencies(t, s, P) for t, s, P in zip(g.total, g.spread_home, g.P)]
    g["e_home"] = [e[0] for e in eff]; g["e_away"] = [e[1] for e in eff]
    g["margin"] = g.pts_home - g.pts_away; g["final_total"] = g.pts_home + g.pts_away
    return g.reset_index(drop=True)


def segment_shape(seg: pd.DataFrame) -> list:
    reg = seg[seg.segment < SEGMENTS]
    by = reg.groupby("segment").poss.sum().reindex(range(SEGMENTS)).fillna(0.0)
    s = by / by.sum() if by.sum() > 0 else pd.Series(DEFAULTS["segment_shape"])
    return [float(x) for x in s]


def fit(seg: pd.DataFrame, lines: pd.DataFrame, write: bool = True, path: Path = MODEL_JSON) -> dict:
    """
    The pace regression (actual possessions on the mean trailing pace), its residual sd as the pace noise, the per-possession
    scoring sd from the segment points around possessions x the game's realised efficiency, and the efficiency shift that
    makes the drawn totals and margins as wide as the real ones. Each number is a moment, nothing is iterated.
    """
    params = dict(DEFAULTS)
    gp = game_pace(seg)
    params["league_pace"] = float(gp.poss.mean())
    params["segment_shape"] = segment_shape(seg)
    g = lined_games(seg, lines, params)
    if len(g) >= 30:
        x = 0.5 * (g.pace_home + g.pace_away); y = g.poss
        b, a0 = np.polyfit(x, y, 1)
        b = float(np.clip(b, 0.8, 1.6))                                   # the priors are shrunk, so the slope sits above 1; the clip keeps 221 games from over-stretching it
        a0 = float((y - b * x).mean())
        params["pace_a"], params["pace_b"] = a0, b
        resid = y - (a0 + b * x)
        params["sigma_pace"] = float(resid.std() / y.mean())
        params["pace_r2"] = float(1 - resid.var() / y.var())
    # per-possession scoring noise: segment points around possessions x the team-game's own efficiency (the shift is taken out);
    # segment count noise: segment possessions around the team-game's possessions x the segment shape
    reg = seg[seg.segment < SEGMENTS].copy()
    tg = reg.groupby(["game_id", "team"]).agg(P=("poss", "sum"), S=("pts", "sum"))
    reg = reg.merge(tg, on=["game_id", "team"])
    reg = reg[reg.P > 0]; eff = reg.S / reg.P
    z = (reg.pts - reg.poss * eff) / np.sqrt(np.maximum(reg.poss, 1.0))
    params["sigma_pts"] = float(z.std())
    shape = np.asarray(params["segment_shape"])
    params["sigma_seg"] = float((reg.poss - reg.P * shape[reg.segment.to_numpy()]).std())
    # the lead's pull-back (kappa) and the shooting-night shift (sigma_eff): the real margin around the spread is narrower than
    # independent possessions make it (the leader eases off), the real total around the line may be wider than pace and
    # possession noise make it (a shared shooting night). Each is matched to its moment by a short search on the draw itself.
    if len(g) >= 30:
        sd_margin = float((g.margin + g.spread_home).std()); sd_total = float((g.final_total - g.total).std())
        params.update(sigma_eff=0.0, lambda_pace=0.0, q_tie=0.0)
        params["kappa"] = _match(lambda k: -_drawn_sd(g, dict(params, kappa=k), "margin"), -sd_margin, lo=0.0, hi=0.02)
        # the total's spread is U-shaped in the coupling (past full decoupling the overshoot widens it again): a grid up to 1.2
        grid = np.linspace(0.0, 1.2, 13)
        sds = [_drawn_sd(g, dict(params, lambda_pace=float(l_)), "total") for l_ in grid]
        params["lambda_pace"] = float(grid[int(np.argmin(np.abs(np.asarray(sds) - sd_total)))])
        if min(sds) > sd_total:
            params["lambda_pace"] = float(grid[int(np.argmin(sds))])
        elif params["lambda_pace"] == 0.0 and sds[0] < sd_total:          # too narrow with no coupling: a shared shooting night widens it
            params["sigma_eff"] = _match(lambda s_: _drawn_sd(g, dict(params, sigma_eff=s_), "total"), sd_total, lo=0.0, hi=0.15)
        p_ot = float(g.overtime.astype(bool).mean())
        params["q_tie"] = _match(lambda q: _drawn_rate(g, dict(params, q_tie=q)), p_ot, lo=0.0, hi=0.6)
        params["real_sd_margin"] = sd_margin; params["real_sd_total"] = sd_total; params["p_ot"] = p_ot
    params["fit"] = dict(games=int(len(g)), lined_from=str(g.game_date.min()) if len(g) else None, lined_to=str(g.game_date.max()) if len(g) else None,
                         commit=_git_commit(), fitted_at=str(date.today()))
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(params, indent=1))
    return params


def _drawn_sd(g: pd.DataFrame, params: dict, what: str, n: int = 400, seed: int = 1) -> float:
    """The sd of the drawn final margin (around the spread) or total (around the line) pooled over a sample of the lined games."""
    rng = np.random.default_rng(seed)
    take = g.iloc[:: max(1, len(g) // 60)]
    devs = []
    for r in take.itertuples(index=False):
        d = draw(n, r.P, r.e_home, r.e_away, params, rng)
        devs.append(d["margin"] + r.spread_home if what == "margin" else d["total"] - r.total)
    return float(np.concatenate(devs).std())


def _drawn_rate(g: pd.DataFrame, params: dict, n: int = 400, seed: int = 1) -> float:
    """The drawn overtime rate pooled over a sample of the lined games."""
    rng = np.random.default_rng(seed)
    take = g.iloc[:: max(1, len(g) // 60)]
    return float(np.mean([(draw(n, r.P, r.e_home, r.e_away, params, rng)["overtime"] > 0).mean() for r in take.itertuples(index=False)]))


def _match(f, target: float, lo: float, hi: float, steps: int = 12) -> float:
    """The x in [lo, hi] where the increasing f(x) meets target (bisection on a noisy monotone draw; lo when f(lo) is already above target)."""
    if f(lo) >= target:
        return lo
    if f(hi) <= target:
        return hi
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_params(path: Path = MODEL_JSON) -> dict:
    p = dict(DEFAULTS)
    if Path(path).exists():
        try:
            p.update(json.loads(Path(path).read_text()))
        except Exception:   # noqa: BLE001
            pass
    return p


def pit(sample: np.ndarray, actual: float) -> float:
    s = np.sort(sample)
    return float((np.searchsorted(s, actual, side="left") + np.searchsorted(s, actual, side="right")) / (2.0 * len(s)))


def check(seg: pd.DataFrame, lines: pd.DataFrame, params: Optional[dict] = None, n: int = 2000, seed: int = 0, stints: Optional[pd.DataFrame] = None) -> dict:
    """
    Every lined game drawn n times with its own P and efficiencies: PIT of the real total and margin (flat = calibrated,
    reported as the share inside the 10-90 band and the mean), the drawn vs real overtime rate, the drawn garbage-time rate
    against the stints' (when given), and the pace check: the real possessions against P, MAE and the share inside the
    pace band. Returns {"games": frame, "summary": dict}.
    """
    params = params or load_params()
    g = lined_games(seg, lines, params)
    rng = np.random.default_rng(seed)
    rows = []
    for r in g.itertuples(index=False):
        d = draw(n, r.P, r.e_home, r.e_away, params, rng)
        rows.append(dict(game_id=r.game_id, game_date=r.game_date, home=r.home_abbr, away=r.away_abbr, P=r.P, poss=r.poss, total_line=r.total, spread_home=r.spread_home,
                         final_total=r.final_total, margin=r.margin, pit_total=pit(d["total"], r.final_total), pit_margin=pit(d["margin"], r.margin),
                         pit_poss=pit(d["pace"], r.poss), p_ot=float((d["overtime"] > 0).mean()), overtime=int(r.overtime),
                         p_garbage=float((d["garbage_at"] >= 0).mean()), mean_total=float(d["total"].mean()), mean_margin=float(d["margin"].mean())))
    games = pd.DataFrame(rows)
    def band(p):
        return float(((p >= 0.1) & (p <= 0.9)).mean())
    summary = dict(games=int(len(games)))
    if len(games):
        summary.update(pace_mae=float((games.poss - games.P).abs().mean()), pace_bias=float((games.poss - games.P).mean()),
                       pace_in_band=band(games.pit_poss), total_in_band=band(games.pit_total), total_pit_mean=float(games.pit_total.mean()),
                       margin_in_band=band(games.pit_margin), margin_pit_mean=float(games.pit_margin.mean()),
                       total_bias=float((games.final_total - games.mean_total).mean()), margin_bias=float((games.margin - games.mean_margin).mean()),
                       ot_rate_real=float(games.overtime.astype(bool).mean()), ot_rate_drawn=float(games.p_ot.mean()), garbage_rate_drawn=float(games.p_garbage.mean()))
        if stints is not None and len(stints):
            from ivan.sports.nba.projection.rotation import garbage_onsets
            st = stints[stints.game_id.isin(games.game_id)]
            on = {}
            for team, d in st.groupby("team"):
                for gid, (m, _) in garbage_onsets(d).items():
                    on[gid] = on.get(gid, False) or (m is not None)
            summary["garbage_rate_real"] = float(np.mean([on.get(g, False) for g in games.game_id]))
    return {"games": games, "summary": summary}
