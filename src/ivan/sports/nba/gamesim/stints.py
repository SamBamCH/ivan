# src/ivan/sports/nba/gamesim/stints.py
"""
Substitutions as the coach makes them: the measurements a stint-by-stint rotation model rests on
(docs/nba/notes/stint_model_2026-11-16.md; the model itself is the next step and is not here yet).

The picture: at every exchange (a stoppage where a substitution can happen) each player on the floor has a chance to
come out that rises with the time he has been on, and the man who comes in is drawn from the bench by who usually
replaces him and by how long each has rested. Both chances are read against the state of the game (period, margin,
fouls). This module builds the person-step tables from the stored stints and asks, on held-out games, which of those
terms carry information:

timeline(game_ids)        per game, from the raw feed: every dead ball (kind timeout | whistle | other) and the home
                          margin after every score; cached in data/nba/derived/gamesim/timeline_*.parquet.
on_panel / bench_panel    one row per player per STEP seconds on the floor (did his stint end in the step) / on the
                          bench (did he come in), with the time on or off, period clock, margin, fouls, role.
ladder(panel, terms)      held-out log loss of a logistic model as each term is added (the decay curve alone first).
swap_moments(stints)      every moment a team changed players: the set that left and the set that came in.
em_pairs / replacement_test   who comes in for whom: recorded pairs, single swaps only, EM over the swap sets, position
                          flow, rest, usual minutes; scored on the bench that was really available at each moment.
context_test              a player's minutes from his past games, weighted by recency alone or by how closely each
                          game's available regulars match tonight's.
cross_team                whether the two teams substitute at the same dead balls more than the dead ball explains.
star_out                  the nights one regular sat: where his minutes and points went, and how often to one man.
"""
from __future__ import annotations

from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.sports.nba.gamesim.possessions import DERIVED_DIR

STEP = 20.0                  # seconds
REG = 2880.0
N_STEPS = int(REG / STEP)
PER_STEPS = int(720 / STEP)
BREAKS = (720.0, 1440.0, 2160.0)
BANDS = [("0-12", 0.0, 12.0), ("12-20", 12.0, 20.0), ("20-28", 20.0, 28.0), ("28+", 28.0, 99.0)]
USUAL_GAMES = 10             # a player's usual minutes: his last this many played games of the season before the game
MARGIN_EDGES = [-99, -21, -13, -6, 6, 13, 21, 99]       # team margin buckets (trailing .. leading)
MARGIN_LABELS = ["-21", "-13", "-6", "close", "+6", "+13", "+21"]
LIVE_TURNOVERS = ("Bad Pass", "Lost Ball")              # a steal: play goes on
TEST_SPANS = [("2026-01-01", "2026-01-31"), ("2026-11-01", "2026-12-31")]   # held out; playoffs (May, June) are in neither half
PLAYOFF_MONTHS = (4, 5, 6)


def season_of(d) -> int:
    return d.year if d.month >= 10 else d.year - 1


def is_test(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    out = np.zeros(len(d), dtype=bool)
    for a, b in TEST_SPANS:
        out |= ((d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))).to_numpy()
    return out


def is_train(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return ~is_test(dates) & ~d.dt.month.isin(PLAYOFF_MONTHS).to_numpy()


# ----------------------------------------------------------------------------- data
def load_stints(session: Session) -> pd.DataFrame:
    q = ("select game_id, game_date, team, player_name, stint_no, start_sec, end_sec, minutes, margin_in, margin_out, to_buzzer, starter "
         "from nba_stint where substr(game_id,1,3) in ('002','004')")
    df = pd.read_sql(text(q), session.get_bind())
    df["game_date"] = pd.to_datetime(df.game_date)
    return df.sort_values(["game_date", "game_id", "team", "player_name", "start_sec"]).reset_index(drop=True)


def load_fouls(session: Session) -> pd.DataFrame:
    return pd.read_sql(text("select game_id, player_name, elapsed from nba_foul where personal = 1 order by elapsed"), session.get_bind())


def load_subs(session: Session) -> pd.DataFrame:
    return pd.read_sql(text("select game_id, team, player_out, player_in, elapsed from nba_sub"), session.get_bind())


def load_box(session: Session) -> pd.DataFrame:
    from ivan.sports.nba.projection.history import load_box_scores
    b = load_box_scores(session)
    b = b[b.game_id.astype(str).str[:3].isin(["002", "004"])].copy()
    b["game_date"] = pd.to_datetime(b.game_date); b["game_id"] = b.game_id.astype(str)
    b["season"] = b.game_date.map(season_of)
    return b.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def usual_minutes(box: pd.DataFrame, n: int = USUAL_GAMES) -> pd.DataFrame:
    """game_id, player_name -> usual (mean minutes of his last n played games of the season before this one), usual_fpts, n_prior, start_share."""
    p = box[box.played].sort_values(["player_name", "season", "game_date", "game_id"]).copy()
    g = p.groupby(["player_name", "season"])
    roll = lambda s: s.shift().rolling(n, min_periods=1).mean()   # noqa: E731
    p["usual"] = g.minutes.transform(roll); p["usual_fpts"] = g.dk_fpts.transform(roll); p["start_share"] = g.starter.transform(roll)
    p["n_prior"] = g.cumcount()
    return p[["game_id", "player_name", "team", "usual", "usual_fpts", "start_share", "n_prior"]]


def band_of(usual: float) -> str:
    if usual is None or np.isnan(usual):
        return "new"
    for name, lo, hi in BANDS:
        if lo <= usual < hi:
            return name
    return BANDS[-1][0]


def timeline(game_ids: Iterable[str], rebuild: bool = False, log=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    (stops, scores) from the raw feeds. stops: game_id, elapsed, kind (timeout | whistle (a foul or free throws) | other
    (violation, dead-ball turnover, jump ball, review)), one row per distinct second of regulation that is not a period
    boundary. scores: game_id, elapsed, home_margin (after the score), home (tricode).
    """
    from ivan.sports.nba.data_clean import pbp as P
    sp, cp = DERIVED_DIR / "timeline_stops.parquet", DERIVED_DIR / "timeline_scores.parquet"
    ids = [str(g) for g in game_ids]
    stops = pd.read_parquet(sp) if sp.exists() and not rebuild else pd.DataFrame(columns=["game_id", "elapsed", "kind"])
    scores = pd.read_parquet(cp) if cp.exists() and not rebuild else pd.DataFrame(columns=["game_id", "elapsed", "home_margin", "home"])
    have = set(scores.game_id.unique())
    new_s, new_c = [], []
    for k, gid in enumerate(g for g in ids if g not in have):
        raw = P.load_raw(gid)
        if raw is None or raw.empty:
            continue
        el = np.array([P.elapsed_at(int(p), c) for p, c in zip(raw.period, raw.clock)], dtype=float)
        at = raw.actionType.astype(str); st = raw.subType.astype(str)
        kind = np.where(at == "Timeout", "timeout",
               np.where(at.isin(["Foul", "Free Throw"]), "whistle",
               np.where(at.isin(["Violation", "Jump Ball", "Instant Replay"]) | ((at == "Turnover") & ~st.isin(LIVE_TURNOVERS)), "other", "")))
        d = pd.DataFrame({"elapsed": el, "kind": kind})
        d = d[(d.kind != "") & (d.elapsed > 0) & (d.elapsed < REG) & ~d.elapsed.isin(BREAKS)]
        rank = d.kind.map({"timeout": 0, "whistle": 1, "other": 2})
        d = d.assign(rank=rank).sort_values(["elapsed", "rank"]).drop_duplicates("elapsed").drop(columns="rank")
        d.insert(0, "game_id", gid); new_s.append(d)
        sh = pd.to_numeric(raw.scoreHome, errors="coerce"); sa = pd.to_numeric(raw.scoreAway, errors="coerce")
        ok = sh.notna() & sa.notna()
        teams = [t for t in raw.teamTricode.astype(str).unique() if t]
        c = pd.DataFrame({"elapsed": el[ok.to_numpy()], "home_margin": (sh - sa)[ok].to_numpy()})
        c = c.drop_duplicates("elapsed", keep="last")
        c.insert(0, "game_id", gid); c["home"] = P.home_of(raw, teams) or ""
        new_c.append(c)
        if log and k % 100 == 0:
            log(f"timeline: {k} feeds read")
    if new_s:
        stops = pd.concat([stops] + new_s, ignore_index=True); scores = pd.concat([scores] + new_c, ignore_index=True)
        DERIVED_DIR.mkdir(parents=True, exist_ok=True)
        stops.to_parquet(sp, index=False); scores.to_parquet(cp, index=False)
    return stops[stops.game_id.isin(ids)], scores[scores.game_id.isin(ids)]


class _Game:
    """One game's margin (home side) and fouls by player, read at any second."""
    def __init__(self, scores: Optional[pd.DataFrame], fouls: Optional[pd.DataFrame]):
        self.t = scores.elapsed.to_numpy(float) if scores is not None and len(scores) else np.array([0.0])
        self.m = scores.home_margin.to_numpy(float) if scores is not None and len(scores) else np.array([0.0])
        self.home = str(scores.home.iloc[0]) if scores is not None and len(scores) else ""
        self.f = {n: g.elapsed.to_numpy(float) for n, g in fouls.groupby("player_name")} if fouls is not None and len(fouls) else {}

    def margin(self, team: str, at: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.t, at, side="right") - 1
        m = np.where(i >= 0, self.m[np.clip(i, 0, None)], 0.0)
        return m if team == self.home else -m

    def fouls(self, player: str, at: np.ndarray) -> np.ndarray:
        f = self.f.get(player)
        return np.searchsorted(f, at, side="right") if f is not None else np.zeros(len(at), dtype=int)


def _games(stints: pd.DataFrame, fouls: pd.DataFrame, scores: pd.DataFrame) -> Dict[str, _Game]:
    sc = {g: d.sort_values("elapsed") for g, d in scores.groupby("game_id")}
    fl = {g: d for g, d in fouls.groupby("game_id")}
    return {g: _Game(sc.get(g), fl.get(g)) for g in stints.game_id.unique()}


def _state_cols(d: pd.DataFrame) -> pd.DataFrame:
    """The categorical terms every panel shares, from step, margin, fouls, starter."""
    d["period"] = (d.step // PER_STEPS + 1).astype(int)
    d["clock"] = ((d.step % PER_STEPS) // 6).astype(int)                 # two-minute bins of the period
    d["brk"] = (((d.step + 1) % PER_STEPS == 0) & (d.step + 1 < N_STEPS)).astype(int)
    d["mbin"] = pd.cut(d.margin, MARGIN_EDGES, labels=MARGIN_LABELS, right=False).astype(str)
    d["phase"] = np.where(d.period == 4, np.where(d.step >= N_STEPS - 18, "last6", "q4"), "q1-3")
    d["trouble"] = ((d.fouls >= np.minimum(d.period + 1, 5))).astype(int)
    return d


def on_panel(stints: pd.DataFrame, fouls: pd.DataFrame, scores: pd.DataFrame, usual: pd.DataFrame, groups: Dict[str, str]) -> pd.DataFrame:
    """One row per player per step on the floor in regulation: out (his stint ended in the step), tis (seconds on by the step's end)."""
    games = _games(stints, fouls, scores)
    cols: Dict[str, list] = {k: [] for k in ("row", "step", "tis", "out", "margin", "fouls")}
    s = stints[(stints.start_sec < REG) & (stints.end_sec > stints.start_sec)].reset_index(drop=True)
    for i, (gid, team, nm, a, b, buz) in enumerate(zip(s.game_id, s.team, s.player_name, s.start_sec, s.end_sec, s.to_buzzer)):
        j0 = int(a // STEP); j1 = min(int(np.ceil(min(b, REG) / STEP)) - 1, N_STEPS - 1)
        if j1 < j0:
            continue
        steps = np.arange(j0, j1 + 1)
        out = np.zeros(len(steps), dtype=np.int8)
        if b < REG and not buz:
            out[-1] = 1
        at = steps * STEP
        g = games[gid]
        cols["row"].append(np.full(len(steps), i)); cols["step"].append(steps); cols["tis"].append((steps + 1) * STEP - a); cols["out"].append(out)
        cols["margin"].append(g.margin(team, at)); cols["fouls"].append(g.fouls(nm, at))
    d = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    meta = s[["game_id", "game_date", "team", "player_name", "stint_no", "starter"]].reset_index(names="row")
    d = d.merge(meta, on="row").drop(columns="row")
    d = d.merge(usual[["game_id", "player_name", "usual"]], on=["game_id", "player_name"], how="left")
    d["band"] = d.usual.map(band_of); d["group"] = d.player_name.map(lambda n: groups.get(n, "wing"))
    d["starter"] = d.starter.fillna(0).astype(int)
    d["tbin"] = np.minimum((d.tis - 1e-6) // 60, 15).astype(int)          # minutes on the floor, 15 = longer
    d["t3"] = np.where(d.tis <= 240, "0-4", np.where(d.tis <= 480, "4-8", "8+"))
    d["sno"] = np.minimum(d.stint_no, 4).astype(int)
    return _state_cols(d)


def bench_panel(stints: pd.DataFrame, fouls: pd.DataFrame, scores: pd.DataFrame, usual: pd.DataFrame, groups: Dict[str, str]) -> pd.DataFrame:
    """One row per step on the bench in regulation for every player who played the game: enter (he came in in the step), rest (seconds off by the step's end; since the tip for a man not yet used)."""
    games = _games(stints, fouls, scores)
    rows = []
    for (gid, team, nm), s in stints.groupby(["game_id", "team", "player_name"], sort=False):
        a = s.start_sec.to_numpy(float); b = s.end_sec.to_numpy(float)
        gaps = [(0.0, a[0], 0)] if a[0] > 0 else []
        gaps += [(b[k], a[k + 1], k + 1) for k in range(len(a) - 1)]
        if b[-1] < REG:
            gaps.append((b[-1], REG + 1.0, len(a)))                       # to the buzzer: never returned
        for off, on, done in gaps:
            if off >= REG:
                continue
            j0 = int(off // STEP); j1 = min(int(np.ceil(min(on, REG) / STEP)) - 1, N_STEPS - 1)
            if j1 < j0:
                continue
            steps = np.arange(j0, j1 + 1)
            ent = np.zeros(len(steps), dtype=np.int8)
            if on <= REG:
                ent[-1] = 1
            at = steps * STEP
            g = games[gid]
            rows.append(pd.DataFrame({"game_id": gid, "team": team, "player_name": nm, "step": steps, "rest": (steps + 1) * STEP - off, "enter": ent,
                                      "margin": g.margin(team, at), "fouls": g.fouls(nm, at), "used": done, "starter": int(s.starter.iloc[0] or 0),
                                      "game_date": s.game_date.iloc[0]}))
    d = pd.concat(rows, ignore_index=True)
    d = d.merge(usual[["game_id", "player_name", "usual"]], on=["game_id", "player_name"], how="left")
    d["band"] = d.usual.map(band_of); d["group"] = d.player_name.map(lambda n: groups.get(n, "wing"))
    d["rbin"] = np.minimum((d.rest - 1e-6) // 120, 10).astype(int)        # two-minute bins of rest, 10 = twenty minutes or more
    d["fresh"] = (d.used == 0).astype(int)                                # not yet used tonight
    return _state_cols(d)


# ----------------------------------------------------------------------------- the ladder
def _design(d: pd.DataFrame, terms: Sequence[Sequence[str]], fit_on: Optional[np.ndarray] = None):
    from scipy import sparse
    from sklearn.preprocessing import OneHotEncoder
    mats = []
    for t in terms:
        key = d[list(t)].astype(str).agg("|".join, axis=1) if len(t) > 1 else d[t[0]].astype(str)
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
        enc.fit(key[fit_on].to_frame() if fit_on is not None else key.to_frame())
        mats.append(enc.transform(key.to_frame()))
    return sparse.hstack(mats).tocsr()


def ladder(d: pd.DataFrame, y: str, steps: Sequence[Tuple[str, Sequence[Sequence[str]]]], C: float = 1.0, log=None) -> pd.DataFrame:
    """
    Held-out log loss as terms are added. steps: (name, the terms this rung adds); a name starting with '=' is fitted
    on its own terms alone (a side rung, not added to the ladder). Returns name, columns, train and test log loss
    (per 1,000 steps), the gain over the rung before and over the base rate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    tr = is_train(d.game_date); te = is_test(d.game_date)
    yv = d[y].to_numpy()
    base = log_loss(yv[te], np.full(te.sum(), yv[tr].mean()), labels=[0, 1])
    rows = [dict(rung="base rate", cols=0, test=1000 * base, gain=0.0, vs_base=0.0)]
    acc: List[Sequence[str]] = []; prev = base
    for name, terms in steps:
        side = name.startswith("=")
        use = list(terms) if side else acc + list(terms)
        X = _design(d, use, fit_on=tr)
        m = LogisticRegression(C=C, max_iter=300, solver="lbfgs").fit(X[tr], yv[tr])
        ll = log_loss(yv[te], m.predict_proba(X[te])[:, 1], labels=[0, 1])
        rows.append(dict(rung=name, cols=X.shape[1], test=1000 * ll, gain=100 * (prev - ll) / base if not side else float("nan"), vs_base=100 * (base - ll) / base))
        if not side:
            acc = use; prev = ll
        if log:
            log(f"  {name}: {1000 * ll:.2f} ({100 * (base - ll) / base:.1f}% under the base rate)")
    return pd.DataFrame(rows)


def rate_table(d: pd.DataFrame, y: str, by: Sequence[str], min_n: int = 400) -> pd.DataFrame:
    """The raw chance per step (as a chance per minute) by the given columns."""
    g = d.groupby(list(by), observed=True)[y].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_n]
    g["per_min"] = 1 - (1 - g["mean"]) ** (60 / STEP)
    return g.rename(columns={"size": "steps"}).drop(columns="mean")


def first_stints(stints: pd.DataFrame) -> pd.DataFrame:
    """A starter's first stint in every game he started: its length in minutes (12.0 = the whole first quarter or more is capped at the stint's own end)."""
    s = stints[(stints.stint_no == 1) & (stints.starter == 1) & (stints.start_sec == 0)]
    return s.assign(first=s.end_sec / 60.0)[["game_id", "game_date", "team", "player_name", "first"]]


def regularity(stints: pd.DataFrame, min_games: int = 10) -> dict:
    """How much of the spread in first-stint length is the player (between players) and how much is night to night (within)."""
    f = first_stints(stints)
    f = f[f["first"] <= 24]
    n = f.groupby("player_name")["first"].transform("size")
    f = f[n >= min_games]
    per = f.groupby("player_name")["first"].agg(["mean", "std", "size"])
    within = float(np.sqrt((per["std"] ** 2 * (per["size"] - 1)).sum() / (per["size"] - 1).sum()))
    return dict(players=int(len(per)), games=int(len(f)), mean=float(f["first"].mean()), sd_all=float(f["first"].std()), sd_between=float(per["mean"].std()),
                sd_within=within, within_median=float(per["std"].median()), share_player=float(1 - within ** 2 / f["first"].var()),
                full_quarter=float((f["first"] >= 11.9).mean()))


# ----------------------------------------------------------------------------- who comes in for whom
def swap_moments(stints: pd.DataFrame) -> pd.DataFrame:
    """Every second of regulation at which a team's five changed: game_id, game_date, team, t, outs, ins (tuples), bench (who had played or would play and was off the floor just before)."""
    rows = []
    for (gid, team), g in stints.groupby(["game_id", "team"], sort=False):
        date = g.game_date.iloc[0]
        ends = g[(g.end_sec > 0) & (g.end_sec < REG)].groupby("end_sec").player_name.apply(tuple).to_dict()
        starts = g[(g.start_sec > 0) & (g.start_sec < REG)].groupby("start_sec").player_name.apply(tuple).to_dict()
        players = list(dict.fromkeys(g.player_name))
        a = g.start_sec.to_numpy(float); b = g.end_sec.to_numpy(float); nm = g.player_name.to_numpy()
        for t in sorted(set(ends) | set(starts)):
            ins = tuple(p for p in starts.get(t, ()))
            outs = tuple(p for p in ends.get(t, ()) if p not in ins)
            if not ins or not outs:
                continue
            on = set(nm[(a < t) & (b >= t)])
            last_off = {}
            for p in players:
                if p in on:
                    continue
                e = b[(nm == p) & (b <= t)]
                last_off[p] = float(t - e.max()) if len(e) else float(t)      # since the tip for a man not yet used
            rows.append(dict(game_id=gid, game_date=date, team=team, t=float(t), outs=outs, ins=ins, bench=tuple(last_off), rest=tuple(last_off.values()),
                             brk=t in BREAKS))
    return pd.DataFrame(rows)


def em_pairs(moments: Iterable[Tuple[Sequence[str], Sequence[str]]], prior: Optional[Dict[Tuple[str, str], float]] = None, rounds: int = 6,
             max_m: int = 4) -> Dict[Tuple[str, str], float]:
    """
    Expected (out, in) counts from swap sets. A one-for-one swap counts as recorded. An m-for-m swap is a set: every
    pairing of its outs with its ins is weighed by the product of the current pair weights and credited in proportion.
    Unequal or larger sets credit every out-in pair evenly. prior: pseudo-counts per pair (position flow).
    """
    moments = [(tuple(o), tuple(i)) for o, i in moments]
    prior = prior or {}
    fixed: Dict[Tuple[str, str], float] = {}
    multi = []
    for o, i in moments:
        if len(o) == 1 and len(i) == 1:
            fixed[(o[0], i[0])] = fixed.get((o[0], i[0]), 0.0) + 1.0
        elif len(o) == len(i) <= max_m:
            multi.append((o, i))
        else:
            w = min(len(o), len(i)) / (len(o) * len(i))
            for a in o:
                for b in i:
                    fixed[(a, b)] = fixed.get((a, b), 0.0) + w
    counts = dict(fixed)
    for o, i in multi:                                                    # start even
        for a in o:
            for b in i:
                counts[(a, b)] = counts.get((a, b), 0.0) + 1.0 / len(i)
    for _ in range(rounds if multi else 0):
        tot: Dict[str, float] = {}
        for (a, b), v in counts.items():
            tot[a] = tot.get(a, 0.0) + v + prior.get((a, b), 0.0)
        w = lambda a, b: (counts.get((a, b), 0.0) + prior.get((a, b), 0.0) + 0.05) / (tot.get(a, 0.0) + 0.5)   # noqa: E731
        new = dict(fixed)
        for o, i in multi:
            perms = list(permutations(range(len(i))))
            p = np.array([np.prod([w(o[k], i[s[k]]) for k in range(len(o))]) for s in perms])
            p = p / p.sum()
            for pk, s in zip(p, perms):
                for k in range(len(o)):
                    new[(o[k], i[s[k]])] = new.get((o[k], i[s[k]]), 0.0) + pk
        counts = new
    return counts


def _shares(counts: Dict[Tuple[str, str], float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for (a, b), v in counts.items():
        out.setdefault(a, {})[b] = v
    return {a: {b: v / sum(d.values()) for b, v in d.items()} for a, d in out.items() if sum(d.values()) > 0}


def group_flow(moments: pd.DataFrame, groups: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """out group -> {in group: how much likelier than his share of the bench}, from the one-for-one swaps."""
    num: Dict[Tuple[str, str], float] = {}; den: Dict[Tuple[str, str], float] = {}
    for o, i, bench in zip(moments.outs, moments.ins, moments.bench):
        if len(o) != 1 or len(i) != 1 or not bench:
            continue
        go = groups.get(o[0], "wing"); gi = groups.get(i[0], "wing")
        num[(go, gi)] = num.get((go, gi), 0.0) + 1.0
        for b in bench:
            k = (go, groups.get(b, "wing")); den[k] = den.get(k, 0.0) + 1.0 / len(bench)
    return {go: {gi: num.get((go, gi), 0.0) / den[(go, gi)] for gi in ("big", "guard", "wing") if den.get((go, gi), 0) > 0} for go in ("big", "guard", "wing")}


REPL_RULES = ("uniform", "usual", "rest", "flow", "pairs_recorded", "pairs_single", "pairs_em", "em+flow", "em+flow x rest")
PAIR_GAMES = 15              # a team's pair history: this many games before the block
PAIR_BLOCK = 5               # refitted every this many games


def replacement_test(moments: pd.DataFrame, subs: pd.DataFrame, usual: pd.DataFrame, groups: Dict[str, str], rest_rate, mix: float = 0.5, log=None) -> pd.DataFrame:
    """
    Each rule's chance for the men who really came in, among the bench that was there (so a usual replacement who was
    out that night is already masked and the rest renormalised). Per moment: the mean log chance of the real ins and
    the share of them inside the rule's top len(ins). rest_rate(rest seconds, starter, fresh) -> the entry chance.
    Pair histories come from the team's PAIR_GAMES games before the block of PAIR_BLOCK games the moment is in.
    Returns one row per moment and rule: kind (single | multi | break), ll, hit.
    """
    us = {(g, n): (u, s) for g, n, u, s in zip(usual.game_id, usual.player_name, usual.usual, usual.start_share)}
    flow = group_flow(moments[is_train(moments.game_date)], groups)
    rec_by_game = {g: d for g, d in subs.groupby("game_id")}
    rows = []
    for (team, season), tm in moments.assign(season=moments.game_date.map(season_of)).groupby(["team", "season"], sort=False):
        order = list(dict.fromkeys(tm.sort_values(["game_date", "game_id"]).game_id))
        by_game = {g: d for g, d in tm.groupby("game_id")}
        for b0 in range(PAIR_BLOCK, len(order), PAIR_BLOCK):
            hist = order[max(0, b0 - PAIR_GAMES):b0]
            hm = pd.concat([by_game[g] for g in hist])
            ms = [(o, i) for o, i in zip(hm.outs, hm.ins)]
            prior_of = lambda o, i: 0.5 * flow.get(groups.get(o, "wing"), {}).get(groups.get(i, "wing"), 1.0)   # noqa: E731
            names = set(n for o, i in ms for n in o + i)
            em = _shares(em_pairs(ms, prior={(a, b): prior_of(a, b) for a in names for b in names if a != b}))
            single = _shares(em_pairs([m for m in ms if len(m[0]) == 1 and len(m[1]) == 1], rounds=0))
            rc: Dict[Tuple[str, str], float] = {}
            for g in hist:
                r = rec_by_game.get(g)
                if r is None:
                    continue
                for o, i in zip(r[r.team == team].player_out, r[r.team == team].player_in):
                    rc[(o, i)] = rc.get((o, i), 0.0) + 1.0
            recorded = _shares(rc)
            for g in order[b0:b0 + PAIR_BLOCK]:
                for mo in by_game[g].itertuples(index=False):
                    bench = list(mo.bench); rest = np.array(mo.rest, dtype=float)
                    if len(bench) < 2 or any(i not in bench for i in mo.ins):
                        continue
                    u = np.array([us.get((g, b), (np.nan, 0.0))[0] for b in bench]); u = np.where(np.isnan(u), 8.0, u)
                    st = np.array([us.get((g, b), (0.0, 0.0))[1] or 0.0 for b in bench]) >= 0.5
                    fresh = np.isclose(rest, mo.t)
                    r_rest = np.array([rest_rate(rest[k], bool(st[k]), bool(fresh[k])) for k in range(len(bench))])
                    f = np.array([np.mean([flow.get(groups.get(o, "wing"), {}).get(groups.get(b, "wing"), 1.0) for o in mo.outs]) for b in bench])
                    pair = lambda tab: np.array([np.mean([tab.get(o, {}).get(b, 0.0) for o in mo.outs]) for b in bench])   # noqa: E731
                    pe = pair(em)
                    fn = f / f.sum()
                    scores = {"uniform": np.ones(len(bench)), "usual": u + 1.0, "rest": r_rest, "flow": f, "pairs_recorded": pair(recorded) + 0.02,
                              "pairs_single": pair(single) + 0.02, "pairs_em": pe + 0.02, "em+flow": mix * pe + (1 - mix) * fn + 0.01,
                              "em+flow x rest": (mix * pe + (1 - mix) * fn + 0.01) * r_rest}
                    idx = [bench.index(i) for i in mo.ins]
                    kind = "break" if mo.brk else ("single" if len(mo.ins) == 1 and len(mo.outs) == 1 else "multi")
                    for rule, sc in scores.items():
                        p = sc / sc.sum()
                        top = set(np.argsort(-p)[:len(idx)])
                        rows.append((g, mo.game_date, kind, rule, float(np.mean(np.log(p[idx]))), float(np.mean([k in top for k in idx])), len(bench)))
        if log:
            log(f"  replacements: {team} {season}")
    return pd.DataFrame(rows, columns=["game_id", "game_date", "kind", "rule", "ll", "hit", "bench"])


# ----------------------------------------------------------------------------- which games count
def context_test(box: pd.DataFrame, stints: pd.DataFrame, taus: Sequence[Optional[float]] = (None, 0.30, 0.15, 0.08, 0.04), n_games: int = 15,
                 halflife: float = 6.0, regular_min: float = 18.0) -> pd.DataFrame:
    """
    A player's minutes (and, for a starter, his first stint) predicted from his own last n_games played games of the
    season, weighted by recency alone (tau None) or by recency x exp(-d / tau), d = the sum over the team's other
    regulars of (usual minutes / 240) x |played that game - plays tonight|. Returns one row per player-game and tau:
    err (minutes), err_first, n_eff, absent (a 24-minute regular of the team sat tonight), band.
    """
    fs = {(g, n): f for g, n, f in zip(*[first_stints(stints)[c] for c in ("game_id", "player_name", "first")])}
    rows = []
    b = box[box.played]
    for (team, season), tb in b.groupby(["team", "season"], sort=False):
        order = list(dict.fromkeys(tb.game_id))
        mins = {g: dict(zip(d.player_name, d.minutes.astype(float))) for g, d in tb.groupby("game_id", sort=False)}
        for gi in range(8, len(order)):
            G = order[gi]; prior = order[max(0, gi - 30):gi]
            recent = prior[-10:]
            tally: Dict[str, List[float]] = {}
            for g in recent:
                for n, m in mins[g].items():
                    tally.setdefault(n, []).append(m)
            reg = {n: np.mean(v) for n, v in tally.items() if len(v) >= 4 and np.mean(v) >= regular_min}
            absent = any(n not in mins[G] for n, m in reg.items() if m >= 24.0)
            for nm, actual in mins[G].items():
                mine = [g for g in prior if nm in mins[g]][-n_games:]
                if len(mine) < 5:
                    continue
                others = {n: m / 240.0 for n, m in reg.items() if n != nm}
                d = np.array([sum(s * ((n in mins[g]) != (n in mins[G])) for n, s in others.items()) for g in mine])
                rec = 0.5 ** ((len(mine) - 1 - np.arange(len(mine))) / halflife)
                y = np.array([mins[g][nm] for g in mine])
                f = np.array([fs.get((g, nm), np.nan) for g in mine]); f_act = fs.get((G, nm), np.nan)
                for tau in taus:
                    w = rec * (np.exp(-d / tau) if tau else 1.0)
                    w = w / w.sum()
                    ok = ~np.isnan(f)
                    ef = float(abs((w[ok] * f[ok]).sum() / w[ok].sum() - f_act)) if ok.sum() >= 3 and not np.isnan(f_act) and w[ok].sum() > 0 else np.nan
                    rows.append((G, nm, "recency" if tau is None else f"tau {tau}", float(abs((w * y).sum() - actual)), ef, float(1.0 / (w ** 2).sum()), absent,
                                 float(d[-1] if len(d) else 0.0), band_of(float(np.mean(y)))))
    return pd.DataFrame(rows, columns=["game_id", "player_name", "weights", "err", "err_first", "n_eff", "absent", "d_last", "band"])


# ----------------------------------------------------------------------------- the two teams
def cross_team(stints: pd.DataFrame, stops: pd.DataFrame, scores: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Per dead ball marked by something other than a substitution: did the home team change players, did the away team.
    Returns (the stoppage table, summary): the away team's substitution rate with and without a home substitution at
    the same dead ball, the Mantel-Haenszel odds ratio across kind x period x two-minute clock bin (what is left once
    the dead ball itself is accounted for), the same for the NEXT dead ball, and the share of all substitution moments
    that fall on a marked dead ball.
    """
    home = scores.drop_duplicates("game_id").set_index("game_id").home.to_dict()
    s = stints[(stints.start_sec > 0) & (stints.start_sec < REG) & ~stints.start_sec.isin(BREAKS)]
    sub_at = s.groupby(["game_id", "team"]).start_sec.apply(lambda x: set(x.astype(float))).to_dict()
    teams = stints.groupby("game_id").team.unique().to_dict()
    rows = []; on_mark = 0; total = 0
    for gid, d in stops.groupby("game_id"):
        h = home.get(gid); ts = [t for t in teams.get(gid, []) if t != h]
        if not h or len(ts) != 1:
            continue
        a = ts[0]
        hs = sub_at.get((gid, h), set()); as_ = sub_at.get((gid, a), set())
        t = d.elapsed.to_numpy(float); marked = set(t)
        total += len(hs) + len(as_); on_mark += len(hs & marked) + len(as_ & marked)
        hsub = np.array([x in hs for x in t]); asub = np.array([x in as_ for x in t])
        rows.append(pd.DataFrame({"game_id": gid, "elapsed": t, "kind": d.kind.to_numpy(), "home_sub": hsub, "away_sub": asub,
                                  "home_next": np.append(hsub[1:], False), "away_next": np.append(asub[1:], False)}))
    tab = pd.concat(rows, ignore_index=True)
    tab["period"] = (tab.elapsed // 720 + 1).astype(int); tab["clock"] = ((tab.elapsed % 720) // 120).astype(int)

    def mh(x: str, y: str) -> float:
        num = den = 0.0
        for _, g in tab.groupby(["kind", "period", "clock"]):
            a_ = float((g[x] & g[y]).sum()); b_ = float((g[x] & ~g[y]).sum()); c_ = float((~g[x] & g[y]).sum()); d_ = float((~g[x] & ~g[y]).sum())
            n = a_ + b_ + c_ + d_
            if n:
                num += a_ * d_ / n; den += b_ * c_ / n
        return num / den if den else float("nan")
    by_kind = tab.groupby("kind").agg(stops=("home_sub", "size"), home=("home_sub", "mean"), away=("away_sub", "mean"),
                                      both=("home_sub", lambda x: float((x & tab.loc[x.index, "away_sub"]).mean()))).reset_index()
    by_kind["if_independent"] = by_kind.home * by_kind.away
    summary = dict(stops=int(len(tab)), subs_on_marked_stop=on_mark / max(total, 1), away_given_home=float(tab.away_sub[tab.home_sub].mean()),
                   away_given_no_home=float(tab.away_sub[~tab.home_sub].mean()), odds_ratio_same_stop=mh("home_sub", "away_sub"),
                   odds_ratio_next_stop=(mh("home_sub", "away_next") * mh("away_sub", "home_next")) ** 0.5, by_kind=by_kind)
    return tab, summary


# ----------------------------------------------------------------------------- a regular sits
STAR_MIN = 28.0              # the man who sat: usual minutes at least this ...
STAR_PLAYED = 5              # ... having played this many of the team's previous STAR_WINDOW games
STAR_WINDOW = 8
GAIN_MIN = 5.0               # a teammate 'gained' when he played this many minutes over his usual


def star_out(box: pd.DataFrame, groups: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    The nights exactly one such regular sat while every other 24-minute regular played. (nights, heirs): per night the
    freed minutes (his usual), the top gainer and his share of them, how many gained GAIN_MIN, whether the top gainer
    is the obvious one by each pre-game guess (the new starter; the highest-minute bench player of his position
    group; the highest-minute bench player), the same for fantasy points, and the night's number in the absence.
    heirs: every teammate's usual and actual minutes and points that night. A teammate's usual is per team game over
    the STAR_WINDOW games before (a night he did not play counts as zero), so the gains of a night add up to the
    minutes the absent men left.
    """
    nights, heirs = [], []
    for (team, season), tb in box.groupby(["team", "season"], sort=False):
        order = list(dict.fromkeys(tb.game_id))
        played = {g: d[d.played] for g, d in tb.groupby("game_id", sort=False)}
        run: Dict[str, int] = {}
        for gi in range(STAR_WINDOW, len(order)):
            G = order[gi]; prior = order[gi - STAR_WINDOW:gi]
            hist = pd.concat([played[g] for g in prior])
            agg = hist.groupby("player_name").agg(usual=("minutes", "mean"), fpts=("dk_fpts", "mean"), n=("minutes", "size"), starts=("starter", "mean"))
            now = played[G].set_index("player_name")
            regs = agg[(agg.usual >= 24.0) & (agg.n >= STAR_PLAYED)]
            out = [n for n in regs.index if n not in now.index]
            for n in list(run):
                if n not in out:
                    run.pop(n)
            for n in out:
                run[n] = run.get(n, 0) + 1
            stars = [n for n in out if regs.usual[n] >= STAR_MIN]
            if len(out) != 1 or len(stars) != 1 or len(now) < 8:
                continue
            star = stars[0]; freed = float(regs.usual[star])
            mates = agg[agg.index != star].index.union(now.index)
            rows = []
            for m in mates:
                share = agg.n[m] / len(prior) if m in agg.index else 0.0      # per team game: a night he did not play counts as zero
                u = float(agg.usual[m]) * share if m in agg.index else 0.0
                uf = float(agg.fpts[m]) * share if m in agg.index else 0.0
                a = float(now.minutes[m]) if m in now.index else 0.0; af = float(now.dk_fpts[m]) if m in now.index else 0.0
                rows.append(dict(game_id=G, team=team, star=star, player=m, usual=u, actual=a, gain=a - u, usual_fpts=uf, fpts=af, fgain=af - uf,
                                 starts=float(agg.starts[m]) if m in agg.index else 0.0, started=bool(now.starter[m]) if m in now.index else False,
                                 group=groups.get(m, "wing"), played=m in now.index))
            h = pd.DataFrame(rows)
            if h.empty or not h.played.any():
                continue
            top = h.loc[h.gain.idxmax()]; ftop = h.loc[h.fgain.idxmax()]
            bench = h[h.starts < 0.5]
            new_st = h[h.started & (h.starts < 0.5)]
            same = bench[bench.group == groups.get(star, "wing")]
            guess_start = new_st.sort_values("usual", ascending=False).player.iloc[0] if len(new_st) else None
            guess_group = same.sort_values("usual", ascending=False).player.iloc[0] if len(same) else None
            guess_bench = bench.sort_values("usual", ascending=False).player.iloc[0] if len(bench) else None
            pos = h.gain.clip(lower=0)
            nights.append(dict(game_id=G, game_date=tb[tb.game_id == G].game_date.iloc[0], team=team, star=star, freed=freed, night=run.get(star, 1),
                               top=top.player, top_gain=float(top.gain), top_share=float(top.gain / freed), top_of_gains=float(top.gain / pos.sum()) if pos.sum() > 0 else np.nan,
                               gainers=int((h.gain >= GAIN_MIN).sum()), top_is_new_starter=top.player == guess_start, top_is_group_bench=top.player == guess_group,
                               top_is_bench=top.player == guess_bench, has_new_starter=guess_start is not None,
                               new_starter_gain=float(h.gain[h.player == guess_start].iloc[0]) if guess_start else np.nan,
                               ftop=ftop.player, ftop_gain=float(ftop.fgain), ftop_is_top=ftop.player == top.player, ftop_is_new_starter=ftop.player == guess_start,
                               fgainers=int((h.fgain >= 8.0).sum())))
            heirs.append(h)
    return pd.DataFrame(nights), (pd.concat(heirs, ignore_index=True) if heirs else pd.DataFrame())


def quiet_nights(box: pd.DataFrame) -> pd.DataFrame:
    """
    The control for star_out: the nights no 24-minute regular sat. The biggest gainer of any night gains something by
    chance alone, so a star-out night's top gain is read against these: per night top_gain, gainers (GAIN_MIN over
    usual), ftop_gain, fgainers, with the same usual minutes as star_out.
    """
    rows = []
    for (team, season), tb in box.groupby(["team", "season"], sort=False):
        order = list(dict.fromkeys(tb.game_id))
        played = {g: d[d.played] for g, d in tb.groupby("game_id", sort=False)}
        for gi in range(STAR_WINDOW, len(order)):
            G = order[gi]; prior = order[gi - STAR_WINDOW:gi]
            hist = pd.concat([played[g] for g in prior])
            agg = hist.groupby("player_name").agg(usual=("minutes", "mean"), fpts=("dk_fpts", "mean"), n=("minutes", "size"))
            now = played[G].set_index("player_name")
            regs = agg[(agg.usual >= 24.0) & (agg.n >= STAR_PLAYED)]
            if any(n not in now.index for n in regs.index) or len(now) < 8:
                continue
            idx = agg.index.union(now.index)
            share = agg.n / len(prior)
            u = (agg.usual * share).reindex(idx).fillna(0.0); uf = (agg.fpts * share).reindex(idx).fillna(0.0)
            gain = now.minutes.astype(float).reindex(idx).fillna(0.0) - u; fgain = now.dk_fpts.astype(float).reindex(idx).fillna(0.0) - uf
            rows.append(dict(game_id=G, team=team, top_gain=float(gain.max()), gainers=int((gain >= GAIN_MIN).sum()), ftop_gain=float(fgain.max()),
                             fgainers=int((fgain >= 8.0).sum()), ftop_is_top=fgain.idxmax() == gain.idxmax()))
    return pd.DataFrame(rows)


def repeat_heirs(nights: pd.DataFrame) -> dict:
    """Over absences of two nights or more: how often the top gainer of night k is the top gainer of night k+1, in minutes and in fantasy points."""
    same = []; fsame = []
    for _, d in nights.sort_values("game_date").groupby(["team", "star"]):
        d = d.sort_values("night")
        for a, b in zip(d.itertuples(), d.iloc[1:].itertuples()):
            if b.night == a.night + 1:
                same.append(a.top == b.top); fsame.append(a.ftop == b.ftop)
    return dict(pairs=len(same), same_minutes_top=float(np.mean(same)) if same else float("nan"), same_fpts_top=float(np.mean(fsame)) if fsame else float("nan"))
