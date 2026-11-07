# src/ivan/sports/nba/gamesim/fouls.py
"""
Mid-game exits: who inherits the minutes of a player who leaves for good.

A pre-game absence is settled (2026-10-23, docs/nba/notes/minutes_model_plan.md): substitution pairs are diffuse and
position groups carry the bias, so the skeleton routes an absent player's minutes to his group. A player who fouls out
or leaves hurt in the third quarter is a different question, and this module measures it on the stints:

exits(stints, fouls, groups)   every player who left for good before the buzzer with at least EXIT_MIN_LEFT minutes of
                               regulation left, not in garbage time, and whose trailing curve owed him at least
                               OWED_MIN more minutes: one row per exit (game, team, player, minute, kind foul_out |
                               other, owed minutes) and, per teammate, the minutes he gained over his own curve after
                               the exit (the heirs frame).
predictors                     how the owed minutes would be split by each rule: pairs (the coach's recorded
                               substitution habit, sub_shares), group (the exiting player's position group first, the
                               pre-game rule), slot (the best teammate off the floor by curve at those minutes, what
                               rotation.sample_worlds does), proportional (by usual minutes, the baseline).
score(exits, heirs, preds)     per rule: mean absolute error of the predicted gain per teammate, the share of the owed
                               minutes the rule's top heir really took, and how often the rule's top heir was the real
                               one; overall and by the exiting player's group and by kind.
midgame_shares(...)            per team: out -> {heir: share of the gained minutes} over the measured exits, the number
                               the team page shows next to the pre-game habit.

Minutes 'after the exit' are regulation minutes from the exit to minute 48; the curve is the trailing CURVE_GAMES games
of the same team before that game (on-floor share per minute), so 'owed' and 'gained' are each player's own usual
pattern, not a league rule.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ivan.sports.nba.projection.groups import group_of, position_map
from ivan.sports.nba.projection.rotation import garbage_onsets

MINUTES = 48
EXIT_MIN_LEFT = 4.0          # minutes of regulation left at the exit for it to count
OWED_MIN = 3.0               # the curve must owe him at least this after the exit
BLOWOUT = [(0, 30.0), (24, 25.0), (30, 18.0), (36, 15.0), (40, 12.0)]   # (from minute, |margin| at the exit that makes it a blowout pull, not a loss)
CURVE_GAMES = 10
CURVE_HALFLIFE = 4.0
PAIR_GAMES = 15
GROUP_TILT = {"big": {"big": 1.0, "guard": 0.0, "wing": 0.0}, "guard": {"guard": 4.0, "wing": 1.0, "big": 1.0}, "wing": {"wing": 1.0, "guard": 1.0, "big": 1.0}}
RULES = ("pairs", "group", "slot", "proportional", "flow", "none")      # none: nobody gains (the noise floor)
# the measured mid-game flow (2026-11-07, 1,468 exits of 2025-26 and the first weeks of 2026-27, docs/nba/notes/midgame_exits_2026-11-07.md):
# out group -> share of his owed minutes that lands on each heir group; the rest never reaches a rotation player. fit_flow refits it.
DEFAULT_FLOW = {"big": {"big": 0.40, "guard": 0.30, "wing": 0.08}, "guard": {"big": 0.0, "guard": 0.47, "wing": 0.36}, "wing": {"big": 0.06, "guard": 0.24, "wing": 0.46}}


def blowout_margin(minute: int) -> float:
    m = BLOWOUT[0][1]
    for start, v in BLOWOUT:
        if minute >= start:
            m = v
    return m


# ----------------------------------------------------------------------------- data
def load_stints(session: Session, since=None) -> pd.DataFrame:
    q = "select game_id, game_date, team, player_name, start_sec, end_sec, minutes, margin_out, to_buzzer, starter from nba_stint"
    df = pd.read_sql(text(q), session.get_bind())
    df["game_date"] = pd.to_datetime(df.game_date).dt.date
    if since is not None:
        df = df[df.game_date >= pd.Timestamp(since).date()]
    return df.sort_values(["game_date", "game_id", "team", "start_sec"]).reset_index(drop=True)


def load_fouls(session: Session) -> pd.DataFrame:
    return pd.read_sql(text("select game_id, team, player_name, elapsed, foul_no from nba_foul where personal = 1"), session.get_bind())


def load_subs(session: Session) -> pd.DataFrame:
    """Single swaps only (several at one second are the scorer's typing order, not roles)."""
    df = pd.read_sql(text("select game_id, game_date, team, player_out, player_in, elapsed from nba_sub"), session.get_bind())
    if df.empty:
        return df
    n = df.groupby(["game_id", "team", "elapsed"]).player_out.transform("size")
    df = df[n == 1].copy(); df["game_date"] = pd.to_datetime(df.game_date).dt.date
    return df


def minute_vector(stints: pd.DataFrame) -> np.ndarray:
    """A player's on-floor share per minute of regulation from his stints of one game."""
    v = np.zeros(MINUTES)
    for a, b in zip(stints.start_sec, stints.end_sec):
        b = min(float(b), 2880.0); a = float(a)
        if b <= a:
            continue
        lo, hi = int(a // 60), int(np.ceil(b / 60))
        for m in range(max(0, lo), min(MINUTES, hi)):
            v[m] = max(v[m], min(1.0, (min(b, (m + 1) * 60) - max(a, m * 60)) / 60.0))
    return v


class TeamHistory:
    """One team's games in order with each player's minute vector per game; trailing curves before any game."""
    def __init__(self, stints: pd.DataFrame):
        self.games: List[str] = list(dict.fromkeys(stints.sort_values(["game_date", "game_id"]).game_id))
        self.vectors: Dict[str, Dict[str, np.ndarray]] = {}
        for (gid, nm), s in stints.groupby(["game_id", "player_name"]):
            self.vectors.setdefault(gid, {})[nm] = minute_vector(s)

    def curve(self, before_game: str, n_games: int = CURVE_GAMES, halflife: float = CURVE_HALFLIFE) -> Dict[str, np.ndarray]:
        """player -> trailing weighted on-floor share per minute (a game he missed counts as zeros: the curve is his usual minutes)."""
        i = self.games.index(before_game)
        prev = self.games[max(0, i - n_games):i]
        if not prev:
            return {}
        w = np.array([0.5 ** ((len(prev) - 1 - k) / halflife) for k in range(len(prev))]); w = w / w.sum()
        out: Dict[str, np.ndarray] = {}
        for k, gid in enumerate(prev):
            for nm, v in self.vectors[gid].items():
                out[nm] = out.get(nm, np.zeros(MINUTES)) + w[k] * v
        return out


def pair_shares(subs: pd.DataFrame, team: str, before_game: str, games_order: List[str], n_games: int = PAIR_GAMES) -> Dict[str, Dict[str, float]]:
    """out -> {in: share} from the team's single swaps in the n_games games before `before_game`."""
    i = games_order.index(before_game)
    prev = set(games_order[max(0, i - n_games):i])
    d = subs[(subs.team == team) & subs.game_id.isin(prev)]
    out: Dict[str, Dict[str, float]] = {}
    for o, g in d.groupby("player_out"):
        c = g.player_in.value_counts(); tot = float(c.sum())
        out[o] = {k: float(v) / tot for k, v in c.items()}
    return out


# ----------------------------------------------------------------------------- the exits
def exits(stints: pd.DataFrame, fouls: pd.DataFrame, subs: pd.DataFrame, groups: Dict[str, str],
          min_left: float = EXIT_MIN_LEFT, owed_min: float = OWED_MIN, flow: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    (exits, heirs). exits: game_id, game_date, team, player, group, minute, kind, owed, margin, n_heirs.
    heirs: one row per (exit, teammate): expected (curve minutes after the exit), actual, gain, group, on_floor (at the
    exit), and each rule's predicted gain: pred_pairs, pred_group, pred_slot, pred_proportional.
    """
    ex_rows, heir_rows = [], []
    foul_outs = set()
    if fouls is not None and not fouls.empty:
        f6 = fouls[fouls.foul_no == 6]
        foul_outs = set(zip(f6.game_id, f6.player_name))
    for team, ts in stints.groupby("team"):
        hist = TeamHistory(ts)
        onsets = garbage_onsets(ts)
        for gid, g in ts.groupby("game_id"):
            curve = hist.curve(gid)
            if not curve:
                continue
            onset, _ = onsets.get(gid, (None, set()))
            date = g.game_date.iloc[0]
            per_player = {nm: s for nm, s in g.groupby("player_name")}
            for nm, s in per_player.items():
                if nm not in curve or int(s.to_buzzer.max()) == 1:
                    continue
                last_out = float(s.end_sec.max())
                if last_out > 2880.0 - min_left * 60.0:
                    continue
                minute = int(last_out // 60)
                if onset is not None and minute >= onset:
                    continue
                margin = s.sort_values("end_sec").margin_out.iloc[-1]
                kind = "foul_out" if (gid, nm) in foul_outs else "other"
                if kind != "foul_out" and pd.notna(margin) and abs(int(margin)) >= blowout_margin(minute):
                    continue
                owed = float(curve[nm][minute:].sum()) - float(minute_vector(s)[minute:].sum())
                if owed < owed_min:
                    continue
                mates = [m for m in curve if m != nm]
                if not mates:
                    continue
                # the floor just before the exit: whoever came in at that very second is the heir, not a man already on
                on_floor = {m for m, ms in per_player.items() if ((ms.start_sec < last_out - 1e-6) & (ms.end_sec >= last_out)).any()} - {nm}
                exp = {m: float(curve[m][minute:].sum()) for m in mates}
                act = {m: float(minute_vector(per_player[m])[minute:].sum()) if m in per_player else 0.0 for m in mates}
                usual = {m: float(curve[m].sum()) for m in mates}
                preds = predictors(nm, mates, owed, groups, usual, curve, minute, on_floor, pair_shares(subs, team, gid, hist.games) if subs is not None and not subs.empty else {},
                                   flow or DEFAULT_FLOW)
                for m in mates:
                    heir_rows.append(dict(game_id=gid, team=team, player=nm, heir=m, group=groups.get(m, "wing"), expected=exp[m], actual=act[m], gain=act[m] - exp[m],
                                          on_floor=m in on_floor, **{f"pred_{r}": preds[r].get(m, 0.0) for r in RULES}))
                ex_rows.append(dict(game_id=gid, game_date=date, team=team, player=nm, group=groups.get(nm, "wing"), minute=minute, kind=kind, owed=owed,
                                    margin=int(margin) if pd.notna(margin) else None, n_heirs=len(mates)))
    ex_cols = ["game_id", "game_date", "team", "player", "group", "minute", "kind", "owed", "margin", "n_heirs"]
    heir_cols = ["game_id", "team", "player", "heir", "group", "expected", "actual", "gain", "on_floor"] + [f"pred_{r}" for r in RULES]
    return pd.DataFrame(ex_rows, columns=ex_cols), pd.DataFrame(heir_rows, columns=heir_cols)


def flow_split(out_group: str, mates: List[str], owed: float, groups: Dict[str, str], room: Dict[str, float], flow: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """The flow rule: each heir group gets its measured share of the owed minutes, split inside the group by the room each player has left."""
    pred = {m: 0.0 for m in mates}
    for hg, share in flow.get(out_group, {}).items():
        members = [m for m in mates if groups.get(m, "wing") == hg]
        tot = sum(room[m] for m in members)
        if tot <= 0 or share <= 0:
            continue
        for m in members:
            pred[m] = owed * share * room[m] / tot
    return pred


def predictors(out: str, mates: List[str], owed: float, groups: Dict[str, str], usual: Dict[str, float], curve: Dict[str, np.ndarray],
               minute: int, on_floor: set, pairs: Dict[str, Dict[str, float]], flow: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Each rule's split of the owed minutes over the teammates (minutes each would gain)."""
    def norm(w: Dict[str, float]) -> Dict[str, float]:
        tot = sum(max(0.0, v) for v in w.values())
        return {m: owed * max(0.0, v) / tot for m, v in w.items()} if tot > 0 else {m: owed / len(mates) for m in mates}
    g_out = groups.get(out, "wing"); tilt = GROUP_TILT[g_out]
    # headroom: a teammate can only take what his own pattern leaves free after the exit (48 - minute minus his expected)
    room = {m: max(0.0, (MINUTES - minute) - float(curve[m][minute:].sum())) for m in mates}
    prop = norm({m: usual.get(m, 0.0) * (room[m] > 0) for m in mates})
    grp = norm({m: tilt.get(groups.get(m, "wing"), 0.0) * max(usual.get(m, 0.0), 1.0) * (room[m] > 0) for m in mates})
    if sum(grp.values()) <= 0:
        grp = prop
    pr = pairs.get(out) or {}
    pairs_pred = norm({m: pr.get(m, 0.0) for m in mates}) if any(m in pr for m in mates) else grp
    # slot: the teammates off the floor at the exit, by how much they usually play those minutes
    off = [m for m in mates if m not in on_floor]
    slot = norm({m: float(curve[m][minute:].mean()) + 1e-3 for m in off}) if off else prop
    return {"pairs": pairs_pred, "group": grp, "slot": slot, "proportional": prop, "flow": flow_split(g_out, mates, owed, groups, room, flow),
            "none": {m: 0.0 for m in mates}}


def fit_flow(exits_df: pd.DataFrame, heirs: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """out group -> {heir group: mean over exits of the group's summed gain / owed}, negatives floored at 0."""
    fl = group_flow(exits_df, heirs)
    fl = fl[fl.kind == "all"]
    out: Dict[str, Dict[str, float]] = {}
    for r in fl.itertuples(index=False):
        if r.out_group == "all":
            continue
        out.setdefault(r.out_group, {})[r.heir_group] = float(max(0.0, r.share_of_owed))
    return out


# ----------------------------------------------------------------------------- scores
def score(exits_df: pd.DataFrame, heirs: pd.DataFrame) -> pd.DataFrame:
    """Per rule and per cut (all, by the exiting player's group, by kind): n exits, MAE of the gain per teammate, top-heir hit rate, the share of owed the top heir took."""
    if heirs.empty:
        return pd.DataFrame(columns=["cut", "rule", "exits", "mae", "top_hit", "top_share", "real_top_share"])
    h = heirs.merge(exits_df[["game_id", "team", "player", "group", "kind", "owed"]].rename(columns={"group": "out_group"}), on=["game_id", "team", "player"])
    rows = []
    cuts = [("all", h)] + [(f"out={g}", h[h.out_group == g]) for g in ("big", "guard", "wing")] + [(f"kind={k}", h[h.kind == k]) for k in ("foul_out", "other")]
    for cut, d in cuts:
        if d.empty:
            continue
        keys = ["game_id", "team", "player"]
        real_top = d.loc[d.groupby(keys).gain.idxmax()][keys + ["heir", "gain", "owed"]].rename(columns={"heir": "real_top", "gain": "real_gain"})
        for r in RULES:
            col = f"pred_{r}"
            if r == "none":
                rows.append(dict(cut=cut, rule=r, exits=int(real_top.shape[0]), mae=float(d.gain.abs().mean()), top_hit=float("nan"), top_share=float("nan"),
                                 real_top_share=float((real_top.real_gain / real_top.owed).mean())))
                continue
            top = d.loc[d.groupby(keys)[col].idxmax()][keys + ["heir", "gain"]].rename(columns={"heir": "pred_top", "gain": "pred_top_gain"})
            m = real_top.merge(top, on=keys)
            rows.append(dict(cut=cut, rule=r, exits=int(len(m)), mae=float((d[col] - d.gain).abs().mean()), top_hit=float((m.real_top == m.pred_top).mean()),
                             top_share=float((m.pred_top_gain / m.owed).mean()), real_top_share=float((m.real_gain / m.owed).mean())))
    return pd.DataFrame(rows)


def group_flow(exits_df: pd.DataFrame, heirs: pd.DataFrame) -> pd.DataFrame:
    """
    Where the owed minutes went, by the exiting player's group and the heirs' group: the mean over exits of (the group's
    summed gain / owed), the mean gain per heir in that group, and the share taken by the group's single biggest heir
    (the one with the most usual minutes off the floor: 'the backup'), with the count of exits.
    """
    if heirs.empty:
        return pd.DataFrame(columns=["out_group", "kind", "heir_group", "exits", "share_of_owed", "gain_per_heir", "backup_share", "second_share"])
    h = heirs.merge(exits_df[["game_id", "team", "player", "group", "kind", "owed"]].rename(columns={"group": "out_group"}), on=["game_id", "team", "player"])
    rows = []
    for (og, kind), d in list(h.groupby(["out_group", "kind"])) + [((og, "all"), d) for og, d in h.groupby("out_group")] + [(("all", "all"), h)]:
        keys = ["game_id", "team", "player"]
        for hg, e in d.groupby("group"):
            per = e.groupby(keys).agg(gain=("gain", "sum"), owed=("owed", "first"), n=("heir", "size"))
            # the backup: the group's heir with the largest predicted proportional gain among those off the floor (else any), his share of owed
            e2 = e.assign(rank_key=e.pred_proportional * (~e.on_floor) + 1e-6 * e.pred_proportional)
            top = e2.loc[e2.groupby(keys).rank_key.idxmax()]
            second = e2.drop(top.index); second = second.loc[second.groupby(keys).rank_key.idxmax()] if len(second) else second
            rows.append(dict(out_group=og, kind=kind, heir_group=hg, exits=int(len(per)), share_of_owed=float((per.gain / per.owed).mean()),
                             gain_per_heir=float(e.gain.mean()), backup_share=float((top.gain / top.owed).mean()) if len(top) else float("nan"),
                             second_share=float((second.gain / second.owed).mean()) if len(second) else float("nan")))
    return pd.DataFrame(rows)


def midgame_shares(exits_df: pd.DataFrame, heirs: pd.DataFrame, team: Optional[str] = None) -> pd.DataFrame:
    """team, out, heir, share (of the positive gains over that player's exits), exits: who really covered whom."""
    h = heirs if team is None else heirs[heirs.team == team]
    if h.empty:
        return pd.DataFrame(columns=["team", "out", "heir", "share", "exits"])
    pos = h.assign(gain=h.gain.clip(lower=0.0))
    tot = pos.groupby(["team", "player"]).gain.transform("sum")
    pos = pos.assign(share=np.where(tot > 0, pos.gain / tot.replace(0, np.nan), 0.0))
    out = pos.groupby(["team", "player", "heir"], as_index=False).agg(share=("share", "sum"), exits=("game_id", "nunique"))
    n_ex = pos.groupby(["team", "player"]).game_id.nunique().rename("n")
    out = out.merge(n_ex, left_on=["team", "player"], right_index=True)
    out["share"] = out.share / out.n
    return out.rename(columns={"player": "out"})[["team", "out", "heir", "share", "exits"]].sort_values(["team", "out", "share"], ascending=[True, True, False])


def run(session: Session, since=None) -> dict:
    """
    Load, measure, score. The flow rule is held out: its matrix is fitted on the first half of the exits by date and every
    rule is scored on the second half ("scores_heldout"); "scores" and "flow" use every exit with DEFAULT_FLOW.
    Returns {"exits", "heirs", "scores", "scores_heldout", "flow", "flow_fitted", "summary"}.
    """
    stints = load_stints(session, since); fouls = load_fouls(session); subs = load_subs(session)
    groups = {n: group_of(p) for n, p in position_map(session).items()}
    ex, heirs = exits(stints, fouls, subs, groups)
    sc = score(ex, heirs)
    held = pd.DataFrame(); fitted = {}
    if len(ex) > 40:
        mid = ex.game_date.sort_values().iloc[len(ex) // 2]
        first = ex[ex.game_date < mid][["game_id", "team", "player"]]
        h_first = heirs.merge(first, on=["game_id", "team", "player"])
        fitted = fit_flow(ex[ex.game_date < mid], h_first)
        ex2, heirs2 = exits(stints[stints.game_date >= mid], fouls, subs, groups, flow=fitted)
        # the second half only has curves from its own games: refit the curves on the whole history instead by filtering the full run
        later = ex[ex.game_date >= mid][["game_id", "team", "player"]]
        _, heirs_full = exits(stints, fouls, subs, groups, flow=fitted)
        held = score(ex.merge(later, on=["game_id", "team", "player"]), heirs_full.merge(later, on=["game_id", "team", "player"]))
    summary = dict(exits=int(len(ex)), foul_outs=int((ex.kind == "foul_out").sum()), owed_mean=float(ex.owed.mean()) if len(ex) else float("nan"),
                   by_group=ex.group.value_counts().to_dict() if len(ex) else {})
    return {"exits": ex, "heirs": heirs, "scores": sc, "scores_heldout": held, "flow": group_flow(ex, heirs), "flow_fitted": fitted, "summary": summary}
