# src/ivan/sports/nba/validation.py
"""
Projected vs actual for the minutes / per-minute model, and the pre-game
rotation review that tells a person what to check before lock.

scorecard   the last projection capture before a slate's lock (nba_projection_capture,
            source "ours", with the history context the capture carries) joined to the
            box scores (nba_stats) by player. Every fantasy-point miss is split into
            the part the minutes explain and the part the per-minute rate explains:
                fpts_proj - fpts_act = (min_proj - min_act) * fpm_proj  +  min_act * (fpm_proj - fpm_act)
            and each player is classed by where his minutes came from (same team as
            his history, a new team, no history at all, a pin). Summaries by starter
            label, priced, salary tier and history class; DK's own average and the
            books' implied points as yardsticks; plain-language findings. One CSV +
            JSON per slate under data/nba/diagnostics/scorecard/, so the slates
            accumulate; scorecard_history() reads them back as a trend table.
            run_pending_scorecards() is what the nightly job calls after box scores.

review      for the slate in nba_player right now: per team, every player's minutes
            with where they came from (history team and games, trailing minutes, the
            last five games, play rate, baseline before allocation, market-implied
            minutes for priced players) and flags for the patterns that produced the
            misses so far: new team, no history, DNP-prone, above recent maximum,
            deep bench projected like a rotation player, market minutes far from ours.
            The desk shows the same flags; scripts/nba/validate.py prints them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.sports.nba.models import NBAMarketRate, NBAProjectionCapture
from ivan.sports.nba.paths import DIAG_DIR

Log = Callable[[str], None]
SCORECARD_DIR = DIAG_DIR / "scorecard"
SALARY_TIERS = ((0, 4000, "<=4.0k"), (4000, 6000, "4-6k"), (6000, 8000, "6-8k"), (8000, 10 ** 6, "8k+"))
ROTATION_MIN = 10.0        # projected minutes above which a miss is "a rotation player"
DNP_PRONE = 0.6            # trailing play rate below this is a DNP risk
DEEP_BENCH_MIN = 22.0      # a player who rarely starts projected above this is worth a look
RECENT_MAX_MARGIN = 6.0    # minutes above his recent maximum
MARKET_MIN_GAP = 6.0       # market-implied minutes this far from ours
TOP_N = 15


# ----------------------------------------------------------------------------- inputs
def captures_for(session: Session, draft_group: int) -> tuple[pd.DataFrame, pd.DataFrame, Optional[datetime]]:
    """(ours, dk_avg, lock): the last 'ours' capture at or before the slate's lock (else the last one) and DK's FPPG rows taken at the same moment."""
    q = session.query(NBAProjectionCapture).filter(NBAProjectionCapture.draft_group_id == int(draft_group))
    df = pd.read_sql(q.statement, session.get_bind())
    if df.empty:
        return df, df, None
    ours = df[df.source == "ours"]
    lock = pd.to_datetime(ours["lock"].dropna().max()) if ours["lock"].notna().any() else None
    before = ours[ours.captured_at <= lock] if lock is not None else ours
    stamp = (before if len(before) else ours).captured_at.max()
    ours = ours[ours.captured_at == stamp].copy()
    dk = df[(df.source == "dk_avg") & (df.captured_at == stamp)][["player_name", "fpts_mean"]].rename(columns={"fpts_mean": "dk_avg_fpts"})
    return ours, dk, (lock.to_pydatetime() if lock is not None else None)


def actuals_for(session: Session, day: date) -> pd.DataFrame:
    """Box score rows for the games of `day` (dressed DNPs included with 0 minutes)."""
    q = "select player_name, team_abbreviation as team_act, minutes as min_act, dk_fpts as fpts_act, position as pos_act from nba_stats where game_date = :d"
    from sqlalchemy import text
    return pd.read_sql(text(q).bindparams(d=day.isoformat()), session.get_bind())


def slate_day(lock: Optional[datetime], ours: pd.DataFrame) -> Optional[date]:
    if lock is not None:
        return lock.date()
    gt = pd.to_datetime(ours["gametime"].dropna())
    return gt.min().date() if len(gt) else None


# ----------------------------------------------------------------------------- the join and the decomposition
def history_class(r) -> str:
    """pinned | no history | new team | same team | unknown (a capture from before the context columns existed)."""
    if r.get("pinned"):
        return "pinned"
    g = r.get("hist_games")
    if g is None or (isinstance(g, float) and np.isnan(g)):
        return "unknown"
    if int(g) == 0:
        return "no history"
    if r.get("hist_team") and r.get("team") and str(r["hist_team"]) != str(r["team"]):
        return "new team"
    return "same team"


def salary_tier(s) -> str:
    try:
        s = float(s)
    except (TypeError, ValueError):
        return "?"
    for lo, hi, label in SALARY_TIERS:
        if lo < s <= hi or (lo == 0 and s <= hi):
            return label
    return "?"


def join_actuals(ours: pd.DataFrame, act: pd.DataFrame, dk: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per captured player with his actuals (0 when no box row) and the error decomposition."""
    df = ours.merge(act, on="player_name", how="left")
    if dk is not None and len(dk):
        df = df.merge(dk, on="player_name", how="left")
    else:
        df["dk_avg_fpts"] = np.nan
    df["box_row"] = df["min_act"].notna()
    df["min_act"] = df["min_act"].fillna(0.0); df["fpts_act"] = df["fpts_act"].fillna(0.0)
    df["played"] = df["min_act"] > 0
    df["started_act"] = df["pos_act"].notna() & (df["pos_act"] != "BENCH") & df["played"]
    for c in ("minutes_proj", "fpts_mean", "fpm", "hist_games", "market_fpts", "baseline_min", "hist_min"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    df["minutes_proj"] = df["minutes_proj"].fillna(0.0)
    df["fpts_proj"] = df["fpts_mean"].fillna(0.0)
    df["fpm_proj"] = df["fpm"].where(df["fpm"].notna(), df["fpts_proj"] / df["minutes_proj"].replace(0, np.nan))
    df["fpm_act"] = (df["fpts_act"] / df["min_act"]).where(df["played"])
    df["min_err"] = df["minutes_proj"] - df["min_act"]
    df["fpts_err"] = df["fpts_proj"] - df["fpts_act"]
    df["err_minutes"] = df["min_err"] * df["fpm_proj"].fillna(0.0)
    df["err_rate"] = (df["min_act"] * (df["fpm_proj"] - df["fpm_act"])).where(df["played"], 0.0)
    df["err_other"] = df["fpts_err"] - df["err_minutes"] - df["err_rate"]      # the sim's mean vs minutes x rate, bonuses
    df["history_class"] = df.apply(history_class, axis=1)
    df["salary_tier"] = df["salary"].map(salary_tier)
    df["miss"] = ""
    df.loc[(df.minutes_proj >= ROTATION_MIN) & ~df.played, "miss"] = "dnp"
    df.loc[(df.minutes_proj < 8) & (df.min_act >= 16), "miss"] = "surprise"
    df.loc[(df.minutes_proj >= ROTATION_MIN) & df.played & (df.min_err.abs() >= 10), "miss"] = "minutes"
    return df


# ----------------------------------------------------------------------------- summaries
def _agg(d: pd.DataFrame) -> dict:
    played = d[d.played]
    out = dict(n=int(len(d)), n_played=int(played.shape[0]),
               min_proj=float(d.minutes_proj.mean()) if len(d) else np.nan, min_act=float(d.min_act.mean()) if len(d) else np.nan,
               min_bias=float(d.min_err.mean()) if len(d) else np.nan, min_mae=float(d.min_err.abs().mean()) if len(d) else np.nan,
               fpts_proj=float(d.fpts_proj.mean()) if len(d) else np.nan, fpts_act=float(d.fpts_act.mean()) if len(d) else np.nan,
               fpts_bias=float(d.fpts_err.mean()) if len(d) else np.nan, fpts_mae=float(d.fpts_err.abs().mean()) if len(d) else np.nan,
               err_minutes=float(d.err_minutes.mean()) if len(d) else np.nan, err_rate=float(d.err_rate.mean()) if len(d) else np.nan,
               fpm_bias=float((played.fpm_proj - played.fpm_act).mean()) if len(played) else np.nan,
               dnp=int(((d.minutes_proj >= ROTATION_MIN) & ~d.played).sum()))
    return out


def group_table(df: pd.DataFrame, by: str, min_n: int = 1) -> pd.DataFrame:
    rows = {k: _agg(g) for k, g in df.groupby(by, dropna=False) if len(g) >= min_n}
    t = pd.DataFrame(rows).T
    return t.sort_values("n", ascending=False) if len(t) else t


def yardsticks(df: pd.DataFrame) -> dict:
    """fpts MAE of ours vs DK's season average vs the books' implied points, on the same players."""
    out = {}
    rot = df[df.played & (df.minutes_proj >= ROTATION_MIN)]
    both = rot[rot.dk_avg_fpts.notna()]
    if len(both):
        out["vs_dk_avg"] = dict(n=int(len(both)), ours_mae=float(both.fpts_err.abs().mean()), dk_mae=float((both.dk_avg_fpts - both.fpts_act).abs().mean()),
                                ours_bias=float(both.fpts_err.mean()), dk_bias=float((both.dk_avg_fpts - both.fpts_act).mean()))
    priced = rot[rot.market_fpts.notna() & rot.priced.astype(bool)]
    if len(priced):
        out["vs_market"] = dict(n=int(len(priced)), ours_mae=float(priced.fpts_err.abs().mean()), market_mae=float((priced.market_fpts - priced.fpts_act).abs().mean()),
                                ours_bias=float(priced.fpts_err.mean()), market_bias=float((priced.market_fpts - priced.fpts_act).mean()))
    return out


def findings(df: pd.DataFrame) -> List[dict]:
    """Rule-based sentences about where the model is off, each with its n so a one-slate fluke reads as one."""
    out: List[dict] = []

    def add(level, text, n):
        out.append({"level": level, "text": text, "n": int(n)})

    rot = df[df.minutes_proj >= ROTATION_MIN]
    bench = rot[rot.starter.isin(["bench", "unknown"])]
    if len(bench) >= 5:
        b = bench.min_err.mean()
        add("warn" if abs(b) >= 2.5 else "info", f"bench players projected >= {ROTATION_MIN:.0f} min: minutes bias {b:+.1f} (MAE {bench.min_err.abs().mean():.1f}), fpts bias {bench.fpts_err.mean():+.1f}", len(bench))
    st = rot[rot.starter.isin(["confirmed", "expected"])]
    if len(st) >= 5:
        add("warn" if abs(st.min_err.mean()) >= 2.0 else "info", f"starters: minutes bias {st.min_err.mean():+.1f} (MAE {st.min_err.abs().mean():.1f}), fpts bias {st.fpts_err.mean():+.1f}, of which rate {st.err_rate.mean():+.1f}", len(st))
    dnp = rot[~rot.played]
    if len(dnp):
        names = ", ".join(f"{r.player_name} ({r.minutes_proj:.0f})" for r in dnp.sort_values("minutes_proj", ascending=False).head(6).itertuples())
        add("bad" if (dnp.minutes_proj >= 15).any() else "warn", f"projected to play but did not: {names}", len(dnp))
    sur = df[df.miss == "surprise"]
    if len(sur):
        names = ", ".join(f"{r.player_name} ({r.minutes_proj:.0f} -> {r.min_act:.0f})" for r in sur.sort_values("min_act", ascending=False).head(6).itertuples())
        add("warn", f"played a rotation role we did not project: {names}", len(sur))
    for cls in ("new team", "no history"):
        d = rot[rot.history_class == cls]
        same = rot[rot.history_class == "same team"]
        if len(d) >= 3 and len(same):
            add("warn" if d.min_err.abs().mean() > same.min_err.abs().mean() * 1.5 else "info",
                f"{cls}: minutes MAE {d.min_err.abs().mean():.1f} vs {same.min_err.abs().mean():.1f} for players whose history is on the same team; bias {d.min_err.mean():+.1f}", len(d))
    ys = yardsticks(df)
    if "vs_dk_avg" in ys:
        y = ys["vs_dk_avg"]
        add("warn" if y["ours_mae"] > y["dk_mae"] else "info", f"fpts MAE {y['ours_mae']:.1f} ours vs {y['dk_mae']:.1f} DK season average (rotation players)", y["n"])
    if "vs_market" in ys:
        y = ys["vs_market"]
        add("warn" if y["ours_mae"] > y["market_mae"] + 0.5 else "info", f"priced players: fpts MAE {y['ours_mae']:.1f} ours vs {y['market_mae']:.1f} the books' implied points; bias {y['ours_bias']:+.1f} vs {y['market_bias']:+.1f}", y["n"])
    played = df[df.played & (df.minutes_proj >= ROTATION_MIN)]
    if len(played) >= 10:
        rate = played.err_rate.mean(); mins = played.err_minutes.mean()
        add("info", f"fpts bias {played.fpts_err.mean():+.1f} = minutes {mins:+.1f} + rate {rate:+.1f} (+ bonuses/sim {played.err_other.mean():+.1f})", len(played))
    for team, g in df.groupby("team"):
        if g.played.any():
            add("info" if abs(g.minutes_proj.sum() - g.min_act.sum()) < 12 else "warn",
                f"{team}: projected {g.minutes_proj.sum():.0f} minutes, played {g.min_act.sum():.0f}", len(g))
    return out


def worst(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    cols = ["player_name", "team", "starter", "history_class", "salary", "priced", "hist_min", "baseline_min", "minutes_proj", "min_act",
            "fpm_proj", "fpm_act", "fpts_proj", "fpts_act", "fpts_err", "err_minutes", "err_rate", "miss"]
    return df.reindex(df.fpts_err.abs().sort_values(ascending=False).index)[cols].head(n)


# ----------------------------------------------------------------------------- one slate
@dataclass
class Scorecard:
    draft_group: int
    day: Optional[date]
    lock: Optional[datetime]
    captured_at: Optional[datetime]
    rows: pd.DataFrame
    summary: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.day.isoformat() if self.day else 'nodate'}_{self.draft_group}"


def build_scorecard(session: Session, draft_group: int) -> Optional[Scorecard]:
    ours, dk, lock = captures_for(session, draft_group)
    if ours.empty:
        return None
    day = slate_day(lock, ours)
    if day is None:
        return None
    act = actuals_for(session, day)
    if act.empty:
        return Scorecard(int(draft_group), day, lock, ours.captured_at.max().to_pydatetime(), pd.DataFrame())
    rows = join_actuals(ours, act, dk)
    rot = rows[rows.played & (rows.minutes_proj >= ROTATION_MIN)]
    summary = {
        "draft_group": int(draft_group), "day": day.isoformat(), "lock": lock.isoformat(timespec="minutes") if lock else None,
        "captured_at": ours.captured_at.max().isoformat(timespec="minutes"), "minutes_before_lock": float((lock - ours.captured_at.max()).total_seconds() / 60) if lock else None,
        "n": int(len(rows)), "n_box": int(rows.box_row.sum()), "n_rotation": int(len(rot)), "games": int(act.groupby("team_act").ngroups / 2) if len(act) else 0,
        "all": _agg(rows), "rotation": _agg(rot),
        "by_starter": group_table(rows, "starter").to_dict("index"), "by_priced": group_table(rot, "priced").to_dict("index"),
        "by_salary": group_table(rot, "salary_tier").to_dict("index"), "by_history": group_table(rot, "history_class").to_dict("index"),
        "by_team": group_table(rows, "team").to_dict("index"),
        "yardsticks": yardsticks(rows), "findings": findings(rows),
        "worst": worst(rows).to_dict("records"),
    }
    return Scorecard(int(draft_group), day, lock, ours.captured_at.max().to_pydatetime(), rows, summary)


def save_scorecard(card: Scorecard, out_dir: Path = SCORECARD_DIR) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    card.rows.to_csv(out_dir / f"{card.key}.csv", index=False)
    (out_dir / f"{card.key}.json").write_text(json.dumps(_jsonable(card.summary), indent=1))
    return out_dir / f"{card.key}.json"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (datetime, date, pd.Timestamp)):
        return o.isoformat()
    return o


def format_scorecard(card: Scorecard, top: int = TOP_N) -> str:
    s = card.summary
    if not s:
        return f"slate {card.draft_group} ({card.day}): captured {card.captured_at:%m-%d %H:%M}, no box scores yet"
    L = [f"slate {card.draft_group} on {s['day']}: capture {s['captured_at']} ({s['minutes_before_lock']:.0f} min before lock), "
         f"{s['n']} players, {s['n_box']} with a box row, {s['n_rotation']} rotation players played"]
    a = s["rotation"]
    L.append(f"  rotation players (proj >= {ROTATION_MIN:.0f} min): minutes bias {a['min_bias']:+.1f} MAE {a['min_mae']:.1f} | fpts bias {a['fpts_bias']:+.1f} MAE {a['fpts_mae']:.1f} "
             f"= minutes {a['err_minutes']:+.1f} + rate {a['err_rate']:+.1f} | fpm bias {a['fpm_bias']:+.2f} | DNP {a['dnp']}")
    for name, key in (("by starter", "by_starter"), ("by history", "by_history"), ("by priced", "by_priced"), ("by salary", "by_salary")):
        t = pd.DataFrame(s[key]).T
        if len(t):
            L.append(f"  {name}:")
            for k, r in t.iterrows():
                L.append(f"    {str(k):12s} n={int(r['n']):3d}  min {r['min_proj']:5.1f} vs {r['min_act']:5.1f} (bias {r['min_bias']:+5.1f}, mae {r['min_mae']:4.1f})"
                         f"  fpts {r['fpts_proj']:5.1f} vs {r['fpts_act']:5.1f} (bias {r['fpts_bias']:+5.1f}, mae {r['fpts_mae']:4.1f})  dnp {int(r['dnp'])}")
    L.append("  findings:")
    for f in s["findings"]:
        L.append(f"    [{f['level']}] {f['text']}  (n={f['n']})")
    L.append(f"  worst {top} fpts misses:")
    w = pd.DataFrame(s["worst"]).head(top)
    for r in w.itertuples():
        L.append(f"    {r.player_name:24s} {r.team:4s} {str(r.starter):9s} {r.history_class:11s} {'$' if r.priced else ' '} "
                 f"min {r.minutes_proj:5.1f}->{r.min_act:5.1f}  fpm {r.fpm_proj or 0:4.2f}->{(r.fpm_act if r.fpm_act == r.fpm_act else 0):4.2f}  "
                 f"fpts {r.fpts_proj:5.1f}->{r.fpts_act:5.1f}  err {r.fpts_err:+6.1f} = min {r.err_minutes:+5.1f} + rate {r.err_rate:+5.1f}  {r.miss}")
    return "\n".join(L)


# ----------------------------------------------------------------------------- many slates
def scored_keys(out_dir: Path = SCORECARD_DIR) -> set:
    return {p.stem for p in Path(out_dir).glob("*.json")}


def pending_draft_groups(session: Session, out_dir: Path = SCORECARD_DIR) -> List[int]:
    """Draft groups with an 'ours' capture whose slate day has box scores but no scorecard file yet."""
    from sqlalchemy import text
    caps = pd.read_sql(text("select draft_group_id, min(lock) lock, min(gametime) gametime from nba_projection_capture where source='ours' group by draft_group_id"), session.get_bind())
    if caps.empty:
        return []
    days = {d for (d,) in session.execute(text("select distinct game_date from nba_stats")).all()}
    days = {str(d)[:10] for d in days}
    done = scored_keys(out_dir)
    out = []
    for r in caps.itertuples():
        when = r.lock or r.gametime
        if when is None:
            continue
        day = str(pd.to_datetime(when).date())
        if day in days and f"{day}_{int(r.draft_group_id)}" not in done:
            out.append(int(r.draft_group_id))
    return out


def run_pending_scorecards(session: Session, out_dir: Path = SCORECARD_DIR, log: Log = print) -> List[Scorecard]:
    cards = []
    for dg in pending_draft_groups(session, out_dir):
        card = build_scorecard(session, dg)
        if card is None or card.rows.empty:
            continue
        save_scorecard(card, out_dir); cards.append(card)
        log(format_scorecard(card))
    return cards


def scorecard_history(out_dir: Path = SCORECARD_DIR) -> pd.DataFrame:
    """One row per scored slate: the headline numbers, oldest first."""
    rows = []
    for p in sorted(Path(out_dir).glob("*.json")):
        s = json.loads(p.read_text())
        a = s.get("rotation", {}); y = s.get("yardsticks", {})
        bench = s.get("by_starter", {}).get("bench", {}); st = s.get("by_starter", {}).get("expected", {})
        rows.append(dict(day=s["day"], draft_group=s["draft_group"], games=s.get("games"), n_rot=s["n_rotation"],
                         min_bias=a.get("min_bias"), min_mae=a.get("min_mae"), fpts_bias=a.get("fpts_bias"), fpts_mae=a.get("fpts_mae"),
                         err_minutes=a.get("err_minutes"), err_rate=a.get("err_rate"), dnp=a.get("dnp"),
                         bench_min_bias=bench.get("min_bias"), starter_min_bias=st.get("min_bias"),
                         dk_mae=y.get("vs_dk_avg", {}).get("dk_mae"), market_mae=y.get("vs_market", {}).get("market_mae"),
                         ours_mae_priced=y.get("vs_market", {}).get("ours_mae")))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- pre-game rotation review
def recent_minutes(box: pd.DataFrame, n: int = 5) -> Dict[str, List[float]]:
    """player -> his last n played games' minutes (most recent last)."""
    played = box[box["played"]].sort_values(["game_date", "game_id"])
    return {name: g["minutes"].tail(n).round(1).tolist() for name, g in played.groupby("player_name")}


def market_points(session: Session) -> Dict[str, float]:
    return {r.player_name: float(r.market_mean) for r in session.query(NBAMarketRate).filter(NBAMarketRate.stat == "points").all()}


def review_team(slate, team: str, proj: pd.DataFrame, recent: Dict[str, List[float]], mk_points: Dict[str, float],
                overrides: Optional[Dict[str, float]] = None) -> dict:
    """
    rows: one per player on the team (out players included) with the minutes' provenance and flags.
    alerts: team-level notes (rotation depth, history sum). Flags, in rough order of cost when wrong:
      new team, no history, dnp-prone, above recent max, deep bench, market minutes gap, questionable, pinned
    """
    from ivan.core.utils import normalize_name
    from ivan.sports.nba.projection.live import OUT_STATUSES
    overrides = overrides or {}
    state = slate.state
    starters = slate.starters.get(team)
    rows = []
    for p in slate.players:
        if p.team != team:
            continue
        name = normalize_name(p.name)
        s = state.loc[name] if name in state.index else None
        r = proj.loc[name] if name in proj.index else None
        status = (p.status or "").upper()
        out = r is None or status in OUT_STATUSES
        starter = None if starters is None else (name in starters)
        last = recent.get(name, [])
        hist_min = None if s is None else float(s["min_ewm"]) * float(s["play_ewm"])
        mins = None if r is None else float(r["minutes"])
        implied = None
        if s is not None and name in mk_points and float(s["eff_min"]) >= 100 and float(s["rate_points"]) > 0:
            implied = mk_points[name] / float(s["rate_points"])
        flags = []
        if not out:
            if s is None:
                flags.append("no history: salary prior")
            elif str(s["team"]) != str(team):
                flags.append(f"new team: history is {s['team']} ({int(s['games'])} g, {float(s['min_ewm']):.0f} min)")
            if s is not None and float(s["play_ewm"]) < DNP_PRONE and (mins or 0) >= 8:
                flags.append(f"dnp-prone: played {100 * float(s['play_ewm']):.0f}% of dressed games")
            if mins is not None and last and starter is not True and name not in overrides and mins > max(last) + RECENT_MAX_MARGIN:
                flags.append(f"above recent max: last {len(last)} games {', '.join(f'{m:.0f}' for m in last)}")
            if mins is not None and mins >= DEEP_BENCH_MIN and starter is not True and s is not None and float(s["start_ewm"]) < 0.2:
                flags.append(f"deep bench projected {mins:.0f} min (starts {100 * float(s['start_ewm']):.0f}% of games)")
            if implied is not None and mins is not None and abs(implied - mins) >= MARKET_MIN_GAP:
                flags.append(f"market implies {implied:.0f} min at his history rate (ours {mins:.0f})")
            if status in ("Q", "GTD", "D", "P"):
                flags.append(f"questionable ({status})")
            if name in overrides:
                flags.append(f"pinned {overrides[name]:g}")
        rows.append(dict(name=name, status=status, starter=starter, out=out, salary=p.salary, position=p.position,
                         hist_team=None if s is None else str(s["team"]), hist_games=0 if s is None else int(s["games"]),
                         hist_min=hist_min, start_ewm=None if s is None else float(s["start_ewm"]), play_ewm=None if s is None else float(s["play_ewm"]),
                         last=last, baseline=None if r is None else float(r["baseline"]), minutes=mins, pinned=name in overrides,
                         priced=bool(r is not None and r.get("priced")), market_points=mk_points.get(name), implied_min=implied,
                         fpm=None if r is None else float(r["fpm"]), fpts=None if r is None else float(r["fpts_mean"]), flags=flags))
    rows.sort(key=lambda x: (x["out"], -(x["minutes"] or 0)))
    avail = [x for x in rows if not x["out"]]
    depth = sum(1 for x in avail if (x["minutes"] or 0) >= 12)
    alerts = []
    if depth < 8 or depth > 11:
        alerts.append(("warn", f"rotation depth {depth} players at >= 12 min (coaches run 8-11)"))
    hist_sum = sum((x["hist_min"] or 0) for x in avail)
    if abs(hist_sum - 241) >= 30:
        alerts.append(("info", f"history minutes of the available roster sum to {hist_sum:.0f}; the allocation {'spread' if hist_sum < 241 else 'cut'} {abs(241 - hist_sum):.0f} min"))
    n_new = sum(1 for x in avail if any(f.startswith("new team") for f in x["flags"]) and (x["minutes"] or 0) >= ROTATION_MIN)
    if n_new:
        alerts.append(("warn", f"{n_new} rotation players projected from another team's history"))
    for x in avail:
        for f in x["flags"]:
            level = "bad" if f.startswith(("dnp-prone", "deep bench")) else ("warn" if f.startswith(("new team", "above recent", "market implies", "no history")) else "info")
            alerts.append((level, f"{x['name']}: {f}"))
    return dict(team=team, rows=rows, alerts=alerts, depth=depth, hist_sum=hist_sum,
                score=sum({"bad": 10, "warn": 3, "info": 1}[k] for k, _ in alerts))


def review_slate(session: Session, slate, projections: Dict[str, pd.DataFrame], overrides: Optional[Dict[str, float]] = None,
                 teams: Optional[List[str]] = None) -> List[dict]:
    recent = recent_minutes(slate.box)
    mk = market_points(session)
    out = [review_team(slate, t, projections[t], recent, mk, overrides) for t in (teams or slate.teams()) if t in projections]
    return sorted(out, key=lambda c: -c["score"])


def format_review(card: dict) -> str:
    L = [f"=== {card['team']}: depth {card['depth']}, history sum {card['hist_sum']:.0f}, alert score {card['score']}"]
    L.append(f"  {'player':24s} {'st':4s} {'sal':>5s} {'hist':>14s} {'last 5':>22s} {'play':>4s} {'base':>5s} {'min':>5s} {'mkt':>5s} {'fpm':>4s} {'fpts':>5s}  flags")
    for r in card["rows"]:
        if r["out"]:
            continue
        hist = f"{r['hist_team'] or '-'} {r['hist_games']:2d}g {r['hist_min'] or 0:4.1f}" if r["hist_games"] else "none"
        last = ",".join(f"{m:.0f}" for m in r["last"]) if r["last"] else "-"
        st = "S" if r["starter"] else ("?" if r["starter"] is None else "")
        L.append(f"  {r['name'][:24]:24s} {st:4s} {r['salary'] or 0:5d} {hist:>14s} {last:>22s} {('' if r['play_ewm'] is None else f'{r['play_ewm']:.2f}'):>4s} "
                 f"{r['baseline'] or 0:5.1f} {r['minutes'] or 0:5.1f} {('' if r['implied_min'] is None else f'{r['implied_min']:.0f}'):>5s} {r['fpm'] or 0:4.2f} {r['fpts'] or 0:5.1f}  "
                 + " · ".join(r["flags"]))
    outs = [r["name"] for r in card["rows"] if r["out"]]
    if outs:
        L.append("  out: " + ", ".join(outs))
    return "\n".join(L)
