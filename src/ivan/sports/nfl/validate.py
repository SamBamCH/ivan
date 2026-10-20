# src/ivan/sports/nfl/validate.py
"""
The NFL side of ivan.core.validation: find what the pipeline wrote down
before lock (run artifacts: projections, sim, choose), find what happened
(box scores, scraped contests), and build the four frames the core module
scores.

For every contest with a pre-lock choose run (the same "last run before
lock" rule as the tax backtest) the sim artifact that run read from and the
projections file that sim read from are paired by time, so live and
replayed slates are handled by one code path. Actual points come from the
scraped contest when there is one (DraftKings' own per-slot points) and
from nfl_stats otherwise, through backtest_tax.score_lineup, so a lineup is
scored here exactly as the settlement scores it.

Slates with no run artifacts yet (the 2026 weeks before artifacts existed)
still feed the projection check through projection_snapshot; those rows
carry source="snapshot" and a normal-approximation PIT.

A replay that stopped before the chooser (scripts/replay.py --through) has
no choose run; its contests come from the slate's replay.json instead, with
whatever the stages that ran left behind: the projections file, and the
field artifact (the sampled field, no outcomes) or the sim artifact. A
sampled field is scored on the actual box scores here, so the field checks
(cash line, winning score, duplication, exposure) test the field model
without a simulation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ivan.core import artifacts, validation as V
from ivan.sports.nfl import actuals
from ivan.sports.nfl import backtest_tax as bt
from ivan.sports.nfl.models import HistoricalContest, ProjectionSnapshot, ShowdownLineup, ShowdownPlayer
from ivan.sports.nfl.placement import load_payout_tiers, payout_vector, places_paid, rank_payout_from_table, remove_our_copies
from ivan.sports.nfl.refactor import environment as E
from ivan.sports.nfl.run_artifacts import load_usage, load_environment, load_projections, load_sim_artifact

SLOT_LABEL = {511: "CPT", 512: "FLEX", 66: "QB", 67: "RB", 68: "WR", 69: "TE", 70: "FLEX", 71: "DST"}
Key = Tuple[int, str, str, int]                 # (roster slot code, team, position, salary): how scraped players are matched


def slot_code(label: str) -> int:
    return bt.SLOT_CODE.get(str(label).upper().rstrip("0123456789"), 512)


# --------------------------------------------------------------------------- what we have per contest
@dataclass
class ContestCase:
    contest_id: int
    name: str
    lock: Optional[datetime]                 # UTC naive
    run: Optional[bt.ChooseRun]              # None for a replay that stopped before the chooser
    sim: Optional[artifacts.SimRun] = None
    field_run: Optional[artifacts.SimRun] = None     # the field stage's sampled field (no outcomes), when there is no sim
    projections: Optional[artifacts.ProjectionFile] = None
    notes: List[str] = field(default_factory=list)

    @property
    def artifact(self) -> Optional[artifacts.SimRun]:
        """The sim artifact, else the field artifact: both hold the sampled field."""
        return self.sim or self.field_run

    @property
    def slate(self) -> str:
        return bt.utc_to_local(self.lock).strftime("%Y-%m-%d %H:%M") if self.lock else "unknown"


def _manifest_contests(dirs: Sequence[Path]) -> Dict[int, Tuple[Path, dict]]:
    """Contest id -> (run directory, manifest) for every replayed slate in `dirs` (the manifest sits in the directory or its parent)."""
    out: Dict[int, Tuple[Path, dict]] = {}
    for d in dirs:
        for cand in (Path(d) / artifacts.REPLAY_MANIFEST, Path(d).parent / artifacts.REPLAY_MANIFEST):
            if not cand.exists():
                continue
            try:
                m = json.loads(cand.read_text())
            except (OSError, json.JSONDecodeError):
                break
            for c in m.get("contests") or []:
                try:
                    out.setdefault(int(c["contest_id"]), (Path(d), m))
                except (KeyError, TypeError, ValueError):
                    continue
            break
    return out


def discover(session: Session, dirs: Sequence[Path], since: Optional[datetime] = None, until: Optional[datetime] = None) -> List[ContestCase]:
    """
    One case per contest with a pre-lock choose run in `dirs`, paired with its
    sim run and projections file; plus one per contest of a replayed slate
    whose run stopped before the chooser (no choose run, the manifest lists
    the contests), paired with the field or sim artifact and projections
    the earlier stages wrote.
    """
    dirs = [Path(d) for d in dirs]
    runs = bt.find_choose_runs(dirs)
    sims = artifacts.find_sim_runs(dirs)
    fields = artifacts.find_field_runs(dirs)
    projs = artifacts.find_projection_files(dirs + [d.parent for d in dirs])
    from_manifest = _manifest_contests(dirs)
    info = bt.contest_info(session, {c for r in runs for c in bt._contests_in_run(r)} | set(from_manifest))
    by_contest = bt.select_prelock_runs(runs, info)

    def in_window(lock: Optional[datetime]) -> bool:
        return not ((since and lock and lock < since) or (until and lock and lock > until))

    cases = []
    for cid, run in by_contest.items():
        hc = info.get(cid)
        lock = hc.date if hc is not None else None
        if not in_window(lock):
            continue
        case = ContestCase(cid, hc.name if hc is not None else "", lock, run)
        with_cid = [s for s in sims if cid in s.contests()]
        same_tree = [s for s in with_cid if s.dir.parent == run.lineups_csv.parent]       # a replay repeat: sim/ beside the choose files
        case.sim = artifacts.latest_before(same_tree or with_cid, run.when_local)
        case.field_run = artifacts.latest_before([f for f in fields if cid in f.contests() and f.dir.parent == run.lineups_csv.parent], run.when_local)
        anchor = case.sim.when if case.sim else run.when_local
        same_tree = [p for p in projs if p.path.parent == run.lineups_csv.parent or p.path.parent == run.lineups_csv.parent.parent]
        case.projections = artifacts.latest_before(same_tree or projs, anchor)
        if case.sim is None:
            case.notes.append("no sim artifact (run predates artifacts)")
        if case.projections is None:
            case.notes.append("no projections file")
        cases.append(case)
    for cid, (d, m) in from_manifest.items():
        if cid in by_contest:
            continue
        hc = info.get(cid)
        lock = hc.date if hc is not None else None
        if not in_window(lock):
            continue
        case = ContestCase(cid, hc.name if hc is not None else "", lock, None)
        case.sim = artifacts.latest_before([s for s in sims if s.dir.parent == d and cid in s.contests()], None)
        case.field_run = artifacts.latest_before([f for f in fields if f.dir.parent == d and cid in f.contests()], None)
        case.projections = artifacts.latest_before([p for p in projs if p.path.parent in (d, d.parent)], None)
        case.notes.append(f"no choose run (replay stopped after the {m.get('through') or '?'} stage)")
        if case.sim is None and case.field_run is None:
            case.notes.append("no field or sim artifact")
        if case.projections is None:
            case.notes.append("no projections file")
        cases.append(case)
    return sorted(cases, key=lambda c: (c.lock or datetime.max, c.contest_id))


# --------------------------------------------------------------------------- actuals
@dataclass
class Actuals:
    scraped: Dict[Key, float]                      # DK points per (slot, team, pos, salary)
    scraped_own: Dict[Key, float]                  # actual ownership % per slot key
    week_points: Dict                              # nfl_stats fallback
    field: Optional[np.ndarray]                    # every entry's points, descending
    tiers: List[dict]
    real_dupes: Dict[frozenset, int]               # lineup (as a set of slot keys) -> copies in the real field
    field_size: Optional[int]
    unique_lineups: Optional[int]

    @property
    def has_points(self) -> bool:
        return bool(self.scraped) or bool(self.week_points)


def load_actuals(session: Session, case: ContestCase, week_cache: Dict) -> Actuals:
    sp = session.query(ShowdownPlayer).filter(ShowdownPlayer.contest_id == case.contest_id).all()
    key_of: Dict[int, Key] = {}
    scraped, scraped_own = {}, {}
    for r in sp:
        k = (int(r.roster_slot_id), str(r.team), str(r.position).upper(), int(r.salary))
        key_of[int(r.player_id)] = k
        if r.fpts_actual is not None:
            scraped[k] = float(r.fpts_actual)
        if r.ownership_actual is not None:
            scraped_own[k] = float(r.ownership_actual)
    wp: Dict = {}
    if case.lock is not None:
        sw = actuals.week_for_date(session, bt.utc_to_local(case.lock), week_cache.setdefault("_weeks", actuals.week_lookup(session)))
        if sw is not None:
            if sw not in week_cache:
                week_cache[sw] = actuals.actual_points(session, *sw)
            wp = week_cache[sw]
    real_dupes: Dict[frozenset, int] = {}
    n_entries = 0
    for lu in session.query(ShowdownLineup).filter(ShowdownLineup.contest_id == case.contest_id).all():
        try:
            ks = frozenset([key_of[int(lu.cpt_player_id)]] + [key_of[int(p)] for p in lu.flex_player_ids])
        except KeyError:
            continue
        real_dupes[ks] = real_dupes.get(ks, 0) + int(lu.dupe_count or 1)
        n_entries += int(lu.dupe_count or 1)
    hc = session.get(HistoricalContest, case.contest_id)
    return Actuals(scraped, scraped_own, wp, bt.field_scores(session, case.contest_id), load_payout_tiers(session, case.contest_id),
                   real_dupes, int(hc.total_entries) if hc is not None else None, len(real_dupes) or None)


def place_alone(scores: np.ndarray, field_desc: np.ndarray, pay: np.ndarray, atol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Rank and prize of each score entered on its own into a field (ties share), vectorised."""
    asc = np.sort(np.asarray(field_desc, dtype=float))
    s = np.asarray(scores, dtype=float)
    n_above = asc.size - np.searchsorted(asc, s + atol, side="right") + 0      # entries strictly above
    n_tied = np.searchsorted(asc, s + atol, side="right") - np.searchsorted(asc, s - atol, side="left")
    ranks = n_above + 1
    pays = np.array([rank_payout_from_table(int(r), int(t) + 1, pay) for r, t in zip(ranks, n_tied)])
    return ranks, pays


# --------------------------------------------------------------------------- the frames for one contest
def _player_table(case: ContestCase) -> pd.DataFrame:
    """Players of the slate from the projections file (preferred) or the sim artifact's player list."""
    if case.projections is not None:
        df = load_projections(case.projections.path)
        df = df.rename(columns={"fpts_mean": "proj_mean", "fpts_std": "proj_sd"})
        for q, c in zip(V.QGRID, V.Q_COLS):
            df[c] = df.get(f"fpts_p{q:02d}")
        df["slot_code"] = df.roster_slot_id.astype(int)
        df["slot"] = df.slot_code.map(SLOT_LABEL).fillna(df.slot_label)
        df["own_proj"] = df.ownership_proj
        for c in ("volume_source", "unpriced_stats"):                 # from projections written after layer 2 existed
            if c not in df.columns:
                df[c] = None
        return df[["player_id", "slot", "slot_code", "name", "team", "position", "salary", "proj_mean", "proj_sd", "own_proj",
                   "volume_source", "unpriced_stats"] + V.Q_COLS]
    if case.artifact is not None:
        _, meta = load_sim_artifact(*artifacts.contest_artifact_paths(case.artifact.dir, case.contest_id))
        rows = [{"player_id": p["id"], "slot": p.get("slot"), "slot_code": slot_code(p.get("slot") or "FLEX"), "name": p.get("name"), "team": p.get("team"),
                 "position": p.get("pos"), "salary": p.get("salary"), "proj_mean": p.get("fpts_mean"), "proj_sd": np.nan, "own_proj": np.nan}
                for p in meta.get("players", [])]
        df = pd.DataFrame(rows)
        for c in V.Q_COLS:
            df[c] = np.nan
        return df
    return pd.DataFrame()


def _score_sampled(arrays: dict, players: pd.DataFrame, act: Actuals) -> Tuple[np.ndarray, List[Optional[frozenset]]]:
    """Every unique lineup of a sampled field scored the way the settlement scores it (NaN when a player is unknown), with its slot-key set."""
    pm = {int(r.player_id): (str(r.slot), int(r.player_id), str(r.name), str(r.team), str(r.position), int(r.salary)) for r in players.itertuples()}
    scores, keys_ = [], []
    for row in arrays["lineup_players"]:
        pl = [pm[int(pid)] for pid in row if int(pid) >= 0 and int(pid) in pm]
        if len(pl) != int((row >= 0).sum()):
            scores.append(np.nan); keys_.append(None); continue
        s, _missing, _ = bt.score_lineup(pl, act.week_points, act.scraped)
        scores.append(np.nan if s is None else s)
        keys_.append(frozenset((slot_code(sl), t, ps.upper(), sal) for sl, _, _, t, ps, sal in pl))
    return np.asarray(scores, dtype=float), keys_


# --------------------------------------------------------------------------- the frames for one contest
def contest_frames(session: Session, case: ContestCase, act: Actuals, in_field: str = "auto") -> Dict[str, pd.DataFrame]:
    """players / field / lineups / pool frames for one contest (empty frames where the inputs are missing)."""
    out = {"players": pd.DataFrame(), "field": pd.DataFrame(), "lineups": pd.DataFrame(), "pool": pd.DataFrame()}
    players = _player_table(case)
    art = case.artifact
    arrays, meta = (load_sim_artifact(*artifacts.contest_artifact_paths(art.dir, case.contest_id)) if art is not None else ({}, {}))
    has_sim = bool(arrays) and "score_pct" in arrays            # a field artifact holds the sampled field but no outcomes
    tag = {"contest_id": case.contest_id, "contest_name": case.name, "slate": case.slate}

    # ---- players: projected vs actual points and ownership
    if not players.empty:
        p = players.copy()
        keys = [(int(r.slot_code), str(r.team), str(r.position).upper(), int(r.salary)) for r in p.itertuples()]
        p["actual_fpts"] = [act.scraped.get(k, actuals.lookup_points(act.week_points, str(r.name), str(r.team), str(r.position)))
                            for k, r in zip(keys, p.itertuples())]
        # a player with no stat line while his team's box score is in scored 0 (same rule as the settlement)
        p["actual_fpts"] = [0.0 if (v is None and actuals.team_played(act.week_points, str(r.team))) else v for v, r in zip(p.actual_fpts, p.itertuples())]
        p["actual_fpts"] = pd.to_numeric(p.actual_fpts, errors="coerce")
        # DK points for a captain row are 1.5x the player's points
        p.loc[p.slot_code == bt.CPT_SLOT, "actual_fpts"] = p.loc[p.slot_code == bt.CPT_SLOT, "actual_fpts"] * bt.CPT_MULT
        p["actual_own"] = [act.scraped_own.get(k, np.nan) for k in keys]
        if arrays:
            expo = dict(zip(arrays["player_id"].tolist(), arrays["player_exposure_pct"].tolist()))
            p["own_field"] = p.player_id.map(expo)
            if "player_mean" in arrays:
                p["sim_mean"] = p.player_id.map(dict(zip(arrays["player_id"].tolist(), arrays["player_mean"].tolist())))
        p["source"] = "artifact"
        for k, v in tag.items():
            p[k] = v
        p["actual_own"] = pd.to_numeric(p.actual_own, errors="coerce")
        out["players"] = p[p.actual_fpts.notna() | p.actual_own.notna()] if act.has_points or act.scraped_own else p.iloc[0:0]

    # ---- field: the sim's field against the real one; a field artifact's sampled field is scored on the box scores instead
    sampled_scores: Optional[np.ndarray] = None
    if arrays and not has_sim and act.has_points and not players.empty:
        sampled_scores, _ = _score_sampled(arrays, players, act)
    if arrays and act.field is not None and act.tiers and (has_sim or sampled_scores is not None):
        q = list(arrays["qgrid"].tolist())
        fld = np.asarray(act.field, dtype=float)
        paid = min(places_paid(act.tiers), fld.size)
        dupes = arrays["dupes"]
        if has_sim:
            sp = np.percentile(arrays["score_pct"], [10, 50, 90], axis=0)      # (3, len(q))
            qi = {v: i for i, v in enumerate(q)}
            cash = [float(np.percentile(arrays["cash_line"], k)) for k in (10, 50, 90)]
            win = [float(np.percentile(arrays["winning_score"], k)) for k in (10, 50, 90)]
            score = {k: float(sp[1, qi[k]]) for k in (10, 50, 90, 99)}
        else:
            # one deterministic field: every sampled entry scored on what the players actually did
            ok = np.isfinite(sampled_scores)
            desc = np.sort(np.repeat(sampled_scores[ok], dupes[ok]))[::-1] if ok.any() else np.zeros(0)
            paid_sim = min(desc.size, max(1, int(round(paid * desc.size / max(1, act.field_size or fld.size))))) if desc.size else 0
            cash = [float(desc[paid_sim - 1]) if paid_sim else np.nan] * 3
            win = [float(desc[0]) if desc.size else np.nan] * 3
            score = {k: (float(np.percentile(desc, k)) if desc.size else np.nan) for k in (10, 50, 90, 99)}
        real_d = np.asarray(list(act.real_dupes.values()), dtype=float) if act.real_dupes else np.zeros(0)
        row = {**tag, "sim_size": int(meta.get("simulated_size", 0)), "actual_size": int(fld.size), "contest_size": act.field_size,
               "paid_places": int(paid), "n_sims": int(meta.get("n_sims", 0)),
               "sim_cash_p10": cash[0], "sim_cash_p50": cash[1], "sim_cash_p90": cash[2], "actual_cash_line": float(fld[paid - 1]) if paid else np.nan,
               "sim_win_p10": win[0], "sim_win_p50": win[1], "sim_win_p90": win[2], "actual_win": float(fld[0]),
               "sim_score_p50": score[50], "sim_score_p90": score[90], "sim_score_p99": score[99], "sim_score_p10": score[10],
               "actual_score_p50": float(np.percentile(fld, 50)), "actual_score_p90": float(np.percentile(fld, 90)),
               "actual_score_p99": float(np.percentile(fld, 99)), "actual_score_p10": float(np.percentile(fld, 10)),
               # the sim samples `sim_size` entries; dupe counts are scaled to the contest size before comparing
               "sim_unique_share": float(len(dupes) / max(1, dupes.sum())), "sim_dupe_max": int(dupes.max()) if dupes.size else 0,
               "sim_dupe_p90": float(np.percentile(dupes, 90)) if dupes.size else np.nan,
               "scale": float(act.field_size / max(1, dupes.sum())) if act.field_size else np.nan,
               "sim_dupe_max_scaled": float(dupes.max() * act.field_size / max(1, dupes.sum())) if (dupes.size and act.field_size) else np.nan,
               "sim_dupe_p90_scaled": float(np.percentile(dupes, 90) * act.field_size / max(1, dupes.sum())) if (dupes.size and act.field_size) else np.nan,
               "sim_top_share": float(dupes.max() / max(1, dupes.sum())), "actual_top_share": float(real_d.max() / real_d.sum()) if real_d.size else np.nan,
               "actual_unique_share": float(real_d.size / real_d.sum()) if real_d.size else np.nan,
               "actual_dupe_max": float(real_d.max()) if real_d.size else np.nan,
               "actual_dupe_p90": float(np.percentile(real_d, 90)) if real_d.size else np.nan,
               "field_coverage": float(fld.size / act.field_size) if act.field_size else np.nan}
        out["field"] = pd.DataFrame([row])

    # ---- our lineups and the pool, scored the way the settlement scores them
    ours = bt.our_entries(case.run, case.contest_id) if case.run is not None else []
    pay = payout_vector(act.tiers, (act.field.size if act.field is not None else 0) + 1) if act.tiers else None
    by_lid = {}
    if arrays:
        for i, lid in enumerate(arrays["lineup_id"].tolist()):
            by_lid[int(lid)] = i
    if ours and act.has_points:
        rows = []
        field_for_us = act.field
        if act.field is not None and in_field in ("auto", "yes"):
            our_scores = [bt.score_lineup(o["players"], act.week_points, act.scraped)[0] for o in ours]
            field_for_us, _ = remove_our_copies(act.field, [s for s in our_scores if s is not None])
        for o in ours:
            s, missing, _ = bt.score_lineup(o["players"], act.week_points, act.scraped)
            ks = frozenset((slot_code(sl), str(t), str(ps).upper(), int(sal)) for sl, _, _, t, ps, sal in o["players"])
            i = by_lid.get(int(o["lineup_id"]))
            r = {**tag, "entry_id": o["entry_id"], "lineup_id": o["lineup_id"], "entry_fee": o["entry_fee"],
                 "sim_cash_pct": o["sim_cash_pct"], "sim_avg_roi": o["sim_avg_roi"],
                 "pred_dupes": float(arrays["predicted_dupes"][i]) if i is not None else np.nan,
                 "sim_dupes": int(arrays["dupes"][i]) if i is not None else np.nan,
                 "actual_points": s, "actual_dupes": (act.real_dupes.get(ks, 0) if act.real_dupes else np.nan),
                 "actual_rank": np.nan, "actual_payout": np.nan, "field_size": (field_for_us.size + 1) if field_for_us is not None else np.nan,
                 "missing": ", ".join(missing)}
            if s is not None and field_for_us is not None and pay is not None:
                rk, py = place_alone(np.array([s]), field_for_us, payout_vector(act.tiers, field_for_us.size + 1))
                r["actual_rank"], r["actual_payout"] = int(rk[0]), float(py[0])
            rows.append(r)
        luck = contest_luck(ours, rows)
        honest = honest_roi(case.run, case.contest_id)
        for r in rows:
            r.update(luck)
            r["sim_honest_roi"] = honest
        out["lineups"] = pd.DataFrame(rows)

    if has_sim and act.has_points and act.field is not None and pay is not None and not players.empty:
        picked_counts: Dict[int, int] = {}
        for o in ours:
            picked_counts[int(o["lineup_id"])] = picked_counts.get(int(o["lineup_id"]), 0) + 1
        scores, keys_ = _score_sampled(arrays, players, act)
        ok = np.isfinite(scores)
        ranks = np.full(scores.size, np.nan); pays = np.full(scores.size, np.nan)
        if ok.any():
            rk, py = place_alone(scores[ok], act.field, pay)
            ranks[ok], pays[ok] = rk, py
        fee = float(meta.get("contest", {}).get("entry_fee") or (ours[0]["entry_fee"] if ours else np.nan))
        pool = pd.DataFrame({**{k: [v] * scores.size for k, v in tag.items()},
                             "lineup_id": arrays["lineup_id"], "entry_fee": fee, "sim_avg_roi": arrays["avg_roi"], "sim_cash_pct": arrays["cash_pct"],
                             "sim_dupes": arrays["dupes"], "pred_dupes": arrays["predicted_dupes"], "sim_mean_score": arrays["mean_score"],
                             "actual_points": scores, "actual_rank": ranks, "actual_payout": pays,
                             "actual_dupes": [act.real_dupes.get(k, 0) if (k is not None and act.real_dupes) else np.nan for k in keys_]})
        pool["n_picked"] = pool.lineup_id.map(picked_counts).fillna(0).astype(int)
        pool["picked"] = pool.n_picked > 0
        out["pool"] = pool
    return out


# --------------------------------------------------------------------------- the game environment
def environment_frame(session: Session, cases: Sequence[ContestCase], actual_games: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    One row per team-game per slate: the environment file written at projection
    time (environment.<stamp>.csv beside the projections) with the actual
    points / plays / pass rate from nfl_schedule + nfl_stats. A slate's file is
    used once even when several contests share it.
    """
    actual = E.team_games(session) if actual_games is None else actual_games
    seen, rows = set(), []
    for case in cases:
        if case.projections is None or case.projections.path in seen:
            continue
        seen.add(case.projections.path)
        env = load_environment(case.projections.path)
        if env.empty:
            continue
        env = env.copy(); env["slate"] = case.slate
        if env.season.isna().all():
            when = bt.utc_to_local(case.lock) if case.lock else None
            sw = actuals.week_for_date(session, when) if when else None
            if sw:
                env["season"], env["week"] = sw
        rows.append(env)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    # a game is on every slate of its day (two classics, the showdowns): one row per team-game
    df["_day"] = df.slate.astype(str).str.slice(0, 10)
    df = df.drop_duplicates(["_day", "team"], keep="first").drop(columns=["_day"])
    a = actual[["season", "week", "team", "points", "plays", "pass_rate", "dropbacks", "rush_att"]].rename(
        columns={"points": "actual_points", "plays": "actual_plays", "pass_rate": "actual_pass_rate", "dropbacks": "actual_dropbacks", "rush_att": "actual_rushes"})
    df["season"] = pd.to_numeric(df.season, errors="coerce"); df["week"] = pd.to_numeric(df.week, errors="coerce")
    return df.merge(a, on=["season", "week", "team"], how="left")


def usage_frame(cases: Sequence[ContestCase]) -> pd.DataFrame:
    """One row per slate, team and stat from the usage file written at projection time (usage.<stamp>.csv beside the projections)."""
    seen, rows = set(), []
    for case in cases:
        if case.projections is None or case.projections.path in seen:
            continue
        seen.add(case.projections.path)
        u = load_usage(case.projections.path)
        if u.empty:
            continue
        u = u.copy(); u["slate"] = case.slate
        rows.append(u)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["_day"] = df.slate.astype(str).str.slice(0, 10)          # one row per team-game and stat, as for the environment
    return df.drop_duplicates(["_day", "team", "stat"], keep="first").drop(columns=["_day"])


# --------------------------------------------------------------------------- one row per player-game, and an honest baseline
def dedupe_player_games(players: pd.DataFrame) -> pd.DataFrame:
    """
    One row per player-game: the same game appears on every slate of the day
    (two classics, the showdowns) and as a captain row beside the flex row.
    The flex row is kept, so the 1.5x captain copy never double counts, and
    the first slate's row stands for the rest. Without this the salary
    baseline, fitted slate by slate, has already seen the scores it is judged
    on, and every error statistic counts the big games several times.
    """
    if players.empty or "slate" not in players.columns:
        return players
    df = players.copy()
    df["_day"] = df.slate.astype(str).str.slice(0, 10)
    df["_cpt"] = (pd.to_numeric(df.slot_code, errors="coerce") == bt.CPT_SLOT).astype(int) if "slot_code" in df.columns else 0
    df = df.sort_values(["_day", "name", "team", "_cpt", "slate"], kind="stable")
    df = df.drop_duplicates(["_day", "name", "team"], keep="first")
    return df.drop(columns=["_day", "_cpt"])


PRIOR_WEEKS_DAYS = 70          # the salary line is fitted on this many days of snapshots before the window
MIN_PRIOR_ROWS = 20            # per position


def prior_salary_baseline(session: Session, players: pd.DataFrame) -> Tuple[Optional[np.ndarray], str]:
    """
    A salary-only projection for every row, fitted per position on the
    projection snapshots of the weeks *before* the earliest slate in the
    frame (actuals from nfl_stats, see backtest_projections). Out of sample by
    construction. Returns (predictions aligned to players, note); predictions
    None when no earlier snapshots exist.
    """
    from ivan.sports.nfl import backtest_projections as BP
    if players.empty or "slate" not in players.columns:
        return None, ""
    first = pd.to_datetime(players.slate.astype(str).str.slice(0, 10), errors="coerce").min()
    if pd.isna(first):
        return None, ""
    until = (first - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    since = (first - pd.Timedelta(days=PRIOR_WEEKS_DAYS)).strftime("%Y-%m-%d")
    try:
        hist = BP.scored_projections(session, since, until)
    except ValueError:
        return None, ""
    hist = hist[pd.to_numeric(hist.actual, errors="coerce").notna()]
    pred = np.full(len(players), np.nan)
    pos = players.position.astype(str).str.upper().values
    sal = pd.to_numeric(players.salary, errors="coerce").values.astype(float)
    cpt = (pd.to_numeric(players.slot_code, errors="coerce") == bt.CPT_SLOT).values if "slot_code" in players.columns else np.zeros(len(players), bool)
    fitted = []
    for p_ in sorted(set(pos)):
        h = hist[hist.position.astype(str).str.upper() == p_]
        if len(h) < MIN_PRIOR_ROWS or h.salary.astype(float).std() == 0:
            continue
        b, a = np.polyfit(h.salary.astype(float).values, h.actual.astype(float).values, 1)
        m = pos == p_
        pred[m] = (a + b * sal[m]) * np.where(cpt[m], bt.CPT_MULT, 1.0)
        fitted.append(p_)
    if not fitted:
        return None, ""
    return pred, f"salary line per position fitted on {since}..{until} snapshots ({len(hist)} player-games)"


# --------------------------------------------------------------------------- snapshot projections (no artifacts yet)
def snapshot_players(session: Session, since: datetime, until: datetime, week_cache: Optional[Dict] = None) -> pd.DataFrame:
    """Projection-snapshot rows in [since, until] (local snapshot dates) with actual points from nfl_stats; source='snapshot'."""
    week_cache = {} if week_cache is None else week_cache
    weeks = week_cache.setdefault("_weeks", actuals.week_lookup(session))
    rows = session.query(ProjectionSnapshot).filter(ProjectionSnapshot.snapshot_date.between(since, until)).all()
    out = []
    for r in rows:
        sw = actuals.week_for_date(session, r.snapshot_date, weeks)
        if sw is None:
            continue
        if sw not in week_cache:
            week_cache[sw] = actuals.actual_points(session, *sw)
        wp = week_cache[sw]
        if not wp:
            continue
        a = actuals.lookup_points(wp, r.player_name, r.team, r.position)
        if a is None and actuals.team_played(wp, r.team):
            a = 0.0
        if a is None:
            continue
        code = int(r.roster_slot_id)
        mult = bt.CPT_MULT if code == bt.CPT_SLOT else 1.0
        out.append({"contest_id": None, "contest_name": "", "name": r.player_name, "slate": r.snapshot_date.strftime("%Y-%m-%d %H:%M"), "player_id": r.id,
                    "slot": SLOT_LABEL.get(code, str(code)), "slot_code": code, "team": r.team, "position": r.position, "salary": r.salary,
                    "proj_mean": r.fpts_mean, "proj_sd": r.fpts_std, "own_proj": r.ownership_proj, "actual_fpts": a * mult,
                    "actual_own": np.nan, "source": "snapshot"})
    df = pd.DataFrame(out)
    for c in V.Q_COLS:
        df[c] = np.nan
    return df


# --------------------------------------------------------------------------- the window
def replay_models(dirs: Sequence[Path]) -> List[dict]:
    """The model versions the replayed slates in `dirs` used (replay.json "models" beside or above each run directory), distinct."""
    seen, out = set(), []
    for d in dirs:
        for cand in (Path(d) / "replay.json", Path(d).parent / "replay.json"):
            if cand.exists():
                try:
                    m = json.loads(cand.read_text()).get("models") or {}
                except (OSError, json.JSONDecodeError):
                    m = {}
                key = (m.get("cutoff"), bool(m.get("fallback")))
                if m and key not in seen:
                    seen.add(key)
                    out.append({"cutoff": m.get("cutoff"), "fallback": bool(m.get("fallback")), "built": m.get("built") or []})
                break
    return out


def contest_luck(ours: Sequence[dict], rows: Sequence[dict]) -> dict:
    """
    Where the contest's realized P/L sits in the sim's own spread for the entries we played:
    the per-world P/L of the book (every entry's fee x its ROI in that world, summed) against
    the actual. sim_pl_pctile is the share of worlds at or below the actual (uniform over
    contests when the sim is calibrated), sim_p_loss the sim's P(the book loses) and
    sim_pl_p50_roi its median-world ROI.
    """
    traces, fees, actual = [], [], 0.0
    by_eid = {int(o["entry_id"]): o for o in ours}
    for r in rows:
        o = by_eid.get(int(r["entry_id"]))
        t = o.get("sim_roi") if o else None
        pay = r.get("actual_payout")
        if t is None or pay is None or not np.isfinite(pay):
            continue
        traces.append(np.asarray(t, dtype=np.float64)); fees.append(float(r["entry_fee"])); actual += float(pay) - float(r["entry_fee"])
    if not traces:
        return {"sim_pl_pctile": np.nan, "sim_p_loss": np.nan, "sim_pl_p50_roi": np.nan}
    S = min(t.size for t in traces)
    if S == 0:
        return {"sim_pl_pctile": np.nan, "sim_p_loss": np.nan, "sim_pl_p50_roi": np.nan}
    pl = np.vstack([f * t[:S] for f, t in zip(fees, traces)]).sum(axis=0)
    return {"sim_pl_pctile": float((pl <= actual).mean()), "sim_p_loss": float((pl < 0).mean()),
            "sim_pl_p50_roi": float(np.median(pl) / sum(fees))}


def honest_roi(run, contest_id: int) -> float:
    """The chooser's cross-fit (out-of-sample) mean ROI for a contest, from the run's choose json; nan when absent."""
    if run is None or run.lineups_csv is None:
        return np.nan
    base, stamp = run.lineups_csv.name.split(".")[0], run.stamp
    f = run.lineups_csv.with_name(f"{base}.choose.{stamp}.json")
    try:
        d = json.loads(f.read_text()).get(str(int(contest_id))) or {}
        return float(d["book"]["cross_fit"]["out_of_sample"]["mean"])
    except (OSError, ValueError, KeyError, TypeError):
        return np.nan


def validate_window(session: Session, dirs: Sequence[Path], out: Path, since: Optional[datetime] = None, until: Optional[datetime] = None,
                    in_field: str = "auto", include_snapshots: bool = True, title: str = "", echo=print) -> dict:
    """
    Run every check over the contests found in `dirs` (and, with
    include_snapshots, the projection snapshots of the same window) and
    write players/field/lineups/pool CSVs plus summary.json under `out`.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cases = discover(session, dirs, since, until)
    week_cache: Dict = {}
    frames = {k: [] for k in ("players", "field", "lineups", "pool")}
    coverage = []
    for case in cases:
        act = load_actuals(session, case, week_cache)
        fr = contest_frames(session, case, act, in_field=in_field)
        for k, df in fr.items():
            if not df.empty:
                frames[k].append(df)
        coverage.append({"contest_id": case.contest_id, "name": case.name, "slate": case.slate, "run": case.run.stamp if case.run else None,
                         "sim": str(case.sim.dir) if case.sim else None, "field_artifact": str(case.field_run.dir) if case.field_run else None,
                         "projections": str(case.projections.path) if case.projections else None,
                         "actual_points": act.has_points, "scraped": bool(act.scraped), "field": act.field is not None,
                         "field_coverage": (act.field.size / act.field_size) if (act.field is not None and act.field_size) else None,
                         "payouts": bool(act.tiers), "players": int(len(fr["players"])), "lineups": int(len(fr["lineups"])),
                         "pool": int(len(fr["pool"])), "notes": "; ".join(case.notes)})
        echo(f"{case.slate}  {case.contest_id}  {case.name[:50]:50s} players {len(fr['players']):3d}  lineups {len(fr['lineups']):3d}  "
             f"pool {len(fr['pool']):4d}  field {'y' if act.field is not None else '-'}  {'; '.join(case.notes)}")
    players = pd.concat(frames["players"], ignore_index=True) if frames["players"] else pd.DataFrame()
    # a contest's players appear once per contest and a game on every slate of the day; the projection check wants one row per player-game
    players_unique = dedupe_player_games(players) if not players.empty else players
    snap = pd.DataFrame()
    if include_snapshots and since is not None and until is not None:
        snap = snapshot_players(session, bt.utc_to_local(since), bt.utc_to_local(until), week_cache)
        if not snap.empty and not players_unique.empty:
            have = set(players_unique.slate.str.slice(0, 10))
            snap = snap[~snap.slate.str.slice(0, 10).isin(have)]        # artifacts win over snapshots for the same day
        echo(f"snapshot player-games with actuals (no artifacts): {len(snap)}")
    all_players = pd.concat([players_unique, snap], ignore_index=True) if not snap.empty else players_unique
    all_players = dedupe_player_games(all_players) if not all_players.empty else all_players
    # the salary baseline comes from the weeks before the window, never from the scores it is judged on
    baseline_note = ""
    for frame in (all_players, players_unique):
        if frame.empty:
            continue
        pred, note = prior_salary_baseline(session, frame)
        if pred is not None:
            frame["baseline"] = pred
            baseline_note = note
    if baseline_note:
        echo(f"salary baseline: {baseline_note}")
    else:
        echo("salary baseline: no earlier snapshots; leave-one-slate-out line (optimistic when slates share player-games)")

    env = environment_frame(session, cases)
    use = usage_frame(cases)
    reports = {"environment": V.environment_report(env) if not env.empty else {"n": 0},
               "usage": V.usage_report(use) if not use.empty else {"n": 0},
               "players": V.player_report(all_players, baseline_note=baseline_note or None) if not all_players.empty else {"n": 0},
               "players_artifacts": V.player_report(players_unique, baseline_note=baseline_note or None) if not players_unique.empty else {"n": 0},
               "field": V.field_report(pd.concat(frames["field"], ignore_index=True)) if frames["field"] else {"n": 0, "rows": []},
               "lineups": V.lineup_report(pd.concat(frames["lineups"], ignore_index=True)) if frames["lineups"] else {"n": 0},
               "pool": V.pool_report(pd.concat(frames["pool"], ignore_index=True)) if frames["pool"] else {"n": 0}}
    checks = V.build_checks(reports["players"], reports["field"], reports["lineups"], reports["pool"], environment=reports["environment"],
                            usage=reports["usage"])

    for k in ("environment", "usage", "players", "field", "lineups", "pool"):
        fr = reports[k].get("frame")
        if isinstance(fr, pd.DataFrame) and not fr.empty:
            fr.to_csv(out / f"{k}.csv", index=False)
    if reports["pool"].get("deciles_by_contest"):
        pd.DataFrame(reports["pool"]["deciles_by_contest"]).to_csv(out / "pool_deciles.csv", index=False)
    summary = {"title": title, "created_at": datetime.now().isoformat(timespec="seconds"), "dirs": [str(d) for d in dirs],
               "models": replay_models(dirs),
               "since": since.isoformat() if since else None, "until": until.isoformat() if until else None, "in_field": in_field,
               "contests": len(cases), "coverage": coverage, "checks": checks, "check_counts": V.summarise_checks(checks),
               "confidence": V.confidence_report(reports),
               "reports": {k: {kk: vv for kk, vv in v.items() if kk != "frame"} for k, v in reports.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    echo(f"\nchecks: {summary['check_counts']}")
    for c in checks:
        echo(f"  [{c['status']:4s}] {c['component']:12s} {c['name']:28s} {c['detail']}")
    echo(f"wrote {out}/")
    return summary


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (datetime, Path)):
        return str(o)
    return str(o)
