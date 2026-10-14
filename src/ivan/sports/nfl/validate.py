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
    run: bt.ChooseRun
    sim: Optional[artifacts.SimRun] = None
    projections: Optional[artifacts.ProjectionFile] = None
    notes: List[str] = field(default_factory=list)

    @property
    def slate(self) -> str:
        return bt.utc_to_local(self.lock).strftime("%Y-%m-%d %H:%M") if self.lock else "unknown"


def discover(session: Session, dirs: Sequence[Path], since: Optional[datetime] = None, until: Optional[datetime] = None) -> List[ContestCase]:
    """One case per contest with a pre-lock choose run in `dirs`, paired with its sim run and projections file."""
    dirs = [Path(d) for d in dirs]
    runs = bt.find_choose_runs(dirs)
    sims = artifacts.find_sim_runs(dirs)
    projs = artifacts.find_projection_files(dirs + [d.parent for d in dirs])
    info = bt.contest_info(session, {c for r in runs for c in bt._contests_in_run(r)})
    by_contest = bt.select_prelock_runs(runs, info)
    cases = []
    for cid, run in by_contest.items():
        hc = info.get(cid)
        lock = hc.date if hc is not None else None
        if (since and lock and lock < since) or (until and lock and lock > until):
            continue
        case = ContestCase(cid, hc.name if hc is not None else "", lock, run)
        with_cid = [s for s in sims if cid in s.contests()]
        same_tree = [s for s in with_cid if s.dir.parent == run.lineups_csv.parent]       # a replay repeat: sim/ beside the choose files
        case.sim = artifacts.latest_before(same_tree or with_cid, run.when_local)
        anchor = case.sim.when if case.sim else run.when_local
        same_tree = [p for p in projs if p.path.parent == run.lineups_csv.parent or p.path.parent == run.lineups_csv.parent.parent]
        case.projections = artifacts.latest_before(same_tree or projs, anchor)
        if case.sim is None:
            case.notes.append("no sim artifact (run predates artifacts)")
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
    if case.sim is not None:
        _, meta = load_sim_artifact(*artifacts.contest_artifact_paths(case.sim.dir, case.contest_id))
        rows = [{"player_id": p["id"], "slot": p.get("slot"), "slot_code": slot_code(p.get("slot") or "FLEX"), "name": p.get("name"), "team": p.get("team"),
                 "position": p.get("pos"), "salary": p.get("salary"), "proj_mean": p.get("fpts_mean"), "proj_sd": np.nan, "own_proj": np.nan}
                for p in meta.get("players", [])]
        df = pd.DataFrame(rows)
        for c in V.Q_COLS:
            df[c] = np.nan
        return df
    return pd.DataFrame()


def contest_frames(session: Session, case: ContestCase, act: Actuals, in_field: str = "auto") -> Dict[str, pd.DataFrame]:
    """players / field / lineups / pool frames for one contest (empty frames where the inputs are missing)."""
    out = {"players": pd.DataFrame(), "field": pd.DataFrame(), "lineups": pd.DataFrame(), "pool": pd.DataFrame()}
    players = _player_table(case)
    arrays, meta = (load_sim_artifact(*artifacts.contest_artifact_paths(case.sim.dir, case.contest_id)) if case.sim is not None else ({}, {}))
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
            smean = dict(zip(arrays["player_id"].tolist(), arrays["player_mean"].tolist()))
            p["own_field"] = p.player_id.map(expo)
            p["sim_mean"] = p.player_id.map(smean)
        p["source"] = "artifact"
        for k, v in tag.items():
            p[k] = v
        p["actual_own"] = pd.to_numeric(p.actual_own, errors="coerce")
        out["players"] = p[p.actual_fpts.notna() | p.actual_own.notna()] if act.has_points or act.scraped_own else p.iloc[0:0]

    # ---- field: the sim's field against the real one
    if arrays and act.field is not None and act.tiers:
        q = list(arrays["qgrid"].tolist())
        fld = np.asarray(act.field, dtype=float)
        paid = min(places_paid(act.tiers), fld.size)
        sp = np.percentile(arrays["score_pct"], [10, 50, 90], axis=0)      # (3, len(q))
        qi = {v: i for i, v in enumerate(q)}
        dupes = arrays["dupes"]
        real_d = np.asarray(list(act.real_dupes.values()), dtype=float) if act.real_dupes else np.zeros(0)
        row = {**tag, "sim_size": int(meta.get("simulated_size", 0)), "actual_size": int(fld.size), "contest_size": act.field_size,
               "paid_places": int(paid), "n_sims": int(meta.get("n_sims", 0)),
               "sim_cash_p10": float(np.percentile(arrays["cash_line"], 10)), "sim_cash_p50": float(np.percentile(arrays["cash_line"], 50)),
               "sim_cash_p90": float(np.percentile(arrays["cash_line"], 90)), "actual_cash_line": float(fld[paid - 1]) if paid else np.nan,
               "sim_win_p10": float(np.percentile(arrays["winning_score"], 10)), "sim_win_p50": float(np.percentile(arrays["winning_score"], 50)),
               "sim_win_p90": float(np.percentile(arrays["winning_score"], 90)), "actual_win": float(fld[0]),
               "sim_score_p50": float(sp[1, qi[50]]), "sim_score_p90": float(sp[1, qi[90]]), "sim_score_p99": float(sp[1, qi[99]]),
               "sim_score_p10": float(sp[1, qi[10]]),
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
    ours = bt.our_entries(case.run, case.contest_id)
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
        out["lineups"] = pd.DataFrame(rows)

    if arrays and act.has_points and act.field is not None and pay is not None and not players.empty:
        pm = {int(r.player_id): (str(r.slot), int(r.player_id), str(r.name), str(r.team), str(r.position), int(r.salary)) for r in players.itertuples()}
        picked_counts: Dict[int, int] = {}
        for o in ours:
            picked_counts[int(o["lineup_id"])] = picked_counts.get(int(o["lineup_id"]), 0) + 1
        scores, keys_ = [], []
        for row in arrays["lineup_players"]:
            pl = [pm[int(pid)] for pid in row if int(pid) >= 0 and int(pid) in pm]
            if len(pl) != int((row >= 0).sum()):
                scores.append(np.nan); keys_.append(None); continue
            s, missing, _ = bt.score_lineup(pl, act.week_points, act.scraped)
            scores.append(np.nan if s is None else s)
            keys_.append(frozenset((slot_code(sl), t, ps.upper(), sal) for sl, _, _, t, ps, sal in pl))
        scores = np.asarray(scores, dtype=float)
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
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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
        coverage.append({"contest_id": case.contest_id, "name": case.name, "slate": case.slate, "run": case.run.stamp,
                         "sim": str(case.sim.dir) if case.sim else None, "projections": str(case.projections.path) if case.projections else None,
                         "actual_points": act.has_points, "scraped": bool(act.scraped), "field": act.field is not None,
                         "field_coverage": (act.field.size / act.field_size) if (act.field is not None and act.field_size) else None,
                         "payouts": bool(act.tiers), "players": int(len(fr["players"])), "lineups": int(len(fr["lineups"])),
                         "pool": int(len(fr["pool"])), "notes": "; ".join(case.notes)})
        echo(f"{case.slate}  {case.contest_id}  {case.name[:50]:50s} players {len(fr['players']):3d}  lineups {len(fr['lineups']):3d}  "
             f"pool {len(fr['pool']):4d}  field {'y' if act.field is not None else '-'}  {'; '.join(case.notes)}")
    players = pd.concat(frames["players"], ignore_index=True) if frames["players"] else pd.DataFrame()
    # a contest's players appear once per contest; the projection check wants one row per player per slate
    if not players.empty:
        players_unique = players.drop_duplicates(subset=["slate", "player_id"]).copy()
    else:
        players_unique = players
    snap = pd.DataFrame()
    if include_snapshots and since is not None and until is not None:
        snap = snapshot_players(session, bt.utc_to_local(since), bt.utc_to_local(until), week_cache)
        if not snap.empty and not players_unique.empty:
            have = set(players_unique.slate.str.slice(0, 10))
            snap = snap[~snap.slate.str.slice(0, 10).isin(have)]        # artifacts win over snapshots for the same day
        echo(f"snapshot player-games with actuals (no artifacts): {len(snap)}")
    all_players = pd.concat([players_unique, snap], ignore_index=True) if not snap.empty else players_unique

    env = environment_frame(session, cases)
    use = usage_frame(cases)
    reports = {"environment": V.environment_report(env) if not env.empty else {"n": 0},
               "usage": V.usage_report(use) if not use.empty else {"n": 0},
               "players": V.player_report(all_players) if not all_players.empty else {"n": 0},
               "players_artifacts": V.player_report(players_unique) if not players_unique.empty else {"n": 0},
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
