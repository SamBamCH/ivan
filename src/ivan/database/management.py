#!/usr/bin/env python
from __future__ import annotations
import importlib
import os
import json
from sqlalchemy import text, exists
from sqlalchemy.orm import Session
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from ivan.database.session import init_db, get_session, Base
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.mlb.models import Lineup
from ivan.sports.mlb.sample_lineups import fetch_latest_map, PITCHER_POS
from ivan.sports.mlb.sample_lineups import DummyPlayer


def _resolve_models(sport: str) -> Dict[str, Any]:
    """
    Dynamically import ivan.sports.<sport>.models and pull the common classes we need.
    Expected (per sport):
      - HistoricalContest (required)
      - SlateKDEModel    (optional but recommended)
      - ContestIngestStatus (optional; if present, used for skip/attempt tracking)
      - ContestPayout, ActualContestEntry (optional; used to protect deletes)
    """
    mod = importlib.import_module(f"ivan.sports.{sport}.models")
    want = [
        "HistoricalContest",
        "SlateKDEModel",
        "ContestIngestStatus",
        "ContestPayout",
        "ActualContestEntry",
    ]
    out = {}
    for name in want:
        out[name] = getattr(mod, name, None)
    if out["HistoricalContest"] is None:
        raise RuntimeError(f"{sport}: models.HistoricalContest is required.")
    return out

def _to_aware_utc(dt: datetime) -> datetime:
    """Make naive datetimes UTC-aware; pass through aware ones."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # treat stored naive timestamps as UTC (change to local if that's your convention)
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def window_start(days: int = 7) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)

def within_window(dt: datetime, days: int = 7) -> bool:
    """True iff dt is within [now - days, now]."""
    if dt is None:
        return False
    dt_utc = _to_aware_utc(dt)
    now = datetime.now(timezone.utc)
    return (now - timedelta(days=days)) <= dt_utc <= now

# --- Selection helpers --------------------------------------------------------

def select_contests_to_process(db, sport: str, days: int = 7) -> list:
    m = _resolve_models(sport)
    HistoricalContest = m["HistoricalContest"]
    SlateKDEModel     = m["SlateKDEModel"]
    ContestIngestStatus = m["ContestIngestStatus"]

    now = datetime.now(timezone.utc)
    start = window_start(days)

    q = db.query(HistoricalContest)

    # DB-side coarse window (works for most backends)
    q = q.filter(
        HistoricalContest.date <= now,
        HistoricalContest.date >= start,
    )

    if SlateKDEModel is not None:
        modeled_subq = db.query(SlateKDEModel.contest_id)
        q = q.filter(~HistoricalContest.contest_id.in_(modeled_subq))

    if ContestIngestStatus is not None:
        failed_subq = db.query(ContestIngestStatus.contest_id).filter(
            ContestIngestStatus.permanently_failed.is_(True)
        )
        q = q.filter(~HistoricalContest.contest_id.in_(failed_subq))

    candidates = q.all()

    # Python-side strict guard (handles tz-naive rows, odd drivers, etc.)
    to_process = [hc for hc in candidates if within_window(hc.date, days)]
    return to_process

def counts_snapshot(db, sport: str, days: int = 7) -> dict:
    m = _resolve_models(sport)
    HistoricalContest = m["HistoricalContest"]
    SlateKDEModel     = m["SlateKDEModel"]
    ContestIngestStatus = m["ContestIngestStatus"]

    now = datetime.now(timezone.utc)
    start = window_start(days)

    # Coarse DB window
    window_q = db.query(HistoricalContest).filter(
        HistoricalContest.date >= start,
        HistoricalContest.date <= now,
    )

    already_modeled = 0
    if SlateKDEModel is not None:
        already_modeled = window_q.filter(
            HistoricalContest.contest_id.in_(db.query(SlateKDEModel.contest_id))
        ).count()

    permanently_skipped = 0
    if ContestIngestStatus is not None:
        permanently_skipped = window_q.filter(
            HistoricalContest.contest_id.in_(
                db.query(ContestIngestStatus.contest_id).filter(
                    ContestIngestStatus.permanently_failed.is_(True)
                )
            )
        ).count()

    return {"already_modeled": already_modeled, "permanently_skipped": permanently_skipped}

# --- Ingest status (retry/skip) ----------------------------------------------

def mark_success(db: Session, sport: str, contest_id: str) -> None:
    m = _resolve_models(sport)
    CIS = m["ContestIngestStatus"]
    if CIS is None:
        return  # no status table for this sport; silently ignore

    now = datetime.now(timezone.utc)
    st = db.query(CIS).get(contest_id)
    if st:
        st.attempts = 0
        st.permanently_failed = False
        st.last_error = None
        st.last_attempt_at = now
    else:
        db.add(CIS(
            contest_id=contest_id,
            attempts=0,
            permanently_failed=False,
            last_error=None,
            last_attempt_at=now,
        ))

def mark_failure(db: Session, sport: str, contest_id: str, err_msg: str, max_attempts: int = 3) -> None:
    m = _resolve_models(sport)
    CIS = m["ContestIngestStatus"]
    if CIS is None:
        return  # no status table; silently ignore

    now = datetime.now(timezone.utc)
    st = db.query(CIS).get(contest_id)
    if not st:
        db.add(CIS(
            contest_id=contest_id,
            attempts=1,
            permanently_failed=False,
            last_error=err_msg,
            first_failed_at=now,
            last_attempt_at=now,
        ))
    else:
        st.attempts = (st.attempts or 0) + 1
        st.last_error = err_msg
        st.last_attempt_at = now
        if st.attempts >= max_attempts:
            st.permanently_failed = True

# --- Purge tracking older than window ----------------------------------------

def purge_stale_contest_tracking(db, sport: str, days: int = 7) -> dict:
    m = _resolve_models(sport)
    HistoricalContest = m["HistoricalContest"]
    CIS = m["ContestIngestStatus"]

    cutoff = window_start(days)
    deleted_cis = 0

    if CIS is not None:
        stale_hc_ids = db.query(HistoricalContest.contest_id).filter(HistoricalContest.date < cutoff)
        deleted_cis = (
            db.query(CIS)
              .filter(
                  (CIS.contest_id.in_(stale_hc_ids)) |
                  ((CIS.last_attempt_at.isnot(None)) & (CIS.last_attempt_at < cutoff))
              )
              .delete(synchronize_session=False)
        )

    db.commit()
    return {"deleted_cis": deleted_cis}


CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'mlb.json')

def clear_sims():
    # assume this script lives in <project_root>/mlb/

    # load your DB connection info
    with open(CONFIG_PATH, encoding='utf-8') as f:
        cfg = json.load(f)

    # initialize engine + session factory
    cfg    = json.load(open(CONFIG_PATH, encoding='utf-8'))
    engine = init_db('mlb', cfg.get('database_url'))
    Base.metadata.create_all(engine)
    sess   = get_session(engine)

    try:
        print("Clearing lineup_simulation_results table...")
        sess.execute(text("DELETE FROM lineup_simulation_results"))
        sess.commit()
        print("✅ lineup_simulation_results has been emptied.")
    except Exception as e:
        sess.rollback()
        print(f"❌ Error clearing table: {e}")
    finally:
        sess.close()

def check_lineups():
    """
    Standalone script to plot diagnostics for *all* lineups in the database,
    using combined histograms and then reporting exposures vs. projected ownership.
    """
        # Load config & init DB/session
    cfg     = json.load(open(CONFIG_PATH, encoding="utf-8"))
    engine  = init_db("mlb", cfg.get("database_url"))
    session = get_session(engine)

    # Fetch *all* lineups
    all_lineups = session.query(Lineup).all()
    if not all_lineups:
        print("No lineups found in the database.")
        return

    # Fetch latest player rows for every player in those lineups
    all_pids   = {pid for lup in all_lineups for pid in lup.player_ids_list}
    latest_map = fetch_latest_map(session, all_pids)

    # for any pid not found, use a zero‐projection, zero‐ownership DummyPlayer
    for pid in all_pids:
        latest_map.setdefault(pid, DummyPlayer(pid))

    # Compute per‐lineup metrics
    ownership_avgs = []
    proj_sums      = []
    stack_types    = []

    for lup in all_lineups:
        pids = lup.player_ids_list

        # total projection
        proj_sums.append(
            sum(latest_map[pid].fpts_projection for pid in pids)
        )

        # average ownership (convert % → fraction)
        owns = [latest_map[pid].ownership / 100 for pid in pids]
        ownership_avgs.append(sum(owns) / len(owns))

        # stack‐type (top two non‐pitcher counts)
        cnt = Counter(
            latest_map[pid].team
            for pid in pids
            if not any(pos in PITCHER_POS for pos in latest_map[pid].position)
        )
        top_two = sorted(cnt.values(), reverse=True)[:2]
        c1, c2 = (top_two + [0, 0])[:2]
        stack_types.append(f"{c1}-{c2}")

    # Convert to numpy arrays
    own_arr  = np.array(ownership_avgs)
    proj_arr = np.array(proj_sums)
    labels   = stack_types

    # ––– Plot all metrics side by side –––
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Ownership histogram
    axes[0].hist(own_arr, bins=20, edgecolor="black")
    axes[0].set_title("Average Ownership per Lineup")
    axes[0].set_xlabel("Avg Ownership")
    axes[0].set_ylabel("Count of Lineups")

    # 2) Projection‐sum histogram
    axes[1].hist(proj_arr, bins=20, edgecolor="black")
    axes[1].set_title("Total Projection per Lineup")
    axes[1].set_xlabel("Sum of fpts_projection")
    axes[1].set_ylabel("Count of Lineups")

    # 3) Stack‐type bar chart
    stack_counts = Counter(labels)
    types, counts = zip(*sorted(stack_counts.items(), key=lambda x: x[1], reverse=True))
    axes[2].bar(types, counts)
    axes[2].set_title("Stack Types (Top-2 Non-Pitcher Counts)")
    axes[2].set_xlabel("Stack Type (c1-c2)")
    axes[2].set_ylabel("Count of Lineups")
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # ––– Compute player exposures –––
    exposures = Counter(pid for lup in all_lineups for pid in lup.player_ids_list)
    total_lineups = len(all_lineups)

    pitchers = {}
    batters  = {}

    for pid, count in exposures.items():
        exposure_frac = count / total_lineups
        projected_own = latest_map[pid].ownership / 100
        if any(pos in PITCHER_POS for pos in latest_map[pid].position):
            pitchers[pid] = (exposure_frac, projected_own)
        else:
            batters[pid]  = (exposure_frac, projected_own)

    def print_exposures(title, exp_dict):
        print(f"\n{title}")
        print(f"{'Name':<20}  {'Team':<5}  {'Exposure':>10}  {'ProjOwn':>10}")
        for pid, (expv, projv) in sorted(exp_dict.items(), key=lambda x: x[1][0], reverse=True):
            player = latest_map[pid]
            print(f"{player.name:<20}  {player.team:<5}  {expv:>9.2%}  {projv:>9.2%}")

    print_exposures("Pitcher Exposures vs. Projected Ownership", pitchers)
    print_exposures("Batter Exposures vs. Projected Ownership", batters)

   # ––– Top 5 absolute deviations from projected ownership –––
    def print_top_deviators(title, exp_dict, n=5):
        print(f"\n{title}")
        print(f"{'Name':<20}  {'Team':<5}  {'Exposure':>10}  {'ProjOwn':>10}  {'Dev':>8}")
        # build list of (pid, deviation, exposure, proj)
        devs = [
            (pid, abs(expv - projv), expv, projv)
            for pid, (expv, projv) in exp_dict.items()
        ]
        for pid, dev, expv, projv in sorted(devs, key=lambda x: x[1], reverse=True)[:n]:
            player = latest_map[pid]
            print(f"{player.name:<20}  {player.team:<5}  {expv:>9.2%}  {projv:>9.2%}  {dev:>7.2%}")

    print_top_deviators("Top 5 Deviations — Pitchers", pitchers)
    print_top_deviators("Top 5 Deviations — Batters", batters)



