#!/usr/bin/env python3
"""
fit_slate_kde_batch_db.py

Loop through every historical contest CSV in a directory, derive slate
features (num_games, total_entries, max_entries, entry_fee, top ownership stats,
payout percentages), fit 1D Gaussian KDEs on (avg ownership)
and (normalized projection sum), and upsert each model into the database.
"""
import re
import argparse
from collections import Counter
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sqlalchemy import desc, func

from ivan.database.session      import init_db, get_session
from ivan.core.utils            import normalize_name
from ivan.sports.mlb.models     import (
    BatterProjection,
    PitcherProjection,
    HistoricalContest,
    ContestPayout,
    SlateKDEModel,
)

# regex to extract (pos, name) pairs
LINEUP_REGEX = re.compile(
    r"\b(1B|2B|3B|SS|C|OF|P)\s+(.+?)(?=(?:\s(?:1B|2B|3B|SS|C|OF|P)\s)|$)"
)
PITCHER_POS = {"SP", "RP", "P"}


def parse_lineup(cell: str):
    return LINEUP_REGEX.findall(cell)


def batch_fetch_latest(session, Model, key_attr, names, cutoff_dt):
    rows = (
        session.query(Model)
               .filter(getattr(Model, key_attr).in_(names),
                       Model.created_at <= cutoff_dt)
               .order_by(getattr(Model, key_attr), desc(Model.created_at))
               .all()
    )
    latest = {}
    for r in rows:
        k = getattr(r, key_attr)
        if k not in latest:
            latest[k] = r
    return latest


def main():
    p = argparse.ArgumentParser(
        description="Batch-fit slate models and store into database"
    )
    p.add_argument(
        "--dir", "-d",
        default=r"I:\data\mlb\historical_contests",
        help="Directory with contest-standings-<id>.csv files"
    )
    args = p.parse_args()

    engine  = init_db("mlb")
    session = get_session(engine)

    for csv_fp in Path(args.dir).glob("contest-standings-*.csv"):
        m = re.search(r"contest-standings-(\d+)\.csv$", csv_fp.name)
        if not m:
            continue
        cid = int(m.group(1))

        # read & parse CSV
        df = pd.read_csv(csv_fp, engine="python")
        if "Lineup" not in df.columns:
            print(f"⚠️ skipping {csv_fp.name}: no 'Lineup'")
            continue
        raw = df["Lineup"].astype(str).tolist()
        parsed = [parse_lineup(c) for c in raw]
        total_entries = len(parsed)

        # fetch contest metadata
        hc = session.query(HistoricalContest).get(cid)
        if not hc:
            print(f"⚠️ no HistoricalContest for id={cid}")
            continue
        contest_dt   = hc.date
        max_entries  = hc.multi_entry
        entry_fee    = hc.entry_fee
        total_pool   = entry_fee * hc.total_entries
        print(f"\nContest {cid}: {hc.name} @ {contest_dt}")

        # determine most recent projections on that date (else fallback)
        day_start = datetime(contest_dt.year, contest_dt.month, contest_dt.day)
        same_p = session.query(func.max(PitcherProjection.created_at)) \
                       .filter(PitcherProjection.created_at >= day_start,
                               PitcherProjection.created_at <= contest_dt).scalar()
        fallback_p = session.query(func.max(PitcherProjection.created_at)) \
                             .filter(PitcherProjection.created_at <= contest_dt).scalar()
        pitcher_cutoff = same_p or fallback_p

        same_b = session.query(func.max(BatterProjection.created_at)) \
                       .filter(BatterProjection.created_at >= day_start,
                               BatterProjection.created_at <= contest_dt).scalar()
        fallback_b = session.query(func.max(BatterProjection.created_at)) \
                             .filter(BatterProjection.created_at <= contest_dt).scalar()
        batter_cutoff = same_b or fallback_b

        asof_ts = max(pitcher_cutoff, batter_cutoff)
        print(f"  using projections as of {asof_ts}")

        # collect unique normalized names
        p_names, b_names = set(), set()
        for lu in parsed:
            for pos, nm in lu:
                nn = normalize_name(nm)
                (p_names if pos in PITCHER_POS else b_names).add(nn)

        # batch-fetch latest projections
        pmap = batch_fetch_latest(session, PitcherProjection, "player", p_names, pitcher_cutoff)
        bmap = batch_fetch_latest(session, BatterProjection,  "name",   b_names,  batter_cutoff)

        # build arrays & collect teams + owner list
        ownerships, projs, teams, owner_vals = [], [], set(), []
        for rec in list(pmap.values()) + list(bmap.values()):
            pct = (getattr(rec, 'ownership', None) or getattr(rec, 'pown', 0.0)) / 100.0
            owner_vals.append(pct)

        stack_types = []

        for lu in parsed:
            counts = Counter()
            vals, owns = [], []
            for pos, nm in lu:
                nn  = normalize_name(nm)
                rec = pmap.get(nn) if pos in PITCHER_POS else bmap.get(nn)
                if rec and pos not in PITCHER_POS:
                    counts[rec.team] += 1
                if not rec:
                    continue
                vals.append(rec.fpts)
                pct = (getattr(rec, 'ownership', None) or getattr(rec, 'pown', 0.0)) / 100.0
                owns.append(pct)
                teams.add(rec.team)
            if not vals:
                continue
            top_two = sorted(counts.values(), reverse=True)[:2]
            c1, c2 = top_two[0] if len(top_two)>0 else 0, top_two[1] if len(top_two)>1 else 0
            stack_types.append(f"{c1}-{c2}")
            projs.append(sum(vals))
            ownerships.append(np.mean(owns))

        total = len(stack_types)
        stack_counts = Counter(stack_types)
        stack_probs  = {stype: cnt/total for stype, cnt in stack_counts.items()}

        if not projs:
            print(f" no valid lineups for {cid}")
            continue

        # derive slate-level ownership metrics
        owner_vals = sorted(owner_vals, reverse=True)
        top_owned   = owner_vals[0]
        top_5_owned = np.mean(owner_vals[:5]) if len(owner_vals)>=5 else np.mean(owner_vals)
        top_10_owned= np.mean(owner_vals[:10]) if len(owner_vals)>=10 else np.mean(owner_vals)

        # payout percentages
        first_amt = (
            session.query(func.sum(ContestPayout.payout_amount))
                .filter(
                    ContestPayout.contest_id == cid,
                    ContestPayout.min_position <= 1,
                    ContestPayout.max_position >= 1
                ).scalar()
            or 0.0
        )
        top5_amt  = session.query(func.sum(ContestPayout.payout_amount)) \
                          .filter(ContestPayout.contest_id==cid,
                                  ContestPayout.min_position.between(1,5)).scalar() or 0.0
        first_pct = first_amt / total_pool
        top5_pct  = top5_amt  / total_pool

        # normalize & fit KDEs + histograms
        max_proj    = float(max(projs))
        projs_norm  = [p/max_proj for p in projs]
        kde_own     = gaussian_kde(np.array(ownerships))
        kde_proj    = gaussian_kde(np.array(projs_norm))
        cnts_o,bins_o = np.histogram(ownerships,   bins=20)
        cnts_p, bins_p = np.histogram(projs_norm,    bins=20)

        num_games = max(len(teams)//2, 1)
        print(f" derived games={num_games}, entries={total_entries}, "
              f"max_entries={max_entries}, entry_fee={entry_fee}")

        # upsert model record
        model = SlateKDEModel(
            contest_date    = contest_dt,
            num_games       = num_games,
            total_entries   = total_entries,
            max_entries     = max_entries,
            entry_fee       = entry_fee,
            top_owned       = top_owned,
            top_5_owned     = top_5_owned,
            top_10_owned    = top_10_owned,
            first_pct       = first_pct,
            top5_pct        = top5_pct,
            bw_own          = float(kde_own.factor),
            bw_proj         = float(kde_proj.factor),
            max_proj        = max_proj,
            hist_own_bins    = bins_o.tolist(),
            hist_own_counts  = cnts_o.tolist(),
            hist_proj_bins   = bins_p.tolist(),
            hist_proj_counts = cnts_p.tolist(),
            hist_stack_types = list(stack_counts.keys()),
            hist_stack_probs = list(stack_probs.values()),
        )
        session.merge(model)
        session.commit()
        print(f" ✔ upserted model for {num_games}_{total_entries}_{max_entries}")

if __name__ == "__main__":
    main()
