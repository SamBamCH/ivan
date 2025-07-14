import re
from collections import Counter
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sqlalchemy import desc, func

from ivan.database.session  import init_db, get_session
from ivan.core.utils        import normalize_name
from ivan.sports.mlb.models import (
    BatterProjection, PitcherProjection,
    HistoricalContest, ContestPayout, SlateKDEModel,
)

# regex to extract (pos, name) pairs
_LINEUP_REGEX = re.compile(
    r"\b(1B|2B|3B|SS|C|OF|P)\s+(.+?)(?=(?:\s(?:1B|2B|3B|SS|C|OF|P)\s)|$)"
)
PITCHER_POS = {"SP", "RP", "P"}

def _parse_lineup(cell: str):
    return _LINEUP_REGEX.findall(cell)

def _batch_fetch_latest(session, Model, key_attr, names, cutoff_dt):
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

def fit_slate_kdes(contest_dir: str, db_key: str = "mlb"):
    """
    Loop through all contest-standings-<id>.csv in `contest_dir`,
    fit KDEs and upsert into the SlateKDEModel table.
    """
    engine  = init_db(db_key)
    session = get_session(engine)

    for csv_fp in Path(contest_dir).glob("contest-standings-*.csv"):
        m = re.search(r"contest-standings-(\d+)\.csv$", csv_fp.name)
        if not m:
            continue
        cid = int(m.group(1))

        df = pd.read_csv(csv_fp, engine="python")
        if "Lineup" not in df.columns:
            print(f"⚠️ skipping {csv_fp.name}: no 'Lineup'")
            continue

        # parse lineups & fetch contest metadata
        parsed       = [_parse_lineup(c) for c in df["Lineup"].astype(str)]
        total_entries = len(parsed)
        hc = session.query(HistoricalContest).get(cid)
        if not hc:
            print(f"⚠️ no HistoricalContest for id={cid}")
            continue
        contest_dt   = hc.date
        max_entries  = hc.multi_entry
        entry_fee    = hc.entry_fee
        total_pool   = entry_fee * hc.total_entries
        print(f"\nContest {cid}: {hc.name} @ {contest_dt}")

        # find most‐recent projection timestamps
        day_start = datetime(contest_dt.year, contest_dt.month, contest_dt.day)
        same_p   = session.query(func.max(PitcherProjection.created_at))\
                          .filter(PitcherProjection.created_at.between(day_start, contest_dt))\
                          .scalar()
        fallback = session.query(func.max(PitcherProjection.created_at))\
                          .filter(PitcherProjection.created_at <= contest_dt)\
                          .scalar()
        pitcher_cutoff = same_p or fallback

        same_b   = session.query(func.max(BatterProjection.created_at))\
                          .filter(BatterProjection.created_at.between(day_start, contest_dt))\
                          .scalar()
        fallback = session.query(func.max(BatterProjection.created_at))\
                          .filter(BatterProjection.created_at <= contest_dt)\
                          .scalar()
        batter_cutoff = same_b or fallback

        asof_ts = max(pitcher_cutoff, batter_cutoff)
        print(f"  using projections as of {asof_ts}")

        # collect all player names
        p_names, b_names = set(), set()
        for lu in parsed:
            for pos, nm in lu:
                nn = normalize_name(nm)
                (p_names if pos in PITCHER_POS else b_names).add(nn)

        pmap = _batch_fetch_latest(session, PitcherProjection, "player", p_names, pitcher_cutoff)
        bmap = _batch_fetch_latest(session, BatterProjection,  "name",   b_names,  batter_cutoff)

        # build per‐lineup arrays
        ownerships, projs, teams, owner_vals, stack_types = [], [], set(), [], []
        for rec in list(pmap.values()) + list(bmap.values()):
            pct = (getattr(rec, 'ownership', None) or getattr(rec, 'pown', 0.0)) / 100.0
            owner_vals.append(pct)

        for lu in parsed:
            counts, vals, owns = Counter(), [], []
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
            stack_types.append(f"{top_two[0] if top_two else 0}-{top_two[1] if len(top_two)>1 else 0}")
            projs.append(sum(vals))
            ownerships.append(np.mean(owns))

        if not projs:
            print(f" no valid lineups for {cid}")
            continue

        # slate‐level stats
        owner_vals   = sorted(owner_vals, reverse=True)
        top_owned    = owner_vals[0]
        top_5_owned  = np.mean(owner_vals[:5])   if len(owner_vals)>=5  else top_owned
        top_10_owned = np.mean(owner_vals[:10])  if len(owner_vals)>=10 else top_owned

        first_amt = session.query(func.sum(ContestPayout.payout_amount))\
                           .filter(ContestPayout.contest_id==cid,
                                   ContestPayout.min_position<=1,
                                   ContestPayout.max_position>=1).scalar() or 0.0
        top5_amt  = session.query(func.sum(ContestPayout.payout_amount))\
                           .filter(ContestPayout.contest_id==cid,
                                   ContestPayout.min_position.between(1,5)).scalar() or 0.0
        first_pct = first_amt / total_pool
        top5_pct  = top5_amt  / total_pool

        # normalize + fit KDEs
        max_proj      = float(max(projs))
        projs_norm    = [p/max_proj for p in projs]
        kde_own       = gaussian_kde(np.array(ownerships))
        kde_proj      = gaussian_kde(np.array(projs_norm))
        cnts_o,bins_o = np.histogram(ownerships, bins=20)
        cnts_p,bins_p = np.histogram(projs_norm, bins=20)

        num_games = max(len(teams)//2, 1)
        print(f" derived games={num_games}, entries={total_entries}, max_entries={max_entries}, entry_fee={entry_fee}")

        # upsert
        model = SlateKDEModel(
            contest_date     = contest_dt,
            num_games        = num_games,
            total_entries    = total_entries,
            max_entries      = max_entries,
            entry_fee        = entry_fee,
            top_owned        = top_owned,
            top_5_owned      = top_5_owned,
            top_10_owned     = top_10_owned,
            first_pct        = first_pct,
            top5_pct         = top5_pct,
            bw_own           = float(kde_own.factor),
            bw_proj          = float(kde_proj.factor),
            max_proj         = max_proj,
            hist_own_bins    = bins_o.tolist(),
            hist_own_counts  = cnts_o.tolist(),
            hist_proj_bins   = bins_p.tolist(),
            hist_proj_counts = cnts_p.tolist(),
            hist_stack_types = list(Counter(stack_types).keys()),
            hist_stack_probs = list(Counter(stack_types).values()),
        )
        session.merge(model)
        session.commit()
        print(f" ✔ upserted model for {num_games}_{total_entries}_{max_entries}")
