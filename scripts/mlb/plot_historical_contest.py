#!/usr/bin/env python3
"""
plot_lineup_csv.py

Read a lineup CSV, normalize each name, batch-fetch each player’s latest
projection/ownership/team as of a given timestamp, then plot diagnostics.
"""

import re
import json
import argparse
from collections import Counter
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import desc

from ivan.database.session import init_db, get_session
from ivan.core.utils import normalize_name
from ivan.sports.mlb.models import BatterProjection, PitcherProjection

PITCHER_POS = {"SP", "RP", "P"}

# regex to extract e.g. ("1B","Vladimir Guerrero Jr.") pairs
LINEUP_REGEX = re.compile(
    r"\b(1B|2B|3B|SS|C|OF|P)\s+(.+?)(?=(?:\s(?:1B|2B|3B|SS|C|OF|P)\s)|$)"
)

def parse_lineup(cell: str):
    return LINEUP_REGEX.findall(cell)

def main():
    p = argparse.ArgumentParser(
        description="Plot diagnostics for lineups CSV using per-player lookups"
    )
    p.add_argument("--csv", "-i",  required=True, help="Input lineup CSV")
    p.add_argument("--config", "-c", default="config/mlb.json",
                   help="MLB JSON config")
    p.add_argument("--asof", "-a", required=True,
                   help="As-of timestamp e.g. 2025-05-26T16:19:03")
    args = p.parse_args()

    # parse as-of datetime
    asof_dt = datetime.fromisoformat(args.asof)

    # 1) Read CSV
    df = pd.read_csv(args.csv, engine="python")
    if "Lineup" not in df.columns:
        raise ValueError("CSV must have a 'Lineup' column")
    raw_lineups = df["Lineup"].astype(str).tolist()

    # 2) Parse each lineup cell
    parsed = [parse_lineup(cell) for cell in raw_lineups]

    print(f'parsed {len(parsed)}')

    # 3) DB init
    json.load(open(args.config))  # (keep for any config usage)
    engine  = init_db("mlb")
    session = get_session(engine)

    # 4) Collect every unique normalized name per model
    pitcher_nms = set()
    batter_nms  = set()
    for lineup in parsed:
        for pos, raw_name in lineup:
            nm = normalize_name(raw_name)
            if pos == "P":
                pitcher_nms.add(nm)
            else:
                batter_nms.add(nm)

    # 5) Batch-fetch all pitcher projections ≤ asof_dt
    pitcher_rows = (
        session.query(PitcherProjection)
               .filter(
                   PitcherProjection.player.in_(pitcher_nms),
                   PitcherProjection.created_at <= asof_dt
               )
               .order_by(PitcherProjection.player,
                         desc(PitcherProjection.created_at))
               .all()
    )
    pitcher_latest = {}
    for r in pitcher_rows:
        if r.player not in pitcher_latest:
            pitcher_latest[r.player] = r

    # 6) Batch-fetch all batter projections ≤ asof_dt
    batter_rows = (
        session.query(BatterProjection)
               .filter(
                   BatterProjection.name.in_(batter_nms),
                   BatterProjection.created_at <= asof_dt
               )
               .order_by(BatterProjection.name,
                         desc(BatterProjection.created_at))
               .all()
    )
    batter_latest = {}
    for r in batter_rows:
        if r.name not in batter_latest:
            batter_latest[r.name] = r

    # 7) Loop lineups with progress prints
    proj_sums      = []
    ownership_avgs = []
    stack_types    = []
    total = len(parsed)

    for idx, lineup in enumerate(parsed, start=1):
        if idx % 50 == 0 or idx == total:
            print(f"Processed {idx}/{total} lineups")

        vals, owns, teams = [], [], []
        for pos, raw_name in lineup:
            nm = normalize_name(raw_name)
            rec = (pitcher_latest.get(nm) if pos == "P"
                   else batter_latest.get(nm))
            if not rec:
                continue
            vals.append(rec.fpts)
            owns.append((rec.pown or 0.0) / 100.0)
            if pos not in PITCHER_POS:
                teams.append(rec.team)

        if not vals:
            continue

        proj_sums.append(sum(vals))
        ownership_avgs.append(sum(owns) / len(owns))

        cnt = Counter(teams)
        top = [c for _, c in cnt.most_common(2)]
        if len(top) < 2:
            top += [0]
        stack_types.append(f"{top[0]}-{top[1]}")

    # 8) Plot histograms
    plt.figure()
    plt.hist(ownership_avgs, bins=20)
    plt.title(f"CSV Lineups Avg Ownership (as of {asof_dt})")
    plt.xlabel("Avg Ownership")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(proj_sums, bins=20)
    plt.title(f"CSV Lineups Projection Sums (as of {asof_dt})")
    plt.xlabel("Projection Sum")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 9) Print stack-type probabilities
    total_stacks = len(stack_types)
    cnt = Counter(stack_types)
    print("\nStack-Type Probabilities:")
    for stype, count in cnt.most_common():
        print(f"  {stype}: {count/total_stacks:.3f}")

if __name__ == "__main__":
    main()
