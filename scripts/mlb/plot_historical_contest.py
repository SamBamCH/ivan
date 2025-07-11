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

import numpy as np
import pandas as pd

from sqlalchemy import desc

from ivan.database.session import init_db, get_session
from ivan.core.utils import normalize_name
from ivan.sports.mlb.models import BatterProjection, PitcherProjection
from ivan.sports.mlb.diagnostic import plot_lineup_metrics

PITCHER_POS = {"SP", "RP", "P"}

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

    # 1) Read and parse CSV
    df = pd.read_csv(args.csv, engine="python")
    if "Lineup" not in df.columns:
        raise ValueError("CSV must have a 'Lineup' column")
    parsed = [parse_lineup(cell) for cell in df["Lineup"].astype(str)]
    print(f"Parsed {len(parsed)} lineups")

    # 2) DB init
    cfg     = json.load(open(args.config, encoding="utf-8"))
    engine  = init_db("mlb", cfg.get("database_url"))
    session = get_session(engine)

    # 3) Batch-fetch latest projections
    pitcher_nms = {normalize_name(nm) for lup in parsed for pos, nm in lup if pos=="P"}
    batter_nms  = {normalize_name(nm) for lup in parsed for pos, nm in lup if pos!="P"}

    rows = (
        session.query(PitcherProjection)
               .filter(PitcherProjection.player.in_(pitcher_nms),
                       PitcherProjection.created_at <= asof_dt)
               .order_by(PitcherProjection.player, desc(PitcherProjection.created_at))
               .all()
        +
        session.query(BatterProjection)
               .filter(BatterProjection.name.in_(batter_nms),
                       BatterProjection.created_at <= asof_dt)
               .order_by(BatterProjection.name, desc(BatterProjection.created_at))
               .all()
    )
    pitcher_latest = {r.player: r for r in rows if isinstance(r, PitcherProjection)}
    batter_latest  = {r.name: r for r in rows if isinstance(r, BatterProjection)}

    # 4) Build metric lists
    proj_sums      = []
    ownership_avgs = []
    stack_types    = []

    for idx, lineup in enumerate(parsed, start=1):
        if idx % 50 == 0 or idx == len(parsed):
            print(f"Processed {idx}/{len(parsed)} lineups")

        vals, owns, teams = [], [], []
        for pos, raw in lineup:
            nm = normalize_name(raw)
            rec = (pitcher_latest.get(nm) if pos=="P" else batter_latest.get(nm))
            if not rec:
                continue
            vals.append(rec.fpts)
            owns.append((rec.pown or 0.0)/100.0)
            if pos not in PITCHER_POS:
                teams.append(rec.team)

        if not vals:
            continue

        proj_sums.append(sum(vals))
        ownership_avgs.append(sum(owns)/len(owns))

        cnt = Counter(teams)
        top = [c for _, c in cnt.most_common(2)]
        top += [0]*(2-len(top))
        stack_types.append(f"{top[0]}-{top[1]}")

    # 5) Convert to arrays + indices
    own_arr   = np.array(ownership_avgs)
    proj_arr  = np.array(proj_sums)
    labels    = stack_types
    indices   = np.arange(len(own_arr))

    # 6) Plot sampled-only histograms
    plot_lineup_metrics(
        own_arr,
        proj_arr,
        labels,
        indices
    )


if __name__ == "__main__":
    main()
