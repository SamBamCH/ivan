# src/ivan/sports/mma/scoring_fit.py
"""
What a significant strike and a takedown are worth in DraftKings points, measured on real fights.

The sim knows a fighter by two lines, significant strikes and takedowns. DraftKings also pays every
strike (0.2, and 0.2 more when significant), control time (0.03 a second), knockdowns (10) and
reversals (5), none of which has a line. `scoring.sig_strike` and `scoring.takedown` are therefore
proxies: the per-strike weight carries the non-significant strikes and the knockdowns, the takedown
weight carries the control time that comes with takedowns (the two are correlated 0.7).

The fit is on fighters' averages over their fights, the form a line takes (an expected count, not
one night's): production = a x strikes + b x takedowns, no intercept, least squares.

    production(df)      DraftKings points without the win bonus, per fighter per fight
    fit(df)             {"sig_strike", "takedown", fighters, fights, the miss by takedown rate for these and other weights}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ivan.sports.mma.paths import MMA_DATA_DIR

FIGHTS = MMA_DATA_DIR / "derived" / "ufcstats" / "fights.parquet"      # one row per fighter per fight (the ufcstats scrape)
MIN_FIGHTS = 4
BUCKETS = [(-1, 0.5, "under 0.5 takedowns a fight"), (0.5, 1.5, "0.5 to 1.5"), (1.5, 1e9, "1.5 or more")]


def production(df: pd.DataFrame) -> pd.Series:
    return (0.2 * df["total_landed"] + 0.2 * df["sig_landed"] + 5.0 * df["td_landed"] + 5.0 * df["rev"]
            + 10.0 * df["kd"] + 0.03 * df["ctrl_s"])


def fit(df: pd.DataFrame, compare: tuple = (), min_fights: int = MIN_FIGHTS) -> dict:
    df = df[df["result"].isin(["W", "L"])].copy()
    df["prod"] = production(df)
    g = df.groupby("fighter_id").agg(n=("prod", "size"), prod=("prod", "mean"), sig=("sig_landed", "mean"), td=("td_landed", "mean"))
    g = g[g["n"] >= min_fights]
    a, b = np.linalg.lstsq(g[["sig", "td"]].to_numpy(float), g["prod"].to_numpy(float), rcond=None)[0]
    weights = [("fitted", float(a), float(b))] + [(f"{x:g} / {y:g}", float(x), float(y)) for x, y in compare]
    rows = []
    for lo, hi, label in BUCKETS:
        d = g[(g["td"] > lo) & (g["td"] <= hi)]
        rows.append({"bucket": label, "fighters": int(len(d)), "real": float(d["prod"].mean()),
                     "miss": {name: float((x * d["sig"] + y * d["td"] - d["prod"]).mean()) for name, x, y in weights}})
    parts = {"strikes": float((0.2 * df["total_landed"] + 0.2 * df["sig_landed"]).mean()), "takedowns": float(5 * df["td_landed"].mean()),
             "control": float(0.03 * df["ctrl_s"].mean()), "knockdowns": float(10 * df["kd"].mean()), "reversals": float(5 * df["rev"].mean())}
    return {"sig_strike": float(a), "takedown": float(b), "fighters": int(len(g)), "fights": int(len(df) // 2),
            "from": str(df["event_date"].min()), "to": str(df["event_date"].max()), "production": float(df["prod"].mean()), "parts": parts,
            "loser_share": float(df.loc[~df["won"].astype(bool), "prod"].mean() / df.loc[df["won"].astype(bool), "prod"].mean()),
            "buckets": rows, "weights": [w[0] for w in weights]}


def load(path: Path = FIGHTS) -> pd.DataFrame:
    return pd.read_parquet(path)


def format_fit(r: dict) -> str:
    lines = [f"{r['fights']:,} fights {r['from']} to {r['to']}; {r['fighters']} fighters with {MIN_FIGHTS}+ fights",
             f"production (points without the win bonus) averages {r['production']:.1f}: " + ", ".join(f"{k} {v:.1f}" for k, v in r["parts"].items()),
             f"a loser produces {r['loser_share']:.2f} of a winner (scoring.loser_production_share)",
             f"fitted: scoring.sig_strike {r['sig_strike']:.2f}, scoring.takedown {r['takedown']:.1f}",
             "the miss (weights x career averages, minus real production) by how often the fighter takes people down:",
             f"  {'':<30} {'fighters':>8} {'real':>6}" + "".join(f" {w:>10}" for w in r["weights"])]
    for b in r["buckets"]:
        lines.append(f"  {b['bucket']:<30} {b['fighters']:>8} {b['real']:>6.1f}" + "".join(f" {b['miss'][w]:>+10.1f}" for w in r["weights"]))
    return "\n".join(lines)
