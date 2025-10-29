import argparse
import os
import json
import pandas as pd
from collections import Counter


from ivan.database.session import init_db, get_session

BASE_COLS = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
POSITION_ORDER = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

def prompt_float(prompt_text: str, default: float) -> float:
    raw = input(f"{prompt_text} [{default}]: ").strip()
    return float(raw) if raw else default

def prompt_weights_for_contest(
    contest_id: int,
    contest_name: str,
    last_weights: dict[str, float]
) -> dict[str, float]:
    print(f"\n▶ Contest {contest_id}: \"{contest_name}\"")
    w_cash = prompt_float(f"  Weight on cash_pct? ", last_weights["cash_pct"])
    w_roi  = prompt_float(f"  Weight on avg_roi_win? ", last_weights["avg_roi_win"])
    w_div  = prompt_float(f"  Diversity λ? ", last_weights["diversity_lambda"])
    return {"cash_pct": w_cash, "avg_roi_win": w_roi, "diversity_lambda": w_div}

def main():
    parser = argparse.ArgumentParser(description='choose lineups and write draftkings formatted csv')
    parser.add_argument("-o", "--output_csv",
                        type=str,
                        default="data/nba/output/chosen_entries.csv",
                        help="Base filename for output CSV(s). We'll append .classic/.showdown and a timestamp.")
    args = parser.parse_args()

    here    = os.path.dirname(__file__)
    cfg_fp  = os.path.abspath(os.path.join(here, "../../config/nba.json"))
    cfg     = json.load(open(cfg_fp, encoding="utf-8"))

    entries_fp = cfg["paths"]["entries_csv"]
    df_raw = pd.read_csv(entries_fp, engine="python", header=0, usecols=range(10), dtype=str, keep_default_na=False)
    for col in BASE_COLS:
        if col not in df_raw.columns:
            raise RuntimeError(f"Expected column '{col}' in entries CSV but not found.")

    # Cut off at first blank Entry ID
    blank_mask = df_raw["Entry ID"].str.strip() == ""
    if blank_mask.any():
        first_blank_idx = blank_mask.idxmax()
        df_raw = df_raw.loc[: first_blank_idx - 1].copy()

    # Normalize types
    df_raw["Contest ID"] = df_raw["Contest ID"].astype(int)
    contests_grouped = df_raw.groupby("Contest ID", sort=False)

    # Default weights (prompted once per contest)
    last_weights = {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1}

    engine = init_db('nba')
    session = get_session(engine)

    df_out = pd.DataFrame(columns=BASE_COLS + POSITION_ORDER)

    prompted_contests: set[int] = set()

    total_entries = 0

    counts = Counter()
    info = {}

    for cid, group in contests_grouped:
        num_entries = len(group)
        cname = group["Contest Name"].iloc[0]
        print(f"\n=== Contest {cid} (\"{cname}\") → {num_entries} entries ===")

        if cid not in prompted_contests:
            w = prompt_weights_for_contest(cid, cname, last_weights)

