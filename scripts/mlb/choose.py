#!/usr/bin/env python
# scripts/mlb/choose.py

import os
import json
from datetime import date

from ivan.database.session import init_db, get_session
from ivan.sports.mlb.selection import choose_lineup_ids
from ivan.sports.mlb.export    import export_selected_lineups

SALARY_CAP = 50000

def prompt_int(prompt_text: str, default: int) -> int:
    raw = input(f"{prompt_text} [{default}]: ").strip()
    return int(raw) if raw else default

def prompt_float(prompt_text: str, default: float) -> float:
    raw = input(f"{prompt_text} [{default}]: ").strip()
    return float(raw) if raw else default

def prompt_date(prompt_text: str, default: date) -> date:
    raw = input(f"{prompt_text} (YYYY-MM-DD) [{default.isoformat()}]: ").strip()
    return date.fromisoformat(raw) if raw else default

def prompt_str(prompt_text: str, default: str) -> str:
    raw = input(f"{prompt_text} [{default}]: ").strip()
    return raw if raw else default

def main():
    print("🏏 DFS Lineup Selector — Interactive Mode\n")

    # 1) load config
    here   = os.path.dirname(__file__)
    cfg_fp = os.path.abspath(os.path.join(here, "../../config/mlb.json"))
    cfg    = json.load(open(cfg_fp, encoding="utf-8"))

    # 2) prompt user for parameters
    n_lineups      = prompt_int("How many lineups to select?", 50)
    cash_w         = prompt_float("Weight on cash_pct (0–1)?", 0.5)
    roi_w          = prompt_float("Weight on avg_roi_win (0–1)?", 0.5)
    div_lambda     = prompt_float("Diversity λ?", 0.1)
    out_csv        = prompt_str("Output CSV path", "chosen_lineups.csv")

    print("\n🎛️  Running with:")
    print(f" • n_lineups      = {n_lineups}")
    print(f" • cash_weight    = {cash_w}")
    print(f" • roi_weight     = {roi_w}")
    print(f" • diversity_lambda = {div_lambda}")
    print(f" • salary_cap     = {SALARY_CAP}")
    print(f" • output CSV     = {out_csv}\n")

    # 3) init DB
    engine  = init_db("mlb")
    session = get_session(engine)

    # 4) select lineup IDs
    weights = {
        "cash_pct":       cash_w,
        "avg_roi_win":    roi_w,
        "diversity_lambda": div_lambda
    }
    chosen_ids = choose_lineup_ids(
        session     = session,
        weights     = weights,
        num_lineups = n_lineups
    )
    if not chosen_ids:
        print("⚠️  No lineups found for that slate.")
        return

    # 5) build full export
    df = export_selected_lineups(
        session    = session,
        chosen_ids = chosen_ids,
        salary_cap = SALARY_CAP
    )

    # 6) write CSV
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote {len(df)} lineups to {out_csv}")

if __name__ == "__main__":
    main()
