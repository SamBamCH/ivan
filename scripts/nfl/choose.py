#!/usr/bin/env python3
# scripts/nfl/choose.py
"""
Choose lineups for each contest and write CSVs in DraftKings format.
- Supports BOTH Showdown (6-man) and Classic (9-man).
- Output headers EXACTLY match DK upload order.
- Uses your existing greedy chooser (choose_lineup_ids).
"""

from __future__ import annotations
import argparse
import os
import json
import pandas as pd
from collections import Counter
from datetime import datetime
from typing import List, Dict

from ivan.database.session import init_db, get_session
# Robust import: prefer NFL impl, fall back to generic selector if needed.
try:
    from ivan.sports.nfl.choose import choose_lineup_ids  # type: ignore
except Exception:
    from ivan.sports.mlb.selection import choose_lineup_ids  # type: ignore

from ivan.sports.nfl.models import Lineup, NFLPlayer

# ------------------------------- DK SLOT DEFINITIONS -------------------------------
# Showdown order (exactly 6): CPT + 5x FLEX
POSITION_ORDER_SD = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]

# Classic order (exactly 9): QB,RB,RB,WR,WR,WR,TE,FLEX,DST
POSITION_ORDER_CLASSIC = ["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"]

# DK roster_slot_id codes used for mapping
SLOT_CODE = {
    "CPT": 511,
    "FLEX": 512,
    "QB": 66,
    "RB": 67,
    "WR": 68,
    "TE": 69,
    "DST": 71,
}

BASE_COLS = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]


# ------------------------------- HELPERS -------------------------------

def prompt_float(prompt_text: str, default: float) -> float:
    raw = input(f"{prompt_text} [{default}]: ").strip()
    return float(raw) if raw else default


def prompt_weights_for_contest(
    contest_id: int,
    contest_name: str,
    last_weights: dict[str, float]
) -> dict[str, float]:
    print(f"\\n▶ Contest {contest_id}: \"{contest_name}\"")
    w_cash = prompt_float(f"  Weight on cash_pct? ", last_weights["cash_pct"])
    w_roi  = prompt_float(f"  Weight on avg_roi_win? ", last_weights["avg_roi_win"])
    w_div  = prompt_float(f"  Diversity λ? ", last_weights["diversity_lambda"])
    return {"cash_pct": w_cash, "avg_roi_win": w_roi, "diversity_lambda": w_div}


def _rs(p: NFLPlayer) -> int | None:
    try:
        return int(getattr(p, "roster_slot_id"))
    except (TypeError, ValueError):
        return None


def _sal(p: NFLPlayer) -> int:
    try:
        return int(getattr(p, "salary") or 0)
    except (TypeError, ValueError):
        return 0


def assign_player_order_showdown(player_objs: list[NFLPlayer], slots: list[str]) -> list[NFLPlayer]:
    """Fill Showdown slots strictly by roster_slot_id (511 CPT, 512 FLEX)."""
    assigned: list[NFLPlayer | None] = [None] * len(slots)
    used: set[int] = set()

    want_cpt  = slots.count("CPT")
    want_flex = slots.count("FLEX")
    have_cpt  = sum(1 for p in player_objs if _rs(p) == SLOT_CODE["CPT"])
    have_flex = sum(1 for p in player_objs if _rs(p) == SLOT_CODE["FLEX"])
    if have_cpt < want_cpt:
        raise ValueError(f"Expected at least {want_cpt} CPT(511) in lineup but found {have_cpt}.")
    if have_flex < want_flex:
        raise ValueError(f"Expected at least {want_flex} FLEX(512) in lineup but found {have_flex}.")

    for idx, slot_req in enumerate(slots):
        code = SLOT_CODE[slot_req]
        for p in player_objs:
            if p.id in used: continue
            if _rs(p) == code:
                assigned[idx] = p
                used.add(p.id)
                break
        if assigned[idx] is None:
            remain = [(p.id, _rs(p)) for p in player_objs if p.id not in used]
            raise ValueError(f"Cannot fill {slot_req}; remaining {remain}")
    return assigned  # type: ignore


def assign_player_order_classic(player_objs: list[NFLPlayer]) -> list[NFLPlayer]:
    """
    Assign players to classic slots in EXACT order:
      QB,RB,RB,WR,WR,WR,TE,FLEX,DST
    We use roster_slot_id codes when present; otherwise fall back to position.
    FLEX picks the highest-salary remaining RB/WR/TE not already used.
    """
    qbs, rbs, wrs, tes, dsts, flex_eligible = [], [], [], [], [], []
    for p in player_objs:
        slot = _rs(p)
        pos = (getattr(p, "position", None) or "").upper()
        if slot == SLOT_CODE["QB"] or pos == "QB":
            qbs.append(p)
        elif slot == SLOT_CODE["RB"] or pos == "RB":
            rbs.append(p); flex_eligible.append(p)
        elif slot == SLOT_CODE["WR"] or pos == "WR":
            wrs.append(p); flex_eligible.append(p)
        elif slot == SLOT_CODE["TE"] or pos == "TE":
            tes.append(p); flex_eligible.append(p)
        elif slot == SLOT_CODE["DST"] or pos in {"DST","D/ST","DEF"}:
            dsts.append(p)

    key = lambda p: -_sal(p)
    qbs.sort(key=key); rbs.sort(key=key); wrs.sort(key=key); tes.sort(key=key); dsts.sort(key=key)
    flex_eligible = sorted(flex_eligible, key=key)

    if not qbs:  raise ValueError("Classic lineup missing QB.")
    if len(rbs) < 2: raise ValueError("Classic lineup requires 2 RB.")
    if len(wrs) < 3: raise ValueError("Classic lineup requires 3 WR.")
    if len(tes) < 1: raise ValueError("Classic lineup requires TE.")
    if not dsts: raise ValueError("Classic lineup missing DST.")

    order: list[NFLPlayer] = []
    order.append(qbs[0])               # QB
    order += rbs[:2]                   # RB, RB
    order += wrs[:3]                   # WR, WR, WR
    order.append(tes[0])               # TE
    used_ids = {p.id for p in order}
    flex_pick = next((p for p in flex_eligible if p.id not in used_ids), None)
    if flex_pick is None:
        raise ValueError("No remaining RB/WR/TE to fill FLEX.")
    order.append(flex_pick)            # FLEX
    order.append(dsts[0])              # DST

    if len(order) != 9:
        raise ValueError(f"Classic assignment produced {len(order)} players, expected 9.")
    return order


# ------------------------------- MAIN -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Choose lineups and write DK-formatted CSVs (Classic & Showdown).")
    parser.add_argument("-o", "--output_csv",
                        type=str,
                        default="data/nfl/output/chosen_entries.csv",
                        help="Base filename for output CSV(s). We'll append .classic/.showdown and a timestamp.")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    cfg_fp = os.path.abspath(os.path.join(here, "../../config/nfl.json"))
    cfg = json.load(open(cfg_fp, encoding="utf-8"))

    # Entries CSV
    entries_fp = cfg["paths"]["entries_csv"]
    print(f"Loading entries from {entries_fp}…")
    df_raw = pd.read_csv(entries_fp, engine="python", header=0, usecols=range(14), dtype=str, keep_default_na=False)

    # Ensure required columns exist
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

    # DB session
    engine = init_db("nfl")
    session = get_session(engine)

    # Output frames (separate per format)
    df_out_sd      = pd.DataFrame(columns=BASE_COLS + POSITION_ORDER_SD)
    df_out_classic = pd.DataFrame(columns=BASE_COLS + POSITION_ORDER_CLASSIC)

    prompted_contests: set[int] = set()

    for cid, group in contests_grouped:
        num_entries = len(group)
        cname = group["Contest Name"].iloc[0]
        print(f"\\n=== Contest {cid} (\"{cname}\") → {num_entries} entries ===")

        # Prompt weights once per contest
        if cid not in prompted_contests:
            w = prompt_weights_for_contest(cid, cname, last_weights)
            last_weights = w
            prompted_contests.add(cid)
        else:
            w = last_weights

        # Choose lineup IDs
        chosen_ids = choose_lineup_ids(
            session=session,
            weights=w,
            num_lineups=num_entries,
            contest_name=cname
        )
        if not chosen_ids:
            print(f"⚠️  No lineups chosen for contest {cid}. Skipping.")
            continue

        # Load lineups and players
        lineups = session.query(Lineup).filter(Lineup.id.in_(chosen_ids)).all()
        id_to_lineup = {L.id: L for L in lineups}
        all_pids = {pid for L in lineups for pid in L.player_ids_list}
        players_map: Dict[int, NFLPlayer] = {
            p.id: p
            for p in session.query(NFLPlayer).filter(NFLPlayer.id.in_(all_pids)).all()
        }

        # Build rows
        orig_rows = group.reset_index(drop=True)
        for i, lid in enumerate(chosen_ids):
            lineup = id_to_lineup.get(lid)
            if not lineup:
                continue
            pids: List[int] = lineup.player_ids_list
            objs: List[NFLPlayer] = [players_map[pid] for pid in pids]
            base_data = orig_rows.iloc[i % num_entries][BASE_COLS].to_list()

            if len(pids) == 6:
                assigned = assign_player_order_showdown(objs, POSITION_ORDER_SD)
                slots = [f"{pl.name} ({pl.id})" for pl in assigned]
                df_out_sd.loc[len(df_out_sd)] = base_data + slots
            elif len(pids) == 9:
                assigned = assign_player_order_classic(objs)
                slots = [f"{pl.name} ({pl.id})" for pl in assigned]
                df_out_classic.loc[len(df_out_classic)] = base_data + slots
            else:
                raise ValueError(f"Unsupported lineup size {len(pids)} for lineup {lid} (expected 6 or 9).")

    # Write outputs
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    root, ext = os.path.splitext(args.output_csv)

    wrote = False
    if not df_out_sd.empty:
        out_sd = f"{root}.showdown.{ts}{ext}"
        os.makedirs(os.path.dirname(out_sd) or ".", exist_ok=True)
        df_out_sd.to_csv(out_sd, index=False)
        print(f"✅ Wrote {len(df_out_sd)} showdown rows → {out_sd}")
        wrote = True

    if not df_out_classic.empty:
        out_cl = f"{root}.classic.{ts}{ext}"
        os.makedirs(os.path.dirname(out_cl) or ".", exist_ok=True)
        df_out_classic.to_csv(out_cl, index=False)
        print(f"✅ Wrote {len(df_out_classic)} classic rows → {out_cl}")
        wrote = True

    if not wrote:
        print("⚠️  No rows written — no valid chosen lineups were found.")

if __name__ == "__main__":
    main()
