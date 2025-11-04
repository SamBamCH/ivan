#!/usr/bin/env python3
# scripts/nba/choose.py
"""
Choose lineups for each NBA contest and write CSVs in DraftKings format.
- Supports classic format only (8-man: PG, SG, SF, PF, C, G, F, UTIL)
- Output headers EXACTLY match DK upload order.
- Uses your existing greedy chooser (choose_lineup_ids).
- Position assignment uses recursive backtracking to guarantee valid assignments.
- ALSO prints exposure DataFrames for all chosen lineups,
  grouped by position.
"""

from __future__ import annotations
import argparse
import os
import json
import pandas as pd
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from ivan.database.session import init_db, get_session
from ivan.sports.nba.choose import choose_lineup_ids
from ivan.sports.nba.models import Lineup, NBAPlayer
from ivan.sports.nba.utils import _coerce_positions

# ------------------------------- DK SLOT DEFINITIONS -------------------------------
# Classic order (exactly 8): PG,SG,SF,PF,C,G,F,UTIL
POSITION_ORDER = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

# Which base positions are eligible for each slot (indexed 0-7)
POSITION_MAP = {
    0: ["PG"],                          # PG
    1: ["SG"],                          # SG
    2: ["SF"],                          # SF
    3: ["PF"],                          # PF
    4: ["C"],                           # C
    5: ["PG", "SG"],                   # G
    6: ["SF", "PF"],                   # F
    7: ["PG", "SG", "SF", "PF", "C"],  # UTIL
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
    print(f"\n▶ Contest {contest_id}: \"{contest_name}\"")
    w_cash = prompt_float(f"  Weight on cash_pct? ", last_weights["cash_pct"])
    w_roi  = prompt_float(f"  Weight on avg_roi_win? ", last_weights["avg_roi_win"])
    w_div  = prompt_float(f"  Diversity λ? ", last_weights["diversity_lambda"])
    return {"cash_pct": w_cash, "avg_roi_win": w_roi, "diversity_lambda": w_div}


def _sal(p: NBAPlayer) -> int:
    try:
        return int(getattr(p, "salary") or 0)
    except (TypeError, ValueError):
        return 0


def _team(p: NBAPlayer) -> str:
    for attr in ("team", "team_abbrev", "team_abbreviation", "TeamAbbrev", "team_short", "team_code"):
        v = getattr(p, attr, None)
        if v:
            return str(v)
    return "UNK"


def _get_player_positions(p: NBAPlayer) -> tuple[str, ...]:
    """
    Get normalized position tuple for a player using the same logic as LP.
    Returns a tuple like ("PG",) or ("SF", "PF")
    """
    pos = getattr(p, "position", None)
    pos_tuple = _coerce_positions(pos or [])
    
    # ENSURE it's always a tuple (same fix as in LP)
    if isinstance(pos_tuple, str):
        pos_tuple = (pos_tuple,)
    elif not isinstance(pos_tuple, tuple):
        pos_tuple = tuple(pos_tuple) if pos_tuple else ()
    
    return pos_tuple


def _pos_display(p: NBAPlayer) -> str:
    """Get position string for display (e.g., 'PG', 'SF/PF')."""
    pos_tuple = _get_player_positions(p)
    return "/".join(pos_tuple) if pos_tuple else "?"


def _name(p: NBAPlayer) -> str:
    """Get player name from various possible attributes."""
    for attr in ("name", "player_name", "player_display_name", "full_name", "display_name"):
        v = getattr(p, attr, None)
        if v:
            return str(v)
    return f"#{getattr(p, 'id', '??')}"


def assign_player_order_classic(player_objs: list[NBAPlayer]) -> list[NBAPlayer]:
    """
    Assign players to classic NBA slots using recursive backtracking.
    
    Given 8 players that form a valid lineup (from the LP), find a valid assignment
    to the 8 slots: PG, SG, SF, PF, C, G, F, UTIL.
    
    Returns:
        List of 8 NBAPlayer objects in slot order.
    """
    
    def is_eligible(player: NBAPlayer, slot_idx: int) -> bool:
        """Check if a player can fill a specific slot."""
        slot_positions = POSITION_MAP[slot_idx]
        player_positions = _get_player_positions(player)
        return any(pos in slot_positions for pos in player_positions)
    
    def backtrack(slot_idx: int, assignment: list[NBAPlayer], remaining: list[NBAPlayer]) -> Optional[list[NBAPlayer]]:
        """
        Recursively try to fill slots.
        
        :param slot_idx: Current slot index (0-7)
        :param assignment: Current partial assignment
        :param remaining: Players not yet assigned
        :return: Complete valid assignment, or None if no valid assignment exists
        """
        # Base case: all slots filled
        if slot_idx == 8:
            return assignment
        
        # Try each remaining player in this slot
        for i, player in enumerate(remaining):
            if is_eligible(player, slot_idx):
                # Try this player in this slot
                new_assignment = assignment + [player]
                new_remaining = remaining[:i] + remaining[i+1:]
                
                # Recurse to fill next slot
                result = backtrack(slot_idx + 1, new_assignment, new_remaining)
                if result is not None:
                    return result
        
        # No valid assignment found with current choices
        return None
    
    # Start backtracking from slot 0
    result = backtrack(0, [], player_objs)
    
    if result is None:
        # This should never happen if the LP is generating valid lineups
        player_summary = ", ".join([f"{_name(p)} ({_pos_display(p)})" for p in player_objs])
        raise ValueError(
            f"Could not find valid slot assignment for players: {player_summary}. "
            f"This indicates a bug in the LP constraints."
        )
    
    return result


def _print_exposures(df: pd.DataFrame, header: str):
    """Pretty-print exposures sorted by exposure (high to low)."""
    if df.empty:
        return
    
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.max_columns", None):
        # Sort by exposure descending
        sorted_df = df.sort_values("exposure", ascending=False).copy()
        
        # Format exposure as percent
        sorted_df["exposure_pct"] = (sorted_df["exposure"] * 100).round(2).astype(str) + "%"
        sorted_df["ownership_pct"] = (sorted_df["ownership"]).round(2).astype(str) + "%"
        
        print(f"\n{header}")
        print(sorted_df[["name", "position", "team", "salary", "exposure_pct", "ownership_pct", "projection"]].to_string(index=False))


# ------------------------------- MAIN -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Choose NBA lineups and write DK-formatted CSV (Classic 8-man).")
    parser.add_argument("-o", "--output_csv",
                        type=str,
                        default="data/nba/output/chosen_entries.csv",
                        help="Base filename for output CSV. We'll append a timestamp.")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    cfg_fp = os.path.abspath(os.path.join(here, "../../config/nba.json"))
    cfg = json.load(open(cfg_fp, encoding="utf-8"))

    # Entries CSV
    entries_fp = cfg["paths"]["entries_csv"]
    print(f"Loading entries from {entries_fp}…")
    df_raw = pd.read_csv(entries_fp, engine="python", header=0, usecols=range(10), dtype=str, keep_default_na=False)

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
    engine = init_db("nba")
    session = get_session(engine)

    # Output frame (NBA classic only)
    df_out = pd.DataFrame(columns=BASE_COLS + POSITION_ORDER)

    prompted_contests: set[int] = set()

    # Exposure aggregators
    total_entries = 0
    classic_counts: Counter[int] = Counter()
    classic_info: Dict[int, Dict[str, object]] = {}

    for cid, group in contests_grouped:
        num_entries = len(group)
        cname = group["Contest Name"].iloc[0]
        print(f"\n=== Contest {cid} (\"{cname}\") → {num_entries} entries ===")

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
        players_map: Dict[int, NBAPlayer] = {
            p.id: p
            for p in session.query(NBAPlayer).filter(NBAPlayer.id.in_(all_pids)).all()
        }

        # Build rows
        orig_rows = group.reset_index(drop=True)
        for i, lid in enumerate(chosen_ids):
            lineup = id_to_lineup.get(lid)
            if not lineup:
                continue
            pids: List[int] = lineup.player_ids_list
            objs: List[NBAPlayer] = [players_map[pid] for pid in pids]
            base_data = orig_rows.iloc[i % num_entries][BASE_COLS].to_list()

            if len(pids) == 8:
                try:
                    assigned = assign_player_order_classic(objs)
                    slots = [f"{_name(pl)} ({pl.id})" for pl in assigned]
                    df_out.loc[len(df_out)] = base_data + slots

                    # Exposure tracking
                    total_entries += 1
                    for pl in assigned:
                        key = pl.id
                        classic_counts[key] += 1
                        classic_info[key] = {
                            "name": _name(pl),
                            "team": _team(pl),
                            "position": _pos_display(pl),
                            "salary": _sal(pl),
                            "ownership": getattr(pl, "ownership_proj", 0.0),
                            "projection": getattr(pl, "fpts_projection", 0.0) or getattr(pl, "projection", 0.0)
                        }
                except ValueError as e:
                    print(f"⚠️  Error assigning lineup {lid}: {e}")
                    continue
            else:
                print(f"⚠️  Skipping lineup {lid}: expected 8 players but got {len(pids)}")

    # Write output
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    root, ext = os.path.splitext(args.output_csv)

    if not df_out.empty:
        out_path = f"{root}.{ts}{ext}"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"\n✅ Wrote {len(df_out)} NBA classic lineups → {out_path}")
    else:
        print("\n⚠️  No rows written — no valid chosen lineups were found.")

    # Print exposures
    if total_entries > 0 and classic_counts:
        cl_rows = []
        for pid, cnt in classic_counts.items():
            info = classic_info[pid]
            cl_rows.append({
                "position": info["position"],
                "player_id": pid,
                "name": info["name"],
                "team": info["team"],
                "salary": info["salary"],
                "exposure": cnt / total_entries,
                "ownership": info["ownership"],
                "projection": info["projection"]
            })
        df_cl_expo = pd.DataFrame(cl_rows)
        _print_exposures(df_cl_expo, header=f"[NBA Classic exposures] (n={total_entries} lineups)")
    else:
        print("\n[NBA Classic exposures] No lineups selected.")


if __name__ == "__main__":
    main()