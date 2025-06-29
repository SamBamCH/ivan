#!/usr/bin/env python
# scripts/mlb/choose_all.py

import os
import json
import argparse
from collections import Counter
import itertools

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from ivan.database.session import init_db, get_session
from ivan.sports.mlb.models import (
    HistoricalContest,
    MLBPlayer,
    Lineup,
)
from ivan.sports.mlb.selection import choose_lineup_ids
from ivan.core.utils import show_player_exposures, show_team_exposures

SALARY_CAP = 50000

# ────────────────────────────
# Fixed DFS‐slot order for lineup assignment
# ────────────────────────────
POSITION_ORDER = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]


def prompt_int(prompt_text: str, default: int) -> int:
    raw = input(f"{prompt_text} [{default}]: ").strip()
    return int(raw) if raw else default

def prompt_float(prompt_text: str, default: float) -> float:
    raw = input(f"{prompt_text} [{default}]: ").strip()
    return float(raw) if raw else default


def prompt_weights_for_contest(
    contest_id: int,
    contest_name: str,
    last_weights: dict[str, float]
) -> dict[str, float]:
    """
    Prompt once for this contest’s cash_pct, avg_roi_win, diversity_lambda.
    Defaults = last_weights (press Enter to reuse).
    """
    print(f"\n▶ Contest {contest_id}: \"{contest_name}\"")
    w_cash = prompt_float(
        f"  Weight on cash_pct for contest {contest_id}? ",
        last_weights["cash_pct"]
    )
    w_roi = prompt_float(
        f"  Weight on avg_roi_win for contest {contest_id}? ",
        last_weights["avg_roi_win"]
    )
    w_div = prompt_float(
        f"  Diversity λ for contest {contest_id}? ",
        last_weights["diversity_lambda"]
    )
    return {"cash_pct": w_cash, "avg_roi_win": w_roi, "diversity_lambda": w_div}


def assign_player_order(player_objs: list[MLBPlayer], slots: list[str]) -> list[MLBPlayer]:
    """
    Given 10 distinct MLBPlayer objects and the fixed slot requirements
    slots = ["P","P","C","1B","2B","3B","SS","OF","OF","OF"], assign exactly
    one player to each slot in order. Return a length‐10 list of MLBPlayer,
    in the same order as slots. Raises ValueError if an assignment is impossible.
    """
    assigned: list[MLBPlayer | None] = [None] * len(slots)
    used = set()

    def norm_positions(player: MLBPlayer) -> set[str]:
        parts = player.position.split("/")  # e.g. "SP" or "1B/OF"
        out = set()
        for p in parts:
            if p in ("SP", "RP"):
                out.add("P")
            else:
                out.add(p)
        return out

    for idx, slot_req in enumerate(slots):
        placed = False
        for p in player_objs:
            if p.id in used:
                continue
            norms = norm_positions(p)
            if slot_req in norms:
                assigned[idx] = p
                used.add(p.id)
                placed = True
                break
        if not placed:
            remaining = [p.id for p in player_objs if p.id not in used]
            raise ValueError(
                f"Cannot fill slot '{slot_req}' with any remaining player. "
                f"Remaining IDs: {remaining}"
            )

    return assigned  # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="For each entry in entries_csv, choose a new lineup and write out a combined CSV."
    )
    parser.add_argument(
        "--output_csv", "-o",
        type=str,
        default="chosen_entries.csv",
        help="Filename for the combined output (default: chosen_entries.csv)."
    )
    args = parser.parse_args()

    # ────────────────────────────────────────────
    # 1) Load config + read only first 14 columns of entries_csv
    # ────────────────────────────────────────────
    here = os.path.dirname(__file__)
    cfg_fp = os.path.abspath(os.path.join(here, "../../config/mlb.json"))
    cfg = json.load(open(cfg_fp, encoding="utf-8"))

    entries_fp = cfg["paths"]["entries_csv"]
    print(f"Loading first 14 columns of entries from {entries_fp}…")

    # Only read columns 0..13 (ignore everything after the third OF)
    df_raw = pd.read_csv(
        entries_fp,
        engine="python",
        usecols=range(14),
        header=0,
        dtype=str,
        keep_default_na=False
    )

    # Ensure base columns exist
    base_cols = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
    for col in base_cols:
        if col not in df_raw.columns:
            raise RuntimeError(f"Expected column '{col}' in entries CSV but not found.")

    # Drop any row at or below the first blank "Entry ID"
    blank_mask = df_raw["Entry ID"].str.strip() == ""
    if blank_mask.any():
        first_blank_idx = blank_mask.idxmax()
        df_raw = df_raw.loc[: first_blank_idx - 1].copy()

    # Convert Contest ID to int and group by contest
    df_raw["Contest ID"] = df_raw["Contest ID"].astype(int)
    contests_grouped = df_raw.groupby("Contest ID", sort=False)

    # ────────────────────────────────────────────
    # 2) Prompt once for default weights (press Enter to accept)
    # ────────────────────────────────────────────
    print("Enter default weights for any contest (press Enter to accept):")
    last_weights = {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1}

    # ────────────────────────────────────────────
    # 3) Initialize DB session
    # ────────────────────────────────────────────
    engine = init_db("mlb")
    session: Session = get_session(engine)

    # ────────────────────────────────────────────
    # 4) Prepare empty DataFrame with headers:
    #    base_cols + ["P","P","C","1B","2B","3B","SS","OF","OF","OF"]
    # ────────────────────────────────────────────
    desired_pos_cols = POSITION_ORDER.copy()  # ["P","P","C","1B","2B","3B","SS","OF","OF","OF"]
    final_header = base_cols + desired_pos_cols
    df_out = pd.DataFrame(columns=final_header)

    # ────────────────────────────────────────────
    # 5) Track which contests have been prompted
    # ────────────────────────────────────────────
    prompted_contests = set()

    # 6) Loop over each unique Contest ID, and accumulate exposures
    # ────────────────────────────────────────────
    # We'll keep a per-contest map of player and team-stack exposures
    player_exposures_map: dict[int, dict[str, float]] = {}
    team_exposures_map: dict[int, dict[str, float]] = {}
    num_lineups_map: dict[int, int] = {}
    STACK_SIZE = 3

    # ────────────────────────────────────────────
    # 6) Loop over each unique Contest ID
    # ────────────────────────────────────────────
    for cid, group in contests_grouped:
        # 6b) Determine how many entries → number of lineups to select
        num_entries = len(group)
        num_lineups_map[cid] = num_entries
        cname = group["Contest Name"].iloc[0]
        print(f"\n=== Contest {cid} (\"{cname}\") → {num_entries} entries ===")

        # 6c) Prompt per‐contest weights once
        if cid not in prompted_contests:
            w = prompt_weights_for_contest(
                contest_id=cid,
                contest_name=cname,
                last_weights=last_weights
            )
            last_weights = w
            prompted_contests.add(cid)
        else:
            w = last_weights

        # 6d) Run greedy‐selection logic (filters by contest_id internally)
        chosen_ids = choose_lineup_ids(
            session=session,
            weights=w,
            num_lineups=num_entries,
            contest_name=cname
        )
        if not chosen_ids:
            print(f"⚠️  No lineups chosen for contest {cid}. Moving on.")
            del num_lineups_map[cid]
            continue

        # Preload Lineup objects and all referenced MLBPlayer rows
        lineups = session.query(Lineup).filter(Lineup.id.in_(chosen_ids)).all()
        all_pids = {pid for L in lineups for pid in L.player_ids_list}
        players_map = {
            p.id: p
            for p in session.query(MLBPlayer).filter(MLBPlayer.id.in_(all_pids)).all()
        }

        # 6e) Build each chosen‐lineup row
        orig_rows = group.reset_index(drop=True)
        lineup_counter = 0

        for lid in chosen_ids:
            # Fetch the Lineup directly by its primary key
            lineup = session.query(Lineup).filter(Lineup.id == lid).one_or_none()
            if lineup is None:
                continue

            # Exactly ten player IDs (unique)
            pids = lineup.player_ids_list
            if len(pids) != 10 or len(set(pids)) != 10:
                raise ValueError(f"Lineup {lid} must have exactly 10 unique players")

            # Convert IDs → MLBPlayer objects
            objs = [players_map[pid] for pid in pids]

            # 1) Assign players to slots in POSITION_ORDER
            assigned = assign_player_order(objs, POSITION_ORDER)

            # 2) Build "Name (ID)" strings in the same order
            slots_str = [f"{pl.name} ({pl.id})" for pl in assigned]

            # 3) Pick original entry row's base data (cycle through)
            idx = lineup_counter % num_entries
            base_data = orig_rows.iloc[idx][base_cols].to_list()

            # 4) Create one combined row as a list of values,
            #    matching final_header’s order exactly.
            row_values = base_data + slots_str

            # 5) Append the row by positional assignment
            df_out.loc[len(df_out)] = row_values

            lineup_counter += 1

        # 6f) (Optional) Show exposures for this contest
        id_lists = [session.query(Lineup).filter(Lineup.id == lid).one().player_ids_list
                    for lid in chosen_ids]
        # print(f"● Exposures for contest {cid}:")
        # show_player_exposures(session, id_lists)
        # show_team_exposures(session, id_lists, stack_size=3)

        # --- now compute and store exposures for recap tables ---
        num_lineups = len(id_lists)

         # 1) Player exposures
        p_counter = Counter(pid for lst in id_lists for pid in lst)
        player_exposures_map[cid] = {f"{players_map[pid].name} ({pid})": p_counter[pid] / num_lineups for pid in p_counter}

        # 2) Team exposures: count each team if count >= STACK_SIZE in lineup
        t_counter: Counter[str] = Counter()
        for lst in id_lists:
            by_team = Counter(players_map[pid].team for pid in lst)
            for team, cnt in by_team.items():
                if cnt >= STACK_SIZE:
                    t_counter[team] += 1
        team_exposures_map[cid] = {team: t_counter[team] / num_lineups for team in t_counter}

   # Player recap
    if player_exposures_map:
        df_p = pd.DataFrame(player_exposures_map).fillna(0)
        total_lineups = sum(num_lineups_map.values())
        overall = {pname: sum(df_p.at[pname, cid] * num_lineups_map[cid] for cid in num_lineups_map) / total_lineups for pname in df_p.index}
        df_p["Overall Exposure"] = pd.Series(overall)

        ids = df_p.index.to_series().str.extract(r"\((\d+)\)$")[0].astype(int)
        players = session.query(MLBPlayer).filter(MLBPlayer.id.in_(ids.tolist())).all()
        pmap = {p.id: p for p in players}

        info = pd.DataFrame({
            "Position": ids.map(lambda i: pmap[i].position),
            "Team":     ids.map(lambda i: pmap[i].team),
            "Salary":   ids.map(lambda i: pmap[i].salary),
            "Ownership": ids.map(lambda i: getattr(pmap[i], "ownership", 0.0) / 100),
        }, index=df_p.index)

        # reorder: info, Overall Exposure, then per-contest
        contest_cols = [c for c in df_p.columns if c != "Overall Exposure"]
        cols = info.columns.tolist() + ["Overall Exposure"] + contest_cols
        df_final = pd.concat([info, df_p], axis=1)[cols]
        df_final = df_final.sort_values("Overall Exposure", ascending=False)

        # separate pitchers and batters
        pitchers_mask = df_final["Position"].str.contains("P")
        df_pitchers = df_final[pitchers_mask]
        df_batters = df_final[~pitchers_mask]

        print("\n=== Exposure Recap: Pitchers ===")
        print(df_pitchers.applymap(lambda x: f"{x:.1%}" if isinstance(x, float) else x))
        print("\n=== Exposure Recap: Batters ===")
        print(df_batters.applymap(lambda x: f"{x:.1%}" if isinstance(x, float) else x))

    # ─── Team recap (overall only) ────────────────────────────
    if team_exposures_map:
        # 1) build a DataFrame of per‐contest team exposures (teams × contests)
        df_t = pd.DataFrame(team_exposures_map).fillna(0)

        # 2) compute weighted overall exposure across all contests
        total_lineups = sum(num_lineups_map.values())
        overall_team = {
            team: sum(df_t.at[team, cid] * num_lineups_map[cid] for cid in num_lineups_map)
            / total_lineups
            for team in df_t.index
        }

        # 3) sort and print
        s = pd.Series(overall_team).sort_values(ascending=False)
        print("\n=== Exposure Recap: Teams ===")
        for team, expo in s.items():
            print(f"{team} – {expo:.1%}")

    # ────────────────────────────────────────────
    # 7) Write out the combined CSV with EXACT headers (14 columns total)
    # ────────────────────────────────────────────
    out_fp = args.output_csv
    df_out.to_csv(out_fp, index=False)
    print(f"\n✅ Wrote {len(df_out)} chosen‐lineup rows → {out_fp}")


if __name__ == "__main__":
    main()
