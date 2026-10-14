#!/usr/bin/env python3
# scripts/nfl/choose.py
"""
Choose lineups for each contest and write CSVs in DraftKings format.
- Supports BOTH Showdown (6-man) and Classic (9-man).
- Output headers EXACTLY match DK upload order.
- Uses your existing greedy chooser (choose_lineup_ids).
- ALSO prints exposure DataFrames for all chosen lineups:
  - Showdown: separated by roster_slot (CPT, FLEX)
  - Classic: grouped by position (QB, RB, WR, TE, DST)
- Writes, next to the upload files, an exposures CSV and a per-lineup detail CSV
  (one row per player per chosen entry, with the lineup's simulated cash % and
  ROI) that the web UI's Exposures page reads.
"""

from __future__ import annotations
import argparse
import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from ivan.core.config import load_config
from ivan.sports.nfl.paths import OUTPUT_DIR
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.choose import choose_lineup_ids, sim_roi_traces

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
CODE_TO_SLOT = {v: k for k, v in SLOT_CODE.items()}

BASE_COLS = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]


def parse_money(x) -> float:
    """'$0.50' -> 0.5"""
    try:
        return float(str(x).replace("$", "").replace(",", "").strip() or 0)
    except ValueError:
        return 0.0


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


def _build_weights_resolver(args, defaults: dict[str, float]):
    """
    Per-contest weight resolution, highest priority first:
      --weights-json {"<contest id or name>": {...}}  >  --weights-json {"default": {...}}
      >  --cash-pct / --avg-roi-win / --diversity-lambda  >  interactive prompt.
    Any CLI weight, a weights file, or --no-prompt makes the run non-interactive.
    """
    overrides: dict = {}
    if args.weights_json:
        raw = args.weights_json
        overrides = json.loads(Path(raw).read_text() if Path(raw).exists() else raw)
    flags = {k: v for k, v in (("cash_pct", args.cash_pct),
                               ("avg_roi_win", args.avg_roi_win),
                               ("diversity_lambda", args.diversity_lambda)) if v is not None}
    non_interactive = args.no_prompt or bool(flags) or bool(overrides)

    def weights_for(cid: int, cname: str, last: dict[str, float]) -> dict[str, float]:
        per = overrides.get(str(cid)) or overrides.get(cname) or {}
        if per or non_interactive:
            return {**defaults, **flags, **overrides.get("default", {}), **per}
        return prompt_weights_for_contest(cid, cname, last)

    return weights_for


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


def _team(p: NFLPlayer) -> str:
    for attr in ("team", "team_abbrev", "team_abbreviation", "TeamAbbrev", "team_short", "team_code"):
        v = getattr(p, attr, None)
        if v:
            return str(v)
    return "UNK"


def _pos(p: NFLPlayer) -> str:
    v = getattr(p, "position", None)
    if v:
        return str(v).upper()
    slot = _rs(p)
    if slot in CODE_TO_SLOT:
        return CODE_TO_SLOT[slot]
    return "?"


def _name(p: NFLPlayer) -> str:
    for attr in ("name", "player_name", "player_display_name", "full_name", "display_name"):
        v = getattr(p, attr, None)
        if v:
            return str(v)
    return f"#{getattr(p, 'id', '??')}"


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
            if p.id in used:
                continue
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
        pos = _pos(p)
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


def _print_exposures(df: pd.DataFrame, header: str, by_col: str):
    """Pretty-print exposures grouped by `by_col` (e.g., roster_slot or position)."""
    if df.empty:
        return
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.max_columns", None):
        groups = df[by_col].unique().tolist()
        # optional: stable, sensible group order
        desired_order = ["CPT", "FLEX", "QB", "RB", "WR", "TE", "DST"]
        groups.sort(key=lambda g: desired_order.index(g) if g in desired_order else len(desired_order))
        print(f"\n{header}")
        for g in groups:
            sub = df[df[by_col] == g].copy().sort_values("exposure", ascending=False)
            if sub.empty:
                continue
            # format exposure as percent
            sub["exposure"] = (sub["exposure"] * 100).round(2).astype(str) + "%"
            sub["ownership"] = (sub["ownership"]).round(2).astype(str) + "%"
            print(f"\n[{g}]")
            print(sub[["name", "team", "position", "salary", "exposure", "ownership", "projection"]].to_string(index=False))


# ------------------------------- MAIN -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Choose lineups and write DK-formatted CSVs (Classic & Showdown).")
    parser.add_argument("-o", "--output_csv",
                        type=str,
                        default=str(OUTPUT_DIR / "chosen_entries.csv"),
                        help="Base filename for output CSV(s). We'll append .classic/.showdown and a timestamp.")
    parser.add_argument("--entries", default=None,
                        help="Entries CSV (default: paths.entries_csv from config, or $IVAN_ENTRIES_CSV)")
    w = parser.add_argument_group("weights (any of these disables the interactive prompt)")
    w.add_argument("--cash-pct", type=float, default=None, help="Weight on cash_pct for every contest")
    w.add_argument("--avg-roi-win", type=float, default=None, help="Weight on avg_roi_win for every contest")
    w.add_argument("--diversity-lambda", type=float, default=None, help="Diversity lambda for every contest")
    w.add_argument("--weights-json", default=None,
                   help='JSON file or literal: {"default": {...}, "<contest id>": {...}} keyed by contest id or name')
    w.add_argument("--no-prompt", action="store_true", help="Use defaults for anything not given; never prompt")
    args = parser.parse_args()

    cfg = load_config("nfl")

    # Entries CSV (default: first 10 cols which match Showdown; Classic has more but we only need BASE_COLS)
    entries_fp = args.entries or cfg["paths"]["entries_csv"]
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

    # Default weights; resolved once per contest from CLI/JSON, or prompted when interactive
    last_weights = {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1}
    weights_for = _build_weights_resolver(args, last_weights)

    # DB session
    engine = init_db("nfl")
    session = get_session(engine)

    # Output frames (separate per format)
    df_out_sd      = pd.DataFrame(columns=BASE_COLS + POSITION_ORDER_SD)
    df_out_classic = pd.DataFrame(columns=BASE_COLS + POSITION_ORDER_CLASSIC)

    prompted_contests: set[int] = set()

    # ---------------- EXPO AGGREGATORS ----------------
    total_sd_entries = 0
    total_classic_entries = 0

    # Showdown exposures keyed by (player_id, roster_slot)
    sd_counts: Counter[Tuple[int, str]] = Counter()
    sd_info: Dict[Tuple[int, str], Dict[str, object]] = {}

    # Classic exposures keyed by player_id (group later by position)
    classic_counts: Counter[int] = Counter()
    classic_info: Dict[int, Dict[str, object]] = {}

    # Per-lineup detail: one row per player per chosen entry
    lineup_rows: List[Dict[str, object]] = []
    # What the chooser saw per contest (candidates, picks, greedy trace): for the web UI's Choose page
    choose_viz: Dict[str, dict] = {}
    # Per-entry simulated ROI traces of the picks, kept for the tax backtest (the sim tables are cleared weekly)
    sim_rows: List[tuple] = []

    def _detail_rows(base_data, fmt: str, lid: int, slots: List[str], assigned: List[NFLPlayer]):
        sim = stats.get("by_lineup", {}).get(lid, {})
        for k, (slot, pl) in enumerate(zip(slots, assigned)):
            lineup_rows.append({
                "entry_id": base_data[0], "contest_name": base_data[1], "contest_id": base_data[2],
                "entry_fee": base_data[3], "format": fmt, "lineup_id": lid, "slot_no": k, "slot": slot,
                "player_id": pl.id, "name": _name(pl), "team": _team(pl), "opponent": getattr(pl, "opponent", None),
                "position": _pos(pl), "salary": _sal(pl), "projection": pl.fpts_mean, "ownership": pl.ownership_proj,
                "cash_pct": sim.get("cash_pct"), "avg_roi": sim.get("avg_roi"),
            })

    for cid, group in contests_grouped:
        num_entries = len(group)
        cname = group["Contest Name"].iloc[0].strip()
        print(f"\n=== Contest {cid} \"{cname}\" → {num_entries} entries ===")

        # Resolve weights once per contest
        if cid not in prompted_contests:
            w = weights_for(cid, cname, last_weights)
            last_weights = w
            prompted_contests.add(cid)
        else:
            w = last_weights
        print(f"  weights: cash_pct={w['cash_pct']} avg_roi_win={w['avg_roi_win']} diversity_lambda={w['diversity_lambda']}")

        # Choose lineup IDs
        stats: dict = {}
        chosen_ids = choose_lineup_ids(
            session=session,
            weights=w,
            num_lineups=num_entries,
            contest_name=cname,
            stats=stats,
        )
        if stats:
            print(f"  candidates: {stats['results']:,} simulated"
                  + (f" → {stats['after_roi_filter']:,} with ROI ≥ 0" if stats['after_roi_filter'] != stats['results'] else "")
                  + (f" → {stats['valid']:,} with all players still on the slate" if stats['valid'] != stats['after_roi_filter'] else "")
                  + f" → {stats['unique']:,} unique")
            if stats["missing_players"]:
                print(f"  ⚠️  {stats['missing_players']} player id(s) in simulated lineups are no longer on the slate")
        if not chosen_ids:
            print(f"  ⚠️  No lineups chosen for contest {cid}. Skipping.")
            continue
        if stats.get("viz"):
            choose_viz[str(cid)] = {"contest_name": cname, "entries": num_entries, **stats["viz"]}
        print(f"  chosen {stats['chosen']}: mean ROI {stats['chosen_roi']:+.1%}, cash {stats['chosen_cash']:.1%}"
              f"  (top-{stats['chosen']} by quality alone: ROI {stats['top_quality_roi']:+.1%}, cash {stats['top_quality_cash']:.1%})")

        traces = sim_roi_traces(session, cname, chosen_ids)

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
            sim_rows.append((int(base_data[0]), int(cid), int(lid), parse_money(base_data[3]), traces.get(lid)))

            if len(pids) == 6:
                assigned = assign_player_order_showdown(objs, POSITION_ORDER_SD)
                slots = [f"{_name(pl)} ({pl.id})" for pl in assigned]
                df_out_sd.loc[len(df_out_sd)] = base_data + slots
                _detail_rows(base_data, "showdown", lid, POSITION_ORDER_SD, assigned)

                # EXPO: showdown by roster_slot
                total_sd_entries += 1
                for pl in assigned:
                    slot_label = CODE_TO_SLOT.get(_rs(pl), "UNK")  # prefer explicit roster_slot
                    key = (pl.id, slot_label)
                    sd_counts[key] += 1
                    sd_info[key] = {
                        "name": _name(pl),
                        "team": _team(pl),
                        "position": _pos(pl),
                        "salary": _sal(pl),
                        "roster_slot": slot_label,
                        "ownership": pl.ownership_proj,
                        "projection": pl.fpts_p50
                    }

            elif len(pids) == 9:
                assigned = assign_player_order_classic(objs)
                slots = [f"{_name(pl)} ({pl.id})" for pl in assigned]
                df_out_classic.loc[len(df_out_classic)] = base_data + slots
                _detail_rows(base_data, "classic", lid, POSITION_ORDER_CLASSIC, assigned)

                # EXPO: classic by position (player base position)
                total_classic_entries += 1
                for pl in assigned:
                    key = pl.id
                    classic_counts[key] += 1
                    classic_info[key] = {
                        "name": _name(pl),
                        "team": _team(pl),
                        "position": _pos(pl),  # base position used for grouping
                        "salary": _sal(pl),
                        "ownership": pl.ownership_proj,
                        "projection": pl.fpts_p50
                    }
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

    # ---------------- PRINT EXPOSURES ----------------
    # Showdown exposures DataFrame (separated by roster_slot)
    if total_sd_entries > 0 and sd_counts:
        sd_rows = []
        for (pid, slot_label), cnt in sd_counts.items():
            info = sd_info[(pid, slot_label)]
            sd_rows.append({
                "roster_slot": slot_label,
                "player_id": pid,
                "name": info["name"],
                "team": info["team"],
                "position": info["position"],
                "salary": info["salary"],
                "exposure": cnt / total_sd_entries,  # fraction of SD lineups
                "ownership": info["ownership"],
                "projection": info["projection"]
            })
        df_sd_expo = pd.DataFrame(sd_rows)
        _print_exposures(df_sd_expo, header=f"[Showdown exposures] (n={total_sd_entries} SD lineups)", by_col="roster_slot")
    else:
        print("\n[Showdown exposures] No showdown lineups selected.")

    # Classic exposures DataFrame (grouped by position)
    if total_classic_entries > 0 and classic_counts:
        cl_rows = []
        for pid, cnt in classic_counts.items():
            info = classic_info[pid]
            cl_rows.append({
                "position": info["position"],
                "player_id": pid,
                "name": info["name"],
                "team": info["team"],
                "salary": info["salary"],
                "exposure": cnt / total_classic_entries,  # fraction of Classic lineups
                "ownership": info["ownership"],
                "projection": info["projection"]
            })
        df_cl_expo = pd.DataFrame(cl_rows)
        _print_exposures(df_cl_expo, header=f"[Classic exposures] (n={total_classic_entries} Classic lineups)", by_col="position")
    else:
        print("\n[Classic exposures] No classic lineups selected.")

    # Exposures as data, next to the chosen-entries CSVs (consumed by the web UI)
    frames = []
    if total_sd_entries > 0 and sd_counts:
        frames.append(df_sd_expo.assign(format="showdown"))
    if total_classic_entries > 0 and classic_counts:
        frames.append(df_cl_expo.assign(format="classic"))
    if frames:
        out_expo = f"{root}.exposures.{ts}{ext}"
        pd.concat(frames, ignore_index=True).to_csv(out_expo, index=False)
        print(f"✅ Wrote exposures → {out_expo}")
    if lineup_rows:
        out_detail = f"{root}.lineups.{ts}{ext}"
        pd.DataFrame(lineup_rows).to_csv(out_detail, index=False)
        print(f"✅ Wrote per-lineup detail → {out_detail}")
    if choose_viz:
        out_viz = f"{root}.choose.{ts}.json"
        Path(out_viz).write_text(json.dumps(choose_viz))
        print(f"✅ Wrote chooser diagnostics → {out_viz}")
    if sim_rows:
        n_sims = max((len(t) for *_, t in sim_rows if t is not None), default=0)
        roi = np.full((len(sim_rows), n_sims), np.nan, dtype=np.float32)
        for k, (*_, t) in enumerate(sim_rows):
            if t is not None:
                roi[k, :len(t)] = t[:n_sims]
        out_sim = f"{root}.simroi.{ts}.npz"
        np.savez_compressed(out_sim, entry_id=np.array([r[0] for r in sim_rows], dtype=np.int64),
                            contest_id=np.array([r[1] for r in sim_rows], dtype=np.int64),
                            lineup_id=np.array([r[2] for r in sim_rows], dtype=np.int64),
                            entry_fee=np.array([r[3] for r in sim_rows], dtype=np.float64), roi=roi)
        print(f"✅ Wrote simulated ROI traces ({len(sim_rows)} entries x {n_sims} sims) → {out_sim}")


if __name__ == "__main__":
    main()
