#!/usr/bin/env python3
"""
scripts/nfl/choose.py

Choose lineups for each contest in the entries file and write DraftKings upload
CSVs (showdown and classic), the exposures and per-lineup detail the web UI
reads, the chooser diagnostics and the per-entry simulated ROI traces. The
work is ivan.sports.nfl.lineup_export; this is the command line around it.
"""
from __future__ import annotations

import argparse

from ivan.core.config import load_config
from ivan.database.session import get_session, init_db
from ivan.sports.nfl import lineup_export as LE
from ivan.sports.nfl.paths import OUTPUT_DIR


def prompt_float(prompt_text: str, default: float) -> float:
    raw = input(f"{prompt_text} [{default}]: ").strip()
    return float(raw) if raw else default


def prompt_weights(contest_id: int, contest_name: str, last: dict) -> dict:
    print(f"\n> Contest {contest_id}: \"{contest_name}\"")
    return {"cash_pct": prompt_float("  Weight on cash_pct?", last["cash_pct"]),
            "avg_roi_win": prompt_float("  Weight on avg_roi_win?", last["avg_roi_win"]),
            "diversity_lambda": prompt_float("  Diversity lambda?", last["diversity_lambda"])}


def main() -> int:
    ap = argparse.ArgumentParser(description="Choose lineups and write DK-formatted CSVs (classic and showdown).")
    ap.add_argument("-o", "--output_csv", default=str(OUTPUT_DIR / "chosen_entries.csv"),
                    help="Base filename for the output CSVs; .classic/.showdown and a timestamp are appended.")
    ap.add_argument("--entries", default=None, help="Entries CSV (default: paths.entries_csv from config, or $IVAN_ENTRIES_CSV)")
    w = ap.add_argument_group("weights (any of these disables the interactive prompt)")
    w.add_argument("--cash-pct", type=float, default=None, help="Weight on cash_pct for every contest")
    w.add_argument("--avg-roi-win", type=float, default=None, help="Weight on avg_roi_win for every contest")
    w.add_argument("--diversity-lambda", type=float, default=None, help="Diversity lambda for every contest")
    w.add_argument("--weights-json", default=None,
                   help='JSON file or literal: {"default": {...}, "<contest id>": {...}} keyed by contest id or name')
    w.add_argument("--no-prompt", action="store_true", help="Use defaults for anything not given; never prompt")
    a = ap.parse_args()

    cfg = load_config("nfl")
    weights_for = LE.weights_resolver(flags={"cash_pct": a.cash_pct, "avg_roi_win": a.avg_roi_win, "diversity_lambda": a.diversity_lambda},
                                      overrides=LE.load_weights_json(a.weights_json), no_prompt=a.no_prompt, prompter=prompt_weights)
    session = get_session(init_db("nfl"))
    LE.run_choose(session, a.entries or cfg["paths"]["entries_csv"], a.output_csv, weights_for)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
