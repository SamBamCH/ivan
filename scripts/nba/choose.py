#!/usr/bin/env python3
"""
scripts/nba/choose.py

Choose lineups for each contest in the entries file and write the DraftKings
upload CSV, the exposures and per-lineup detail, the chooser diagnostics and
the per-entry simulated ROI traces. The work is ivan.sports.nba.contest.export;
this is the command line around it.

    scripts/nba/choose.py                                   prompts for weights per contest
    scripts/nba/choose.py --no-prompt                       config choose.* defaults, no prompts
    scripts/nba/choose.py --method portfolio --bankroll 3
    scripts/nba/choose.py --method slate [--bankroll-dollars 1000]   the whole slate as one book from one bankroll (the default method; the
                                                            bankroll is read from our results when none is given, scripts/bankroll.py)
    scripts/nba/choose.py --weights-json '{"default": {"cash_pct": 0.3}, "123456": {"method": "portfolio"}}'
"""
from __future__ import annotations

import argparse

from ivan.core.config import config_path, load_config
from ivan.sports.nba.contest import export as LE
from ivan.sports.nba.contest.choose import DEFAULT_WEIGHTS
from ivan.sports.nba import slates as S


def prompt_float(prompt_text: str, default: float) -> float:
    raw = input(f"{prompt_text} [{default}]: ").strip()
    return float(raw) if raw else default


def prompt_weights(contest_id: int, contest_name: str, last: dict) -> dict:
    print(f"\n> Contest {contest_id}: \"{contest_name}\"")
    return {**last,
            "cash_pct": prompt_float("  Weight on cash_pct?", last["cash_pct"]),
            "avg_roi_win": prompt_float("  Weight on avg_roi_win?", last["avg_roi_win"]),
            "diversity_lambda": prompt_float("  Diversity lambda?", last["diversity_lambda"])}


def main() -> int:
    ap = argparse.ArgumentParser(description="Choose NBA lineups and write the DraftKings upload CSV.")
    S.add_arg(ap)
    ap.add_argument("--config", default=str(config_path("nba")), help="path to the nba json config file")
    ap.add_argument("-o", "--output_csv", default=None, help="base name of the files written (stamped); default: the slate's output/chosen_entries.csv")
    ap.add_argument("--entries", default=None, help="entries CSV (default: paths.entries_csv from config, or $IVAN_ENTRIES_CSV)")
    w = ap.add_argument_group("weights (any of these disables the interactive prompt)")
    w.add_argument("--cash-pct", type=float, default=None, help="weight on cash_pct for every contest")
    w.add_argument("--avg-roi-win", type=float, default=None, help="weight on avg_roi_win for every contest")
    w.add_argument("--diversity-lambda", type=float, default=None, help="diversity lambda for every contest")
    w.add_argument("--method", choices=("greedy", "portfolio", "slate"), default=None,
                   help="slate (every contest at once as one book in dollars; the config default), greedy (quality + diversity) or portfolio (one contest's book log utility)")
    w.add_argument("--bankroll", type=float, default=None, help="portfolio method: utility scale in multiples of the entries' fees (default 3)")
    w.add_argument("--bankroll-dollars", type=float, default=None, help="slate method: the bankroll in dollars (default config choose.bankroll_dollars; null there = read from our results, scripts/bankroll.py)")
    w.add_argument("--weights-json", default=None, help="file or JSON literal: {\"default\": {...}, \"<contest id or name>\": {...}}")
    w.add_argument("--no-prompt", action="store_true", help="use defaults for anything not given; never prompt")
    a = ap.parse_args()
    cfg = load_config("nba", a.config)
    defaults = {**DEFAULT_WEIGHTS, **{k: v for k, v in (cfg.get("choose") or {}).items() if k in DEFAULT_WEIGHTS}}
    weights_for = LE.weights_resolver(flags={"cash_pct": a.cash_pct, "avg_roi_win": a.avg_roi_win, "diversity_lambda": a.diversity_lambda,
                                             "method": a.method, "bankroll": a.bankroll, "bankroll_dollars": a.bankroll_dollars},
                                      overrides=LE.load_weights_json(a.weights_json), no_prompt=a.no_prompt, prompter=prompt_weights,
                                      defaults=defaults)
    dg, session = S.open(a.slate)
    LE.run_choose(session, a.entries or str(S.entries_csv()), a.output_csv or str(S.chosen_entries_csv()), weights_for)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
