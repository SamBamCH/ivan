#!/usr/bin/env python3
# scripts/mma/scoring_weights.py
"""
What a significant strike and a takedown are worth in DraftKings points on real fights
(ivan.sports.mma.scoring_fit, on data/mma/derived/ufcstats/fights.parquet), beside the weights in
config/mma.json. Prints only; put the figures in config/mma.json -> scoring by hand.

    scripts/mma/scoring_weights.py [--compare 0.8,5 0.63,5]
"""
import argparse

from ivan.core.config import config_path, load_config
from ivan.sports.mma.scoring_fit import FIGHTS, fit, format_fit, load


def main() -> int:
    p = argparse.ArgumentParser(description="Fit the MMA strike and takedown weights on real fights")
    p.add_argument("--config", default=str(config_path("mma")))
    p.add_argument("--compare", nargs="*", default=[], help="other weight pairs to show, as strike,takedown")
    args = p.parse_args()
    if not FIGHTS.exists():
        print(f"{FIGHTS} is not on disk (the ufcstats scrape)")
        return 1
    sc = load_config("mma", args.config).get("scoring", {})
    pairs = [(float(sc.get("sig_strike", 0)), float(sc.get("takedown", 0)))] + [tuple(float(v) for v in c.split(",")) for c in args.compare]
    print(format_fit(fit(load(), compare=tuple(pairs))))
    print(f"config now: scoring.sig_strike {pairs[0][0]:g}, scoring.takedown {pairs[0][1]:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
