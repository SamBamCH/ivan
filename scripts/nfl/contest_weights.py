#!/usr/bin/env python3
"""
scripts/nfl/contest_weights.py

One contest's own chooser knobs (data/nfl/input/web/weights.json, which Choose passes to choose.py as --weights-json). A
contest without its own knob gets the default (method greedy, cash 0.5, roi 0.5, diversity 0.1, bankroll 3). The desk's
contests table saves through the same call as you type.

    contest_weights.py list
    contest_weights.py set 197208274 method=portfolio bankroll=4
    contest_weights.py set 197208274 cash_pct=          # blank: back to the default
    contest_weights.py clear 197208274
"""
from __future__ import annotations

import argparse
import json

from ivan.web.nfl.pipeline import DEFAULT_WEIGHTS, WEIGHT_KEYS, load_weights, set_contest_weights


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("list", "set", "clear"))
    ap.add_argument("contest_id", nargs="?")
    ap.add_argument("knobs", nargs="*", help=f"knob=value, knobs: {', '.join(WEIGHT_KEYS)}")
    a = ap.parse_args(argv)
    if a.action == "list":
        data = load_weights()
        print(f"default: {json.dumps(DEFAULT_WEIGHTS)}")
        for cid, w in data.items():
            if cid != "default":
                print(f"{cid}: {json.dumps({k: v for k, v in w.items() if k in WEIGHT_KEYS})}")
        return 0
    if not a.contest_id:
        ap.error("a contest id")
    if a.action == "clear":
        own = set_contest_weights(a.contest_id, {k: "" for k in WEIGHT_KEYS})
    else:
        values = {}
        for kv in a.knobs:
            k, _, v = kv.partition("=")
            if k not in WEIGHT_KEYS:
                ap.error(f"unknown knob {k!r} (one of {', '.join(WEIGHT_KEYS)})")
            values[k] = v
        own = set_contest_weights(a.contest_id, values)
    print(f"{a.contest_id}: {json.dumps(own) if own else 'the defaults'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
