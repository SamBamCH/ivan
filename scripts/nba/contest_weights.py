#!/usr/bin/env python3
"""
scripts/nba/contest_weights.py

The chooser's knobs per contest: what the desk's contests section sets (data/nba/input/contest_weights.json). The desk's
Choose passes the file as --weights-json, so a contest's knobs win over the config panel's; a knob left blank takes the
panel's value. Run choose.py --weights-json data/nba/input/contest_weights.json to do the same from a terminal.

    scripts/nba/contest_weights.py                                  show every contest's knobs
    scripts/nba/contest_weights.py set 123456 method=portfolio bankroll=2
    scripts/nba/contest_weights.py set 123456 cash_pct=             (blank: back to the panel's value)
    scripts/nba/contest_weights.py clear 123456
"""
from __future__ import annotations

import argparse
import json

from ivan.sports.nba import lineup_export as LE


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-contest chooser knobs (the desk's contests section).")
    ap.add_argument("action", nargs="?", default="show", choices=("show", "set", "clear"))
    ap.add_argument("contest_id", nargs="?", type=int)
    ap.add_argument("knobs", nargs="*", help=f"knob=value, knobs: {', '.join(LE.CONTEST_KNOBS)}")
    a = ap.parse_args()
    if a.action == "show":
        print(json.dumps(LE.contest_weights(), indent=1, sort_keys=True))
        return 0
    if a.contest_id is None:
        ap.error("a contest id is needed")
    if a.action == "clear":
        LE.set_contest_weights(a.contest_id, {k: None for k in LE.CONTEST_KNOBS})
    else:
        LE.set_contest_weights(a.contest_id, dict(kv.split("=", 1) for kv in a.knobs))
    print(json.dumps(LE.contest_weights().get(str(a.contest_id), {}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
