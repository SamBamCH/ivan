#!/usr/bin/env python3
"""
scripts/nba/autopilot.py

The autopilot switch (ivan.sports.nba.autopilot): with it on, the capture tick (scripts/nba/capture.py, a user
timer every 15 minutes) refreshes the loaded slate, fetches props at the offsets before lock and publishes +
pushes the feed to the other desk while nobody is editing here. Same switch as the desk header's "autopilot" box.

    scripts/nba/autopilot.py status        the switch, the last tick, whether the tick would stand down right now, the timer
    scripts/nba/autopilot.py on            turn it on   (the desk shows "autopilot on")
    scripts/nba/autopilot.py off           turn it off  (only Publish sends projections to the other desk)
    scripts/nba/autopilot.py tick          run one tick now, honouring the switch and the guard (what the timer does)
    scripts/nba/autopilot.py tick --force  run one tick now whatever the switch says (a test); extra args go to capture.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nba import autopilot


def show_status() -> None:
    st = autopilot.state()
    print(f"autopilot: {'ON' if st['on'] else 'off'}" + (f" since {st['since']} (by {st['by']})" if st["since"] else ""))
    lt = st.get("last_tick")
    if lt:
        print(f"last tick: {lt.get('at')} · {'ok' if lt.get('ok') else 'FAILED'} · {lt.get('result')}")
    else:
        print("last tick: none recorded")
    hold = autopilot.desk_hold()
    print("right now: " + (f"the tick would stand down ({hold})" if hold else "the tick would run"))
    try:
        out = subprocess.run(["systemctl", "--user", "list-timers", "--no-pager", "ivan-nba-capture.timer"], capture_output=True, text=True, timeout=10).stdout
        lines = [ln for ln in out.splitlines() if "ivan-nba-capture" in ln]
        print("timer: " + (lines[0].strip() if lines else "ivan-nba-capture.timer is not installed or not enabled (deploy/install_nba_timers.sh)"))
    except (OSError, subprocess.TimeoutExpired):
        pass
    print(f"switch file: {autopilot.AUTOPILOT_JSON.relative_to(PROJECT_ROOT)} · tick log: data/nba/logs/capture.log")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["status", "on", "off", "tick"])
    args, rest = ap.parse_known_args()
    if args.action == "on":
        autopilot.set_on(True, by="cli")
    elif args.action == "off":
        autopilot.set_on(False, by="cli")
    elif args.action == "tick":
        cmd = [sys.executable, str(Path(__file__).with_name("capture.py")), *rest]
        print(f"{datetime.now():%H:%M:%S} running: {' '.join(cmd)}")
        return subprocess.call(cmd)
    show_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
