#!/usr/bin/env python3
"""
scripts/nba/clear_all.py

Empty the NBA working tables: every generated lineup, the lineup-pool snapshots taken at generate, every
per-lineup simulation result and every contest sim report. The pool (nba_player), the props, the captures,
the box scores, contests and payouts, and the files on disk (edits, rotations, output/) are untouched.
The same clear as `generate_lineups.py --fresh`, plus the pool snapshots and the sim reports.

    scripts/nba/clear_all.py              # show the counts, ask, then delete
    scripts/nba/clear_all.py --yes        # no prompt
    scripts/nba/clear_all.py --dry-run    # counts only
"""
import argparse

from sqlalchemy import text

from ivan.sports.nba import slates as S

# children before parents (sim results point at lineups)
TABLES = [
    ("lineup_simulation_results", "simulation results"),
    ("contest_sim_report", "contest sim reports"),
    ("lineup", "lineups"),
    ("lineup_pool", "pool snapshots at generate"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("--yes", "-y", action="store_true", help="delete without asking")
    ap.add_argument("--dry-run", action="store_true", help="print the counts and exit")
    args = ap.parse_args()
    dg, session = S.open(args.slate)
    try:
        counts = {t: int(session.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() or 0) for t, _ in TABLES}
        for t, label in TABLES:
            print(f"{counts[t]:>8,}  {label} ({t})")
        if args.dry_run:
            return 0
        if not any(counts.values()):
            print("nothing to clear")
            return 0
        if not args.yes and input("delete all of these? [y/N] ").strip().lower() not in ("y", "yes"):
            print("left as they were")
            return 1
        for t, label in TABLES:
            n = session.execute(text(f"DELETE FROM {t}")).rowcount
            print(f"cleared {int(n or 0):,} {label}")
        session.commit()
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
