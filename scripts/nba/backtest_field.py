#!/usr/bin/env python3
"""
scripts/nba/backtest_field.py

The field backtest (ivan.sports.nba.field_backtest): this season's real contest fields (the final standings exports,
derived by scripts/nba/standings.py) against the field model rebuilt on each slate's last capture before lock, the
ownership the desk showed at lock, and the softmax ownership model. Ownership error per contest, the fields' shape
side by side, calibration, bias by salary tier and starter, the largest misses.

    scripts/nba/backtest_field.py                                 # every final derived contest, model field of 3,000 entries
    scripts/nba/backtest_field.py --since 2026-10-25 --field-size 10000 --workers 8
    scripts/nba/backtest_field.py --contest-ids 197118943,197177737
    scripts/nba/backtest_field.py --no-model                      # only the lock and softmax ownership (no ILPs, seconds)

Output: data/nba/diagnostics/field_backtest/<stamp>/{report.txt, contests.csv, players.csv, summary.json}; the report is
also printed. The model field per slate is cached under data/nba/derived/field_backtest/ (a second run is instant until
the public projection, the parameters or the seed change). About 45 s per slate at 3,000 entries on eight workers.
"""
import argparse

from ivan.sports.nba import field_backtest as FB
from ivan.sports.nba import slates as S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_arg(ap)
    ap.add_argument("--since", default=None, help="first slate day (YYYY-MM-DD)")
    ap.add_argument("--until", default=None, help="last slate day")
    ap.add_argument("--contest-ids", default=None, help="only these derived contests")
    ap.add_argument("--field-size", type=int, default=3000, help="entries in the model field per slate (default 3,000, the desk's ownership field)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-model", action="store_true", help="skip the two-group field (lock and softmax ownership only)")
    a = ap.parse_args()
    dg, session = S.open(a.slate)
    ids = [int(x) for x in a.contest_ids.split(",") if x.strip()] if a.contest_ids else None
    out = FB.run(session, field_size=a.field_size, workers=a.workers, seed=a.seed, since=a.since, until=a.until, contest_ids=ids,
                 with_model=not a.no_model, log=lambda m: print(m, flush=True))
    print()
    print(out["report"])
    print(f"written {out['dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
