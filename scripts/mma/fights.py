#!/usr/bin/env python3
# scripts/mma/fights.py
"""
Each fight's simulation results for the card on file (what the slate page's
Fights section shows): who wins and how, the average length, and each
fighter's fantasy points: mean, quartiles, the mean when they win and when
they lose, the share of worlds at 100+, points per $1,000, strikes and
takedowns landed. Above the fights, what the card's inputs are missing.

    scripts/mma/fights.py               20,000 worlds per fight, a fixed seed
    scripts/mma/fights.py -n 100000

Reads only: nothing is stored.
"""
import argparse

from ivan.core.config import config_path, load_config
from ivan.database.session import get_session, init_db
from ivan.sports.mma.fight_report import N_SIMS, card_report, format_card
from ivan.sports.mma.models import MMABase


def main() -> int:
    p = argparse.ArgumentParser(description="Per-fight simulation results for the MMA card on file")
    p.add_argument("--config", default=str(config_path("mma")))
    p.add_argument("-n", "--sims", type=int, default=N_SIMS, help=f"worlds per fight (default {N_SIMS:,})")
    p.add_argument("--seed", type=int, default=None, help="default: fixed, or derived from IVAN_SEED")
    args = p.parse_args()
    cfg = load_config("mma", args.config)
    engine = init_db("mma")
    MMABase.metadata.create_all(engine)
    session = get_session(engine)
    try:
        rep = card_report(session, cfg, n_sims=args.sims, seed=args.seed)
    finally:
        session.close()
    if not rep["fights"]:
        print("no fights on file; run scripts/mma/ingest.py")
        return 1
    print(format_card(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
