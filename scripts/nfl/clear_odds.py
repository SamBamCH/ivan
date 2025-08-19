# scripts/nfl/clear_odds.py
"""
Delete ALL rows from the NFLOdds table.

Usage:
  python scripts/nfl/clear_odds.py --dry-run
  python scripts/nfl/clear_odds.py --force
"""

import argparse
from sqlalchemy import delete

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import NFLOdds


def clear_odds(*, dry_run: bool = False) -> int:
    """
    Deletes all rows from NFLOdds.
    Returns the number of rows that were (or would be) deleted.
    """
    db = init_db('nfl')
    session = get_session(db)
    try:
        total = session.query(NFLOdds).count()
        if dry_run:
            print(f"[DRY RUN] Would delete {total} NFLOdds rows.")
            return total

        # SQLAlchemy 2.x preferred path
        try:
            session.execute(delete(NFLOdds))
        except Exception:
            # Fallback for older SQLAlchemy
            session.query(NFLOdds).delete(synchronize_session=False)

        session.commit()
        print(f"[OK] Deleted {total} NFLOdds rows.")
        return total
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Delete ALL rows from NFLOdds.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many rows would be deleted without deleting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.force:
        ans = input(
            "This will DELETE ALL rows from NFLOdds. Type 'yes' to continue: "
        ).strip().lower()
        if ans != "yes":
            print("Aborted.")
            return

    clear_odds(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
