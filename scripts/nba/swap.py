import argparse
import csv
import json
from pathlib import Path
from ivan.database.session import get_session, init_db
from ivan.sports.nba.refactor.post_lock.entry_parsing import parse_entries_csv
from ivan.sports.nba.refactor.post_lock.slate import lateswap_slate
from ivan.sports.nba.refactor.pre_lock.lineup import SLOT_NAMES
from ivan.sports.nba.models import HistoricalContest


def _write_best(session, path: Path, rows):
    """Write best lineups in DK-like format with only player IDs per slot."""
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Entry ID', 'Contest Name', 'Contest ID', 'Entry Fee', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'])
        for entry, best_lineup, exp_pay, _ in rows:
            # Ensure length 8
            lineup_ids = list(best_lineup)[:8] + [""] * max(0, 8 - len(best_lineup))
            contest = session.query(HistoricalContest).filter_by(contest_id=entry.contest_id).first()
            contest_name = contest.name if contest else ""
            entry_fee = contest.entry_fee if contest else ""
            w.writerow([
                entry.entry_id,
                contest_name,
                entry.contest_id,
                entry_fee,
                *lineup_ids[:8]
            ])


def _write_candidates(path: Path, rows):
    """Write candidate lineups with player IDs only."""
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Entry ID', 'Contest ID', 'Candidate Index', 'Expected Payout', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'])
        for entry_id, contest_id, cand_idx, lineup_players, exp_pay, _ in rows:
            lineup_ids = list(lineup_players)[:8] + [""] * max(0, 8 - len(lineup_players))
            w.writerow([
                entry_id,
                contest_id,
                cand_idx,
                f"{exp_pay:.2f}",
                *lineup_ids[:8]
            ])


def main():
    parser = argparse.ArgumentParser(description="NBA Late Swap Optimization")
    parser.add_argument("--entries-csv", default='data/nba/input/entries.csv', help="Path to DK entries export CSV")
    parser.add_argument("--num-sims", type=int, default=1000, help="Number of Monte Carlo simulations")
    parser.add_argument("--num-candidates", type=int, default=20, help="Number of candidates per entry")
    parser.add_argument("--field-samples", type=int, default=5000, help="Max field lineups sampled from DK")
    parser.add_argument("--player-min-fpts", type=float, default=0.0, help="Minimum fpts_mean threshold for player pool")
    parser.add_argument("--out-best", default="data/nba/output/lateswap_best.csv", help="Output CSV for best lineups")
    parser.add_argument("--out-candidates", default="data/nba/output/lateswap_candidates.csv", help="Output CSV for all candidate stats")
    args = parser.parse_args()

    engine = init_db('nba')
    session = get_session(engine)

    entries_by_contest = parse_entries_csv(args.entries_csv, session)

    best_rows, cand_rows = lateswap_slate(
        session=session,
        entries=entries_by_contest,
        num_candidates_per_entry=args.num_candidates,
        num_field_samples=args.field_samples,
        num_sims=args.num_sims,
        player_min_fpts=args.player_min_fpts,
    )

    # Extract id_to_fp by contest (will be populated by slate function)
    _write_best(session, Path(args.out_best), best_rows)
    _write_candidates(Path(args.out_candidates), cand_rows)
    print(f"Best lineups written -> {args.out_best}")
    print(f"All candidates written -> {args.out_candidates}")


if __name__ == "__main__":
    main()