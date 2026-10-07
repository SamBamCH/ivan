import argparse
import csv
import json
from pathlib import Path
from ivan.database.session import get_session, init_db
from ivan.sports.nba.post_lock.entry_parsing import parse_entries_csv
from ivan.sports.nba.post_lock.slate import lateswap_slate
from ivan.sports.nba.pre_lock.lineup import SLOT_NAMES
from ivan.sports.nba.pre_lock.slate import assign_player_order_classic
from ivan.sports.nba.models import HistoricalContest, NBAPlayer
from ivan.sports.nba.paths import DEFAULT_ENTRIES_CSV, LATESWAP_BEST_CSV

def _apply_locked_slots(entry, lineup_pids):
    locked_ids = set(entry.locked_player_ids or [])
    original_slot_player = [entry.slots[s] for s in SLOT_NAMES]

    # Validate all locked players still present
    missing_locked = [pid for pid in locked_ids if pid not in lineup_pids]
    if missing_locked:
        raise ValueError(f"Locked players missing from lineup: {missing_locked}")

    # Place locked players in their original slots
    final_slots = [None] * 8
    for i, pid in enumerate(original_slot_player):
        if pid in locked_ids:
            final_slots[i] = pid

    # Fill remaining slots with remaining players (preserve order)
    remaining = [pid for pid in lineup_pids if pid not in locked_ids]
    it = iter(remaining)
    for i in range(8):
        if final_slots[i] is None:
            final_slots[i] = next(it, None)

    if any(p is None for p in final_slots):
        raise ValueError("Lineup has fewer than 8 players after enforcing locks.")

    return final_slots


def _write_best(session, path: Path, rows):
    """Write best lineups in DK-compatible format: 'PlayerName (ID)' per slot.

    Assumes best_lineup is already in DK roster order: [PG, SG, SF, PF, C, G, F, UTIL].
    """
    # Gather all player IDs across all lineups
    all_pids = set()
    for entry, best_lineup, exp_pay, _ in rows:
        all_pids.update(best_lineup)

    players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(all_pids)).all()
    player_map = {p.id: p for p in players}

    contest_ids = set(entry.contest_id for entry, _, _, _ in rows)
    contests = session.query(HistoricalContest).filter(HistoricalContest.contest_id.in_(contest_ids)).all()
    contest_map = {c.contest_id: c for c in contests}

    errors = 0
    rows_written = 0

    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Entry ID', 'Contest Name', 'Contest ID', 'Entry Fee', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'])

        for entry, best_lineup, exp_pay, _ in rows:
            contest = contest_map.get(entry.contest_id)
            contest_name = contest.name if contest else ""
            entry_fee = contest.entry_fee if contest else ""

            if len(best_lineup) != 8:
                print(f"⚠ Skipping entry {entry.entry_id}: expected 8 players, got {len(best_lineup)}")
                errors += 1
                continue

            try:
                best_lineup = _apply_locked_slots(entry, best_lineup)
            except ValueError as e:
                print(f"⚠ Skipping entry {entry.entry_id}: {e}")
                errors += 1
                continue


            # Assume best_lineup is already in slot order [PG, SG, SF, PF, C, G, F, UTIL]
            slot_players = []
            valid = True
            for pid in best_lineup:
                p = player_map.get(pid)
                if not p:
                    print(f"⚠ Skipping entry {entry.entry_id}: player id {pid} not found in DB")
                    valid = False
                    break
                slot_players.append(f"{p.name} ({p.id})")

            if not valid or len(slot_players) != 8:
                errors += 1
                continue

            w.writerow([
                entry.entry_id,
                contest_name,
                entry.contest_id,
                entry_fee,
                *slot_players
            ])
            rows_written += 1

    print(f"✓ Wrote {rows_written} lineups to {path}")
    if errors > 0:
        print(f"⚠ {errors} lineups could not be written due to errors")


def main():
    parser = argparse.ArgumentParser(description="NBA Late Swap Optimization")
    parser.add_argument("--entries-csv", default=str(DEFAULT_ENTRIES_CSV), help="Path to DK entries export CSV")
    parser.add_argument("--num-sims", type=int, default=1000, help="Number of Monte Carlo simulations")
    parser.add_argument("--num-candidates", type=int, default=20, help="Number of candidates per entry")
    parser.add_argument("--field-samples", type=int, default=5000, help="Max field lineups sampled from DK")
    parser.add_argument("--player-min-fpts", type=float, default=0.0, help="Minimum fpts_mean threshold for player pool")
    parser.add_argument("--out-best", default=str(LATESWAP_BEST_CSV), help="Output CSV for best lineups")
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
    )

    # Extract id_to_fp by contest (will be populated by slate function)
    _write_best(session, Path(args.out_best), best_rows)
    print(f"Best lineups written -> {args.out_best}")


if __name__ == "__main__":
    main()
