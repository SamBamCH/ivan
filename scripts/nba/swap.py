import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from rich.console import Console

from ivan.database.session import get_session, init_db
from ivan.sports.nba.swap_logic import (
    parse_entries_csv,
    lateswap_slate,
    write_lateswap_csv_entries_format,
    write_lateswap_candidates_csv,
)

def main():
    parser = argparse.ArgumentParser(description="NBA Late Swap Optimization")
    parser.add_argument(
        "--entries-csv", 
        required=False, 
        default='/home/sam-pratt/ivan/data/nba/input/entries.csv', 
        help="DK entries CSV path"
    )
    parser.add_argument(
        "--config", 
        default="config/nba.json",
        help="Config JSON path"
    )
    parser.add_argument(
        "--output-csv", 
        default="data/nba/output/lateswap_results.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--candidates-csv",
        default="data/nba/output/lateswap_candidates.csv",
        help="Output CSV for all candidates with ROI (DK field mode)"
    )
    parser.add_argument(
        "--lock-time", 
        required=False, 
        help="Lock time (YYYY-MM-DD HH:MM). If omitted, uses current Central Time (CT) for testing."
    )
    parser.add_argument(
        "--num-sims", 
        type=int, 
        default=1000,
        help="Number of Monte Carlo simulations"
    )
    parser.add_argument(
        "--num-candidates", 
        type=int, 
        default=10,
        help="Number of candidate lineups to generate per entry"
    )
    parser.add_argument(
        "--field-samples",
        type=int,
        default=5000,
        help="Max number of field lineups to sample when using DK field"
    )
    parser.add_argument(
        "--field-workers",
        type=int,
        default=8,
        help="Parallel workers for DK field parsing/completion (0 to disable)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed diagnostic output for debugging"
    )
    args = parser.parse_args()

    console = Console()
    # Process lock time early so we can display the effective value
    if args.lock_time:
        try:
            lock_time = datetime.strptime(args.lock_time, "%Y-%m-%d %H:%M")
        except ValueError:
            console.print("[red]✗ Invalid --lock-time format. Use YYYY-MM-DD HH:MM[/red]")
            return
    else:
        # Default to current Central Time (naive) for comparisons
        lock_time = datetime.now(ZoneInfo('America/Chicago')).replace(tzinfo=None)

    quiet = not args.debug
    if not quiet:
        console.print("[bold cyan]NBA Late Swap Optimizer[/bold cyan]")
        console.print(f"[cyan]Lock Time: {lock_time.strftime('%Y-%m-%d %H:%M')} CT[/cyan]")
        console.print(f"[cyan]Simulations: {args.num_sims} | Candidates/Entry: {args.num_candidates} | Field Samples: {args.field_samples}[/cyan]")
        console.print()
    
    # Initialize database
    engine = init_db('nba')
    session = get_session(engine)

    # Parse entries (quiet mode suppresses per-row output inside helper)
    entries_by_contest = parse_entries_csv(args.entries_csv, session, quiet=quiet)
    
    if not entries_by_contest:
        console.print("[red]No entries found in CSV[/red]")
        return

    if not quiet:
        console.print("[cyan]Using DK field lineups source[/cyan]")
    cfg = json.load(open(args.config, encoding="utf-8"))
    # Force DK field source for this run
    cfg.setdefault("classic", {})["field_source"] = "dk"
    # Configure parallelism for field parsing/completion
    cfg["classic"]["field_parallel_workers"] = int(args.field_workers)

    # Flatten all entries; coordinator groups by contest internally
    all_entries = []
    for _, entries in entries_by_contest.items():
        all_entries.extend(entries)

    # Run late swap (swap_logic internally handles minimal field sampling progress)
    best_results, candidate_rows = lateswap_slate(
        session=session,
        entries=all_entries,
        lock_time=lock_time,
        config=cfg,
        num_candidates_per_entry=args.num_candidates,
        num_field_samples=args.field_samples,
        num_sims=args.num_sims,
        seed=42,
    )
    if not quiet:
        console.print(f"[cyan]Writing results to {args.output_csv} and {args.candidates_csv}[/cyan]")
    write_lateswap_csv_entries_format(best_results, args.output_csv, session=session, quiet=quiet)
    write_lateswap_candidates_csv(candidate_rows, args.candidates_csv, quiet=quiet)
    console.print(f"[green]✓ Late swap complete[/green]")


if __name__ == "__main__":
    main()