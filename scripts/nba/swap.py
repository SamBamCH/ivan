import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from rich.console import Console

from ivan.database.session import get_session, init_db
from ivan.sports.nba.swap_logic import (
    parse_entries_csv,
    lateswap_slate,
    write_lateswap_csv_simple,
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
        default="lateswap_results.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--candidates-csv",
        default="lateswap_candidates.csv",
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
        default=10,
        help="Max number of field lineups to sample when using DK field"
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
            # Treat provided time as Central Time (naive, for compatibility with DB values)
            lock_time_display = f"{lock_time.strftime('%Y-%m-%d %H:%M')} CT (as provided)"
        except ValueError:
            console.print("[red]✗ Invalid --lock-time format. Use YYYY-MM-DD HH:MM[/red]")
            return
    else:
        # Default to current Central Time for testing; keep naive for comparisons
        now_ct = datetime.now(ZoneInfo('America/Chicago'))
        lock_time = now_ct.replace(tzinfo=None)
        lock_time_display = f"{lock_time.strftime('%Y-%m-%d %H:%M')} CT (defaulted to current for testing)"
        console.print("[yellow]![/yellow] --lock-time not provided; defaulting to current Central Time (CT) for testing.")

    console.print("[bold cyan]NBA Late Swap Optimizer[/bold cyan]")
    console.print(f"[cyan]Entries CSV: {args.entries_csv}[/cyan]")
    console.print(f"[cyan]Config: {args.config}[/cyan]")
    console.print(f"[cyan]Lock Time: {lock_time_display}[/cyan]")
    console.print(f"[cyan]Simulations: {args.num_sims}[/cyan]")
    console.print(f"[cyan]Candidates per entry: {args.num_candidates}[/cyan]")
    console.print()
    
    # Initialize database
    engine = init_db('nba')
    session = get_session(engine)

    # Parse entries
    console.print("[cyan]Parsing entries CSV...[/cyan]")
    entries_by_contest = parse_entries_csv(args.entries_csv, session)
    
    if not entries_by_contest:
        console.print("[red]No entries found in CSV[/red]")
        return

    console.print("\n[bold cyan]Using DK field lineups for simulation[/bold cyan]")
    cfg = json.load(open(args.config, encoding="utf-8"))
    # Force DK field source for this run
    cfg.setdefault("classic", {})["field_source"] = "dk"

    # Flatten all entries; coordinator groups by contest internally
    all_entries = []
    for _, entries in entries_by_contest.items():
        all_entries.extend(entries)

    best_results, candidate_rows = lateswap_slate(
        session=session,
        entries=all_entries,
        lock_time=lock_time,
        config=cfg,
        num_candidates_per_entry=args.num_candidates,
        num_field_samples=args.field_samples,
        num_sims=args.num_sims,
        seed=42,
        debug=args.debug,
    )
    console.print(f"\n[cyan]Writing results to {args.output_csv} and {args.candidates_csv}...[/cyan]")
    write_lateswap_csv_simple(best_results, args.output_csv)
    write_lateswap_candidates_csv(candidate_rows, args.candidates_csv)
    console.print(f"[green]✓ Late swap (DK field) complete! Results saved to {args.output_csv} and {args.candidates_csv}[/green]")


if __name__ == "__main__":
    main()