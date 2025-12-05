from ivan.sports.nba.refactor.pre_lock.lineup import generate_lineups, deduplicate_lineups
from ivan.sports.nba.refactor.pre_lock.slate import simulate_one_slate, choose_lineups_for_slate
from ivan.sports.nba.refactor.cli import print_exposures, analyze_lineups
from ivan.database.session import init_db, get_session
from sqlalchemy import text
import argparse

from rich.console import Console


def main():
    p = argparse.ArgumentParser(description="NBA pre-lock simulation flow")
    p.add_argument("--entries-csv", default="data/nba/input/entries.csv", help="Path to DK entries CSV")
    p.add_argument("--num-lineups", default=25000, type=int, help="Number of lineups to generate before simming")
    p.add_argument("--num-jobs", default=8, type=int, help="Number of parallel jobs for lineup generation")
    p.add_argument("--fresh", action="store_true", help="Clear all lineups in DB prior to run")
    p.add_argument("--min-fpts", default=None, type=float, help="Optional minimum fantasy points constraint for lineups")
    p.add_argument("--field-cap", default=10000, type=int, help="Number of lineups to cap field at when simulating")
    p.add_argument("--num-sims", default=1000, type=int, help="Number of Monte Carlo simulations to run")
    p.add_argument("--output-csv", default="data/nba/output/chosen_lineups.csv", help="Output CSV for chosen lineups")

    args = p.parse_args()
    console = Console() 
    
    # Initialize DB session for exposures
    engine = init_db('nba')
    session = get_session(engine)

    # Always clear simulation results before new run
    session.execute(text("DELETE FROM lineup_simulation_results"))
    session.commit()
    console.print("[dim]✓ Cleared lineup_simulation_results table[/dim]")
    
    # Optionally clear lineups with --fresh flag
    if args.fresh:
        session.execute(text("DELETE FROM lineup"))
        session.commit()
        console.print("[yellow]✓ Cleared lineup table (--fresh)[/yellow]")
    
    # Generate lineups
    console.print("\n[bold cyan]═══ LINEUP GENERATION ═══[/bold cyan]")
    generate_lineups(args.num_lineups, args.num_jobs, min_fpts=args.min_fpts)
    deduplicate_lineups()
    
    # Analyze lineup pool (histogram + exposures)
    analyze_lineups(session, console=console)
    print_exposures(session, lineups=None, title="Generated Lineup Pool Exposures", console=console)
    
    # Interactive loop to regenerate if needed
    while True:
        console.print("\n[bold cyan]Lineup Generation Complete[/bold cyan]")
        
        new_min_fpts = console.input("[yellow]Set min_fpts threshold (or press Enter to skip):[/yellow] ").strip()
        new_num_lineups = console.input("[yellow]How many additional lineups to generate? (or press Enter to skip):[/yellow] ").strip()
        
        if not new_min_fpts and not new_num_lineups:
            console.print("[green]✓ Proceeding with current lineups[/green]")
            break
        
        if new_min_fpts:
            try:
                args.min_fpts = float(new_min_fpts)
            except ValueError:
                console.print("[red]✗ Invalid min_fpts value[/red]")
                continue
        
        if new_num_lineups:
            try:
                args.num_lineups = int(new_num_lineups)
            except ValueError:
                console.print("[red]✗ Invalid num_lineups value[/red]")
                continue
        
        console.print("[cyan]Regenerating lineups with new parameters...[/cyan]")
        generate_lineups(args.num_lineups, args.num_jobs, min_fpts=args.min_fpts)
        deduplicate_lineups()
        
        # Analyze updated lineup pool
        analyze_lineups(session, console=console)
        print_exposures(session, lineups=None, title="Generated Lineup Pool Exposures", console=console)
    
    # Simulate
    console.print("\n[bold cyan]═══ CONTEST SIMULATION ═══[/bold cyan]")
    entries_by_contest = simulate_one_slate(
        entries_csv=args.entries_csv,
        console=console,
        field_cap=args.field_cap,
        num_sims=args.num_sims,
    )
    
    # Choose lineups
    console.print("\n[bold cyan]═══ LINEUP SELECTION ═══[/bold cyan]")
    choose_lineups_for_slate(
        entries_csv=args.entries_csv,
        entries_by_contest=entries_by_contest,
        console=console,
        output_csv=args.output_csv,
    )
    
    console.print("\n[bold green]✓ Complete![/bold green]")


if __name__ == "__main__":
    main()
