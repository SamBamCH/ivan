"""
Slate-level orchestration for NBA Classic pre-lock simulation and lineup selection.

This module provides:
- simulate_one_slate: Orchestrate simulation across contests from entries CSV
- choose_lineups_for_slate: Select optimal diverse lineups for each entry
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import csv

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import Session

from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import (
    HistoricalContest,
    Lineup,
    LineupSimulationResult,
    NBAPlayer,
)
from ivan.sports.nba.pre_lock.contest import (
    simulate_one_contest,
    sample_field_with_projection_model,
    choose_lineups_for_contest,
)
from ivan.sports.nba.pre_lock.payouts import fetch_and_store_all_payouts
from ivan.sports.nba.cli import (
    plot_field_sampling_diagnostics,
    print_exposures,
    print_exposures_from_counts,
)
from ivan.core.utils import PROJECT_ROOT
from ivan.sports.nba.utils import _coerce_positions


# ------------------------------- DK SLOT DEFINITIONS -------------------------------
# Classic order (exactly 8): PG, SG, SF, PF, C, G, F, UTIL
POSITION_ORDER = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

# Which base positions are eligible for each slot (indexed 0-7)
# Slots 5-7 (G, F, UTIL) are more flexible
SLOT_ELIGIBILITY_MAP = {
    0: ["PG"],                          # PG
    1: ["SG"],                          # SG
    2: ["SF"],                          # SF
    3: ["PF"],                          # PF
    4: ["C"],                           # C
    5: ["PG", "SG"],                    # G (flexible)
    6: ["SF", "PF"],                    # F (flexible)
    7: ["PG", "SG", "SF", "PF", "C"],   # UTIL (most flexible)
}

def _prompt_weights_for_contest(
    contest_name: str,
    defaults: Optional[dict],
    console: Console,
) -> dict:
    """
    Prompt the user for weights for a single contest. Press Enter to accept the default.

    Returns a dict with keys: cash_pct, avg_roi_win, diversity_lambda (all floats).    If console is not interactive, returns defaults without prompting.
    """
    if defaults is None:
        defaults = {"cash_pct": 0.0, "avg_roi_win": 1.0, "diversity_lambda": 0.5}

    # If console not interactive, skip prompting and return defaults
    if not getattr(console, "is_terminal", False):
        console.print(f"[dim]Non-interactive session detected; using defaults for {contest_name}[/dim]")
        return defaults

    console.print(f"\n[bold]Provide weights for contest:[/bold] {contest_name}")
    console.print("[dim]Press Enter to accept the default value shown in brackets.[/dim]")

    def _prompt_float(prompt_label: str, key: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        default = defaults.get(key, 0.0)
        attempts = 0
        while True:
            attempts += 1
            try:
                raw = console.input(f"{prompt_label} [{default}]: ").strip()
            except Exception:
                # Fallback if console.input is not available for some reason
                raw = input(f"{prompt_label} [{default}]: ").strip()

            if raw == "":
                return float(default)
            try:
                val = float(raw)
                return val
            except ValueError:
                console.print("[red]Invalid number. Try again.[/red]")
            if attempts >= 5:
                console.print("[yellow]Too many invalid attempts, using default.[/yellow]")
                return float(default)

    cash_pct = _prompt_float("Cash % target (as fraction, e.g. 0.0 for ignore or 0.1 for 10%)", "cash_pct", 0.0, None)
    avg_roi_win = _prompt_float("Avg ROI weight (relative importance)", "avg_roi_win", 0.0, None)
    diversity_lambda = _prompt_float("Diversity lambda (0-5)", "diversity_lambda", 0.0, 5)

    return {"cash_pct": float(cash_pct), "avg_roi_win": float(avg_roi_win), "diversity_lambda": float(diversity_lambda)}



def _get_player_positions(player: NBAPlayer) -> Tuple[str, ...]:
    """Get normalized position tuple for a player."""
    pos = getattr(player, "position", None)
    pos_tuple = _coerce_positions(pos or [])
    
    if isinstance(pos_tuple, str):
        pos_tuple = (pos_tuple,)
    elif not isinstance(pos_tuple, tuple):
        pos_tuple = tuple(pos_tuple) if pos_tuple else ()
    
    return pos_tuple


def _is_eligible_for_slot(player: NBAPlayer, slot_idx: int) -> bool:
    """Check if a player can fill a specific slot."""
    slot_positions = SLOT_ELIGIBILITY_MAP[slot_idx]
    player_positions = _get_player_positions(player)
    return any(pos in slot_positions for pos in player_positions)


def assign_player_order_classic(
    player_objs: List[NBAPlayer],
    prefer_late_in_flex: bool = True,
) -> List[NBAPlayer]:
    """
    Assign players to classic NBA slots using recursive backtracking.
    
    Given 8 players that form a valid lineup (from the LP), find a valid assignment
    to the 8 slots: PG, SG, SF, PF, C, G, F, UTIL.
    
    If prefer_late_in_flex=True, sorts players by gametime ascending so that
    earlier-game players fill specific slots first, leaving flexible slots (G, F, UTIL)
    for late-game players who might need to be swapped before lock.
    
    Args:
        player_objs: List of 8 NBAPlayer objects
        prefer_late_in_flex: If True, prefer late-game players in flexible slots
        
    Returns:
        List of 8 NBAPlayer objects in slot order [PG, SG, SF, PF, C, G, F, UTIL]
    """
    if prefer_late_in_flex:
        # Sort by gametime ascending (earliest first) so they fill specific slots first
        # Players with no gametime go first (safest assumption)
        def gametime_key(p):
            gt = getattr(p, "gametime", None)
            if gt is None:
                return 0  # No gametime = earliest (fill specific slots)
            return gt.timestamp() if hasattr(gt, 'timestamp') else 0
        
        player_objs = sorted(player_objs, key=gametime_key)
    
    def backtrack(
        slot_idx: int,
        assignment: List[NBAPlayer],
        remaining: List[NBAPlayer]
    ) -> Optional[List[NBAPlayer]]:
        """Recursively try to fill slots."""
        # Base case: all slots filled
        if slot_idx == 8:
            return assignment
        
        # Try each remaining player in this slot
        for i, player in enumerate(remaining):
            if _is_eligible_for_slot(player, slot_idx):
                new_assignment = assignment + [player]
                new_remaining = remaining[:i] + remaining[i+1:]
                
                result = backtrack(slot_idx + 1, new_assignment, new_remaining)
                if result is not None:
                    return result
        
        return None
    
    result = backtrack(0, [], player_objs)
    
    if result is None:
        player_summary = ", ".join([
            f"{getattr(p, 'name', p.id)} ({'/'.join(_get_player_positions(p))})"
            for p in player_objs
        ])
        raise ValueError(
            f"Could not find valid slot assignment for players: {player_summary}. "
            f"This indicates a bug in the LP constraints."
        )
    
    return result


@dataclass
class PreLockEntry:
    """Simple entry representation for pre-lock (no lineup needed)."""
    entry_id: str
    contest_id: int
    contest_name: str


def _parse_entries_csv_prelock(csv_path: str) -> Tuple[Dict[int, List[PreLockEntry]], pd.DataFrame]:
    """
    Parse DraftKings entries CSV for pre-lock scenarios.
    
    For pre-lock, lineups are typically empty. We just need entry IDs and contest IDs.
    Uses the same logic as scripts/nba/sim.py.
    
    Returns:
        Tuple of (entries_by_contest dict, unique_contests dataframe)
    """
    df = pd.read_csv(
        csv_path,
        engine="python",
        usecols=["Entry ID", "Contest ID", "Contest Name"],
        dtype={"Entry ID": str, "Contest ID": str, "Contest Name": str},
    )
    
    # Clean up
    df["Entry ID"] = df["Entry ID"].str.strip()
    df["Contest ID"] = df["Contest ID"].str.strip()
    df["Contest Name"] = df["Contest Name"].str.strip()
    df = df.dropna(subset=["Entry ID", "Contest ID", "Contest Name"]).copy()
    df["Contest ID"] = df["Contest ID"].astype(int)
    
    # Get unique contests
    unique_contests = df.drop_duplicates(subset=["Contest ID"], keep="last").sort_values("Contest ID")
    
    # Group entries by contest
    entries_by_contest: Dict[int, List[PreLockEntry]] = {}
    
    for _, row in df.iterrows():
        entry = PreLockEntry(
            entry_id=row["Entry ID"],
            contest_id=int(row["Contest ID"]),
            contest_name=row["Contest Name"],
        )
        entries_by_contest.setdefault(entry.contest_id, []).append(entry)
    
    return entries_by_contest, unique_contests


def simulate_one_slate(
    entries_csv: str = "data/nba/input/entries.csv",
    console: Optional[Console] = None,
    field_cap: int = 10000,
    num_sims: int = 1000,
    config_path: Optional[str] = None,
) -> Dict[int, List[PreLockEntry]]:
    """
    Orchestrate simulation for contests from entries CSV.
    
    This function:
    1. Parses entries from CSV to identify which contests to simulate
    2. Samples field lineups using the projection model
    3. Runs Monte Carlo simulation for each contest
    4. Stores results in LineupSimulationResult table
    
    Args:
        entries_csv: Path to DraftKings entries CSV
        console: Rich console for output (optional)
        field_cap: Maximum field size to cap sampling at
        num_sims: Number of Monte Carlo simulations per contest
        config_path: Path to config JSON (defaults to config/nba.json)
        
    Returns:
        Dict of contest_id -> List[PreLockEntry] for use in choose step
    """
    if console is None:
        console = Console()
    
    if config_path is None:
        config_path = str(Path(PROJECT_ROOT) / "config" / "nba.json")
    
    engine = init_db('nba')
    session = get_session(engine)

    fetch_and_store_all_payouts()
    
    # Load config
    try:
        cfg = json.load(open(config_path, encoding="utf-8"))
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        return {}
    
    # Parse entries from CSV
    console.print(f"[cyan]Parsing entries from: {entries_csv}[/cyan]")
    try:
        entries_by_contest, unique_contests = _parse_entries_csv_prelock(entries_csv)
    except Exception as e:
        console.print(f"[red]Failed to parse entries CSV: {e}[/red]")
        return {}
    
    total_entries = sum(len(v) for v in entries_by_contest.values())
    console.print(f"[green]✓ Parsed {total_entries} entries across {len(entries_by_contest)} contests[/green]")
    
    # Check we have lineups
    lineup_count = session.query(Lineup).count()
    if lineup_count == 0:
        console.print("[red]No lineups in database. Run lineup generation first.[/red]")
        return entries_by_contest
    
    console.print(f"[cyan]Simulating with {lineup_count} lineups in pool[/cyan]")
    
    # Loop through unique contests from entries CSV
    for _, row in unique_contests.iterrows():
        contest_id = int(row["Contest ID"])
        contest_name = row["Contest Name"]
        entries = entries_by_contest.get(contest_id, [])
        
        # Get contest info from DB
        contest = session.query(HistoricalContest).filter_by(contest_id=contest_id).first()
        
        if not contest:
            console.print(f"[yellow]Contest {contest_id} ({contest_name}) not found in DB, skipping...[/yellow]")
            continue
        
        console.print(f"\n[bold]Simulating: {contest_name}[/bold]")
        console.print(f"  Entry Fee: ${contest.entry_fee}, Field Size: {contest.total_entries}")
        console.print(f"  Your Entries: {len(entries)}")
        
        try:
            # Sample field for diagnostics visualization
            contest_params = {
                "entry_fee": float(contest.entry_fee),
                "total_entries": int(contest.total_entries),
                "multi_entry": int(contest.multi_entry or 150),
            }
            
            counts_by_lineup = sample_field_with_projection_model(
                session,
                field_size=min(field_cap, int(contest.total_entries)),
                cfg=cfg,
                contest_params=contest_params
            )
            
            # Plot diagnostics (includes field exposures)
            plot_field_sampling_diagnostics(session, counts_by_lineup, contest_params)
            
            # Run full simulation
            simulate_one_contest(
                session=session,
                contest_id=contest_id,
                contest_name=contest_name,
                config_path=config_path,
                field_cap=field_cap,
                num_sims=num_sims,
            )
            
            console.print(f"[green]✓ Completed simulation for {contest_name}[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to simulate {contest_name}: {e}[/red]")
            import traceback
            traceback.print_exc()
            continue
    
    return entries_by_contest


def choose_lineups_for_slate(
    entries_csv: str = "data/nba/input/entries.csv",
    entries_by_contest: Optional[Dict[int, List[PreLockEntry]]] = None,
    console: Optional[Console] = None,
    weights: Optional[dict] = None,
    output_csv: str = "data/nba/output/chosen_lineups.csv",
    prompt_per_contest: bool = True,    
) -> Dict[str, List[int]]:
    """
    Choose optimal diverse lineups for each entry in the slate.
    
    Orchestrates lineup selection across all contests by calling
    choose_lineups_for_contest for each contest in the slate.
    
    Args:
        entries_csv: Path to DraftKings entries CSV (used if entries_by_contest not provided)
        entries_by_contest: Pre-parsed entries dict (optional, from simulate_one_slate)
        console: Rich console for output (optional)
        weights: Dict with keys: cash_pct, avg_roi_win, diversity_lambda
        output_csv: Path to write chosen lineups CSV
        
    Returns:
        Dict of entry_id -> chosen lineup player IDs
    """
    if console is None:
        console = Console()
    
    if weights is None:
        weights = {
            "cash_pct": 0.0,
            "avg_roi_win": 1.0,
            "diversity_lambda": 0.5,
        }
    
    engine = init_db('nba')
    session = get_session(engine)
    
    # Parse entries if not provided
    if entries_by_contest is None:
        console.print(f"[cyan]Parsing entries from: {entries_csv}[/cyan]")
        try:
            entries_by_contest, _ = _parse_entries_csv_prelock(entries_csv)
        except Exception as e:
            console.print(f"[red]Failed to parse entries CSV: {e}[/red]")
            return {}
    
    total_entries = sum(len(v) for v in entries_by_contest.values())
    console.print(f"[cyan]Choosing lineups for {total_entries} entries across {len(entries_by_contest)} contests[/cyan]")
    
    chosen_by_entry: Dict[str, List[int]] = {}
    all_chosen_lineup_ids: List[int] = []
    
    # Loop through each contest
    for contest_id, entries in entries_by_contest.items():
        contest = session.query(HistoricalContest).filter_by(contest_id=contest_id).first()
        contest_name = contest.name if contest else entries[0].contest_name if entries else f"Contest_{contest_id}"
        num_to_choose = len(entries)
        
        console.print(f"\n[bold]Choosing for: {contest_name} ({num_to_choose} entries)[/bold]")

        # Determine per-contest weights: prompt interactively if allowed, otherwise use provided weights
        if prompt_per_contest and console is not None and getattr(console, "is_terminal", False):
            local_weights = _prompt_weights_for_contest(contest_name, weights, console)
        else:
            if prompt_per_contest:
                console.print(f"[dim]Non-interactive; using provided/default weights for {contest_name}[/dim]")
            local_weights = weights or {
                "cash_pct": 0.0,
                "avg_roi_win": 1.0,
                "diversity_lambda": 0.5,
            }

        # Call contest-level diversity choosing algorithm with the per-contest weights
        chosen_ids = choose_lineups_for_contest(
            session=session,
            contest_id=contest_id,
            num_to_choose=num_to_choose,
            weights=local_weights,
            console=console,
        )
        
        if not chosen_ids:
            console.print(f"[red]No lineups selected for contest {contest_id}[/red]")
            continue
        
        # Fetch the lineup objects for selected IDs
        lineup_map = {
            l.id: l for l in 
            session.query(Lineup).filter(Lineup.id.in_(chosen_ids)).all()
        }
        
        # Map selected lineup IDs to entries
        for i, entry in enumerate(entries):
            if i < len(chosen_ids):
                lineup = lineup_map.get(chosen_ids[i])
                if lineup:
                    chosen_by_entry[entry.entry_id] = lineup.player_ids_list
                    all_chosen_lineup_ids.append(chosen_ids[i])
        
        # Display results for this contest
        _display_contest_choices_from_ids(
            session, console, contest_name, entries, chosen_ids, weights
        )
    
    # Print final slate exposures
    if all_chosen_lineup_ids:
        console.print("\n")
        print_exposures(
            session,
            lineups=all_chosen_lineup_ids,
            title="Final Slate Exposures (All Chosen Lineups)",
            console=console,
            top_n=30,
        )
    
    # Write output CSV
    _write_chosen_lineups_csv(session, chosen_by_entry, entries_by_contest, output_csv, console)
    
    return chosen_by_entry


def _display_contest_choices_from_ids(
    session: Session,
    console: Console,
    contest_name: str,
    entries: List[PreLockEntry],
    chosen_lineup_ids: List[int],
    weights: dict,
) -> None:
    """Display chosen lineups for a contest in a formatted table."""
    
    table = Table(title=f"Chosen Lineups: {contest_name}", show_header=True, header_style="bold cyan")
    table.add_column("Entry ID", style="cyan", width=12)
    table.add_column("Lineup ID", style="dim")
    table.add_column("Players", style="white")
    table.add_column("Avg ROI", style="green")
    table.add_column("Cash %", style="yellow")
    table.add_column("Salary", style="magenta")
    
    # Fetch lineups and simulation results for chosen IDs
    lineups = session.query(Lineup).filter(Lineup.id.in_(chosen_lineup_ids)).all()
    lineup_map = {l.id: l for l in lineups}
    
    contest_id = entries[0].contest_id if entries else None
    sim_results = {}
    if contest_id:
        sims = (
            session.query(LineupSimulationResult)
            .filter(
                LineupSimulationResult.lineup_id.in_(chosen_lineup_ids),
                LineupSimulationResult.contest_id == contest_id
            )
            .all()
        )
        sim_results = {s.lineup_id: s for s in sims}
    
    # Get player info
    all_pids = set()
    for lid in chosen_lineup_ids:
        lineup = lineup_map.get(lid)
        if lineup:
            all_pids.update(lineup.player_ids_list)
    
    players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(all_pids)).all()
    player_map = {p.id: p for p in players}
    
    for i, entry in enumerate(entries):
        if i >= len(chosen_lineup_ids):
            break
        
        lineup_id = chosen_lineup_ids[i]
        lineup = lineup_map.get(lineup_id)
        sim_res = sim_results.get(lineup_id)
        
        if not lineup:
            continue
        
        # Get player names
        names = []
        total_salary = 0
        for pid in lineup.player_ids_list:
            p = player_map.get(pid)
            if p:
                names.append(p.name.split()[-1])  # Last name only
                total_salary += p.salary or 0
            else:
                names.append(f"?{pid}")
        
        # Get ROI/cash from simulation results
        if sim_res:
            avg_roi = np.array(sim_res.roi, dtype=float).mean() * 100
            payouts = np.array(sim_res.payout, dtype=float)
            cash_pct = (payouts > 0).mean() * 100
        else:
            avg_roi = 0.0
            cash_pct = 0.0
        
        table.add_row(
            entry.entry_id,
            str(lineup_id),
            ", ".join(names[:4]) + "...",
            f"{avg_roi:.1f}%",
            f"{cash_pct:.1f}%",
            f"${total_salary:,}",
        )
    
    console.print(table)


def _write_chosen_lineups_csv(
    session: Session,
    chosen_by_entry: Dict[str, List[int]],
    entries_by_contest: Dict[int, List[PreLockEntry]],
    output_csv: str,
    console: Console,
) -> None:
    """
    Write chosen lineups to DK-compatible CSV format.
    
    Uses recursive backtracking to assign players to valid slots,
    preferring late-game players in flexible slots (G, F, UTIL).
    """
    
    # Build entry_id -> PreLockEntry map
    entry_map: Dict[str, PreLockEntry] = {}
    for entries in entries_by_contest.values():
        for entry in entries:
            entry_map[entry.entry_id] = entry
    
    # Get player info for all chosen players
    all_pids = set()
    for pids in chosen_by_entry.values():
        all_pids.update(pids)
    
    players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(all_pids)).all()
    player_map = {p.id: p for p in players}
    
    # Get contest info
    contest_ids = set()
    for entry in entry_map.values():
        contest_ids.add(entry.contest_id)
    
    contests = session.query(HistoricalContest).filter(HistoricalContest.contest_id.in_(contest_ids)).all()
    contest_map = {c.contest_id: c for c in contests}
    
    # Write CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows_written = 0
    errors = 0
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Entry ID', 'Contest Name', 'Contest ID', 'Entry Fee', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'])
        
        for entry_id, player_ids in chosen_by_entry.items():
            entry = entry_map.get(entry_id)
            if not entry:
                continue
            
            contest = contest_map.get(entry.contest_id)
            contest_name = contest.name if contest else entry.contest_name
            entry_fee = contest.entry_fee if contest else ""
            
            # Get player objects
            player_objs = [player_map.get(pid) for pid in player_ids if player_map.get(pid)]
            
            if len(player_objs) != 8:
                console.print(f"[yellow]⚠ Skipping entry {entry_id}: expected 8 players, got {len(player_objs)}[/yellow]")
                errors += 1
                continue
            
            try:
                # Assign players to slots with backtracking
                # Prefers late-game players in flexible slots (G, F, UTIL)
                assigned = assign_player_order_classic(player_objs, prefer_late_in_flex=True)
                
                # Format as "Name (ID)" for DK upload
                slots = [f"{getattr(p, 'name', '')} ({p.id})" for p in assigned]
                
                row = [
                    entry_id,
                    contest_name,
                    entry.contest_id,
                    entry_fee,
                ]
                row.extend(slots)
                writer.writerow(row)
                rows_written += 1
                
            except ValueError as e:
                console.print(f"[red]✗ Error assigning lineup for entry {entry_id}: {e}[/red]")
                errors += 1
                continue
    
    console.print(f"\n[green]✓ Wrote {rows_written} lineups to {output_csv}[/green]")
    if errors > 0:
        console.print(f"[yellow]⚠ {errors} lineups could not be written due to errors[/yellow]")
