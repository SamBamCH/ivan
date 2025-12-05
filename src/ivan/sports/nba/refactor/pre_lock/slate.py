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
from scipy.spatial.distance import pdist, squareform
from sqlalchemy.orm import Session

from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import (
    HistoricalContest,
    Lineup,
    LineupSimulationResult,
    NBAPlayer,
)
from ivan.sports.nba.refactor.pre_lock.contest import (
    simulate_one_contest,
    sample_field_with_projection_model,
)
from ivan.sports.nba.refactor.pre_lock.payouts import fetch_and_store_all_payouts
from ivan.sports.nba.refactor.cli import (
    plot_field_sampling_diagnostics,
    print_exposures,
    print_exposures_from_counts,
)
from ivan.core.utils import PROJECT_ROOT


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
    unique_contests = df.drop_duplicates(subset=["Contest Name"], keep="last").sort_values("Contest ID")
    
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
) -> Dict[str, List[int]]:
    """
    Choose optimal diverse lineups for each entry in the slate.
    
    Uses a greedy diversity-aware selection algorithm that balances:
    - Expected ROI / cash rate
    - Lineup diversity (measured by cosine distance in FPTS space)
    
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
    
    for contest_id, entries in entries_by_contest.items():
        contest = session.query(HistoricalContest).filter_by(contest_id=contest_id).first()
        contest_name = contest.name if contest else entries[0].contest_name if entries else f"Contest_{contest_id}"
        
        console.print(f"\n[bold]Choosing for: {contest_name} ({len(entries)} entries)[/bold]")
        
        # Get simulation results for this contest
        sims = (
            session.query(LineupSimulationResult)
            .filter(LineupSimulationResult.contest_id == contest_id)
            .all()
        )
        
        if not sims:
            console.print(f"[red]No simulation results for contest {contest_id}. Run simulation first.[/red]")
            continue
        
        console.print(f"[dim]Found {len(sims)} simulation results[/dim]")
        
        # Filter to positive ROI if large result set
        if len(sims) > 5000:
            sims = [
                res for res in sims
                if np.array(res.roi, dtype=float).mean() >= 0.0
            ]
            console.print(f"[yellow]Filtered to {len(sims)} lineups with non-negative ROI[/yellow]")
        
        if not sims:
            continue
        
        # Filter out lineups with missing players
        all_pids = {pid for res in sims for pid in res.lineup.player_ids_list}
        existing = {
            pid for (pid,) in
            session.query(NBAPlayer.id).filter(NBAPlayer.id.in_(all_pids)).all()
        }
        
        filtered = []
        for res in sims:
            missing = set(res.lineup.player_ids_list) - existing
            if not missing:
                filtered.append(res)
        sims = filtered
        
        if not sims:
            console.print(f"[red]No valid lineups after dropping missing players[/red]")
            continue
        
        # Remove duplicates
        unique = []
        seen_comps = set()
        for res in sims:
            comp = tuple(sorted(res.lineup.player_ids_list))
            if comp not in seen_comps:
                seen_comps.add(comp)
                unique.append(res)
        sims = unique
        
        if not sims:
            console.print(f"[red]No unique lineups remain[/red]")
            continue
        
        N = len(sims)
        num_to_choose = len(entries)
        
        # Build feature matrix
        raw = np.zeros((N, 2), dtype=float)
        for i, res in enumerate(sims):
            payouts = np.array(res.payout, dtype=float)
            rois = np.array(res.roi, dtype=float)
            raw[i, 0] = (payouts > 0).mean() if payouts.size else 0.0  # cash%
            raw[i, 1] = rois.mean()
        
        fpts_rows = [np.array(res.fpts, dtype=float) for res in sims]
        fpts_matrix = np.vstack(fpts_rows)

        # Normalize features
        mins, maxs = raw.min(axis=0), raw.max(axis=0)
        norm = (raw - mins) / (maxs - mins + 1e-6)

        # Blend into single score
        w_feat = np.array([weights.get("cash_pct", 0), weights.get("avg_roi_win", 1)], dtype=float)
        q = (norm * w_feat).sum(axis=1)

        # Build diversity matrix
        dist_mat = squareform(pdist(fpts_matrix, metric="cosine"))

        lam = float(weights.get("diversity_lambda", 0.5))

        # Greedy diversity-aware selection
        remaining = set(range(N))
        selected = []

        first = int(np.argmax(q))
        selected.append(first)
        remaining.remove(first)

        def lineup_score(j: int, S: list) -> float:
            if S:
                diversity_gain = np.mean(dist_mat[j, S])
            else:
                diversity_gain = 0.0
            return q[j] + lam * diversity_gain
        
        while len(selected) < num_to_choose and remaining:
            best_j = max(remaining, key=lambda j: lineup_score(j, selected))
            selected.append(best_j)
            remaining.remove(best_j)

        if len(selected) < num_to_choose:
            console.print(f'[yellow]Not enough lineups, filling with top remaining[/yellow]')
            for j in np.argsort(-q):
                if j not in selected:
                    selected.append(j)
                if len(selected) >= num_to_choose:
                    break
        
        # Map selected indices to entries
        for i, entry in enumerate(entries):
            if i < len(selected):
                res = sims[selected[i]]
                chosen_by_entry[entry.entry_id] = res.lineup.player_ids_list
                all_chosen_lineup_ids.append(res.lineup_id)
        
        # Display results for this contest
        _display_contest_choices(session, console, contest_name, entries, sims, selected, weights)
    
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


def _display_contest_choices(
    session: Session,
    console: Console,
    contest_name: str,
    entries: List[PreLockEntry],
    sims: list,
    selected_indices: list,
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
    
    # Get player info
    all_pids = set()
    for idx in selected_indices:
        if idx < len(sims):
            lineup = sims[idx].lineup
            all_pids.update(lineup.player_ids_list)
    
    players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(all_pids)).all()
    player_map = {p.id: p for p in players}
    
    for i, entry in enumerate(entries):
        if i >= len(selected_indices):
            break
        
        idx = selected_indices[i]
        res = sims[idx]
        lineup = res.lineup
        
        # Get player names
        names = []
        total_salary = 0
        for pid in lineup.player_ids_list:
            p = player_map.get(pid)
            if p:
                names.append(p.name.split()[-1])  # Last name only
                total_salary += p.salary
            else:
                names.append(f"?{pid}")
        
        avg_roi = np.array(res.roi, dtype=float).mean() * 100
        payouts = np.array(res.payout, dtype=float)
        cash_pct = (payouts > 0).mean() * 100
        
        table.add_row(
            entry.entry_id,
            str(res.lineup_id),
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
    """Write chosen lineups to DK-compatible CSV format."""
    
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
            
            # Format player IDs for DK (just the IDs)
            row = [
                entry_id,
                contest_name,
                entry.contest_id,
                entry_fee,
            ]
            row.extend(player_ids[:8] + [""] * max(0, 8 - len(player_ids)))
            writer.writerow(row)
    
    console.print(f"\n[green]✓ Wrote {len(chosen_by_entry)} lineups to {output_csv}[/green]")
