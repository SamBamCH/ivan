from datetime import datetime
import sys
import json
from typing import Optional, List, Dict, Union
from rich.console import Console
from rich.table import Table
from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import StartingLineups, NBAPlayer, Lineup
from sqlalchemy.orm import Session
import pickle
import plotext as plt
import numpy as np
from collections import Counter


def view_lineups(session: Session, prompt_missing: bool = False) -> list[str]:
    console = Console()
    now = datetime.utcnow()

    rows = session.query(StartingLineups).order_by(StartingLineups.team).all()
    if not rows:
        console.print("[yellow]No starting lineups found.[/yellow]")
        return []

    # collect all player names and group by team
    teams = {}
    names = []
    for r in rows:
        name = getattr(r, "player_name", None)
        team = getattr(r, "team", "") or ""
        pos = getattr(r, "position", "") or ""
        if not name:
            continue
        teams.setdefault(team, []).append((name, pos))
        names.append(name)

    players = {}
    if names:
        db_players = session.query(NBAPlayer).filter(NBAPlayer.name.in_(set(names))).all()
        players = {p.name: p for p in db_players}

    table = Table(show_header=True, header_style="bold")
    table.add_column("Team", style="cyan", no_wrap=True)
    table.add_column("Pos", style="magenta", width=6)
    table.add_column("Player — Salary — FPTS", style="white")

    missing_count = 0
    unmatched = []

    # Determine which teams have at least one matched player record
    team_has_match: dict[str, bool] = {}
    for team, roster in teams.items():
        team_has_match[team] = any(name in players for name, _ in roster)

    # Build table rows, skipping teams with zero matched players (likely off-slate)
    for team in sorted(teams.keys()):
        if not team_has_match.get(team, False):
            # Skip printing entire team if no player records exist
            continue
        for idx, (name, pos) in enumerate(teams[team], start=1):
            p = players.get(name)
            if p is None:
                table.add_row(team if idx == 1 else "", pos, f"[red]MISSING RECORD: {name}[/red]")
                missing_count += 1
                unmatched.append(name)
                continue

            salary = getattr(p, "salary", None)
            fpts = getattr(p, "fpts_mean", None)

            salary_str = f"${salary}" if salary is not None else "N/A"
            fpts_str = f"{fpts:.1f}" if fpts is not None else "MISSING"

            player_text = f"{p.name} ({getattr(p, 'position', pos)}) — {salary_str} — {fpts_str}"

            if fpts is None:
                player_text = f"[red]{player_text}[/red]"
                missing_count += 1

            # dim excluded, mark started games
            if getattr(p, "exclude", False):
                player_text = f"[dim red]{player_text}[/dim red]"
            elif getattr(p, "gametime", None) and p.gametime <= now:
                player_text = f"[yellow]{player_text}[/yellow]"

            table.add_row(team if idx == 1 else "", pos, player_text)

    console.print(table)

    if missing_count:
        console.print(f"[bold red]WARNING:[/bold red] {missing_count} starting player(s) missing projections or records.")

    # Optionally prompt user to add manual projections for unmatched players that are on teams we printed
    if prompt_missing and unmatched and sys.stdin.isatty():
        console.print("[cyan]Manual projection entry for unmatched starters (blank to skip).[/cyan]")
        for name in unmatched:
            # Only prompt if the player's team had at least one matched record (i.e., team printed)
            # Find team & position from starting lineups
            sl_entry = next((r for r in rows if r.player_name == name), None)
            if not sl_entry:
                continue
            if not team_has_match.get(sl_entry.team, False):
                # Skip players on fully missing teams (off-slate)
                continue
            try:
                proj_in = input(f"Projection (mean FPTS) for {name} ({sl_entry.team} {sl_entry.position}) [skip]: ").strip()
            except EOFError:
                break
            if not proj_in:
                continue
            try:
                fpts_mean = float(proj_in)
            except ValueError:
                console.print(f"[red]Invalid number; skipping {name}[/red]")
                continue
            # Ask for salary (needed for optimization); default to 0 if blank
            try:
                sal_in = input(f"Salary for {name} (blank=0): ").strip()
            except EOFError:
                sal_in = ""
            salary_val = 0
            if sal_in:
                try:
                    salary_val = int(sal_in)
                except ValueError:
                    console.print(f"[yellow]Invalid salary input; using 0 for {name}[/yellow]")
            # Gametime default now
            gametime_val = now
            # Insert new NBAPlayer record with minimal defaults
            new_player = NBAPlayer(
                name=name,
                team=sl_entry.team,
                opponent=None,
                position=sl_entry.position,
                roster_slot_id=0,
                gametime=gametime_val,
                salary=salary_val,
                status="MANUAL",
                fpts_mean=fpts_mean,
                fpts_std=0.0,
                ownership=0.0,
                fpts_updated_at=now,
            )
            session.add(new_player)
            try:
                session.commit()
                console.print(f"[green]Added manual projection for {name} (mean={fpts_mean:.2f})[/green]")
            except Exception as e:
                session.rollback()
                console.print(f"[red]Failed to add {name}: {e}[/red]")

    # return unique unmatched names in original encounter order
    return list(dict.fromkeys(unmatched))


def print_exposures(
    session: Session,
    lineups: Union[List[int], List[List[int]], None] = None,
    title: str = "Player Exposures",
    console: Optional[Console] = None,
    top_n: int = 20,
) -> None:
    """
    Print a formatted table of player exposures from a set of lineups.
    Shows players from NBAPlayer table, sorted by ownership high to low.
    
    Args:
        session: Database session
        lineups: Either:
            - List of lineup IDs (will fetch from Lineup table)
            - List of player ID lists [[pid1, pid2, ...], ...]
            - None to use all lineups in database
        title: Title for the exposure table
        console: Rich console for output (optional)
        top_n: Number of top players to show by ownership (default 20)
    """
    if console is None:
        console = Console()
    
    # Resolve lineups to list of player ID lists
    lineup_player_ids: List[List[int]] = []
    
    if lineups is None:
        # Fetch all lineups from database
        all_lineups = session.query(Lineup).all()
        for lu in all_lineups:
            pids = lu.player_ids_list
            if isinstance(pids, str):
                try:
                    pids = json.loads(pids)
                except:
                    pids = []
            pids = [int(x) for x in (pids or []) if x is not None]
            if pids:
                lineup_player_ids.append(pids)
    elif lineups and isinstance(lineups[0], int):
        # List of lineup IDs
        lineup_ids = lineups
        fetched = session.query(Lineup).filter(Lineup.id.in_(lineup_ids)).all()
        id_to_lineup = {lu.id: lu for lu in fetched}
        for lid in lineup_ids:
            lu = id_to_lineup.get(lid)
            if lu:
                pids = lu.player_ids_list
                if isinstance(pids, str):
                    try:
                        pids = json.loads(pids)
                    except:
                        pids = []
                pids = [int(x) for x in (pids or []) if x is not None]
                if pids:
                    lineup_player_ids.append(pids)
    else:
        # Already list of player ID lists
        lineup_player_ids = lineups
    
    total_lineups = len(lineup_player_ids) if lineup_player_ids else 0
    
    # Count exposures
    exposure_counter: Counter = Counter()
    for pids in lineup_player_ids:
        for pid in pids:
            exposure_counter[pid] += 1
    
    # Get players from NBAPlayer table, sorted by ownership descending, limited to top_n
    all_players = session.query(NBAPlayer).order_by(NBAPlayer.ownership.desc()).limit(top_n).all()
    total_players = session.query(NBAPlayer).count()
    
    if not all_players:
        console.print(f"[yellow]No players found in NBAPlayer table.[/yellow]")
        return
    
    # Build table
    table = Table(title=f"{title} ({total_lineups} lineups)", show_header=True, header_style="bold cyan")
    table.add_column("Player", style="white", no_wrap=True)
    table.add_column("Team", style="cyan", width=5)
    table.add_column("Pos", style="magenta", width=8)
    table.add_column("Salary", style="green", justify="right")
    table.add_column("FPTS", style="yellow", justify="right")
    table.add_column("Own%", style="blue", justify="right")
    table.add_column("Count", style="dim", justify="right")
    table.add_column("Exposure", justify="right")
    
    for p in all_players:
        name = p.name
        team = p.team or "?"
        pos = p.position or "?"
        salary = f"${p.salary:,}" if p.salary else "N/A"
        fpts = f"{p.fpts_mean:.1f}" if p.fpts_mean else "N/A"
        ownership = p.ownership or 0.0
        own = f"{ownership:.1f}%"
        
        count = exposure_counter.get(p.id, 0)
        exposure_pct = (count / total_lineups) * 100 if total_lineups > 0 else 0.0
        
        # Color based on difference from ownership
        diff = exposure_pct - ownership
        if diff <= -5:
            # More than 5 points below ownership -> red
            exp_str = f"[bold red]{exposure_pct:.1f}%[/bold red]"
        elif diff >= 5:
            # 5 points or more above ownership -> green
            exp_str = f"[bold green]{exposure_pct:.1f}%[/bold green]"
        else:
            # Within ±5 points -> default
            exp_str = f"{exposure_pct:.1f}%"
        
        table.add_row(
            name,
            team,
            pos,
            salary,
            fpts,
            own,
            str(count),
            exp_str,
        )
    
    console.print(table)
    
    # Summary stats
    console.print(f"\n[dim]Showing top {len(all_players)} of {total_players} players by ownership[/dim]")


def print_exposures_from_counts(
    session: Session,
    counts_by_lineup: Dict[int, int],
    title: str = "Field Exposures",
    console: Optional[Console] = None,
    top_n: int = 20,
) -> None:
    """
    Print player exposures from a field sample with counts per lineup.
    
    Preprocesses the counts into expanded lineup lists, then calls print_exposures.
    
    Args:
        session: Database session
        counts_by_lineup: Dict mapping lineup_id -> count (number of times sampled)
        title: Title for the exposure table
        console: Rich console for output
        top_n: Number of top exposed players to show
    """
    if console is None:
        console = Console()
    
    # Fetch lineups
    lineup_ids = list(counts_by_lineup.keys())
    fetched = session.query(Lineup).filter(Lineup.id.in_(lineup_ids)).all()
    id_to_lineup = {lu.id: lu for lu in fetched}
    
    # Expand counts into list of player ID lists (weighted by count)
    expanded_lineups: List[List[int]] = []
    for lid, count in counts_by_lineup.items():
        lu = id_to_lineup.get(lid)
        if not lu:
            continue
        pids = lu.player_ids_list
        if isinstance(pids, str):
            try:
                pids = json.loads(pids)
            except:
                pids = []
        pids = [int(x) for x in (pids or []) if x is not None]
        if pids:
            # Add this lineup `count` times to represent field duplication
            for _ in range(count):
                expanded_lineups.append(pids)
    
    # Update title to show unique count
    total_field_size = sum(counts_by_lineup.values())
    enhanced_title = f"{title} ({total_field_size:,} entries, {len(lineup_ids)} unique)"
    
    # Delegate to print_exposures
    print_exposures(
        session=session,
        lineups=expanded_lineups,
        title=enhanced_title,
        console=console,
        top_n=top_n,
    )


def analyze_lineups(
    session: Session,
    console: Optional[Console] = None,
    num_buckets: int = 8,
) -> None:
    """
    Analyze all lineups in the database and plot a histogram of projected FPTS sums.
    
    Args:
        session: Database session
        console: Rich console for output (optional)
        num_buckets: Number of histogram buckets (default 15)
    """
    if console is None:
        console = Console()
    
    # Fetch all lineups
    lineups = session.query(Lineup).all()
    
    if not lineups:
        console.print("[yellow]No lineups found in database.[/yellow]")
        return
    
    # Get all unique player IDs
    all_player_ids = set()
    for lineup in lineups:
        pids = lineup.player_ids_list
        if isinstance(pids, str):
            try:
                pids = json.loads(pids)
            except:
                pids = []
        pids = [int(x) for x in (pids or []) if x is not None]
        all_player_ids.update(pids)
    
    # Fetch all players
    players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(all_player_ids)).all()
    player_fpts_map = {p.id: (p.fpts_mean or 0.0) for p in players}
    
    # Calculate FPTS sum for each lineup
    lineup_fpts_sums = []
    for lineup in lineups:
        pids = lineup.player_ids_list
        if isinstance(pids, str):
            try:
                pids = json.loads(pids)
            except:
                pids = []
        pids = [int(x) for x in (pids or []) if x is not None]
        fpts_sum = sum(player_fpts_map.get(pid, 0.0) for pid in pids)
        lineup_fpts_sums.append(fpts_sum)
    
    if not lineup_fpts_sums:
        console.print("[yellow]No lineup FPTS data to analyze.[/yellow]")
        return
    
    # Calculate stats
    fpts_array = np.array(lineup_fpts_sums)
    mean_fpts = float(np.mean(fpts_array))
    median_fpts = float(np.median(fpts_array))
    std_fpts = float(np.std(fpts_array))
    min_fpts = float(np.min(fpts_array))
    max_fpts = float(np.max(fpts_array))
    
    # Print summary
    console.print(f"\n[bold cyan]═══ Lineup Pool Analysis ({len(lineups):,} lineups) ═══[/bold cyan]")
    console.print(f"  Mean:   {mean_fpts:>8.1f}")
    console.print(f"  Median: {median_fpts:>8.1f}")
    console.print(f"  Std:    {std_fpts:>8.1f}")
    console.print(f"  Min:    {min_fpts:>8.1f}")
    console.print(f"  Max:    {max_fpts:>8.1f}")
    
    # Plot histogram using plotext
    plt.clf()
    plt.theme('clear')  # Transparent background
    plt.hist(lineup_fpts_sums, bins=num_buckets)
    plt.title(f"Lineup Projected FPTS Distribution ({len(lineups):,} lineups)")
    plt.xlabel("Projected FPTS Sum")
    plt.yticks([])  # Hide y-axis tick values
    plt.plotsize(100, 20)
    plt.show()


def plot_field_sampling_diagnostics(session, counts_by_lineup, contest_params, model_path="data/nba/models/field_distribution_model.pkl"):
    """
    Plot the model's predicted projection distribution vs. the empirical distribution from the sampled field.
    Prints in terminal using plotext if available, otherwise prints text summary.
    Also prints player exposures.
    """
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler = model['scaler']
    proj_model = model['proj_model']
    bin_edges = model['bin_edges']
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    BINS = len(bin_centers)

    # Predict model distribution for this contest
    X = np.array([[contest_params['entry_fee'], contest_params['total_entries'], contest_params['multi_entry']]])
    X_scaled = scaler.transform(X)
    proj_dist = proj_model.predict(X_scaled)[0]

    # Gather all sampled lineups and their projection sums
    lineup_ids = list(counts_by_lineup.keys())
    lineup_counts = list(counts_by_lineup.values())
    all_lineups = session.query(Lineup).filter(Lineup.id.in_(lineup_ids)).all()
    id_to_lineup = {l.id: l for l in all_lineups}
    lineup_proj = []
    for lid in lineup_ids:
        l = id_to_lineup.get(lid)
        if not l:
            continue
        pids = l.player_ids_list
        if isinstance(pids, str):
            try:
                pids = json.loads(pids)
            except:
                pids = []
        pids = [int(x) for x in (pids or []) if x is not None]
        players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(pids)).all()
        player_dict = {p.id: p for p in players}
        total_proj = sum(player_dict[pid].fpts_mean for pid in pids if pid in player_dict and player_dict[pid].fpts_mean is not None)
        # Add the projection sum for each time this lineup was sampled
        lineup_proj.extend([total_proj] * counts_by_lineup[lid])

    lineup_proj = np.array(lineup_proj)
    
    # Load projection_min from model
    projection_min = model.get('projection_min', 200.0)
    
    # Filter lineups below projection_min (matching sampling logic)
    valid_mask = lineup_proj >= projection_min
    lineup_proj = lineup_proj[valid_mask]
    
    # Normalize lineup projections to [0, 1] using projection_min as min and max as max
    if len(lineup_proj) > 0:
        max_proj = lineup_proj.max()
        if max_proj > projection_min:
            norm_proj = (lineup_proj - projection_min) / (max_proj - projection_min)
        else:
            norm_proj = np.zeros_like(lineup_proj)
        
        # Empirical histogram (normalized to sum to 1 for comparison)
        hist, _ = np.histogram(norm_proj, bins=bin_edges, density=False)
        hist = hist / hist.sum() if hist.sum() > 0 else hist

        # Map bin centers to projection space for plotting on the new range
        proj_bin_centers = projection_min + bin_centers * max(1e-6, (max_proj - projection_min))

        # Print terminal plot using projection-space x-axis to "shrink" to the active range
        print("\n" + "="*70)
        print("Field Sampling Diagnostics: Model vs Sampled Distribution")
        print("="*70)
        
        plt.clf()
        plt.bar(proj_bin_centers.tolist(), proj_dist.tolist(), label='Model Prediction', color='green')
        plt.bar(proj_bin_centers.tolist(), hist.tolist(), label='Sampled Field', color='blue', fill=False)
        plt.title(f'Projection Distribution (sum): {projection_min:.0f}–{max_proj:.0f}')
        plt.xlabel('Lineup Projection Sum')
        plt.ylabel('Probability Density')
        plt.plotsize(100, 25)
        # limit x-axis to the active projection range for better detail
        try:
            plt.xlim(projection_min, max_proj)
        except Exception:
            pass
        plt.show()
    
    # Use the new modular exposure function
    print_exposures_from_counts(session, counts_by_lineup, title="Sampled Field Exposures")
