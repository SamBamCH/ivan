from datetime import datetime
import sys
from typing import Optional
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
        l = id_to_lineup[lid]
        pids = l.player_ids_list
        if isinstance(pids, str):
            import json
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
    # --- Player exposures ---
    player_counter = Counter()
    for lid in lineup_ids:
        l = id_to_lineup[lid]
        pids = l.player_ids_list
        if isinstance(pids, str):
            import json
            try:
                pids = json.loads(pids)
            except:
                pids = []
        pids = [int(x) for x in (pids or []) if x is not None]
        for pid in pids:
            player_counter[pid] += counts_by_lineup[lid]

    # Print top 20 player exposures
    print("\n" + "="*70)
    print("Top 20 Player Exposures in Sampled Field")
    print("="*70)
    if player_counter:
        player_ids = list(player_counter.keys())
        players = session.query(NBAPlayer).filter(NBAPlayer.id.in_(player_ids)).all()
        id_to_name = {p.id: p.name for p in players}
        total = len(all_lineups)
        top = player_counter.most_common(20)
        for pid, count in top:
            pct = 100.0 * count / total
            name = id_to_name.get(pid, str(pid))
            print(f"  {name:30s}  {count:6d}  ({pct:5.2f}%)")
    else:
        print("  (No player exposures found)")
    print("="*70 + "\n")
