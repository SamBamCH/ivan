#!/usr/bin/env python3
"""
NFL Lineup Generation Monitor - Terminal UI
Displays real-time status of lineup generation and simulation pipeline
"""

import time
import json
import os
from datetime import datetime, timezone
from dateutil import parser as dateutil_parser
import pytz
from pathlib import Path
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import Session

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box
from rich.align import Align

from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import (
    NFLPlayer,
    Lineup,
    LineupSimulationResult,
    HistoricalContest,
    NFLGameOdds
)


class NFLMonitor:
    """Real-time monitor for NFL lineup generation pipeline"""

    def __init__(self, config_path: str = "config/nfl.json"):
        self.engine = init_db("nfl")
        self.session = get_session(self.engine)
        self.console = Console()
        self.config_path = config_path
        self.cst_tz = pytz.timezone('America/Chicago')

        # Load config to get paths
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def to_cst(self, dt):
        """Convert a datetime to CST, handling both naive and aware datetimes"""
        if dt is None:
            return None
        # If naive, assume it's UTC
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        return dt.astimezone(self.cst_tz)

    def get_pipeline_status(self):
        """Determine current pipeline status"""
        with self.engine.connect() as conn:
            player_count = conn.execute(text('SELECT COUNT(*) FROM nfl_player')).scalar()
            lineup_count = conn.execute(text('SELECT COUNT(*) FROM lineup')).scalar()
            sim_count = conn.execute(text('SELECT COUNT(DISTINCT lineup_id) FROM lineup_simulation_results')).scalar()
            total_sim_results = conn.execute(text('SELECT COUNT(*) FROM lineup_simulation_results')).scalar()

            if player_count == 0:
                status = "⚪ Empty - No players loaded"
                phase = "INIT"
            elif lineup_count == 0:
                status = "🔵 Ready - Players loaded, no lineups generated"
                phase = "READY"
            elif sim_count == 0:
                status = "🟡 Lineups Generated - Waiting for simulation"
                phase = "PENDING_SIM"
            elif sim_count < lineup_count:
                status = f"🟠 Simulating - {sim_count}/{lineup_count} lineups simulated"
                phase = "SIMULATING"
            else:
                status = "🟢 Complete - All lineups simulated"
                phase = "COMPLETE"

        return {
            "status": status,
            "phase": phase,
            "player_count": player_count,
            "lineup_count": lineup_count,
            "sim_count": sim_count,
            "total_sim_results": total_sim_results
        }

    def get_game_matchups(self):
        """Get list of unique game matchups"""
        try:
            # Get unique team-opponent-gametime combinations
            games = self.session.query(
                NFLPlayer.team,
                NFLPlayer.opponent,
                NFLPlayer.gametime
            ).distinct().order_by(NFLPlayer.gametime).all()

            # Deduplicate by creating game pairs (sort teams alphabetically to avoid duplicates)
            seen_games = set()
            unique_games = []

            for team, opponent, gametime in games:
                if opponent:
                    game_key = tuple(sorted([team, opponent]))
                    if game_key not in seen_games:
                        seen_games.add(game_key)
                        unique_games.append({
                            "team1": game_key[0],
                            "team2": game_key[1],
                            "gametime": gametime
                        })

            return unique_games
        except Exception as e:
            return []

    def get_slate_info(self):
        """Get current slate information from players"""
        try:
            # Get earliest game time from players
            earliest_game = self.session.query(func.min(NFLPlayer.gametime)).scalar()

            if earliest_game is None:
                return None

            # Get game matchups
            matchups = self.get_game_matchups()

            # Get team count
            team_count = self.session.query(func.count(func.distinct(NFLPlayer.team))).scalar()

            # Check if it's locked (game time in past)
            now = datetime.now(timezone.utc).replace(tzinfo=None)  # Make naive for comparison
            is_locked = earliest_game < now if earliest_game else False
            time_until = earliest_game - now if earliest_game and not is_locked else None

            return {
                "earliest_game": earliest_game,
                "game_count": len(matchups),
                "matchups": matchups,
                "team_count": team_count,
                "is_locked": is_locked,
                "time_until": time_until
            }
        except Exception as e:
            return None

    def get_operation_timestamps(self):
        """Get timestamps for recent operations"""
        try:
            timestamps = {}

            # 1. Data ingest - when players were loaded from DraftKings
            # We'll track this separately in the future, for now use player data timestamp
            # Check if there's a created_at or similar, otherwise use current approach
            player_count = self.session.query(func.count(NFLPlayer.id)).scalar()
            if player_count > 0:
                # Get any player to check when data was last loaded
                sample_player = self.session.query(NFLPlayer).first()
                # For now, we'll rely on when you tell us the data was ingested
                # This is a placeholder - ideally track separately
                timestamps["ingest"] = None  # Will be set by separate tracking later
            else:
                timestamps["ingest"] = None

            # 2. Last odds pull (from NFLGameOdds last_update field, which is a string)
            last_odds_str = self.session.query(func.max(NFLGameOdds.last_update)).scalar()
            if last_odds_str:
                try:
                    # Parse the ISO string to datetime
                    timestamps["odds_pull"] = dateutil_parser.isoparse(last_odds_str)
                except:
                    timestamps["odds_pull"] = None
            else:
                timestamps["odds_pull"] = None

            # 3. Last lineup generation
            last_lineup = self.session.query(func.max(Lineup.created_at)).scalar()
            timestamps["lineup_gen"] = last_lineup

            # 4. Entries CSV - when the file was downloaded
            entries_path = self.config.get("paths", {}).get("entries_csv")
            if entries_path and os.path.exists(entries_path):
                mtime = os.path.getmtime(entries_path)
                # fromtimestamp returns local time, so we need to localize it properly
                dt = datetime.fromtimestamp(mtime)
                # Assume local timezone, convert to aware datetime
                local_tz = pytz.timezone('America/Chicago')  # Assuming you're in CST
                timestamps["entries_csv"] = local_tz.localize(dt)
            else:
                timestamps["entries_csv"] = None

            # 5. Last simulation
            last_sim = self.session.query(func.max(LineupSimulationResult.created_at)).scalar()
            timestamps["simulation"] = last_sim

            return timestamps
        except Exception as e:
            self.console.print(f"[red]Error getting timestamps: {e}[/red]")
            return {}

    def is_showdown_format(self):
        """Check if the current slate is showdown format"""
        try:
            # Check if any player has roster_slot_id 511 (CPT) or 512 (FLEX)
            showdown_count = self.session.query(func.count(NFLPlayer.id)).filter(
                NFLPlayer.roster_slot_id.in_([511, 512])
            ).scalar()
            return showdown_count > 0
        except:
            return False

    def get_exposure_summary(self, limit=30):
        """Get top player exposures from lineups, split by CPT/FLEX if showdown"""
        try:
            # Get all lineups
            lineups = self.session.query(Lineup.player_ids_list).all()

            if not lineups:
                return {"is_showdown": False, "exposures": [], "cpt_exposures": [], "flex_exposures": []}

            # Count player occurrences
            from collections import Counter

            player_counts = Counter()
            total_lineups = len(lineups)

            for lineup in lineups:
                if isinstance(lineup.player_ids_list, list):
                    player_ids = lineup.player_ids_list
                elif isinstance(lineup.player_ids_list, str):
                    player_ids = json.loads(lineup.player_ids_list)
                else:
                    continue

                for pid in player_ids:
                    player_counts[pid] += 1

            # Check if showdown format
            is_showdown = self.is_showdown_format()

            if is_showdown:
                # Split into CPT and FLEX
                cpt_exposures = []
                flex_exposures = []

                top_players = player_counts.most_common(limit * 2)  # Get more to ensure we have enough for both

                for pid, count in top_players:
                    player = self.session.query(NFLPlayer).filter(NFLPlayer.id == pid).first()
                    if player:
                        exp_data = {
                            "name": player.name,
                            "team": player.team,
                            "position": player.position,
                            "count": count,
                            "exposure": count / total_lineups,
                            "ownership": player.ownership_proj/100 if player.ownership_proj else 0.0
                        }

                        # Check if this player has CPT slot (roster_slot_id 511)
                        if player.roster_slot_id == 511:
                            cpt_exposures.append(exp_data)
                        else:
                            flex_exposures.append(exp_data)

                return {
                    "is_showdown": True,
                    "exposures": [],
                    "cpt_exposures": cpt_exposures[:limit],
                    "flex_exposures": flex_exposures[:limit]
                }
            else:
                # Regular format - single list
                exposures = []
                top_players = player_counts.most_common(limit)

                for pid, count in top_players:
                    player = self.session.query(NFLPlayer).filter(NFLPlayer.id == pid).first()
                    if player:
                        exposures.append({
                            "name": player.name,
                            "team": player.team,
                            "position": player.position,
                            "count": count,
                            "exposure": count / total_lineups,
                            "ownership": player.ownership_proj/100 if player.ownership_proj else 0.0
                        })

                return {
                    "is_showdown": False,
                    "exposures": exposures,
                    "cpt_exposures": [],
                    "flex_exposures": []
                }
        except Exception as e:
            self.console.print(f"[red]Error getting exposures: {e}[/red]")
            return {"is_showdown": False, "exposures": [], "cpt_exposures": [], "flex_exposures": []}

    def get_projection_distribution(self):
        """Get lineup projection distribution for graphing"""
        try:
            lineups = self.session.query(Lineup.player_ids_list).all()

            if not lineups:
                return []

            projection_sums = []

            for lineup in lineups:
                if isinstance(lineup.player_ids_list, list):
                    player_ids = lineup.player_ids_list
                elif isinstance(lineup.player_ids_list, str):
                    player_ids = json.loads(lineup.player_ids_list)
                else:
                    continue

                # Get players and sum their projections
                players = self.session.query(NFLPlayer).filter(NFLPlayer.id.in_(player_ids)).all()
                total_proj = sum(p.fpts_mean for p in players if p.fpts_mean is not None)

                if total_proj > 0:
                    projection_sums.append(total_proj)

            # Normalize to 0-1 if we have data
            if projection_sums:
                min_proj = min(projection_sums)
                max_proj = max(projection_sums)

                if max_proj > min_proj:
                    normalized = [(p - min_proj) / (max_proj - min_proj) for p in projection_sums]
                    return {
                        "raw": projection_sums,
                        "normalized": normalized,
                        "min": min_proj,
                        "max": max_proj,
                        "mean": sum(projection_sums) / len(projection_sums)
                    }

            return None
        except Exception as e:
            self.console.print(f"[red]Error getting projection distribution: {e}[/red]")
            return None

    def create_projection_graph(self, proj_data, width=70, height=8):
        """Create ASCII histogram of projection distribution"""
        if proj_data is None or not proj_data.get("normalized"):
            return "[yellow]No projection data available[/yellow]"

        normalized = proj_data["normalized"]

        # Create histogram bins
        num_bins = min(width, 60)
        bins = [0] * num_bins

        for val in normalized:
            bin_idx = min(int(val * num_bins), num_bins - 1)
            bins[bin_idx] += 1

        # Normalize bin heights to fit in display
        max_bin = max(bins) if bins else 1

        # Build graph with Y-axis labels
        lines = []

        # Add title line
        lines.append(f"Lineups: {len(normalized):,} | Min: {proj_data['min']:.1f} | Mean: {proj_data['mean']:.1f} | Max: {proj_data['max']:.1f}")
        lines.append("")

        for row in range(height, 0, -1):
            # Y-axis label (count)
            y_val = int((row / height) * max_bin)
            y_label = f"{y_val:>4} │ "

            line = y_label
            threshold = (row / height) * max_bin
            for bin_val in bins:
                if bin_val >= threshold:
                    line += "█"
                else:
                    line += " "
            lines.append(line)

        # X-axis
        lines.append("     └" + "─" * num_bins)

        # Calculate raw projection values for labels
        min_proj = proj_data['min']
        max_proj = proj_data['max']
        q1 = min_proj + (max_proj - min_proj) * 0.25
        q2 = min_proj + (max_proj - min_proj) * 0.50
        q3 = min_proj + (max_proj - min_proj) * 0.75

        # Create evenly spaced label positions
        label_positions = [0, num_bins // 4, num_bins // 2, 3 * num_bins // 4, num_bins - 1]
        raw_values = [min_proj, q1, q2, q3, max_proj]

        # Build x-axis label with both normalized and raw values
        lines.append("")
        lines.append("      Projection Values (Fantasy Points):")

        # Raw projection values
        raw_label = "      "
        for i, (pos, val) in enumerate(zip(label_positions, raw_values)):
            if i == 0:
                raw_label += f"{val:.1f}"
            else:
                prev_pos = label_positions[i-1]
                prev_val_str = f"{raw_values[i-1]:.1f}"
                curr_val_str = f"{val:.1f}"
                spacing = (pos - prev_pos) - len(prev_val_str) // 2 - len(curr_val_str) // 2
                raw_label += " " * spacing + curr_val_str
        lines.append(raw_label)

        lines.append("")
        lines.append("      X-axis: Lineup Projection Sum | Y-axis: Number of Lineups")

        return "\n".join(lines)

    def create_header(self) -> Panel:
        """Create header panel with title and timestamp"""
        text = Text()
        text.append("NFL Lineup Monitor", style="bold cyan")
        # Display in CST
        now_cst = datetime.now(self.cst_tz)
        text.append(f"\n{now_cst.strftime('%Y-%m-%d %I:%M:%S %p %Z')}", style="dim")
        return Panel(text, box=box.DOUBLE, style="cyan")

    def create_status_panel(self, data: dict) -> Panel:
        """Create status panel showing pipeline state"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Status", data["status"])
        table.add_row("Phase", data["phase"])
        table.add_row("Players Loaded", f"{data['player_count']:,}")
        table.add_row("Lineups Generated", f"{data['lineup_count']:,}")
        table.add_row("Lineups Simulated", f"{data['sim_count']:,} / {data['lineup_count']:,}")

        if data['lineup_count'] > 0:
            pct = (data['sim_count'] / data['lineup_count']) * 100
            table.add_row("Simulation Progress", f"{pct:.1f}%")

        table.add_row("Total Sim Results", f"{data['total_sim_results']:,}")

        return Panel(table, title="[bold]Pipeline Status[/bold]", border_style="green", box=box.ROUNDED)

    def create_slate_panel(self, slate_info: dict | None) -> Panel:
        """Create panel showing slate information"""
        if slate_info is None:
            return Panel(
                "[yellow]No slate information available[/yellow]",
                title="[bold]Slate Information[/bold]",
                border_style="yellow",
                box=box.ROUNDED
            )

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Games", str(slate_info["game_count"]))
        table.add_row("Teams", str(slate_info["team_count"]))

        if slate_info["earliest_game"]:
            # Convert to CST
            earliest_cst = self.to_cst(slate_info["earliest_game"])
            table.add_row("First Lock", earliest_cst.strftime("%Y-%m-%d %I:%M %p %Z"))

        if slate_info["is_locked"]:
            table.add_row("Status", "🔒 LOCKED")
        elif slate_info["time_until"]:
            hours = slate_info["time_until"].total_seconds() / 3600
            if hours < 1:
                mins = slate_info["time_until"].total_seconds() / 60
                table.add_row("Time Until Lock", f"⏰ {mins:.0f} minutes")
            else:
                table.add_row("Time Until Lock", f"⏰ {hours:.1f} hours")

        # Show matchups
        if slate_info.get("matchups"):
            table.add_row("", "")  # Spacer
            table.add_row("[bold]Matchups[/bold]", "")
            for matchup in slate_info["matchups"][:5]:  # Show first 5 games
                game_time_cst = self.to_cst(matchup["gametime"])
                game_str = f"{matchup['team1']} vs {matchup['team2']}"
                time_str = game_time_cst.strftime("%I:%M %p")
                table.add_row(game_str, time_str)
            if len(slate_info["matchups"]) > 5:
                table.add_row("...", f"({len(slate_info['matchups']) - 5} more)")

        return Panel(table, title="[bold]Slate Information[/bold]", border_style="blue", box=box.ROUNDED)

    def create_cpt_exposure_table(self, exposures: list) -> Table:
        """Create table for CPT player exposures (compact for side-by-side display)"""
        table = Table(box=None, show_edge=False, padding=(0, 1), title="CAPTAIN", title_style="bold cyan")
        table.add_column("Player", style="white", no_wrap=True, width=20)
        table.add_column("Pos", style="cyan", width=3)
        table.add_column("Exp", style="green", justify="right", width=6)
        table.add_column("Own", style="yellow", justify="right", width=6)

        # Show up to 25 rows to match FLEX
        for exp in exposures[:25]:
            table.add_row(
                exp["name"][:19],
                exp["position"],
                f"{exp['exposure']:.1%}",
                f"{exp['ownership']:.1%}"
            )

        return table

    def create_flex_exposure_table(self, exposures: list) -> Table:
        """Create table for FLEX player exposures (compact for side-by-side display)"""
        table = Table(box=None, show_edge=False, padding=(0, 1), title="FLEX", title_style="bold cyan")
        table.add_column("Player", style="white", no_wrap=True, width=20)
        table.add_column("Pos", style="cyan", width=3)
        table.add_column("Exp", style="green", justify="right", width=6)
        table.add_column("Own", style="yellow", justify="right", width=6)

        # Show up to 25 rows
        for exp in exposures[:25]:
            table.add_row(
                exp["name"][:19],
                exp["position"],
                f"{exp['exposure']:.1%}",
                f"{exp['ownership']:.1%}"
            )

        return table

    def create_exposure_panel(self, exposure_data: dict) -> Panel:
        """Create panel showing top player exposures, split by CPT/FLEX if showdown"""
        is_showdown = exposure_data.get("is_showdown", False)

        if is_showdown:
            # Create two separate tables for CPT and FLEX
            cpt_exposures = exposure_data.get("cpt_exposures", [])
            flex_exposures = exposure_data.get("flex_exposures", [])

            if not cpt_exposures and not flex_exposures:
                return Panel(
                    "[yellow]No exposure data available[/yellow]",
                    title="[bold]Top Player Exposures[/bold]",
                    border_style="magenta",
                    box=box.ROUNDED
                )

            # Return two separate panels
            return {
                "is_showdown": True,
                "cpt_exposures": cpt_exposures,
                "flex_exposures": flex_exposures
            }
        else:
            # Regular format - single table
            exposures = exposure_data.get("exposures", [])

            if not exposures:
                return Panel(
                    "[yellow]No exposure data available[/yellow]",
                    title="[bold]Top Player Exposures[/bold]",
                    border_style="magenta",
                    box=box.ROUNDED
                )

            table = Table(box=None, show_edge=False, padding=(0, 1))
            table.add_column("Player", style="white", no_wrap=True, width=25)
            table.add_column("Team", style="cyan", width=4)
            table.add_column("Pos", style="cyan", width=3)
            table.add_column("Exposure", style="green", justify="right", width=8)
            table.add_column("Own Proj", style="yellow", justify="right", width=8)
            table.add_column("Count", style="dim", justify="right", width=7)

            for exp in exposures:
                table.add_row(
                    exp["name"][:24],
                    exp["team"],
                    exp["position"],
                    f"{exp['exposure']:.1%}",
                    f"{exp['ownership']:.1%}",
                    f"{exp['count']:,}"
                )

            return Panel(table, title="[bold]Top Player Exposures[/bold]", border_style="magenta", box=box.ROUNDED)

    def create_operations_panel(self, timestamps: dict) -> Panel:
        """Create panel showing recent operation timestamps"""
        if not timestamps:
            return Panel(
                "[yellow]No timestamp data available[/yellow]",
                title="[bold]Recent Operations[/bold]",
                border_style="yellow",
                box=box.ROUNDED
            )

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Operation", style="cyan")
        table.add_column("Time (CST)", style="white")
        table.add_column("Age", style="dim")

        now = datetime.now()

        def format_time_ago(dt):
            if dt is None:
                return "Never", ""

            # Convert to CST for display
            dt_cst = self.to_cst(dt)

            # Calculate age based on UTC
            if dt.tzinfo is None:
                dt_utc = pytz.utc.localize(dt)
            else:
                dt_utc = dt.astimezone(pytz.utc)

            now_utc = pytz.utc.localize(now) if now.tzinfo is None else now.astimezone(pytz.utc)
            delta = now_utc - dt_utc

            if delta.total_seconds() < 0:
                # Future time (shouldn't happen, but handle it)
                return dt_cst.strftime("%I:%M:%S %p"), "future"
            elif delta.total_seconds() < 60:
                return dt_cst.strftime("%I:%M:%S %p"), "just now"
            elif delta.total_seconds() < 3600:
                mins = int(delta.total_seconds() / 60)
                return dt_cst.strftime("%I:%M:%S %p"), f"{mins}m ago"
            elif delta.total_seconds() < 86400:
                hours = int(delta.total_seconds() / 3600)
                return dt_cst.strftime("%I:%M:%S %p"), f"{hours}h ago"
            else:
                days = int(delta.total_seconds() / 86400)
                return dt_cst.strftime("%m-%d %I:%M %p"), f"{days}d ago"

        # Add rows for each operation (in chronological order)
        operations = [
            ("Data Ingest", timestamps.get("ingest")),
            ("Odds Pull", timestamps.get("odds_pull")),
            ("Lineup Gen", timestamps.get("lineup_gen")),
            ("Entries CSV", timestamps.get("entries_csv")),
            ("Simulation", timestamps.get("simulation")),
        ]

        for op_name, dt in operations:
            time_str, age_str = format_time_ago(dt)
            table.add_row(op_name, time_str, age_str)

        return Panel(table, title="[bold]Recent Operations[/bold]", border_style="yellow", box=box.ROUNDED)

    def create_projection_panel(self, proj_data) -> Panel:
        """Create panel with projection distribution graph"""
        graph = self.create_projection_graph(proj_data, width=100, height=10)
        return Panel(
            graph,
            title="[bold]Lineup Projection Distribution[/bold]",
            border_style="yellow",
            box=box.ROUNDED
        )

    def create_layout(self) -> Layout:
        """Create the overall layout"""
        layout = Layout()

        # Main vertical split: header, body
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body")
        )

        # Body split: top (exposures), middle (histogram), bottom (operations/slate)
        layout["body"].split_column(
            Layout(name="exposures", ratio=3),  # Top large section
            Layout(name="projections", ratio=2),  # Middle histogram
            Layout(name="bottom", ratio=2)  # Bottom info
        )

        # Bottom split: operations on left, slate on right
        layout["bottom"].split_row(
            Layout(name="operations"),
            Layout(name="slate")
        )

        return layout

    def update_layout(self, layout: Layout):
        """Update all panels in the layout"""
        # Get fresh data
        slate_info = self.get_slate_info()
        exposure_data = self.get_exposure_summary(limit=30)
        timestamps = self.get_operation_timestamps()
        proj_data = self.get_projection_distribution()

        # Check if showdown format
        is_showdown = exposure_data.get("is_showdown", False)

        # Update panels
        layout["header"].update(self.create_header())

        if is_showdown:
            # Create side-by-side tables for CPT and FLEX
            from rich.columns import Columns
            cpt_table = self.create_cpt_exposure_table(exposure_data.get("cpt_exposures", []))
            flex_table = self.create_flex_exposure_table(exposure_data.get("flex_exposures", []))
            exposure_panel = Panel(
                Columns([cpt_table, flex_table], equal=True, expand=True),
                title="[bold]Player Exposures (Showdown)[/bold]",
                border_style="magenta",
                box=box.ROUNDED
            )
            layout["exposures"].update(exposure_panel)
        else:
            # For classic format, show regular exposure table
            layout["exposures"].update(self.create_exposure_panel(exposure_data))

        layout["projections"].update(self.create_projection_panel(proj_data))
        layout["operations"].update(self.create_operations_panel(timestamps))
        layout["slate"].update(self.create_slate_panel(slate_info))

    def run(self, refresh_seconds: float = 2.0):
        """Run the monitor with live updates"""
        layout = self.create_layout()

        try:
            with Live(layout, console=self.console, refresh_per_second=1, screen=True):
                while True:
                    self.update_layout(layout)
                    time.sleep(refresh_seconds)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Monitor stopped by user[/yellow]")
        finally:
            self.session.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monitor NFL lineup generation pipeline")
    parser.add_argument(
        "-r", "--refresh",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2.0)"
    )
    parser.add_argument(
        "-c", "--config",
        default="config/nfl.json",
        help="Path to config file"
    )

    args = parser.parse_args()

    monitor = NFLMonitor(config_path=args.config)
    monitor.run(refresh_seconds=args.refresh)


if __name__ == "__main__":
    main()
