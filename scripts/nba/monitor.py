"""
Terminal lineup display script that projects starters based on FPTS projections.
Groups players by team and displays projected lineups in grid format ordered by game time.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import os
import shutil
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from ivan.sports.nba.models import NBAPlayer
from ivan.database.session import get_session, init_db

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


class PlayerStatus(Enum):
    """Player availability status indicators"""
    HEALTHY = None
    QUESTIONABLE = "Q"
    OUT = "Out"
    MAY_NOT_PLAY = "MAY NOT PLAY"
    OUT_FOR_SEASON = "OFS"
    DOUBT = "Doubt"


@dataclass
class ProjectedPlayer:
    """Data class for a player with their projection info"""
    name: str
    position: str  # 'G', 'F', 'C'
    specific_position: str  # 'PG', 'SG', 'SF', 'PF', 'C'
    fpts_mean: float
    fpts_std: float
    salary: int
    team: str
    status: Optional[str]
    
    def __repr__(self) -> str:
        status_str = f" - {self.status}" if self.status else ""
        return f"{self.specific_position}: {self.name}{status_str} (${self.salary:,} | {self.fpts_mean:.1f}±{self.fpts_std:.1f})"


@dataclass
class TeamLineup:
    """Data class for a team's projected lineup"""
    team: str
    opponent: Optional[str]
    game_time: Optional[datetime]
    starters: List[ProjectedPlayer]
    bench: List[ProjectedPlayer]
    lineup_status: str = "Expected Lineup"  # or "Confirmed Lineup"


class LineupProjector:
    """Projects starting lineups based on FPTS projections"""
    
    # Position hierarchy for starter selection
    # Maps simplified position to specific positions and how many at each
    POSITION_REQUIREMENTS = {
        'G': ['PG', 'SG'],      # 2 guards
        'F': ['SF', 'PF'],      # 2 forwards
        'C': ['C'],             # 1 center
    }
    
    # Standard NBA starter count: 2G, 2F, 1C
    STARTER_SLOTS = 5
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_players_by_team(self) -> Dict[str, List[NBAPlayer]]:
        """Fetch all active players grouped by team"""
        stmt = select(NBAPlayer).where(
            NBAPlayer.status != 'OUT'  # Exclude OUT players
        ).order_by(NBAPlayer.team, NBAPlayer.fpts_mean.desc())
        
        players = self.session.execute(stmt).scalars().all()
        
        # Group by team
        by_team = {}
        for player in players:
            if player.team not in by_team:
                by_team[player.team] = []
            by_team[player.team].append(player)
        
        return by_team
    
    def _convert_to_specific_position(self, position: str) -> str:
        """
        Convert simplified position (G, F, C) to specific positions.
        This is a helper that uses the position field from DB.
        """
        if position in ['PG', 'SG', 'SF', 'PF', 'C']:
            return position
        return position  # fallback to original
    
    def _parse_position_value(self, position_str: str) -> str:
        """
        Parse position string from DB and return simplified form.
        Handles formats like "PG", "SG/PF", etc.
        """
        if not position_str:
            return 'G'  # default to guard
        
        pos_upper = position_str.strip().upper()
        
        # Handle multi-position entries
        if '/' in pos_upper:
            positions = [p.strip() for p in pos_upper.split('/')]
        else:
            positions = [pos_upper]
        
        # Categorize
        guards = {'PG', 'SG'}
        forwards = {'SF', 'PF'}
        centers = {'C'}
        
        for pos in positions:
            if pos in guards:
                return 'G'
            elif pos in forwards:
                return 'F'
            elif pos in centers:
                return 'C'
        
        return 'G'  # default
    
    def project_team_lineup(self, team: str, players: List[NBAPlayer]) -> TeamLineup:
        """
        Project a team's starting lineup based on FPTS projections with flexible position assignment.
        
        Strategy:
        1. Select top 2 G (from PG/SG)
        2. Select top 2 F (from SF/PF)
        3. Select top 1 C (from C)
        4. Fallbacks with visual warnings (red position):
           - If no C available, use F (mark red)
           - If not enough F, use additional C (mark red)
           - If not enough total starters, fill from bench (mark red)
        """
        
        # Group by simplified position
        by_position = {'G': [], 'F': [], 'C': []}
        all_players_list = []
        
        for player in players:
            if player.fpts_mean is None:
                logger.warning(f"Player {player.name} has no FPTS projection, skipping")
                continue
            
            simplified_pos = self._parse_position_value(player.position)
            specific_pos = player.position if player.position in ['PG', 'SG', 'SF', 'PF', 'C'] else 'G'
            
            projected = ProjectedPlayer(
                name=player.name,
                position=simplified_pos,
                specific_position=specific_pos,
                fpts_mean=player.fpts_mean,
                fpts_std=player.fpts_std,
                salary=player.salary,
                team=team,
                status=player.status
            )
            by_position[simplified_pos].append(projected)
            all_players_list.append(projected)
        
        # Sort each position by FPTS (descending)
        for pos in by_position:
            by_position[pos].sort(key=lambda p: p.fpts_mean, reverse=True)
        
        # Track which players are starters
        starters = []
        used_players = set()
        
        # Step 1: Get top 2 Guards
        guards_added = 0
        for guard in by_position['G']:
            if guards_added < 2:
                starters.append(guard)
                used_players.add(id(guard))
                guards_added += 1
        
        # Step 2: Get top 2 Forwards
        forwards_added = 0
        for forward in by_position['F']:
            if forwards_added < 2 and id(forward) not in used_players:
                starters.append(forward)
                used_players.add(id(forward))
                forwards_added += 1
        
        # Step 3: Get top 1 Center (or fallback with red indicator)
        center_added = False
        
        # Try to add a center
        for center in by_position['C']:
            if not center_added and id(center) not in used_players:
                starters.append(center)
                used_players.add(id(center))
                center_added = True
                break
        
        # If no center and we have extra forwards, use one (mark red)
        if not center_added and forwards_added < len(by_position['F']):
            for forward in by_position['F']:
                if id(forward) not in used_players:
                    forward.specific_position = f"{Colors.RED}F{Colors.RESET}"
                    starters.append(forward)
                    used_players.add(id(forward))
                    center_added = True
                    break
        
        # If still no center and we have extra centers or no centers at all, use what we have (mark red)
        if not center_added:
            for center in by_position['C']:
                if id(center) not in used_players:
                    center.specific_position = f"{Colors.RED}C{Colors.RESET}"
                    starters.append(center)
                    used_players.add(id(center))
                    center_added = True
                    break
        
        # If still not 5 starters, fill from remaining best players (mark red)
        while len(starters) < 5:
            for player in all_players_list:
                if id(player) not in used_players:
                    # Only mark red if position isn't already marked
                    if Colors.RED not in player.specific_position:
                        player.specific_position = f"{Colors.RED}{player.specific_position}{Colors.RESET}"
                    starters.append(player)
                    used_players.add(id(player))
                    break
        
        # Collect remaining players as bench
        bench = []
        for player in all_players_list:
            if id(player) not in used_players:
                bench.append(player)
        
        # Get game time from first player
        game_time = players[0].gametime if players else None
        
        return TeamLineup(
            team=team,
            opponent=players[0].opponent if players else None,
            game_time=game_time,
            starters=starters,
            bench=bench
        )
    
    def project_all_lineups(self) -> Dict[str, TeamLineup]:
        """Project lineups for all teams"""
        players_by_team = self.get_players_by_team()
        lineups = {}
        
        for team, players in players_by_team.items():
            if not players:
                continue
            lineups[team] = self.project_team_lineup(team, players)
        
        return lineups
    
    def group_lineups_by_gametime(self, lineups: Dict[str, TeamLineup]) -> List[Tuple[Optional[datetime], List[TeamLineup]]]:
        """
        Group lineups by game time, sorted chronologically.
        
        Returns: List of (game_time, [lineups_at_that_time]) tuples
        """
        by_time = defaultdict(list)
        
        for team, lineup in lineups.items():
            by_time[lineup.game_time].append(lineup)
        
        # Sort by game time (None values go first)
        sorted_times = sorted(by_time.keys(), key=lambda x: x if x else datetime.min)
        
        return [(t, by_time[t]) for t in sorted_times]


class LineupDisplayFormatter:
    """Formats and displays lineups in terminal with grid layout"""
    
    def __init__(self, show_salary: bool = True, show_std: bool = True):
        self.show_salary = show_salary
        self.show_std = show_std
        self.terminal_width, self.terminal_height = shutil.get_terminal_size()
    
    def _get_status_color(self, status: Optional[str]) -> str:
        """Get color code based on player status"""
        if status == 'Out':
            return Colors.RED
        elif status in ['Q', 'MAY NOT PLAY', 'Doubt']:
            return Colors.YELLOW
        elif status == 'OFS':
            return Colors.RED
        return Colors.WHITE
    
    def _format_player_line(self, player: ProjectedPlayer, compact: bool = False) -> str:
        """Format a single player line with position, name, and projection"""
        status_color = self._get_status_color(player.status)
        status_str = f" {Colors.YELLOW}{player.status}{Colors.RESET}" if player.status else ""
        
        if compact:
            # Compact format for grid: "PG Name 12.3±1.2"
            name_short = player.name[:20]  # Truncate long names
            proj_str = f"{player.fpts_mean:.1f}±{player.fpts_std:.1f}" if self.show_std else f"{player.fpts_mean:.1f}"
            return f"{player.specific_position} {name_short:20s} {proj_str:>10s}{status_str}"
        else:
            # Full format
            proj_str = f"{player.fpts_mean:.1f}"
            if self.show_std:
                proj_str += f"±{player.fpts_std:.1f}"
            
            line = f"  {Colors.CYAN}{player.specific_position:3s}{Colors.RESET} {status_color}{player.name:25s}{Colors.RESET}"
            
            if self.show_salary:
                line += f" ${player.salary:>6,}"
            
            line += f" {Colors.GREEN}{proj_str:>8}{Colors.RESET}"
            line += status_str
            
            return line
    
    def format_team_lineup_compact(self, lineup: TeamLineup, width: int = 38) -> List[str]:
        """
        Format a team lineup for grid display with starters and bench separated.
        
        Returns list of lines that fit within width.
        """
        lines = []
        
        # Header with team and time
        header = f"{Colors.BOLD}{Colors.CYAN}{lineup.team:3s}{Colors.RESET}"
        if lineup.game_time:
            header += f" {lineup.game_time.strftime('%I:%M')}"
        lines.append(self._pad_to_width(header, width))
        
        # Separator
        sep_visual = '─' * (width - 2)
        lines.append(self._pad_to_width(f"{Colors.DIM}{sep_visual}{Colors.RESET}", width))
        
        # Column headers
        col_header = f"Pos Name        FPTS Salary  $/K"
        lines.append(self._pad_to_width(col_header, width))
        
        # Starters header
        lines.append(self._pad_to_width(f"{Colors.BOLD}STARTERS{Colors.RESET}", width))
        
        # Starters
        for player in lineup.starters:
            player_line = self._format_player_line_compact(player)
            lines.append(self._pad_to_width(player_line, width))
        
        # Bench header
        if lineup.bench:
            lines.append(self._pad_to_width(f"{Colors.BOLD}BENCH{Colors.RESET}", width))
            
            # Bench players
            for player in lineup.bench:
                player_line = self._format_player_line_compact(player)
                lines.append(self._pad_to_width(player_line, width))
        
        return lines
    
    def _format_player_line_compact(self, player: ProjectedPlayer) -> str:
        """Format player line for compact grid display with salary and value"""
        # Use yellow if questionable ('Q'), normal color otherwise
        name_color = Colors.YELLOW if player.status and player.status.upper() == 'Q' else Colors.WHITE
        
        # Truncate name to 15 chars
        name_short = player.name[:15]
        
        # Calculate value (fpts / (salary/1000))
        value = player.fpts_mean / (player.salary / 1000) if player.salary > 0 else 0
        
        # Conditional formatting for value - highlight if > 5.5
        if value > 5.5:
            value_str = f"{Colors.GREEN}{value:5.2f}{Colors.RESET}"
        else:
            value_str = f"{value:5.2f}"
        
        # Standardize position to 2 chars (pad with space if needed)
        pos_std = f"{player.specific_position:2s}"
        
        # Format: "PG name           12.3 $ 8,500  1.45"
        line = f"{Colors.CYAN}{pos_std}{Colors.RESET} {name_color}{name_short:15s}{Colors.RESET} {player.fpts_mean:5.1f} ${player.salary:>6,} {value_str}"
        
        return line
    
    def _pad_to_width(self, text: str, target_width: int) -> str:
        """Pad text to target width, accounting for ANSI codes"""
        clean_text = self._strip_ansi(text)
        padding_needed = target_width - len(clean_text)
        return text + (" " * max(0, padding_needed))
    
    def calculate_grid_layout(self, lineups_by_time: List[Tuple[Optional[datetime], List[TeamLineup]]]) -> Tuple[int, int]:
        """
        Calculate optimal grid layout (cols, rows) based on terminal size and number of games.
        
        Each column is 55 chars wide
        """
        total_games = sum(len(games) for _, games in lineups_by_time)
        
        col_width = 57  # 55 chars + 2 char separator
        cols = max(1, self.terminal_width // col_width)
        
        rows = max(1, (total_games + cols - 1) // cols)  # Ceiling division
        
        return cols, rows
    
    def display_lineups_grid(self, lineups_by_time: List[Tuple[Optional[datetime], List[TeamLineup]]]):
        """
        Display all lineups in a grid layout filling the terminal.
        Earliest games top-left, latest bottom-right.
        """
        # Flatten all lineups in chronological order
        all_lineups = []
        for game_time, lineups in lineups_by_time:
            # Sort teams at same game time alphabetically for consistency
            for lineup in sorted(lineups, key=lambda x: x.team):
                all_lineups.append(lineup)
        
        cols, rows = self.calculate_grid_layout(lineups_by_time)
        col_width = 55  # Fixed column width to show all data
        
        # Print header
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * (self.terminal_width - 2)}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}NBA PROJECTED LINEUPS - Grid View{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'=' * (self.terminal_width - 2)}{Colors.RESET}\n")
        
        # Build grid row by row
        for row in range(rows):
            row_lineups = all_lineups[row * cols:(row + 1) * cols]
            
            if not row_lineups:
                break
            
            # Get formatted lines for each lineup in this row
            formatted_lineups = [self.format_team_lineup_compact(lu, col_width) for lu in row_lineups]
            
            # Find max height in this row
            max_height = max(len(fu) for fu in formatted_lineups)
            
            # Print each line across all lineups in row
            for line_idx in range(max_height):
                line_parts = []
                for formatted_lineup in formatted_lineups:
                    if line_idx < len(formatted_lineup):
                        part = formatted_lineup[line_idx]
                    else:
                        part = self._pad_to_width("", col_width)
                    
                    line_parts.append(part)
                
                print("  ".join(line_parts))
            
            # Spacing between rows
            if row < rows - 1:
                print()
        
        # Print footer
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * (self.terminal_width - 2)}{Colors.RESET}")
        print(f"Total Teams: {len(all_lineups)} | Games by Tip-off Time | $/K = FPTS / (Salary/1000)")
        print(f"{Colors.BOLD}{Colors.GREEN}{'=' * (self.terminal_width - 2)}{Colors.RESET}\n")
    
    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI color codes from text"""
        ansi_codes = [
            Colors.GREEN, Colors.YELLOW, Colors.RED, Colors.CYAN, Colors.WHITE,
            Colors.BOLD, Colors.DIM, Colors.RESET
        ]
        result = text
        for code in ansi_codes:
            result = result.replace(code, "")
        return result


# ============================================================================
# VALIDATION & COMPARISON
# ============================================================================

class LineupValidator:
    """Validates projected lineups against actual lineups"""
    
    @staticmethod
    def compare_projected_to_actual(
        projected: TeamLineup,
        actual_starters: List[str]  # List of actual starter names
    ) -> Dict:
        """
        Compare projected starters to actual starters.
        
        Returns:
        {
            'matches': int,
            'mismatches': List[str],
            'benched': List[str],  # Projected starters not in actual
            'called_up': List[str],  # In actual but not projected starter
        }
        """
        projected_names = {p.name for p in projected.starters}
        actual_set = set(actual_starters)
        
        matches = len(projected_names & actual_set)
        mismatches = list(projected_names - actual_set)
        called_up = list(actual_set - projected_names)
        
        return {
            'matches': matches,
            'mismatches': mismatches,
            'called_up': called_up,
            'accuracy': matches / len(projected.starters) if projected.starters else 0
        }


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def display_lineups(session: Session):
    """
    Main entry point: Project and display all team lineups in grid format.
    
    Args:
        session: SQLAlchemy session
    """
    projector = LineupProjector(session)
    formatter = LineupDisplayFormatter(show_salary=False, show_std=True)
    
    # Project all lineups
    lineups = projector.project_all_lineups()
    
    # Group by game time
    lineups_by_time = projector.group_lineups_by_gametime(lineups)
    
    # Display in grid
    formatter.display_lineups_grid(lineups_by_time)
    
    return lineups


if __name__ == "__main__":
    engine = init_db('nba')
    session = get_session(engine)
    
    lineups = display_lineups(session)
