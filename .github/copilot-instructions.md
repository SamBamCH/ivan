# GitHub Copilot Instructions for IVAN

## ⚡ Response Style
- **BE CONCISE** - No verbose explanations unless asked
- **MINIMAL OUTPUT** - Only show essential info (errors, warnings, key results)
- **NO DOCUMENTATION** - Don't create docs unless explicitly requested

## 🎯 Project Overview
IVAN is a multi-sport DFS (Daily Fantasy Sports) optimization framework supporting NBA, NFL, MLB, and LOL. Each sport follows a unified architecture with sport-specific modeling while sharing core infrastructure for database management, lineup generation, Monte Carlo simulation, and optimization.

**Core Pipeline (All Sports):**
1. **Ingest** - Fetch DraftKings contest/player data + betting odds
2. **Model** - Fit statistical distributions to player performance
3. **Generate** - Create optimized lineups using linear programming
4. **Simulate** - Monte Carlo contest simulation with field sampling
5. **Choose** - Select diverse subset of lineups to enter
6. **Swap** - Late-swap optimization after games start (NBA/NFL)

## 📁 Architecture Overview

### Project Structure
```
ivan/
├── src/ivan/
│   ├── core/              # Shared utilities across all sports
│   │   ├── player.py      # BasePlayer mixin for ORM models
│   │   ├── lineup.py      # BaseLineup mixin for ORM models
│   │   ├── config.py      # Configuration loader
│   │   └── utils.py       # File I/O, logging helpers
│   ├── database/
│   │   └── session.py     # init_db(), get_session() - CRITICAL
│   └── sports/
│       ├── common/        # Cross-sport modeling utilities
│       │   ├── kdes.py    # Gaussian KDE utilities
│       │   ├── regressors.py  # Bin regressors
│       │   └── io.py      # Data I/O helpers
│       ├── nba/           # NBA Classic (8-player lineups)
│       ├── nfl/           # NFL Showdown + Classic
│       ├── mlb/           # MLB Classic with stacking
│       └── lol/           # League of Legends
├── scripts/               # CLI entrypoints (per sport)
│   ├── nba/              # NBA workflow scripts
│   ├── nfl/              # NFL workflow scripts
│   ├── mlb/              # MLB workflow scripts
│   └── lol/              # LOL workflow scripts
├── config/               # Sport-specific JSON configs
│   ├── nba.json
│   ├── nfl.json
│   ├── mlb.json
│   └── lol.json
├── data/                 # SQLite databases (gitignored)
│   ├── nba/nba_history.sqlite
│   ├── nfl/nfl_history.sqlite
│   └── ...
└── tests/                # Visualization & test scripts
```

### Multi-Sport Database Pattern
Each sport has **separate SQLite database** with **own metadata namespace**:
- NBA: `data/nba/nba_history.sqlite` (no schema prefix)
- NFL: `data/nfl/nfl_history.sqlite` (uses `nfl_metadata`)
- MLB: `data/mlb/mlb_history.sqlite` (uses shared `Base`)
- LOL: `data/lol/lol_history.sqlite`

### Sport Module Structure (Consistent Pattern)
Each `src/ivan/sports/{sport}/` contains:
- `models.py` - SQLAlchemy ORM models (Player, Stat, Lineup, etc.)
- `lineup_generation.py` or `generate_lineups_logic.py` - PuLP optimization
- `simulate_contest.py` - Monte Carlo simulation engine
- `ownership.py` - Ownership projection
- `sample_field.py` - Field sampling for opponents
- Sport-specific files (e.g., `showdown_models.py` for NFL)

## 🏀 NBA Module (Primary Reference Implementation)

### Core Module Location: `src/ivan/sports/nba/`

**Key Modules:**
- `models.py` - SQLAlchemy ORM models for all NBA entities
- `lineup_generation.py` - PuLP-based optimization for generating lineups
- `fpts_calculations.py` - Statistical modeling and distribution fitting
- `generate_samples.py` - Monte Carlo sampling for player projections
- `simulate_contest.py` - Contest simulation engine
- `ownership.py` - Ownership projection and field sampling
- `sample_field.py` - Field sampling for opponent lineups
- `choose.py` - Diversity-based lineup selection
- `swap_logic.py` - Late-swap optimization logic
- `utils.py` - Helper functions for position coercion, config lookup

### Scripts Location: `scripts/nba/`

**Workflow Scripts (in typical execution order):**
1. `ingest.py` - Fetch DraftKings contest/player data via API
2. `update_odds.py` - Update player prop odds
3. `build_models.py` - Fit statistical distributions (currently empty/TODO)
4. `generate_lineups.py` - Generate optimized lineups using multiprocessing
5. `sim.py` - Simulate contest outcomes with Monte Carlo
6. `choose.py` - Select diverse subset of lineups to enter
7. `swap.py` - Optimize entries after some games have started (late swap)

**Utility Scripts:**
- `create_db.py` - Initialize NBA database schema
- `clear_all.py` - Clear player/lineup data
- `clear_sims.py` - Clear simulation results
- `get_contest_ids.py` - List available contests
- `get_payouts.py` - Fetch contest payout structures
- `analyze_lineups.py` - Analyze lineup characteristics
- `monitor.py` - Monitor running processes

## 🏈 NFL Module

### Dual Contest Types: Showdown vs Classic

**NFL supports TWO distinct contest formats with separate workflows:**

#### Showdown (Captain Mode)
- **6-player roster**: 1 CPT (1.5x points) + 5 FLEX
- **Salary cap**: $50,000
- **Models**: `ShowdownPlayer`, `ShowdownLineup`, `ShowdownContestMetadata`
- **Key scripts**: `scripts/nfl/generate_lineups.py`, `scripts/nfl/sim.py`
- **Unique feature**: Duplication prediction model (predicts how many times lineups appear in field)

#### Classic (Traditional DFS)
- **9-player roster**: QB, RB, RB, WR, WR, WR, TE, FLEX, DST
- **Models**: `NFLPlayer`, `Lineup` (shared with NBA pattern)
- **Stacking**: QB-WR/TE correlation, bring-back stacks

### NFL-Specific Modeling: KDE + Odds Calibration

**Philosophy**: Instead of simple normal distributions, NFL uses:
1. **Individual stat modeling** (pass_yds, rush_yds, rec_yds, TDs, etc.)
2. **Odds calibration** - Adjust KDEs to match betting market probabilities
3. **Correlation preservation** - Gaussian copula maintains stat dependencies
4. **FPTS derivation** - Combine stats using DraftKings scoring weights

**Key Module**: `src/ivan/sports/nfl/refactor/stat.py`

```python
from ivan.sports.nfl.refactor.stat import fit_stat_distribution, sample_stat

# Fit distribution with odds calibration
stat_dist = fit_stat_distribution(
    odds_df,              # DataFrame with betting odds
    historical_values,    # List of historical stat values
    stat_name='passing_yards'
)

# Sample from calibrated distribution
samples = sample_stat(stat_dist, n_samples=10000, rng=np.random.default_rng(42))
```

**StatDistribution dataclass**:
- `dist_name`: 'kde' or parametric ('norm', 'poisson')
- `kde_bandwidth`, `kde_support`: For KDE-based distributions
- `quantile_map`: Calibration points mapping raw values to odds-adjusted CDFs
- `odds_points`: List of (line, target_cdf) from betting markets

**Position-specific stat sets** (in `refactor/player.py`):
```python
STATS = {
    'QB': ['passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards', 'rushing_tds'],
    'RB': ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards'],
    'WR': ['receptions', 'receiving_yards', 'receiving_tds'],
    'TE': ['receptions', 'receiving_yards', 'receiving_tds'],
    'K':  ['fg_made', 'pat_made'],
    'DST': ['def_tds', 'def_sacks', 'def_interceptions', 'special_teams_tds', 'points_allowed']
}
```

### Showdown Duplication Model v2

**Problem**: Predict how many times each lineup appears in a contest (critical for ownership/ROI)

**Two-stage architecture** (see `docs/nfl_showdown_dupe_v2_summary.md`):

**Stage 1: Contest Distribution Model**
- **Inputs**: `log_field_size`, `entry_fee_tier`, `ownership_gini`, `num_games_tier`
- **Outputs**: Quantiles of duplicate distribution (q50, q75, q90, q95, q99)
- **Model**: Multi-output Gradient Boosting

**Stage 2: Lineup Positioning Model**
- **Inputs**: `cpt_ownership`, `flex_ownership_mean`, `flex_ownership_std`, `salary_remaining_pct`
- **Output**: Duplication score (0-1) → percentile in contest distribution
- **Model**: Gradient Boosting Regressor

**Combination**: Interpolate lineup percentile into contest distribution to get duplicate count

### NFL Workflow Scripts

**Setup/Ingestion**:
1. `create_db.py` - Initialize NFL database schema
2. `ingest.py` - Fetch DraftKings player/contest data
3. `update_odds.py` - Pull betting odds from sportsbooks
4. `get_contest_data.py` - Get contest metadata and entries

**Modeling**:
5. `build_models.py` - Fit stat distributions with odds calibration
6. `train_showdown_model.py` - Train duplication prediction model (Showdown only)

**Generation/Simulation**:
7. `generate_lineups.py` - Generate optimized lineups (Showdown or Classic)
8. `sim.py` - Monte Carlo contest simulation
9. `choose.py` - Select diverse lineup subset

**Maintenance**:
- `clear_all.py` - Clear player/lineup data
- `clear_sims.py` - Clear simulation results
- `monitor.py` - Monitor worker processes

### NFL Config Pattern (`config/nfl.json`)

**Key sections**:
```json
{
  "showdown": {
    "use_learned_model": true,         // Use v2 dupe model vs simple formula
    "ownership_alpha": 1.0,            // Ownership projection sharpness
    "lineup_generation": {
      "n_samples": 5000,
      "salary_cap": 50000,
      "max_cap_drop": 5000,            // Allow salary under cap for diversity
      "cap_drop_exponent": 1,          // Weight distribution of cap drops
      "jitter": 0.30                   // Randomness in projections
    },
    "sampling": {                      // Field sampling parameters
      "n_samples": 100,
      "alpha": 0.0,                    // Ownership concentration
      "rho_game": 0.0,                 // Game-level correlation
      "kicker_rho": 0.40               // Kicker correlation factor
    }
  },
  "classic": {
    "lineup_generation": {
      "min_qb_wrte_stack": 1,          // Force QB stacks
      "min_qb_bringback": 1,           // Require bring-back player
      "max_team_players": 4,           // Team exposure limit
      "max_off_vs_dst": 2              // Max offensive players vs owned DST
    }
  }
}
```

### NFL Refactor Pattern

**Active refactoring**: `src/ivan/sports/nfl/refactor/`

**Goal**: Extract stat-level modeling into clean, testable modules:
- `stat.py` - Stat distribution fitting (KDE + odds calibration)
- `player.py` - Player-level FPTS aggregation with correlation
- `historical/` - Historical data ingestion utilities

**Migration approach**: New code in `refactor/`, gradually migrate scripts to use it

## ⚾ MLB Module

### MLB-Specific Features: Stacking + Pitcher-Batter Correlation

**Roster structure** (9 players):
- 2 Pitchers (P, P)
- 1 Catcher (C)
- 1 First Baseman (1B)
- 1 Second Baseman (2B)
- 1 Third Baseman (3B)
- 1 Shortstop (SS)
- 2 Outfielders (OF, OF)

**Stacking strategies** (config-driven):
```json
{
  "stack_probabilities": {
    "5-3": 0.20,   // 5-man primary stack + 3-man secondary
    "5-2": 0.40,   // 5-man primary + 2-man secondary
    "5-x": 0.20,   // 5-man primary + no secondary
    "4-4": 0.20,   // Two 4-man stacks
    "4-3": 0.20,   // 4-man + 3-man
    "4-x": 0.20    // 4-man primary only
  }
}
```

**Key constraint**: `max_opposing_batters` - Limit batters facing your pitchers

**Models**:
- `MLBStat` - Historical box score data (PA, AB, HR, RBI, SB, etc.)
- `BatterProjection` - Per-game projections with park factors
- `PitcherProjection` - Pitching projections

**Workflow scripts** (`scripts/mlb/`):
- `ingest.py` - Load projection CSVs
- `generate_lineups.py` - Generate stacked lineups
- `sim.py` - Simulate contests
- `analyze.py` - Post-analysis of lineup quality

## 🎮 LOL Module (League of Legends)

**Roster structure** (8 players):
- 2 Top laners
- 2 Junglers  
- 2 Mid laners
- 2 ADCs
- **Team constraint**: Must select full 5-man teams (no mixing)

**Unique challenge**: Team-level correlation (all 5 players win/lose together)

**Models**:
- `LOLPlayer` - Individual player projections
- `LOLTeam` - Team-level metadata
- `LOLLineup` - Generated lineups

## 🗄️ Database Patterns

### ALWAYS Use This Pattern:
```python
from ivan.database.session import init_db, get_session

# Initialize database connection
engine = init_db('nba')  # sport key must be 'nba', 'mlb', 'nfl', or 'lol'
session = get_session(engine)

# Query data
from ivan.sports.nba.models import NBAPlayer
players = session.query(NBAPlayer).filter(NBAPlayer.gametime > lock_time).all()

# Don't forget to commit changes
session.commit()
```

### NEVER:
- Create engines directly with `create_engine()`
- Use hardcoded database URLs
- Forget to call `init_db()` before `get_session()`

### Database Schema Pattern:
Each sport has its own metadata and Base class:
```python
# In models.py
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

nba_metadata = MetaData(schema="nba")
NBABase = declarative_base(metadata=nba_metadata)

class NBAPlayer(NBABase):
    __tablename__ = "players"
    # ... columns ...
```

## 🏗️ ORM Models (src/ivan/sports/nba/models.py)

### Primary Models:

**NBAPlayer** - Player pool with projections
- `id` (int, PK): DraftKings player ID
- `name` (str): Player full name
- `team` (str): 3-letter team abbreviation
- `opponent` (str): 3-letter opponent abbreviation
- `position` (str): Position string (e.g., "PG/SG")
- `gametime` (datetime): Game start time (UTC)
- `salary` (int): DraftKings salary
- `fpts_mean` (float): Projected fantasy points (mean)
- `fpts_std` (float): Projected fantasy points (std dev)
- `ownership` (float): Projected ownership percentage
- `exclude` (bool): Whether to exclude from lineups

**NBAStat** - Historical box score data
- `player_id`, `game_date`, `opponent`: Composite key
- Box score stats: `points`, `rebounds`, `assists`, `steals`, `blocks`, `turnovers`
- Shooting: `fgm`, `fga`, `fg3m`, `fg3a`, `ftm`, `fta`
- `minutes`, `dk_fpts`: Minutes played and DraftKings points

**NBAOdds** - Player prop betting lines
- `player_id`, `stat_type`, `book`: Composite key
- `line`, `over_odds`, `under_odds`: Betting line and odds

**Lineup** - Generated lineup entries
- `id` (int, PK): Auto-increment
- `player_ids` (JSON): List of 8 player IDs (in roster order)
- `total_salary` (int): Sum of player salaries
- `projected_fpts` (float): Expected lineup score
- Helper methods: `player_ids()`, `to_dict()`, `save()`

**HistoricalContest** - Contest metadata
- `contest_id` (int, PK): DraftKings contest ID
- `name`, `entry_fee`, `total_prizes`, `entries_max`: Contest details
- Used for simulation configuration

**ContestPayout** - Payout structure
- `contest_id`, `start_rank`, `end_rank`: Composite key
- `prize`: Payout amount for rank range

**LineupSimulationResult** - Simulation outcomes
- `lineup_id`, `contest_id`: Foreign keys
- `roi_mean`, `roi_std`, `win_pct`: Key metrics
- Percentile distributions: `p10_winnings`, `p50_winnings`, `p90_winnings`
- `simulations`: Number of Monte Carlo iterations

## ⚙️ Lineup Generation Workflow

### 1. Core Optimization (lineup_generation.py)

**ROSTER_STRUCTURE** - NBA Classic DFS roster:
```python
ROSTER_STRUCTURE = [
    ("PG", ["PG"]),
    ("SG", ["SG"]),
    ("SF", ["SF"]),
    ("PF", ["PF"]),
    ("C", ["C"]),
    ("G", ["PG", "SG"]),
    ("F", ["SF", "PF"]),
    ("UTIL", ["PG", "SG", "SF", "PF", "C"])
]
```

**FieldPlayer** dataclass:
```python
@dataclass(frozen=True)
class FieldPlayer:
    id: int
    positions: tuple[str, ...]  # eligible positions
    salary: int
    projection: float
    ownership: float = 0.0
    exclude: bool = False
```

**generate_single_classic_lineup()** - Core optimizer:
```python
def generate_single_classic_lineup(
    field_players: List[FieldPlayer],
    config: dict,
    locked_assignments: Dict[str, int] | None = None,  # {slot: player_id}
    excluded_ids: Set[int] | None = None
) -> List[int] | None:
    """
    Returns list of 8 player IDs in roster slot order, or None if infeasible.
    
    Uses PuLP with GLPK solver.
    Constraints: salary cap, roster structure, uniqueness, locked players.
    Objective: maximize projected points (with optional ownership penalties).
    """
```

**generate_and_store_field_lineups()** - Batch generation:
- Generates multiple unique lineups
- Implements uniqueness constraints (configurable overlap threshold)
- Calls `on_progress()` callback for progress tracking
- Commits to database in batches

### 2. Multiprocess Generation (scripts/nba/generate_lineups.py)

Pattern for parallelization:
```python
from multiprocessing import Process, Queue
from rich.progress import Progress

# Each worker runs in separate process
def worker(worker_id: int, num_lineups: int, config_path: str, q: Queue):
    engine = init_db('nba')  # Separate DB connection per process
    session = get_session(engine)
    
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Progress callback sends updates via queue
    def on_progress(delta: int, note: str | None = None):
        q.put(("advance", worker_id, delta, note or ""))
    
    created = generate_and_store_field_lineups(
        session, config, num_lineups, 
        on_progress=on_progress
    )
```

**ALWAYS** use separate database connections in multiprocessing contexts.

### 3. Configuration (config/nba.json)

Key configuration sections:
```json
{
  "classic": {
    "salary_cap": 50000,
    "uniqueness_threshold": 6,  // Max shared players between lineups
    "lineup_generation": {
      "min_salary": 49000,
      "max_salary": 50000,
      "randomness": 0.05,  // Add noise to projections
      "ownership_penalty": 0.0  // Penalize high ownership
    }
  }
}
```

## 🎲 Monte Carlo Simulation

### Sampling Pattern (generate_samples.py)

```python
from ivan.sports.nba.generate_samples import generate_player_samples

# Generate samples for all players
player_samples = generate_player_samples(
    session, 
    n_samples=10000,
    seed=42
)
# Returns: {player_id: np.array of shape (n_samples,)}
```

Default: Uses normal distribution with `fpts_mean` and `fpts_std` from NBAPlayer table.

### Simulation Engine (simulate_contest.py)

**simulate_one_contest()** - Main simulation function:
```python
def simulate_one_contest(
    session: Session,
    contest_id: int,
    our_lineup_ids: List[int],  # Lineups to evaluate
    num_sims: int = 10000,
    num_field_samples: int = 5000,  # Opponent lineups to simulate
    num_workers: int = 4,
    seed: int = 42
) -> None:
    """
    Simulates contest outcomes and stores results in LineupSimulationResult table.
    
    Process:
    1. Load contest metadata and payouts from HistoricalContest/ContestPayout
    2. Sample opponent field using ownership projections
    3. Generate player performance samples
    4. Simulate all lineups in parallel chunks
    5. Calculate ROI, win%, percentiles for each lineup
    6. Store results in database
    """
```

**Helper Functions:**
- `_load_and_scale_payouts_from_db()` - Load and normalize payouts
- `_expand_sampled_field()` - Replicate field lineups to contest size
- `_build_player_samples_matrix()` - Build matrix of all player samples
- `_simulate_chunk()` - Simulate subset of iterations (for parallelization)

### Parallel Simulation Pattern:
```python
from concurrent.futures import ProcessPoolExecutor, as_completed

# Split simulations across workers
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    for chunk_start in range(0, num_sims, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_sims)
        future = executor.submit(
            _simulate_chunk,
            chunk_start, chunk_end, 
            player_samples_matrix, all_lineups, scaled_payouts
        )
        futures.append(future)
    
    for future in as_completed(futures):
        chunk_results = future.result()
        # Aggregate results...
```

## 🔄 Late Swap Optimization

Late swap allows updating lineups after early games start but before late games lock.

### Data Structures (swap_logic.py)

```python
class OriginalEntry:
    """Represents an entry already submitted to DraftKings"""
    entry_id: str          # Unique entry identifier
    contest_id: int        # DraftKings contest ID
    lineup_id: int         # Database lineup ID
    player_ids: List[int]  # 8 player IDs in roster order

class SwapCandidate:
    """Represents a potential swap optimization"""
    parent_entry: OriginalEntry
    locked_positions: Set[int]    # Roster slots that can't change (games started)
    unlocked_positions: Set[int]  # Roster slots that can be swapped
    new_player_ids: List[int]     # New player IDs for unlocked slots
    projected_fpts: float          # Expected points for swapped lineup
```

### Swap Workflow:

1. **Identify Locked Players** - Players whose games have started:
```python
def _identify_locked_players(
    player_ids: List[int],
    lock_time: datetime,
    session: Session
) -> Set[int]:
    """
    Returns set of roster slot indices (0-7) for locked players.
    Compares player.gametime against lock_time.
    """
```

2. **Generate Swap Candidates** - Create new optimized lineups:
```python
def _generate_swap_candidates(
    original_entry: OriginalEntry,
    locked_positions: Set[int],
    available_players: List[FieldPlayer],
    config: dict,
    num_candidates: int
) -> List[SwapCandidate]:
    """
    For each candidate:
    - Lock players in locked_positions
    - Optimize remaining roster slots
    - Return multiple unique alternatives
    """
```

3. **Simulate Candidates** - Evaluate expected outcomes:
- Build player samples matrix for all players
- Simulate each candidate lineup against sampled field
- Calculate ROI for each swap option

4. **Output CSV** - Export recommendations:
```python
def write_lateswap_csv(
    results: List[Tuple[OriginalEntry, SwapCandidate, float]],  # (entry, best_swap, roi)
    output_path: str
):
    """
    Write CSV with columns:
    - entry_id, original_lineup, swap_lineup
    - original_roi, swap_roi, roi_improvement
    - locked_players, swapped_players
    """
```

### Integration with Lineup Generation:

IMPORTANT: Swap logic reuses `generate_single_classic_lineup()` with `locked_assignments`:
```python
# Example: Lock PG and SG slots, optimize rest
locked_assignments = {
    "PG": locked_player_ids[0],
    "SG": locked_player_ids[1]
}

new_lineup = generate_single_classic_lineup(
    field_players,
    config,
    locked_assignments=locked_assignments
)
```

## 📊 Ownership Projection

### Ownership Workflow (ownership.py)

```python
from ivan.sports.nba.ownership import project_ownership

project_ownership(
    session,
    num_field_lineups=10000,
    config_path="config/nba.json"
)
# Updates NBAPlayer.ownership column based on simulated field
```

**How it works:**
1. Generate sample field lineups using `sample_field()`
2. Count player appearances across all lineups
3. Calculate ownership % = appearances / total_lineups
4. Write ownership values back to NBAPlayer table

### Field Sampling (sample_field.py)

```python
from ivan.sports.nba.sample_field import sample_field

lineups = sample_field(
    session,
    num_lineups=5000,
    use_db_lineups=True  # Sample from existing Lineup table
)
# Returns: List[List[int]] - list of player ID lists
```

## 🎨 UI Patterns

### ALWAYS Use Rich for Progress/Output:
```python
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn

console = Console()

# Simple output
console.print("[green]✓[/green] Lineups generated successfully")
console.print(f"[yellow]![/yellow] Warning: {message}")
console.print(f"[red]✗[/red] Error: {error_message}")

# Progress bars
with Progress() as progress:
    task = progress.add_task("[cyan]Generating lineups...", total=num_lineups)
    for i in range(num_lineups):
        # ... work ...
        progress.advance(task)
```

### Script CLI Pattern (argparse):
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate NBA lineups")
    parser.add_argument("--config", default="config/nba.json", help="Config file")
    parser.add_argument("--lineups", type=int, default=150, help="Number of lineups")
    parser.add_argument("--workers", type=int, default=4, help="Worker processes")
    args = parser.parse_args()
    
    # Use args.config, args.lineups, args.workers
    
if __name__ == "__main__":
    main()
```

## 🔧 Utility Functions

### Position Handling (utils.py):
```python
from ivan.sports.nba.utils import _coerce_positions

# Normalize position strings
positions = _coerce_positions("PG/SG")  # Returns ("PG", "SG")
positions = _coerce_positions(["PG", "SG"])  # Returns ("PG", "SG")
```

### Config Lookup (utils.py):
```python
from ivan.sports.nba.utils import _cfg

# Safely get nested config values
value = _cfg(config, "classic.lineup_generation.randomness", default=0.0)
# Equivalent to config["classic"]["lineup_generation"]["randomness"]
# Returns default if any key in path doesn't exist
```

### Player ID Coercion:
```python
from ivan.sports.nba.utils import _coerce_player_ids

# Handle JSON strings or lists
player_ids = _coerce_player_ids("[123, 456, 789]")  # Returns [123, 456, 789]
player_ids = _coerce_player_ids([123, 456, 789])    # Returns [123, 456, 789]
```

## 🚨 Common Pitfalls

### ❌ Don't:
1. Create engines manually: `engine = create_engine("postgresql://...")`
2. Forget `init_db()` before `get_session()`
3. Share database sessions across processes
4. Use print() instead of Rich console
5. Hardcode roster structure (use ROSTER_STRUCTURE constant)
6. Forget to commit database changes
7. Use blocking operations in multiprocessing without queues
8. Store player_ids as strings (always List[int])
9. Mix sport-specific imports (e.g., import NFLPlayer when working with NBA)
10. Forget sport parameter in `init_db()` - it determines database path

### ✅ Do:
1. Always call `init_db('sport')` → `get_session(engine)` (sport = 'nba', 'nfl', 'mlb', 'lol')
2. Use separate DB connections in each process
3. Use Rich for all user-facing output
4. Import ROSTER_STRUCTURE from lineup_generation (NBA) or use sport-specific patterns
5. Call `session.commit()` after modifications
6. Use Queue for inter-process communication
7. Store player_ids as JSON arrays in database
8. Use FieldPlayer dataclass for player data in optimization
9. Leverage locked_assignments for late swap (NBA/NFL)
10. Generate multiple candidates for diversity
11. Check config/{sport}.json for sport-specific parameters before adding new ones
12. Use `tests/{sport}/` for visualization/validation scripts (not pytest)

## 📈 Typical Development Workflow

### Adding a New Feature:
1. **Model changes** - Update `models.py` if new data needed
2. **Data ingestion** - Extend `ingest.py` to fetch new data
3. **Core logic** - Add functions to appropriate module (lineup_generation, simulate_contest, etc.)
4. **Script** - Create/update script in `scripts/{sport}/`
5. **Config** - Add config parameters to `config/{sport}.json`
6. **Test** - Verify with sample data using `tests/{sport}/` visualization scripts

### Example: Adding Player Prop Constraints (NBA)
1. Model: Add `prop_targets` column to NBAPlayer
2. Ingest: Fetch props in `ingest.py`, store in NBAOdds
3. Logic: Update `generate_single_classic_lineup()` to add prop constraints
4. Config: Add `"use_prop_constraints": true` to config
5. Test: Run `generate_lineups.py` and verify constraints work

### Example: Adding New Stat Market (NFL)
1. Model: Ensure stat column exists in NFLStat (e.g., `passing_2pt_conversions`)
2. Mapping: Add market name to `MARKET_TO_STAT` in `refactor/stat.py`
3. Ingestion: Update `update_odds.py` to fetch new market odds
4. Scoring: Add stat weight to `FPTS_WEIGHTS` in `refactor/player.py`
5. Test: Use `tests/nfl/visualize_stat_to_fpts.py` to verify calibration

## 🔍 Debugging Tips

### Check Database State:
```python
from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import NBAPlayer, Lineup

engine = init_db('nba')
session = get_session(engine)

# Count records
print(f"Players: {session.query(NBAPlayer).count()}")
print(f"Lineups: {session.query(Lineup).count()}")

# Check specific player
player = session.query(NBAPlayer).filter_by(name="LeBron James").first()
if player:
    print(f"Salary: ${player.salary}, Projection: {player.fpts_mean:.2f}")
```

### Test Lineup Generation:
```python
from ivan.sports.nba.lineup_generation import generate_single_classic_lineup, FieldPlayer

# Create minimal test case
players = [
    FieldPlayer(id=1, positions=("PG",), salary=8000, projection=45.0),
    FieldPlayer(id=2, positions=("SG",), salary=7000, projection=40.0),
    # ... add 6 more players covering all positions
]

lineup = generate_single_classic_lineup(
    players,
    config={"classic": {"salary_cap": 50000}},
    locked_assignments=None
)
print(f"Generated lineup: {lineup}")
```

### Trace Multiprocessing Issues:
```python
# Add detailed logging in worker function
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[Worker %(process)d] %(message)s"
)
logger = logging.getLogger(__name__)

def worker(...):
    logger.info(f"Worker started with {num_lineups} lineups")
    # ... rest of worker code
```

### Visualize NFL Stat Modeling (tests/nfl/)
```bash
# Comprehensive player modeling pipeline
python tests/nfl/visualize_stat_to_fpts.py "Patrick Mahomes"

# Compare multiple players
python tests/nfl/visualize_player_comparison.py "Patrick Mahomes" "Josh Allen" "Lamar Jackson"

# Check odds calibration quality
python tests/nfl/analyze_odds_calibration.py
```

## 📚 External Dependencies

**Critical packages:**
- `sqlalchemy` - ORM and database interaction
- `pulp` - Linear programming solver (with GLPK backend)
- `numpy` - Array operations and Monte Carlo sampling
- `pandas` - Data manipulation (especially in ingestion)
- `scipy` - Statistical distributions, KDE fitting
- `rich` - Terminal UI (progress bars, formatted output)

**When adding dependencies:**
1. Add to `requirements.txt`
2. Document usage in this file
3. Consider impact on multiprocessing (pickleable?)

## 🎯 Key Design Principles

1. **Separation of Concerns**: Core logic (src/ivan/sports/{sport}/) vs Scripts (scripts/{sport}/)
2. **Database-Centric**: All data flows through SQLAlchemy models
3. **Configuration-Driven**: Behavior controlled by config/{sport}.json
4. **Parallel-First**: Design for multiprocessing from start
5. **Type Hints**: Use dataclasses and type hints for clarity
6. **Progress Visibility**: Always show progress for long operations
7. **Reusability**: Core functions like generate_single_classic_lineup() used everywhere
8. **Sport Independence**: Each sport module is self-contained with shared utilities in core/

---

**Remember:** This documentation reflects the ACTUAL codebase structure as of the last update. When in doubt, check the source code. Happy coding! 🚀
