# IVAN

A comprehensive, multi-sport Daily Fantasy Sports (DFS) optimization and simulation framework built in Python. IVAN supports NBA, NFL, MLB, League of Legends (LoL), and MMA with advanced statistical modeling, lineup generation, Monte Carlo simulation, and contest analysis. please note that the code probably sucks and I'm not a good programmer. you can feel free to let me know how much it sucks directly @sambamch@Gmail.com.

## 🚀 Overview

IVAN is designed to support end-to-end DFS workflows for multiple sports with shared core infrastructure. The framework provides:

- **Data Ingestion**: Automated DraftKings API integration for player data, contests, and payouts
- **Statistical Modeling**: Advanced distribution fitting (KDE, parametric) with betting odds calibration
- **Lineup Optimization**: Linear programming-based lineup generation with sport-specific constraints
- **Monte Carlo Simulation**: Contest outcome simulation with field sampling and ownership projection
- **Late-Swap Optimization**: Post-lock lineup optimization for NBA and NFL
- **Ownership Projection**: Field-level ownership modeling and exposure analysis
- **Database Management**: SQLite-based storage with SQLAlchemy ORM

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SamBamCH/ivan.git
   cd ivan
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install GLPK solver** (required for lineup optimization):
   - Linux: `sudo apt-get install glpk-utils`
   - macOS: `brew install glpk`
   - Windows: Download from [GLPK website](https://www.gnu.org/software/glpk/) or use the bundled version

5. **Configure sport-specific settings:**
   - Edit JSON config files in `config/` directory for each sport
   - Set database paths, solver paths, and sport-specific parameters

## 🗂️ Project Structure

```
ivan/
├── src/ivan/                    # Core Python package
│   ├── core/                    # Shared utilities
│   │   └── utils.py            # File I/O, logging, name normalization
│   ├── database/               # Database management
│   │   ├── session.py         # Database initialization and session management
│   │   └── management.py      # Contest tracking and cleanup utilities
│   └── sports/                 # Sport-specific implementations
│       ├── common/             # Cross-sport utilities
│       │   ├── draftkings.py  # DraftKings API integration
│       │   ├── odds.py        # Betting odds API integration
│       │   └── io.py          # Data I/O helpers
│       ├── nba/               # NBA Classic (8-player lineups)
│       ├── nfl/               # NFL Showdown + Classic
│       ├── mlb/               # MLB Classic with stacking
│       ├── lol/               # League of Legends
│       └── mma/               # Mixed Martial Arts
│
├── scripts/                    # Command-line entrypoints
│   ├── nba/                   # NBA workflow scripts
│   ├── nfl/                   # NFL workflow scripts
│   ├── mlb/                   # MLB workflow scripts
│   └── lol/                   # LoL workflow scripts
│
├── config/                     # Sport-specific JSON configurations
│   ├── nba.json
│   ├── nfl.json
│   ├── mlb.json
│   └── lol.json
│
├── data/                       # SQLite databases and data files
│   ├── nba/
│   │   ├── nba_history.sqlite
│   │   ├── input/
│   │   ├── output/
│   │   └── models/
│   ├── nfl/
│   ├── mlb/
│   └── lol/
│
├── models/                     # Trained models (pickle files)
│   └── nfl/
│       ├── contest_dupe_quantiles.pkl
│       └── lineup_dupe_model.pkl
│
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## 🏀 NBA Module

### Features
- **Classic DFS Format**: 8-player rosters (PG, SG, SF, PF, C, G, F, UTIL)
- **Pre-lock and Post-lock workflows**: Separate modules for lineup generation and late-swap
- **Statistical Modeling**: Normal distribution-based projections with configurable variance
- **Ownership Projection**: Field-level ownership modeling
- **Late-Swap Optimization**: Optimize lineups after early games start

### Workflow Scripts

1. **`scripts/nba/ingest.py`** - Fetch DraftKings player/contest data
   ```bash
   python scripts/nba/ingest.py
   ```

2. **`scripts/nba/overnight.py`** - Automated overnight pipeline (ingest → generate → simulate → choose)

3. **`scripts/nba/sim.py`** - Run Monte Carlo contest simulation
   ```bash
   python scripts/nba/sim.py --contest-id 123456789 --num-sims 10000
   ```

4. **`scripts/nba/swap.py`** - Late-swap optimization
   ```bash
   python scripts/nba/swap.py --contest-id 123456789
   ```

### Key Modules
- `src/ivan/sports/nba/pre_lock/lineup_generation.py` - PuLP-based lineup optimization
- `src/ivan/sports/nba/pre_lock/contest.py` - Contest simulation and lineup selection
- `src/ivan/sports/nba/post_lock/contest.py` - Late-swap logic
- `src/ivan/sports/nba/models.py` - SQLAlchemy ORM models

## 🏈 NFL Module

### Features
- **Dual Contest Types**:
  - **Showdown**: 6-player rosters (1 CPT at 1.5x points + 5 FLEX), $50k salary cap
  - **Classic**: 9-player rosters (QB, RB, RB, WR, WR, WR, TE, FLEX, DST), $50k salary cap
- **Advanced Statistical Modeling**:
  - Individual stat distributions (pass_yds, rush_yds, rec_yds, TDs, etc.)
  - KDE-based distributions with betting odds calibration
  - Gaussian copula for stat correlation preservation
  - FPTS derived from stat combinations using DraftKings scoring
- **Duplication Prediction Model**: Two-stage model predicting lineup duplication in contests
- **Team Correlation**: QB-WR/TE stacking, bring-back stacks, team-level correlation

### Workflow Scripts

1. **`scripts/nfl/ingest.py`** - Fetch DraftKings player data
   ```bash
   python scripts/nfl/ingest.py -p --update-odds
   ```

2. **`scripts/nfl/update_odds.py`** - Update player prop betting odds
   ```bash
   python scripts/nfl/update_odds.py
   ```

3. **`scripts/nfl/build_models.py`** - Fit stat distributions with odds calibration

4. **`scripts/nfl/train_showdown_model.py`** - Train duplication prediction model (Showdown only)

5. **`scripts/nfl/generate_lineups.py`** - Generate optimized lineups
   ```bash
   python scripts/nfl/generate_lineups.py --mode showdown --num-lineups 150
   ```

6. **`scripts/nfl/sim.py`** - Monte Carlo contest simulation
   ```bash
   python scripts/nfl/sim.py --contest-id 123456789 --num-sims 10000
   ```

7. **`scripts/nfl/choose.py`** - Select diverse lineup subset based on simulation results

8. **`scripts/nfl/get_contest_data.py`** - Fetch contest metadata and entries

### Key Modules
- `src/ivan/sports/nfl/generate_lineups_logic.py` - Showdown and Classic lineup generation
- `src/ivan/sports/nfl/simulate_contest.py` - Contest simulation engine
- `src/ivan/sports/nfl/refactor/stat.py` - Stat distribution fitting with odds calibration
- `src/ivan/sports/nfl/refactor/player.py` - Player-level FPTS aggregation
- `src/ivan/sports/nfl/showdown_models.py` - Duplication prediction models
- `src/ivan/sports/nfl/ownership.py` - Ownership projection
- `src/ivan/sports/nfl/sample_field.py` - Field sampling for opponents

## ⚾ MLB Module

### Features
- **Roster Structure**: 9 players (2 P, C, 1B, 2B, 3B, SS, 2 OF)
- **Team Stacking**: Configurable stacking strategies (5-3, 5-2, 4-4, etc.)
- **Pitcher-Batter Correlation**: Constraints limiting opposing batters vs. owned pitchers
- **Historical Modeling**: Bin regressors for projection distributions
- **Park Factors**: Park-adjusted projections

### Workflow Scripts

1. **`scripts/mlb/ingest.py`** - Load projection CSVs
   ```bash
   python scripts/mlb/ingest.py
   ```

2. **`scripts/mlb/fetch_boxscores.py`** - Fetch historical box score data

3. **`scripts/mlb/fit_historical_fields.py`** - Fit bin regressors from historical contests

4. **`scripts/mlb/generate_lineups.py`** - Generate stacked lineups
   ```bash
   python scripts/mlb/generate_lineups.py --num-lineups 150
   ```

5. **`scripts/mlb/sim_all.py`** - Simulate all contests in database

6. **`scripts/mlb/choose_all.py`** - Select lineups across all contests

7. **`scripts/mlb/overnight.py`** - Automated overnight pipeline

### Key Modules
- `src/ivan/sports/mlb/generate_lineups.py` - PuLP-based lineup optimization with stacking
- `src/ivan/sports/mlb/simulation.py` - Contest simulation with field sampling
- `src/ivan/sports/mlb/selection.py` - Quality-aware diversity-based lineup selection
- `src/ivan/sports/mlb/sample_lineups.py` - Fast lineup sampling for field generation
- `src/ivan/sports/mlb/models.py` - SQLAlchemy ORM models

## 🎮 League of Legends (LoL) Module

### Features
- **Roster Structure**: 8 players (2 TOP, 2 JNG, 2 MID, 2 ADC)
- **Team Constraint**: Must select full 5-man teams (no mixing)
- **Team-Level Correlation**: All 5 players win/lose together
- **Moneyline-Based Projections**: Implied win probabilities from betting odds

### Workflow Scripts

1. **`scripts/lol/ingest.py`** - Fetch DraftKings player data and odds
2. **`scripts/lol/generate_lineups.py`** - Generate lineups with team constraints
3. **`scripts/lol/check_lineups.py`** - Validate lineup constraints
4. **`scripts/lol/get_contest_ids.py`** - List available contests
5. **`scripts/lol/get_payouts.py`** - Fetch contest payout structures

### Key Modules
- `src/ivan/sports/lol/lineup_generation.py` - Team-constrained lineup generation
- `src/ivan/sports/lol/models.py` - SQLAlchemy ORM models

## 🔧 Core Functionality

### Database Management

Each sport uses a separate SQLite database:

```python
from ivan.database.session import init_db, get_session

# Initialize database connection
engine = init_db('nba')  # or 'nfl', 'mlb', 'lol'
session = get_session(engine)

# Query data
from ivan.sports.nba.models import NBAPlayer
players = session.query(NBAPlayer).filter(NBAPlayer.gametime > lock_time).all()

# Commit changes
session.commit()
```

**Database Locations:**
- NBA: `data/nba/nba_history.sqlite`
- NFL: `data/nfl/nfl_history.sqlite`
- MLB: `data/mlb/mlb_history.sqlite`
- LoL: `data/lol/lol_history.sqlite`

### DraftKings Integration

The framework provides comprehensive DraftKings API integration:

```python
from ivan.sports.common.draftkings import ingest_players, fetch_all_contests

# Fetch upcoming contests
contests = fetch_all_contests(sport='NBA')

# Ingest players for a slate
players = ingest_players(sport='nba', dedupe_by_name=True)
```

**Features:**
- Automated slate selection
- Player data extraction (salary, position, gametime, etc.)
- Contest metadata retrieval
- Payout structure fetching
- Entry export (post-contest analysis)

### Betting Odds Integration

Integration with The Odds API for player prop odds:

```python
from ivan.sports.common.odds import get_odds, parse_odds

# Fetch odds for relevant teams
odds_by_team = get_odds(
    sport="americanfootball_nfl",
    api_key="your_api_key",
    relevant_teams={"KC", "BUF", "MIA"}
)

# Parse into DataFrame
df_odds = parse_odds(odds_by_team)
```

**Use Cases:**
- Calibrating stat distributions (NFL)
- Implied probability calculations (LoL)
- Market-based projections

### Lineup Generation

All sports use linear programming (PuLP with GLPK) for lineup optimization:

**Common Constraints:**
- Salary cap (typically $50,000)
- Position requirements
- Player uniqueness
- Sport-specific rules (stacking, team limits, etc.)

**Objective Function:**
- Maximize projected fantasy points
- Optional ownership penalties
- Configurable randomness/jitter

### Monte Carlo Simulation

Contest simulation with field sampling:

1. **Field Sampling**: Generate opponent lineups using ownership projections
2. **Player Sampling**: Sample player performance from fitted distributions
3. **Scoring**: Calculate lineup scores for all iterations
4. **Ranking**: Determine finish positions and payouts
5. **Metrics**: Calculate ROI, win%, percentile distributions

**Parallel Processing:**
- Simulations split across multiple processes
- Efficient NumPy-based scoring
- Configurable worker count

### Ownership Projection

Field-level ownership modeling:

- Generate sample field lineups
- Count player appearances
- Calculate ownership percentages
- Update player ownership projections

### Late-Swap Optimization

Post-lock lineup optimization (NBA/NFL):

1. Identify locked players (games started)
2. Generate swap candidates for unlocked slots
3. Simulate candidate outcomes
4. Recommend optimal swaps

## 📊 Statistical Modeling

### Distribution Fitting

**NFL (Advanced):**
- Individual stat distributions (pass_yds, rush_yds, etc.)
- KDE-based with betting odds calibration
- Gaussian copula for correlation
- FPTS derived from stat combinations

**NBA/MLB (Standard):**
- Normal distributions (mean, std dev)
- Historical data-based fitting
- Optional KDE for non-normal stats

### Odds Calibration

NFL-specific feature that adjusts distributions to match betting market probabilities:

- Extract quantile points from betting lines
- Calibrate CDF to match market expectations
- Preserve historical distribution shape
- Improve projection accuracy

## 🎯 Usage Examples

### Complete NBA Workflow

```bash
# 1. Ingest player data
python scripts/nba/ingest.py

# 2. Generate lineups
python scripts/nba/overnight.py --generate --num-lineups 150

# 3. Simulate contest
python scripts/nba/sim.py --contest-id 123456789 --num-sims 10000

# 4. Choose final lineups
python scripts/nba/overnight.py --choose --num-lineups 20
```

### Complete NFL Showdown Workflow

```bash
# 1. Ingest players and odds
python scripts/nfl/ingest.py -p --update-odds

# 2. Build stat models
python scripts/nfl/build_models.py

# 3. Train duplication model (if needed)
python scripts/nfl/train_showdown_model.py

# 4. Generate lineups
python scripts/nfl/generate_lineups.py --mode showdown --num-lineups 150

# 5. Simulate contest
python scripts/nfl/sim.py --contest-id 123456789 --num-sims 10000

# 6. Choose final lineups
python scripts/nfl/choose.py --contest-id 123456789 --num-lineups 20
```

### Complete MLB Workflow

```bash
# 1. Ingest projections
python scripts/mlb/ingest.py

# 2. Generate lineups
python scripts/mlb/generate_lineups.py --num-lineups 150

# 3. Simulate all contests
python scripts/mlb/sim_all.py

# 4. Choose lineups
python scripts/mlb/choose_all.py
```

## ⚙️ Configuration

Each sport has a JSON configuration file in `config/`:

**Common Settings:**
- `database_url`: SQLite database path
- `solver.glpk_path`: Path to GLPK solver executable
- `min_salary` / `max_salary`: Salary constraints
- `randomness`: Jitter factor for projections
- `num_sims`: Default simulation count

**Sport-Specific Settings:**
- **NFL**: Showdown duplication model parameters, ownership alpha, correlation factors
- **MLB**: Stack probabilities, pitcher-batter constraints
- **NBA**: Max salary per team, position constraints

## 🗄️ Database Schema

### Common Models (All Sports)

- **Player**: Player pool with projections, salary, ownership
- **Lineup**: Generated lineup entries
- **HistoricalContest**: Contest metadata
- **ContestPayout**: Payout structure
- **LineupSimulationResult**: Simulation outcomes (ROI, win%, percentiles)

### Sport-Specific Models

**NFL:**
- `ShowdownPlayer`, `ShowdownLineup` (Showdown format)
- `NFLPlayer`, `Lineup` (Classic format)
- `NFLOdds` (Betting odds)
- `NFLStat` (Historical stats)

**MLB:**
- `MLBPlayer`, `MLBStat` (Box scores)
- `BatterProjection`, `PitcherProjection`

**NBA:**
- `NBAPlayer`, `NBAStat` (Box scores)
- `NBAOdds` (Player props)

## 🔍 Key Design Principles

1. **Separation of Concerns**: Core logic (`src/ivan/sports/`) vs. Scripts (`scripts/`)
2. **Database-Centric**: All data flows through SQLAlchemy models
3. **Configuration-Driven**: Behavior controlled by JSON configs
4. **Parallel-First**: Multiprocessing for lineup generation and simulation
5. **Type Hints**: Dataclasses and type hints throughout
6. **Progress Visibility**: Rich library for progress bars and formatted output
7. **Sport Independence**: Each sport module is self-contained with shared utilities

## 🚨 Common Pitfalls

### ❌ Don't:
- Create database engines manually (use `init_db()`)
- Share database sessions across processes
- Forget to call `init_db()` before `get_session()`
- Use `print()` instead of Rich console
- Hardcode roster structures
- Forget to commit database changes

### ✅ Do:
- Always use `init_db('sport')` → `get_session(engine)`
- Use separate DB connections in each process
- Use Rich for all user-facing output
- Call `session.commit()` after modifications
- Use Queue for inter-process communication
- Store player_ids as JSON arrays in database

## 📚 Dependencies

**Critical Packages:**
- `sqlalchemy` - ORM and database interaction
- `pulp` - Linear programming solver (with GLPK backend)
- `numpy` - Array operations and Monte Carlo sampling
- `pandas` - Data manipulation
- `scipy` - Statistical distributions, KDE fitting
- `scikit-learn` - Machine learning models
- `rich` - Terminal UI (progress bars, formatted output)
- `selenium` - Browser automation for DraftKings
- `requests` - HTTP requests for APIs

**Sport-Specific:**
- `nflreadpy` - NFL data
- `nba_api` - NBA data
- `MLB-StatsAPI` - MLB data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/awesome-feature`
3. Commit your changes: `git commit -am 'Add awesome feature'`
4. Push to the branch: `git push origin feature/awesome-feature`
5. Open a pull request

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- DraftKings for providing the DFS platform and API
- The Odds API for betting odds data
- GLPK project for the linear programming solver
- All contributors and users of the framework

---

**Note**: This framework is for educational and research purposes. Always comply with DraftKings Terms of Service and applicable laws when using this software.
