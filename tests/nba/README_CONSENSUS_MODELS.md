# NBA Consensus Player Models

## Overview

`build_player_consensus_models.py` builds individualized fantasy point distributions for each NBA player by combining:

1. **Historical position-level marginals** - empirical stat distributions per position
2. **Consensus odds constraints** - aggregated over/under lines from multiple sportsbooks
3. **Historical correlations** - how stats co-vary within each position
4. **DraftKings scoring** - fantasy point calculations with bonuses

The result is a realistic, market-informed projection of each player's fantasy point distribution.

## Pipeline Overview

```
NBAPlayer records
       ↓
   [Parse Position]
       ↓
Load Position Marginals & Correlations
       ↓
Get Consensus Odds (from NBAOdds)
       ↓
   [Sample Base Marginals]
       ↓
   [Adjust Each Stat's Marginal with Odds Constraints]
       ↓
   [Impose Position-Level Correlation Structure]
       ↓
   [Calculate Fantasy Points (with bonuses)]
       ↓
   [Fit Normal Distribution to FPTS]
       ↓
player_consensus_distributions.csv
```

## Key Components

### 1. Position Parsing

Converts raw position strings to normalized categories:
- **Guard (G)**: PG, SG
- **Forward (F)**: SF, PF
- **Center (C)**: C

Handles dual positions (e.g., "SG/SF" → "G", "PF/C" → "F")

### 2. Consensus Probability Calculation

**Why consensus?**
Different sportsbooks have slightly different odds due to their own models, market conditions, and risk management. Averaging across books creates a more robust, market-wide view.

**How it works:**

For each player and stat:
1. Collect all over/under lines from all sportsbooks
2. Group by (stat, line, side)
3. Convert American odds to implied probabilities
4. Average probabilities across sportsbooks
5. Match over/under pairs to create probability constraints

**Example:**
```
Luka Doncic Points at 30.5:
  DraftKings: -110 over, +107 under → 52.4% over, 47.6% under
  BetMGM:     -115 over, +105 under → 53.5% over, 46.5% under
  FanDuel:    -120 over, +100 under → 54.5% over, 45.5% under

Consensus: 53.5% over, 46.5% under (average)
```

### 3. Marginal Adjustment: Quantile Stretching

**Goal:** Match odds constraints while preserving the empirical distribution shape.

**Method:** Quantile stretching (Iman-Conover style)

1. Sort base samples to identify quantiles
2. Create piecewise linear transformation mapping quantiles to values
3. Use odds-derived probabilities as quantile targets
4. Interpolate smoothly between constraint points
5. Reorder samples to original indices (preserves independence)

**Key property:** Preserves sample ordering → maintains ranking relationships

**Example:**
```
Base distribution (position-level):
  Quantile 0.0 → 0 points
  Quantile 0.25 → 10 points
  Quantile 0.50 → 18 points
  Quantile 0.75 → 28 points
  Quantile 1.0 → 45 points

Odds constraint: 30.5 points → 46.5% quantile

Adjusted distribution:
  Quantile 0.0 → 0 points
  Quantile 0.25 → 10 points
  Quantile 0.465 → 30.5 points  ← from odds
  Quantile 0.50 → ~32 points
  Quantile 0.75 → ~40 points
  Quantile 1.0 → 45 points
```

### 4. Correlation Imposition: Iman-Conover Method

**Goal:** Add realistic cross-stat correlations while keeping adjusted marginals.

**Why needed:** Stats aren't independent. When a player scores more, they tend to:
- Shoot 3-pointers at slightly different rates (weak correlation)
- Get more assists (moderate positive correlation)
- Take more chances → turnovers (weak positive correlation)

**How Iman-Conover works:**

1. **Generate target ranks:** Sample multivariate normal with target correlation, get ranks
2. **Get current ranks:** Apply rankdata to samples (ignoring values)
3. **Reorder:** Rearrange samples to match target ranks

**Why this preserves marginals:**
- Reordering just changes which sample goes where
- Actual values never change, just their order
- Spearman rank correlation depends only on order, not values

**Example (simplified):**
```
Original samples (3 stats):
  Sample 1: [10 pts, 2 3pm, 4 reb]
  Sample 2: [25 pts, 3 3pm, 6 reb]
  Sample 3: [18 pts, 1 3pm, 5 reb]

After Iman-Conover with positive correlation (pts ↔ 3pm ↔ reb):
  Sample 1: [18 pts, 1 3pm, 4 reb]  ← low values grouped
  Sample 2: [25 pts, 3 3pm, 6 reb]  ← high values grouped
  Sample 3: [10 pts, 2 3pm, 5 reb]  ← medium values grouped

Correlation improved without changing marginal distributions!
```

### 5. Fantasy Points Calculation

DraftKings scoring:
```
Base:
  Points: 1.0x
  3-pointers made: 0.5x
  Rebounds: 1.25x
  Assists: 1.5x
  Steals: 2.0x
  Blocks: 2.0x
  Turnovers: -0.5x

Bonuses:
  Double-double (2+ stats ≥ 10): +1.5
  Triple-double (3+ stats ≥ 10): +3.0
```

### 6. Distribution Fitting

Fit normal distribution to 10,000 FPTS samples:
- **Mean:** Average FPTS across samples
- **Std:** Sample standard deviation (unbiased)

Stored for later use in:
- Player exposure calculations
- Stack variance modeling
- Risk management

## Output

**File:** `/data/nba/models/player_consensus_distributions.csv`

**Columns:**
- `player_id`: NBA API player ID
- `name`: Player name
- `position`: Original position string (e.g., "SG", "SF/PF")
- `norm_position`: Normalized position (G, F, C)
- `fpts_mean`: Mean fantasy points (from 10k samples)
- `fpts_std`: Std dev of fantasy points
- `sample_size`: Always 10,000
- `num_odds_lines`: Number of odds constraints applied

**Example:**
```
name,position,norm_position,fpts_mean,fpts_std,num_odds_lines
luka doncic,PG,G,63.91,26.09,23
stephen curry,PG,G,57.37,29.00,20
kevin durant,SF/PF,F,55.76,27.96,16
```

## Usage

### Basic Run
```bash
cd /home/sam-pratt/ivan
python tests/nba/build_player_consensus_models.py
```

### With Custom Sample Size
Edit the script and change `NUM_SAMPLES`:
```python
NUM_SAMPLES = 50000  # For more precision
```

### Checking Specific Player
```python
from ivan.database.session import init_db, get_session
from ivan.sports.nba.models import NBAPlayer
from pathlib import Path
import pandas as pd

# Load results
results_df = pd.read_csv(Path("/data/nba/models/player_consensus_distributions.csv"))

# Find player
player_stats = results_df[results_df['name'].str.lower() == 'luka doncic']
print(player_stats)
```

## Data Sources

1. **Historical stats:** `NBAStat` table (from `fit_base_marginals.py`)
2. **Odds:** `NBAOdds` table (from `src/ivan/sports/common/odds.py`)
3. **Base models:** Pickled files in `/data/nba/models/`
   - `marginals.pkl` - Position-level stat distributions
   - `correlations.pkl` - Position-level Spearman correlations

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| NUM_SAMPLES | 10,000 | Samples per player; increase for more precision |
| Positions | G, F, C | Guards, Forwards, Centers |
| Stats | 6 | points, fg3m, rebounds, assists, steals, blocks |
| Constraint threshold | 0.01-0.99 | Quantile limits for odds |

## How It Differs from `fit_player_distributions.py`

**Both scripts:**
- Load position-level models
- Adjust marginals with odds
- Impose correlations
- Calculate FPTS
- Fit distributions

**Key enhancement in `build_player_consensus_models.py`:**
- **Explicit consensus calculation** - averages odds across sportsbooks
- **Better logging** - detailed progress and diagnostics
- **Cleaner architecture** - separated concerns, documented functions
- **Sportsbook-aware** - tracks which books contributed to each constraint

## Validating Results

### Quick checks:
1. **All 81 players processed:** Check "Successfully processed: X players"
2. **Odds being applied:** Check "Avg odds lines per player" > 0
3. **Reasonable distributions:** Most players 30-50 FPTS mean, 10-30 std
4. **Position distribution:** ~37 guards, ~31 forwards, ~13 centers

### Detailed validation:
```python
import pandas as pd

results = pd.read_csv("/data/nba/models/player_consensus_distributions.csv")

# Check for anomalies
print("Players with no odds constraints:")
print(results[results['num_odds_lines'] == 0]['name'].tolist())

# Check FPTS reasonableness
print(f"Mean FPTS range: {results['fpts_mean'].min():.1f} - {results['fpts_mean'].max():.1f}")
print(f"Typical player FPTS std: {results['fpts_std'].mean():.1f}")

# By position
for pos in ['G', 'F', 'C']:
    pos_data = results[results['norm_position'] == pos]
    print(f"\n{pos}: mean={pos_data['fpts_mean'].mean():.1f}, "
          f"std={pos_data['fpts_std'].mean():.1f}, "
          f"n={len(pos_data)}")
```

## Performance

- **Runtime:** ~0.8 seconds for 81 players (10k samples each)
- **Per-player time:** ~10ms with 10,000 samples
- **Output size:** ~30KB CSV

## Debugging

Enable verbose logging for specific player:
```python
result = process_player(session, player, marginals, correlations, verbose=True)
```

Check odds collection:
```python
odds = get_player_consensus_odds(session, "luka doncic")
for stat, constraints in odds.items():
    print(f"{stat}: {len(constraints)} constraints")
    for line, over_prob, under_prob in constraints:
        print(f"  {line}: {under_prob:.1%} under, {over_prob:.1%} over")
```

## Future Enhancements

1. **Game-level correlations** - account for opponent strength, pace
2. **Team stacks** - model co-variance between teammates
3. **Injury adjustments** - downweight recent games by injured players
4. **Recency weighting** - give more weight to recent seasons
5. **Venue effects** - home/away adjustments
6. **Vine copulas** - more sophisticated correlation (like NFL implementation)

## References

- **Iman-Conover method**: Iman & Conover (1982) "A distribution-free approach to inducing rank correlation among input variables"
- **Spearman correlation**: Rank-based correlation, invariant to monotonic transformations
- **Quantile stretching**: Preserves marginals while matching constraints (copula-inspired)
