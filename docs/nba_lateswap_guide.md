# NBA Late Swap Guide

## Overview

The NBA late swap functionality allows you to optimize your DraftKings lineups after some games have already started (locked) while other games are still open for changes. This maximizes your expected value by using updated information and optimizing only the players whose games haven't started yet.

## Architecture

### Core Logic (`src/ivan/sports/nba/swap_logic.py`)

All late swap logic lives in `swap_logic.py` and **reuses the core lineup generation functions** from `lineup_generation.py`:

**Key Functions:**

1. **`_map_players_to_slots()`** - Maps player IDs to their roster slots (PG, SG, SF, PF, C, G, F, UTIL) based on position eligibility. This is essential for determining which slots are locked.

2. **`_identify_locked_players()`** - Compares each player's gametime against the lock time to determine which are locked (game started) vs. swappable.

3. **`_generate_swap_candidates()`** - The core optimization function that:
   - Maps locked players to their original roster slots
   - Gets all available unlocked players from the pool
   - Uses `generate_single_classic_lineup()` with `locked_assignments` parameter
   - Generates diverse candidates using Monte Carlo sampling
   - Returns candidate lineups with locked + newly optimized players

4. **`lateswap_one_contest()`** - Main entry point that:
   - Identifies locked vs. swappable players for each entry
   - Generates candidates for each entry
   - Simulates all candidates using Monte Carlo
   - Selects the best candidate per entry based on ROI
   - Returns results for CSV output

5. **`parse_entries_csv()`** - Parses DraftKings native export format CSV with entries

6. **`write_lateswap_csv()`** - Writes optimization results to CSV for review

### Script (`scripts/nba/swap.py`)

Simple CLI wrapper that:
- Parses command-line arguments
- Initializes database connection
- Calls `parse_entries_csv()` → `lateswap_one_contest()` → `write_lateswap_csv()`
- Provides Rich console output for progress tracking

## How It Works

### 1. Reusing Core Lineup Generation

The late swap logic **reuses** `generate_single_classic_lineup()` from `lineup_generation.py` by using the `locked_assignments` parameter:

```python
# Example: Lock PG and SG slots, optimize rest
locked_assignments = {
    123456: "PG",  # Player ID 123456 locked in PG slot
    789012: "SG"   # Player ID 789012 locked in SG slot
}

lineup = generate_single_classic_lineup(
    players=available_players,      # Only unlocked players
    base_scores=projected_scores,   # Fantasy point projections
    config=config,                  # NBA config (salary cap, etc.)
    locked_assignments=locked_assignments  # Force these players into slots
)
```

This ensures:
- ✅ Same optimization logic as pre-lock lineups
- ✅ All constraints respected (salary cap, roster structure, multi-game)
- ✅ No code duplication

### 2. Slot Mapping Algorithm

The `_map_players_to_slots()` function uses a **greedy assignment** that matches the optimizer's natural slot order:

```python
# ROSTER_STRUCTURE defines the order:
# PG → SG → SF → PF → C → G → F → UTIL

# For each slot, assign the first eligible player
for slot_name, _, eligible_positions in ROSTER_STRUCTURE:
    for player_id in remaining_players:
        if player.position in eligible_positions:
            assign(player_id → slot_name)
            break
```

This ensures consistency with how the optimizer fills slots.

### 3. Candidate Diversity

To generate diverse swap candidates (not just one optimal lineup), we use **Monte Carlo sampling**:

```python
# Generate player samples (random performance variations)
samples = generate_player_samples(session, n_samples=150)

# For each candidate:
for i in range(num_candidates):
    # Pick a random sample for score variation
    sample_idx = random.choice(sample_indices)
    base_scores = {pid: samples[pid][sample_idx] for pid in player_ids}
    
    # Add jitter for extra diversity
    jittered_scores = {pid: score * (1 + random.uniform(-0.05, 0.05))}
    
    # Optimize with this score set
    lineup = generate_single_classic_lineup(
        players=available_players,
        base_scores=jittered_scores,
        locked_assignments=locked_assignments
    )
```

This creates diverse yet optimized alternatives.

### 4. Simulation & Selection

After generating candidates, we:

1. **Simulate** each candidate's expected performance using Monte Carlo
2. **Calculate** ROI, win%, expected payout for each
3. **Select** the best candidate per entry (highest expected ROI)

## Usage

### Step 1: Export Entries from DraftKings

Export your contest entries to CSV. The expected format is DraftKings native export:

```csv
Entry ID,Contest Name,Contest ID,Entry Fee,PG,SG,SF,PF,C,G,F,UTIL
12345,NBA $3 Single,...,3.00,Luka Doncic (40618539) (LOCKED),Damian Lillard (40618540),...
```

Save to: `/home/sam-pratt/ivan/data/nba/input/entries.csv`

### Step 2: Set Lock Time

Determine when you want to make the swap (before late games lock). Format: `YYYY-MM-DD HH:MM`

Example: `2025-11-05 22:30` (10:30 PM)

### Step 3: Run the Script

```bash
cd /home/sam-pratt/ivan

python scripts/nba/swap.py \
  --lock-time "2025-11-05 22:30" \
  --num-candidates 50 \
  --num-sims 1000 \
  --output-csv lateswap_results.csv
```

**Optional Arguments:**
- `--entries-csv` - Path to entries CSV (default: `data/nba/input/entries.csv`)
- `--config` - Path to config file (default: `config/nba.json`)
- `--num-candidates` - Candidates per entry (default: 50)
- `--num-sims` - Monte Carlo simulations (default: 1000)

### Step 4: Review Results

The output CSV contains:

| Column | Description |
|--------|-------------|
| Entry ID | Your original DraftKings entry ID |
| Original Lineup ID | Database reference |
| Locked Players | Player IDs that can't change (games started) |
| Swapped Players | New optimized player IDs |
| Player 1-8 | Full lineup after swap |
| Mean FPTS | Expected fantasy points |
| Mean Payout | Expected payout amount |
| Mean ROI | Expected return on investment |
| Mean Rank | Expected finishing rank |

### Step 5: Execute Swaps on DraftKings

1. Log into DraftKings
2. Navigate to "My Contests"
3. For each entry in the CSV, manually swap the players listed in "Swapped Players"

*Note: DraftKings doesn't provide an API for automated swaps, so this must be done manually.*

## Configuration

The swap logic uses your existing `config/nba.json`:

```json
{
  "min_salary": 48000,
  "max_salary": 50000,
  "randomness": 0.25,
  "solver": {
    "glpk_path": "/path/to/glpsol"
  }
}
```

All lineup generation settings apply to late swap optimization.

## Example Workflow

```bash
# 1. Make sure you have entries CSV
ls data/nba/input/entries.csv

# 2. Check current time and games
# Games at 7:00 PM have locked, games at 10:00 PM still open

# 3. Run swap optimization for 10:00 PM lock time
python scripts/nba/swap.py \
  --lock-time "2025-11-05 22:00" \
  --num-candidates 100 \
  --num-sims 5000

# 4. Review results
cat lateswap_results.csv

# 5. Execute swaps on DraftKings before 10:00 PM
```

## Technical Details

### How Locked Assignments Work

The `generate_single_classic_lineup()` function accepts `locked_assignments: Dict[int, str]`:

```python
locked_assignments = {
    player_id: slot_name,
    123456: "PG",
    789012: "C"
}
```

When provided, the optimizer:
1. Creates constraint: `x[player_id][slot_name] == 1` (forces assignment)
2. Ensures player is eligible for that slot (raises error if not)
3. Optimizes remaining slots normally

This is implemented in `lineup_generation.py` lines ~175-185:

```python
if locked_player_id is not None:
    if locked_player_id in x and slot_name in x[locked_player_id]:
        prob += x[locked_player_id][slot_name] == 1, f"LockedSlot_{slot_name}"
    else:
        raise ValueError(f"Locked player {locked_player_id} not eligible for {slot_name}")
```

### Position Coercion

The `_coerce_positions()` utility ensures position data is always a tuple:

```python
from ivan.sports.nba.utils import _coerce_positions

positions = _coerce_positions("PG/SG")  # → ("PG", "SG")
positions = _coerce_positions(["PG"])    # → ("PG",)
```

This prevents type mismatches between string/list/tuple position formats.

### Game ID Creation

To enforce multi-game constraints, we create canonical game IDs:

```python
# Create game_id from team matchups (alphabetically sorted)
if team and opponent:
    game_id = "-".join(sorted([team.upper(), opponent.upper()]))
    # Example: "BOS" vs "LAL" → "BOS-LAL"
```

This ensures both teams in the same game share the same `game_id`.

## Troubleshooting

### "No available players for swapping"

**Cause:** All games have already started (past lock_time)

**Solution:** Check your `--lock-time` argument. It should be BEFORE the late games start.

### "Could not assign any player to slot X"

**Cause:** Position eligibility issue in slot mapping

**Solution:** Verify player positions in database are correct. Check that `NBAPlayer.position` contains valid values (PG, SG, SF, PF, C).

### "Player ID not in database"

**Cause:** Entries CSV references players not in your database

**Solution:** Run `python scripts/nba/ingest.py` to refresh player data from DraftKings API.

### Simulation is very slow

**Cause:** Too many candidates or simulations

**Solution:** Reduce `--num-candidates` (try 20-30) or `--num-sims` (try 500-1000 for quick runs).

### Different results each run

**Cause:** Monte Carlo randomness in candidate generation and simulation

**Solution:** This is expected behavior. The randomness creates diversity. If you need reproducibility, you can add a seed parameter (would require code modification).

## Best Practices

1. **Run early** - Generate candidates 15-30 minutes before lock time to review results
2. **Multiple candidates** - Use at least 50 candidates for good diversity
3. **Sufficient simulations** - Use 1000+ simulations for stable ROI estimates
4. **Verify manually** - Always review the recommended swaps before executing on DraftKings
5. **Check salary** - Ensure swapped lineups stay under salary cap (enforced by optimizer but good to verify)
6. **Test first** - Try with a small contest before using on large-field tournaments

## Future Enhancements

Potential improvements to consider:

- [ ] Auto-submit to DraftKings via unofficial API
- [ ] Real-time odds integration for better projections
- [ ] Injury/news alerts before lock time
- [ ] Multi-entry correlation optimization
- [ ] Historical swap performance tracking
- [ ] Web UI for easier swap review

---

**Questions?** Check the inline code comments in `swap_logic.py` or refer to the project's Copilot instructions.
