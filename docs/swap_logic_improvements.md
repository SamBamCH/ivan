# Late Swap Output Improvements - Summary

## Changes Made

### 1. **Missing Player Report** (`swap_logic.py`)

Added `_print_missing_player_report()` function that provides a concise, formatted report of players using fallback projections:

**Features:**
- Shows player name, ID, mean/std values used
- Displays frequency: how many times each player appears across all candidate lineups
- Sorts by frequency (most common first)
- Clean table format with clear headers and separators

**Example Output:**
```
⚠ Missing Player Report (5 players using fallback projections)
──────────────────────────────────────────────────────────────────────────────────────
Player                         ID         Mean     Std      Frequency   
──────────────────────────────────────────────────────────────────────────────────────
LeBron James                   1234       45.30    8.50      125/150  (83.3%)
Stephen Curry                  5678       42.10    7.20       98/150  (65.3%)
...
──────────────────────────────────────────────────────────────────────────────────────
Note: These players were assigned projections from ProjectionSnapshot or used default values
```

### 2. **Ownership Tracking** (`swap_logic.py`)

Added `_calculate_and_write_ownership()` function that:

**For Unlocked Players (games not started):**
- Calculates ownership % across all swap candidates
- Writes to `NBAPlayer.ownership` column
- Commits to database after all updates

**For Locked Players (games already started):**
- Calculates ownership % across all swap candidates  
- Writes to `ProjectionSnapshot.ownership` column
- Uses snapshot from the lock date (same day)
- Gracefully handles if ownership column doesn't exist

**Output:**
```
✓ Updated ownership for 45 unlocked players in NBAPlayer table
✓ Updated ownership for 12 locked players in ProjectionSnapshot table
```

### 3. **Database Schema Update** (`models.py`)

Added `ownership` column to `ProjectionSnapshot` model:
```python
ownership = Column(Float, nullable=True, default=0.0)  # Projected ownership %
```

This allows tracking ownership for locked players whose data is preserved in snapshots.

### 4. **Migration Script** (`scripts/nba/add_ownership_to_snapshot.py`)

Created database migration script to add the ownership column:

**Usage:**
```bash
python scripts/nba/add_ownership_to_snapshot.py
```

**Features:**
- Checks if column already exists (idempotent)
- Uses raw SQL for schema modification
- Clear console output with Rich formatting
- Safe and reversible

### 5. **Enhanced `_build_player_samples_matrix()`** (`swap_logic.py`)

Modified to return additional information:
- **Before:** `(P, ns)`
- **After:** `(P, ns, missing_players)`

Where `missing_players` is a dict mapping player_id → (mean, std) for all players that used fallback projections.

## Integration Points

The new functions are called in `lateswap_one_contest()` at Step 5b:

```python
# Generate FPTS samples
P, num_sims_actual, missing_players = _build_player_samples_matrix(...)

# Print missing player report
_print_missing_player_report(session, missing_players, all_candidates)

# Calculate and write ownership to database
_calculate_and_write_ownership(session, all_candidates, lock_time)
```

## Benefits

1. **Cleaner Output:** Single consolidated report instead of individual warnings
2. **Better Visibility:** See which players are most frequently affected by missing projections
3. **Ownership Tracking:** Capture actual field composition for both locked and unlocked players
4. **Analytics:** Ownership data can be used for future analysis and optimization
5. **Backward Compatible:** Gracefully handles missing ownership column in ProjectionSnapshot

## Migration Steps

To use these changes:

1. **Run the migration:**
   ```bash
   python scripts/nba/add_ownership_to_snapshot.py
   ```

2. **Run swap as normal:**
   ```bash
   python scripts/nba/swap.py --entries entries.csv --lock-time "2025-11-07 19:00:00"
   ```

3. **Review output:**
   - Missing player report shows at Step 5
   - Ownership updates show at Step 5b
   - All data persisted to database

## Notes

- Ownership is calculated as: `(appearances in candidates / total candidates) * 100`
- Locked player snapshots are matched by date (same calendar day as lock_time)
- If ProjectionSnapshot lacks ownership column, script warns but continues
- All changes are backward compatible with existing code
