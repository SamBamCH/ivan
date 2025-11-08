"""
NFL Showdown Duplication Model v2 - Visual Flow Diagram
ASCII art representation of the two-stage architecture
"""

# ============================================================================
# TRAINING FLOW
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                          HISTORICAL CONTESTS                             │
│  (ShowdownContestMetadata, ShowdownLineup, ShowdownPlayer tables)       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ Query & Join
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING DATA PIPELINE                           │
│                    (build_training_data_v2.py)                          │
│                                                                          │
│  For each contest:                                                       │
│    - Extract contest features (4):                                       │
│      • log_field_size = log10(total_entries)                           │
│      • entry_fee_tier = bucket(entry_fee)                              │
│      • ownership_gini = gini(all_player_ownerships)                    │
│      • num_games_tier = 1.0 (showdown)                                 │
│                                                                          │
│  For each lineup in contest:                                             │
│    - Extract lineup features (4):                                        │
│      • cpt_ownership = ownership[captain, CPT]                         │
│      • flex_ownership_mean = mean([ownership[p, FLEX] for p in flex]) │
│      • flex_ownership_std = std([ownership[p, FLEX] for p in flex])   │
│      • salary_remaining_pct = (50000 - total_salary) / 50000 * 100    │
│                                                                          │
│    - Extract target:                                                     │
│      • duplicate_count (from ShowdownLineup)                           │
│      • percentile_rank = rank_within_contest(duplicate_count)          │
│                                                                          │
│  Output: DataFrame with ~50-100K rows                                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ Save to Parquet
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              showdown_dupe_v2_training.parquet                          │
│                                                                          │
│  Columns:                                                                │
│    contest_id, lineup_id                                                │
│    log_field_size, entry_fee_tier, ownership_gini, num_games_tier      │
│    cpt_ownership, flex_ownership_mean, flex_ownership_std,              │
│    salary_remaining_pct                                                 │
│    duplicate_count, percentile_rank                                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ Load & Split
                                 ▼
        ┌────────────────────────┴────────────────────────┐
        │                                                  │
        ▼                                                  ▼
┌──────────────────────────┐                 ┌──────────────────────────┐
│   STAGE 1 TRAINING       │                 │   STAGE 2 TRAINING       │
│                          │                 │                          │
│  Group by contest_id     │                 │  Use all lineup rows     │
│  Compute quantiles:      │                 │                          │
│    q50, q75, q90,        │                 │  Input (4 features):     │
│    q95, q99              │                 │    - cpt_ownership       │
│                          │                 │    - flex_ownership_mean │
│  Input (4 features):     │                 │    - flex_ownership_std  │
│    - log_field_size      │                 │    - salary_remaining_pct│
│    - entry_fee_tier      │                 │                          │
│    - ownership_gini      │                 │  Target:                 │
│    - num_games_tier      │                 │    logit(percentile_rank)│
│                          │                 │                          │
│  Target (5 outputs):     │                 │  Model:                  │
│    - [q50, q75, q90,     │                 │    GradientBoosting      │
│       q95, q99]          │                 │    n_est=150, depth=5    │
│                          │                 │                          │
│  Model:                  │                 │  Output:                 │
│    MultiOutputRegressor  │                 │    Duplication score     │
│    (GradientBoosting)    │                 │    (logit space)         │
│    n_est=100, depth=4    │                 │                          │
│                          │                 │                          │
│  80/20 train/test split  │                 │  80/20 split by contest  │
│                          │                 │  (avoid leakage)         │
└──────────┬───────────────┘                 └──────────┬───────────────┘
           │                                            │
           │ Save                                       │ Save
           ▼                                            ▼
┌──────────────────────────┐                 ┌──────────────────────────┐
│  stage1_model.pkl        │                 │  stage2_model.pkl        │
│  stage1_scaler.pkl       │                 │  stage2_scaler.pkl       │
└──────────────────────────┘                 └──────────────────────────┘
"""

# ============================================================================
# INFERENCE FLOW
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                        RUNTIME: NEW SLATE                                │
│                                                                          │
│  Inputs:                                                                 │
│    - Candidate lineups (from lineup generation)                         │
│    - Player ownership projections                                       │
│    - Contest parameters (field_size, entry_fee)                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LOAD TRAINED MODELS                                   │
│  - stage1_model.pkl + stage1_scaler.pkl                                 │
│  - stage2_model.pkl + stage2_scaler.pkl                                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────┴────────────────────────┐
        │                                                  │
        ▼                                                  ▼
┌──────────────────────────┐                 ┌──────────────────────────┐
│   STAGE 1: PREDICT       │                 │   STAGE 2: PREDICT       │
│   CONTEST DISTRIBUTION   │                 │   LINEUP SCORES          │
│                          │                 │                          │
│  Extract contest features│                 │  For each lineup:        │
│    log_field_size        │                 │    Extract features:     │
│    entry_fee_tier        │                 │      cpt_ownership       │
│    ownership_gini        │                 │      flex_ownership_mean │
│    num_games_tier        │                 │      flex_ownership_std  │
│                          │                 │      salary_remaining_pct│
│  Scale with scaler       │                 │                          │
│                          │                 │  Scale with scaler       │
│  Predict quantiles:      │                 │                          │
│    [q50, q75, q90,       │                 │  Predict scores (logit)  │
│     q95, q99]            │                 │                          │
│                          │                 │  Transform with expit    │
│  Add boundaries:         │                 │    → scores in [0, 1]    │
│    [0.0, 0.50, 0.75,     │                 │                          │
│     0.90, 0.95, 0.99,    │                 │  Rank scores to get      │
│     1.0]                 │                 │    percentiles           │
│                          │                 │                          │
│  Example output:         │                 │  Example output:         │
│    [1, 2, 5, 12, 25,     │                 │    [0.23, 0.45, 0.87,    │
│     78, 156]             │                 │     0.12, 0.99, ...]     │
│    duplicates            │                 │    percentile ranks      │
└──────────┬───────────────┘                 └──────────┬───────────────┘
           │                                            │
           │                                            │
           └──────────────────┬─────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       INTERPOLATION                                      │
│                                                                          │
│  For each lineup:                                                        │
│    percentile = lineup_percentiles[i]        (from Stage 2)             │
│    quantile_levels = [0.0, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]         │
│    quantile_values = [1, 2, 5, 12, 25, 78, 156]  (from Stage 1)       │
│                                                                          │
│    predicted_dupes[i] = np.interp(                                      │
│        percentile,                                                       │
│        quantile_levels,                                                  │
│        quantile_values                                                   │
│    )                                                                     │
│                                                                          │
│  Example:                                                                │
│    Lineup with percentile = 0.85                                        │
│    → Interpolate between q75 (5 dupes) and q90 (12 dupes)              │
│    → 0.85 is 2/3 of the way from 0.75 to 0.90                          │
│    → Prediction: 5 + (12-5) * 2/3 = 9.67 ≈ 10 duplicates               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PREDICTED DUPLICATE COUNTS                            │
│                                                                          │
│  Array of shape (n_lineups,) with predicted duplicates                  │
│  Range: [1.0, max_duplicates]                                           │
│  Example:                                                                │
│    [2.3, 5.1, 78.4, 1.2, 45.6, ...]                                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FIELD SAMPLING                                      │
│                                                                          │
│  Use predicted duplicates as λ (lambda) for Poisson sampling            │
│  Sample field according to these probabilities                          │
│  Return: {lineup_id: duplicate_count} dictionary                        │
└─────────────────────────────────────────────────────────────────────────┘
"""

# ============================================================================
# FEATURE FLOW DETAIL
# ============================================================================

"""
CONTEST-LEVEL FEATURES (Stage 1)
─────────────────────────────────

Database → Extract → Transform → Model Input
───────────────────────────────────────────

total_entries    →  log10()  →  log_field_size
(e.g., 50000)       (4.699)

entry_fee        →  bucket() →  entry_fee_tier
(e.g., 5.0)         (5.0)

ownership_pct[]  →  gini()   →  ownership_gini
(all players)       (0.45)

num_games        →  float()  →  num_games_tier
(1 for showdown)    (1.0)

────────────────────────────────────────────
Vector: [4.699, 5.0, 0.45, 1.0]
→ StandardScaler
→ Stage 1 Model
→ Output: [2, 5, 12, 25, 78] (quantiles)


LINEUP-LEVEL FEATURES (Stage 2)
────────────────────────────────

Database → Extract → Calculate → Model Input
─────────────────────────────────────────────

captain_id       →  lookup   →  cpt_ownership
+ roster_slot       ownership    (e.g., 42.3%)
= CPT               map

flex_ids[]       →  lookup   →  flex_ownership_mean
+ roster_slot       ownership    mean([25, 18, 30, 15, 22])
= FLEX              map          (22.0%)

flex_ids[]       →  lookup   →  flex_ownership_std
+ roster_slot       ownership    std([25, 18, 30, 15, 22])
= FLEX              map          (5.8%)

total_salary     →  calc     →  salary_remaining_pct
(e.g., 48500)       remain%      (1500 / 50000 * 100)
                                 (3.0%)

────────────────────────────────────────────
Vector: [42.3, 22.0, 5.8, 3.0]
→ StandardScaler
→ Stage 2 Model
→ Output: -1.23 (logit)
→ expit(-1.23) = 0.23 (score)
→ rank to percentile = 0.45
→ interpolate into Stage 1 quantiles
→ Final: 3.5 duplicates
"""

# ============================================================================
# EXAMPLE WALKTHROUGH
# ============================================================================

"""
CONCRETE EXAMPLE: Predicting duplicates for one lineup
───────────────────────────────────────────────────────

Contest Setup:
  - 50,000 total entries
  - $5 entry fee
  - Player ownerships: [45%, 30%, 25%, 20%, 15%, 10%, 8%, 5%, 3%, ...]
  - Ownership Gini: 0.45 (moderate concentration)

Lineup:
  - CPT: Travis Kelce (42.3% ownership)
  - FLEX: Patrick Mahomes (25%), Rashee Rice (18%), Xavier Worthy (30%), 
          Isiah Pacheco (15%), Noah Gray (22%)
  - Total salary: $48,500 (remaining: $1,500 = 3%)

STAGE 1: PREDICT CONTEST DISTRIBUTION
──────────────────────────────────────
Input features:
  log_field_size = log10(50000) = 4.699
  entry_fee_tier = 5.0
  ownership_gini = 0.45
  num_games_tier = 1.0

Model predicts quantiles:
  Q50: 2 duplicates    (50% of lineups have ≤2 dupes)
  Q75: 5 duplicates    (75% of lineups have ≤5 dupes)
  Q90: 12 duplicates   (90% of lineups have ≤12 dupes)
  Q95: 25 duplicates   (95% of lineups have ≤25 dupes)
  Q99: 78 duplicates   (99% of lineups have ≤78 dupes)

Build full CDF:
  Percentiles: [0.00, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
  Duplicates:  [1,    2,    5,    12,   25,   78,   1000]

STAGE 2: PREDICT LINEUP PERCENTILE
───────────────────────────────────
Input features:
  cpt_ownership = 42.3
  flex_ownership_mean = mean([25, 18, 30, 15, 22]) = 22.0
  flex_ownership_std = std([25, 18, 30, 15, 22]) = 5.8
  salary_remaining_pct = 3.0

Model predicts:
  score (logit) = 1.45
  score (prob) = expit(1.45) = 0.81

Relative to all candidate lineups, this scores at:
  percentile_rank = 0.85 (85th percentile)

INTERPOLATION: MAP PERCENTILE TO DUPLICATES
────────────────────────────────────────────
We have:
  lineup_percentile = 0.85
  
Interpolate into Stage 1's CDF:
  0.85 is between Q75 (0.75 → 5 dupes) and Q90 (0.90 → 12 dupes)
  
  Linear interpolation:
    ratio = (0.85 - 0.75) / (0.90 - 0.75) = 0.10 / 0.15 = 0.667
    predicted_dupes = 5 + (12 - 5) * 0.667 = 5 + 4.67 = 9.67
  
FINAL PREDICTION: ~10 duplicates

INTERPRETATION
──────────────
This lineup has:
  - High captain ownership (42%) → chalk
  - Moderate flex ownership (22% avg) → some chalk, some value
  - Low flex diversity (5.8% std) → all similar ownership levels
  - Very little salary remaining (3%) → optimized build
  
Result: This is a fairly chalky, well-optimized lineup
→ Predicted to be in the 85th percentile
→ ~10 duplicates expected in a 50K-entry contest
→ About 1 in 5000 of the field will use this lineup
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nVisual flow diagrams for NFL Showdown Duplication Model v2")
    print("=" * 70)
    print("\nSee the ASCII diagrams above for:")
    print("  1. Training flow (historical data → models)")
    print("  2. Inference flow (new slate → predictions)")
    print("  3. Feature extraction detail")
    print("  4. Concrete example walkthrough")
