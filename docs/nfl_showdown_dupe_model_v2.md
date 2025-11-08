# NFL Showdown Duplication Model v2.0
## Simplified Historical Data-Driven Architecture

**Author**: GitHub Copilot  
**Date**: November 6, 2025  
**Status**: Design Document

---

## 🎯 Design Goals

1. **Simplicity First**: Minimal features, maximum interpretability
2. **Purely Historical**: Train directly from past contest results
3. **Two-Stage Structure**: Keep contest-level vs lineup-level separation (this works!)
4. **Fast Inference**: Predict duplicates in milliseconds, not seconds
5. **Easy to Debug**: Every feature should be explainable and verifiable

---

## 📊 What We Learned from v1 (Current System)

### ✅ What Works
- **Two-stage hierarchy** (contest distribution → lineup positioning) is conceptually sound
- Contest-level features (field size, ownership concentration) do predict duplicate distributions
- Using quantile interpolation works well for combining stages
- Historical contest data is rich and available

### ❌ What's Overcomplicated
- **Too many lineup features** (14+ features including correlation_score, stack_pattern, game_stack_type, etc.)
- **Unclear feature engineering** - What is `correlation_score`? How is it calculated?
- **Duplicate database models** - ShowdownLineup has unused columns like `qb_wr_stack`, `rb_dst_negative`
- **Fragile preprocessing** - Heavy reliance on external normalization (percentiles, gini, etc.)
- **Hard to validate** - Can't easily verify if features match reality

---

## 🏗️ Proposed Architecture v2.0

### Core Principle: **Histogram Matching**

Instead of trying to predict *exact* duplicate counts, we:
1. Learn what **distribution shapes** look like for different contest types
2. Map each lineup to a **percentile** in that distribution
3. Sample from the learned distribution

This is simpler, more robust, and matches what actually happens in DFS.

---

## 📐 Stage 1: Contest Distribution Model

### Inputs (Contest-Level Features)
```python
STAGE1_FEATURES = [
    'log_field_size',      # log10(total_entries)
    'entry_fee_tier',      # Categorical: [1, 3, 5, 10, 20, 50, 100+]
    'ownership_gini',      # 0-1, measures ownership concentration
    'num_games_tier',      # Showdown = 1 game always, but keep for future multi-game
]
```

**Rationale**:
- `log_field_size`: Small fields (100 entries) behave very differently than large fields (100K entries)
- `entry_fee_tier`: $1 contests have different duplicate patterns than $100 contests (multi-entry behavior)
- `ownership_gini`: High concentration = more duplicates on chalk lineups
- `num_games_tier`: For future extension to main slate (more games = more diversity)

### Outputs (Distribution Quantiles)
```python
STAGE1_TARGETS = [
    'q50',  # Median duplicate count
    'q75',  # 75th percentile
    'q90',  # 90th percentile
    'q95',  # 95th percentile
    'q99',  # 99th percentile
]
```

**Why these quantiles?**
- Most lineups (50-75%) have low duplicate counts (1-5)
- The interesting action is in the tail (top 10%)
- We care most about high-duplicate lineups (those are the field chalk)

### Model Type
```python
# Multi-output Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

stage1_model = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,          # Shallow trees = better generalization
        learning_rate=0.1,
        min_samples_leaf=10,   # Prevent overfitting to rare contests
    )
)
```

---

## 📊 Stage 2: Lineup Positioning Model

### Inputs (Lineup-Level Features)
```python
STAGE2_FEATURES = [
    'cpt_ownership',           # Captain's ownership % (0-100)
    'flex_ownership_mean',     # Mean ownership of 5 flex players
    'flex_ownership_std',      # Std dev of flex ownership (diversity)
    'salary_remaining_pct',    # % of $50K cap unused (0-100)
]
```

**Rationale**:
- `cpt_ownership`: Captain is 1.5x multiplier - most important player
- `flex_ownership_mean`: Average chalkiness of the supporting cast
- `flex_ownership_std`: High std = contrarian mix; low std = all chalk or all fade
- `salary_remaining_pct`: Leftover salary often indicates suboptimal builds

**NO MORE**:
- ❌ Correlation scores (too complex, hard to validate)
- ❌ Stack patterns (QB-WR-WR is implicitly captured by ownership)
- ❌ Game stack types (not worth the engineering cost)
- ❌ Percentile normalization within contest (chicken-and-egg problem)

### Output (Lineup Score)
```python
# Predict a single "duplication propensity score" in [0, 1]
# Higher score = lineup is more likely to be duplicated
# This score is then converted to a percentile rank within the candidate pool
```

### Model Type
```python
# Gradient Boosting Regressor (logit target)
from scipy.special import logit, expit

stage2_model = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
)

# Target transformation: logit(percentile_rank)
# This ensures output stays in (0, 1) range
```

---

## 🔄 Inference Pipeline

### Step-by-Step Process

```python
def predict_duplicate_counts(
    session: Session,
    candidate_lineups: List[Dict],  # Each dict: {cpt_id, flex_ids}
    contest_params: Dict,            # {total_entries, entry_fee, ownership_gini}
) -> np.ndarray:
    """
    Returns predicted duplicate counts for each lineup.
    """
    
    # ========================================
    # STAGE 1: Predict Contest Distribution
    # ========================================
    
    # Build contest feature vector
    X_contest = np.array([[
        np.log10(contest_params['total_entries']),
        map_entry_fee_to_tier(contest_params['entry_fee']),
        contest_params['ownership_gini'],
        1.0,  # num_games (always 1 for showdown)
    ]])
    
    # Predict quantiles [q50, q75, q90, q95, q99]
    quantiles = stage1_model.predict(X_contest)[0]
    
    # Build full CDF by adding boundary points
    quantile_levels = np.array([0.0, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    quantile_values = np.array([
        1.0,                    # Minimum: everyone appears at least once
        quantiles[0],           # q50
        quantiles[1],           # q75
        quantiles[2],           # q90
        quantiles[3],           # q95
        quantiles[4],           # q99
        contest_params['total_entries'] * 0.02,  # Max cap: 2% of field
    ])
    
    # ========================================
    # STAGE 2: Compute Lineup Scores
    # ========================================
    
    # Extract features for each lineup
    X_lineups = []
    for lineup in candidate_lineups:
        cpt_own = get_player_ownership(lineup['cpt_id'])
        flex_owns = [get_player_ownership(pid) for pid in lineup['flex_ids']]
        
        X_lineups.append([
            cpt_own,
            np.mean(flex_owns),
            np.std(flex_owns),
            lineup['salary_remaining_pct'],
        ])
    
    X_lineups = np.array(X_lineups)
    
    # Predict duplication scores (in logit space)
    scores_logit = stage2_model.predict(X_lineups)
    scores = expit(scores_logit)  # Transform to [0, 1]
    
    # ========================================
    # COMBINE: Map Scores to Duplicates
    # ========================================
    
    # Convert scores to percentile ranks relative to candidate pool
    percentiles = rankdata(scores, method='average') / len(scores)
    
    # Interpolate into Stage 1's quantile distribution
    duplicates = np.interp(
        percentiles,
        quantile_levels,
        quantile_values,
    )
    
    return duplicates
```

---

## 🗃️ Data Requirements

### Training Data Schema

We already have most of this in `ShowdownLineup` and `ShowdownContestMetadata`!

#### Required Tables
1. **ShowdownContestMetadata** (contest-level)
   - ✅ `total_entries`
   - ✅ `entry_fee`
   - ✅ `ownership_gini`
   - ✅ `unique_lineup_count` (to compute field density)
   
2. **ShowdownLineup** (lineup-level)
   - ✅ `duplicate_count` (TARGET)
   - ⚠️ Need to compute: `cpt_ownership`, `flex_ownership_mean`, `flex_ownership_std`
   - ✅ `salary_remaining_pct`
   
3. **ShowdownPlayer** (player-level)
   - ✅ `ownership_pct` (historical actual ownership)
   - ✅ `roster_slot` (CPT vs FLEX)

### Data Cleaning Pipeline

```python
def build_training_dataset(session: Session, min_entries: int = 5000):
    """
    Build clean training dataset from historical contests.
    """
    
    # Step 1: Get all contests with sufficient size
    contests = (
        session.query(ShowdownContestMetadata)
        .filter(ShowdownContestMetadata.total_entries >= min_entries)
        .all()
    )
    
    records = []
    
    for contest in contests:
        # Step 2: Get all lineups for this contest
        lineups = (
            session.query(ShowdownLineup)
            .filter(ShowdownLineup.contest_id == contest.contest_id)
            .all()
        )
        
        # Step 3: Build player ownership lookup for this contest
        players = (
            session.query(ShowdownPlayer)
            .filter(ShowdownPlayer.contest_id == contest.contest_id)
            .all()
        )
        
        ownership_map = {
            (p.normalized_name, p.roster_slot): p.ownership_pct
            for p in players
        }
        
        # Step 4: Compute features for each lineup
        for lineup in lineups:
            cpt_name = normalize_name(lineup.captain_name)
            flex_names = [normalize_name(n) for n in lineup.flex_players]
            
            cpt_own = ownership_map.get((cpt_name, 'CPT'), 0.0)
            flex_owns = [
                ownership_map.get((name, 'FLEX'), 0.0)
                for name in flex_names
            ]
            
            # Contest features
            log_field_size = np.log10(contest.total_entries)
            entry_fee_tier = map_entry_fee_to_tier(contest.entry_fee)
            ownership_gini = contest.ownership_gini or 0.5
            
            # Lineup features
            cpt_ownership = cpt_own
            flex_ownership_mean = np.mean(flex_owns)
            flex_ownership_std = np.std(flex_owns)
            salary_remaining_pct = lineup.salary_remaining_pct
            
            # Target
            duplicate_count = lineup.duplicate_count
            
            records.append({
                'contest_id': contest.contest_id,
                'lineup_id': lineup.id,
                # Stage 1 features
                'log_field_size': log_field_size,
                'entry_fee_tier': entry_fee_tier,
                'ownership_gini': ownership_gini,
                'num_games_tier': 1.0,
                # Stage 2 features
                'cpt_ownership': cpt_ownership,
                'flex_ownership_mean': flex_ownership_mean,
                'flex_ownership_std': flex_ownership_std,
                'salary_remaining_pct': salary_remaining_pct,
                # Target
                'duplicate_count': duplicate_count,
            })
    
    df = pd.DataFrame(records)
    
    # Compute percentile ranks within each contest (for Stage 2 target)
    df['percentile_rank'] = df.groupby('contest_id')['duplicate_count'].rank(pct=True)
    
    return df
```

---

## 🎓 Training Procedure

### Stage 1: Contest Distribution Model

```python
def train_stage1(df: pd.DataFrame):
    """Train contest-level quantile predictor."""
    
    # Aggregate to contest level
    contest_df = df.groupby('contest_id').agg({
        'log_field_size': 'first',
        'entry_fee_tier': 'first',
        'ownership_gini': 'first',
        'num_games_tier': 'first',
        'duplicate_count': ['median', 
                           lambda x: np.percentile(x, 75),
                           lambda x: np.percentile(x, 90),
                           lambda x: np.percentile(x, 95),
                           lambda x: np.percentile(x, 99)],
    }).reset_index()
    
    # Flatten column names
    contest_df.columns = ['contest_id', 'log_field_size', 'entry_fee_tier',
                          'ownership_gini', 'num_games_tier',
                          'q50', 'q75', 'q90', 'q95', 'q99']
    
    # Features and targets
    X = contest_df[['log_field_size', 'entry_fee_tier', 'ownership_gini', 'num_games_tier']]
    Y = contest_df[['q50', 'q75', 'q90', 'q95', 'q99']]
    
    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )
    )
    
    model.fit(X_train_scaled, Y_train)
    
    # Evaluate
    Y_pred = model.predict(X_test_scaled)
    for i, q in enumerate(['q50', 'q75', 'q90', 'q95', 'q99']):
        mae = mean_absolute_error(Y_test.iloc[:, i], Y_pred[:, i])
        print(f"{q} MAE: {mae:.2f} duplicates")
    
    return model, scaler
```

### Stage 2: Lineup Positioning Model

```python
def train_stage2(df: pd.DataFrame):
    """Train lineup-level percentile predictor."""
    
    # Features and target
    X = df[['cpt_ownership', 'flex_ownership_mean', 'flex_ownership_std', 'salary_remaining_pct']]
    
    # Target: logit-transformed percentile rank
    y_percentile = df['percentile_rank'].clip(0.001, 0.999)  # Avoid division by zero
    y_logit = logit(y_percentile)
    
    # Train/test split (stratified by contest to avoid leakage)
    train_contests = df['contest_id'].sample(frac=0.8, random_state=42)
    train_mask = df['contest_id'].isin(train_contests)
    
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y_logit[train_mask]
    y_test = y_logit[~train_mask]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_logit = model.predict(X_test_scaled)
    y_pred_percentile = expit(y_pred_logit)
    y_test_percentile = expit(y_test)
    
    mae = mean_absolute_error(y_test_percentile, y_pred_percentile)
    print(f"Percentile MAE: {mae:.3f} (0-1 scale)")
    
    return model, scaler
```

---

## 🚀 Implementation Plan

### Phase 1: Data Pipeline (Week 1)
- [ ] Create `scripts/nfl/build_training_data_v2.py`
  - Query historical contests (min 5000 entries)
  - Compute all 8 features (4 contest + 4 lineup)
  - Export to `data/nfl/training/showdown_dupe_training_v2.parquet`
- [ ] Validate data quality
  - Check for missing values
  - Verify ownership percentages are reasonable
  - Inspect duplicate count distribution

### Phase 2: Model Training (Week 1)
- [ ] Create `scripts/nfl/train_showdown_dupe_v2.py`
  - Train Stage 1 (contest quantile model)
  - Train Stage 2 (lineup percentile model)
  - Save models to `data/nfl/models/showdown_dupe_v2/`
- [ ] Model evaluation notebook
  - Compare predictions vs actuals on held-out contests
  - Analyze error patterns (underestimate vs overestimate)

### Phase 3: Integration (Week 2)
- [ ] Update `src/ivan/sports/nfl/sample_field.py`
  - Replace existing `predict_lineups_with_twostage()` with v2
  - Keep same function signature for backward compatibility
  - Add flag to switch between v1 and v2
- [ ] Test on upcoming slate
  - Generate field samples with v2 model
  - Compare ownership projections with v1
  - Validate field diversity metrics

### Phase 4: Validation (Week 2-3)
- [ ] Wait for contest results
- [ ] Compare predicted vs actual duplicates
- [ ] Iterate on feature engineering if needed
- [ ] Deprecate v1 if v2 performs better

---

## 📊 Success Metrics

### Quantitative
1. **Stage 1 Accuracy**: MAE < 5 duplicates for q90, q95, q99
2. **Stage 2 Accuracy**: MAE < 0.05 on percentile scale (0-1)
3. **Combined Accuracy**: Correlation > 0.7 between predicted and actual duplicates
4. **Runtime**: < 1 second to predict 10,000 candidate lineups

### Qualitative
1. **Interpretability**: Can explain why a lineup has high/low duplicates
2. **Robustness**: Works across different contest sizes and entry fees
3. **Maintainability**: Easy to retrain with new data
4. **Debuggability**: Can trace back feature values for any lineup

---

## 🔧 Configuration

### Model Hyperparameters

```python
# config/nfl.json
{
  "showdown_dupe_model_v2": {
    "version": "2.0",
    "min_training_contest_size": 5000,
    "stage1": {
      "n_estimators": 100,
      "max_depth": 4,
      "learning_rate": 0.1,
      "min_samples_leaf": 10
    },
    "stage2": {
      "n_estimators": 150,
      "max_depth": 5,
      "learning_rate": 0.05,
      "subsample": 0.8
    },
    "inference": {
      "max_duplicate_pct": 0.02,  # Cap at 2% of field
      "min_duplicate_count": 1.0,
      "smooth_percentiles": true,
      "smoothing_sigma": 0.1
    }
  }
}
```

---

## 🎯 Appendix: Feature Justification

### Why These 8 Features?

| Feature | Type | Rationale | Data Source |
|---------|------|-----------|-------------|
| `log_field_size` | Contest | Duplicate behavior scales logarithmically with field size | `ShowdownContestMetadata.total_entries` |
| `entry_fee_tier` | Contest | Higher stakes = different multi-entry patterns | `ShowdownContestMetadata.entry_fee` |
| `ownership_gini` | Contest | Concentrated ownership = more duplicates | `ShowdownContestMetadata.ownership_gini` |
| `num_games_tier` | Contest | Future-proofing (currently always 1 for showdown) | Hardcoded to 1.0 |
| `cpt_ownership` | Lineup | Captain is 1.5x multiplier = most impactful | `ShowdownPlayer.ownership_pct` where `roster_slot='CPT'` |
| `flex_ownership_mean` | Lineup | Average chalkiness of supporting cast | Mean of `ShowdownPlayer.ownership_pct` where `roster_slot='FLEX'` |
| `flex_ownership_std` | Lineup | Variance in chalk/contrarian mix | Std dev of flex ownerships |
| `salary_remaining_pct` | Lineup | Leftover salary = suboptimal lineup | `ShowdownLineup.salary_remaining_pct` |

### What We're NOT Using (and Why)

| Feature (v1) | Why Removed |
|-------------|-------------|
| `correlation_score` | Undefined calculation, hard to validate |
| `stack_pattern` | Implicitly captured by ownership; adds complexity |
| `game_stack_type` | Minimal predictive power for duplicates |
| `qb_wr_stack` | Same as above - ownership already captures this |
| `projection_normalized` | Ownership is sufficient; projection adds noise |
| `ownership_percentile` | Chicken-and-egg problem - requires knowing contest distribution first |

---

## 🚨 Known Limitations

1. **Cold Start Problem**: New player (no historical ownership) requires fallback
2. **Multi-Entry Modeling**: Doesn't explicitly model user multi-entry strategies
3. **Time Sensitivity**: Pre-lock vs late-swap ownership may differ
4. **Correlation Assumption**: Assumes ownership independence (not always true)

### Mitigation Strategies

1. **Cold Start**: Use position-based average ownership as fallback
2. **Multi-Entry**: Entry fee tier captures some of this behavior
3. **Time Sensitivity**: Train on lock-time snapshots only
4. **Correlation**: Stage 2 model learns implicit correlations through data

---

## ✅ Summary

**Goal**: Predict duplicate counts for NFL Showdown lineups

**Approach**: Two-stage hierarchical model
- Stage 1: Contest distribution quantiles (4 features → 5 quantiles)
- Stage 2: Lineup percentile rank (4 features → 1 score)

**Key Improvements over v1**:
- ✂️ Fewer features (8 vs 14+)
- 📊 Direct historical data usage (no synthetic normalization)
- 🔍 Fully interpretable (every feature has clear meaning)
- ⚡ Faster inference (simpler feature computation)
- 🛠️ Easier to maintain (less preprocessing logic)

**Next Steps**: Implement Phase 1 (data pipeline) and validate data quality
