# NFL Showdown Dupe Model v2 - Quick Reference Card

## 🎯 One-Sentence Summary
**Predict lineup duplicate counts using an 8-feature two-stage model that maps contest parameters to distribution quantiles, then positions each lineup within that distribution.**

---

## 📊 The 8 Features

### Stage 1: Contest Features (4)
```python
1. log_field_size      = log10(total_entries)         # Scale: 3-6 (1K-1M entries)
2. entry_fee_tier      = bucket(entry_fee)            # Values: [1, 3, 5, 10, 20, 50, 100]
3. ownership_gini      = gini_coefficient(ownerships) # Range: 0-1 (concentration)
4. num_games_tier      = 1.0 (showdown) or N (slate)  # Typically 1.0
```

### Stage 2: Lineup Features (4)
```python
5. cpt_ownership          = ownership_pct[captain_id]           # Range: 0-100
6. flex_ownership_mean    = mean(ownership_pct[flex_ids])       # Range: 0-100
7. flex_ownership_std     = std(ownership_pct[flex_ids])        # Range: 0-50
8. salary_remaining_pct   = (50000 - total_salary) / 50000 * 100  # Range: 0-100
```

---

## 🔄 Two-Stage Flow

```
Stage 1: Contest → Quantiles
    Input:  [log_field_size, entry_fee_tier, ownership_gini, num_games_tier]
    Model:  MultiOutputRegressor(GradientBoosting)
    Output: [Q50, Q75, Q90, Q95, Q99] duplicate quantiles

Stage 2: Lineup → Percentile
    Input:  [cpt_ownership, flex_own_mean, flex_own_std, salary_remaining_pct]
    Model:  GradientBoostingRegressor
    Output: Score → Percentile rank

Combine: Interpolate percentile into quantile distribution
    Result: Predicted duplicate count
```

---

## 📁 File Locations

### Models
```
data/nfl/models/showdown_dupe_v2/
├── stage1_model.pkl       # Contest quantile predictor
├── stage1_scaler.pkl      # Stage 1 feature scaler
├── stage2_model.pkl       # Lineup percentile predictor
├── stage2_scaler.pkl      # Stage 2 feature scaler
└── metadata.json          # Version, training date, metrics
```

### Code
```
src/ivan/sports/nfl/dupe_model_v2/
├── features.py            # Feature extraction functions
├── training.py            # Model training logic
└── inference.py           # Prediction pipeline

scripts/nfl/
├── build_training_data_v2.py   # Data pipeline
├── train_showdown_dupe_v2.py   # Train models
└── validate_dupe_model_v2.py   # Validation
```

### Documentation
```
docs/
├── nfl_showdown_dupe_v2_summary.md      # This file (overview)
├── nfl_showdown_dupe_model_v2.md        # Full architecture spec
├── nfl_showdown_dupe_comparison.md      # v1 vs v2 comparison
├── nfl_showdown_dupe_v2_roadmap.md      # Implementation plan
└── nfl_showdown_dupe_v2_visual_flow.py  # ASCII diagrams
```

---

## ⚡ Quick Start Commands

### Train Model
```bash
# 1. Build training data
python scripts/nfl/build_training_data_v2.py \
    --min-entries 5000 \
    --output data/nfl/training/showdown_dupe_v2_training.parquet

# 2. Train models
python scripts/nfl/train_showdown_dupe_v2.py \
    --input data/nfl/training/showdown_dupe_v2_training.parquet \
    --output data/nfl/models/showdown_dupe_v2/

# 3. Validate
python scripts/nfl/validate_dupe_model_v2.py \
    --model-dir data/nfl/models/showdown_dupe_v2/ \
    --test-data data/nfl/training/showdown_dupe_v2_test.parquet
```

### Use Model at Runtime
```python
from ivan.sports.nfl.dupe_model_v2.inference import load_models_v2, predict_duplicate_counts_v2

# Load models
models = load_models_v2("data/nfl/models/showdown_dupe_v2")

# Predict for candidate lineups
duplicates = predict_duplicate_counts_v2(
    candidate_lineups=[
        {'cpt_id': 123, 'flex_ids': [456, 789, ...], 'salary_remaining': 1500},
        # ... more lineups
    ],
    contest_params={
        'total_entries': 50000,
        'entry_fee': 5.0,
        'ownership_values': [45.0, 30.0, 25.0, ...],  # All player ownerships
    },
    ownership_map={123: 42.3, 456: 25.0, ...},  # player_id → ownership_pct
    models=models,
)

# Result: array([9.67, 2.34, 78.5, ...])  # duplicates per lineup
```

---

## 🎓 Training Data Schema

```python
# Parquet file: data/nfl/training/showdown_dupe_v2_training.parquet
columns = [
    # Identifiers
    'contest_id',
    'lineup_id',
    
    # Stage 1 features (contest-level)
    'log_field_size',
    'entry_fee_tier',
    'ownership_gini',
    'num_games_tier',
    
    # Stage 2 features (lineup-level)
    'cpt_ownership',
    'flex_ownership_mean',
    'flex_ownership_std',
    'salary_remaining_pct',
    
    # Targets
    'duplicate_count',      # Raw count (≥1)
    'percentile_rank',      # Within-contest rank (0-1)
]

# Example row:
{
    'contest_id': 12345,
    'lineup_id': 67890,
    'log_field_size': 4.699,          # log10(50000)
    'entry_fee_tier': 5.0,            # $5 entry
    'ownership_gini': 0.45,           # Moderate concentration
    'num_games_tier': 1.0,            # Showdown
    'cpt_ownership': 42.3,            # Travis Kelce 42.3%
    'flex_ownership_mean': 22.0,      # Average flex ownership
    'flex_ownership_std': 5.8,        # Low diversity
    'salary_remaining_pct': 3.0,      # $1500 remaining
    'duplicate_count': 8,             # This lineup appeared 8 times
    'percentile_rank': 0.82,          # 82nd percentile in contest
}
```

---

## 📈 Expected Performance

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Stage 1 MAE** | < 5 duplicates | Compare predicted vs actual quantiles on held-out contests |
| **Stage 2 MAE** | < 0.05 | Compare predicted vs actual percentile ranks |
| **Overall Correlation** | > 0.70 | Correlation between final predictions and actual duplicates |
| **Runtime** | < 1ms/lineup | Time to predict 10K lineups (excluding model load) |

---

## 🛠️ Configuration

```json
// config/nfl.json
{
  "dupe_model_version": "v2",  // "v1" or "v2" (feature flag)
  
  "showdown_dupe_model_v2": {
    "model_dir": "data/nfl/models/showdown_dupe_v2",
    
    "training": {
      "min_contest_size": 5000,
      "test_split": 0.2,
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
      }
    },
    
    "inference": {
      "min_duplicate_count": 1.0,
      "max_duplicate_pct": 0.02,
      "smooth_percentiles": false
    }
  }
}
```

---

## 🔍 Debugging Checklist

### If predictions seem off:

1. **Check feature values**
   ```python
   # Print features for a lineup
   print(f"CPT: {cpt_ownership}%")
   print(f"Flex mean: {flex_ownership_mean}%")
   print(f"Flex std: {flex_ownership_std}%")
   print(f"Salary left: {salary_remaining_pct}%")
   ```

2. **Check Stage 1 quantiles**
   ```python
   # Are quantiles reasonable?
   # q50 should be low (1-5), q99 should be high (20-200)
   print(f"Quantiles: {quantiles}")
   ```

3. **Check Stage 2 scores**
   ```python
   # Scores should be in [0, 1] range
   # Higher scores = more duplicates expected
   print(f"Min score: {scores.min():.3f}")
   print(f"Max score: {scores.max():.3f}")
   print(f"Mean score: {scores.mean():.3f}")
   ```

4. **Check percentile ranks**
   ```python
   # Should span [0, 1] range somewhat evenly
   plt.hist(percentiles, bins=20)
   plt.show()
   ```

5. **Compare with actuals (if available)**
   ```python
   from scipy.stats import spearmanr
   corr, pval = spearmanr(predictions, actuals)
   print(f"Correlation: {corr:.3f} (p={pval:.4f})")
   ```

---

## 🎯 Common Use Cases

### 1. Generate Field Sample
```python
from ivan.sports.nfl.sample_field import sample_showdown_field

counts_by_lineup, meta = sample_showdown_field(
    session=session,
    field_size=50000,
    config=config,
    contest_id=None,  # Will use v2 if config['dupe_model_version'] == 'v2'
)

# Result: {lineup_id: duplicate_count}
```

### 2. Project Ownership
```python
from ivan.sports.nfl.ownership import project_ownership

project_ownership(
    session=session,
    num_field_lineups=50000,
    config_path="config/nfl.json",
)

# Updates NFLPlayer.ownership_proj column
```

### 3. Simulate Contest
```python
from ivan.sports.nfl.simulate_contest import simulate_one_contest

simulate_one_contest(
    session=session,
    contest_id=12345,
    our_lineup_ids=[1, 2, 3, ...],
    num_sims=10000,
    num_field_samples=5000,
)

# Stores results in LineupSimulationResult table
```

---

## 📚 Key Insights

### Why These Features?

1. **log_field_size** - Duplicate behavior scales logarithmically
   - 1K entries: Rare to see any duplicates
   - 100K entries: Top lineup might appear 200+ times

2. **entry_fee_tier** - Captures multi-entry behavior
   - $1 contests: Casual, one entry each
   - $100 contests: Sharks, 20-150 entries, more concentration

3. **ownership_gini** - Measures field concentration
   - 0.0 = perfect equality (all players same ownership)
   - 1.0 = perfect inequality (one player 100%, rest 0%)
   - Higher gini = more duplicates on chalk lineups

4. **cpt_ownership** - Captain is 1.5x multiplier
   - Most important decision in showdown
   - High ownership captain = high duplication

5. **flex_ownership_mean** - Average chalkiness
   - High mean = chalky supporting cast
   - Low mean = contrarian supporting cast

6. **flex_ownership_std** - Diversity measure
   - Low std = all chalk or all fade (more duplicates)
   - High std = mixed chalk/fade (fewer duplicates)

7. **salary_remaining_pct** - Build optimality
   - 0-2%: Optimized (spend it all)
   - 5-10%: Suboptimal (leaving value on table)
   - High remaining often indicates contrarian punt plays

---

## ✅ Validation Checklist

Before deploying v2:

- [ ] Training data has ≥ 50K examples from ≥ 100 contests
- [ ] No missing values in critical features
- [ ] Stage 1 MAE < 5 duplicates for all quantiles
- [ ] Stage 2 MAE < 0.05 on percentile scale
- [ ] Overall correlation > 0.65 with actuals
- [ ] No crashes on edge cases (empty field, single lineup, etc.)
- [ ] Runtime < 1ms per lineup (excluding model load)
- [ ] Feature values match manual calculations
- [ ] Predictions are stable across multiple runs (no randomness leaks)

---

## 🚨 Known Limitations

1. **Cold Start**: New players with no historical ownership require fallback
   - **Solution**: Use position-based average ownership

2. **Distribution Shift**: If ownership projections are systematically biased
   - **Solution**: Calibrate projections or train on projected ownership

3. **Multi-Entry Not Explicit**: Model learns aggregate behavior, not individual user strategies
   - **Solution**: Entry fee tier captures some of this

4. **Correlation Ignored**: No explicit correlation features
   - **Solution**: Ownership implicitly captures correlation patterns

5. **Time Sensitivity**: Pre-lock vs late-swap ownership may differ
   - **Solution**: Train on lock-time snapshots only

---

## 📞 Need Help?

### Documentation
1. **Overview**: `docs/nfl_showdown_dupe_v2_summary.md` (this file)
2. **Full Spec**: `docs/nfl_showdown_dupe_model_v2.md`
3. **Comparison**: `docs/nfl_showdown_dupe_comparison.md`
4. **Roadmap**: `docs/nfl_showdown_dupe_v2_roadmap.md`
5. **Visuals**: `docs/nfl_showdown_dupe_v2_visual_flow.py`

### Key Functions
- `build_training_dataset()` in `build_training_data_v2.py`
- `train_stage1()` / `train_stage2()` in `train_showdown_dupe_v2.py`
- `predict_duplicate_counts_v2()` in `inference.py`
- `sample_showdown_field_v2()` in `sample_field.py`

### Debugging
- Print feature values at each step
- Check model files exist and load correctly
- Verify ownership values are reasonable (0-100 range)
- Compare predictions with v1 on same input

---

**Ready to implement? Start with Phase 1 (data pipeline)! 🚀**
