# NFL Showdown Dupe Model v2 - Implementation Roadmap

## 📁 File Structure

```
src/ivan/sports/nfl/
├── dupe_model_v2/              # NEW: Clean v2 implementation
│   ├── __init__.py
│   ├── features.py             # Feature extraction logic
│   ├── training.py             # Model training functions
│   └── inference.py            # Prediction pipeline
├── sample_field.py             # MODIFY: Add v2 flag
├── models.py                   # KEEP: Database models
└── ownership.py                # KEEP: Ownership projection

scripts/nfl/
├── train_showdown_dupe_v2.py   # NEW: V2 training script
├── build_training_data_v2.py   # NEW: Data pipeline
├── validate_dupe_model_v2.py   # NEW: Validation script
├── train_showdown_model.py     # KEEP: Legacy v1 training
└── sim.py                      # MODIFY: Add --dupe-model-version flag

data/nfl/
├── models/
│   ├── showdown_dupe_v1/       # KEEP: Current models
│   └── showdown_dupe_v2/       # NEW: V2 models
│       ├── stage1_model.pkl
│       ├── stage1_scaler.pkl
│       ├── stage2_model.pkl
│       ├── stage2_scaler.pkl
│       └── metadata.json       # Model version, training date, etc.
└── training/
    └── showdown_dupe_v2_training.parquet  # NEW: Training dataset

config/nfl.json
# ADD: showdown_dupe_model_v2 section

tests/nfl/
└── validate_dupe_model_v2.ipynb  # NEW: Jupyter notebook for analysis
```

---

## 🎯 Implementation Phases

### Phase 1: Data Pipeline (2-3 hours)

**Goal**: Create clean training dataset from historical contests

**File**: `scripts/nfl/build_training_data_v2.py`

**Tasks**:
1. Query historical contests (≥ 5K entries)
2. Extract contest-level features
3. Extract lineup-level features
4. Compute percentile ranks
5. Export to Parquet

**Key Functions**:
```python
def build_training_dataset(
    session: Session,
    min_entries: int = 5000,
    output_path: str = "data/nfl/training/showdown_dupe_v2_training.parquet"
) -> pd.DataFrame:
    """
    Build complete training dataset.
    
    Returns DataFrame with columns:
    - contest_id, lineup_id
    - log_field_size, entry_fee_tier, ownership_gini, num_games_tier
    - cpt_ownership, flex_ownership_mean, flex_ownership_std, salary_remaining_pct
    - duplicate_count, percentile_rank
    """
    pass

def extract_contest_features(contest: ShowdownContestMetadata) -> dict:
    """Extract 4 contest-level features."""
    pass

def extract_lineup_features(
    lineup: ShowdownLineup,
    ownership_map: dict
) -> dict:
    """Extract 4 lineup-level features."""
    pass
```

**Validation Checks**:
- [ ] No missing values in critical features
- [ ] Ownership values in [0, 100] range
- [ ] Duplicate counts ≥ 1
- [ ] Percentile ranks in [0, 1] range
- [ ] ~50-100K training examples from ~100-200 contests

---

### Phase 2: Model Training (2-3 hours)

**Goal**: Train two-stage model on historical data

**File**: `scripts/nfl/train_showdown_dupe_v2.py`

**Tasks**:
1. Load training data
2. Train Stage 1 (contest quantiles)
3. Train Stage 2 (lineup percentiles)
4. Evaluate on held-out data
5. Save models with metadata

**Key Functions**:
```python
def train_stage1(
    df: pd.DataFrame,
    config: dict
) -> Tuple[MultiOutputRegressor, StandardScaler]:
    """
    Train contest-level quantile predictor.
    
    Input: [log_field_size, entry_fee_tier, ownership_gini, num_games_tier]
    Output: [q50, q75, q90, q95, q99]
    """
    pass

def train_stage2(
    df: pd.DataFrame,
    config: dict
) -> Tuple[GradientBoostingRegressor, StandardScaler]:
    """
    Train lineup-level percentile predictor.
    
    Input: [cpt_ownership, flex_ownership_mean, flex_ownership_std, salary_remaining_pct]
    Output: logit(percentile_rank)
    """
    pass

def evaluate_models(
    stage1_model, stage1_scaler,
    stage2_model, stage2_scaler,
    df_test: pd.DataFrame
) -> dict:
    """Compute MAE, R2, correlation on held-out test set."""
    pass
```

**Output Files**:
```
data/nfl/models/showdown_dupe_v2/
├── stage1_model.pkl          # MultiOutputRegressor
├── stage1_scaler.pkl         # StandardScaler
├── stage2_model.pkl          # GradientBoostingRegressor
├── stage2_scaler.pkl         # StandardScaler
└── metadata.json             # Version, training date, performance metrics
```

**Validation Checks**:
- [ ] Stage 1 MAE < 5 duplicates for each quantile
- [ ] Stage 2 MAE < 0.05 on percentile scale
- [ ] No overfitting (train vs test gap < 10%)
- [ ] Models load/predict correctly

---

### Phase 3: Feature Extraction Module (1-2 hours)

**Goal**: Clean, reusable feature extraction logic

**File**: `src/ivan/sports/nfl/dupe_model_v2/features.py`

**Key Functions**:
```python
def extract_contest_features_runtime(
    total_entries: int,
    entry_fee: float,
    ownership_values: List[float],  # All player ownerships
    num_games: int = 1
) -> np.ndarray:
    """
    Extract contest features at runtime (no database needed).
    
    Returns: [log_field_size, entry_fee_tier, ownership_gini, num_games_tier]
    """
    log_field_size = np.log10(total_entries)
    entry_fee_tier = map_entry_fee_to_tier(entry_fee)
    ownership_gini = calculate_gini(ownership_values)
    num_games_tier = float(num_games)
    
    return np.array([[log_field_size, entry_fee_tier, ownership_gini, num_games_tier]])

def extract_lineup_features_runtime(
    cpt_id: int,
    flex_ids: List[int],
    ownership_map: Dict[int, float],
    salary_remaining: int,
    salary_cap: int = 50000
) -> np.ndarray:
    """
    Extract lineup features at runtime.
    
    Returns: [cpt_ownership, flex_ownership_mean, flex_ownership_std, salary_remaining_pct]
    """
    cpt_own = ownership_map.get(cpt_id, 0.0)
    flex_owns = [ownership_map.get(fid, 0.0) for fid in flex_ids]
    
    flex_own_mean = np.mean(flex_owns)
    flex_own_std = np.std(flex_owns)
    salary_remaining_pct = (salary_remaining / salary_cap) * 100.0
    
    return np.array([[cpt_own, flex_own_mean, flex_own_std, salary_remaining_pct]])

def map_entry_fee_to_tier(entry_fee: float) -> float:
    """Map entry fee to discrete tier [1, 3, 5, 10, 20, 50, 100]."""
    if entry_fee <= 1:
        return 1.0
    elif entry_fee <= 3:
        return 3.0
    elif entry_fee <= 5:
        return 5.0
    elif entry_fee <= 10:
        return 10.0
    elif entry_fee <= 20:
        return 20.0
    elif entry_fee <= 50:
        return 50.0
    else:
        return 100.0

def calculate_gini(values: List[float]) -> float:
    """Calculate Gini coefficient [0-1] for ownership concentration."""
    if not values or len(values) < 2:
        return 0.5  # Default moderate concentration
    
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    
    gini = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))
```

**Unit Tests**:
```python
def test_entry_fee_tiers():
    assert map_entry_fee_to_tier(0.25) == 1.0
    assert map_entry_fee_to_tier(1.0) == 1.0
    assert map_entry_fee_to_tier(5.0) == 5.0
    assert map_entry_fee_to_tier(200.0) == 100.0

def test_gini_coefficient():
    # Perfect equality: all same value
    assert calculate_gini([10, 10, 10, 10]) == 0.0
    
    # Perfect inequality: one person has everything
    assert calculate_gini([0, 0, 0, 100]) > 0.9
    
    # Moderate inequality
    gini = calculate_gini([5, 10, 15, 20, 25, 35])
    assert 0.2 < gini < 0.4
```

---

### Phase 4: Inference Module (2-3 hours)

**Goal**: Prediction pipeline for runtime use

**File**: `src/ivan/sports/nfl/dupe_model_v2/inference.py`

**Key Functions**:
```python
def load_models_v2(
    model_dir: str = "data/nfl/models/showdown_dupe_v2"
) -> dict:
    """
    Load trained v2 models from disk.
    
    Returns:
        {
            'stage1_model': MultiOutputRegressor,
            'stage1_scaler': StandardScaler,
            'stage2_model': GradientBoostingRegressor,
            'stage2_scaler': StandardScaler,
            'metadata': dict,
        }
    """
    pass

def predict_duplicate_counts_v2(
    candidate_lineups: List[Dict],  # [{cpt_id, flex_ids, salary_remaining}]
    contest_params: Dict,            # {total_entries, entry_fee, ownership_values}
    ownership_map: Dict[int, float], # {player_id: ownership_pct}
    models: dict,
    config: dict = None
) -> np.ndarray:
    """
    Predict duplicate counts for candidate lineups.
    
    Returns:
        Array of shape (n_lineups,) with predicted duplicate counts
    """
    # Step 1: Extract contest features
    X_contest = extract_contest_features_runtime(
        total_entries=contest_params['total_entries'],
        entry_fee=contest_params['entry_fee'],
        ownership_values=contest_params['ownership_values'],
        num_games=1,
    )
    
    # Step 2: Predict contest quantiles
    X_contest_scaled = models['stage1_scaler'].transform(X_contest)
    quantiles = models['stage1_model'].predict(X_contest_scaled)[0]  # [q50, q75, q90, q95, q99]
    
    # Build full CDF with boundary points
    quantile_levels = np.array([0.0, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    quantile_values = np.array([
        1.0,                                         # Min: everyone appears once
        quantiles[0], quantiles[1], quantiles[2],   # Predicted quantiles
        quantiles[3], quantiles[4],
        contest_params['total_entries'] * 0.02       # Max: 2% of field
    ])
    
    # Step 3: Extract lineup features
    X_lineups = []
    for lineup in candidate_lineups:
        features = extract_lineup_features_runtime(
            cpt_id=lineup['cpt_id'],
            flex_ids=lineup['flex_ids'],
            ownership_map=ownership_map,
            salary_remaining=lineup['salary_remaining'],
        )
        X_lineups.append(features[0])
    
    X_lineups = np.array(X_lineups)
    
    # Step 4: Predict lineup scores
    X_lineups_scaled = models['stage2_scaler'].transform(X_lineups)
    scores_logit = models['stage2_model'].predict(X_lineups_scaled)
    scores = expit(scores_logit)  # Transform to [0, 1]
    
    # Step 5: Convert scores to percentiles
    from scipy.stats import rankdata
    percentiles = rankdata(scores, method='average') / len(scores)
    percentiles = np.clip(percentiles, 0.0, 1.0)
    
    # Step 6: Interpolate percentiles into quantile distribution
    duplicates = np.interp(
        percentiles,
        quantile_levels,
        quantile_values,
    )
    
    # Apply config constraints if provided
    if config:
        min_dupes = config.get('min_duplicate_count', 1.0)
        max_dupes_pct = config.get('max_duplicate_pct', 0.02)
        max_dupes = contest_params['total_entries'] * max_dupes_pct
        
        duplicates = np.clip(duplicates, min_dupes, max_dupes)
    
    return duplicates
```

**Integration Point**:
```python
# In sample_field.py
def sample_showdown_field_v2(
    session: Session,
    field_size: int,
    config: dict,
    contest_id: Optional[int] = None,
) -> Tuple[Dict[int, int], Dict]:
    """
    Sample showdown field using v2 duplication model.
    
    This is the main entry point called by ownership.py and sim.py
    """
    # Load v2 models
    models = load_models_v2(config.get('model_dir', 'data/nfl/models/showdown_dupe_v2'))
    
    # Get candidate lineups from database
    candidate_lineups = session.query(Lineup).limit(10000).all()
    
    # Build ownership map
    players = session.query(NFLPlayer).all()
    ownership_map = {p.id: p.ownership_proj for p in players}
    
    # Predict duplicates
    duplicates = predict_duplicate_counts_v2(
        candidate_lineups=[...],
        contest_params={
            'total_entries': field_size,
            'entry_fee': config.get('entry_fee', 5.0),
            'ownership_values': list(ownership_map.values()),
        },
        ownership_map=ownership_map,
        models=models,
        config=config.get('showdown_dupe_model_v2', {}),
    )
    
    # Sample field using predicted duplicates (same as v1)
    # ... rest of sampling logic
```

---

### Phase 5: Integration & Testing (2-3 hours)

**Goal**: Integrate v2 into existing pipeline with feature flag

**Modified Files**:
- `src/ivan/sports/nfl/sample_field.py`
- `scripts/nfl/sim.py`
- `config/nfl.json`

**Feature Flag Pattern**:
```python
# In config/nfl.json
{
  "dupe_model_version": "v2",  # "v1" or "v2"
  "showdown_dupe_model_v2": {
    "min_duplicate_count": 1.0,
    "max_duplicate_pct": 0.02,
    "smooth_percentiles": false
  }
}

# In sample_field.py
def sample_showdown_field(session, field_size, config, contest_id=None):
    version = config.get('dupe_model_version', 'v1')
    
    if version == 'v2':
        return sample_showdown_field_v2(session, field_size, config, contest_id)
    else:
        return sample_showdown_field_learned(session, field_size, config, contest_id)
```

**Testing Checklist**:
- [ ] v2 produces valid duplicate counts (≥ 1, ≤ 0.02 * field_size)
- [ ] v2 runs faster than v1 (measure with `time.time()`)
- [ ] v2 produces different but reasonable ownership projections
- [ ] Feature flag switches between v1 and v2 correctly
- [ ] No crashes or errors on edge cases (empty field, single lineup, etc.)

---

### Phase 6: Validation & Analysis (3-5 hours)

**Goal**: Validate model performance on held-out contests

**File**: `tests/nfl/validate_dupe_model_v2.ipynb`

**Analyses**:
1. **Prediction Accuracy**
   - Scatter plot: predicted vs actual duplicates
   - MAE by quantile (q50, q75, q90, q95, q99)
   - Correlation coefficient
   
2. **Feature Importance**
   - SHAP values for Stage 2 model
   - Partial dependence plots for key features
   - Feature interaction analysis

3. **Error Analysis**
   - Which lineups are most mispredicted?
   - Are errors systematic (always over/under)?
   - Stratify by contest size, entry fee

4. **Comparison with v1**
   - Side-by-side predictions on same contests
   - Runtime comparison
   - Which model is more stable?

5. **Real-World Simulation**
   - Generate field with v2
   - Compare ownership distributions
   - Check field diversity metrics

**Notebook Outline**:
```python
# Cell 1: Load test data
df_test = pd.read_parquet("data/nfl/training/showdown_dupe_v2_test.parquet")

# Cell 2: Load models
models = load_models_v2()

# Cell 3: Generate predictions
predictions = []
for contest_id in df_test['contest_id'].unique():
    contest_lineups = df_test[df_test['contest_id'] == contest_id]
    preds = predict_duplicate_counts_v2(...)
    predictions.extend(preds)

# Cell 4: Accuracy metrics
mae = mean_absolute_error(df_test['duplicate_count'], predictions)
r2 = r2_score(df_test['duplicate_count'], predictions)
corr = np.corrcoef(df_test['duplicate_count'], predictions)[0, 1]

# Cell 5: Scatter plot
plt.scatter(df_test['duplicate_count'], predictions, alpha=0.3)
plt.xlabel('Actual Duplicates')
plt.ylabel('Predicted Duplicates')
plt.title(f'v2 Model Predictions (R²={r2:.3f}, Corr={corr:.3f})')

# Cell 6: Feature importance
import shap
explainer = shap.TreeExplainer(models['stage2_model'])
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test)

# ... more cells
```

---

## 🚦 Decision Gates

### Gate 1: After Phase 1 (Data Pipeline)
**Question**: Is training data quality acceptable?

**Criteria**:
- [ ] ≥ 50K training examples
- [ ] ≥ 100 unique contests
- [ ] < 1% missing values
- [ ] Features match expected distributions

**If NO**: Debug data pipeline, check database queries

---

### Gate 2: After Phase 2 (Model Training)
**Question**: Are model metrics acceptable?

**Criteria**:
- [ ] Stage 1 MAE < 5 duplicates for q90, q95, q99
- [ ] Stage 2 MAE < 0.05 on percentile scale
- [ ] No severe overfitting (train/test gap < 15%)

**If NO**: Tune hyperparameters, check for data leakage

---

### Gate 3: After Phase 5 (Integration)
**Question**: Does v2 work in production pipeline?

**Criteria**:
- [ ] No runtime errors
- [ ] Produces valid predictions
- [ ] Faster than v1 (or comparable)
- [ ] Ownership projections are reasonable

**If NO**: Debug integration, check edge cases

---

### Gate 4: After Phase 6 (Validation)
**Question**: Is v2 better than v1?

**Criteria**:
- [ ] Equal or better accuracy
- [ ] More interpretable
- [ ] Easier to debug
- [ ] Faster runtime

**If MIXED**: Consider using v2 for certain contest types only (e.g., large fields)

**If NO**: Keep v1, iterate on v2 design

---

## ⏱️ Timeline Estimate

| Phase | Time | Dependencies |
|-------|------|--------------|
| 1. Data Pipeline | 2-3 hours | Database access |
| 2. Model Training | 2-3 hours | Phase 1 complete |
| 3. Feature Module | 1-2 hours | Phase 2 complete |
| 4. Inference Module | 2-3 hours | Phase 3 complete |
| 5. Integration | 2-3 hours | Phase 4 complete |
| 6. Validation | 3-5 hours | Phase 5 complete |

**Total**: 12-19 hours (1.5 - 2.5 days of focused work)

**Add buffer**: +50% for debugging, iteration → **18-29 hours**

**Realistic timeline**: 3-4 days with breaks

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP)
- [ ] v2 model trains successfully
- [ ] v2 produces valid predictions
- [ ] v2 integrates into existing pipeline
- [ ] v2 accuracy ≥ 90% of v1 accuracy

### Full Success
- [ ] v2 accuracy > v1 accuracy
- [ ] v2 runtime < v1 runtime
- [ ] v2 is easier to debug and interpret
- [ ] Documentation complete
- [ ] Team adopts v2 as default

---

## 📝 Next Steps

1. **Review this roadmap** - Any questions or concerns?
2. **Set up environment** - Ensure dependencies installed
3. **Start Phase 1** - Build training data pipeline
4. **Daily check-ins** - Review progress, adjust timeline

---

## 🤔 Open Questions for Discussion

1. **Training Data Scope**
   - How far back should we go? (Last 6 months? 1 year? All time?)
   - Should we filter by sport (NFL only) or include other showdowns?
   - Minimum contest size: 5K entries, or lower?

2. **Feature Engineering**
   - Should `entry_fee_tier` be categorical or continuous?
   - Should we add interaction features (e.g., `log_field_size * entry_fee_tier`)?
   - Any other features worth considering?

3. **Model Hyperparameters**
   - Start with recommended defaults, or do grid search?
   - How much time to allocate for hyperparameter tuning?

4. **Deployment Strategy**
   - Feature flag in config, or separate script versions?
   - A/B test on live contests, or full cutover?
   - Rollback plan if v2 underperforms?

5. **Ownership Source**
   - Train on actual ownership (historical) or projected ownership?
   - At runtime, use projected or some blend?

---

**Let's discuss and then start Phase 1! 🚀**
