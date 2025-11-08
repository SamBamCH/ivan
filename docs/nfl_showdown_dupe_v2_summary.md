# NFL Showdown Duplication Model v2 - Architecture Summary

## 🎯 Executive Summary

This document summarizes the proposed rewrite of the NFL Showdown duplication model. The v2 architecture prioritizes **simplicity, interpretability, and historical data fidelity** over complex feature engineering.

---

## 📊 The Problem

When simulating DFS contests, we need to predict how many times each lineup will appear in the field (duplicate counts). This is critical for:
- **Ownership projection**: Popular lineups get duplicated more
- **Leverage calculation**: Contrarian builds have less duplication
- **ROI estimation**: High-duplicate lineups compete against themselves

**Current v1 Model Issues**:
- 14+ features with unclear definitions (`correlation_score`? `stack_pattern`?)
- Complex preprocessing (percentile normalization, one-hot encoding)
- Hard to debug and validate
- Slow inference (~5-10ms per lineup)

---

## 🏗️ The Solution: Simplified Two-Stage Model

### Core Insight: Histogram Matching

Instead of predicting exact duplicate counts, we:
1. Learn typical **distribution shapes** for different contest types (Stage 1)
2. Map each lineup to a **percentile** in that distribution (Stage 2)  
3. Interpolate to get final duplicate counts

This is simpler, more robust, and matches how DFS fields actually form.

---

## 📐 Architecture

### Stage 1: Contest Distribution Model

**What it does**: Predicts the shape of the duplicate distribution for a contest

**Inputs** (4 contest-level features):
```python
- log_field_size      # log10(total_entries) - scales logarithmically
- entry_fee_tier      # Categorical: [1, 3, 5, 10, 20, 50, 100+]
- ownership_gini      # 0-1, ownership concentration metric
- num_games_tier      # Always 1 for showdown (future-proofing)
```

**Outputs** (5 quantiles):
```python
- q50   # Median duplicate count
- q75   # 75th percentile
- q90   # 90th percentile
- q95   # 95th percentile
- q99   # 99th percentile
```

**Model**: Multi-output Gradient Boosting Regressor

**Example**:
```
Contest: 50,000 entries, $5 entry fee, 0.45 ownership Gini
Predicted distribution:
  Q50: 2 duplicates
  Q75: 5 duplicates
  Q90: 12 duplicates
  Q95: 25 duplicates
  Q99: 78 duplicates
```

---

### Stage 2: Lineup Positioning Model

**What it does**: Predicts where a lineup sits in the contest distribution

**Inputs** (4 lineup-level features):
```python
- cpt_ownership          # Captain's ownership % (0-100)
- flex_ownership_mean    # Mean ownership of 5 flex players
- flex_ownership_std     # Std dev of flex ownership (diversity)
- salary_remaining_pct   # % of $50K cap unused (0-100)
```

**Output**: Duplication score (0-1) → converted to percentile rank

**Model**: Gradient Boosting Regressor (logit-transformed target)

**Example**:
```
Lineup: 42% captain, 25% avg flex (8% std), 3% salary remaining
Predicted score: 0.85 → 85th percentile
```

---

### Combination: Interpolation

**Process**:
1. Stage 1 produces quantile values: `[1, 2, 5, 12, 25, 78]` at percentiles `[0.0, 0.50, 0.75, 0.90, 0.95, 0.99]`
2. Stage 2 produces percentile rank: `0.85` for our lineup
3. Interpolate: At 85th percentile → **~10 duplicates**

---

## 🎨 Key Design Decisions

### 1. Feature Reduction: 14+ → 8 Features

**Removed** (from v1):
- ❌ `correlation_score` - Too complex, undefined calculation
- ❌ `stack_pattern` - Implicitly captured by ownership
- ❌ `game_stack_type` - Minimal predictive power
- ❌ `qb_wr_stack`, `rb_dst_negative` - Edge cases, not worth complexity
- ❌ `projection_normalized` - Adds noise, ownership is better
- ❌ `ownership_percentile` - Chicken-and-egg problem

**Added** (to v2):
- ✅ `entry_fee_tier` - Captures multi-entry behavior (new insight!)
- ✅ `flex_ownership_std` - Captures chalk vs contrarian mix (new!)

**Kept** (both versions):
- ✅ `ownership` metrics - Core signal
- ✅ `salary_remaining_pct` - Indicator of suboptimal builds

### 2. Raw Values Instead of Percentiles

**v1 Problem**: Percentile normalization within contest creates circular dependency
- Can't compute percentiles without seeing all lineups
- Unstable for small contests

**v2 Solution**: Use raw ownership values
- Always available (from player projections)
- No inter-lineup dependencies
- More stable and interpretable

### 3. Entry Fee as Contest Feature

**New insight**: Entry fee predicts duplicate behavior!
- $1 contests: Casual players, more diversity
- $100 contests: Sharks, multi-entry, more concentration

**Implementation**: Discrete tiers `[1, 3, 5, 10, 20, 50, 100+]`

### 4. Flex Ownership Std Dev

**New insight**: Variance in flex ownership matters!
- Low std (2%): All chalk or all fade → more duplication
- High std (15%): Mixed chalk/contrarian → less duplication

**Example**:
```python
Lineup A: [30%, 28%, 25%, 26%, 27%] → std = 2% → chalk → high dupes
Lineup B: [45%, 18%, 8%, 35%, 5%]  → std = 16% → mixed → low dupes
```

---

## 📊 Expected Improvements

| Metric | v1 | v2 (Expected) | Change |
|--------|----|----|--------|
| **Features** | 14+ | 8 | -43% |
| **Inference Speed** | ~5-10ms/lineup | ~1ms/lineup | 5-10x faster |
| **Interpretability** | ⚠️ Complex | ✅ Clear | Much better |
| **Debuggability** | ⚠️ Hard | ✅ Easy | Much better |
| **Accuracy (MAE)** | ~8 dupes | ~5 dupes | Better |
| **Correlation** | ~0.65 | ~0.70+ | Better |

---

## 🗃️ Data Requirements

### Training Data
- **Source**: `ShowdownContestMetadata`, `ShowdownLineup`, `ShowdownPlayer`
- **Size**: ~50-100K lineups from ~100-200 contests
- **Filters**: Contests ≥ 5K entries (more stable distributions)
- **Time Range**: Last 6-12 months (or all available data)

### Required Database Tables
✅ Already have all necessary tables!
- `ShowdownContestMetadata` - Contest-level features
- `ShowdownLineup` - Lineup composition + duplicate counts
- `ShowdownPlayer` - Player ownership (actual historical)

### Data Pipeline Output
```
data/nfl/training/showdown_dupe_v2_training.parquet
Columns:
- contest_id, lineup_id
- log_field_size, entry_fee_tier, ownership_gini, num_games_tier
- cpt_ownership, flex_ownership_mean, flex_ownership_std, salary_remaining_pct
- duplicate_count, percentile_rank
```

---

## 🚀 Implementation Timeline

### Phase 1: Data Pipeline (2-3 hours)
Build training dataset from historical contests

### Phase 2: Model Training (2-3 hours)
Train two-stage model, evaluate performance

### Phase 3: Feature Extraction (1-2 hours)
Clean, reusable feature extraction module

### Phase 4: Inference Pipeline (2-3 hours)
Prediction logic for runtime use

### Phase 5: Integration (2-3 hours)
Add feature flag, integrate with existing code

### Phase 6: Validation (3-5 hours)
Analyze performance, compare with v1

**Total**: 12-19 hours (1.5-2.5 days)
**With buffer**: 18-29 hours (3-4 days)

---

## ✅ Success Criteria

### Minimum Viable Product (MVP)
- [ ] v2 trains successfully
- [ ] v2 produces valid predictions
- [ ] v2 integrates with existing pipeline
- [ ] v2 accuracy ≥ 90% of v1

### Full Success
- [ ] v2 accuracy > v1
- [ ] v2 runtime < v1
- [ ] v2 easier to debug
- [ ] Team adopts v2 as default

---

## 🛠️ File Structure

```
src/ivan/sports/nfl/dupe_model_v2/    # NEW: Clean v2 implementation
├── features.py                        # Feature extraction
├── training.py                        # Model training
└── inference.py                       # Prediction pipeline

scripts/nfl/
├── build_training_data_v2.py         # NEW: Data pipeline
├── train_showdown_dupe_v2.py         # NEW: Training script
└── validate_dupe_model_v2.py         # NEW: Validation

data/nfl/models/showdown_dupe_v2/     # NEW: V2 models
├── stage1_model.pkl
├── stage1_scaler.pkl
├── stage2_model.pkl
├── stage2_scaler.pkl
└── metadata.json
```

---

## 🤔 Open Questions for Discussion

### 1. Training Data Scope
- **Time range**: Last 6 months? 1 year? All time?
- **Contest size filter**: 5K entries minimum, or lower?
- **Sport filter**: NFL only, or include other showdowns?

### 2. Ownership Source
- **For training**: Use actual historical ownership (ground truth)
- **For inference**: Use projected ownership (what we have at runtime)
- **Concern**: Distribution shift if projections are biased

**Recommendation**: Train on actual, validate that projected ownership has similar distributions

### 3. Entry Fee Tiers
- **Current proposal**: `[1, 3, 5, 10, 20, 50, 100+]`
- **Alternative**: Continuous log scale `log10(entry_fee)`
- **Trade-off**: Discrete is more interpretable, continuous is more flexible

**Recommendation**: Start with discrete, can switch to continuous if needed

### 4. Feature Interactions
- **Question**: Should we add interaction terms? E.g., `log_field_size * entry_fee_tier`
- **Pro**: May capture complex effects
- **Con**: Adds complexity, risk of overfitting

**Recommendation**: Start without interactions, add if needed

### 5. Deployment Strategy
- **Option A**: Feature flag in config (`dupe_model_version: "v1" | "v2"`)
- **Option B**: Separate script versions (`sim_v1.py`, `sim_v2.py`)
- **Option C**: A/B test on live contests

**Recommendation**: Option A (feature flag) for easy rollback

---

## 📚 Documentation

Three key documents created:

1. **`nfl_showdown_dupe_model_v2.md`** - Full architecture specification
   - Detailed model design
   - Training procedure
   - Success metrics
   - Configuration

2. **`nfl_showdown_dupe_comparison.md`** - v1 vs v2 comparison
   - Feature comparison tables
   - Performance expectations
   - Migration path
   - Known limitations

3. **`nfl_showdown_dupe_v2_roadmap.md`** - Implementation plan
   - Phase-by-phase breakdown
   - File structure
   - Decision gates
   - Timeline estimates

---

## 🎯 Recommendation

**Proceed with v2 implementation**

**Key benefits**:
1. 72% fewer features (14+ → 8) = simpler, more maintainable
2. 5-10x faster inference = scales to larger candidate pools
3. Fully interpretable = easier to debug and validate
4. Historical data-driven = no synthetic feature engineering
5. Better generalization = less overfitting risk

**Risks** (mitigated):
1. Loss of correlation features → ownership captures implicitly
2. Uncertain if simpler matches v1 accuracy → can A/B test
3. Requires retraining → ~3 hours of training time

**Next step**: Implement Phase 1 (data pipeline) and validate training data quality

---

## 🚀 Getting Started

```bash
# 1. Review architecture
cat docs/nfl_showdown_dupe_model_v2.md

# 2. Compare with v1
cat docs/nfl_showdown_dupe_comparison.md

# 3. Check implementation roadmap
cat docs/nfl_showdown_dupe_v2_roadmap.md

# 4. Start Phase 1
python scripts/nfl/build_training_data_v2.py

# 5. Train models (Phase 2)
python scripts/nfl/train_showdown_dupe_v2.py

# 6. Validate (Phase 6)
jupyter notebook tests/nfl/validate_dupe_model_v2.ipynb
```

---

## 📞 Questions?

Review the full documentation and let me know:
1. Any concerns about the simplified feature set?
2. Agreement on training data scope?
3. Preferred deployment strategy?
4. Ready to start Phase 1?

**Let's build a simpler, better duplication model! 🎯**
