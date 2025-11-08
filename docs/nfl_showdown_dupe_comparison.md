# NFL Showdown Dupe Model: v1 vs v2 Comparison

## 🔄 High-Level Flow Comparison

### v1 (Current)
```
Historical Contest
       ↓
   [Feature Engineering]
   - Stack patterns (QB-WR-WR, etc.)
   - Correlation scores (complex calculation)
   - Percentile normalization within contest
   - Game stack type (5-1, 4-2, etc.)
       ↓
   [Stage 1: Contest Model]
   Input: 4 features
   Output: Quantiles [Q25, Q50, Q75, Q90, Q95, Q99]
       ↓
   [Stage 2: Lineup Model]
   Input: 14+ features (including correlation, stacks, percentiles)
   Output: Duplication score (logit)
       ↓
   [Combine]
   Scores → Percentiles → Interpolate into Quantiles
       ↓
   Predicted Duplicates
```

### v2 (Proposed)
```
Historical Contest
       ↓
   [Feature Extraction]
   - Raw ownership values
   - Salary remaining %
   - Entry fee / field size
   - Ownership concentration (Gini)
       ↓
   [Stage 1: Contest Model]
   Input: 4 features (log_field_size, entry_fee_tier, ownership_gini, num_games)
   Output: Quantiles [Q50, Q75, Q90, Q95, Q99]
       ↓
   [Stage 2: Lineup Model]  
   Input: 4 features (cpt_own, flex_own_mean, flex_own_std, salary_remaining_pct)
   Output: Duplication score (logit)
       ↓
   [Combine]
   Scores → Percentiles → Interpolate into Quantiles
       ↓
   Predicted Duplicates
```

---

## 📊 Feature Comparison

### Stage 1: Contest-Level Features

| v1 | v2 | Change |
|----|----|----|
| `total_entries` | `log_field_size` | ✅ Better: Log scale handles wide range |
| ❌ Not used | `entry_fee_tier` | ✅ New: Captures multi-entry behavior |
| `ownership_gini` | `ownership_gini` | ✅ Same: Good metric |
| `projection_std` | `num_games_tier` | ⚠️ Changed: More general for future |
| `field_density` | ❌ Removed | ⚠️ Redundant with field size |

### Stage 2: Lineup-Level Features

| v1 | v2 | Change | Reason |
|----|----|----|--------|
| `cpt_ownership_pct` | `cpt_ownership` | ✅ Same | Core feature |
| `ownership_sum` | `flex_ownership_mean` | ✅ Better | Mean is more interpretable than sum |
| ❌ Not used | `flex_ownership_std` | ✅ New | Captures chalk vs contrarian mix |
| `salary_remaining_pct` | `salary_remaining_pct` | ✅ Same | Good indicator |
| `correlation_score` | ❌ Removed | ✅ Good | Too complex, undefined calculation |
| `stack_pattern` (categorical) | ❌ Removed | ✅ Good | Captured by ownership |
| `game_stack_type` (categorical) | ❌ Removed | ✅ Good | Minimal predictive power |
| `qb_wr_stack` (boolean) | ❌ Removed | ✅ Good | Redundant with ownership |
| `rb_dst_negative` (boolean) | ❌ Removed | ✅ Good | Edge case, not worth it |
| `same_team_count` | ❌ Removed | ✅ Good | Captured by ownership |
| `projection_normalized` | ❌ Removed | ✅ Good | Adds noise, ownership is better |
| `ownership_percentile` | ❌ Removed | ✅ Good | Chicken-and-egg problem |
| `projection_percentile` | ❌ Removed | ✅ Good | Not needed |

**Feature Count**: 14+ → 4 (72% reduction!)

---

## 🎯 Advantages of v2

### 1. Simplicity
- **v1**: 14+ features, many with unclear definitions
- **v2**: 8 total features (4+4), all directly measurable

### 2. Interpretability
**v1 Lineup Score Explanation**:
> "This lineup has a high duplication score because... uh... the correlation_score is 0.73? And the stack_pattern is QB-WR-WR? But what does that mean exactly?"

**v2 Lineup Score Explanation**:
> "This lineup has 35% captain ownership, 22% average flex ownership (low std dev = all chalk), and 3% salary remaining. That's a classic field chalk build."

### 3. Validation
**v1**: Hard to verify
- What is the formula for `correlation_score`?
- How are stack patterns encoded?
- Why are percentiles computed within contest (circular dependency)?

**v2**: Easy to verify
```python
# Every feature is directly queryable
cpt_own = session.query(ShowdownPlayer.ownership_pct)\
    .filter(normalized_name == 'patrick mahomes', roster_slot == 'CPT')\
    .scalar()

# Result: 42.3% ✓ Matches reality
```

### 4. Training Data Quality
**v1**: 
- Requires all lineups in a contest to compute percentiles
- Can't train on individual lineups in isolation
- Normalization can be unstable for small contests

**v2**:
- Each lineup is an independent training example
- Raw ownership values are always available
- No inter-lineup dependencies during feature extraction

### 5. Runtime Performance
**v1**: 
```python
# For each lineup:
# - Compute stack patterns (string matching)
# - Calculate correlation scores (complex formula)
# - Normalize across all candidates (requires full dataset)
# - One-hot encode categorical features
# Estimated: ~5-10ms per lineup
```

**v2**:
```python
# For each lineup:
# - Look up ownership values (database query or dict lookup)
# - Compute mean/std of 5 numbers
# - Divide salary remaining by cap
# Estimated: <1ms per lineup (5-10x faster)
```

---

## 📈 Expected Performance

### Accuracy (Estimated)

| Metric | v1 | v2 (Expected) | Reasoning |
|--------|----|----|-----------|
| Stage 1 Q90 MAE | ~8 dupes | ~5 dupes | Better contest features (entry fee) |
| Stage 2 Percentile MAE | ~0.08 | ~0.05 | Cleaner target (no percentile circularity) |
| Overall Correlation | ~0.65 | ~0.70+ | Simpler model = less overfitting |

### Robustness

**v1 Failure Modes**:
1. New stack pattern not seen in training → defaults poorly
2. Contest with unusual ownership distribution → percentile normalization breaks
3. Small contests (<1000 entries) → not enough data for percentiles

**v2 Failure Modes**:
1. Player with no historical ownership → use position average (graceful degradation)
2. Extreme field size (100K+ entries) → log scale handles this naturally
3. Zero variance in flex ownership → std = 0, still interpretable

---

## 🔄 Migration Path

### Phase 1: Parallel Implementation
- Keep v1 model running in production
- Train v2 model on same historical data
- Compare predictions side-by-side on new slates

### Phase 2: A/B Testing
- Use v1 for 50% of lineups, v2 for 50%
- Track which model produces better field simulation
- Measure runtime differences

### Phase 3: Deprecation
- Once v2 proves superior, switch fully
- Archive v1 model and code
- Update documentation

---

## 🛠️ Implementation Checklist

### Data Pipeline
- [ ] `scripts/nfl/build_training_data_v2.py`
  - [ ] Query historical contests (min 5K entries)
  - [ ] Extract 8 features per lineup
  - [ ] Compute percentile ranks within contest
  - [ ] Export to Parquet

### Model Training
- [ ] `scripts/nfl/train_showdown_dupe_v2.py`
  - [ ] Stage 1: Contest quantile model
  - [ ] Stage 2: Lineup percentile model
  - [ ] Save to `data/nfl/models/showdown_dupe_v2/`

### Integration
- [ ] `src/ivan/sports/nfl/sample_field_v2.py`
  - [ ] `predict_lineups_with_twostage_v2()`
  - [ ] `sample_showdown_field_v2()`
  - [ ] Add feature flag to switch versions

### Validation
- [ ] Jupyter notebook: `tests/nfl/validate_dupe_model_v2.ipynb`
  - [ ] Compare predictions on held-out contests
  - [ ] Feature importance analysis
  - [ ] Error distribution plots

---

## 💭 Open Questions

### For Discussion

1. **Entry Fee Tiers**: Should we use [1, 3, 5, 10, 20, 50, 100+] or finer buckets?
2. **Ownership Source**: Should we use projected or actual ownership for training?
   - **Actual**: More accurate (ground truth)
   - **Projected**: More realistic for inference (we won't have actual at runtime)
3. **Percentile Smoothing**: Should we apply Gaussian smoothing to percentiles before interpolation?
4. **Cold Start**: How to handle brand new players with no ownership data?
   - Option A: Position-based average
   - Option B: Salary-based imputation
   - Option C: Skip lineup entirely

### Future Enhancements

1. **Multi-Entry Modeling**: Explicitly model max entry limits (3-entry vs 150-entry)
2. **Time Decay**: Weight recent contests more heavily than old ones
3. **Late Swap**: Adjust model for post-lock ownership shifts
4. **Main Slate Extension**: Generalize to multi-game slates (already designed for this!)

---

## 📚 References

### Code Files to Review
- Current: `src/ivan/sports/nfl/sample_field.py` (lines 200-400)
- Current: `scripts/nfl/train_showdown_model.py`
- Current: `src/ivan/sports/nfl/showdown_models.py`

### Database Tables
- `ShowdownContestMetadata`
- `ShowdownLineup`
- `ShowdownPlayer`
- `ProjectionSnapshot`

### Related Docs
- `docs/nfl_showdown_dupe_model_v2.md` (full architecture spec)
- `config/nfl.json` (configuration)

---

## ✅ Decision Summary

**Recommendation**: Proceed with v2 implementation

**Key Benefits**:
1. 72% fewer features (14+ → 8)
2. 5-10x faster inference
3. Fully interpretable
4. Easy to validate and debug
5. Better generalization (less overfitting risk)

**Risks**:
1. Loss of correlation features (mitigated: ownership captures this implicitly)
2. Requires retraining (expected: not a blocker)
3. Uncertain if simpler model will match v1 accuracy (can test in parallel)

**Next Step**: Implement Phase 1 (data pipeline) and validate training data quality
