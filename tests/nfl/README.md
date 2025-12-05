# NFL Stat → FPTS Visualization Suite

This suite provides comprehensive visualization tools for analyzing and explaining the NFL statistical modeling methodology used in IVAN. These tools help visualize how individual player stats are modeled, calibrated to betting odds, and combined into final fantasy point projections.

## 📊 Overview

The visualization pipeline demonstrates the complete modeling workflow:

1. **Historical Data Analysis** - KDE fitting to player's historical performance
2. **Odds Integration** - Extracting market-implied probabilities from betting lines
3. **Distribution Calibration** - Adjusting KDEs to match odds-derived CDFs
4. **Dependent Sampling** - Applying correlation structure via Gaussian copula
5. **FPTS Projection** - Computing final fantasy point distributions

## 🔧 Scripts

### 1. `visualize_stat_to_fpts.py`

**Comprehensive single-player visualization showing the complete modeling pipeline.**

Creates a multi-panel visualization showing:
- Individual stat PDFs (historical + KDE + calibrated samples)
- Stat CDFs with odds calibration overlay
- Correlation matrix between stats
- Sample distribution box plots with scoring weights
- Final FPTS distribution with percentiles

**Usage:**
```bash
# Basic usage
python visualize_stat_to_fpts.py "Patrick Mahomes"

# With custom parameters
python visualize_stat_to_fpts.py "Christian McCaffrey" --samples 20000 --seed 123

# Save to file
python visualize_stat_to_fpts.py "Travis Kelce" --output kelce_analysis.png

# Different positions
python visualize_stat_to_fpts.py "Tyreek Hill"  # WR
python visualize_stat_to_fpts.py "Mark Andrews"  # TE
```

**Arguments:**
- `player_name` - Full name of NFL player (required)
- `--samples` - Number of Monte Carlo samples (default: 10000)
- `--output` - Output file path (if omitted, displays interactively)
- `--seed` - Random seed for reproducibility (default: 42)

**Output:**
- 4x4 grid showing complete modeling pipeline
- Individual stat distributions (top 4 stats)
- CDF calibration quality
- Correlation structure
- Final FPTS distribution

---

### 2. `visualize_player_comparison.py`

**Side-by-side comparison of multiple players' FPTS distributions.**

Compares players to understand relative projections and uncertainty.

**Usage:**
```bash
# Compare specific players
python visualize_player_comparison.py "Patrick Mahomes" "Josh Allen" "Lamar Jackson"

# Top N players at a position
python visualize_player_comparison.py --position QB --top 5

# Save comparison
python visualize_player_comparison.py --output qb_elite.png "Patrick Mahomes" "Josh Allen" "Joe Burrow"

# Running backs
python visualize_player_comparison.py --position RB --top 3
```

**Arguments:**
- `players` - List of player names (space-separated)
- `--position` - Position filter (QB, RB, WR, TE)
- `--top` - Number of top players by salary (used with --position)
- `--samples` - Number of Monte Carlo samples (default: 10000)
- `--output` - Output file path
- `--seed` - Random seed (default: 42)

**Output:**
- Individual FPTS distributions for each player
- Overlay CDF comparison
- Summary statistics bar chart with error bars
- Metadata table (games, odds, stats modeled)

---

### 3. `analyze_odds_calibration.py`

**Deep-dive analysis of odds calibration quality.**

Shows how well the calibrated distributions match market odds, before and after calibration.

**Usage:**
```bash
# Single player analysis
python analyze_odds_calibration.py "Patrick Mahomes"

# Save to file
python analyze_odds_calibration.py "Travis Kelce" --output kelce_calibration.png

# Batch analysis for position
python analyze_odds_calibration.py --position QB --top 10 --output-dir calibration_results/
```

**Arguments:**
- `player_name` - Full name of NFL player (optional if using --position)
- `--position` - Position to analyze (QB, RB, WR, TE)
- `--top` - Number of top players by salary (default: 10)
- `--output` - Output file path for single player
- `--output-dir` - Output directory for batch analysis

**Output:**
For each stat:
- Predicted vs Target CDF scatter (base vs calibrated)
- Absolute error by odds point
- CDF curves comparison
- Summary statistics (MAE, improvement %)

---

## 📈 Example Workflows

### Explaining Your Methodology

**For presentations or documentation:**

```bash
# Generate comprehensive visualization for a star player
python visualize_stat_to_fpts.py "Patrick Mahomes" --samples 20000 --output mahomes_methodology.png

# Compare multiple QBs to show consistency
python visualize_player_comparison.py "Patrick Mahomes" "Josh Allen" "Jalen Hurts" --output qb_comparison.png

# Analyze calibration quality
python analyze_odds_calibration.py "Patrick Mahomes" --output mahomes_calibration.png
```

### Research & Development

**Testing improvements to the model:**

```bash
# Analyze calibration quality across top QBs
python analyze_odds_calibration.py --position QB --top 15 --output-dir calibration_qb/

# Compare RB projections
python visualize_player_comparison.py --position RB --top 10 --output rb_elite.png

# Deep dive on specific players
python visualize_stat_to_fpts.py "Christian McCaffrey" --samples 50000
```

### Position-Specific Analysis

```bash
# Quarterbacks (many stats, complex correlations)
python visualize_stat_to_fpts.py "Patrick Mahomes"
python visualize_player_comparison.py --position QB --top 5

# Running backs (fewer stats, tight distributions)
python visualize_stat_to_fpts.py "Christian McCaffrey"
python visualize_player_comparison.py --position RB --top 5

# Wide receivers (high variance)
python visualize_stat_to_fpts.py "Tyreek Hill"
python visualize_player_comparison.py --position WR --top 5

# Tight ends
python visualize_stat_to_fpts.py "Travis Kelce"
python visualize_player_comparison.py --position TE --top 5
```

---

## 🎨 Visualization Components

### Stat Distribution Plots (PDF)
- **Blue histogram**: Historical data
- **Green dashed line**: Base KDE fit
- **Purple filled**: Calibrated samples
- **Vertical lines**: Mean values (historical vs calibrated)

### Stat CDF Plots
- **Green dashed**: Base KDE CDF
- **Purple solid**: Calibrated CDF
- **Red points**: Odds-derived CDF targets
- **Red dotted lines**: Vertical lines from odds points

### Correlation Matrix
- **Heatmap**: Spearman rank correlation between stats
- **Values**: -1 (negative) to +1 (positive)
- Shows statistical dependencies used for copula

### FPTS Distribution
- **Teal histogram**: Simulated FPTS samples
- **Red curve**: Normal distribution fit
- **Dashed lines**: Key percentiles (P10, P25, P50, P75, P90)
- **Labels**: Mean and standard deviation

### Calibration Analysis
- **Scatter plots**: Predicted vs target CDF
- **Bar charts**: Absolute errors by odds point
- **Line plots**: CDF curves comparison

---

## 🔍 Interpreting Results

### Good Calibration
- Calibrated CDF closely matches odds points (red dots on purple line)
- Calibrated MAE << Base MAE
- Improvement percentage > 50%
- Scatter points near the diagonal line

### Poor Calibration
- Large gaps between calibrated CDF and odds points
- Similar MAE for base and calibrated
- Low improvement percentage
- Scatter points far from diagonal

### Correlation Insights
- **Strong positive** (red): Stats that tend to increase together (e.g., passing yards + passing TDs)
- **Strong negative** (blue): Stats inversely related (rare in football)
- **Weak** (white): Independent stats

### FPTS Distribution Quality
- **Narrow distribution** (low σ): Consistent, low-ceiling player
- **Wide distribution** (high σ): Boom-bust potential
- **High mean**: Strong projection
- **Skewed right**: Upside potential

---

## 📦 Requirements

These scripts require the following Python packages (already in `requirements.txt`):

```
numpy
pandas
scipy
matplotlib
seaborn
sqlalchemy
```

Plus IVAN's internal modules:
- `ivan.database.session`
- `ivan.sports.nfl.models`
- `ivan.sports.nfl.refactor.stat`
- `ivan.sports.nfl.refactor.player`

---

## 🗄️ Data Requirements

### Required Database Tables
- `NFLPlayer` - Player pool with projections
- `NFLStat` - Historical box score data (minimum 5-10 games recommended)
- `NFLOdds` - Player prop betting lines from various books

### Minimum Data for Visualization
- **Historical stats**: ≥5 games (10+ recommended)
- **Odds lines**: ≥1 betting line (more is better)
- **Stats modeled**: Depends on position (see `STATS` dict in `player.py`)

### Position Requirements
- **QB**: 5 stats (passing_yards, passing_tds, interceptions, rushing_yards, rushing_tds)
- **RB**: 5 stats (rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds)
- **WR/TE**: 3 stats (receptions, receiving_yards, receiving_tds)
- **K/DST**: Currently skipped (TODO in player.py)

---

## 🚀 Performance Notes

### Sample Sizes
- **10,000 samples**: Fast, good for development (~2-5 seconds per player)
- **20,000 samples**: Balanced, recommended for presentations
- **50,000+ samples**: High precision, use for research

### Batch Processing
- Use `--position` flag for automated batch analysis
- Results are cached in database (reuses fitted distributions)
- Parallel processing possible (run multiple scripts)

### Memory Usage
- Each player: ~50-100 MB RAM
- Comparison plots: Scales linearly with number of players
- Large batch jobs: Consider processing in chunks

---

## 🐛 Troubleshooting

### "Player not found"
- Check exact name spelling (case-sensitive)
- Verify player exists in database: `SELECT name FROM nfl.players WHERE name LIKE '%Mahomes%';`

### "Insufficient historical data"
- Player needs ≥5 historical games
- Check: `SELECT COUNT(*) FROM nfl.stats WHERE player_display_name = 'Patrick Mahomes';`

### "No odds found"
- Player needs betting lines in database
- Check: `SELECT COUNT(*) FROM nfl.odds WHERE player = 'Patrick Mahomes';`

### Import errors
- Ensure you're running from project root: `/home/sam-pratt/ivan/`
- Verify IVAN is installed: `pip install -e .`
- Check database connection: `init_db('nfl')` works

### Visualization issues
- For display issues: Try saving to file with `--output` instead
- For memory errors: Reduce `--samples`
- For slow rendering: Close other plot windows

---

## 📝 Output Files

### Image Formats Supported
- PNG (recommended): `--output player.png`
- PDF (vector): `--output player.pdf`
- SVG (web): `--output player.svg`
- JPG: `--output player.jpg`

### Recommended Settings
- **Presentations**: PNG at 300 DPI (default)
- **Publications**: PDF (vector graphics)
- **Web**: PNG at 150 DPI or SVG

---

## 🔬 Technical Details

### Methodology Summary

1. **KDE Fitting**: Gaussian KDE with Scott's rule bandwidth on historical data
2. **Odds Extraction**: De-vigged over/under lines → CDF points
3. **Calibration**: Gaussian kernel smoothing to adjust CDF toward odds points
4. **Sampling**: Inverse CDF method from calibrated distribution
5. **Correlation**: Gaussian copula (Iman-Conover) on ranks
6. **FPTS**: Linear combination with DraftKings scoring weights

### Scoring Weights (DraftKings)
```python
{
    'passing_yards': 0.04,      # 1 pt per 25 yards
    'passing_tds': 4.0,
    'passing_interceptions': -1.0,
    'rushing_yards': 0.1,       # 1 pt per 10 yards
    'rushing_tds': 6.0,
    'receptions': 1.0,          # PPR
    'receiving_yards': 0.1,     # 1 pt per 10 yards
    'receiving_tds': 6.0,
}
```

### Correlation Method
- **Spearman rank correlation**: Robust to outliers
- **Gaussian copula**: Preserves marginal distributions while imposing dependence
- **Cholesky decomposition**: Efficient generation of correlated normals

---

## 🎯 Future Enhancements

### Planned Features
- [ ] Support for K and DST positions
- [ ] Animated GIF showing calibration iterations
- [ ] Interactive HTML output with Plotly
- [ ] Export to LaTeX/Beamer for presentations
- [ ] Batch comparison across multiple weeks
- [ ] Sensitivity analysis to odds/history weights

### Ideas for Extension
- Add game environment factors (weather, opponent strength)
- Show confidence intervals on calibration
- Compare multiple calibration methods
- Visualize full correlation matrix structure
- Add statistical tests for calibration quality

---

## 📚 Related Documentation

- **Main methodology**: `/docs/nba_lateswap_guide.md` (similar concepts for NBA)
- **Copilot instructions**: `/.github/copilot-instructions.md`
- **Core modules**: 
  - `/src/ivan/sports/nfl/refactor/stat.py` - Distribution fitting
  - `/src/ivan/sports/nfl/refactor/player.py` - Player-level modeling

---

## 💡 Tips & Best Practices

### For Presentations
1. Start with a star player everyone knows (e.g., Patrick Mahomes)
2. Show the full pipeline visualization first
3. Then show calibration quality to prove it works
4. Finally compare players to show practical application

### For Research
1. Batch analyze by position to find systematic issues
2. Compare calibration quality across different stat types
3. Test sensitivity to number of historical games
4. Analyze correlation structure across player archetypes

### For Development
1. Use low sample counts (1000-5000) during testing
2. Test with edge cases (rookies, injury returns)
3. Verify calibration improvements are consistent
4. Profile performance on large batches

---

## ❓ Questions or Issues?

For questions about:
- **Methodology**: See core modules in `/src/ivan/sports/nfl/refactor/`
- **Data issues**: Check database schema in `/src/ivan/sports/nfl/models.py`
- **Visualization bugs**: Review matplotlib/seaborn documentation

---

**Happy visualizing! 🏈📊**
