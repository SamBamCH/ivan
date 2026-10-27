# NFL: how a player's distribution is built

Code: `src/ivan/sports/nfl/refactor/` (`player.py`, `stat.py`, `volume.py`, `environment.py`,
`usage.py`, `dst.py`, `spread.py`, `team_correlation.py`) and `correlated_samples.py`. The ingest's
`project_slate` runs it and prints one validation line: how many stat distributions came from a market
versus the salary model and where each K/DST projection came from.

## The odds model (`refactor/stat.py`, `refactor/player.py`)

Per player and stat (passing / rushing / receiving yards, receptions, TDs, ...): a base fit, a KDE
marginal, then every priced line's de-vigged over probability becomes a CDF target and a monotone
quantile mapping bends the marginal through them (`fit_stat_distribution`). An odds outlier filter
drops lines that disagree with the rest. Fantasy points are the DraftKings scoring applied to the
joint stat draws. Players with nothing priced for a stat go to the volume models below; each player's
`volume_source` (market / environment / salary / lines) and `unpriced_stats` are recorded so the
validation splits the error by source.

## Unpriced volume (`config/nfl.json → projection.unpriced_volume`)

- **`salary`** (`refactor/volume.py`): the absence of a yards market is itself the signal. A per
  (position, stat) linear model on the same slate's priced players from DK salary plus anytime-TD
  probability, non-negative slopes, falling back to default coefficients. Poisson TDs, proportional
  tails, `UNPRICED_SHADE` (0.5 since 2026-10-15) shades the result down.
- **`environment`** (the live setting; `IVAN_UNPRICED_VOLUME` overrides for an A/B): the two layers below.

### Layer 1: the game environment (`refactor/environment.py`)

Per team in a game, points, offensive plays and pass rate drawn jointly from the lines (total, spread)
and the teams' tendencies, with the within-game residual dependence (a scoring team runs more, the
trailing team passes more, both teams' play counts move together) estimated from past games. Training
rows are `nfl_schedule` (closing lines, final scores) and `nfl_stats`, strictly before the history
cutoff. Team tendencies are shrunk toward the league and used only where the rolling backtest says
they help (pass rate yes, plays a little, points never: the market prices team strength). In-game
line leaks were fixed 2026-10-15 (`matchup_lines`); one-sided alternate lines are de-vigged with a
power rule. `scripts/nfl/environment.py backtest|grid|slate|draw TEAM OPP -n` are the tools; the
projection step writes `environment.<stamp>.csv` beside the projections.

### Layer 2: usage allocation (`refactor/usage.py`)

For each team and stat (receiving yards, receptions, rushing yards): the team total (the QB's
passing-yards line when one exists, else the environment's dropbacks or rushes times league rates),
minus every mean already settled (priced players' market means, the QB's rushing), floored at 5% of the
total, split among the unpriced RB / WR / TE by depth-rank share priors measured on the previous four
seasons, normalised over the unpriced players on the slate. Writes `usage.<stamp>.csv` (team, stat, total
and source, settled lines, residual, sharers); the Validation page shows **coverage** = settled lines /
team total, an identity on the marginals that needs no actuals. Finding (November 2025 slates): priced
receiving lines already claim a median 105% of the QB line, so the allocation mostly hands out the
floor; reconciling the priced marginals to the team total is the next thing to build. The joint draws
of layer 1 are not consumed by the sampler yet. `scripts/nfl/backtest_usage.py --modes salary,environment`
is the projection-only A/B.

## Kickers and defenses (`refactor/player.py`, `refactor/dst.py`, `dst_scoring.py`)

Kickers are fit from their markets: `player_kicking_points` sets the centre, `player_field_goals` the FG
count; DK's distance bonus is added per made FG at the kicker's own historical rate shrunk to the
league's 0.65. A defense has no market: its points-allowed bracket comes from the opponent's drawn
score (the opponent's implied total plus a bootstrapped residual), takeaways from a linear fit on the
implied totals plus a residual, the two residuals keeping their historical rank correlation
(`dst_odds_params.json`, written by `build_models.py`; `project_dst_samples` at ingest). Without lines a
salary scale. Team-points latents sit in the sampler so DST and the opposing offense move together
(2026-10-01). `nfl_dst_game` (DK DST points per team-game) is rebuilt by the nightly job.

## Predictive spread (`refactor/spread.py`, `sd_calibration.json`)

The fitted marginals are too wide at the top: a flat factor per position leaves the high-sd players
lineups are built from 15 to 25% too wide, which a convex payout ladder turns into a 5 to 7x overclaim
at the top rungs (found 2026-10-19). The live calibration is the **power** form, realized sd = a × model
sd^b, fitted by `build_models.py` (`--sd-since`) and written per position; `fpts_std_model` keeps the
uncalibrated value. The `factor` form is kept.

## Correlation (`refactor/team_correlation.py`, `correlated_samples.py`)

Per team a Spearman matrix over the slots QB, RB1-2, WR1-3, TE1-2, K, DST plus team points, shrunk
toward the league by `LEAGUE_SHRINK_GAMES` (300; recency weighting was backtested and dropped
2026-10-06); cross-team correlation pooled over every opponent; DST is the takeaway part only, its
bracket comes from the opponent's points. Sampling is a Gaussian copula (Spearman → Pearson, near-SPD
fix) through each player's own marginal. `generate_lineups.py` prints a per-game check that the sampled
correlation matches the target; the Contests page's Correlation tab shows it per contest.
`scripts/nfl/backtest_correlation.py --shrink ...` chooses the shrink strength.

## History the models read

`nfl_stats` and `nfl_schedule` from nflverse (`refactor/historical/data_fetch.py`, the nightly job),
`nfl_dst_game`, and the scraped contests. Everything is read "before the history cutoff"
(`actuals.history_cutoff`, the replay clock), so a replayed slate cannot see later weeks.

## Checking a projection

- The Validation page / `scripts/nfl/validate.py --since ... --until ...`: bias, MAE against a
  salary-only baseline, PIT histogram and 10-90 coverage, by position, salary tier, ownership quintile
  and `volume_source`.
- `scripts/nfl/backtest_projections.py [--since 2025-09-01] [--positions K,DST] [--write-calibration]`
  scores every projection snapshot against actual DK points: bias, error, rank correlation, whether the
  stated spread covers outcomes, model sd vs calibrated sd.
- The Models page (`/models`) keeps the per-component scoreboard over commits.
