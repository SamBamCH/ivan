# How an NBA player projection is built

Written 2026-10-22 from the code in `src/ivan/sports/nba/projection/`. Each step names the file that does it. The last section lists what the model does not do, because that is where the questions were.

## 1. The props become a stat mean (`market.py`)

Every book posts a line and an over and under price for points, rebounds, assists, threes, steals and blocks (plus alternate lines). For each (book, line) pair:

1. The two prices are turned into implied probabilities and de-vigged, giving P(stat > line). A one-sided quote gets a 0.955 haircut instead.
2. A negative binomial with the stat's calibrated dispersion (`data/nba/models/projection_calibration.json`) is solved for the mean that reproduces that probability. So a 24.5 line at -130 over and 27.5 alt at +180 both yield an estimate of the mean.
3. The player's market mean for the stat is the median of those estimates, once at least 2 pairs exist. The "line" shown on the desk is the median of the main (non-alternate) lines.

Turnovers are never priced, so they always come from history.

## 2. The mean becomes a per-minute rate (`market.py`, `live.py`)

The market mean is divided by the player's *basis minutes*: the minutes the model projected for him, with that day's availability and history-only rates, at the moment the props were fetched. The result is stored in `nba_market_rate` (rate, basis minutes, books, line).

This is the one assumption that matters most: the books price the role as it stands, and the projection treats their mean as rate x minutes. When the minutes move later (a starter is ruled out, you pin minutes on the desk), the rate rides along and fantasy points move linearly. That is what the desk's green rows mean.

## 3. The per-minute rate set (`rates.py`, `engine.py`)

For every player on the roster:

- History rate per stat = his minute-weighted trailing rate, shrunk toward the league rate with 60 pseudo-minutes. A player with 600 trailing minutes is 91% himself; a rookie with none is the league.
- Where the player is priced, the market rate replaces the history rate for that stat (config `projection.market_weight`, default 1.0 = market only). Unpriced stats and unpriced players stay on history.
- fppm = DraftKings weights applied to the rates. That is the "fppm" column.

## 4. Minutes (`minutes.py`, `team_minutes.py`, `rotation.py`)

- Baseline per player: play rate x trailing minutes, plus 6 minutes if he is a confirmed starter who has been coming off the bench (minus 6 the other way). No history: a salary prior.
- Team allocation: available players' baselines are scaled to 241 minutes. A shortfall (someone out) is handed out with weight minutes x (38 - minutes), so a 30-minute player absorbs more than a 5-minute one and a 36-minute player has little room. A surplus comes off the bottom of the rotation. Hand pins are fixed first.
- The distribution of minutes comes from the rotation model (config `projection.minutes_model = rotation`): the rotation grid (last game, a proposal from stint history, or one you drew) is the shape, and each world jitters stint boundaries, draws a game margin from N(spread, 14.75) with 5.2% overtime, and moves starter minutes to the bench in blowouts. Each player's draws are rescaled so their mean equals the allocation. The 10-90 band on the desk is read off these draws.

The usage bump (`rates.py`): after allocation, the roster's projected fantasy total is compared with the team's trailing fantasy production. Half the gap is closed by scaling the *unpriced* players' rates by one common factor, clipped to 0.7 to 1.4. Priced players are never scaled; the books already priced their role. This is how a star's absence raises the rest of the roster's efficiency, not just their minutes.

## 5. Fantasy points (`fpts.py`)

Per world: that world's minutes x the rate per stat gives each stat's mean; each stat is drawn negative binomial with its dispersion; double-double and triple-double bonuses are counted on the joint draw. The projection shown is the mean over worlds and the sd is the sd over worlds. The quantiles are kept for the sampler.

## 6. The team projection and the game environment

There is no separate team projection. The team total on the desk is the sum of the players' means. The game enters in two places only:

- The spread, through the minutes draw: a big favourite's starters lose minutes to the bench in blowout worlds, which lowers their mean minutes a little and widens their band.
- The props, implicitly: whatever the books think about pace, matchup or the opponent's defence is inside the lines they post.

The game total (over/under) is not used anywhere in the player projection. Trailing team fantasy production, not the market's implied team points, is the anchor for the usage bump.

## 7. Correlation (`correlation.py`, `pre_lock/player.py`)

Measured on 2025-26 walk-forward residuals, same-game correlations are near zero: two starters 0.00, starter with bench -0.05, bench with bench -0.03, opponents 0.00. A pair's own history predicts its future at r = 0.08, so pair history is shrunk toward the league table with 200 pseudo-games and barely moves anything.

The lineup generator and the sim draw a game as: one shared margin and overtime draw for both teams, each team's minutes drawn jointly from its grid under that margin, then fantasy points per player through a Gaussian copula with the league correlation table, mapped back to each player's own quantiles so skew and bonus bumps survive. The structural correlation that matters in NBA (a star out raises teammates' minutes) is in the means via the allocation, not in the correlation matrix.

## 8. Where I would look first

1. **The team total is not anchored to the market.** If the books have a team at 128 implied points and its trailing production says 112, nothing moves except what is already in individual props. An implied-points anchor for the usage bump (replace or blend trailing team fpts with a figure derived from spread and total) is the cheapest way to let the game environment in. It mostly affects unpriced players, since priced ones are pinned to the books anyway.
2. **The rate-times-minutes assumption for priced players.** A priced bench player promoted to 34 minutes keeps the per-minute rate the books priced at 18 minutes. Starters' per-minute production usually drops with more minutes. The usage bump cannot correct this because it skips priced players.
3. **Trailing production as the usage anchor** lags roster changes early in the season (trades, new rotations). The salary blend and the market cover some of it.
4. **Correlation is effectively off**, which the backtest supports. If you want a stack-aware sim for a specific reason (game-script bets), the margin draw already ties both teams' minutes together; the player-level copula adds little.
