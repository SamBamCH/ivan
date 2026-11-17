# How the trailing history is weighted (2026-11-16)

Why: on slate 156745 the sim had Nikola Vucevic (ORL, $3,300) at 20.4 fantasy points and 76% of our lineups. Three inputs
stacked: a per-minute rate of 1.11 of which 81% of the weight was his 39 games as a 30-minute starter for Chicago last
season (1.23 a minute there, 0.85 in 12 games for Orlando); a minutes baseline of 14.4 (10.0 a game for Orlando) that put
him ahead of Goga Bitadze (12.1) for the backup-centre row; and the pinned rotation row, which sets the scratch chance to
zero (the mean went from 12.4 to 19.9 when the row was pinned). This note covers the first two.

## The measurement (`scripts/nba/history_weights.py`)

Walk-forward on the box scores, 13,679 player-games from 2025-11-15 (players with five games behind them): the state
strictly before each game predicts the minutes (trailing mean) and the fantasy points at the minutes really played (shrunk
rates alone). MAE / bias. 1,270 player-games fall in a player's first 15 for a new team, 4,037 in his first 15 of a season.

| setting | minutes: all | new team | new season | fpts: all | new team | new season |
|---|---|---|---|---|---|---|
| before: half-lives 5 / 20, no discount | 5.00 / +0.10 | 5.77 / +1.22 | 5.22 / +0.50 | 5.49 / +0.26 | 5.06 / +0.54 | 5.46 / +0.28 |
| **live: 3 / 10, team keeps 0.1 (minutes) and 0.25 (rates), season keeps 0.25 (minutes)** | 4.85 / -0.01 | 5.02 / +0.40 | 4.86 / +0.03 | 5.50 / +0.23 | 5.07 / +0.39 | 5.47 / +0.16 |
| live, minutes half-life 5 | 4.89 / -0.03 | 5.04 / +0.56 | 4.88 / +0.11 | | | |
| live, rate half-life 20 | | | | 5.49 / +0.25 | 5.04 / +0.43 | 5.45 / +0.25 |
| live, nothing kept of another team | 4.85 / -0.02 | 5.04 / +0.26 | 4.86 / -0.01 | 5.51 / +0.22 | 5.16 / +0.32 | 5.50 / +0.15 |
| live, team change only | 4.87 / +0.04 | 5.03 / +0.41 | 4.94 / +0.20 | 5.50 / +0.23 | 5.07 / +0.39 | 5.47 / +0.16 |
| half-lives only (3 / 10) | 4.92 / +0.09 | 5.52 / +0.87 | 5.08 / +0.33 | 5.50 / +0.23 | 5.05 / +0.46 | 5.46 / +0.18 |

- **Minutes: the discounts are a clear gain.** New-team error 5.77 to 5.02 and the +1.2 minute bias mostly gone; the first
  weeks of a season 5.22 to 4.86; everything 5.00 to 4.85. The shorter half-life adds a little on top.
- **Rates: no setting changes the error** (5.49 to 5.51 across half-lives 10 to 30 and every discount). The discount and the
  shorter half-life only trim the bias (new team +0.54 to +0.39). Keeping nothing of another team is worse (5.16).
- **What the weighting cannot fix:** a player who plays well under the minutes behind his rate produces less per minute that
  night whatever the weighting (under 75% of his usual minutes: 0.74 to 0.86 a minute actual against 0.93 to 0.95
  predicted, +1.5 to +1.8 points, identical at half-life 5). Part of that is the night itself (foul trouble, a blowout, a
  bad game cut short), so it is not a projection input as it stands.

Vucevic under the live weighting: minutes 11.3 (was 14.7), 0.975 a minute (was 1.127), so behind Bitadze for the row.

## The rerun gate

`scripts/nba/rerun_slate.py --compare stints -n 400`, 25 slates with tip records, 2,709 player-games, old weighting against
new. The rerun takes the minutes from the tip record, so it sees the rates only: fantasy points MAE 7.595 against 7.602,
bias +0.37 against +0.40, 76.9% against 76.4% inside the 10-90 band; minutes 4.98 against 4.97; every stat row within 0.005.
No change, as the box-score test says for the rates.

## Not done

A pinned bench row is certain to play on the stint path (no scratch chance, cut nights off): the third cause above.
