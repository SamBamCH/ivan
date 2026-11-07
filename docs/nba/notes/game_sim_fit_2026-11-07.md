# Game simulation: what each layer's fit and check said (2026-11-07)

The layers of [game_sim_plan_2026-11-07.md](game_sim_plan_2026-11-07.md) as they were fitted on the stored play-by-play
(763 games, 2025-10-21 .. 2026-11-06) and checked. One section per layer, appended as the layers land; the numbers
quoted on the validation page come from here.

## Possessions (step 1)

`gamesim.possessions` against `nba_stats`, 763 games: field-goal attempts, makes, threes, free-throw attempts and makes
match the box score on every player-game; points 99.8%, rebounds 99.6%, assists 99.4% (the rest are feed gaps and
unresolvable name labels, 0.8% of attributed events). Counted possessions 99.1 per team per game, 0.3 below the classic
estimate FGA - OREB + TOV + 0.44 FTA, the two teams 0.9 apart on average, 6 of 1,526 team-games more than 3 apart.
Both fives whole and the acting player among his five on 95.6% of events.

## Layer A: the game script

Fitted on the 221 games with a line (118 of December 2025 from the historical CSV, 103 of this season from
`game_odds`), `scripts/nba/gamesim.py fit-script`:

| parameter | value | how |
|---|---|---|
| league pace | 98.5 possessions | mean of the team-games |
| pace regression | P = -59.4 + 1.60 x mean trailing pace | slope clipped to 1.6 (the trailing pace is shrunk toward the league, so the slope sits above 1); R² 0.16 |
| pace noise | 4.6% of P (sd 4.5 possessions) | residual of the regression |
| segment count noise | 1.16 possessions | segment possessions around team-game possessions x the segment shape (a Poisson count, sd 3.5, was far too wide: both teams' points ride on the same possessions, so a Poisson segment made the drawn total's sd 30 against a real 16.8) |
| per-possession scoring noise | 1.14 points | segment points around possessions x the team-game's own efficiency |
| lead pull-back κ | 0.0022 points per possession per point of lead | the drawn margin's sd matched to the real 13.6 around the spread; independent possessions alone give 16 |
| pace-efficiency coupling λ | 1.0 | the drawn total's sd matched to the real 16.8 around the line: a fast night is a proportionally less efficient night, so the total does not move with pace; past 1.0 the overshoot widens it again (U-shaped: a grid, not a bisection) |
| shooting-night shift σ_eff | 0 | nothing left for a shared shift once κ and λ are in (the pie and the clock already make teammates' points move together) |
| endgame tie bump | 34% of worlds within 3 points at the buzzer are tied | the drawn overtime rate matched to the real 7.7%; possession noise alone ties 2.7% |
| segment shape | 0.126 0.129 0.122 0.130 0.123 0.128 0.117 0.126 | share of regulation possessions per six-minute segment |

Check (`check-script`, 2,000 worlds per game, each game drawn with its own line):

| | real | drawn |
|---|---|---|
| margin inside the drawn 10-90 band | 80% expected | 80.5%, PIT fifths 43 / 50 / 44 / 42 / 42 |
| total inside the band | 80% expected | 84%, PIT fifths 55 / 46 / 60 / 37 / 23 |
| possessions inside the pace band | 80% expected | 81%, flat |
| pace error | | MAE 3.6 possessions, no bias |
| overtime rate | 7.7% | 7.9% |
| garbage-time rate (the stints' threshold tripped) | 47.1% | 48.0% |
| total bias | | the real finals sit 4.9 points under the lines (January 2026: -6.0 on 108 games; this season -1.6 on 104); the lines, not the draw: the drawn mean is the line by construction |

What the layer leaves to the others: in the World the points are the players' makes (layer D); this module supplies the
pace, the segment shape, each team's efficiency target, the pull-back and the tie rule, and its standalone draw is the
margin path the rotation reads for garbage time and the closers (layer B).
