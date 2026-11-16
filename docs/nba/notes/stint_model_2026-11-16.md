# A stint-by-stint rotation model: what the history supports (2026-11-16)

The idea under test: the desk gives one number per available player (tonight's median minutes) and the game sim
draws the rotation itself. At every exchange each player on the floor has a chance to come out that rises with his
time on, the man who comes in is drawn from the bench that is available, and both chances are read against the game
state (period, margin, fouls). A sitting star's minutes then land on different teammates in different worlds
instead of on one heir.

This note is the measurement before any sim change. Everything is from `scripts/nba/gamesim.py stints-study`
(`ivan.sports.nba.gamesim.stints`) on the 828 stored games (2025-10-21 to 2026-11-15, February to April 2026
missing). Chances are fitted on October to December 2025 and October 2026 and scored on January 2026 and
November 2026; playoff games are in neither.

## 1. When a player comes out

Held-out log loss of "his stint ends in this 20-second step", as terms are added (percent under the base rate):

| term added | gain | total |
|---|---|---|
| period break | 7.5 | 7.5 |
| time on the floor (by starter) | 5.7 | 13.2 |
| period clock (by starter) | 6.0 | 19.2 |
| role (usual-minutes band x position group) | 0.7 | 19.8 |
| the player | 0.9 | 20.7 |
| the player's own curve shape | 0.2 | 20.9 |
| stint number | 0.4 | 21.2 |
| game state (phase x margin, by starter) | 0.6 | 21.8 |
| foul trouble | 0.05 | 21.9 |

- **The decay curve is real.** A starter's chance per minute of coming out is 1-2% in his first two minutes on,
  10% in the fifth, 25% in the ninth. A bench player's climbs from 2% to 30% by the twelfth.
- **The clock matters as much as the time on.** The period clock alone (15.8%) beats time on the floor alone
  (13.2%), and each adds about 6 points given the other. Coaches substitute at set points of the quarter, not only
  when a player has run long. A model with the decay curve and no clock would lose half of what is predictable.
- **The player adds little beyond his role.** A starter's first stint averages 7.4 minutes; the spread between
  players is 1.3 minutes and one player's night-to-night spread is 1.6, so the player explains 39% of the
  variance. Only 5% of first stints run the whole quarter. A pooled curve per role with one shift per player is
  enough; a curve shape per player is not worth its parameters.
- **Game state is small on average and large where it applies.** In the last six minutes a starter comes out at
  37% a minute when leading by 21 or more, 25% when trailing by 13 or more, 9% in a close game. Before that the
  margin barely moves the rate.
- **Foul trouble** (fouls at least the period number plus one, five in the fourth) multiplies the out-rate by 2 to
  2.7 in every period (24% a minute against 9% in the first). It is rare, hence the small log-loss gain.
- **Period breaks are a re-pick.** At the end of the first and third quarters about 43% of the starters on the
  floor and 40% of the bench players come out. At the half 95% of the bench players on the floor come out and 93%
  of the resting starters return.

## 2. When a player comes back

The same ladder for "he comes in in this step": break 6.6, rest (by starter and whether he has played yet) 5.3,
period clock 4.0, role 1.0, player 0.5, game state 1.1, foul trouble 0.2; 18.6% under the base rate in all.

- A resting starter returns at 14-20% a minute after two to eight minutes off, peaking at 24% after eight to ten.
  One who has sat fourteen minutes or more almost never returns (2-3%): he was pulled.
- **Starters do not come back in a blowout.** In the last six minutes a benched starter returns at 27% a minute in
  a close game, 9% at a margin of 13 to 20 and 1% at 21 or more. Earlier in the fourth it is 16% against 5%.
  Bench players move the other way (their return rate doubles in a late blowout).
- A player in foul trouble returns at a quarter of the usual rate in the first half (2% against 8-10% a minute),
  half in the third, and at the usual rate in the fourth.

## 3. Who comes in

35,820 moments a team changed players: 63% one man out, 12% at a period break, the rest several at once. Each rule
was scored on the bench that was really there (about six men), so a usual replacement who was out that night is
already removed and the rest renormalised. Pair histories come from the team's previous 15 games.

| rule | one-for-one: named the man | several at once | at a break |
|---|---|---|---|
| any bench man | 18% | 40% | 33% |
| by usual minutes | 32% | 55% | 53% |
| by rest alone | 32% | 53% | 46% |
| by position group | 30% | 50% | 42% |
| recorded pairs | 41% | 59% | 49% |
| pairs from single swaps only | 41% | 58% | 50% |
| pairs by EM over the swap sets | 43% | 60% | 53% |
| EM pairs + position, times rest | **47%** | **63%** | 51% |

- **History matched against who is available works**, and it is the best rule tested: it names the man 47% of the
  time against 18% by chance, and its mean log chance for the real man is -1.40 against -1.75.
- **The feed's pairs are partly wrong at multi-player swaps, as suspected.** Treating those moments as sets and
  letting the single swaps decide the pairing (EM) beats the recorded pairs (43% against 41%, log chance -1.51
  against -1.58). Throwing the multi-swaps away is no better than keeping them as recorded.
- **Rest adds to the pairs** (43% to 47%): the starter who has sat six minutes is the likely pick whoever went out.
- **Even the best rule misses more often than it hits.** The replacement has to be drawn, not assigned.
- **At a break** usual minutes do as well as any pair rule: it is a lineup choice, not a set of swaps.

## 4. Which past games to use

A player's minutes predicted from his own last 15 games, weighted by recency alone or by recency times how closely
each game's available regulars match tonight's:

| | recency | context-matched (tau 0.15) |
|---|---|---|
| all player-games (12,347) | 4.91 | 4.76 |
| nights a 24-minute regular sat | 5.08 | 4.94 |
| those nights, players under 20 usual minutes | 5.96 | 5.74 |
| starter's first-stint length | 1.17 | 1.16 |

Matching the games helps the **amount** of minutes by about 3% (4% for bench players on a night a regular sits) and
the **shape** of the stints not at all. Sharper matching (tau 0.04) is worse than none: it leaves three or four
effective games. So the stint curves can be fitted on all of a player's recent games; the context belongs in the
minutes number, which the desk supplies, and in the default the desk is offered.

## 5. The two teams

50,432 dead balls marked by a whistle, timeout or violation carry 90% of all substitution moments. The away team
substitutes at 48% of the dead balls where the home team does and 21% where it does not. Within the same kind of
dead ball, period and two-minute clock bin the odds ratio is still 2.1; at the next dead ball it is 1.03.

The teams substitute at the same dead balls, more than the kind of dead ball explains, and neither reacts to the
other afterwards. One dead-ball timeline per game shared by both teams, with some dead balls more open to
substitutions than others, covers it. No model of one coach answering the other is needed.

## 6. A regular sits: where the minutes and points went

289 nights exactly one 28-minute regular sat while the team's other regulars played (116 players, all 30 teams),
against 674 nights nobody sat. A teammate's usual is his minutes per team game over the eight games before.

| | nobody sat | one regular sat |
|---|---|---|
| biggest minutes gain on the team | 15.2 | 18.5 |
| teammates 5+ minutes over usual | 3.3 | 4.5 |
| biggest fantasy-point gain | 20.5 | 23.0 |
| teammates 8+ points over usual | 2.6 | 3.6 |

- **His 32 minutes do not go to one man.** The biggest gainer is only 3 minutes further over his usual than the
  biggest gainer of an ordinary night; what changes is that one or two more teammates run well over. By rank the
  gains are 58%, 40%, 30%, 22% and 16% of his minutes (others lose minutes, so these do not sum to one).
- **By role**: 45% of his minutes go to players who averaged under 12 a team game (deep bench and men returning),
  35% to the usual starters, 7% to the rotation bench.
- **The obvious heir is usually not the top gainer.** Someone new starts on 80% of these nights. He is the top
  minutes gainer on 28% of them and in the top two on 44%; he is the top fantasy-point gainer on 21%.
- **The points go elsewhere than the minutes.** The top points gainer is a usual starter 43% of the time, and is
  the top minutes gainer on 47% of the nights (46% on an ordinary night).
- **It does not repeat.** On the next night of the same absence the top minutes gainer is the same man 41% of the
  time, the top points gainer 23%.

This is the case for drawing the heirs per world. The limit of the measurement: a night's gain over a trailing
average is noisy, which is why every figure sits beside the ordinary-night one.

## 7. Usage

`gamesim.py check-usage` already scores the World's possession split on nights a 25-minute teammate sat: shot
attempts are 0.08 low on 8.65 (against 0.32 low for a per-minute rate). The split among the five on the floor
moves the usage to whoever is out there, so no separate star-off usage term is planned.

## 8. The draw against played games

The model built from the above (`gamesim.stints`: `fit`, `Stepper` / `draw`, `calibrate`; fitted on the 690 regular
season games before 2026-11-01) was run on the 216 team-games of 2026-11-01 to 11-15 with the truth plugged in: who
played, who started, the real margin path, and each man's real regulation minutes as the number steered to
(`gamesim.py check-stints`, 200 worlds a team).

| | real | drawn |
|---|---|---|
| steering: minutes off the target (mean absolute) | | 0.40 |
| stints per player | 3.48 | 3.17 |
| a starter's first stint, minutes | 7.16 | 7.23 |
| real first stint inside the drawn 10-90 band | 80% if calibrated | 84% |
| on the floor by minute, Brier (lower is better) | | 0.138 |
| the same for his own ten-game curve scaled to tonight's minutes | | 0.162 |
| the same for a flat share of the game | | 0.185 |

- **Steering works.** One tilt per player lands the median within 0.4 minutes, including targets far from the
  role (a starter asked for 12, a bench man for 34). Targets that do not add to 240 are scaled to it first; without
  that the tilts run away and the team substitutes every minute.
- **The drawn rotation looks like the played one.** Given the minutes, the model places them better than the
  player's own recent pattern for every role except the 28-minute starters, where the two are level (0.145
  against 0.140): their pattern is personal and the pooled curve does not carry it.
- **First stints are calibrated** (the real one falls evenly across the drawn distribution).
- **Open**: the draw substitutes a little less often than teams do (0.3 stints a player). With the median fixed
  a player's minutes still vary by about 5 minutes (sd) across worlds; that spread has not yet been checked against
  the real spread around a projection, which is the rerun gate's job. Regulation only: overtime is not drawn yet.
- **Cost**: 1,000 worlds of one team in 0.14 seconds, the same order as the current rotation draw; steering a team
  takes about 0.6 seconds.

## What this changes in the model

1. The out-chance is the decay curve **and** the period clock, pooled by role, with one shift per player. The
   shift is the same knob the desk's minutes number steers.
2. Game state and fouls are multipliers on both the out-chance and the return, read from the simulated score.
   They matter in the fourth quarter and in foul trouble and hardly anywhere else.
3. The replacement is a draw from EM pair history plus position flow, times rest, over the men available.
4. A period break is a lineup re-pick by usual minutes and rest, not a set of swaps.
5. Stint shapes use all recent games. Context matching is for the default minutes offered to the desk only.
6. Both teams share one dead-ball timeline per game.
7. No usage term beyond what the World does.
