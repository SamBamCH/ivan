# NBA game simulation: plan

Written 2026-11-07. The goal: replace "every player's fantasy points drawn around his projection" with a simulation of
the game itself, so that the dependence between teammates, between the two teams and between a player's minutes and
what happens in the game comes out of the mechanism instead of being imposed. The NFL has this shape already
(`ivan.sports.nfl.conditioned`: worlds are game-level latents, players conditional on them); the NBA gets its own
because the game is different (five on the floor, a rotation, fouls, possessions shared by whoever is on).

Nothing here touches the live pipeline until the gates at the end pass. Everything lives on branch `nba-game-sim`
in a package of its own and is reached through a config switch that stays off.

## What the sampler does today (so the plan starts from the truth)

`pre_lock.player.generate_player_samples` with `projection.minutes_model = rotation` (live since 2026-10-24), per game:

1. **Game**: one final margin `N(-spread, margin_sd)` and one overtime flag, shared by both teams.
2. **Minutes** (`rotation.sample_worlds`): the skeleton grid's substitution times are jittered once per world, a
   scratched player's stints go to the best substitute off the floor, a Brownian bridge to the drawn margin decides
   garbage time (closers off, end of the bench on), overtime adds five minutes to whoever closes.
3. **Stats** (`fpts.fpts_from_minutes`): each stat is negative binomial at minutes x rate with a fitted dispersion,
   independent across stats and across players given the minutes; double-double bonuses from the joint draw.
4. **Correlation**: the only dependence between players is through the shared margin and the minutes. The measured
   residual correlations are tiny (teammate starters ~0.00, starter with bench -0.05, opponents 0.00; `projection/correlation.py`),
   so this is not far wrong on average.

What it cannot do:

- **The score is not made of the players' points.** A world draws a margin, then draws every player's points
  independently of it. A team's simulated fantasy total is not tied to the total the books posted, and a blowout world
  does not have the favourite's starters scoring the points that made it a blowout.
- **Usage is a multiplier, not a pie.** A star's absence raises his teammates' rates by one factor
  (`rates.apply_usage_bump`, 50% recovery) before the draw. Inside a world nobody absorbs anything: the five on the
  floor do not share a fixed number of possessions.
- **Nothing happens during the game.** Foul trouble, a foul-out, an in-game injury, a coach shortening the rotation in
  a close game: the only in-game event is garbage time. The play-by-play fouls (`nba_foul`, 32k rows) are parsed and
  shown on the desk but not used by the sampler.
- **No game state for the late swap.** The in-game path (`post_lock/ingame.py`) rescales projections; it cannot
  condition the rest of the game on the score and the fouls as they stand.

## The data we have

| Source | Rows | Span | What it gives the sim |
|---|---|---|---|
| `nba_stats` (box scores) | 26,327 player-games | 2025-10-02 .. 2026-11-06 | finals per player per game: the marginals |
| `raw/nba_api/pbp/` (PlayByPlayV3) | 763 games | 2025-10-21 .. 2026-11-06 | every action with its clock, team, player, shot result and running score |
| `nba_stint` | 57,805 | same | who was on the floor when, margin at entry and exit |
| `nba_sub` | 42,379 | same | recorded "A in for B" pairs |
| `nba_foul` | 32,318 | same | each personal foul, count after it, kind |
| `data/nba/input/historical_lines.csv` + `game_odds` | 119 games + this season | Dec 2025, 2026-27 | the books' spread and total per game (`fetch_historical_lines.py` for more) |
| stored props (`odds` by game) | 11 days last season + this season | | the books' player marginals, for the marginal check |

The 763 play-by-play games are the fitting set and the validation set of every layer below. On-floor five-man states
per possession come from the stints joined to the actions; possessions are counted from the actions (a possession
ends on a made shot, a defensive rebound, a turnover, or the last free throw of a trip).

## The model: four layers, each one testable with the layer above plugged in from the truth

The sim is one `World` per draw: for every game, the score path, overtime, the possessions per segment, and for every
player the minutes, the stat counts and the fantasy points. Layers:

### A. Game script (possessions and points)

Per game and world: a pace (possessions, one number for both teams) and each team's points per possession, drawn around
means set by the books' total and spread (pace x efficiency difference = margin; pace x efficiency sum = total), with
variances fitted on the play-by-play games. The score path is built segment by segment (eight 6-minute segments, the
same grid `pbp.rotation_map` uses) so garbage time and the close-game closers read the path, and overtime is the path
ending within a basket. This replaces the Normal margin and the Brownian bridge; the margin's law is unchanged in mean,
the variance is now the sum of pace and efficiency noise, which is what makes a high-total game a wider game.

Check (A alone): on games with stored lines, PIT coverage of the final total and margin, the overtime rate, the
garbage-time onset rate and minute against `nba_stint`'s garbage flags. The lines are the mean; only the variance
and the path shape are fitted, so there is little to overfit.

### B. Rotation under game events (the existing layer, extended)

`sample_worlds` stays the engine: the skeleton grid is the projection and its worlds are the minutes. Three additions,
each driven by the layer A path or by a fitted hazard:

- **Foul trouble and foul-outs.** Each player's personal fouls follow a rate per minute on the floor fitted from
  `nba_foul` (shrunk toward the league by position), drawn per world along his stints. A player at 4 fouls in the third
  quarter or 5 in the fourth sits by a fitted rule (the coach's reaction is measured from the stints: minutes lost after
  the fourth and fifth foul by period); six fouls ends his night. His remaining stints go to his heirs (below).
- **Who inherits a vacated stretch mid-game.** The user's hypothesis: when the centre fouls out the backup centre
  takes most or all of what is left. The pre-game measurement (2026-10-23, [minutes_model_plan.md](minutes_model_plan.md))
  found substitution *pairs* diffuse (a regular's top replacement covers 24% of his exits) and position *groups* carrying
  the bias (a big's minutes go to bigs, guards ~4:1, wings proportionally), which is what `skeleton.inherit` does. A
  mid-game exit for good is a different question from a pre-game absence, so it was measured first
  ([midgame_exits_2026-11-07.md](midgame_exits_2026-11-07.md), 1,468 exits): single exits are noise (no rule beats
  predicting zero gain), the aggregate flow is clear (a big's owed minutes: 40% to bigs, 30% to guards, 8% to wings, a
  fifth lost; guards and wings diffuse to guards and wings), the backup big takes 22% himself (half the group's share, not
  "most or all"), and `sample_worlds`' best-substitute-by-curve rule is the worst of all because the player whose curve is
  highest in those minutes has no room. The heir rule is therefore the **flow matrix across groups, room inside a group**,
  pair habit as a tie-breaker (`fouls.DEFAULT_FLOW`, `flow_split`), for scratches and mid-game exits alike.
- **The path decides the closers.** Garbage time and the closing five already read a margin; they read layer A's
  segment path instead of the bridge, so a world where the bench gets garbage minutes is also the world where the
  favourite's starters scored early and the total ran high.

Check (B given A): the existing walk-forward of minutes (`fit_team_minutes.py --eval`, `backtest_skeleton.py`:
MAE, bias, 10-90 coverage by role) must not worsen, plus two new conditional checks: minutes by final-margin bucket
(starters in 20-point games vs 5-point games, against the stints) and minutes of players who reached 5 fouls.

### C. Usage: the five on the floor share the possessions

Per world, segment and team: the segment's possessions (layer A) are split among the five on the floor (layer B) by
each player's **on-floor shares**, fitted per player from the play-by-play: shots (FGA + 0.44 FTA) per possession on
the floor, turnovers per possession, assists per teammate make, offensive and defensive rebounds per available rebound,
each shrunk toward a position prior with the same pseudo-minutes idea as `rates.shrunk_rates`. Shares renormalise over
whoever is on the floor, so when a star sits the other four absorb his share of the pie in proportion to their own; a
fitted elasticity says how much of the pie really moves (the 2026-10-23 measurement that only ~30% of freed minutes
reach the regular rotation has a usage twin to measure here). This is the mechanism the usage bump approximates.

**The props are the shares for priced players.** A points, rebounds, assists or threes prop is tonight's expected
total for that stat, and divided by the player's expected possessions on the floor (his grid minutes x the game's pace
from layer A) it *is* his per-possession share: no fitted prop-to-usage step, the way the live projection already takes
the book's total over the solved minutes. History fills the shares of unpriced players only, and the team's shares are
then reconciled so the pie adds up to what the lines say the team scores (the NFL found priced lines over-cover the
team by ~5%; the NBA number is measured here). An edit to a player's minutes on the desk moves his possessions, not his
share, which is the right lever.

Check (C given the real minutes and the real possessions): project every player-game in the 763 with his true
minutes and his team's true possessions plugged in; per-stat MAE and bias against the current minutes x rate model
given the same true minutes. This isolates the usage layer completely. Then the same with teammates absent: the
players whose stars sat are where the current model leans on the bump, and the unpriced players are where the public
projection fails most (80% of the field backtest's big ownership misses, 2026-11-05).

### D. Outcomes

Shot attempts become makes by a per-player fitted percentage split into twos and threes, free throws by trips and
percentage, turnovers become the opponent's steals in proportion to the defenders' steal shares, missed shots become
blocks in proportion to the defenders' block shares and then rebounds by the ten players' rebounding shares. Opponent
dependence enters here mechanically: a team that turns it over feeds the other team's steals and possessions. All of
it is drawn at the **segment level as counts** (binomial and multinomial over the segment's possessions), not event by
event, so a slate's draw is vectorised numpy at the current sim's speed; the event-by-event engine that the live sim
would want later is a drop-in behind the same `World` interface, not part of this plan.

**The draw takes a game state.** `World.draw(slate, state=None)`: with a state (the clock, the score, each player's
minutes, stats and fouls so far, who is on the floor, from the nba_api live box score the late swap already stores
under `raw/nba_live/boxscore`) only the remaining segments are simulated and a player's total is what he has plus the
remainder, with the in-play total and spread from the odds API anchoring the rest of the script. That is the late
swap's sim: a game in the third quarter is conditioned on its third quarter instead of rescaled.

Check (D, the whole thing, walk-forward): fantasy-point MAE, bias and 10-90 coverage per player against the live
sampler and against the Stokastic replica (the live model is at parity, MAE 7.44 vs 7.41; the game sim may not be
worse); the **joint** checks the live sampler cannot pass by construction: the team fantasy total's spread, the
pairwise teammate and opponent correlation by role pair (the sim must *produce* the measured table, ~0.00 / -0.05 /
-0.03, from the mechanism, with no correlation imposed), and the correlation between a team's fantasy total and its
score.

## The backtest is one scorecard with a row per layer

The checks above are not separate reports: `gamesim/backtest.py` scores every layer on the same games and prints one
table, the truth plugged in above each row, so a miss shows at the layer that caused it:

| row | truth plugged in | what is scored |
|---|---|---|
| possessions | the lines | pace per team-game and per segment |
| points and margin | the lines, the possessions | points per possession, the final, overtime, garbage onset |
| minutes and rotation | the pre-game pool and the margin path | minutes by role, 10-90 coverage, who was on with whom |
| attempts and usage | true minutes and possessions | shots, free-throw trips, turnovers, rebounds, assists per player |
| makes and outcomes | true attempts | points, threes, free throws, steals, blocks |
| fantasy points | nothing | the marginals and the joint checks (team totals, pair correlations) |
| ownership and the field | the public projection | the field backtest's rows |

A player who is modelled with the right attempts and simply missed his shots that night shows as a miss on the makes row
and nowhere else, which is the point. This scorecard replaces the per-player performance checks of
`backtest_projection.py` and `backtest_skeleton.py` once the layers it scores are the live ones (step 8 removes them).

**On a past slate's backtest screen** the same table is per slate: what we published at the time (the projection
store's `ours` record), what the current code says as of that slate's tips, and what happened. The second column needs
an **as-of runner**: the pipeline fed from the slate's tip records (`tips.read_tip`: the pool with salaries, status and
starters, the props, the game lines, the news and edits since 2026-11-05) with a history cutoff at the tip, written to
the projection store under `rerun:<commit>`. The data exists for every slate since 2026-10-23 (twelve held slates); the
backfilled tips of 10-23 to 11-04 carry no news, so the injury report as of tip is re-derived from the raw Rotowire
captures; last season has eleven days of historical props and lines but no DK pool, so a rerun there scores minutes and
stats, not salaries or ownership. The runner is step 7 below.

## The gate before anything goes live

The standing rule applies ([../validation-and-backtests.md](../validation-and-backtests.md), `feedback_sim_changes_need_backtest`):
no live sim change without the field checks passing. For the NBA that means, on the slates whose real fields we hold
(the standings scrapes, `field_backtest.py`):

1. every real lineup in the field scored under the new draw: the PIT histogram of each lineup's actual score within
   its simulated distribution is flat, and the field's actual top scores sit where the simulated field's top sits
   (the live sampler's field top was 0.49x too flat in the NFL; the NBA number is unknown and gets measured first);
2. the marginal and joint checks of layer D on the same slates;
3. a chooser A/B on the results ledger (`slate_backtest.py`): the lineups we would have chosen under each draw, settled
   against the real contests;
4. speed: a slate draw of 1,000 worlds within the current `slate_draw` time, so the desk's sim and refresh do not slow.

Only then does `config/nba.json → sim.engine` flip from `players` (today's path) to `game`, in its own commit, with
`docs/nba/field-and-simulation.md` and `projections.md` rewritten to describe the game sim as the running system.

## Where the code goes

```
src/ivan/sports/nba/gamesim/
    script.py       layer A: pace and efficiency fit, score paths, overtime, garbage onset
    fouls.py        layer B: foul hazards per player, the sit rule, foul-out heirs; the mid-game exit measurement
    usage.py        layer C: on-floor shares from the play-by-play, renormalisation, elasticity
    outcomes.py     layer D: counts from attempts, opponent coupling
    world.py        the World object; draw(slate) -> World; to_player_samples() -> {player_id: fpts[n]} (the
                    shape generate_player_samples returns, so the field, the contest sim and the chooser need nothing)
    possessions.py  possession counting and five-man states from the stored play-by-play (fitting input for A, C, D)
    backtest.py     the layer checks above, each a function returning a frame and a report
scripts/nba/gamesim.py   fit | backtest --layer A|B|C|D|all [--start --end] | draw --team BOS | inputs --team BOS
tests/nba/test_nba_gamesim_*.py
data/nba/models/gamesim/*.json   fitted parameters, with the fit window and the commit in each file
```

`pre_lock.player.generate_player_samples` becomes a thin wrapper over `World.to_player_samples`, so the field, the
contest sim, the chooser and the desk's sim button do not change; `IVAN_SEED` seeds the game draw as it seeds
everything else. The old draws (the Gaussian copula on stored quantiles, the independent and the team minutes models,
the negative-binomial stats at minutes x rate) and the checks the scorecard replaces are deleted on this branch (step
8), not kept behind a switch: the branch is the gate, and it merges clean or not at all.

## The team page shows the sim's inputs

Every number the game sim fits per player or per team is a panel on the team page, read-only, in this order of
arrival (each panel is its script's `inputs --team` output rendered; `docs/web/nba-desk.md` names both):

- **Who covers whom**: the mid-game heir shares (and the pre-game ones `pbp.sub_shares` already computes) as a small
  matrix of the rotation players, with the counts behind each share, next to the position groups and the group
  budgets the solve uses. This is the panel the user's question about the fouled-out centre is answered on.
- **Fouls**: fouls per 36 on the floor, the chance of reaching 5 and of fouling out at tonight's minutes, minutes
  lost after the 4th and 5th foul in his history.
- **Usage on the floor**: shots, turnovers, assists and rebounds per possession on the floor, with the position prior
  and the shrinkage weight shown, and the five-man states the fit saw (how many possessions with and without each
  teammate).
- **Game script**: the books' total and spread, the fitted pace and efficiency means and spreads, the overtime and
  garbage-time probabilities for tonight.
- **Drawn worlds**: the existing rotation grid view gains the layer A path for a drawn world (score by segment,
  garbage onset) so a drawn grid can be read against the game that produced it.

## Order of work

Each step is one branch merged to `nba-game-sim` (never to master until the gate), with its test and its note in
`docs/nba/notes/`:

1. **Possessions and five-man states** from the stored play-by-play (`possessions.py`), checked against the box
   scores (team FGA, FTA, TOV, points per game must reconcile). Everything else fits on this.
2. **The mid-game exit measurement** (`fouls.py`): who inherits a foul-out or an exit for good; pairs vs groups.
   Written up as a dated note and shown on the team page. This answers the user's hypothesis first because every
   later heir rule depends on it.
3. **Layer A** fit and check on the lined games. More historical lines fetched if 119 games are too few for the
   variance fit (the API is billed; the note says how many were bought).
4. **Layer C** fit and the true-minutes check against minutes x rate. The first result that can change a projection:
   if the shares given true minutes beat the rates, the usage bump has a mechanistic replacement and the backtest says
   by how much.
5. **Layer B** additions (fouls, heirs, the path's closers) and the conditional minutes checks.
6. **Layer D** and the World object; the marginal and joint checks; `to_player_samples` wired behind `sim.engine`;
   the layered scorecard (`backtest.py`) complete.
7. **The as-of runner** and the per-slate scorecard on the backtest screen (three columns per layer).
8. **Removal** of the old draw path and the checks the scorecard replaces (the branch merges clean, not with two
   samplers).
9. **The gate**: field checks, chooser A/B, speed. Merge or do not merge.

Steps 1-4 touch no live code at all. From step 6 the branch's `generate_player_samples` draws from the World, and step
8 removes the old draw; the branch merges to master only when step 9 passes, so master never carries two samplers.
