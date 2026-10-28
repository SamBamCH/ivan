# NBA: how a projection is built

Code: `src/ivan/sports/nba/projection/`. A player is **minutes times per-minute rates**; minutes are
the volume signal (a team hands out 241 minutes, so an absence moves minutes around the roster
instead of being guessed per player), rates come from the books where a player is priced and from
box-score history elsewhere. The desk's grid is the minutes projection: its row sums are the
numbers. Every switch below is in `config/nba.json → projection` with a `_note` beside it.

## The steps

```
props (odds)            ──► market means per stat (market.py) ──► book-implied minutes (book_minutes.py) ─┐
box scores (nba_stats)  ──► trailing minutes, rates, starter share, play rate (history.py) ───────────────┤
DK statuses, Rotowire   ──► availability, named starters                                                 ├─► anchors
desk pins / locked rows ──► held exactly                                                                 ┘
                                                                                                          ▼
                              per-team minutes solve within position groups (solve.py, groups.py)
                                                                                                          ▼
                              the rotation skeleton: a 48-minute grid, five on the floor (skeleton.py, rotation.py, team_profile.py)
                                                                                                          ▼
                              per-world minutes draw (team_minutes.py) × per-minute rates (rates.py) ──► fantasy points by Monte Carlo (fpts.py)
                                                                                                          ▼
                              fpts_mean, fpts_std, 101 quantiles on nba_player; a Gaussian copula per game for the sampler (correlation.py)
```

`engine.project_team()` is the one call from roster + availability + history to a table; `live.py`
(`LiveSlate`) feeds it the live pool, the starters, the market rates, the override files and the
grids, and writes back. The walk-forward backtest feeds it box-score rosters instead
(`walkforward.py`), so the same code is measured.

### 1. Props to market means (`market.py`)

For each (book, line) pair of points, rebounds, assists, threes, steals, blocks (and alternates): the
over/under prices are de-vigged into P(stat > line) (a one-sided quote gets a 0.955 haircut), and a
negative binomial with the stat's calibrated dispersion is solved for the mean that reproduces it.
The player's market mean is the median over pairs once at least two exist. Turnovers are never
priced and stay on history. The solve is one vectorised bisection over every pair's distinct
(line, probability, dispersion) triple (`market.means_from_tails`), a slate's props in well under a
second.

### 2. Market means to per-minute rates (`market.py`, `live.py`)

A market mean is divided by the player's *basis minutes*, the minutes the model projected for him
with that day's availability and history-only rates at the moment the props were fetched
(`live.basis_minutes_for_market`). Stored in `nba_market_rate` (rate, basis minutes, books, line).
When his minutes move later (news, a pin), the rate rides along and the points move linearly.

### 3. Book-implied minutes (`book_minutes.py`)

A minutes market is used as is when a book offers one. Otherwise each priced counting stat's market
mean divided by the player's history rate gives an implied minutes figure; they are averaged with the
rates as weights and scaled by 0.924 (market means over history rates land about two minutes high).
Needs two of points, rebounds, assists. Measured 2026-10-23: mean miss 4.06 minutes against 4.92 for
the minutes model and 5.42 for history alone.

### 4. History (`history.py`)

Everything is "as of" a date, from `nba_stats`: exponentially weighted minutes (half-life 5 games),
minute-weighted per-minute rates (half-life 20), starter share (half-life 3), how often he plays
when dressed (half-life 5). Per team, trailing fantasy production. Players with no history get a
salary prior (`minutes.SALARY_CURVE`; `projection.salary_blend`).

### 5. The minutes solve (`solve.py`, `groups.py`; config `alloc = solve`)

Per team, a small convex quadratic program: every available player brings an anchor (mean, sd):
the books' implied minutes for priced players (sd 4), history for the rest (sd 5, deep bench 6, no
history 8), a desk edit (sd 0.05, held almost exactly). Each position group (big / guard / wing from DK
eligibility) brings a soft target: the team's trailing minutes in that group (half-life 10 games,
spread floor 8, active after 8 games). Minimise the squared standardised misses subject to the sum
being 241 and 0 ≤ x ≤ 42. An absent player has no row, so his minutes flow to his group first (bigs'
minutes go to bigs; measured 2026-10-23) and then to everyone. Deep-bench players enter with their
expected minutes and a wide sd and are never removed by the solve.

The older allocator (`minutes.py`, config `alloc = headroom`) is kept: baseline = play rate × trailing
minutes ± 6 for a starter change, shortfall handed out with weight μ(38 − μ), surplus cut from the
bottom with weight μ·exp(−μ/8). The baseline and `START_SHIFT`/`STARTER_PRIOR_W` (a named starter who
rarely started is pulled 0.6 of the way toward the team's starter minutes) still feed the solve's
history anchors.

### 6. The pool: who is in tonight's rotation (config `depth_cut`, `depth_extra`, `pool`)

`depth_cut = true`: the grid plays the team's measured rotation depth (players with 5+ minutes over
the last 10 games, recency weighted, 8 to 12) plus `depth_extra`; the rest get no minutes and leave the
pool. `pool = last_game` instead takes the rotation of the last ordinary game this group played
together, minus tonight's outs, plus the returners; the depth cut stands when there is no such game.
Backtest before switching: `scripts/nba/backtest_skeleton.py --pool`. A player the cut dropped can be
added back on the desk (Map), starting at his history or 6 minutes.

### 7. The rotation skeleton (`skeleton.py`, `rotation.py`, `team_profile.py`; config `minutes_model = rotation`)

`rotation.curves` gives, per player, the recency-weighted share of recent games he was on the floor
in each of the 48 minutes (from the stint table built by `pbp.py`). Each regular's curve is a slot.
Tonight the slots are filled from the solve and the injury report: an available player keeps his own
slot; an absent owner's slot is inherited by the players the solve says are owed minutes beyond their
own, same position group first; a player with no history takes what he inherits or a flat prior.
`build_grid()` then schedules the game minute by minute, five on the floor, stints kept whole by
hysteresis, the named starters at the tip, breathers after long stretches, locked rows fixed, and a
balance pass moves boundary minutes until each row is within tolerance of the solve's target. **The
grid's minutes are the solve's minutes**; what the desk shows is what the pipeline uses.

`team_profile = true` gives the scheduler the coach's own pattern from his ordinary games this season
(final margin under 15, five named starters), shrunk toward the league by n/(n+6): how long the
starters hold the tip and the Q3 tip before the first sub, how many starters close the half, starter
and bench minutes. Measured 2026-10-26 on 898 team-games: a team's opening span and starter minutes
repeat across its games (r 0.72 and 0.68); who closes does not (r −0.03), so the closers stay a league
rule (the most-played five).

### 8. Minutes as a distribution (`team_minutes.py`)

In one world: the final margin is drawn N(spread, sd) with a 5.2% overtime chance (the spread comes
from `game_odds` through `lines.py`; without a priced game the model falls back to independent
minutes); who plays is drawn from each player's availability; the team's minutes are split by a
softmax of log μ + β[role]·|margin| + ε, with ε ~ N(0, τ[role]). The softmax keeps the team at 241
so one player's extra minutes are someone else's; β < 0 for starters moves their minutes to the bench
in blowouts; τ by role makes a star's draw tight and a bench player's wide. With the rotation model
the grid supplies the shape (stints, closers) that the draw jitters. Parameters: `data/nba/models/team_minutes.json`
(`scripts/nba/fit_team_minutes.py`).

### 9. Rates and the usage bump (`rates.py`)

History rate per stat = minute-weighted trailing rate shrunk to the league with 60 pseudo-minutes
(600 trailing minutes is 91% himself). Where the player is priced, the market rate replaces it
(`market_weight`, default 1.0). Then one factor per team scales the **unpriced** players' rates so
that half the gap between the team's trailing fantasy production and the present roster's projection
closes (clipped 0.7 to 1.4). Priced players are never scaled: the books already priced their role, so
with nothing changed since the fetch a priced player projects to exactly the market mean.

### 10. Fantasy points (`fpts.py`)

Per world: that world's minutes × rate gives each stat's mean; each stat is drawn negative binomial
with its dispersion (`projection_calibration.json`, refit by `backtest_projection.py --calibrate`);
double-double and triple-double bonuses are counted on the joint draw. The projection is the mean
over worlds (`n_sims` 4000 from the CLI, 2000 from the desk's Publish), the sd the sd over worlds, and
the marginal is kept as 101 quantiles on `nba_player.fpts_quantiles` for the sampler.

### 11. Same-game correlation (`correlation.py`, used by `pre_lock/player.py`)

Measured on 2025-26 walk-forward residuals the structure is small: two starters 0.00, starter with
bench −0.05, bench with bench −0.03, opponents 0.00, and a pair's own history predicts its future at
r 0.08, so pair history is shrunk toward the league table by 200 pseudo-games. The sampler draws one
shared margin per game, each team's minutes jointly from its grid under that margin, then fantasy
points through a Gaussian copula mapped back to each player's own quantiles so skew and bonus bumps
survive. The correlation that matters (a star out raises teammates' minutes) is in the means via the
allocation, not in the matrix.

## The public side (`public_projection.py`, `public_field.py`)

Kept apart from ours and never moved by our edits. `pub_fpts` on `nba_player` is a replica of the
projection the field plays to (Stokastic, the provider the field model was fitted on): for priced
players a fitted scale and shift on the DK points implied by the consensus prop means (within 1.13
points of Stokastic held out), for unpriced players a ridge on history fantasy points, salary, starts
and recent minutes (within 3.05). The two-group field and the ownership column are built on it
(`refresh_public`, called by the pool refresh, not by Publish). The desk shows **proj** (ours) next
to **pub**.

## Late news and overrides (`live.py`, `scenarios.py`)

`data/nba/input/minutes_overrides.csv` holds the pins (minutes, fppm, band, fpts, starter calls; 0
minutes is a scratch); `data/nba/input/rotations/<TEAM>.json` the painted grid with its locked rows.
Both are read by every projection, by any entry point. On a consumer the producer's published copies
sit under `data/nba/input/published/` as the base layer and the consumer's own files on top
(`override_layers`). Scenarios (`data/nba/input/scenarios/<TEAM>/`) are named copies of a team's two
files plus the assumptions read off them. `scripts/nba/news.py "Name"` builds the overrides for one
scratch from the terminal; `--conditional` scales teammates toward their "without him" per-minute
history (opt-in: those samples are small and confounded).

## Where to look first when a projection is off

1. `scripts/nba/validate.py review --team XXX`: every player's minutes with where they came from
   (history team and games, trailing minutes, last five games, play rate, market-implied minutes) and
   the flags (new team, no history, DNP-prone, above recent maximum, deep bench projected like a
   rotation player, market minutes far from ours). The desk shows the same flags on the card.
2. The grid: a locked row from a previous slate that was not filed away, or a row the depth cut dropped.
3. The props: a stale fetch (the market rate is forward-looking only from its basis minutes) or a
   player the books have not priced yet (he is on history plus the usage bump).
4. Known gaps, in the order they matter: the team total is anchored to trailing production, not the
   market's implied points (the game total is unused); a priced bench player promoted to 34 minutes
   keeps the per-minute rate the books priced at 18; trailing production lags trades early in the
   season. The 2026-10-22 explainer with the older allocator is in
   [notes/projection_explainer_2026-10-22.md](notes/projection_explainer_2026-10-22.md); the
   measurements behind the solve, the skeleton and the depth cut in
   [notes/minutes_model_plan.md](notes/minutes_model_plan.md).
