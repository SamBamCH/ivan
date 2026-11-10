# NBA: validation, backtests and refits

Projected against actual is the bread and butter: every claim the model made before lock is lined
up with the box score after it. This page lists what exists, what each one measures and how to
run it. Findings with dates are in [notes/](notes/).

## The scorecard (`ivan.sports.nba.backtest.validation`, `scripts/nba/validate.py`)

```bash
venv/bin/python scripts/nba/validate.py scorecard                          # every slate with box scores and no scorecard yet (what nightly runs)
venv/bin/python scripts/nba/validate.py scorecard --draft-group 155647 --force
venv/bin/python scripts/nba/validate.py scorecard --date 2026-10-25
venv/bin/python scripts/nba/validate.py history                            # the headline numbers per scored slate, as a trend
venv/bin/python scripts/nba/validate.py review [--team BOS] [--all] [--min-score 4]   # the current pool: minutes provenance + flags per team
venv/bin/python scripts/nba/validate.py replay --draft-group 155647 [--no-compare]    # re-project a past slate with the salary blend and history only
```

The scorecard takes the last projection capture before the slate's lock (`nba_projection_capture`,
source `ours`, with the history context each row carries: baseline minutes, history team and games,
play rate, fppm, usage multiplier, market fpts, minutes model) and joins the box scores. Every
fantasy-point miss is split into the part the minutes explain and the part the rate explains:

```
fpts_proj − fpts_act = (min_proj − min_act) · fpm_proj + min_act · (fpm_proj − fpm_act)
```

Players are classed by where their minutes came from (same team as history, new team, no history, a
pin); summaries by starter label, priced or not, salary tier and history class; DK's own average and
the books' implied points stand beside as yardsticks; plain-language findings. One CSV + JSON per slate
under `data/nba/diagnostics/scorecard/`, read back by `history`. The desk's player notes that stood at lock
(`ivan.sports.nba.slate.notes.at`: written before it, not cleared by then) are in the JSON as `notes`, in the CSV as a `note`
column, and in the printed report with each player's minutes and points projected against actual; a scorecard
built before a note existed gets it with `--force`. The desk's cards carry the same flags
as `review`.

## Is a day ready to backtest? (`ivan.sports.nba.backtest.readiness`, `scripts/nba/readiness.py`)

```bash
venv/bin/python scripts/nba/readiness.py                 # every day with a held past slate: ready / pending / failed, why
venv/bin/python scripts/nba/readiness.py --slate 156176  # one slate: the box scores, the money, the scrape record
```

Per held past slate: **ready** when the box scores of every team of the slate are on file (`nba_stats` on the slate's day)
and, where we played, the results ledger is complete (a final standings export of every contest of ours); **failed** when
a fetch failed and nothing can be backtested until it is fixed (a standings scrape recorded failures or a needed login,
`slates/<dg>/results/scrape.json`; box scores still missing 36 hours after lock; a ledger that cannot be read);
**pending** otherwise. A day is failed when any of its slates is, ready when all are. The calendar in the bar and the
picker's day list colour the day by it (green, red, none); the status is cached two minutes per process.

## One slate's results (`ivan.sports.nba.backtest.slate`, `scripts/nba/backtest_slate.py`, the desk's backtest view)

Per past slate: our entries' results from DraftKings' **final** standings exports (a contest with only a mid-slate
export or none is listed unscored, never given a number) against the chooser's claim, each contest (drilled into on the
desk: our entries there with their field copies, the field's ownership against ours, the payout curve), and every player's
projection at lock against the box score, with each team's game as played over our grid. Read only. How the numbers are
made is on [../web/nba-desk.md](../web/nba-desk.md#backtest-view-backtest-dgdraft-group-_nba_backtesthtml); what a past
slate is missing (box scores, final exports) is fetched from the same page.
`--date` lists every held slate whose `slate.json` day is that date. The redesign this is the first step of (one lock
snapshot per slate, each date's data on its own, the pre-lock models re-run on it against the actuals) is
[notes/backtest_redesign_2026-11-05.md](notes/backtest_redesign_2026-11-05.md).

## The money (`ivan.sports.nba.backtest.results`, `scripts/nba/results.py`)

```bash
venv/bin/python scripts/nba/results.py                       # every held past slate summed per day and overall
venv/bin/python scripts/nba/results.py --slate 156078        # one slate, per contest
venv/bin/python scripts/nba/results.py --slate 156078 --force
```

What our entries made, from DraftKings' word only: each contest's **final** standings export and the payout table (ties
share the places they cover). Our entries are the ones DraftKings lists under our user name (`config/nba.json`
`backtest.dk_user`, "SamBamCH (3/10)") plus any Entry ID the desk's upload files name; the contests are those the slate's
entries file names (unscored, with their fees, until scraped final) plus any final export where our name appears. Per
slate the result is cached in `data/nba/slates/<dg>/results/ledger.json`: a slate whose every contest has a final export
is **complete** and never recomputed (past results do not change); an incomplete one is re-checked at most every ten
minutes and recomputed only when a newer final export is on disk. The nightly warms the ledgers after the standings
derive, so the first page after a scrape does not pay the parse (about 2 s a slate).

Where it shows, **on this machine only** (a consumer desk, the server, gets none of it: `Sport.ledger` is None there and
the backtest view keeps only the model's columns): the Ivan home tile (the season), the NBA picker's strip under its
heading (the season) and each day's row and heading (that day), and the slate's backtest view (that slate). The model's
checks below are separate from this: the money is the actual, they are the claims.

## The record at each game's tip (`ivan.sports.nba.backtest.tips`, `scripts/nba/tips.py`)

```bash
venv/bin/python scripts/nba/tips.py list --slate 156078
venv/bin/python scripts/nba/tips.py backfill --slate 156078 | --all      # from the archive's at_tip captures (slates before 2026-11-05)
venv/bin/python scripts/nba/tips.py show --slate 156078 --tip 2026-11-04T18:00
```

The source of truth for a backtest of a game: what the system knew when that game tipped, written once by the archive
timer to `data/nba/slates/<dg>/tips/<YYYYMMDD-HHMM>/` (the pool with its projections, ownership and grid rows; the props
as stored; the game lines; the slate's news seen by then; a copy of the day's edits). A model re-run for a game may read
that tip's record, the box scores and stints dated before the slate's day, and the slate's contests, payouts and
entries: nothing dated after the tip. Tips backfilled from the archive captures carry players, props and lines only (no
edits or news were kept then) and say so in `meta.json`. The layout and what is deliberately not stored are in the
module's docstring and [data.md](data.md).

## The field backtest (`ivan.sports.nba.backtest.field`, `scripts/nba/backtest_field.py`)

```bash
venv/bin/python scripts/nba/backtest_field.py                                   # every final derived contest, model field of 3,000 entries
venv/bin/python scripts/nba/backtest_field.py --since 2026-10-25 --field-size 10000 --workers 8
venv/bin/python scripts/nba/backtest_field.py --contest-ids 197118943 --no-model   # lock and softmax ownership only, seconds
```

The real fields of the season (every derived contest from a **final** standings export, [overnight.md](overnight.md))
against what the field model says they should have been, rebuilt from the slate's last archive capture before lock
exactly as the live path builds it: the public projection (`public_projection.project` on the capture's pool, the
props posted for that night, the box scores before that day), then the two-group field
(`field.model.simulate_field` with `data/nba/models/field_two_group.json`) at `--field-size` entries, one per slate,
every contest of the slate compared to it the way the sim slices one field per run. Two more columns stand beside
it: **lock**, the ownership the desk showed when the slate locked (the `at_tip` capture's ownership column, archived
since 2026-10-23), and **softmax**, the ownership model fitted on real contests (`field.model.predict`).

Per contest: ownership MAE over the players the public could roster, MAE over the real top 20, Spearman, how many
of the real top 10 the model also ranks there, and the chalk's concentration (the eight most-owned players' summed
ownership) real against model; then the shape, real subsampled to the model's size against the model's field:
lineup strength under the public projection relative to its optimal (p10/p50/p90, within 5 %/10 %), duplicate
share, max copies, salary left. Across contests: calibration (players binned by predicted ownership against the
real mean), bias by salary tier, by starter and by real ownership, the largest misses. Output under
`data/nba/diagnostics/field_backtest/<stamp>/` (`report.txt`, `contests.csv`, `players.csv`, `summary.json`); the
model field per slate is cached under `data/nba/derived/field_backtest/` by capture, public projection, parameters
and seed, so a second run costs seconds. Findings: [notes/field_backtest_2026-11-05.md](notes/field_backtest_2026-11-05.md);
the sampler's knobs against simpler variants by slate size (one noise knob scaled by the game count matches the real
fields better than the five-knob two-group model; not applied): [notes/field_sampler_ablation_2026-11-05.md](notes/field_sampler_ablation_2026-11-05.md).

## The field's parameters (`scripts/nba/field_model.py`)

```bash
venv/bin/python scripts/nba/field_model.py validate | fit | check
```

Needs derived real contests from the standings scrape ([overnight.md](overnight.md)). `validate` is
leave-one-slate-out on every derived contest against baselines, `fit` writes
`data/nba/models/field_ownership.json`, `check` scores the saved model on slates it was not fitted on (the honest
test; run after every scrape). The two-group field's own parameters (`field_two_group.json`: optimizer share,
provider and personal noise) were fitted 2026-10-24 on the first 29 contests by a one-off that was not kept; the
field backtest above is how they are checked, and a refit is a grid over those knobs scored by it. The generator's
`randomness` 0.25 and floor were set the same way against real fields in 2026-10.

## Historical market data (the-odds-api, billed)

```bash
venv/bin/python scripts/nba/fetch_historical_lines.py --start 2025-10-21 --end 2026-01-31 --dry-run     # game lines, one snapshot per day
venv/bin/python scripts/nba/fetch_historical_props.py --stokastic-days --dry-run                        # props for the days projection_snapshot has Stokastic
venv/bin/python scripts/nba/fetch_historical_props.py --field-days --before 60 --max-credits 3000 --reserve 3000
venv/bin/python scripts/nba/fetch_historical_props.py --rebuild                                         # re-derive the parquet files from stored raw, no requests
```

Historical requests cost about ten times a live one per market (about 60 credits a game for the six
main markets, 120 with alternates); both scripts print the cost first, `--dry-run` only lists days,
`--max-credits` and `--reserve` stop a run. Raw under `data/nba/raw/odds/{events,lines,props}_hist/`,
derived parquet under `data/nba/derived/props_hist/`, lines CSV at `data/nba/input/historical_lines.csv`.
Eleven days of last season's props are stored; they are what the public projection replica and the
book-implied minutes were fitted on.

## Play-by-play checks (`scripts/nba/pbp.py`)

```bash
venv/bin/python scripts/nba/pbp.py check --game-ids 0022600002            # stint minutes vs the box score, per player
venv/bin/python scripts/nba/pbp.py rotation --team NYK --games 10 [--before 2026-10-20]
venv/bin/python scripts/nba/pbp.py rebuild [--since 2026-10-01]           # re-reconstruct stints, fouls, pairs from stored raw (no fetch)
```

## The game simulation's data (`ivan.sports.nba.gamesim`, `scripts/nba/gamesim.py`)

```bash
venv/bin/python scripts/nba/gamesim.py events [--since 2026-10-01] [--rebuild]     # the events store from the stored feeds (incremental)
venv/bin/python scripts/nba/gamesim.py check [--game-ids 0022500001]              # the parse against the box score
venv/bin/python scripts/nba/gamesim.py exits [--team BOS]                         # who inherits a mid-game exit: the heir rules scored
venv/bin/python scripts/nba/gamesim.py fit-script [--no-write]                     # layer A: the game script's parameters from the lined games
venv/bin/python scripts/nba/gamesim.py check-script [-n 2000]                      # layer A: PIT coverage of totals and margins, pace, overtime, garbage time
venv/bin/python scripts/nba/gamesim.py check-usage [--start --end]                 # layer C: shares over the real fives vs per-minute rates, true minutes plugged in
venv/bin/python scripts/nba/gamesim.py check-rotation [--start --end] [-n 300]     # layer B: the rotation draw vs the current sampler, the real margin path plugged in
venv/bin/python scripts/nba/gamesim.py draw-world --slate DG [-n 1000]             # the World on a slate: moments vs the lines, every player's line vs the pool
venv/bin/python scripts/nba/gamesim.py calibrate-world --slate DG [-n 400]         # the World's four game-level knobs against the real spread of totals, margins, team points, overtime
```

`events` turns every game with stints into one row per event (shot, free throw, rebound, turnover, steal, block, foul,
substitution) with the possession it ends and the five on the floor for both teams
(`data/nba/derived/gamesim/events.parquet`, 763 games in under a minute). `check` reconciles it with `nba_stats`:
shots, makes and free throws match exactly, points and rebounds on 99.6-99.8% of player-games (the rest are feed gaps),
counted possessions sit 0.3 per game below the classic estimate FGA - OREB + TOV + 0.44 FTA and the two teams' counts
differ by 0.9 on average (2026-11-07, 763 games). The layers of the game sim fit on this store
([notes/game_sim_plan_2026-11-07.md](notes/game_sim_plan_2026-11-07.md)).

`exits` measures who inherits the minutes of a player who leaves for good mid-game (`gamesim.fouls`): every exit with
four or more minutes left outside garbage time and blowouts, each teammate's gain over his own trailing curve, and six
heir rules scored on a held-out half (pairs, group, the sampler's best-substitute slot, proportional, the fitted flow
matrix, nobody). Finding 2026-11-07: single exits are noise, the aggregate flow is by position group (a big's minutes
40% to bigs, the backup big 22% himself, a fifth lost), and the sampler's slot rule is the worst
([notes/midgame_exits_2026-11-07.md](notes/midgame_exits_2026-11-07.md)). `--team` adds the team's own exits and who
gained; `--csv` writes `data/nba/diagnostics/gamesim/exits.csv` (+ `_heirs.csv`).

`fit-script` / `check-script` are layer A, the game script (`gamesim.script`): each team's trailing pace and the books'
total and spread give the game's expected possessions and each team's points per possession; a world draws the pace,
the possessions per six-minute segment, the points per segment with the lead pulling the efficiencies back (the
leader eases off), garbage time off the margin path and overtime from a tie (close games tie more than possession
noise says). `fit-script` takes every parameter from a moment of the lined games (the historical lines CSV and this
season's `game_odds`, matched to the stored feeds by date and teams; 221 games on 2026-11-07) and writes
`data/nba/models/gamesim/script.json` with the window and the commit. `check-script` draws every lined game with its own
line: the real total and margin should sit inside the drawn 10-90 band 80% of the time with flat PIT fifths, the drawn
overtime and garbage-time rates should match the stints'. Result 2026-11-07 in
[notes/game_sim_fit_2026-11-07.md](notes/game_sim_fit_2026-11-07.md).

`check-usage` is layer C (`gamesim.usage`): each player's share of the possessions he is on the floor for (attempts,
threes, trips, turnovers per own possession; assists per teammate make; rebounds per miss; steals per opponent
possession; blocks per opponent attempt), trailing and shrunk toward his position group, and the team's pie (its own
endings per possession, its shares of misses and makes) that the five on the floor split. The check replays every game
of the window possession by possession with the real fives and scores the shares' expected counts against the real
ones, next to the current model's way (the trailing stat per minute times the true minutes). Result 2026-11-07 in the
same note: level on attempts when the minutes are known (7 of 9 stats a hair better), and the pie removes the bias on
nights a 25-minute teammate sat.

`check-rotation` is layer B (`gamesim.rotation`): the skeleton grid's substitutions jittered once per world and rasterised
to each player's coverage of every minute (five on the floor in every minute of every world, by construction), then the
game happening to the plan: scratches and foul-outs handed to heirs by the measured flow matrix and the room inside a
group, fouls as a process along each player's floor time at his own trailing rate with the coach's measured reaction
after the fourth and fifth, garbage time off the margin path (closers out only while the lead holds, the end of the
bench in), overtime for the five at the buzzer. The check proposes each team-game's grid from the stints before that
date, plugs in the real margin path and overtime, and scores the draw against the box score next to the current sampler
on the same grid: minutes MAE, bias and 10-90 coverage by role, by final margin and for the players who really reached
five fouls. Result 2026-11-07 in the same note.

`draw-world` / `calibrate-world` are the World itself (`gamesim.world`, the sampler the field, the sim and the chooser
read; [field-and-simulation.md](field-and-simulation.md#simulate-contestsimulatepy)). `calibrate-world` sets its four
game-level knobs on a slate's games so the drawn totals, margins, team points and overtime rate have the real spread
(the script's fit holds the real numbers); `draw-world` prints a slate's moments, the per-team reconciliation factor
(how far the pool's points sit from the lines) and every player's simulated line next to the pool's mean.

## The as-of rerun of a past slate (`ivan.sports.nba.backtest.rerun`, `scripts/nba/rerun_slate.py`, the backtest view)

```bash
venv/bin/python scripts/nba/rerun_slate.py --slate 156184 [-n 1000] [--seed 0] [--tip 1800]      # the first tip record unless --tip
venv/bin/python scripts/nba/rerun_slate.py --slate 156184 --list
```

The World drawn from the slate's **tip record** ([data.md](data.md): the pool with its minutes, grid rows and scratch
probabilities as the desk had them, the props and the game lines at the tip) with the history cut at that day (the
per-minute rates rebuilt the way the live projection built them: history before the day, the books' means over the
projected minutes where priced; shooting profiles and foul rates before the day; each team's pace from the events before
the day), then scored against the actuals **one layer at a time**: possessions and the finals per game from the events
store, every player's minutes, attempts, makes and outcomes and fantasy points from `nba_stats`, each as the drawn mean's
MAE and bias, the share of actuals inside the drawn 10-90 band and the mean percentile of the actual, with the error of
what was published at the tip (the projection, the line) beside it; since 2026-11-09 the published projection is itself the
World's mean (the desk draws each game when it projects), so the two fantasy-point columns differ by the inputs' drift
between the tip record's mapping and the rerun's history cut, not by model. The record (`slates/<dg>/rerun/<stamp>-<commit>/`)
adds a **minutes by role** block (planned ≤12 / 12-22 / 22-30 / 30+: bias, MAE, in-band, the actuals below the drawn p10
and above the p90, the share under 8 minutes, the drawn sd against the error's; DNPs counted) and a **pool minutes a
team** row (the World's drawn minutes against what the pool's players really played, and the minutes that went to players
outside the World), the gate for anything that touches who plays; it
keeps every stat of every player in every world, so the backtest view draws each player's distribution with his actual
inside it ([../web/nba-desk.md](../web/nba-desk.md#backtest-view-backtest-viewbacktestdgdraft-group-_nba_backtesthtml)).
A slate archived before 2026-10-23 has no tip record and cannot be rerun. First result (slate 156184, 2026-11-06, 500
worlds): [notes/game_sim_fit_2026-11-07.md](notes/game_sim_fit_2026-11-07.md).

## What does not exist yet

- A replay adapter for NBA (`ivan.core.replay` runs archived slates with today's code; only NFL plugs in),
  so there is no end-to-end "replay a month with the current pipeline" and no settle/tax report.
- A refit of the two-group field's parameters from the field backtest (the one-off fit of 2026-10-24 was not kept).
- Lineup and pool validation (did sim ROI rank lineups the way the real field did) from the sim
  artifacts, which `ivan.core.validation.lineup_report` / `pool_report` are built for; the NBA adapter is
  not written.
- Foul-trouble and blowout sampler models beyond the team minutes draw (deferred 2026-10-24); the game simulation
  that would carry them, with a check per layer, is planned in [notes/game_sim_plan_2026-11-07.md](notes/game_sim_plan_2026-11-07.md).

The decision rule for paying for Stokastic (phase 5 of [notes/plan_2026-27.md](notes/plan_2026-27.md))
waits on a season of `--compare` numbers: the market consensus and Stokastic correlate at 0.994, so
the real test of the odds layer is the slates captured through the season.
