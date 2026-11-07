# NBA: validation, backtests and refits

Projected against actual is the bread and butter: every claim the model made before lock is lined
up with the box score after it. This page lists what exists, what each one measures and how to
run it. Findings with dates are in [notes/](notes/).

## The scorecard (`ivan.sports.nba.validation`, `scripts/nba/validate.py`)

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
(`ivan.sports.nba.notes.at`: written before it, not cleared by then) are in the JSON as `notes`, in the CSV as a `note`
column, and in the printed report with each player's minutes and points projected against actual; a scorecard
built before a note existed gets it with `--force`. The desk's cards carry the same flags
as `review`.

## Is a day ready to backtest? (`ivan.sports.nba.readiness`, `scripts/nba/readiness.py`)

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

## One slate's results (`ivan.sports.nba.slate_backtest`, `scripts/nba/backtest_slate.py`, the desk's backtest view)

Per past slate: our entries' results from DraftKings' **final** standings exports (a contest with only a mid-slate
export or none is listed unscored, never given a number) against the chooser's claim, each contest, our lineups
with their field copies, and every player's projection at lock against the box score. Read only. How the numbers are
made is on [../web/nba-desk.md](../web/nba-desk.md#backtest-view-backtest-dgdraft-group-_nba_backtesthtml).
`--date` lists every held slate whose `slate.json` day is that date. The redesign this is the first step of (one lock
snapshot per slate, each date's data on its own, the pre-lock models re-run on it against the actuals) is
[notes/backtest_redesign_2026-11-05.md](notes/backtest_redesign_2026-11-05.md).

## The money (`ivan.sports.nba.results`, `scripts/nba/results.py`)

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

## The record at each game's tip (`ivan.sports.nba.tips`, `scripts/nba/tips.py`)

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

## The walk-forward backtest of the projection (`scripts/nba/backtest_projection.py`)

```bash
venv/bin/python scripts/nba/backtest_projection.py [--start 2025-11-15 --end 2026-01-15] [--dnp-available]
    [--grid | --kgrid | --rgrid | --sgrid] [--calibrate] [--compare] [--fit-correlation]
    [--market-day 2025-11-30] [--minutes-model independent|team|both]
```

Every team-game from `--start` on is projected using only games before that date, with availability
and starters from the box score (`projection/walkforward.py`), so it measures the minutes and rate
model and not injury-news timing. `--dnp-available` is the live condition (dressed players may not
play). `--grid` and friends search the model's knobs; `--calibrate` fits the per-stat dispersion and
the minutes sd into `data/nba/models/projection_calibration.json`; `--compare` scores the same
player-games against the Stokastic projections kept in `projection_snapshot`; `--fit-correlation`
refits the role-pair correlation table (`correlation.json`); `--market-day` evaluates market rates
from the stored props of the games that tipped that day (rows without a game: by the day the book posted
them). Reports: `data/nba/diagnostics/backtest_projection_*.txt`.
Result 2026-10-06 under live conditions: level with Stokastic (MAE 7.44 vs 7.41 on 3,410 player-games),
a 50/50 blend beats both, the parameters survive a held-out half and the playoffs.

## The rotation grid as the projection (`scripts/nba/backtest_skeleton.py`)

```bash
venv/bin/python scripts/nba/backtest_skeleton.py --start 2025-12-01 --end 2025-12-21 [--sims 400] [--dnp-available]
    [--no-depth] [--depth-extra 1] [--pool] [--no-profile] [--max-cases N]
```

A/B of the skeleton (`projection/skeleton.py`) against the previous live path (the solve's minutes as
the mean, `rotation.propose`'s grid as the shape) on last season's box scores: minutes and fantasy
error and bias (players with 5+ games), 10-90 coverage, and what the skeleton drops (players it leaves
off the floor who did play). `--pool` adds the last-game pool rule; run it before switching
`projection.pool`. The depth cut's own numbers are in the `_depth_note` of `config/nba.json`.

## The team minutes draw (`scripts/nba/fit_team_minutes.py`)

```bash
venv/bin/python scripts/nba/fit_team_minutes.py [--fit-end 2026-01-31] [--eval-start 2026-02-01 --eval-end 2026-03-15] [-n 300]
    [--no-write] [--no-eval] [--lines data/nba/input/historical_lines.csv]
```

Fits τ (share noise) and β (minutes per point of final margin) per role on the box scores and
checks the draw walk-forward against the independent sampler (minutes MAE of the allocation and of
the draw's mean, 10-90 coverage and width). Writes `data/nba/models/team_minutes.json`. `--lines`
lets the draw know each game's spread (from `fetch_historical_lines.py`).

## The field backtest (`ivan.sports.nba.field_backtest`, `scripts/nba/backtest_field.py`)

```bash
venv/bin/python scripts/nba/backtest_field.py                                   # every final derived contest, model field of 3,000 entries
venv/bin/python scripts/nba/backtest_field.py --since 2026-10-25 --field-size 10000 --workers 8
venv/bin/python scripts/nba/backtest_field.py --contest-ids 197118943 --no-model   # lock and softmax ownership only, seconds
```

The real fields of the season (every derived contest from a **final** standings export, [overnight.md](overnight.md))
against what the field model says they should have been, rebuilt from the slate's last archive capture before lock
exactly as the live path builds it: the public projection (`public_projection.project` on the capture's pool, the
props posted for that night, the box scores before that day), then the two-group field
(`field_model.simulate_field` with `data/nba/models/field_two_group.json`) at `--field-size` entries, one per slate,
every contest of the slate compared to it the way the sim slices one field per run. Two more columns stand beside
it: **lock**, the ownership the desk showed when the slate locked (the `at_tip` capture's ownership column, archived
since 2026-10-23), and **softmax**, the ownership model fitted on real contests (`field_model.predict`).

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

## The field's parameters (`scripts/nba/calibrate_field.py`, `scripts/nba/field_model.py`)

```bash
venv/bin/python scripts/nba/calibrate_field.py --contest-id 196456057 [--randomness 0.15,0.25,0.35,0.5] [--floors 0.85,0.9] [-n 300]
venv/bin/python scripts/nba/field_model.py validate | fit | check
```

Both need derived real contests from the standings scrape ([overnight.md](overnight.md)).
`calibrate_field` describes a real field by two curves measured with our projections at lock (the
ownership of the most-used players and the distribution of the lineups' projected totals below the
optimal), generates candidate pools in process from the slate's capture over a grid of randomness ×
floor and scores each on how closely it reproduces the curves; it is how `generate.randomness` 0.25
and the floor were set. `field_model.py validate` is leave-one-slate-out on every derived contest
against baselines, `fit` writes `data/nba/models/field_ownership.json`, `check` scores the saved
model on slates it was not fitted on (the honest test; run after every scrape). The two-group field's own
parameters (`field_two_group.json`: optimizer share, provider and personal noise) were fitted 2026-10-24 on the first
29 contests by a one-off that was not kept; the field backtest above is how they are checked, and a refit is a
grid over those knobs scored by it.

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
