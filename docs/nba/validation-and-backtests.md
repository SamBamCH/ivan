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
under `data/nba/diagnostics/scorecard/`, read back by `history`. The desk's cards carry the same flags
as `review`.

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
from the props stored for that day. Reports: `data/nba/diagnostics/backtest_projection_*.txt`.
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

## The field (`scripts/nba/calibrate_field.py`, `scripts/nba/field_model.py`)

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
model on slates it was not fitted on (the honest test; run after every scrape).

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
- Lineup and pool validation (did sim ROI rank lineups the way the real field did) from the sim
  artifacts, which `ivan.core.validation.lineup_report` / `pool_report` are built for; the NBA adapter is
  not written.
- Foul-trouble and blowout sampler models beyond the team minutes draw (deferred 2026-10-24).

The decision rule for paying for Stokastic (phase 5 of [notes/plan_2026-27.md](notes/plan_2026-27.md))
waits on a season of `--compare` numbers: the market consensus and Stokastic correlate at 0.994, so
the real test of the odds layer is the slates captured through the season.
