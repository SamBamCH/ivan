# NBA: data

One SQLite file, raw payloads on disk, derived files per contest, model parameters, and the stamped
outputs of each run. Layout constants are in `src/ivan/sports/nba/paths.py`; the database is
`data/nba/nba_history.sqlite` unless `IVAN_NBA_DB_URL` says otherwise. The principle (store the raw
inputs and outcomes of every slate, derive everything else) is [../data_framework.md](../data_framework.md).

## Tables (`src/ivan/sports/nba/models.py` and the shared `ivan.core` tables)

| Table | What | Written by | Feed |
|---|---|---|---|
| `nba_player` | **the pool**: one row per DK player of the loaded slate: salary, position, status, DK FPPG (`dk_avg_fpts`), our `minutes_proj`, `fpts_mean`, `fpts_std`, `fpts_quantiles`, `minutes_p10/p90`, `minutes_role`, `p_out`, `rates`, `rotation_cells`; the public `pub_fpts` and `ownership` | the pool refresh (`ingest.store_slate`), Publish / `project_slate`, `project.py`, `news.py` | live |
| `starting_lineups` | Rotowire's expected and confirmed starters, with `fetched_at` | the pool refresh | live |
| `odds` | player props as (book, market, line, prices) | `ingest.fetch_slate` with `--update-odds` / Refresh props | live |
| `nba_market_rate` | per player and stat: market mean ÷ basis minutes, books, line | `live.store_market_rates` after a props fetch | live |
| `game_odds` | moneyline, spread, total per book per game | `ingest.store_game_lines` | live |
| `historical_contest_information`, `payouts` | DK contests and payout tables | `contests.upsert_contests` / `fetch_payouts` / `ensure_contest` (the sim) | live |
| `nba_stats` | box scores, one row per player-game, both seasons | `nightly.py` (`historical.data_fetch.fetch_boxscores`) | reference |
| `nba_stint`, `nba_sub`, `nba_foul` | when each player was on the floor, substitution pairs, personal fouls, from the play-by-play | `nightly.py`, `pbp.py` | stints only |
| `nba_projection_capture` | our projection per player at a moment, keyed by lock: sources `ours` (Publish, ingest), `at_tip` (the archive timer), `dk_avg`, a Stokastic CSV; carries the history context the scorecard needs | `project_slate`, `archive_at_lock.py` | reference |
| `archived_slate`, `archived_blob` | the slate archive: a capture per ingest/refresh/tip with the pool, the props and the game lines as JSON (`ivan.core.archive`) | `ingest.store_slate`, `archive_at_lock.py` | reference (`archived_slate`) |
| `projection_snapshot` | last season's Stokastic projections and our own, one `fpts_mean` per player-day | the old pipeline; `ingest.py --stok-csv` | reference |
| `classic_contest_metadata` | last season's contest bins the old field sampler was fitted on | historical scrape (old) | reference |
| `nba_player_note` | the desk's player notes: draft group, day, player, text, created / updated / cleared | the note button, `notes.py` | no |
| `nba_news` | Rotowire headlines with `first_seen` | `newsfeed.check` (every pool refresh, the news auto box, `newsfeed.py check`) | no |
| `lineup`, `lineup_pool` | the generated candidate pool and the pool snapshot taken at generate | `generate_lineups.py` | no |
| `lineup_simulation_results` | per unique lineup per contest: fpts / payout / ROI / rank per simulation | `sim.py` (replaces per contest) | no |
| `contest_sim_report` | one report per simulated contest (field, payouts, scores, checks) | `sim.py` | no |
| `raw_artifact` | the index of raw payloads (mirror of `raw/index.jsonl`) | `ivan.core.raw.store` | no |

Working tables (`lineup*`, `lineup_simulation_results`, `contest_sim_report`): each slate keeps its own lineups, sims and contest reports: generate, the field sampler and the sim read only the loaded slate's (`field_generation.slate_lineups`), and a switch (`ingest.isolate_slate`) deletes only the work of slates more than 2 days old (`KEEP_SLATE_DAYS`). `scripts/nba/clear_all.py [--yes|--dry-run]`
empties them by hand.

## On disk (`data/nba/`)

```
data/nba/
├── nba_history.sqlite              the database (WAL mode; -wal/-shm beside it)
├── active_slate.json               the draft group the desk has loaded
├── archive_state.json              which tips archive_at_lock.py has recorded
├── capture_state.json              the (disabled) capture tick's state
├── input/
│   ├── entries.csv                 the DraftKings entries export the sim and chooser read
│   ├── minutes_overrides.csv       the desk's pins: player, minutes (0 = scratch), fppm, band, fpts, starter calls
│   ├── rotations/<TEAM>.json       the painted grid per team with its locked rows
│   ├── scenarios/<TEAM>/*.json     named copies of a team's two files + assumptions; _active.json
│   ├── edits/<draft group>/        the above, filed away when another slate is loaded
│   ├── published/                  consumer only: the producer's pins and grids as the base layer
│   ├── historical_lines.csv        closing-ish game lines for past days (fetch_historical_lines.py)
│   └── stok.csv                    a Stokastic export to record as a yardstick (ingest.py --stok-csv)
├── models/
│   ├── projection_calibration.json per-stat dispersion, minutes sd (backtest_projection.py --calibrate)
│   ├── team_minutes.json           τ, β per role (fit_team_minutes.py)
│   ├── correlation.json, correlation_residuals.csv   role-pair correlation table (backtest_projection.py --fit-correlation)
│   ├── field_two_group.json        the field's two-group lineup model (the live sampler)
│   ├── field_ownership.json        the ownership model fitted on real contests (field_model.py fit)
│   ├── field_distribution_model.pkl, marginals_*.pkl   last season's bin sampler (field.sampler = projection_model)
│   └── salary_minutes.json         (when fitted) the salary → minutes prior curve
├── raw/                            every payload as received, gzip, plus index.jsonl (ivan.core.raw)
│   ├── dk/{lobby,draftables,payouts,standings}/
│   ├── rotowire/{lineups,news}/
│   ├── odds/{events,props,lines}/ and {events,props,lines}_hist/
│   ├── nba_api/pbp/<game id>/
│   └── nba_live/{scoreboard,boxscore}/   (late swap --live)
├── derived/
│   ├── contests/<id>/              lineups.parquet, players.csv, field.json from a standings export
│   ├── field_cache/                the two-group field per public-projection key
│   ├── players/experience.json     each player's first NBA season (nba_api), the rookie marks; nightly
│   └── props_hist/                 parquet of the historical props
├── output/
│   ├── projections/projections.<stamp>.csv       the projection artifact of each Publish / ingest
│   ├── sim/<stamp>/contest_<id>.npz + .json      what the sim assumed per contest
│   ├── chosen_entries.{classic,exposures,lineups}.<stamp>.csv, .choose.<stamp>.json, .simroi.<stamp>.npz
│   ├── lateswap.<stamp>.csv + .json
│   └── nba_*_for_brother_*.csv                   hand exports
├── diagnostics/
│   ├── <contest name>_<id>/report.json           the sim report per contest
│   ├── scorecard/                                one CSV + JSON per scored slate
│   └── backtest_projection_*.txt
├── jobs/                           the desk's job logs and json
└── logs/                           nightly.log, archive.log, capture.log
```

Secrets and per-machine settings are in `.env` at the repo root, never under `data/`. The desk's login
cookie secret is `data/.desk_secret` (delete it to log everyone out).

## Which copy is the truth

- **Before lock**: `nba_player` is what the pipeline reads; the desk's memory is what you see; the
  override files are what both are rebuilt from. [game-day.md](game-day.md) has the table.
- **After lock**: the `at_tip` capture is the record of what we projected (the pool may change later
  by a refresh); the `ours` capture before lock is what the scorecard scores.
- **The field**: the derived contest folders are the real fields; everything under `models/` about
  the field is fitted from them.
- **Raw first**: a parser change is a re-run over `raw/`, never a re-fetch (`pbp.py rebuild`,
  `fetch_historical_props.py --rebuild`, `standings.py derive`).
