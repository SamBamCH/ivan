# Architecture: the shared system

Ivan is one Python package (`src/ivan`, installed with `pip install -e .`) with a thin layer of
scripts per sport. Every sport follows the same pipeline shape and shares the same infrastructure.

```
scripts/<sport>/*.py          command-line entry points: argparse around one function in the package
scripts/replay.py             replay archived slates with today's code (any sport with an adapter)
scripts/feed_server.py        serve the data feed over HTTP
scripts/serve.py              the web UI (ivan.web.main): every sport, one password
scripts/password.py           hash the site's password into .env
src/ivan/
├── core/                     sport-agnostic infrastructure (below)
├── database/                 engine/session factory (init_db, get_session, read_sql, safe_commit, add_missing_columns)
├── sports/common/            DraftKings client, the-odds-api client, payout scaling
├── sports/<sport>/           the sport: models (tables), ingest, projection, field generation, simulation, choose, lineup export, validation
└── web/                      the two FastAPI apps and their templates
config/<sport>.json           tracked, machine-independent tuning; config/replay/<sport>.json contest menus; config/tax.json
deploy/                       server bootstrap and hardening, systemd units, push.sh, the Cloudflare tunnel
data/<sport>/                 git-ignored: the SQLite file, raw payloads, inputs, models, outputs, diagnostics, logs
tests/{core,nba,nfl,mma}/     pytest
```

## The pipeline shape every sport follows

```
ingest ──► project ──► generate the field's candidate lineups ──► simulate each entered contest ──► choose ──► (after lock) late swap / settle
```

| Stage | What it reads | What it writes | Shared numerics |
|---|---|---|---|
| ingest | DraftKings draftables + lobby, the-odds-api, a lineups/news source | the pool table (`<sport>_player`), odds, contests + payouts, an archive capture, raw payloads | `sports.common.draftkings`, `sports.common.odds`, `core.raw`, `core.archive` |
| project | the pool, history, odds | per-player distribution on the pool (mean, sd, quantiles), a projections artifact, a capture row | |
| generate | the pool's distributions | the `lineup` pool (correlated draws → ILP per sample, deduped) | `core.seed` |
| simulate | the pool, the entries file, payouts | `lineup_simulation_results` per contest, a sim artifact, `contest_sim_report` | `sports.common.payouts`, `core.artifacts` |
| choose | the sim results | the DK upload CSV, exposures, lineups, choose json, simroi npz | `core.chooser` (greedy quality, empirical-Bayes shrinkage, cross-fit), `core.portfolio` (log-utility book, the NFL slate book) |
| validate / settle | artifacts + box scores + real standings | scorecards, validation reports, slate status | `core.validation`, `core.slates` |

The sport modules mirror each other by name (`ingest.py`, `field_generation.py`, `simulate_contest.py`,
`choose.py`, `lineup_export.py`, `validation.py`, `standings.py`, `contests.py`, `paths.py`), so a
question answered for one sport usually points at the same file in another.

## `ivan.core`

| Module | Purpose |
|---|---|
| `config` | `load_config(sport)` reads `config/<sport>.json`, resolves its `paths` block against the project root, applies `IVAN_ENTRIES_CSV` and `IVAN_<SPORT>_DB_URL`. `.env` at the root is loaded automatically. Resolution order for anything path-like: CLI argument, environment, config, the sport's `paths.py` convention. |
| `utils` | `PROJECT_ROOT`, `normalize_name` and generic helpers. Every path in the project is resolved against the repository root, never the current directory. |
| `raw` | every payload a fetcher receives, kept as received: `data/<sport>/raw/<kind>/<key>/<stamp>.<ext>.gz`, an append-only `index.jsonl` and the `raw_artifact` table. `store()` runs before anything parses, so a parser change is a re-run over files, not a re-fetch. Nothing deletes or rewrites. |
| `archive` | the slate archive (`archived_slate`, `archived_blob`): every pre-lock input a slate was ingested with (the pool, the props, the game lines) as JSON, one capture per ingest, so a slate can be restored and replayed. Nothing modelled is stored here. |
| `artifacts` | where a run leaves what the validation layer reads back: `<output>/projections/projections.<stamp>.csv`, `<output>/sim/<stamp>/contest_<id>.npz + .json`, `chosen_entries.*`. A replay sets `IVAN_RUN_DIR` and the same files land unstamped in the run directory. |
| `clock` | the pipeline's "now": `datetime.now()` live, `IVAN_AS_OF` in a replay, so every stage that asks the time or which games are history goes through one place. |
| `seed` | `IVAN_SEED=<int>` derives a seed for every named draw, so a run is reproducible and two seeds are two independent runs. Unset, every stage keeps its own behaviour. |
| `model_store` | versioned model directories: `data/<sport>/models/` live, `models/asof/<cutoff>/` for as-of sets a replay resolves by the clock so a backtest never uses a model that has seen the slate. |
| `replay` | restores archived slates into a copy of the sport's database, pins the clock, runs the sport's pipeline through a `SportAdapter` (project → generate → simulate → choose), with repeats per slate under their own seed to measure the pipeline's own noise. Only NFL has an adapter today. |
| `validation` | sport-agnostic frames for projected vs actual: `player_report`, `field_report`, `lineup_report`, `pool_report`, `build_checks`. A sport's `validation.py` builds the frames from its artifacts and actuals. |
| `slates` | one status row per slate from the settle job (open / waiting / login_needed / settled / no_contests) for the Backtest page and the nightly loop. |
| `chooser` | the choosers' numerics shared by the sports: greedy quality, `shrink_rows` (DerSimonian-Laird empirical Bayes on per-world ROI), `cash_feature`, `cross_fit_book` (select on even worlds, score on odd: honest ROI). |
| `portfolio` | the book-level choosers: `pick_portfolio` (per-contest log utility over worlds, bankroll in multiples of fees), `pick_book` + `dollar_book_summary` (the NFL "slate" method: one bankroll in dollars across every contest). |
| `feed` | the data feed between two Ivans: `publish` named tables and files as gzipped bundles with a manifest, `push` them by rsync, `pull`/`apply` on the consumer, which replaces exactly those tables and keeps its own edits. Only NBA has a feed spec today. |

## The database

One SQLite file per sport, `data/<sport>/<sport>_history.sqlite`, WAL mode, created on demand
(`init_db(sport)` then the sport's `ensure_schema` or `create_all`). Reference tables (history, captures,
contests) and the machine's working tables (the pool, the lineup pool, sim results, reports) live in the
same file; the working tables hold one slate at a time and are cleared by the next slate or by
`clear_all.py`. `add_missing_columns` adds columns a model gained since the file was created, so a
schema change is a code change plus a restart. SQLite never returns freed pages on its own; the NFL
overnight job runs `VACUUM` when the freelist is large, and `VACUUM INTO` makes a consistent compact copy
for syncing.

## Configuration and environment

`config/<sport>.json` is tracked and machine-independent; every tuning knob has a `_note` beside it
with the date and the measurement that set it. `.env` at the repository root (git-ignored, loaded by
`ivan.core.config` and `ivan.database.session`) holds secrets and per-machine settings:

| Variable | Used by |
|---|---|
| `ODDS_API_KEY` | every ingest that fetches odds |
| `IVAN_ENTRIES_CSV`, `IVAN_<SPORT>_DB_URL` | overrides of the entries file and the database |
| `IVAN_SEED`, `IVAN_AS_OF`, `IVAN_RUN_DIR` | reproducibility, the replay clock, the run directory (set by the replay driver) |
| `IVAN_SYNC_HOST`, `IVAN_REMOTE_ROOT`, `IVAN_NFL_DB` | the NFL reference sync to the server |
| `IVAN_FEED_PUSH`, `IVAN_FEED_URL`, `IVAN_FEED_TOKEN` | the NBA feed (producer destination; consumer source) |
| `IVAN_PASSWORD_HASH`, `IVAN_PASSWORD`, `IVAN_PUBLIC`, `IVAN_SPORTS` | the site's login (the old `IVAN_DESK_*` names still work); which sports `serve.py` mounts |
| `IVAN_DK_USERNAME` | the DraftKings scrapes through the Chrome profile |
| `GLPK_PATH` | the ILP solver (`glpsol`); otherwise PATH, otherwise PuLP's CBC |

## External sources and what they cost

| Source | Used for | Cost / constraint |
|---|---|---|
| DraftKings public endpoints | lobby, draftables, contest detail and payouts | free, no login |
| DraftKings standings export | the real fields after a contest | needs the logged-in Chrome profile `chrome-selenium/dk-profile`; kept about two weeks |
| the-odds-api | player props, game lines, moneylines; historical endpoints | credits: NBA props about 12 per game, historical about ten times live |
| Rotowire | NBA starting lineups and news | free |
| nba_api (stats.nba.com) | box scores, play-by-play | rate limited: capped per run, three runs a day |
| nflverse | NFL box scores and schedules | free |
| NBA live data | the late swap's live box scores | free, no key |

## Tests

```bash
venv/bin/python -m pytest tests            # everything
venv/bin/python -m pytest tests/nba        # one sport
```
