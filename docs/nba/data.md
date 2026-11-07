# NBA: data

The history database, **one directory per slate**, raw payloads on disk, derived files per contest, model parameters,
and the stamped outputs of each run. Layout constants are in `src/ivan/sports/nba/paths.py` and, for everything that
belongs to one slate, `src/ivan/sports/nba/slates.py`; the history database is `data/nba/nba_history.sqlite` unless
`IVAN_NBA_DB_URL` says otherwise. The principle (store the raw inputs and outcomes of every slate, derive everything
else) is [../data_framework.md](../data_framework.md).

**A slate is a directory**, `data/nba/slates/<draft group>/`: its own `slate.sqlite` (the pool, starters, market rates,
notes, lineups, sims, contests), its edits, its published layer, its projection store, its entries and outputs, and
`slate.json` (lock, day, games, when it was refreshed and by what, when it was published or pulled). A slate's engine
opens `slate.sqlite` as *main* and attaches the history database as `hist`; SQLite resolves an unqualified table name
in main first, then in the attached file, so the same queries read the slate's `nba_player` and the shared `nba_stats`.
Two slates never displace each other: the desk holds as many as were opened, scripts take `--slate DG`, the autopilot
refreshes every held slate that has not locked, and a slate's directory is its pre-lock record (nothing deletes it).
The design note is [notes/plan_slates_2026-11-03.md](notes/plan_slates_2026-11-03.md).

## Tables (`src/ivan/sports/nba/models.py` and the shared `ivan.core` tables)

| Table | Where | What | Written by | Feed |
|---|---|---|---|---|
| `nba_player` | slate | **the pool**: one row per DK player of the slate: salary, position, status, DK FPPG (`dk_avg_fpts`), our `minutes_proj`, `fpts_mean`, `fpts_std`, `fpts_quantiles`, `minutes_p10/p90`, `minutes_role`, `p_out`, `rates`, `rotation_cells`; the public `pub_fpts` and `ownership` | the pool refresh (`ingest.store_slate`), `project_slate`, `project.py`, `news.py` | slate bundle |
| `starting_lineups` | slate | Rotowire's expected and confirmed starters, with `fetched_at` | the pool refresh | slate bundle |
| `odds` | history | player props as (book, market, line, prices), each row with its game (`game_id`, `commence_time`, `home_abbr`, `away_abbr` from the odds API event): the day's slates share them | `ingest.store_props` (a props fetch replaces the games it priced and drops games older than 3 days; the restore of a capture) | live; a slate's projection reads only the rows of its own games (`market.fresh_odds`); `scripts/nba/odds_games.py` shows them by game, `--backfill` gives pre-2026-11-07 rows their game |
| `nba_market_rate` | slate | per player and stat: market mean ÷ basis minutes, books, line | `live.store_market_rates` after a props fetch | slate bundle |
| `game_odds` | history | moneyline, spread, total per book per game | `ingest.store_game_lines` | live |
| `historical_contest_information`, `payouts`, `classic_contest_metadata` | slate | the slate's DK contests and payout tables | `contests.upsert_contests` / `fetch_payouts` / `ensure_contest` (the sim) | slate bundle |
| `nba_stats` | history | box scores, one row per player-game, both seasons | `nightly.py` (`historical.data_fetch.fetch_boxscores`) | reference |
| `nba_stint`, `nba_sub`, `nba_foul` | history | when each player was on the floor, substitution pairs, personal fouls, from the play-by-play | `nightly.py`, `pbp.py` | stints only |
| `nba_projection_capture` | history | our projection per player at a moment, keyed by slate and lock: sources `ours` (every refresh and the snapshot Generate, Sim and the late swap take first), `at_tip` (the archive timer), `dk_avg`, a Stokastic CSV; carries the history context the scorecard needs | `project_slate`, `archive_at_lock.py` | reference |
| `archived_slate`, `archived_blob` | history | the slate archive: a capture per refresh and per tip with the pool, the props and the game lines as JSON (`ivan.core.archive`), by slate key | `ingest.store_slate`, `archive_at_lock.py` | reference (`archived_slate`) |
| `projection_snapshot` | slate | last season's Stokastic projections and our own, one `fpts_mean` per player-day | the old pipeline; `ingest.py --stok-csv` | no |
| `nba_player_note` | slate | the desk's player notes: draft group, day, player, text, created / updated / cleared | the note button, `notes.py` | no |
| `nba_news` | history | Rotowire headlines with `first_seen` | `newsfeed.check` (every pool refresh, the news auto box, `newsfeed.py check`) | live |
| `nba_slate_news` | history | which headlines each slate's refreshes saw (slate, headline id, seen_at): a slate's pre-lock news is `nba_news` joined through it | `ingest.check_news` at every refresh | live |
| `lineup`, `lineup_pool` | slate | the generated candidate pool and the pool snapshot taken at generate | `generate_lineups.py` | no |
| `lineup_simulation_results` | slate | per unique lineup per contest: fpts / payout / ROI / rank per simulation | `sim.py` (replaces per contest) | no |
| `contest_sim_report` | slate | one report per simulated contest (field, payouts, scores, checks) | `sim.py` | no |
| `raw_artifact` | history | the index of raw payloads (mirror of `raw/index.jsonl`) | `ivan.core.raw.store` | no |

**Player names** are keyed one way in every NBA table and frame: `ivan.sports.nba.names.normalize_name` = the shared
`ivan.core.utils.normalize_name` (accents, punctuation, case), then the generational suffix dropped (`jr`, `sr`, `ii`,
`iii`, `iv`, `v` at the end: nba_api writes "Jimmy Butler III", DraftKings and Rotowire "Jimmy Butler"), then the NBA
aliases (`names.ALIASES`: "Gregory Jackson" = "GG Jackson", "Yang Hansen" = "Hansen Yang", ...). Every NBA module imports
`normalize_name` from `names`, so the DK pool, Rotowire's starters and news, the props, the box scores, the play-by-play and
the standings exports meet on the same key on input (2026-11-05: starters and headlines keyed with the suffix found no box
scores). Rows stored before the rule are re-keyed once by `scripts/nba/normalize_names.py` (dry run by default, `--apply`
writes): the history tables, every held slate's database (its pool `nba_player.name` included: a past slate never
refreshes, so its pool would keep the old spelling and its players lose their history on the team page), its projection
store records (the frames' index, the rotation inputs) and the days' edit files (pins CSV, drawn grids, scenarios, the
published layer); a row whose new spelling already exists under the same unique key is left and listed. Stop the desk
first or reload the slate afterwards. A new alias goes into `names.ALIASES` and the script is run again.

The slate tables (`slates.SLATE_TABLES`) are created in the slate's file by `slates.ensure_slate_schema`, the rest in the
history file by `ingest.ensure_schema`; a history table created in a slate file would shadow the real one, so the two
lists are explicit. A slate's lineups, sims and reports are its own and nothing clears them for another slate;
`scripts/nba/clear_all.py --slate DG [--yes|--dry-run]` empties one slate's by hand. The one-time move from the one pool
is `scripts/nba/migrate_slates.py` (the old tables stay in the history file as `_migrated_*` until dropped);
`scripts/nba/migrate_edits_by_day.py` then moved each slate's edits under its day (both run by the server bootstrap, idempotent).

## On disk (`data/nba/`)

```
data/nba/
├── nba_history.sqlite              the history database (WAL mode; -wal/-shm beside it)
├── slates/<draft group>/           one slate (ivan.sports.nba.slates; never deleted by the code)
│   ├── slate.json                  draft_group, lock, day, label, games, created_at, refreshed_at + refreshed_by (desk | autopilot | script | feed | archive), published_at (producer) / pulled_at (consumer)
│   ├── slate.sqlite                the slate's tables: the pool, starters, market rates, notes, lineups, sims, contests (the history attached as `hist`)
│   ├── projection/<TEAM>.pkl       the projection store: the slate's projection per team (frame, history-only frame, the grid behind it),
│   │                               written by whatever projects (ingest.project_team), read by the desk; mirrored to the consumer
│   ├── input/
│   │   ├── entries.csv             the DraftKings entries export the sim and chooser read (the desk's upload box)
│   │   └── contest_weights.json    per-contest chooser knobs set on the desk's contests section (contest_weights.py); Choose passes it as --weights-json
│   ├── output/
│   │   ├── projections/projections.<stamp>.csv       the projection artifact of each refresh and snapshot
│   │   ├── sim/<stamp>/contest_<id>.npz + .json      what the sim assumed per contest
│   │   ├── chosen_entries.{classic,exposures,lineups}.<stamp>.csv, .choose.<stamp>.json, .simroi.<stamp>.npz
│   │   └── lateswap.<stamp>.csv + .json
│   ├── tips/<YYYYMMDD-HHMM>/       the record at each game's tip (ivan.sports.nba.tips, written once by archive_at_lock.py): meta.json,
│   │                               players.json (the pool with projections, ownership, grid rows), odds.parquet (the props as stored:
│   │                               the odds table is replaced per player at every refresh), game_lines.json, news.json (the slate's
│   │                               news seen by then), edits/ (a copy of the day's edits). What a backtest of that game's models may read
│   └── results/ledger.json         the money (ivan.sports.nba.results): our entries' result per contest from DraftKings' final exports,
│                                   cached; complete once every contest has a final export, re-checked every 10 minutes until then
├── days/<YYYY-MM-DD>/              the day's edits, shared by its slates (a game is mapped once; the 6:00 and 6:30 slates read the
│   │                               same files, the next day starts clean); a slate reaches them through the `day` in its slate.json
│   ├── edits/
│   │   ├── minutes_overrides.csv   this machine's pins: player, minutes (0 = scratch), fppm, band, fpts, pub, starter calls, a total per stat
│   │   ├── rotations/<TEAM>.json   the painted grid per team with its locked rows
│   │   └── scenarios/<TEAM>/*.json named copies of a team's two files + assumptions, saved on this day; _active.json
│   └── published/                  consumer only: the laptop's edits of the day as the base layer (minutes_overrides.csv, rotations/)
├── archive_state.json              which tips archive_at_lock.py has recorded
├── capture_state.json              the autopilot tick's state (props offsets done per slate, the day's fetches)
├── input/
│   ├── historical_lines.csv        closing-ish game lines for past days (fetch_historical_lines.py)
│   └── stok.csv                    a Stokastic export to record as a yardstick (ingest.py --stok-csv)
├── models/
│   ├── projection_calibration.json per-stat dispersion, minutes sd (backtest_projection.py --calibrate)
│   ├── team_minutes.json           the projection's game constants: scratch probabilities by status and salary, margin sd, overtime rate (hand-kept; measured 2026-10)
│   ├── correlation.json            the measured role-pair correlation table (backtest_projection.py --fit-correlation): the game sim's check target
│   ├── field_two_group.json        the field's two-group lineup model (the live sampler)
│   ├── field_ownership.json        the ownership model fitted on real contests (field_model.py fit)
│   ├── field_distribution_model.pkl, marginals_*.pkl   last season's bin sampler (field.sampler = projection_model)
│   ├── gamesim/script.json         the game script's parameters (gamesim.py fit-script) and the World's calibrated knobs (world_*, calibrate-world), with the fit window and the commit
│   ├── gamesim/teams.json          every team's trailing pace and the league coupling constants the World falls back on (fit-script)
│   └── salary_minutes.json         (when fitted) the salary → minutes prior curve
├── raw/                            every payload as received, gzip, plus index.jsonl (ivan.core.raw)
│   ├── dk/{lobby,draftables,payouts,standings}/
│   ├── rotowire/{lineups,news}/
│   ├── odds/{events,props,lines}/ and {events,props,lines}_hist/
│   ├── nba_api/pbp/<game id>/
│   └── nba_live/{scoreboard,boxscore}/   (late swap --live)
├── derived/
│   ├── contests/<id>/              lineups.parquet, players.csv, field.json from a FINAL standings export (field.json: final, locked_entries)
│   ├── field_backtest/             the field backtest's model field per capture and public projection (a cache)
│   ├── field_cache/                the two-group field per public-projection key
│   ├── gamesim/events.parquet      one row per play-by-play event with its possession bookkeeping and the five on the floor
│   │                               (a slate's reruns by the current code live with the slate: slates/<dg>/rerun/<stamp>-<commit>/ with meta.json
│   │                               (the scorecard, the games), players.json (published, actual) and arrays.npz (every stat x world x player))
│   │                               (ivan.sports.nba.gamesim.possessions; scripts/nba/gamesim.py events, incremental): the game sim's fitting data
│   ├── players/experience.json     each player's first NBA season (nba_api), the rookie marks; nightly
│   └── props_hist/                 parquet of the historical props
├── output/                         cross-slate outputs only (a slate's are under slates/<dg>/output)
│   └── nba_*_for_brother_*.csv                   hand exports
├── diagnostics/
│   ├── <contest name>_<id>/report.json           the sim report per contest
│   ├── scorecard/                                one CSV + JSON per scored slate
│   ├── field_backtest/<stamp>/                   report.txt, contests.csv, players.csv, summary.json (backtest_field.py)
│   └── backtest_projection_*.txt
├── jobs/                           the desk's job logs and json
└── logs/                           nightly.log, archive.log, capture.log
```

Secrets and per-machine settings are in `.env` at the repo root, never under `data/`. The desk's login
cookie secret is `data/.desk_secret` (delete it to log everyone out).

## Which copy is the truth

- **Before lock**: the slate's `nba_player` is what the pipeline reads and what the desk shows: every projection
  (a refresh, an edit on the desk, `project.py`) writes the team's rows and its record in the slate's projection
  store (`slates/<dg>/projection/`, `ivan.sports.nba.projection.store`) in the same step, through
  `ingest.project_team`; the desk only reads the store. The slate's edit files (pins, grids) are what any
  projection is rebuilt from. [game-day.md](game-day.md) has the table.
- **The pre-lock record of a slate**, for backtests: its archive captures (the pool, props and lines at every refresh),
  its `ours` projection captures (one per refresh and per snapshot), its headlines (`nba_slate_news`), and its directory
  (edits, store, entries, outputs) as it stood.
- **After lock**: the `at_tip` capture is the record of what we projected (the pool may change later
  by a refresh); the `ours` capture before lock is what the scorecard scores.
- **The field**: the derived contest folders are the real fields (final exports only: a mid-slate export is
  refused, see [overnight.md](overnight.md)); everything under `models/` about the field is fitted from them and
  checked against them by `backtest_field.py`.
- **Raw first**: a parser change is a re-run over `raw/`, never a re-fetch (`pbp.py rebuild`,
  `fetch_historical_props.py --rebuild`, `standings.py derive`).
