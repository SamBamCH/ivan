# NBA: timers and the overnight work

Three user systemd timers on the producer (plus the autopilot tick, which only acts when its switch is on), one on the consumer, and one job you run by hand.
Units are in `deploy/`, installed by `deploy/install_nba_timers.sh` (rendered with this checkout's
path into `~/.config/systemd/user/`). They are **user** units: they run as you, read `.env`, and keep
running after logout only with `sudo loginctl enable-linger <user>`.

| Timer | Schedule | Runs | Log |
|---|---|---|---|
| `ivan-nba-nightly.timer` | 06:45, 12:45, 18:45 (+ up to 5 min), `Persistent=true` so a missed slot runs at wake | `scripts/nba/nightly.py` | `data/nba/logs/nightly.log` |
| `ivan-nba-archive.timer` | every 5 min, 09:00–23:59 | `scripts/nba/archive_at_lock.py` | `data/nba/logs/archive.log` |
| `ivan-nba-capture.timer` | every 15 min, 09:00–23:45; does nothing unless the **autopilot** switch is on (`scripts/nba/autopilot.py`, the desk header) | `scripts/nba/capture.py` (every held slate that has not locked through `ingest.run_ingest`, each in its own directory, props at the offsets per slate; the server pulls the result every 30 minutes and on its Refresh) | `data/nba/logs/capture.log` |
| `ivan-nba-pull.timer` (the **server**) | every 30 min, persistent | `scripts/nba/feed.py pull`: every slate the laptop published, the shared props, lines and headlines (installed by `deploy/setup_nba_server.sh`) | `journalctl --user -u ivan-nba-pull` |

```bash
systemctl --user list-timers | grep ivan-nba              # next and last runs
systemctl --user start ivan-nba-nightly.service           # run the nightly job now, exactly as the timer does
tail -f data/nba/logs/nightly.log
```

## The nightly job (`scripts/nba/nightly.py`)

```bash
venv/bin/python scripts/nba/nightly.py [--season 2026-27] [--max-games 60] [--no-scorecard]
```

In order, each step guarded so a failure does not stop the next (except the box scores, which are the job):

1. **Box scores** for the season (`data_fetch.boxscores.fetch_boxscores`, nba_api): capped at
   `--max-games` per run because of nba_api's rate limits; three runs a day catch up. Into `nba_stats`.
2. **Scorecards** (`validation.run_pending_scorecards`): for every slate whose box scores have arrived
   and that has no scorecard yet, the last projection capture before lock against the box scores, written
   to `data/nba/diagnostics/scorecard/` ([validation-and-backtests.md](validation-and-backtests.md)).
3. **Play-by-play**: stints for up to 40 games without them since October 1 (`pbp.process_game`, nba_api
   PlayByPlayV3, raw under `nba_api/pbp/`), then fouls and substitution pairs parsed from the stored raw
   feed for up to 200 games each (no fetch). Tables `nba_stint`, `nba_foul`, `nba_sub`. The rotation grids
   and the skeleton read these.
4. **Standings derive** (`standings.derive_pending`): parses every standings export on disk that has no
   derived folder yet. It never fetches; the scrape is yours (below).
5. **Results ledger** (`results.ledger_days`): the money per held past slate from the final exports, cached under
   `slates/<dg>/results/ledger.json` so the picker and the home tile never pay the parse; a complete slate is not touched
   ([validation-and-backtests.md](validation-and-backtests.md#the-money-ivansportsnbaresults-scriptsnbaresultspy)).
6. **Experience** (`experience.fetch`, also `scripts/nba/experience.py fetch`): nba_api's all-players list, one
   request, each player's first NBA season into `data/nba/derived/players/experience.json` (the desk's rookie marks).
7. **Feed**: `feed.publish("nba", which="both")` writes the reference bundle (box scores, stints, captures, models,
   `data/nba/derived`) and every held slate's bundle into `data/feed/nba/`; the server's next pull rsyncs it over (nothing is pushed).

A laptop that wakes with no network yet fails at step 1 with a DNS error and nothing else runs; the
next slot or `systemctl --user start ivan-nba-nightly.service` repeats it. The job is idempotent.

## Archive at tip (`scripts/nba/archive_at_lock.py`)

At each game's tip of every slate this machine holds (the first tip is the slate lock), within 12 minutes
after it, it writes as they stand: **the tip's record on disk**, `data/nba/slates/<dg>/tips/<YYYYMMDD-HHMM>/`
(`ivan.sports.nba.backtest.tips`: the pool with its projections, ownership and grid rows, the props as stored, the game lines,
the slate's news seen so far, a copy of the day's edits; the source of truth a backtest of that game's models reads,
[validation-and-backtests.md](validation-and-backtests.md#the-record-at-each-games-tip-ivansportsnbatips-scriptsnbatipspy)),
`nba_projection_capture` rows (source `at_tip`, lock = that tip) from the slate's `nba_player`, and an `archived_slate`
capture (source `at_tip`) of the pool, the stored props and the game lines. It never fetches and never writes a pool,
so it cannot change what the desk sees. A tip is archived once per slate (state in `data/nba/archive_state.json`; a
tip record already on disk is left alone). `--dry-run` and `--now <ISO>` for testing. The late swap and the backtests
read these to see what changed after lock. Slates archived before 2026-11-05 get their tip records from the captures
with `scripts/nba/tips.py backfill --all` (players, props and lines; no edits or news were kept then).

## The autopilot tick (`scripts/nba/capture.py`)

With the autopilot switch on (`scripts/nba/autopilot.py on`, the desk header), every 15 minutes: every slate this
machine holds that has not locked (`slates.open_slates`) gets its DK pool, statuses and Rotowire lineups refreshed
through `ingest.run_ingest` (one transaction on the slate's own database), the game lines at the first tick and with
props, player props at fixed offsets before its lock (360/180/90/45/15 minutes, `--offsets`; at most
`--max-props-per-day 20` fetches a day across slates), and its contests and payouts. Nothing is published by the tick:
the server pulls. A slate nobody opened (on the desk or with `ingest.py --draft-group`) is not refreshed. It stands
down while a desk job runs. `--only <draft group>`, `--no-props`, `--force` (ignore the switch) by hand.

## The real fields: standings scrape (by hand, producer)

```bash
venv/bin/python scripts/nba/standings.py status --days 3                      # played slates, which contests are exported / derived
venv/bin/python scripts/nba/standings.py scrape --days 3 [--public 3] [--headless] [--no-derive]
venv/bin/python scripts/nba/standings.py scrape --contest-ids 197054966,197054968
venv/bin/python scripts/nba/standings.py derive [--contest-ids ...]            # parse final exports without a derived folder
```

Per played slate (an archive capture with a lock in the window) it exports **every contest we played** (the
slate's own entries file, `data/nba/slates/<dg>/input/entries.csv`; `--entries FILE` names them for a slate this
machine does not hold) **plus the `--public` largest public classic contests** whether we played them or not,
through the logged-in Chrome profile (`chrome-selenium/dk-profile`; Chrome opens and waits for your login when the
session expired; `--headless` only once the login is fresh). `status` marks each contest ours or public and final,
mid-slate or pending. Until 2026-11-05 the entries file was read from the pre-split shared path, which no longer
existed, so only the public contests were scraped and the desk's backtest scored the rest from the late swap's
mid-slate fetch. Raw exports go to `data/nba/raw/dk/standings/<contest id>/`, derived
contests to `data/nba/derived/contests/<id>/` (`lineups.parquet` one row per unique lineup with copies,
`players.csv` the side table joined to the pool, `field.json` summary with `final` and `locked_entries`).
**DraftKings keeps standings about two weeks**, so scrape within days. The script runs on any day, open slate or
not (`slates.open_any`: the newest held slate's session when none is open today). The derived contests are what
`field_model.py fit` and `backtest_field.py` learn from and are checked against.

`scrape` notes its outcome on every held slate it touched (`standings.record_scrape` → `slates/<dg>/results/scrape.json`:
the contests, which exported, which failed, whether a login was needed); a failure there turns the day **red** on the
calendar and the picker until a later scrape succeeds (`ivan.sports.nba.backtest.readiness`, [../web/README.md](../web/README.md)).

**Only a final export is a field.** The late swap's standings fetch (a minute after each tip) stores an export
too, taken while games were on: the slots of players not yet tipped read `LOCKED`. `standings.export_is_final`
calls an export final when at most 2 % of its entries show a locked slot (`PARTIAL_MAX_SHARE`); `status` lists a
contest whose only export is such a one as **mid-slate** and `scrape` exports it again, `derive` refuses it and
removes a derived folder that was built from one (2026-10-24 .. 11-04 the nightly had derived 31 of them as fields:
16,812 entries into 43 lineups). Contest rows are read from wherever the slate's live since the split into one
database per slate (`standings.slate_contests`: the slate's own file, else the history file's pre-split copy).

## The feed (`ivan.core.feed`, `scripts/nba/feed.py`)

What the producer publishes so the consumer runs the desk, the generator and the sim on the same inputs
without fetching:

| Bundle | Tables | Files | Published |
|---|---|---|---|
| `live` | the history's shared live tables: `odds`, `game_odds`, `nba_news`, `nba_slate_news` | | on every pull's request, every nightly |
| `slates/<dg>/` (one per held slate) | from its `slate.sqlite`: `nba_player`, `starting_lineups`, `nba_market_rate`, `historical_contest_information`, `payouts`, `classic_contest_metadata` | its `edits/` (as the consumer's `published/` layer), its `projection/` store (mirrored), its `slate.json` | with the live part; `publish --slate DG` one slate (fetched first when not held) |
| `reference` | `nba_stats`, `nba_stint`, `nba_foul`, `nba_sub` (the game simulation's foul rates and substitutions), `nba_projection_capture`, `archived_slate` | `data/nba/models/`, `data/nba/derived/`, `data/nba/diagnostics/scorecard/`, `data/nba/input/historical_lines.csv` | every nightly |

Never published: a slate's lineups, sims and reports, the consumer's own edits. A bundle is a gzipped SQL dump plus a
manifest with stamps and digests (per slate: the tables', the edits' and the store's); the consumer replaces exactly
those tables in its own copy of the slate. A slate the producer no longer holds leaves the manifest; the consumer keeps
its copy.

```bash
venv/bin/python scripts/nba/feed.py publish [--which live|reference|both] [--slate DG]   # write data/feed/nba/*
venv/bin/python scripts/nba/feed.py pull [--no-project] [--slate DG] [--loop 30]        # the consumer: ssh publish + rsync + apply (IVAN_FEED_SSH), or --url
venv/bin/python scripts/nba/feed.py apply <dir>                                          # apply bundles already on disk
```

On the consumer a pull (the 30-minute timer's, a slate page's Refresh from laptop, or the bar picking a slate the
server does not hold) applies every slate bundle whose stamps moved into that slate's own directory (created when new:
the feed tables replaced, the laptop's edits as `published/`, the store mirrored, `slate.json` merged with
`pulled_at`), then `ingest.after_pull` re-projects only the teams this machine edited on each (its edits on top of the
published projection); the rest stays exactly as published. State in `data/feed-state/` (`nba.json`: stamps per part
and per slate, the last check, the producer's note from the manifest). The slate page's header shows when the laptop
last refreshed the slate and by what, and when the server last pulled it.
`scripts/feed_server.py` (unit `ivan-feed`) serves the bundles over HTTP with a bearer token for a
consumer that cannot receive rsync.

## Checking that last night happened

```bash
venv/bin/python scripts/nba/readiness.py                              # every past day: ready / pending / failed and why (the calendar's colours)
grep -n "box scores for\|scorecards written\|stints\|standings derived\|pushed" data/nba/logs/nightly.log | tail
venv/bin/python scripts/nba/validate.py history                      # the headline numbers per scored slate
venv/bin/python scripts/nba/standings.py status --days 3
ssh ivan-server 'ls -la /opt/ivan/app/data/feed-state/ /opt/ivan/app/data/feed/nba/'   # the laptop's feed as the server last pulled it, and what it applied
```
