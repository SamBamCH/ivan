# NBA: timers and the overnight work

Three user systemd timers on the producer (plus the autopilot tick, which only acts when its switch is on), one on the consumer, and one job you run by hand.
Units are in `deploy/`, installed by `deploy/install_nba_timers.sh` (rendered with this checkout's
path into `~/.config/systemd/user/`). They are **user** units: they run as you, read `.env`, and keep
running after logout only with `sudo loginctl enable-linger <user>`.

| Timer | Schedule | Runs | Log |
|---|---|---|---|
| `ivan-nba-nightly.timer` | 06:45, 12:45, 18:45 (+ up to 5 min), `Persistent=true` so a missed slot runs at wake | `scripts/nba/nightly.py` | `data/nba/logs/nightly.log` |
| `ivan-nba-archive.timer` | every 5 min, 09:00–23:59 | `scripts/nba/archive_at_lock.py` | `data/nba/logs/archive.log` |
| `ivan-nba-capture.timer` | every 15 min, 09:00–23:45; does nothing unless the **autopilot** switch is on (`scripts/nba/autopilot.py`, the desk header) | `scripts/nba/capture.py` (one slate through `ingest.run_ingest`, props at the offsets, feed publish + push) | `data/nba/logs/capture.log` |
| `ivan-nba-pull.timer` (consumer) | every 5 min | `scripts/nba/feed.py pull` | `journalctl --user -u ivan-nba-pull` |

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

1. **Box scores** for the season (`historical.data_fetch.fetch_boxscores`, nba_api): capped at
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
5. **Feed**: `feed.publish("nba", which="both")` then `feed.push` to `IVAN_FEED_PUSH`, so the server's
   reference tables (box scores, stints, captures, models) follow.

A laptop that wakes with no network yet fails at step 1 with a DNS error and nothing else runs; the
next slot or `systemctl --user start ivan-nba-nightly.service` repeats it. The job is idempotent.

## Archive at tip (`scripts/nba/archive_at_lock.py`)

At each game's tip on the slate the desk holds (the first tip is the slate lock), within 12 minutes
after it, it writes as they stand: `nba_projection_capture` rows (source `at_tip`, lock = that tip)
from `nba_player`, and an `archived_slate` capture (source `at_tip`) of the pool, the stored props and the
game lines. It never fetches and never writes `nba_player`, so it cannot change what the desk sees. A tip
is archived once (state in `data/nba/archive_state.json`). `--dry-run` and `--now <ISO>` for testing.
The late swap and the backtests read these to see what changed after lock.

## The capture tick (`scripts/nba/capture.py`, disabled)

The game-day loop from opening week: every classic draft group in today's lobby that has not locked
gets its DK pool, Rotowire lineups, game lines at the first tick and with props, and player props at
fixed offsets before lock (360/180/90/45/15 minutes, `--offsets`), plus contests and payouts and a feed
publish. Disabled 2026-10-23 because it rewrote the pool under the desk; the desk's Refresh is the
only fetch now. Still runnable by hand (`--only <draft group>`, `--no-props`, `--max-props-per-day 20`).

## The real fields: standings scrape (by hand, producer)

```bash
venv/bin/python scripts/nba/standings.py status --days 3                      # played slates, which contests are exported / derived
venv/bin/python scripts/nba/standings.py scrape --days 3 [--public 3] [--headless] [--no-derive]
venv/bin/python scripts/nba/standings.py scrape --contest-ids 197054966,197054968
venv/bin/python scripts/nba/standings.py derive [--contest-ids ...]            # parse exports without a derived folder
```

Per played slate (an archive capture with a lock in the window) it exports the `--public` largest public
classic contests plus every contest in the entries file, through the logged-in Chrome profile
(`chrome-selenium/dk-profile`; Chrome opens and waits for your login when the session expired; `--headless`
only once the login is fresh). Raw exports go to `data/nba/raw/dk/standings/<contest id>/`, derived
contests to `data/nba/derived/contests/<id>/` (`lineups.parquet` one row per unique lineup with copies,
`players.csv` the side table joined to the pool, `field.json` summary). **DraftKings keeps standings about
two weeks**, so scrape within days. The derived contests are what `calibrate_field.py` and
`field_model.py fit` learn from.

## The feed (`ivan.core.feed`, `scripts/nba/feed.py`)

What the producer publishes so the consumer runs the desk, the generator and the sim on the same inputs
without fetching:

| Bundle | Tables | Files | Published |
|---|---|---|---|
| `live` | `nba_player`, `odds`, `nba_market_rate`, `game_odds`, `starting_lineups`, `historical_contest_information`, `payouts` | the producer's `minutes_overrides.csv` and `rotations/` (as the consumer's `input/published/` layer) | every Publish on the desk, every nightly |
| `reference` | `nba_stats`, `nba_stint`, `nba_projection_capture`, `archived_slate`, `projection_snapshot`, `classic_contest_metadata` | `data/nba/models/`, `data/nba/derived/`, `data/nba/diagnostics/scorecard/`, `data/nba/input/historical_lines.csv` | every nightly |

Never published: the lineup pool, sim results, reports, the consumer's own edits. A bundle is a
gzipped SQL dump plus a manifest with stamps and digests; the consumer replaces exactly those tables.

```bash
venv/bin/python scripts/nba/feed.py publish [--which live|reference|both]    # write data/feed/nba/*
venv/bin/python scripts/nba/feed.py push [DEST]                              # rsync to DEST or $IVAN_FEED_PUSH
venv/bin/python scripts/nba/feed.py pull [--url file:///opt/ivan/app/data/feed] [--no-project] [--loop 5]
venv/bin/python scripts/nba/feed.py apply <dir>                              # apply bundles already on disk
```

On the consumer a pull re-projects only the teams it has edited itself (its edits on top of the
published projection) and leaves the rest exactly as published. State in `data/feed-state/`.
`scripts/feed_server.py` (unit `ivan-feed`) serves the bundles over HTTP with a bearer token for a
consumer that cannot receive rsync.

## Checking that last night happened

```bash
grep -n "box scores for\|scorecards written\|stints\|standings derived\|pushed" data/nba/logs/nightly.log | tail
venv/bin/python scripts/nba/validate.py history                      # the headline numbers per scored slate
venv/bin/python scripts/nba/standings.py status --days 3
ssh ivan-server 'ls -la /opt/ivan/app/data/feed-state/ /opt/ivan/app/data/feed/nba/'   # what was pushed and what the server last applied
```
