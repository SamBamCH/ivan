# NBA: setup

What a machine needs before the NBA side runs. The shared install (venv, `pip install -e .`,
`.env`) is in the [root README](../../README.md); this page is the NBA part on top of it.

## Roles

| Role | Machine | Fetches | Runs |
|---|---|---|---|
| **Producer** | the laptop | DraftKings, Rotowire, the-odds-api, nba_api, the DK standings scrape | the desk on port 8001, the pipeline, the timers, `feed.publish` + `push` |
| **Consumer** | the server | nothing; takes the feed the producer pushes | the same desk at `/nba` inside the NFL web app, `feed.pull` on a timer |

A machine is a consumer when `IVAN_FEED_URL` is set in `.env`; the desk then never fetches and
its Refresh button pulls the feed instead (`src/ivan/web/nba_desk.py`, `CONSUMER`).

## Environment (`.env` at the repo root)

| Variable | Role | Meaning |
|---|---|---|
| `ODDS_API_KEY` | producer | the-odds-api key; player props cost about 12 credits per game, game lines about 3 per fetch |
| `IVAN_FEED_PUSH` | producer | where `feed.push` rsyncs the bundles, e.g. `ivan-server:/opt/ivan/app/data/feed` |
| `IVAN_FEED_URL` | consumer | `file:///opt/ivan/app/data/feed` when the producer pushes to this disk, or `http://host:8790` for a feed server (then `IVAN_FEED_TOKEN` too) |
| `IVAN_DESK_PASSWORD_HASH` | consumer | the desk login, made by `scripts/desk_password.py --set`; `IVAN_DESK_PASSWORD` plain text is the fallback |
| `IVAN_DESK_PUBLIC=1` | consumer | required when the desk is reached through the Cloudflare tunnel; refuses to serve without a password |
| `IVAN_DK_USERNAME` | producer | the DraftKings login the standings scrape and the late swap use through the Chrome profile |
| `IVAN_NBA_DB_URL` | any | SQLAlchemy URL overriding `sqlite:///data/nba/nba_history.sqlite` |
| `IVAN_ENTRIES_CSV` | any | the entries file, overriding `config/nba.json → paths.entries_csv` |
| `IVAN_SEED` | any | one integer makes every draw in a run reproducible (`ivan.core.seed`) |

## Database

There is no `create_db.py` for NBA. The schema is created on demand by every entry point through
`ivan.sports.nba.ingest.ensure_schema(init_db("nba"))`; to do it explicitly:

```bash
venv/bin/python -c "from ivan.database.session import init_db; from ivan.sports.nba.ingest import ensure_schema; ensure_schema(init_db('nba'))"
```

The models read history that is not fetched on a slate day:

- **Box scores** (`nba_stats`) for this season and last: `scripts/nba/nightly.py --season 2025-26 --max-games 400`
  repeated until it reports nothing new (nba_api is rate limited, the run is capped). Last season's stints
  (`nba_stint`, `nba_foul`, `nba_sub`) come from `scripts/nba/pbp.py fetch --since 2025-10-01`.
- **Model parameters** in `data/nba/models/`: `projection_calibration.json` (per-stat dispersion, minutes sd),
  `team_minutes.json`, `correlation.json`, `field_two_group.json`, `field_ownership.json`,
  `field_distribution_model.pkl`. They are tracked files; the refits are in
  [validation-and-backtests.md](validation-and-backtests.md).

A consumer gets all of this from the reference bundle of the feed instead (`scripts/nba/feed.py pull`).

## Timers (user systemd units)

```bash
deploy/install_nba_timers.sh          # producer: archive-at-tip (every 5 min, 09:00-23:59) + nightly (06:45, 12:45, 18:45) + the autopilot tick (every 15 min)
deploy/install_nba_timers.sh --pull   # consumer: the feed pull every 5 minutes instead
deploy/install_nba_timers.sh --feed   # optional: serve the feed over HTTP on the Tailscale address (scripts/feed_server.py)
systemctl --user list-timers | grep ivan-nba
scripts/nba/autopilot.py status       # the autopilot switch, the last tick, whether a tick would run now
```

The autopilot tick (`ivan-nba-capture.timer`, every 15 minutes, 09:00–23:45) is installed and enabled
but does nothing unless the autopilot switch is on (`scripts/nba/autopilot.py on`, or the desk header's
box): with it off, the desk's Refresh is the only fetch on a slate day. Timers keep running after logout
only with `sudo loginctl enable-linger <user>`. Details in [overnight.md](overnight.md) and
[game-day.md](game-day.md#2-refresh-through-the-day).

## The desk

```bash
venv/bin/python scripts/nba/serve.py                 # http://127.0.0.1:8001, open, no login
venv/bin/python scripts/nba/serve.py --host 0.0.0.0  # on the LAN; set IVAN_DESK_PASSWORD_HASH first
```

On the server the desk is mounted at `/nba` inside the NFL app (`scripts/nfl/serve.py`, unit
`ivan-web`), so a code push is `deploy/push.sh` and the one-time NBA setup there is
`deploy/setup_nba_server.sh`. The server has no autoreload and neither does the laptop's `serve.py`:
restart the process to see code changes.

## Entries file

`data/nba/input/entries.csv` is the DraftKings entries export (`Entry ID, Contest Name, Contest ID,
Entry Fee, ...`). The sim and the chooser read it; the desk has an upload box for it (`/entries`).
Contests it names that the database lacks are fetched by `sim.py` (`ivan.sports.nba.contests.ensure_contest`).

## Chrome profile (producer only)

`chrome-selenium/dk-profile` is a Chrome profile logged in to DraftKings. The standings scrape
(`scripts/nba/standings.py scrape`) and the late swap's field export (`scripts/nba/lateswap.py`,
default `--source draftkings`) open Chrome with it and wait for you to log in when the session has
expired. Nothing else needs it.

## Tests

```bash
venv/bin/python -m pytest tests/nba tests/core
```

`tests/nba/` covers the feed, the desk auth, field generation, the in-game update, the late swap,
lineup export, the live feed, play-by-play, the public projection, the rotation grid, the skeleton
and the contest sim.
