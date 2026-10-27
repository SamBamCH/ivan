# Ivan

Daily-fantasy analytics for DraftKings. For a slate: ingest the players and the betting markets,
fit every player's fantasy-point distribution, generate a correlated field of lineups, simulate the
contests you entered against that field, and pick the lineups to upload. After the games: settle,
validate every component against what happened, replay past slates with today's code.

| Sport | Status | Start here |
|---|---|---|
| NBA | live, 2026-27 (the desk, the projections, the feed to a second machine) | [docs/nba/](docs/nba/README.md) |
| NFL | live, showdown and classic (the five-step pipeline, the web UI, the backtests) | [docs/nfl/](docs/nfl/README.md) |
| MMA | live, CSV-driven odds model | [docs/mma/](docs/mma/README.md) |
| MLB, LoL | dormant | [docs/mlb/](docs/mlb/README.md), [docs/lol/](docs/lol/README.md) |

Everything shared (layout, `ivan.core`, the database, config, deployment, the web apps) is in
[docs/](docs/README.md).

## Setup

```bash
git clone git@github.com:SamBamCH/ivan.git && cd ivan
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .     # full install; requirements-server.txt is the thin one for the server
cp .env.example .env                                     # ODDS_API_KEY=... (the-odds-api); other variables per docs/architecture.md
python scripts/nfl/create_db.py                          # NFL database; the NBA schema is created on first use
```

Python 3.12, `sqlite3` and `rsync` on the PATH; PuLP's bundled CBC solves lineups out of the box (GLPK is
used when `glpsol` is on the PATH or `GLPK_PATH` is set). Scrapes of finished contests need Chrome with a
logged-in DraftKings profile in `chrome-selenium/dk-profile`. Every path in the project resolves
against the repository root, so scripts run from anywhere.

## Run

```bash
# NBA: a slate day on the laptop (docs/nba/game-day.md)
python scripts/nba/serve.py                                  # the desk at http://127.0.0.1:8001: load the slate, work the teams, Publish
python scripts/nba/generate_lineups.py 10000 8 && python scripts/nba/sim.py && python scripts/nba/choose.py --no-prompt
python scripts/nba/lateswap.py                               # after lock

# NFL: the five steps (docs/nfl/pipeline.md), or the same five as buttons at http://localhost:8000
python scripts/nfl/ingest.py -p --update-odds true
python scripts/nfl/get_contest_ids.py && python scripts/nfl/get_payouts.py
python scripts/nfl/generate_lineups.py 10000 8
python scripts/nfl/sim.py
python scripts/nfl/choose.py --no-prompt
python scripts/nfl/serve.py                                  # the web UI

# MMA (docs/mma/README.md)
python scripts/mma/ingest.py                                 # then fill the two odds CSVs and run it again
python scripts/mma/generate_lineups.py --universe && python scripts/mma/sim.py && python scripts/mma/choose.py
```

The entries file each sport reads is your DraftKings entries export, at `data/<sport>/input/entries.csv`.
Outputs are the DraftKings upload CSV plus exposures and diagnostics under `data/<sport>/output/`.

## Timers and the second machine

Box scores, settlement and the NBA archive-at-tip run from user systemd timers on the laptop
(`deploy/install_nightly_timer.sh`, `deploy/install_nba_timers.sh`). A server runs the web UI and the NBA
desk for someone else on data the laptop pushes (`deploy/push.sh` for code, the NFL sync and the NBA
feed for data). See [docs/deployment.md](docs/deployment.md). The live timers and servers run from this
working tree, so the checkout stays on `master`.

## Tests

```bash
python -m pytest tests
```

## Layout

```
scripts/<sport>/      command-line entry points, one per step
src/ivan/core/        shared infrastructure: config, raw payloads, the slate archive, artifacts, clock, seed, replay, validation, feed, chooser numerics
src/ivan/sports/      common/ (DraftKings, odds API, payouts) and one package per sport
src/ivan/web/         the NFL app and the NBA desk (FastAPI + Jinja + htmx)
config/               per-sport JSON, replay menus, tax rules
deploy/               server bootstrap and hardening, systemd units, push.sh, the Cloudflare tunnel
docs/                 everything else
data/<sport>/         git-ignored: database, raw, inputs, models, outputs, diagnostics, logs
```
