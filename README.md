# Ivan

Daily-fantasy analytics for DraftKings: ingest a slate, fit per-player fantasy-point
distributions from betting odds, generate a correlated field of lineups, simulate the
actual contests you entered, and pick the lineups to submit.

| Sport | Status | Entry points |
|---|---|---|
| **NFL** | Active — CLI pipeline plus a web UI | `scripts/nfl/`, `src/ivan/sports/nfl/`, `src/ivan/web/` |
| NBA | Mid-refactor | `scripts/nba/`, `src/ivan/sports/nba/` |
| MLB | Dormant (previous seasons) | `scripts/mlb/`, `src/ivan/sports/mlb/` |
| LoL | Dormant | `scripts/lol/`, `src/ivan/sports/lol/` |

Everything below is about NFL unless it says otherwise.

## Setup

```bash
git clone git@github.com:SamBamCH/ivan.git && cd ivan
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # full dev install (model training, scrapers, all sports)
pip install -e .
cp .env.example .env                    # then put your the-odds-api key in it
python scripts/nfl/create_db.py         # creates data/nfl/nfl_history.sqlite
```

Requirements:
- Python 3.12, `sqlite3` and `rsync` on the PATH.
- An [the-odds-api](https://the-odds-api.com) key (`ODDS_API_KEY` in `.env`). Two machines
  on the same slate is a handful of extra requests; the paid tier is plenty.
- An ILP solver: PuLP's bundled CBC works out of the box. GLPK is used instead if `glpsol`
  is on the PATH or `GLPK_PATH` is set.
- Only for building models from scraped contest results: Chrome plus a logged-in DK profile
  in `chrome-selenium/dk-profile` (see *Producer-only work*).

Every path in the project is resolved against the repository root, never the current
directory, so scripts can be run from anywhere.

## The weekly NFL pipeline

Run these top to bottom. Each is one script; the web UI is just buttons for the same five.

| # | Step | Command | What it does |
|---|---|---|---|
| 1 | Ingest the slate | `python scripts/nfl/ingest.py -p --update-odds true` | Players + salaries from DraftKings, player odds from the-odds-api, fits each player's FPTS distribution, then projects ownership by sampling a field of noisy-optimal lineups and snapshots projections for training. `-p` (pre-lock) re-fetches the player pool and prompts for the slate; `--draft-group <id>` skips the prompt. Drop `-p` to refresh odds only. `--no-ownership` / `--no-snapshot` skip those tails; `scripts/nfl/project_ownership.py` re-runs them alone. |
| 2 | Contests & payouts | `python scripts/nfl/get_contest_ids.py && python scripts/nfl/get_payouts.py [--days N \| --all]` | Upcoming DK contests and their payout tables. The simulator needs a payout table for every contest in your entries file. Payouts are fetched only for contests starting within the last day or later; `--all` re-queries old ones too. |
| 3 | Generate the field | `python scripts/nfl/generate_lineups.py 10000 8` | `<lineups> <parallel workers>`. Correlated player samples → ILP → the pool of candidate lineups the simulator picks from. Deduplicates at the end. |
| 4 | Simulate | `python scripts/nfl/sim.py [--entries file.csv] [-n 1000] [-c 10000] [-d]` | For each contest in the entries CSV: sample an opposing field with the duplication model, simulate `-n` outcomes, score every candidate lineup (cash %, ROI, rank). `-c` caps the simulated field size; `-d` allows duplicate lineups in classic. |
| 5 | Choose | `python scripts/nfl/choose.py [--entries file.csv] [--no-prompt] [--cash-pct .5 --avg-roi-win .5 --diversity-lambda .1] [-o base.csv]` | Picks lineups per contest from the sim results and writes DK-format upload CSVs plus an exposures CSV. With no weight flags it prompts per contest; any weight flag, `--weights-json`, or `--no-prompt` makes it non-interactive. |

The entries file is the CSV you export from DraftKings for the contests you have entered
(`Entry ID, Contest Name, Contest ID, Entry Fee, …`). By default it is read from
`config/nfl.json → paths.entries_csv` (`data/nfl/input/entries.csv`); override with
`--entries` or `IVAN_ENTRIES_CSV`.

Outputs land in `data/nfl/output/` as `chosen_entries.<showdown|classic|exposures>.<timestamp>.csv`.

`--weights-json` takes a file or a literal, keyed by contest id or contest name with an
optional `"default"`:

```json
{"default": {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1},
 "193391004": {"diversity_lambda": 0.3}}
```

## Web UI

```bash
python scripts/nfl/serve.py            # http://<host>:8000
```

A single page with the five steps as forms, an entries-file upload, a live log of the
running job, and pages for exposures, downloads (the DK upload files) and job history.
It shells out to the scripts above and runs **one job at a time**, `nice`d, which is also
what keeps SQLite happy. There is no authentication: bind it to a LAN or Tailscale address.

Uploads go to `data/nfl/input/web/`, outputs to `data/nfl/output/web/`, job logs to
`data/nfl/jobs/`. Code is in `src/ivan/web/` (FastAPI + Jinja + a vendored htmx; no build step).
See [docs/web-ui.md](docs/web-ui.md) for the user-facing walkthrough.

## Two machines: producer and consumer

The intended deployment is a laptop that owns the history and models (*producer*) and a
server that runs the web UI for someone else on the same slates (*consumer*). Each has its
own SQLite file, so there is no lock contention between them; the producer pushes reference
data one way.

```
laptop (producer)                          server (consumer)
  overnight.py ──► boxscores, model builds     web UI ──► ingest/generate/sim/choose
       │                                            ▲
       └── sync_to_server.sh ───────────────────────┘
           reference tables + models only; never the server's working tables
```

- **Synced** (`scripts/nfl/sync_to_server.sh [host]`): every table except the per-slate
  working set, plus their indexes, plus `data/nfl/models/`. About 24 MB gzipped.
- **Never synced**: `nfl_player`, `odds`, `projection_snapshot`, `lineup`,
  `lineup_simulation_results` — the server's own slate state.
- `overnight.py` runs the sync automatically when `IVAN_SYNC_HOST` is set (in `.env`).
- The server runs the UI as an unprivileged user in a sandboxed unit, reachable only over
  Tailscale; setup is `deploy/harden_server.sh`. See [docs/server.md](docs/server.md).

## Producer-only work

These need the full `requirements.txt` install and, for `get_contest_data.py`, a logged-in
DraftKings Chrome profile.

| Script | Purpose |
|---|---|
| `scripts/nfl/overnight.py` | Fetch boxscores, clear last slate's lineups and sim results, `VACUUM` when the file is mostly free pages, sync to the server. |
| `scripts/nfl/get_contest_data.py` | Scrape finished contests (lineups, ownership, payouts) into the historical tables. Selenium + `chrome-selenium/dk-profile`. |
| `scripts/nfl/build_models.py` | Player marginals and copulas, team vine, DST/positional tables → `data/nfl/models/`. |
| `scripts/nfl/build_dupe_model.py` | Two-stage showdown duplication model → `data/nfl/models/{contest_dupe_quantiles,lineup_dupe_model}.pkl`. |
| `scripts/nfl/lineup_exposures.py` | Ad-hoc exposure and stacking diagnostics over the current field. |

## Configuration

`config/nfl.json` is machine-independent and tracked. Paths in it are relative to the repo
root. Model tuning lives there too (`field`, `pool`, `sampling`, `simulation`,
`classic.lineup_generation`, `dupe_model_simple`), and `ownership.num_lineups` /
`ownership.num_simulations` control how many sampled lineups ownership is estimated from.

Environment (`.env` at the repo root is loaded automatically):

| Variable | Used by | Meaning |
|---|---|---|
| `ODDS_API_KEY` | `ingest.py` | the-odds-api key. Required for odds. |
| `IVAN_ENTRIES_CSV` | `sim.py`, `choose.py` | Entries file, overriding config. |
| `IVAN_NFL_DB_URL` | everything | SQLAlchemy URL, overriding the default `sqlite:///data/nfl/nfl_history.sqlite`. |
| `IVAN_SYNC_HOST` | `overnight.py` | ssh host to push reference data to after the nightly run. |
| `IVAN_NFL_DB`, `IVAN_REMOTE_ROOT` | sync scripts | Source sqlite file; checkout path on the server (default `ivan`). |
| `GLPK_PATH` | lineup generation | Path to `glpsol`; otherwise PATH lookup, otherwise CBC. |

On-disk layout is defined once in `src/ivan/sports/nfl/paths.py`:

```
data/nfl/
├── nfl_history.sqlite      one file: reference tables + this machine's working tables
├── input/entries.csv       default entries file (web uploads: input/web/)
├── output/                 chosen-entries and exposures CSVs (web: output/web/)
├── models/                 pickles from build_models.py / build_dupe_model.py
├── diagnostics/            per-contest simulation diagnostics (pruned by sim.py)
└── jobs/                   web UI job logs
```

## Repository layout

```
scripts/<sport>/            command-line entry points (see tables above)
src/ivan/
├── core/                   config loader, project-root paths, name normalisation, generic helpers
├── database/               engine/session factory, read_sql
├── sports/common/          DraftKings and odds API clients, payout scaling
├── sports/nfl/             models, ingestion, distributions, field sampling, simulation, choosing
│   └── refactor/           the newer ingestion path (player/stat fitting, team correlation, boxscores)
├── sports/{nba,mlb,lol}/   other sports
└── web/                    FastAPI app, job runner, NFL pipeline definition, templates
config/                     per-sport JSON config
deploy/                     server bootstrap + systemd unit
docs/                       web-ui.md (for users), server.md (for operators)
```

## Housekeeping worth knowing

- `overnight.py` deletes last slate's lineups and simulation results. SQLite never returns
  freed pages to the OS on its own — the history file once reached 18 GB around ~600 MB of
  data — so `overnight.py` now runs `VACUUM` when more than a quarter of the file is freelist.
- `sqlite3 <db> "VACUUM INTO 'compact.sqlite'"` makes a compact, consistent copy without
  touching the original; it is what the sync script snapshots from.
