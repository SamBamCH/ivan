# Ivan

Daily-fantasy analytics for DraftKings: ingest a slate, fit per-player fantasy-point
distributions from betting odds, generate a correlated field of lineups, simulate the
actual contests you entered, and pick the lineups to submit.

| Sport | Status | Entry points |
|---|---|---|
| **NFL** | Active — CLI pipeline plus a web UI | `scripts/nfl/`, `src/ivan/sports/nfl/`, `src/ivan/web/` |
| NBA | Rebuilding for 2026-27 — minutes-anchored projections in, odds layer next | `scripts/nba/`, `src/ivan/sports/nba/` |
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
| 4 | Simulate | `python scripts/nfl/sim.py [--entries file.csv] [-n 1000] [-c 10000] [-d]` | For each contest in the entries CSV: sample an opposing field with the duplication model, simulate `-n` outcomes, score every candidate lineup (cash %, ROI, rank), replacing earlier results for that contest. `-c` caps the simulated field size; `-d` allows duplicate lineups in classic. Each contest also gets a report (`contest_sim_report` table and `data/nfl/diagnostics/contest_*/report.json`) of the sampled field, the payout scaling and the invariants it checked; the web UI's Contests page renders it. |
| 5 | Choose | `python scripts/nfl/choose.py [--entries file.csv] [--no-prompt] [--cash-pct .5 --avg-roi-win .5 --diversity-lambda .1] [-o base.csv]` | Picks lineups per contest from the sim results and writes DK-format upload CSVs, an exposures CSV, a per-lineup detail CSV (one row per player per entry with the lineup's simulated cash % and ROI) and a chooser-diagnostics JSON (every candidate's cash %, ROI, quality and map position, the picks and the greedy trace); the web UI's Exposures and Choose pages read them. With no weight flags it prompts per contest; any weight flag, `--weights-json`, or `--no-prompt` makes it non-interactive. |

The entries file is the CSV you export from DraftKings for the contests you have entered
(`Entry ID, Contest Name, Contest ID, Entry Fee, …`). By default it is read from
`config/nfl.json → paths.entries_csv` (`data/nfl/input/entries.csv`); override with
`--entries` or `IVAN_ENTRIES_CSV`.

Outputs land in `data/nfl/output/` as `chosen_entries.<showdown|classic|exposures|lineups>.<timestamp>.csv`
plus `chosen_entries.choose.<timestamp>.json`.

`--weights-json` takes a file or a literal, keyed by contest id or contest name with an
optional `"default"`:

```json
{"default": {"cash_pct": 0.5, "avg_roi_win": 0.5, "diversity_lambda": 0.1},
 "193391004": {"diversity_lambda": 0.3}}
```

### Kickers and defenses

Kickers are fit like everyone else, from their markets: `player_kicking_points`
(3 per FG, 1 per PAT) sets the centre and `player_field_goals` the FG count;
DraftKings' distance bonus (+1 for 40-49 yards, +2 for 50+) is added per made
FG at the kicker's own historical rate, shrunk to the league's 0.65. A defense
has no market, so its mean comes from the game lines: a fit of historical
DraftKings DST points on the two implied totals (`dst_odds_params.json`,
written by `build_models.py`), with residuals bootstrapped from the closest
historical games so the skew and the -4 floor survive. Without lines it falls
back to a salary scale. The ingest prints one validation line saying how many
stat distributions came from a market versus the salary model and where each
K/DST projection came from; `generate_lineups.py` prints a per-game check that
the sampled correlation matches the team and cross-team history it asked for.

### Backtesting projections

Every ingest snapshots the slate's projections. `python scripts/nfl/backtest_projections.py
[--since 2025-09-01] [--positions K,DST]` scores them against the DraftKings
points the players actually produced (from `nfl_stats` and `nfl_dst_game`):
bias, error, rank correlation, whether the stated spread covers outcomes, and
a tercile table per position. The overnight job keeps `nfl_stats`,
`nfl_schedule` (scores and closing lines) and `nfl_dst_game` current through
the season in progress.

## Web UI

```bash
python scripts/nfl/serve.py            # http://<host>:8000
```

A single page with the five steps as forms (step 5 takes choose weights per contest and
remembers them), an entries-file upload, a live log of the running job, and pages for the
per-contest simulation reports, exposures, downloads (the DK upload files) and job history.
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
  `lineup_simulation_results`, `contest_sim_report` — the server's own slate state.
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

## NBA

The NBA side is being brought up to the NFL approach for 2026-27. What exists today:

| Step | Command | What it does |
|---|---|---|
| Ingest | `python scripts/nba/ingest.py -p --update-odds --no-projections-csv` | Upcoming contests, the DK player pool (`-p`), player props from the-odds-api turned into per-minute market rates (`--update-odds`, needs `ODDS_API_KEY` in `.env`), confirmed starters from Rotowire into `starting_lineups`. Without `--no-projections-csv` it still reads `data/nba/input/stok.csv` if present. |
| Project | `python scripts/nba/project.py [--dry-run] [--team MIL] [--no-market]` | Minutes-anchored projections (below): market rates where a player is priced, history elsewhere. Writes `fpts_mean`, `fpts_std`, `minutes_proj` on `nba_player`; players with DK status OUT get no projection. |
| Late news | `python scripts/nba/news.py "Giannis Antetokounmpo" [--accept] [--conditional]` | Marks him out, shows his team with and without him, asks for minutes player by player (Enter keeps the model's), saves the pins to `minutes_overrides.csv`, writes the projections. Then `sim.py --fresh`. |
| Generate, simulate, choose | `python scripts/nba/sim.py [--num-lineups 25000] [--num-sims 1000]` | One interactive flow: GLPK lineup pool, field sampled from the historical-contest model, contest sims from the entries CSV, lineup selection. |
| Late swap | `python scripts/nba/swap.py` | Re-optimises locked entries against the remaining games. |
| Overnight | `python scripts/nba/overnight.py` | Box scores (nba_api), contest results and payouts, refits the field model. **Runs on import** — do not call it with `--help`. |

### Minutes-anchored projections

`src/ivan/sports/nba/projection/` models a player as minutes times per-minute
rates, the NBA analogue of the salary/TD-odds anchor NFL uses for unpriced
stats. Minutes are the volume signal: a team has 241 minutes (240 plus the
average overtime) to hand out, so availability changes move minutes around
the roster instead of being guessed per player.

* **History** (`history.py`): per player, exponentially weighted minutes
  (half-life 5 games), minute-weighted per-minute rates for points, rebounds,
  assists, steals, blocks, turnovers and threes (half-life 20), starter share,
  and how often he plays when dressed. Per team, trailing fantasy points per game.
* **Minutes** (`minutes.py`): baseline = play-rate × trailing minutes, plus
  6 minutes for a confirmed starter who has been coming off the bench (minus 6
  the other way). The available roster is then reconciled to 241: a shortfall
  (someone out) goes to players with room to grow, a surplus (DK lists the whole
  roster) comes off the bottom of the rotation.
* **Rates** (`rates.py`, `market.py`): for a player the books price, each
  (book, line) pair is de-vigged and solved for the negative-binomial mean
  that matches it; the median is his market mean for the game, and dividing
  by the minutes he projected to when the props were fetched gives a
  forward-looking per-minute rate that travels with him when his minutes
  move (`nba_market_rate` table). Everyone else uses trailing history rates
  shrunk to league with 60 minutes of prior weight. One usage multiplier per
  team closes half the gap between the team's trailing production and what
  the present roster projects to, so teammates pick up efficiency as well as
  minutes when a star sits.
* **Fantasy points** (`fpts.py`): Monte Carlo over minutes and negative-binomial
  stat draws; double-double and triple-double bonuses come out of the joint
  draw. Dispersion and minutes spread live in
  `data/nba/models/projection_calibration.json`, refit by the backtest.

**Late news.** `news.py "Name"` is the one command: it marks the player out,
shows the team with and without him (model minutes, history, and how each
teammate's minutes and production per minute went in last season's games
without him), asks for minutes player by player, saves the pins to
`data/nba/input/minutes_overrides.csv` (columns `player,minutes`; 0 means a
scratch) and writes the projections. The minutes you type are the
forward-looking input; the model only moves what you do not pin. Rates stay
forward looking too: a priced player keeps his market rate, the rest keep
history times the usage bump. `--conditional` scales teammates toward their
"without him" per-minute history instead; it is opt-in because those samples
are small and confounded (LeBron's six games without Luka in 2025-26 show
*lower* production, which is his own early-season form, not usage). For a
player stepping into a role nobody has seen, pin his minutes; his efficiency
comes from the usage bump until the books reprice him, after which
`ingest.py --update-odds` and `project.py` pick up the market's view.

**Backtest.** `python scripts/nba/backtest_projection.py [--dnp-available]
[--grid|--kgrid|--rgrid|--sgrid] [--calibrate] [--compare]` replays 2025-26
from 15 November to 15 January using only earlier games. `--dnp-available`
is the live condition (dressed players may not play); without it the box
score reveals who played. `--compare` scores the same player-games against
the Stokastic projections kept in `projection_snapshot`; `--market-day
2025-11-30` evaluates market rates from the props stored in `odds` for that
slate. As of 2026-10-06, under live conditions, the model's error is level
with Stokastic (MAE 7.44 vs 7.41 DK points on 3,410 player-games) and a 50/50
blend beats both; minutes error is 4.7 per player-game. Overfitting check:
parameters tuned on 15 Nov to 15 Dec are the ones the held-out 16 Dec to 15 Jan
half prefers (model 7.36 vs naive 7.51 vs season average 7.62 there), the
grid surfaces are flat, and the playoffs (never used for tuning) show the
same margin (7.50 vs 7.74 vs 8.81). The one stored props slate (71 priced
players) is too small to rank market against history rates, but the market
consensus and Stokastic correlate at 0.994: Stokastic is the market, so the
real test of the odds layer starts when ingests snapshot projections through
the season. Reports land in `data/nba/diagnostics/`.

Not yet ported from NFL: the correlation layer, the web pipeline and the
overnight job's safety rails. The projection snapshot keeps one `fpts_mean`
per player-day, so keep dropping `stok.csv` on a separate column if you want
Stokastic as a season-long yardstick.

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
├── diagnostics/            per-contest simulation reports (report.json; pruned by sim.py)
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
deploy/                     server bootstrap, systemd unit, push.sh (deploy + restart)
docs/                       web-ui.md (for users), server.md (for operators)
```

## Housekeeping worth knowing

- `overnight.py` deletes last slate's lineups and simulation results. SQLite never returns
  freed pages to the OS on its own — the history file once reached 18 GB around ~600 MB of
  data — so `overnight.py` now runs `VACUUM` when more than a quarter of the file is freelist.
- `sqlite3 <db> "VACUUM INTO 'compact.sqlite'"` makes a compact, consistent copy without
  touching the original; it is what the sync script snapshots from.
