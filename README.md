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
| MMA | Active — CSV-driven odds model ported from the standalone mma_dfs project (2026-10) | `scripts/mma/`, `src/ivan/sports/mma/` |

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

### Replaying past slates with today's code

```bash
python scripts/replay.py --sport nfl --since 2025-11-01 --until 2025-11-30 [--lineups 3000 --jobs 4 --sims 500] [--bankroll 2000]
```

Every pre-lock ingest now archives its raw inputs (the DraftKings pool, the
player props, the game lines) as a capture in `archived_slate` /
`archived_blob`, never modelled. The replay takes each archived slate in the
window, restores those inputs into a *copy* of the database
(`data/nfl/replay/nfl_replay.sqlite`; the live one is never written), pins
the clock to the slate's lock (`IVAN_AS_OF`, honoured wherever the pipeline
reads the time or player history, so a 2025 slate cannot see later box
scores), and runs today's projection, lineup generation, contest simulation
and chooser on it. The contests it enters come from a menu
(`config/replay/nfl.json`: name patterns, entry counts, choose weights)
matched against the slate's real contests. Each slate's outputs land in
`data/nfl/output/replay/nfl/<lock>_<mode>_<slate>/` in the live pipeline's
own files; afterwards the settlement + tax report runs over the window
(`report_<since>_<until>/`). Known gap: the correlation, duplication and
spread-calibration models in `data/nfl/models` were trained on data that
includes the replayed weeks.

One replay is one draw of the pipeline's own randomness (the generator's
jitter, the sim's universes). `--repeats N --seed 0` runs generate ->
simulate -> choose N times per slate (projection once), repeat k seeded with
`IVAN_SEED = seed + k` (every draw in the pipeline derives from it, see
`ivan.core.seed`; unset, the live pipeline behaves as before), into
`rep_000/`, `rep_001/`, ... of the slate directory. The report then settles
every repeat against the same actual results and adds the window's profit as
a distribution (`repeats.csv`, `repeats_by_slate.csv`, `repeats_summary.json`,
`repeats.png`): mean / percentiles / share of repeats in profit, pre- and
after-tax, the same with the slates bootstrapped, the window without its best
slate, and each slate's spread. Profit that only shows in some repeats was
luck of the draw, not edge. A slate-repeat takes about 35 s; `--workers W`
splits the slates over W processes (each on its own copy of the replay
database; keep workers x jobs near the core count), re-running with a larger
`--repeats` only adds the missing ones, and `--report-only` re-reads what is
there. Put a long run in its own root (`--out data/nfl/output/replay_repeats`)
to keep the single replay intact:

```bash
python -u scripts/replay.py --sport nfl --since 2025-11-01 --until 2025-11-30 --repeats 100 --seed 0 --workers 3 --jobs 3 \
    --out data/nfl/output/replay_repeats --bankroll 1000 --tax-year 2025
```

Past slates that predate the archive are rebuilt with
`scripts/nfl/backfill_archive.py --since 2025-11-01 --until 2025-11-30 --odds`:
the pool from the projection snapshot, the lines from `game_odds`, and the
props from the-odds-api historical endpoint as of 30 minutes before lock
(about 210 credits per showdown game). Only showdown slates with a scraped
field are backfilled, since those are the ones that settle with real payouts.

### After-tax backtest

`python scripts/nfl/backtest_tax.py [--dk-history export.csv] [--bankroll 2000] [--cap 0.9 --rate 0.22 | --brackets --other-income 90000] [--mode gross_165d|net_dfs]`

What the chosen lineups would have returned after US federal gambling tax
(2026 rules: only 90% of losses deductible, configurable for the pending
repeal). For each contest the last choose run before lock is what we would
have submitted; each of those entries is settled from the best evidence
available, in this order: a DraftKings contest-history export
(`--dk-history`, the real finish of entries actually played), our own
entries in a scraped contest (`get_contest_data.py --username <dk user>`),
or the lineup's actual points placed in the scraped field and paid from the
real payout table. Entries that cannot be settled are listed with the reason
(no box scores yet: run `overnight.py`; no field: scrape the contest with
`get_contest_data.py --contest-ids ...`).

The report prints the year to date (W, L, deductible, itemize decision,
taxable, phantom income, tax, after-tax profit and ROI), a scenario table
(cap 90% vs 100%, itemize vs standard, gross vs net reporting), the tax each
contest format adds, the sim's expectation against the actual result per
contest, the running per-slate ledger with the point where tax owed exceeds
the bankroll, and a volume sweep (bootstrapped years at 0.25x-8x the
observed slates, from the actual slates and from the sim's traces) with a
plot. Everything is also written under `data/nfl/output/backtest/<stamp>/`.
Rules and 2026 figures live in `config/tax.json` (verify them with a CPA);
the engine itself is `ivan.sports.nfl.tax.compute_tax(W, L, config)`, tested
in `tests/nfl/test_tax.py`.

`choose.py` now also writes `chosen_entries.simroi.<stamp>.npz` (each pick's
per-simulation ROI) because the sim tables are cleared weekly, and the
scraper stores every entry's actual points (`contest_field_scores`) for
showdown and classic contests alike. Run `python -m pytest tests` for the
test suite.

### Ownership: the sampler, corrected by what fields do

The field sampler projects ownership by generating value-driven lineups
and counting exposures. It has the wrong habits for roster-filler
positions (it never plays a 4-point defense; the public plays one 13% of
the time in showdowns, kickers 20%) and overplays the chalk. Two
corrections, both fitted on the scraped showdown fields and both part of
the as-of model build:

- **Calibration lines** (`scripts/nfl/build_ownership_calibration.py` →
  `ownership_calibration.json`): per slot and position, actual ~ a + b ×
  projected; DST and K on projected points instead (a DST's flex ownership
  tracks its projected points at corr 0.89).
- **Learned model** (`scripts/nfl/build_ownership_model.py` →
  `ownership_model.pkl`): a gradient-boosted regressor on the sampler's
  projection plus points, salary, value, ranks, position, slot, team share
  and contest size. On a time split (train November 2025, test October
  2026) it scores MAE 2.51 against 2.74 for the calibrated sampler and 3.00
  for the sampler alone, Spearman 0.961 vs 0.926 vs 0.939; on the last
  third of contests MAE 2.73 vs 3.26 vs 3.80 and chalk bias +1.8 vs +2.6 vs
  +6.8. `backtest` prints the comparison.

`config/nfl.json` → `ownership.model`: `learned` (default; falls back to
the calibration when no model file exists), `sampler_calibrated`, or
`sampler`. Each slot is renormalised to one lineup's worth (captain 100,
flex 500). Classic slates are untouched until their fields are scraped at
the player level. The validation's ownership check (MAE, Spearman, chalk
bias by quintile) is the acceptance test for any change here.

### Duplication: how many copies of a lineup the field holds

`ivan.sports.nfl.dupe_model` fits the product rule on real fields (the
lineups the settle job keeps per contest):

    copies(L) = N × k × ∏ (own_i / 100) ^ beta

by copy-weighted least squares on the log scale, per format, so it targets
the copies of the lineup an entry actually holds rather than the million
singletons. On the first 50 October fields: showdown k 0.044, beta 0.56
(leave-one-slate-out bias within ±0.2 in log terms; the single most
duplicated lineup is still under-predicted, a tail the rule does not
reach); classic k 4e-5, beta 0.07, i.e. classic fields barely duplicate by
ownership. Both samplers use the same function for their lineup weights
and for every candidate's predicted copies, so the sampled field's
concentration and the dupe prediction of our own lineups agree.
`scripts/nfl/build_dupe_model.py backtest|build` re-fits as fields
accumulate; the as-of build writes a version per cutoff. (The earlier
two-stage quantile + GBM model, which under-predicted the chalk about 3x,
was removed in 2026-10.)

Where the dupe model misses: the Dupes page of the web UI (`/dupes`) and
`scripts/nfl/dupe_report.py` put the model's expected copies against the
copies every scraped field actually held, lineup by lineup
(`ivan.sports.nfl.dupe_analysis`): calibration by predicted copies, the
copy-weighted residual by captain, ownership-sum decile, team split, QB
stack, bring-back and salary left on the table, and the most duplicated
lineups. Each residual is shown twice, from the ownership we projected
(what the pipeline runs on) and from the ownership the field showed, so a
miss can be attributed to the dupe rule or to the ownership projection
upstream.

### Raw artifacts: every payload, as received

`ivan.core.raw` keeps every payload a fetcher receives, gzip, under
`data/<sport>/raw/<kind>/<key>/<stamp>.<ext>.gz`, with an append-only
`index.jsonl` and the same index in the database (`raw_artifact`). The
fetchers write it before anything parses it: DraftKings draftables, lobby,
payouts and standings exports; the odds API's events, player props (every
book, every market asked for) and game lines. A parser change later is a
re-run over the files. `docs/data_framework.md` has the design and the
build order.

**Settling a slate** (`scripts/nfl/settle.py`, run nightly by the overnight
job with `--days 3`): for every archived slate past lock, the standings of
the contests we entered and the draft group's largest public contests, the
week's box scores, the derived `lineups.parquet` and `scores.npy` per contest
under `data/nfl/derived/contests/`, and a status row per slate
(`slate_status`: open, waiting, login_needed, settled), plus, for classic
contests, per-player rows (actual ownership and points from the export's
player table, matched to the capture and the projection snapshot) so classic
slates settle at the player level like showdowns. Idempotent. The
scrape needs the saved DraftKings Chrome profile and a display; without
them the job runs with `--no-scrape` and the slate stays `waiting`. When
the profile's session has expired the scraper opens the login page in the
Chrome window and waits up to ten minutes for you to log in there, then
continues; the profile keeps the session for the following runs.

### Backtest: re-run slates with what was known at lock

One page, `/backtest`, organised around a **run** (a window of slates, the
contests chosen on them, the parameters, the results; stored under
`data/nfl/output/backtests/<name>/`). Open an existing run from the menu or
start a new one with a name and a window. A new run walks through:

1. **Data.** What the window lacks, grouped by the fetch that fills it, each
   with its cost: box scores for the missing weeks (free, nflverse), pre-lock
   captures for slates that only have scraped contests (the props come from
   the-odds-api history at about 230 credits per game), the default pool's
   unscraped contests (DraftKings through the saved Chrome profile, with your
   username so your finishes settle), and a model version as of the window
   start. Nothing is fetched until you tick it and confirm; what is fetched is
   written to the database and kept. You can also continue without it.
2. **Contests.** Every contest of every replayable slate; tick what to enter
   and how many entries (defaults follow `config/replay/nfl.json`).
3. **Parameters.** Repeats per contest, workers, lineups, simulations, field
   cap, seed. Each repeat re-generates, re-simulates and re-chooses with seed
   plus k; the projection is fitted once per slate.
4. **Results.** The sim's claim against the actual outcome and after tax, the
   distribution across repeats, the assumption checks (projections,
   environment, usage, ownership, field, dupes, simulator, chooser; full
   component detail one link away), then every slate and contest with the
   sim's expected profit, the actual profit and, over repeats, the mean, p10,
   p90 and the share of repeats that profited.

Going forward, every slate played is settled on its own: the ingest archives
the capture at lock, the overnight job fetches the box scores, and
`scripts/nfl/settle.py --days 3` scrapes the contests the chooser entered in
the last days (the overnight job runs it when `IVAN_DK_USERNAME` is set and
a display exists for Chrome).

Underneath: `ivan.sports.nfl.readiness` (what the window holds),
`ivan.web.backtest_hub` (the steps and the job commands), `scripts/replay.py`
with an explicit selection (`"picks"` in the menu file, written per run as
`replay_<name>/selection.json`), `scripts/nfl/validate.py --replay` for the
checks, and the replay's own report for the settlement (`repeats.csv`,
`repeats_by_slate.csv`, `repeats_by_contest.csv`, `sim_vs_actual.csv`).

What makes the re-run honest:

- **Captures.** Every ingest archives the slate's players, props and game
  lines; older slates are backfilled from projection snapshots. The replay
  restores a capture with the clock frozen at lock minus one minute, so every
  history table cuts off at that week, and enters the slate's own draft
  group's contests.
- **As-of models.** `build_models.py --cutoff 2025-11-01` writes
  `data/nfl/models/asof/2025-11-01/` (DST parameters, classic field models,
  the dupe model, the sd calibration, each trained only on contests and games
  before the cutoff) with a manifest. `ivan.core.model_store` resolves the
  directory for the clock: live runs use the live models, a replay the newest
  version whose cutoff precedes the slate, falling back to the live models
  while saying so in the manifest and the report.

    scripts/nfl/build_models.py --cutoff 2025-11-01
    scripts/replay.py --since 2025-11-01 --until 2025-11-30 --menu data/nfl/output/replay_nov/selection.json --repeats 20 --workers 3 --out data/nfl/output/replay_nov
    scripts/nfl/validate.py --replay data/nfl/output/replay_nov

### Validation: projected vs actual, per component

Every stage writes its predictions down when it makes them (`ivan.core.artifacts`):
the ingest writes `data/nfl/output/projections/projections.<stamp>.csv`, the simulator
writes `data/nfl/output/sim/<stamp>/contest_<id>.npz|json` (the sampled field, every
candidate lineup's simulated cash % / ROI / predicted dupes, the players' sampled score
quantiles and field exposure, the cash line and winning score per simulation), and the
chooser its `chosen_entries.*` files. A replay writes the same files into each slate /
repeat directory (`IVAN_RUN_DIR`). After the games:

```bash
python scripts/nfl/validate.py --since 2026-09-01 --until 2026-10-12            # live runs
python scripts/nfl/validate.py --replay data/nfl/output/replay_artifacts        # a replayed window
```

pairs each contest's last pre-lock choose run with the sim and projections it read, scores
every lineup the way the settlement does (`backtest_tax.score_lineup`, scraped per-slot
points first, `nfl_stats` otherwise) and reports, with a pass / warn / fail row each
(`ivan.core.validation`, NFL adapter `ivan.sports.nfl.validate`):

* **projections** bias, MAE against a salary-only baseline, PIT histogram and 10-90
  coverage, by position / salary tier / ownership quintile; slates without artifacts are
  covered from `projection_snapshot` with a normal-approximation PIT
* **ownership** projected ownership and the sim's sampled-field exposure against actual
  ownership (scraped contests), with the chalk bias
* **field** the sim's cash line, winning score, score percentiles and duplication profile
  against the real field, per contest
* **simulator / chooser** every lineup the sim scored, settled in the real field: does sim
  ROI rank lineups (Spearman, sim-ROI deciles vs actual ROI), and do our picks beat K random
  lineups from the same pool
* **our entries** sim cash % / ROI / predicted dupes against the real finish and copies

Output under `data/nfl/output/validation/<stamp>/` (or `<replay root>/nfl/validation/rep_<k>/`):
`players.csv`, `field.csv`, `lineups.csv`, `pool.csv`, `pool_deciles.csv`, `summary.json`;
the Validation page of the web UI reads them and can submit a run as a job. Live slates
need the week's box scores (overnight job) and the contests scraped with
`get_contest_data.py --contest-ids ... --username <dk user>` to settle.

### The projected game environment (layer 1 of the game model)

`ivan.sports.nfl.refactor.environment` draws, per team in a game, points, offensive plays
and pass rate jointly from the lines (total, spread) and the teams' tendencies in history,
with the within-game residual dependence (a scoring team runs more and passes less, the
trailing team passes more, both teams' play counts move together) estimated from past games.
Training rows are `nfl_schedule` (closing lines, final scores) and `nfl_stats` (attempts,
sacks, carries per team-week), strictly before the history cutoff. Team tendencies are
shrunk toward the league and only used where the rolling backtest says they help
(`DEFAULT_TEAM_EFFECTS`: pass rate yes, plays a little, points never - the market already
prices team strength).

```bash
python scripts/nfl/environment.py backtest --from 2024-1    # rolling refit, projected vs actual, writes a validation report
python scripts/nfl/environment.py grid                      # shrinkage grid behind the defaults
python scripts/nfl/environment.py slate                     # the loaded slate's environment
python scripts/nfl/environment.py draw CIN BAL -n 2000      # joint draws for one game
```

Every projection step writes the slate's environment next to the projections
(`environment.<stamp>.csv`), and the validation joins it to the actual team-games, so the
Validation page shows a Game environment table ahead of the player tables. Layers 2 (usage
shares pinned to the props, residual to unpriced players by depth slot) and 3 (market
marginals through rank mapping) are planned on top of these draws.

### Usage allocation (layer 2 of the game model)

`ivan.sports.nfl.refactor.usage` hands out the volume of the players nobody
prices. For each team and stat (receiving yards, receptions, rushing yards)
it takes the team total (the QB's passing-yards line when one exists, else the
game environment's dropbacks / rushes times the league's yards per dropback,
completions per dropback or yards per carry), subtracts every mean already
settled (the priced players' market means and the QB's salary-model rushing),
floors the residual at 5% of the total, and splits it among the unpriced
RB / WR / TE by depth-rank priors measured on the previous four seasons (share
of the team's stat for the k-th WR / TE / RB by salary order; WR4 .09, WR5 .06,
TE2 .08, RB3 .02 of receiving yards, and so on), normalised over the unpriced
players actually on the slate.

`config/nfl.json` → `projection.unpriced_volume` chooses `salary` (the
salary + TD-odds model in `refactor.volume`, the default) or `environment`
(this allocation); `IVAN_UNPRICED_VOLUME` overrides it for an A/B. The model is
built in both modes because its team rows are a diagnostic on their own: the
projection step writes `usage.<stamp>.csv` beside the projections (team, stat,
total and its source, settled lines, residual, how many unpriced players share
it), and the Validation page shows them as **coverage** = settled lines / team
total. This needs no actuals: it is an identity on the marginals. At 100% the
priced players already account for the whole team; past it their means are
collectively too high and the unpriced players only get the floor. Every
player also records `volume_source` (market / environment / salary / lines)
and `unpriced_stats`, so the Projections table splits the error by it.

    scripts/nfl/backtest_usage.py --since 2025-11-01 --until 2025-11-30 [--modes salary,environment]

is a projection-only replay: it restores each archived slate with the clock
frozen at lock, fits the projections under each mode and scores both against
the actual DraftKings points (usage_ab.csv, a paired comparison on the players
the modes disagree on, and a validation report under
data/nfl/output/validation/usage_<stamp>/ that the page lists). Twelve showdown
slates take about a minute.

What the November 2025 slates said (first run): the two models tie on the 79
unpriced players (MAE 2.22 vs 2.25, both under by about 0.65 points, 48% of
those players scored zero), because the settled receiving lines already claim
a median 105% of each team's passing yards (83% of team-slates past 100%), so
the allocation only ever hands out the floor. Receptions (100%) and rushing
yards (98%) add up; receiving yards run 5-6% over, the mean-above-median
effect of right-skewed yardage props. Reconciling the priced WR / TE
receiving means to the QB line (a crude proportional scaling) halved their
over-projection on those slates (WR bias -0.89 → -0.40, TE -1.83 → -1.37);
that reconciliation is the next thing to build and test properly.

## Web UI

```bash
python scripts/nfl/serve.py            # http://<host>:8000
```

A single page with the five steps as forms (step 5 takes choose weights per contest and
remembers them), an entries-file upload, a live log of the running job, and pages for the
per-contest simulation reports, exposures, downloads (the DK upload files), job history,
and a Backtest page that reads the replay / after-tax reports (`data/nfl/output/replay*/nfl/report_*`,
`data/nfl/output/backtest/*`): headline profit and tax, the repeats distribution, scenarios,
the per-slate ledger, sim vs actual, formats, the volume sweep, plus the progress of a replay
still running and a button that rebuilds its report from the repeats finished so far.
To run it on the laptop only: `python scripts/nfl/serve.py --host 127.0.0.1` and open
http://localhost:8000.
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
| `scripts/nfl/build_models.py` | Every model the pipeline reads (DST table + odds fit, classic field, dupe, sd calibration, ownership calibration + model) → `data/nfl/models/`; `--cutoff` writes the as-of set, `--only` a subset. One function per model in `ivan.sports.nfl.model_build`. |
| `scripts/nfl/build_dupe_model.py` | Product-rule duplication model fitted on scraped fields → `data/nfl/models/dupe_model_v2.json`; `backtest` prints the leave-one-slate-out fit. |

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
  `data/nba/models/projection_calibration.json`, refit by the backtest. The
  resulting marginal is kept as 101 quantiles on `nba_player.fpts_quantiles`.
* **Same-game correlation** (`correlation.py`, used by
  `pre_lock/player.py`): one Gaussian copula per game maps correlated
  uniforms through each player's own quantiles, so skew and the
  double-double bump survive. The correlation is a league table by role pair
  (teammates both starting, starter with bench, bench with bench, and the
  opponent versions) fit on walk-forward residuals, with each specific
  pair's shared-game history shrunk toward it by 200 pseudo-games. Measured
  on 2025-26 the structure is small and stable across half-seasons:
  starter-bench teammates -0.06, bench-bench -0.03, everything else within
  0.01 of zero; and pair-specific history is noise (a pair's first-half
  correlation predicts its second half at r=0.08), which is why it is shrunk
  that hard. Who is out changes the *means* through the minutes allocation,
  not the correlations (teammate correlation is -0.03 with and without the
  team's star). Refit with `backtest_projection.py --fit-correlation`.
* **Odds are gospel.** A priced player keeps his market rate and is exempt
  from the team usage bump, so with nothing changed since the fetch he
  projects to exactly the market mean; only unpriced teammates absorb a gap.

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

Not yet ported from NFL: the web pipeline and the overnight job's safety
rails. DK status Q (questionable) is treated as playing. The projection snapshot keeps one `fpts_mean`
per player-day, so keep dropping `stok.csv` on a separate column if you want
Stokastic as a season-long yardstick.

## MMA

The UFC model from the standalone `mma_dfs` project, on Ivan's layout: a
SQLite file (`data/mma/mma_history.sqlite`), per-sport tables (`mma_player`,
`mma_fight`, the lineup pool, one simulation-result row per lineup per
contest, the DK contest and payout tables), `config/mma.json`, and one
script per step. Run top to bottom:

| # | Step | Command | What it does |
|---|---|---|---|
| 1 | Ingest the slate | `python scripts/mma/ingest.py` | No arguments: picks the Classic MMA slate on DraftKings (the one with the most contests; `--draft-group <id>` for another), pulls its fighter pool, fills each fighter's Win odds from the-odds-api (`ODDS_API_KEY`; DraftKings moneylines, matched to the fights by name similarity), fetches the slate's contests and payout tables, and writes the slate with a win probability and projection per fighter. The finish markets and volume lines are not on the-odds-api, so they are hand-entered: on a new card ingest writes `data/mma/input/fight_odds.csv` (fight, u60 sec, 1st round, itd) and `player_odds.csv` (1st round win, ITD, Win, sig strikes, takedowns) with the names and Win odds in; fill them from the sportsbook and run ingest again. Whenever those files describe the current card they are read as overrides (a non-empty cell wins over the API) and left alone. The summary says what is still blank. The latest `--main-events` fights are 5 rounds. `--players-csv <DK export>` uses DraftKings' CSV instead of the API. |
| 2 | Generate the field's lineups | `python scripts/mma/generate_lineups.py 6000 4` | `<solves> <workers>`. Each solve maximises jittered win probability (main-event fighters boosted) under DK's rules: six fighters, cap, salary floor, one fighter per fight. Unique results form the candidate pool. Clears the previous pool unless `--append`. |
| 3 | Simulate | `python scripts/mma/sim.py` | With no arguments, every contest in `data/mma/input/entries.csv` (the DraftKings entries export, as for NFL; `--entries <file>` for another), each against its own sampled field; contests ingest did not list are fetched by id. One contest instead: `--contest "mini-MAX"` (the upcoming DK contest whose name contains the text), `--contest-id <dk id>`, or hand-described with `--payouts payouts.csv --total-entries 47500 --fee 0.5 --name "UFC Sat"` (rank, prize rows as copied from the contest page). Each run samples a field of `field.size` entries from the pool, duplicating cap-spending and favourite-heavy lineups more, scales the payout table to that field, simulates every fight `n_sims` times (finish type from the fight markets, winner from the fighter markets for that finish, production by time fought) and ranks every entry, ties pooling their prizes. Writes each sampled field to `data/mma/output/field.*.csv`. |
| 4 | Choose | `python scripts/mma/choose.py [--roi-weight 0.8] [--diversity-weight 0.4]` | With no arguments, fills every row of `entries.csv`: per contest, as many lineups as you hold entries, written to `data/mma/output/chosen_entries.<stamp>.csv` in DraftKings' upload format (your entry rows with the six F columns filled), plus `chosen_entries.{lineups,exposures}.<stamp>.csv`. One contest instead: `--name <contest> --lineups 150` writes `chosen_lineups.{lineups,upload,exposures}.<stamp>.csv`. Greedy either way: quality = ROI weight × average ROI + (1 − ROI weight) × cash rate, minus the diversity weight times the largest fighter overlap with anything already picked. |

Scoring constants, the duplication model, the generator and the chooser are
all in `config/mma.json`, with the values the original used. `IVAN_SEED`
makes a run reproducible, as for NFL. Tests: `python -m pytest tests/mma`
(end to end on the sample slate under `tests/mma/data`).

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
├── sports/mma/             fight model, slate ingest, generator, field sampler, simulator, chooser
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
