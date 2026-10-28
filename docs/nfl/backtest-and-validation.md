# NFL: backtests, replay and validation

Every stage writes its prediction down when it makes it (`ivan.core.artifacts`); after the games the
settle job brings in what happened; the tools below line the two up. Two numbers come out of every
backtest: the result and the confidence (how much of it is the pipeline's own randomness).

## Validation: projected vs actual per component (`validate.py`, `/validate`)

```bash
venv/bin/python scripts/nfl/validate.py --since 2026-09-01 --until 2026-10-12 [--in-field] [--no-snapshots]   # live runs
venv/bin/python scripts/nfl/validate.py --replay data/nfl/output/replay_nov [--repeat k]                      # a replayed window
```

Pairs each contest's last pre-lock choose run with the sim and projections it read, scores every
lineup the way the settlement does, and reports with pass / warn / fail rows (`ivan.core.validation`,
adapter `ivan.sports.nfl.validate`): **projections** (bias, MAE vs a salary-only baseline, PIT
histogram, 10-90 coverage, by position / salary tier / ownership quintile / volume source),
**environment** (the drawn team points, plays, pass rate vs actual), **usage coverage**, **ownership**
(projected and the sampled field's exposure vs actual, chalk bias), **field** (cash line, winning
score, percentiles, duplication profile vs the real field), **simulator / chooser** (does sim ROI rank
lineups the way the real field did; do our picks beat K random lineups from the pool), **our entries**
(sim cash % / ROI / dupes vs the real finish). Output `data/nfl/output/validation/<stamp>/`
(`players.csv`, `field.csv`, `lineups.csv`, `pool.csv`, `pool_deciles.csv`, `summary.json`). Live slates
need the box scores and the scraped contests to settle.

## Replay: past slates with today's code (`scripts/replay.py`, `ivan.core.replay`, `replay_adapter.py`)

```bash
venv/bin/python scripts/replay.py --sport nfl --since 2025-11-01 --until 2025-11-30 [--menu config/replay/nfl.json]
    [--lineups 3000 --jobs 4 --sims 500 --field-cap 5000] [--repeats 20 --seed 0 --workers 3] [--through project|generate|field|simulate|choose]
    [--rebuild-db] [--slates KEY,KEY] [--redo] [--report-only] [--out data/nfl/output/replay_nov] [--bankroll 1000 --tax-year 2025]
```

Each archived slate in the window is restored (pool, props, lines) into a copy of the database
(`data/nfl/replay/nfl_replay.sqlite`; the live file is never written), the clock pinned to lock minus
one minute, and today's project → generate → simulate → choose run on it against the contests a menu
picks (`config/replay/nfl.json`: name patterns, entry counts, max fee, weights; or a `selection.json`
from the hub). Outputs per slate in `<out>/nfl/<lock>_<mode>_<key>/` in the live pipeline's own files,
then the settlement and tax report over the window. `--repeats N` runs generate → simulate → choose N
times per slate (projection once) under `IVAN_SEED = seed + k` into `rep_NNN/`, and the report adds the
window's profit as a distribution (`repeats.csv`, `repeats_by_slate.csv`, `repeats_by_contest.csv`,
`repeats_summary.json`, `repeats.png`: mean, percentiles, share of repeats in profit, pre and after tax,
slates bootstrapped, without the best slate). Profit that shows only in some repeats is luck, not
edge. A slate-repeat is about 60 s; `--workers` splits slates over processes with their own database
copy (`replay.w<i>.sqlite`, deleted when the workers finish unless `--keep-db`: they hold only working
tables and grew to 12 GB each before 2026-10-28). Each slate's `replay.json` records the commit the
code was at (`git`: commit, dirty, subject), so a slate's builds can be told apart. `--through STAGE`
stops early for the component scoreboard. Slates that predate the archive are
rebuilt with `scripts/nfl/backfill_archive.py --since ... --until ... --odds` (about 210 credits per
showdown game).

What makes it honest: the captures (every ingest archives the slate's inputs), the clock, and
**as-of models**: `scripts/nfl/build_models.py --cutoff 2025-11-01 [--only ...]` writes
`data/nfl/models/asof/2025-11-01/` (DST table and parameters, classic field models, the dupe model, the
sd calibration, the ownership calibration and model, each trained only before the cutoff) with a
manifest; `ivan.core.model_store` resolves the newest version whose cutoff precedes the slate, falling
back to the live models while saying so.

## The Backtest hub (`/backtest`, `ivan.web.backtest_hub`, `readiness.py`)

One page per **run** (`data/nfl/output/backtests/<name>/`: `run.json`, `selection.json`, the replayed
slates). Steps: **Data** (what the window lacks, grouped by the fetch that fills it with its cost: box
scores, pre-lock captures from odds-api history at about 230 credits per game, unscraped contests through
the Chrome profile, a model version as of the window start; nothing fetched until ticked), **Contests**
(tick what to enter and how many), **Parameters** (repeats, workers, lineups, sims, field cap, seed),
**Results** (the sim's claim vs the actual outcome and after tax, the repeats distribution, the
assumption checks, every slate and contest; **Pin** keeps the run's heavy files past the retention
window, see below). `/backtest/plan` is the readiness per slate (captures, pool contests scraped, our
entries, box scores, model version).

**One slate's builds** (`/backtest/slate/<key>` = `scripts/nfl/slate_runs.py <key>`,
`ivan.sports.nfl.run_index`): what the archive holds for the slate (capture, contests scraped, box
scores, model version, settle status) and every run that replayed it, newest first, across the hub's
runs, the Models page's runs and the lab replay roots: when, the commit, the stage it ran through, the
parameters, the models version, the chooser, and for the slate's contests the fees, the sim's claim,
the actual profit and the mean over repeats. Linked from every slate on the hub's data step, progress
table and results, and from the hub's landing page (`slate_runs.py --list`). The index reads only the
files retention keeps, so a purged run still shows what it claimed and what happened. The 2026-10-19 backtest of the live config
over 10-11..10-19: +1.9% actual, the luck test passes; a 20-repeat run +14% mean ROI per repeat, 18 of 20
profitable, sd 16%; classic slates lose in every seed.

## Retention: what is kept, what is let go (`scripts/nfl/retention.py`, `ivan.sports.nfl.retention`)

Kept for good, because a model build or a backtest needs it and nothing recomputes it: the live
database with the slate archive (captures, raw artifacts, contests, standings, box scores, our
finishes), `data/nfl/raw`, `data/nfl/derived`, `data/nfl/models` (live, `asof/`, `archive/`),
`data/nfl/input`. Kept for good because it is small and it is the record: every run's `run.json`
and `selection.json`, each replayed slate's `replay.json` with its `projections` / `entries` /
`environment` / `usage` csv, the `report_*` directories and the validation `summary.json`.

Let go of, because running the replay again rebuilds it: worker clones of the replay database a day
after their last write (`scripts/replay.py` deletes its own when the workers finish; the sweep takes
what a crash or a cancel left, in the runs and in `data/nfl/replay/`), and 30 days after a run's last
write its heavy files: `rep_*/` (or the chooser files, simroi npz and `sim/` of a flat single-repeat
slate), the validation csvs, the run's `replay.sqlite` and the logs. The sweep writes `purged_at` into
`run.json`, which the run page and the slate page show. A run with `"pinned": true` (the Pin button, or
`retention.py --pin NAME`) keeps its heavy files; a run with a slate still running and written to in
the last day is not touched. Roots: `output/backtests/<name>`, `output/models/<component>/<stamp>`,
`output/replay*`. The nightly job runs the sweep last (`nightly.py --no-retention` skips it);
`retention.py --dry-run` lists what would go, `--days N` changes the window for a manual run.
On 2026-10-28 the first sweep removed 108 GB of leftover clones.

## The Models page: a scoreboard over code iterations (`/models`, `models_view.py`)

A fixed benchmark window replayed per commit with `replay.py --through STAGE` plus the field artifact,
scored per component (projections, environment, ownership, field, dupes, sim, chooser) with repeat noise
bands, so a change to one piece is judged on that piece. The direction chosen 2026-10-14 over any knob or
config UI: evidence per commit, not tuning by hand.

## Settle (`settle.py`, `ivan.core.slates`)

```bash
venv/bin/python scripts/nfl/settle.py [--days 3 | --since ... --until ...] [--public 3] [--no-scrape] [--no-box-scores] [--headless] [--dry-run]
```

For every archived slate past lock: the standings of the contests we entered and the draft group's
largest public contests (DraftKings through the Chrome profile; `IVAN_DK_USERNAME` so our finishes are
recorded), the week's box scores, the derived `lineups.parquet` and `scores.npy` per contest under
`data/nfl/derived/contests/`, per-player rows for classic contests, and a status row per slate
(`slate_status`: open, waiting, login_needed, settled, no_contests). Idempotent. The nightly timer runs it
([overnight.md](overnight.md)). It only knows which contests *we* played from the chooser's output on the
same machine, so ingest, contest fetch and choose on the producer for every slate you enter.

## The after-tax backtest (`backtest_tax.py`, `tax.py`, `/backtest/tax`)

```bash
venv/bin/python scripts/nfl/backtest_tax.py [--dk-history export.csv] [--runs-dir ...] [--since --until] [--bankroll 2000]
    [--cap 0.9 --rate 0.22 | --brackets --other-income 90000] [--mode gross_165d|net_dfs] [--played-only] [--in-field] [--volumes ...] [--n-boot ...]
```

What the chosen lineups would have returned after US federal gambling tax (2026 rules: 90% of losses
deductible; `config/tax.json`, verify with a CPA; engine `tax.compute_tax(W, L, cfg)`, tested in
`tests/nfl/test_tax.py`). Each entry is settled from the best evidence in order: a DraftKings
contest-history export, our own entries in a scraped contest, the lineup's actual points placed in the
scraped field, points only, nothing. The report: year to date (W, L, deductible, itemize decision,
taxable, phantom income, tax, after-tax ROI), a scenario table, the tax each format adds, sim vs actual
per contest, the running ledger with the point where tax owed exceeds the bankroll, a volume sweep
(bootstrapped years at 0.25x to 8x the observed slates) with a plot. Under `data/nfl/output/backtest/<stamp>/`.

## Smaller backtests

| Tool | Question |
|---|---|
| `scripts/nfl/backtest_projections.py` | the projection snapshots vs actual DK points; `--write-calibration` |
| `scripts/nfl/backtest_correlation.py --shrink ...` | how hard to shrink team correlation toward the league |
| `scripts/nfl/backtest_usage.py --modes salary,environment` | projection-only A/B of the unpriced-volume modes |
| `scripts/nfl/environment.py backtest --from 2024-1` | the game environment layer, rolling refit |
| `scripts/nfl/compare_choosers.py` | chooser settings on one slate's existing sim, settled against the real fields |
| `scripts/nfl/dupe_report.py` | the dupe model's misses per scraped field |
| `scripts/nfl/build_ownership_model.py backtest`, `build_dupe_model.py backtest` | held-out fits of the ownership and dupe models |
