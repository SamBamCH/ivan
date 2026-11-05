# 2026-11-05: the showdown dupe model, tested forward over the season

The first run of `scripts/nfl/backtest_dupes.py` (`ivan.sports.nfl.dupe_backtest`), slates 2026-10-11 ..
2026-11-03, 222 showdown fields over 28 slates in the archive (10 slates of November 2025 and 18 of this
season at that point; 16 of this season's had enough history before them to score). Report under
`data/nfl/diagnostics/dupe_backtest/20261105-*/`.

## Why a new test

`build_dupe_model.py backtest` is leave-one-slate-out over every field: the later slates are in every
fold and the fit is dominated by November 2025 (113 of the live model's 140 fields). Its held-out log
bias was -0.01 and the MAE 0.77, which looked fine. The live pipeline never sees that situation: at lock
it holds a model fitted on the past. The season test fits on the fields locked before each slate and
scores that slate, scores the live file only on the slates after its `fitted_on`, and runs the
candidate variants side by side with a bootstrap band over slates.

## What the first run says

Bias is the copy-weighted mean of log(actual / predicted): > 0 means the field duplicated more than
the model said. Bands are 90 % over slates.

| variant | slates | log MAE | log bias | slope | top lineup pred/actual | top-100 share pred / actual |
|---|---|---|---|---|---|---|
| live file (fitted 2026-10-15) | 13 | 0.81 | -0.16 [-0.32, +0.02] | 0.94 | 1.09 | 0.162 / 0.190 |
| as-of, the live terms, all history | 16 | 0.80 | -0.09 [-0.24, +0.07] | 0.93 | 0.98 | 0.151 / 0.192 |
| as-of, this season's fields only | 15 | 0.83 | +0.01 | 0.90 | 0.71 | 0.126 / 0.186 |
| as-of, ownership only | 16 | 1.04 | -0.12 | 0.97 | 0.12 | 0.062 / 0.192 |
| as-of + entry-cap term | 16 | 0.79 | -0.07 | 0.98 | 0.98 | 0.162 / 0.192 |

1. **The level moves slate to slate by a factor of two either way.** The as-of bias runs from -0.56
   (2026-10-18 early slate: the model predicted 1.75x the copies the field held) to +0.77 (2026-11-01
   Saturday slate, 31k entries: the field duplicated 2.2x more than predicted). The mean bias is near
   zero only because the swings cancel. Nothing in the rule (ownership, salary, projection gap)
   carries a per-slate level, and the two worst slates are the two smallest.
2. **By the contest's entries-per-user cap the misses go opposite ways.** Single-entry fields
   duplicate 1.7x more per entry than predicted (bias +0.51 over 11 fields), 150-max fields 0.75x
   (-0.29 over 33). The optimizer crowd spreads in a 150-max contest; a single-entry crowd lands on
   the chalk. The entry-cap term (log entries per user, from `multi_entry`) takes the single-entry
   bias to -0.02 and the 150-max to -0.13, the slope from 0.93 to 0.98, the MAE 0.80 to 0.79. The
   bands overlap, so it is a small gain on the pooled score and a clear one on the by-cap table.
   Turning it on needs the live sampler to pass the contest's cap through (`sample_field` has the
   contest row; the term is in `ALL_TERMS`, not `TERMS`).
3. **Given a prediction, the actual is about half of it through the middle.** Lineups predicted at
   5-50 copies held 0.45x that; lineups predicted under 0.5 held 1.1 (the floor of one). The same
   table from the ownership the field actually showed is almost identical (0.46-0.52 in the same
   bins), so this is the rule's dispersion, not ownership upstream: the predictor explains little
   of the lineup-level variance (contest correlations 0.5-0.7), and the regression to the mean is
   the symptom.
4. **Training on this season only is worse**, not better: MAE 0.83, the top lineup at 0.71x. Fewer
   fields hurt more than the season shift helps; keep all history.
5. **Ownership alone is useless for the top**: the top lineup at 0.12x and the top-100 share at a
   third of the real one. The projection-gap term is doing most of the work, as documented.
6. **Our own entries are the chalk.** 2,180 of our entries were found in 87 scraped fields. Their
   mean copies in the field: 93 (the as-of prediction 85; the live file 111 vs 110 on 1,781 entries),
   so the level of the dupe prediction for what we enter is right on average over the window, with
   the same per-slate swings (-0.60 to +0.78). Only 4 % of our entries were alone in their field;
   on 2026-11-03 we held a copy of the most duplicated lineup of the 237k-entry mini-MAX (1,174
   copies). The average entry in a 150-max field has about 70 copies, so we are more duplicated than
   the field. That is a chooser and sim finding, not a dupe-model one, and it sits with the open
   "sim field top too flat" item.

## What to run, and when

After the nightly settle has scraped a slate's fields (the Monday after a week, or any morning):

```bash
venv/bin/python scripts/nfl/backtest_dupes.py                      # slates since the season start
venv/bin/python scripts/nfl/backtest_dupes.py --no-actual-own      # half the time, without the ownership attribution
```

Read the `asof` row against the last run: the band over slates is the noise, so a term or a refit is
only called better when its band clears the `asof` one, and its by-cap table is flatter. The live
file's row is the error the chooser actually carried; when its bias drifts outside its band, rebuild
the model (`build_models.py --only dupe`). Rebuilding on all history is the current recommendation;
the entry-cap term is the first candidate change, pending the sampler hook.
