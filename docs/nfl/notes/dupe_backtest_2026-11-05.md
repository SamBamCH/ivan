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
| as-of, every past slate weighted equally | 16 | 0.82 | -0.22 [-0.36, -0.06] | 0.96 | 0.93 | 0.155 / 0.192 |
| as-of, slate-balanced + entry-cap term | 16 | 0.81 | -0.19 | 1.02 | 0.92 | 0.166 / 0.192 |
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
6. **Weighting every past slate equally in the fit is worse, not better** (second run, 11:31):
   bias -0.22 against -0.09, MAE 0.82 against 0.80, with the bands shifted the same way. Balancing
   hands the small November 2025 and Saturday slates the weight of the big Sunday ones, and their
   levels are the noisiest. The scores were always one row per slate; the copy weighting in the fit
   stays. The season's imbalance is a coverage problem (below), not a weighting one.
7. **Our own entries carry twice the copies of the field's average entry.** 2,328 of our 2,534
   scraped entries were found in 89 fields (the rest are contests whose field was not derived).
   Mean copies 88, median 28, against 44 for a random entry of the same fields, weighted the same
   way; the ratio is 1.8 to 2.4 at every contest size (under 10k entries: 12 vs 6; over 100k: 141 vs
   77) and every cap (20-max 74 vs 34, 150-max 102 vs 55). Only 4 % of our entries were alone; on
   2026-11-03 we held a copy of the most duplicated lineup of the 237k-entry mini-MAX (1,174 copies).
   The dupe model's level for what we enter is right on average (as-of 85 predicted vs 93 held), but
   by cap the live file runs +0.17 log low on our 20-max entries and -0.14 high on our 150-max ones;
   with the entry-cap term both are within 0.08. So the chalk we play is not the dupe model lying
   about our copies: it is the chooser and sim preferring lineups the field also builds, which sits
   with the open "sim field top too flat" item (the sampled field holds too few copies of the chalk,
   so a chalk lineup that hits splits first place with fewer copies in the sim than in reality).

## Why only 16 slates of this season

The archive holds 51 NFL slates: 12 November 2025 showdowns (backfilled) and 39 of this season, every
one locked on or after 2026-10-11, the day the slate archive started. Of those 39, 24 are showdowns
(the rest classic, whose fields are not stored at the lineup level). Three limits, in order of size:

1. **Only the slates we play are captured.** A week has 14 to 16 showdown games; we ingest about five
   (Thursday, the Sunday early and late games we enter, Sunday night, Monday night). The other games'
   contests are listed in `historical_contest_information` (about 13 draft groups per Sunday, 2,800
   unarchived contests in October) but there is no capture and no projection for them, so no
   pre-lock ownership to fit on. Adding them needs the DraftKings pool at lock, which nothing stores.
2. **Nothing before 2026-10-11 is archived**, but 11 showdown slates from 2026-09-09 to 2026-10-08
   have projection snapshots and listed contests: `backfill_archive.py --since 2026-09-09 --until
   2026-10-10 --odds` rebuilds their captures from the snapshots and the odds-api history (about 230
   credits a game, 2,500 in all) and `settle.py --since 2026-09-09 --until 2026-10-10 --public 5`
   scrapes their public fields through the logged-in Chrome profile. That takes the season from 16
   scored slates to about 27.
3. **Per slate the settle job scrapes our contests plus three public ones** (`--public 3`): 6 to 11
   fields of 150 to 260 per draft group. More public contests per slate add fields of the same game
   (more single-entry and 5-max fields, which the by-cap table is short of: 11 and 2), not slates.
   `--public 6` on the nightly timer is cheap.

The balance question is the other side of this: November 2025 is 12 of the 28 slates but about 55 %
of the copies, and the fit weighs a lineup by its copies, so the parameters are set by the biggest
fields. The `asof balanced` variants give every past slate the same total weight in the fit
(`dupe_model.fit_weights`); the scores were always one row per slate, weighted equally.

## The term-by-term ablation and the model that went live (afternoon)

`backtest_dupes.py --ablation --no-actual-own`, the live terms with each dropped in turn and each candidate
added in turn, 16 slates forward:

| variant | log MAE | log bias | slope | top lineup pred/actual |
|---|---|---|---|---|
| salary + cpt + logproj (the terms until today) | 0.80 | -0.09 | 0.93 | 0.98 |
| without salary | 0.94 | -0.08 | 0.95 | 1.16 |
| without the captain exponent | 0.80 | -0.10 | 0.93 | 0.97 |
| without the projection gap | 0.85 | -0.12 | 0.97 | 0.22 |
| plus the linear projection gap | 0.81 | -0.10 | 0.87 | 1.67 |
| plus the QB-stack deltas | 0.81 | -0.09 | 0.93 | 1.06 |
| plus the entry cap | 0.79 | -0.07 | 0.98 | 0.98 |
| plus contest size (`logn`) | 0.78 | -0.17 [-0.28, -0.06] | 1.02 | 1.18 |

Decision, one term per finding and nothing that the test does not pay for: **ownership, salary left,
projection gap, entry cap** (`TERMS = ("salary", "logproj", "emax")`). The captain exponent came out
(nothing changes without it), the linear gap and the stack deltas came out of the code with the
zero-truncated Poisson fit (worse or nothing, twice now), and contest size stays a candidate only: its
pooled error is the best of the table but its slate-level bias is the worst, and the slate level is the
biggest open error. The samplers now read each contest's cap (`sample_field.contest_entry_cap`); a
contest without one is predicted at the default cap of 20. The live and as-of model files were rebuilt
with the new terms the same afternoon.

## What to run, and when

After the nightly settle has scraped a slate's fields (the Monday after a week, or any morning):

```bash
venv/bin/python scripts/nfl/backtest_dupes.py                      # slates since the season start
venv/bin/python scripts/nfl/backtest_dupes.py --no-actual-own      # half the time, without the ownership attribution
```

Read the `asof` row against the last run: the band over slates is the noise, so a term or a refit is
only called better when its band clears the `asof` one, and its by-cap table is flatter. The live
file's row is the error the chooser actually carried; when its bias drifts outside its band, rebuild
the model (`build_models.py --only dupe`). Rebuilding on all history, copy-weighted, is the current
recommendation; the entry-cap term is the first candidate change, pending the sampler hook. The
"our entries by cap" table is the one to watch for the chalk problem: our copies against the field's,
and whether the ratio comes down once the sim's field top is fixed.
