# NFL: ownership and duplication

## Ownership (`ownership.py`, `ownership_model.py`)

The **sampler** projects ownership by generating value-driven lineups and counting exposures
(`config/nfl.json → ownership.num_lineups` 1000, `num_simulations`). It has the wrong habits for
roster-filler positions (it never plays a 4-point defense; the public plays one 13% of the time in
showdowns, kickers 20%) and overplays the chalk. Two corrections, both fitted on the scraped showdown
fields and both part of the as-of model build (`config → ownership.model`):

- **`sampler_calibrated`**: calibration lines per slot and position, actual ~ a + b × projected, DST
  and K on projected points instead (`build_ownership_calibration.py` → `ownership_calibration.json`).
- **`learned`** (the live setting; falls back to the calibration when no model file exists): a
  gradient-boosted regressor on the sampler's projection plus points, salary, value, ranks, position,
  slot, team share, contest size and fee (`build_ownership_model.py backtest|build` →
  `ownership_model.pkl`). Time split train Nov 2025 / test Oct 2026: MAE 2.51 vs 2.74 calibrated vs 3.00
  sampler. Since 2026-10-24 both train on showdown rows only (classic rows biased the captain fit; CPT
  MAE 1.58 → 1.17).

Each slot is renormalised to one lineup's worth (captain 100, flex 500). Known gap (2026-10-25): the
learned model is about 11 points per contest too flat at the captain top (chalk under, dust over); a
log-target fit fixes most of it held out but is not applied; the `own_sampler` feature has been
self-referential since 2026-10-13 because the raw sampler ownership is stored nowhere. Classic slates
are untouched until their fields are scraped at the player level. The acceptance test for any change
is the Validation page's ownership check (MAE, Spearman, chalk bias by quintile).

`field.ownership_calibration` (`iterations` 30, `damping` .7, `max_ratio` 4, `min_target_pct` .05) rakes
the field sampler's lineup weights to the projected ownership so the sampled field's exposure matches
the projection.

## Duplication (`dupe_model.py`, `dupe_analysis.py`)

How many copies of a lineup the field holds, fitted on the real fields the settle job keeps. Four terms:

```
copies(L) = N · k · ∏ (own_i/100)^beta · exp(gamma · min(salary_left, knee)/1000)
              · exp(delta_logproj · log(1 + points below the best lineup)) · exp(delta_emax · log(entries per user))
```

by copy-weighted least squares on the log scale, per format, `TERMS = ("salary", "logproj", "emax")`:
ownership is the base, money left on the table cuts copies (flat past `knee` = $2,000), points below the
best lineup cut copies (the optimizer crowd lands on the top projected lineups; at prediction time the
reference is the candidate pool's best), and the contest's entries-per-user cap sets the level (a
single-entry crowd lands on the chalk where a 150-max crowd spreads; from
`historical_contest_information.multi_entry`, `EMAX_DEFAULT` 20 when unknown). Both samplers read the cap
through `sample_field.contest_entry_cap` and use the same function for their lineup weights and for every
candidate's predicted copies, so the sampled field's concentration and the dupe prediction of our own
lineups agree. Classic fields barely duplicate by ownership and keep the ownership-only rule.
`build_dupe_model.py backtest|build [--cutoff]` refits (`dupe_model_v2.json`; the as-of build writes a
version per cutoff). It is the only dupe model: there is no config switch, and the archived two-stage and
pre-salary model files were deleted on 2026-11-05. Every term is there because the season backtest below keeps it (2026-11-05, 16
slates forward): without the salary term the log MAE goes 0.80 to 0.94, without the projection gap the
top lineup's predicted copies fall to 0.22x the real ones, the entry-cap term takes the single-entry
bias from +0.51 to -0.02 and the 150-max bias from -0.29 to -0.13. Removed the same day because the test
showed nothing or worse: a separate captain exponent, a linear projection-gap term, the QB-stack deltas,
the zero-truncated Poisson fit; a contest-size exponent (`logn`) stays a candidate in `ALL_TERMS` only.
The two-stage quantile + GBM model that under-predicted the chalk 3x was removed in 2026-10.

**Where it misses**: the Dupes page (`/dupes`) and `scripts/nfl/dupe_report.py [--contests] [--since]
[--showdown-only]` put the model's expected copies against the copies every scraped field held,
lineup by lineup: calibration by predicted copies, copy-weighted residual by captain, ownership-sum
decile, team split, QB stack, bring-back and salary left, the most duplicated lineups. Each residual is
shown from the ownership we projected and from the ownership the field showed, so a miss is attributed
to the dupe rule or to the ownership upstream. The dupe leak (our own lineups duplicated more than
predicted) was one of the 2026-10-11 validation findings.

**The season test** (`scripts/nfl/backtest_dupes.py`, `ivan.sports.nfl.dupe_backtest`) is the acceptance
test for any change to the model: every slate since the season start is scored by a model fitted only on
the fields locked before it (the rolling origin), the live `dupe_model_v2.json` is scored on the slates
after its `fitted_on`, and the variants run side by side with a 90 % bootstrap band over slates: the live
terms on all history (copy-weighted, the fit's default), the same without the entry-cap term (the model
before 2026-11-05), the same with every past slate weighted equally (`dupe_model.fit_weights` "slate"),
this season only, and ownership only; `--ablation` runs the live terms with each dropped in turn and each
candidate of `ALL_TERMS` added in turn, which is how `TERMS` is decided. Per slate it reports the
copy-weighted log MAE and bias (log(actual / pred), > 0 = the field duplicated more than predicted), the
calibration slope, the top lineup's predicted / actual copies and the share of entries in the 100 most
duplicated lineups; pooled, the calibration by predicted copies (from projected and from actual
ownership) and by the contest's entries-per-user cap; and our own entries' predicted copies against the
copies the field held, per slate and by cap next to the copies a random entry of the same fields has.
Output `data/nfl/diagnostics/dupe_backtest/<stamp>/` (`report.txt`, `slates.csv`, `summary.csv`,
`calibration.csv`, `by_entry_max.csv`, `ours.csv`, `ours_by_cap.csv`, `summary.json`); about ten minutes, half of it the actual-ownership rerun (`--no-actual-own`).
`build_dupe_model.py backtest` (leave-one-slate-out) stays for the ablation but is not the acceptance
test: it leaks the later slates into every fold and is weighted toward November 2025. First run
2026-11-05 ([notes/dupe_backtest_2026-11-05.md](notes/dupe_backtest_2026-11-05.md)): the live file
over-predicts 16 % copy-weighted on the 13 slates after its fit, the per-slate level swings a factor of
two either way, single-entry fields are under-predicted 1.7x and 150-max fields over-predicted 0.75x
(the entry-cap term flattens both, and is live since 2026-11-05), training on this season only or weighting
every past slate equally is worse, and our own entries carry twice the copies of a random entry of the same
fields (88 vs 44).
The season has few slates because the archive started 2026-10-11 and holds only the games we play;
the note says what a backfill of the September slates takes.
