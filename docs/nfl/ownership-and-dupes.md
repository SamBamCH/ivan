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

How many copies of a lineup the field holds, fitted on the real fields the settle job keeps:

```
copies(L) = N · k · ∏_flex (own_i/100)^beta · (own_cpt/100)^beta_cpt · exp(gamma · min(salary_left, knee)/1000) · [projection-gap term]
```

by copy-weighted least squares on the log scale (`METHOD = "wls"`; a zero-truncated Poisson fit was
tried 2026-10-14 and was worse held out), per format, with `TERMS = ("salary", "cpt", "logproj")`: a
salary-left term, a separate captain exponent and a projection-gap term, all held-out verified; the
stack deltas are measured but left out. Classic fields barely duplicate by ownership. Both samplers
use the same function for their lineup weights and for every candidate's predicted copies, so the
sampled field's concentration and the dupe prediction of our own lineups agree. `build_dupe_model.py
backtest|build [--cutoff]` refits; the as-of build writes a version per cutoff
(`dupe_model_v2.json`). The two-stage quantile + GBM model that under-predicted the chalk 3x was removed
in 2026-10.

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
terms on all history, on this season only, ownership only, and the live terms plus the entry-cap term
(`emax`, log entries per user from `historical_contest_information.multi_entry`; in `ALL_TERMS`, not in
`TERMS`, because the live sampler does not pass the contest's cap yet). Per slate it reports the
copy-weighted log MAE and bias (log(actual / pred), > 0 = the field duplicated more than predicted), the
calibration slope, the top lineup's predicted / actual copies and the share of entries in the 100 most
duplicated lineups; pooled, the calibration by predicted copies (from projected and from actual
ownership) and by the contest's entries-per-user cap; and our own entries' predicted copies against the
copies the field held. Output `data/nfl/diagnostics/dupe_backtest/<stamp>/` (`report.txt`, `slates.csv`,
`summary.csv`, `calibration.csv`, `by_entry_max.csv`, `ours.csv`, `summary.json`); about ten minutes, half of it the actual-ownership rerun (`--no-actual-own`).
`build_dupe_model.py backtest` (leave-one-slate-out) stays for the ablation but is not the acceptance
test: it leaks the later slates into every fold and is weighted toward November 2025. First run
2026-11-05 ([notes/dupe_backtest_2026-11-05.md](notes/dupe_backtest_2026-11-05.md)): the live file
over-predicts 16 % copy-weighted on the 13 slates after its fit, the per-slate level swings a factor of
two either way, single-entry fields are under-predicted 1.7x and 150-max fields over-predicted 0.75x
(the entry-cap term flattens both), training on this season only is worse, and our own entries average
about 95 copies in their fields, more than the field's average entry.
