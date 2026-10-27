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
