# NFL: the field and the contest simulation

## The field sampler (`sample_field.py`, `model_calculations.py`)

Which of the generated lineups the contest's other entrants hold and how many copies of each.
Showdown: lineup weights from the dupe model's product rule on projected ownership, raked to the
projected ownership (`field.ownership_calibration`), captain handled with its own exponent. Classic: the
field-shape random forests (`classic_*.pkl`, trained on scraped classic contests by `build_models.py`)
plus the same dupe rule. The sampled field's exposure is what the Validation page compares to real
ownership; the dupe model gives every candidate in our pool its predicted copies. Known open item: the
sim field's top is too flat (0.49x the real field's top-score ratio).

## The Monte Carlo sim (`simulate_contest.py`, `sim.method = montecarlo`, the live default)

For every contest in the entries file: a field of `field_cap` entries (default 10000; `-c`) sampled from
the pool, the payout table scaled to it (`ivan.sports.common.payouts.scale_payouts_correctly`), `n_sims`
(default 1000; `-n`) correlated draws of every player on the field, lineups scored and ranked, ties
sharing the pooled payout. Since 2026-10-22 one draw is shared across every contest of the slate
(`SlateContext` / `SlateField`): world *w* is the same game in every contest, which is what lets the
slate chooser treat the book as one. Per unique lineup the result is fpts / payout / ROI / rank per
simulation (`lineup_simulation_results`, replaced per contest), plus the sim artifact
(`output/sim/<stamp>/contest_<id>.npz + .json`: the sampled field, every candidate's cash % / ROI /
predicted dupes, the players' sampled quantiles and exposure, the cash line and winning score per
simulation) and the report. `-d` writes extra diagnostics.

## The conditioned sim (`conditioned.py`, `sim.method = conditioned`, the lab)

Built 2026-10-19 to replace sampling: worlds are the game latents (team points), a lineup's conditional
score is Normal given the world, the field is a mixture of such Normals, the count of entries ahead is a
compound Poisson over duplicated lineups inverted exactly at the top of the payout ladder (inverse FFT),
so every lineup is credited with what it was positioned for in each world and a thousand worlds carry
what millions of counted simulations would (Rao-Blackwell). It found the heteroscedastic sd problem (the
power-law sd calibration) and a shared-lineup dependence (`SHARED_OWN_PCT`: a player left out of the
world makes a spurious uniqueness edge the chooser finds). **Reverted from the live path** the same week
because its count model breaks when the field's lineups are deterministic given the world; it stays
selectable by config for the lab, and its "what to root for" worlds feed the Choose page when present.

## The sim report (`sim_report.py`)

One `ContestSimReport` per contest (`contest_sim_report` row and
`data/nfl/diagnostics/contest_<id>_<name>/report.json`) with sections contest, field, payouts,
simulation, correlation, checks. The Contests page renders it; the checks are the invariants the sim
verified (payouts conserved in every simulation, ties split correctly, the field's average return
equals the rake, sampled scores match projections, sampled correlation matches the model). `bad` means
do not trust that contest's results and usually says what to re-run. The 2026-09-10 verification pass
that built the report also found and fixed sampler and payout bugs.

## Real-contest placement (`placement.py`, `historical_contest.py`)

After the games, a lineup's actual rank and payout in the real field with tie pooling, from the
derived `lineups.parquet` and `scores.npy` the settle job keeps per contest. This is what the
validation and the tax backtest settle against, so the sim's claim and the real finish are compared on
the same payout table.
