# The field sampler against simpler variants, 2026-11-05

Question: does the two-group field (`field_model.simulate_field`, five knobs in `data/nba/models/field_two_group.json`:
optimizer share 0.45, provider noise 0.04, personal noise 0.05, casual noise 0.45, 3 providers) earn its knobs, and does
the slate's size (2 games vs 11) need to enter it? Method: every variant builds its field on the same public projection
(the slate's last capture before lock, as `field_backtest` rebuilds it) at 1,000 entries; the real field is subsampled to
1,000 so the shapes compare at one size. Scored per contest on the real field's shape (duplicate share, unique share,
share within 10 % of the public optimal, median strength, salary left, max copies, the top-8 ownership sum) and on
ownership (MAE, MAE over the real top 20, Spearman). Model fields at 1,000 entries carry sampler noise of about 0.03 in
duplicate share and 0.7 in top-20 MAE (two seeds of the live variant); differences under that are nothing.

Data: the 88 final derived contests on 24 slates (2026-10-20 .. 11-04; 8 two-game, 6 three-game, 4 four-game, 6 of
7-11 games). The shape numbers below are on the 17 **clean** slates (65 contests): props in the capture and the capture
under six hours before lock. The seven others measure the public projection's error (no props, a day-old capture), not
the sampler ([field_backtest_2026-11-05.md](field_backtest_2026-11-05.md)). The harness scripts of this run
(`sampler_ablation.py`, `sampler_confirm.py`, `softmax_ablation.py`, `analyze.py`) were run from a scratch directory and
are not in the repository; the per-contest tables are summarised here.

## The real fields by slate size (clean slates, 1,000-entry subsamples)

| | 2 games (16 contests) | 3-4 games (19) | 7-11 games (30) |
|---|---|---|---|
| duplicate share | 0.22 | 0.12 | 0.06 |
| within 10 % of the public optimal | 0.81 | 0.77 | 0.59 |
| median strength / optimal | 0.940 | 0.934 | 0.910 |
| salary left, mean $ | 187 | 136 | 128 |
| entries at the full $50k | 44 % | 49 % | 50 % |
| top-8 ownership sum | 389 | 332 | 272 |

The field converges on small slates and spreads on big ones, by a factor of 3-4 in duplicate share and 117 points of
top-8 ownership. A fixed set of knobs cannot follow this. Within one slate the contest matters too: duplicate share falls
0.026 per log of entries (t = 6, slate fixed effects) and max copies fall with entries and fee; the 11-04 slate ran from
0.02 (the 79k Zone Special) to 0.17 (a 3k Dime Time) to 0.33 (a 2k turbo). The sim gives every contest of a slate one
field.

## The live two-group field, same slates (model minus real)

| | 2 games | 3-4 games | 7-11 games |
|---|---|---|---|
| duplicate share | +0.11 | +0.12 | +0.03 |
| within 10 % | +0.11 | +0.10 | +0.09 |
| salary left, $ | +142 | +80 | +5 |
| max copies | +26 | +8 | 0 |
| top-8 sum | +14 | +6 | -12 |
| top-20 ownership MAE | 9.7 | 8.3 | 9.6 |

Right on big slates, far too convergent on small ones: twice the real duplication, twice the max copies, $140 more left
on the table. The culprit is the jammer group: 45 % of entrants on three near-identical providers with 5 % personal
noise collapses onto a handful of lineups when the pool is 40 players.

## Ablation over the five knobs (shape score = |d dup| + |d unique| + |d within10| + 5 |d p50| + |d salary| / 300)

| variant | 2 games | 3-4 games | 7-11 games | all | top-20 MAE |
|---|---|---|---|---|---|
| live | 0.98 / 1.01 (two seeds) | 0.67 / 0.64 | 0.39 / 0.41 | 0.62 | 9.2 / 8.5 |
| live, 1 provider | 1.07 | 0.72 | 0.43 | 0.67 | 10.4 |
| live, optimizer share 0.25 | 0.99 | 0.67 | 0.38 | 0.62 | 8.5 |
| live, optimizer share 0.65 | 1.18 | 0.93 | 0.75 | 0.91 | 11.5 |
| live, personal noise 0.10 | 0.89 | 0.60 | 0.37 | 0.57 | 8.6 |
| one group, sigma 0.06 | 1.72 | 1.25 | 1.15 | 1.32 | 14.5 |
| one group, sigma 0.10 | 0.98 | 0.60 | 0.89 | 0.82 | 9.9 |
| one group, sigma 0.15 | **0.53** | **0.55** | 0.70 | 0.62 | 8.0 |
| one group, sigma 0.22 | 0.81 | 0.59 | 0.44 | 0.58 | 8.2 |
| one group, sigma 0.30 | 1.08 | 0.66 | **0.37** | 0.63 | 9.1 |
| one group, sigma 0.45 | 1.66 | 0.97 | 0.60 | 0.97 | 10.5 |

"One group" is every entrant optimizing the public projection times (1 + sigma x noise): one knob. The providers do
nothing (1 vs 3 is noise). The optimizer share moves duplication and the top-8 chalk in opposite directions, so no
value of it fixes both. A single sigma matches the live model's score on average, and the best sigma per slate rises
with the slate: 0.15 on every two-game slate, 0.10-0.15 on three games, 0.15-0.30 on four, 0.22-0.45 on 7-11 games.

## One knob scaled by the slate: sigma = 0.15 (games / 2)^0.4

Confirmation run on the same clean slates (sigma 0.15 on two games, 0.20 on four, 0.26 on eight, 0.30 on eleven):

| variant | 2 games | 3-4 games | 7-11 games | all | top-20 MAE | top-8 sum, 7-11 g |
|---|---|---|---|---|---|---|
| live | 0.98 | 0.67 | 0.39 | 0.62 | 9.2 | -12 |
| one group, sigma(games) | 0.55 / 0.62 | 0.56 / 0.56 | 0.34 / 0.34 | 0.46 | 8.9 | -109 |
| + salary tilt 0.5 / $1k | 0.51 | 0.52 | 0.34 | 0.43 | 8.9 | -108 |
| + salary tilt 1.0 / $1k | 0.53 | 0.52 | 0.33 | 0.43 | 8.8 | -108 |
| sigma = 0.12 (games / 2)^0.55 | 0.68 | 0.53 | 0.35 | 0.48 | 9.3 | -110 |

The salary tilt (each entrant adds lambda x salary to the score, lambda uniform on [0, 2 x tilt] per entrant, so the
optimizer prefers spending the cap) removes the salary-left bias on every size (+13 -> -8 $ mean, +42 -> -11 $ at the
90th percentile) at no cost elsewhere; it stands in for the min-salary rule real optimizers run with.

What the one-group model still misses: its lineups sit too close to the public optimal on small slates (within 10 %:
+0.19 on two games, +0.21 on 3-4, +0.05 on 7-11; the live model +0.10 on all three), and on 7-11 game slates it
under-predicts the chalk by about 13 points a player over the top eight (live: -1.5 a player). Those two are the same
thing: on a big slate the real field is both more spread out and more concentrated at its top than one noise level can
be. The mixed-group section below tests whether a second, casual group fixes that without the providers.

## Two groups without providers: an optimizer group at sigma(games) and a casual group at 0.45

A casual share (lineups built on the public projection times (1 + 0.45 x noise), the live model's second group) beside
the optimizer group at sigma(games) with the salary tilt:

| variant | 2 games | 3-4 games | 7-11 games | all | top-20 MAE | within 10 %, 7-11 g | top-8 sum, 7-11 g |
|---|---|---|---|---|---|---|---|
| one group, sigma(games), tilt 1.0 | 0.53 | 0.52 | 0.33 | 0.43 | 8.8 | +0.05 | -108 |
| + casual 20 % | 0.63 | 0.53 | 0.35 | 0.47 | 8.9 | 0.00 | -116 |
| + casual 35 % | 0.81 | 0.57 | 0.40 | 0.55 | 9.0 | -0.04 | -121 |
| sigma = 0.10 (games / 2)^0.4, casual 20 % | 0.75 | 0.50 | 0.43 | 0.53 | 8.5 | +0.17 | -78 |
| sigma = 0.10 (games / 2)^0.4, casual 35 % | 0.75 | 0.53 | 0.36 | 0.51 | 8.5 | +0.11 | -91 |

The casual group buys the strength spread on big slates (within 10 % goes to zero error) and pays for it in salary left
and in the top-8 chalk; nowhere does it beat the plain one-group model. The live model's second group is not what the
one-group model lacks.

What it lacks on 7-11 game slates is a **convergent core**: at sigma 0.26-0.30 every one of its 1,000 lineups is unique
(duplicate share 0, max copies 1), while the real fields keep a duplicate share of 0.06 and 11-12 copies of their most
common lineup, and that core is where the chalk's top-8 sum comes from. The live model has that core (its jammer group)
and gets both right on big slates, at the cost of drowning the small slates in it.

A small core share beside the one-group model (core lineups: the public projection times (1 + 0.05 x noise), the live
model's personal noise, no providers), the rest at sigma(games) with the salary tilt:

| variant | 2 games | 3-4 games | 7-11 games | all | top-20 MAE all | top-20 MAE, 7-11 g | top-8 sum, 7-11 g | max copies, 7-11 g |
|---|---|---|---|---|---|---|---|---|
| live | 0.98 | 0.67 | 0.39 | 0.62 | 9.2 | 9.6 | -12 | 0 |
| one group, sigma(games), tilt 1.0 | 0.53 | 0.52 | 0.33 | 0.43 | 8.8 | 10.4 | -108 | -11 |
| + core 8 % | 0.59 | 0.49 | 0.33 | 0.44 | 8.5 | 9.3 | -86 | -9 |
| + core 15 % | 0.68 | 0.48 | 0.34 | 0.46 | 8.3 | 8.6 | -62 | -6 |
| + core 8 %, sigma = 0.15 (games / 2)^0.55 | 0.59 | 0.48 | 0.34 | 0.44 | 8.8 | 10.2 | -104 | -9 |

The core does what it should on big slates (the chalk's top-8 sum -108 -> -62, top-20 MAE 10.4 -> 8.6 at 15 %, the
copies of the most common lineup half-way back) and costs the two-game slates duplication (+0.02 -> +0.11), where the
one-group model already had the real core from its low sigma. So the core share wants to scale with the slate as the
noise does: about 0 on two games, 0.15 on eleven. That is the next grid to run (core share c(games) x sigma(games), two
parameters each); it was not run today.

## The ownership model's features (reference only, not in the live path)

`field_model.FEATURES` (ten features, frozen 2026-10-23) by leave-one-slate-out over the same 88 contests:

| features | top-20 MAE all | 2 games | 3-4 games | 8+ games |
|---|---|---|---|---|
| the frozen ten | 13.0 | 13.7 | 11.4 | 14.3 |
| seven (without promoted, questionable, freed minutes) | 12.5 | 13.5 | 11.5 | 12.6 |
| three (value on the props, salary, starter) | 13.6 | 14.8 | 12.2 | 13.9 |
| DK value + salary | 16.1 | 16.6 | 14.6 | 17.1 |

Three of the ten can go with no loss (the news features never helped; they were frozen on seven slates). The model is
a reference row in the field backtest, nothing reads it live, and it is 4 points of top-20 MAE behind the lineup
sampler on the same contests, so it stays as it is.

## What this means

1. The five-knob two-group model can be replaced by one noise knob that scales with the number of games, plus a
   salary tilt: fewer parameters and a closer field on every slate size, by a wide margin on the 2-4 game slates (shape
   score 0.98 -> 0.5 on two games). The providers and the optimizer share are removed. A small convergent core that
   also scales with the slate (none on two games, about 15 % on eleven) brings back the big slates' chalk; with it the
   model has two knobs and two slopes, each with a meaning (how far entrants' projections spread; how many sit on the
   optimal), against five with none.
2. Slate size is the first-order input the field model was missing. Contest size is the second (duplication falls with
   entries within a slate); the sim's one-field-per-slate design cannot carry it, so a per-contest field would need the
   field built once at the largest size and thinned by contest, or a sigma per contest.
3. The top-20 ownership error (8-9 points in every variant) does not move with the sampler: it is the public projection,
   as the 11-05 field backtest found. The sampler change above is about the field's shape (duplication, strength
   spread, salary), which is what the sim's payout scoring reads.
4. Not applied. The change is a model change to the live sim and waits for the field backtest to be rerun with the new
   sampler at the sim's sizes (3,000 and 10,000 entries), and for the per-contest sampler question, before
   `field_two_group.json` and `public_field.slate_field` move.
