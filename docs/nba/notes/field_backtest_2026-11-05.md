# The field backtest: first run, 2026-11-05

`scripts/nba/backtest_field.py` over every final derived contest: 42 contests on 13 slates, 2026-10-20 .. 10-27, the
two-group field at 3,000 entries per slate with the parameters live since 2026-10-24 (optimizer share 0.45, provider
noise 0.04, personal 0.05, casual 0.45). Output `data/nba/diagnostics/field_backtest/20261105-*/`.

## What was wrong with the data first

- 31 of the 73 derived contests were not fields. From 2026-10-24 the nightly derived the late swap's **mid-slate**
  standings exports (taken a minute after a tip, the unplayed slots `LOCKED`): 16,812 entries matched into 43
  lineups, 59,453 into none. `standings.export_is_final` now refuses them, `derive_pending` removed the 31 folders,
  and `status` lists 45 contests (10-23 .. 11-04) as mid-slate, to be scraped final while DraftKings still has them.
  Without the fix the ownership model and the two-group field would have been refit on those.
- `standings.py status` read contests from the open slate's database only since the split into one database per
  slate (2026-11-03) and showed 0 contests everywhere; it now reads each slate's own file or the history file's
  pre-split copy.
- Two of the thirteen slates (10-24 turbo and night) were matched to a capture from the evening before (22-23 h old):
  nobody opened them on the day, so the autopilot never refreshed them. Their comparison measures the capture's age,
  not the model.

## Ownership

| | contests | MAE | MAE top 20 | Spearman | real top 10 found | top-8 share real vs model |
|---|---|---|---|---|---|---|
| model (two-group on the public projection) | 42 | 5.0 | 11.8 | 0.81 | 6.1 / 10 | 374 vs 374 |
| model, slates with props in the capture | 28 | 3.2 | 8.1 | 0.87 | | 365 vs 366 |
| model, slates without props | 14 | 8.6 | 19.2 | 0.70 | | 392 vs 388 |
| lock (what the desk showed; 10-25 .. 10-27 only) | 7 | 2.7 | 7.4 | 0.86 | 7.4 / 10 | 339 vs 337 |
| softmax ownership model (reference) | 42 | 5.0 | 11.9 | 0.86 | 6.0 / 10 | 374 vs 313 |

The chalk's total concentration is right (the top-8 share matches) but **who** the chalk is is not: of the real top
ten the model finds six. By real ownership the model is flat at the top: players the field put in 40 %+ of lineups
(real mean 53 %) get 42 % from the model, the 25-40 % bin (real 31 %) gets 27 %, and the 3-7 % bin gets 8 %. The
calibration from the model's side says the same: the players it puts at 40 %+ (mean 52 %) were really at 40 %.

## The misses are the public projection, not the sampler

Four of the thirteen captures carried **no props at all** (10-20 main slate, 10-21 night, both 10-24 slates), and
where props exist the books price points, rebounds and assists for about **40 % of the available players**. Every
other player's public projection comes from the replica's unpriced path (box-score history, salary, start rate).
Splitting the player-contests by that:

| | n | MAE | real chalk (30 %+) MAE |
|---|---|---|---|
| priced at the capture | 1,377 | 4.4 | 8.3 |
| unpriced | 3,234 | 3.6 | **24.5** |

Eighty percent of the player-contests missed by more than 20 points are unpriced players, and a third of the real
chalk is unpriced. The largest misses are all of one kind: a cheap bench player whose history says 20-29 points
(Gillespie, Goodwin, Caleb Love, Champagnie: model 52-57 %, real 1-11 %) or a star or value play the field knew
about from news the capture did not have (Trae Young 65 %, Brandon Williams 75 %, Dylan Harper 70 % real vs 5-15 %
model; Curry 42 % model, 0.2 % real: ruled out after a day-old capture). The convergent sampler then does what it
is built to do, piling half the field onto whoever the public projection ranks best per dollar, so one mispriced
cheap player becomes 55 % chalk.

## Shape

Real fields subsampled to 3,000 against the model's 3,000, strength measured under the rebuilt public projection
relative to its optimal: median 0.911 real vs 0.953 model, within 10 % of the optimal 61 % vs 85 %, duplicate share
0.29 vs 0.36, unique lineups 80 % vs 70 %, salary left 178 vs 270. The model's field is tighter and more duplicated
than the real one, but on the two slates with good props and no news (10-21 main, 10-22) the gap nearly closes
(within 10 %: 0.66 vs 0.61, 0.93 vs 0.99), and on the slates where the public projection was wrong the real field
looks weak under it by construction (10-21 night: 6 % within 10 %). The shape numbers are therefore not yet a clean
test of the sampler's knobs; they will be once the public projection is right.

## What to do, in order

1. **Scrape the 45 mid-slate contests final** (`scripts/nba/standings.py scrape --days 14`, needs the Chrome login)
   before DraftKings drops them: the 10-24 .. 11-04 slates are the ones with props at every capture.
2. **The unpriced path of the public projection** is the lever. Candidates to test against this backtest: treat a
   missing prop on a player the books priced earlier as news (he is doubtful or out, not a 25-point value); cap the
   unpriced projection for bench players by the depth chart / rotation skeleton rather than raw history; blend
   salary harder for cheap players. Measure with `backtest_field.py --no-model` first (seconds), then the field.
3. **Props at every capture**: a slate must carry its props when it locks (the 10-20 main and 10-21 night slates
   locked without any), and an unopened slate still needs a capture on its day.
4. Only then refit the two-group knobs (optimizer share, provider noise) on the shape table, slate by slate.
