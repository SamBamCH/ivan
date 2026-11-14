# The duplication model per contest (2026-11-14)

Question: the sim looped the contests of the entries file but gave every one the same field (5,000 entries
from one contest-blind model). What should differ per contest?

## Data

81 standings exports were on disk; 68 are Classic contests of three cards with a win probability per
fighter: 2026-10-10 (draft group 154571, 13 contests), 10-17 (154987, 18), 11-07 (156055, 37, the first
with single-entry fields). The 10-24 satellites (155302) have no stored win probabilities and the 11-08
contests are Captain Mode; both are left out. 63 contests hold 100 entries or more: 342,190 entries.
Each contest is laid on every valid lineup of its card (46k to about 120k lineups), zeros included.

## What varies by contest

Fitted one contest at a time, within a card, the coefficients move a little and in the same direction on
every card: fewer entries per user means more weight on spending the cap exactly, on the top-ranked builds
and on the main event, and less on summed win probability. Fitted with a slope per coefficient in
log(entries per user), log(field size) or log(fee), and tested on the card left out:

| terms | deviance explained (held-out cards) | top-100 share error | unique share error |
|---|---|---|---|
| none (one model for every contest) | 0.6498 | 0.79 | 0.160 |
| entries per user | 0.6500 | 0.73 | 0.141 |
| field size | 0.6478 | 0.72 | 0.134 |
| fee | 0.6465 | 0.79 | 0.158 |
| entries per user + size | 0.6452 | 0.72 | 0.132 |
| all three | 0.6427 | 0.71 | 0.128 |

(errors: mean absolute relative error of the expectation, before lumpiness.) The entry cap is the only
term that does not cost fit on a new card, and its five slopes have the same sign and similar size in all
three leave-one-card-out fits. Size and fee are confounded with it (the big fields are the 150-max ones)
and make the held-out deviance worse. The cap is what the model carries; it is also what the NFL model
carries (`emax`).

## The larger miss is the card, not the contest

The expectation alone is far too smooth on a card it was not fitted on: the 100 most copied lineups hold
7.5% of a 150-max field against 14.0% real, the most copied lineup 39 copies against 114. The model knows a
fighter by salary, win probability and the main event; salary and win probability correlate 0.96 to 0.99, and
real ownership correlates only 0.5 to 0.7 with win probability (10-17: Cerminara $9,000 at 71% to win was 7%
owned, Koslow $8,000 44%). An ownership projection per fighter is the missing input.

Until there is one, the lumpiness term stands in for it: a Gamma(alpha) multiplier per lineup. Its size must
come from cards the coefficients did not see. The in-card moments estimate the old file used (2.5 on one
card) is too smooth; maximum likelihood on the held-out rows of fields of 4,000 entries or more gives 0.68
(by card 1.3, 0.67, 0.51), and a field sampled with it has the real shape:

| held-out contests | n | unique lineups per entry (real / model) | top-100 share | most copied |
|---|---|---|---|---|
| 150 max | 12 | 0.393 / 0.414 | 0.140 / 0.143 | 114 / 111 |
| 20 max | 8 | 0.514 / 0.530 | 0.217 / 0.196 | 58 / 46 |
| 2-14 max, under 700 entries | 35 | 0.891 / 0.925 | 0.574 / 0.541 | 4 / 4 |
| single entry | 2 | 0.741 / 0.771 | 0.353 / 0.337 | 10 / 15 |
| single entry, under 700 entries | 6 | 0.873 / 0.916 | 0.813 / 0.789 | 5 / 3 |

The shape is right; which lineups are the chalk is partly a coin flip per lineup, where in a real field it
follows the fighters the public likes.

## What changed

- `ivan.sports.mma.dupe_model`: the fit above (multinomial per contest, entry-cap slopes, held-out alpha);
  `scripts/mma/build_dupe_model.py` writes a version-3 file and prints the coefficients by entry cap and the held-out shape table.
- `sample_field` and `simulate_contest`: each contest's field uses its own entry cap and its own size up to
  `field.size`; with `IVAN_SEED` each contest still draws its own field.
- `cards.py`: ingest stores the card; `scripts/mma/derive_contests.py` rebuilds every field from raw.
  Records for the three fitted cards and for 11-14 were written by hand that day.

Open: the ownership projection; fields above `field.size` at their real size (needs the results stored
outside one JSON row per lineup); three cards is thin, and only one of them has single-entry fields.
