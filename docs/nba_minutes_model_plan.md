# NBA minutes: one model where the books, the rotation and history agree

Written 2026-10-23. The problem: we have good data that does not work together. The books' props are the best
predictor we have, the rotation (who plays with and instead of whom) is visible in the play-by-play, and history fills
the gaps, but each lives in its own patch (book minutes column, rotation grid, team allocator, usage bump, balance to
books). Editing minutes on the desk means balancing by hand, and edits interact in ways nobody can follow.

## What we know (measured 2026-10-23)

| Minutes estimate, this season's priced players (329 player-games) | mean miss | bias |
|---|---|---|
| book-implied (props / per-minute history, x 0.924; scale fitted on last season) | 4.06 | +0.01 |
| our minutes model | 4.92 | +0.13 |
| history alone | 5.42 | -0.10 |

No book offers NBA minutes props (84 markets, none minutes); book minutes are always implied.

Position occupancy over 57,783 team-minutes (645 games of stints):

| on the floor | 0 | 1 | 2 | 3+ |
|---|---|---|---|---|
| primary C | 21% | 77% | 3% | 0% |
| C-eligible (C, PF/C) | 3% | 67% | 28% | 2% |
| primary PG | 7% | 59% | 32% | 2% |

A C-eligible player is on the floor 97% of the time: centre minutes fill ~48 (more for two-big teams). The DK "PG"
label is loose (combo guards), so guard budgets come from each team's history, not a fixed 48.

## Step 1 result (2026-10-23): position groups yes, substitution pairs no

From 62,294 out-in pairings (stints, projection/replacements.py) over 647 games:
- **pairs are diffuse**: a rotation player's top replacement covers 24% of his exits (median), the top two 39%, and seven
  teammates cover >= 5% each. Routing an absent player's minutes through his pairs predicted worse (5.42 min miss) than a
  proportional split (5.21) in 404 games with one absent rotation player. Dropped.
- **position groups carry a real bias**: when a big is out, each other big gains +3.9 min and everyone else +0.2; a guard
  out: other guards +1.9, others +0.6; a wing out: no difference. Held out (fit on games before 2025-12-18, test after),
  a tilt toward the absent player's group removes the bias proportional allocation leaves on those players:
  backup bigs -2.02 -> +0.24, other guards -1.21 -> -0.08 (mean miss 5.63 -> 5.61 and 4.65 -> 4.63: single games are noisy).
  The big tilt is effectively exclusive (bigs' minutes go to bigs); guards ~4:1; wings none.
- **only ~30% of the freed minutes reach the regular rotation** (the rest: players without history, a tighter rotation);
  allocating all of them over-projects everyone (bias +1.6).

So the structure is position-group budgets with exclusive big replacement, not player pairs.

## The model

Per team, per slate:

1. **Budget.** 240 minutes plus the expected overtime share (241).
2. **Anchors**, each with a confidence:
   - priced players: book-implied minutes (sd ~4 minutes, from the table above);
   - unpriced players: history (min_ewm x play rate, sd from their own variance, wider for little history);
   - desk edits: the user's number, held tightest of all.
3. **Structure:**
   - **position groups** (big = C-eligible, guard = PG-eligible, wing = the rest): an absent player's minutes flow to his
     group first (bigs effectively only to bigs, guards ~4:1, wings proportionally), ~30% of them to the known rotation;
   - **position budgets**: each team's recent minutes by group as soft constraints (C-eligible on the floor 97% of minutes).
4. **Solve.** The minutes closest to the anchors (weighted by confidence) that sum to the budget and keep the position
   budgets: a small quadratic program per team, milliseconds.
5. **Rates.** Priced players: the book total over the solved minutes (the books' total holds whatever the minutes);
   unpriced: history with the usage bump.
6. **Distribution.** Unchanged: the rotation sampler and game script draw worlds around the solved means.
7. **Foul trouble.** Pre-game: a player's foul rate widens and lowers his range (fouls per 36 from the play-by-play).
   In-game (late swap): fouls in the live box score cut his remaining minutes, fitted from play-by-play checkpoints
   the same way the in-game update was.

## The desk

One minutes number per player. Rows grouped by position with each group's total against its budget (C 46 / 48).
Typing a minute makes it the strongest anchor and re-solves the team: the replacements move, the team always sums,
the typed number stays. "Balance to books", the fill and the keep-total logic are removed: the solve does all three.
Advanced controls (bands, rate overrides) behind a disclosure.

## Validation before anything switches

Backtest on this season's slates (props at lock) and last season's 11 days with historical props:
- minutes mean miss and bias, overall and for players whose teammates are out (where structure matters most);
- fantasy points mean miss for priced and unpriced players.
Against: the current model and book minutes alone. Ship only if it beats both; the desk switches after that.

## Steps

1. DONE 2026-10-23: substitution pairs measured (diffuse, not predictive) and position groups (a real bias, removed by a
   group tilt held out). Analysis script: the job tmp repl_test.py; pairs code kept in projection/replacements.py.
2. Position budgets per team from stints.
3. The solve (projection/solve.py) with anchors and constraints; unit tests on hand-built teams.
4. Backtest harness for minutes + fpts on both windows; compare.
5. Desk minutes editor on the solve.
6. Foul trouble: pre-game range, then the in-game update.
