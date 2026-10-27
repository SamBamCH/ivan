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

## Engine backtest (2026-10-24): headroom allocator vs the solve on the real code path

project_team with actual availability, this season's 7 pre-lock captures (801 projected player-games) and last season's 11
historical-props days (1,430):

| | minutes MAE this / last | fpts MAE (played) this / last | priced minutes this / last | unpriced rotation minutes this / last |
|---|---|---|---|---|
| headroom (old) | 5.34 / 5.17 | 7.48 / 8.13 | 4.77 / 4.95 | 6.10 / 7.33 |
| solve | 5.43 / 5.16 | 7.25 / 7.78 | 4.42 / 4.96 | 6.63 / 7.04 |

Fantasy points improve on both windows (the thing that pays); minutes tie overall; priced players improve this season; this
season's unpriced rotation is the one loss (early-season group targets on 3-6 games). Switched: config projection.alloc = solve.
Both allocators over-project priced players' points this season (+2 to +3 bias): the books themselves ran high in week 1.
Anchor widths (book 3-4 / history 5-8 / deep 6-12) move the result by 0.1-0.2 minutes either way: kept at 4 / 5 / 6 / 8.
Group budgets: on this season's 3-6 games of history they cost 0.1-0.2 minutes (5.43 vs 5.33 without), on last season's full
history they gained it back on the unpriced rotation (7.04 vs 7.42), so they act only once a team has 8 games.

## Steps

1. DONE 2026-10-23: substitution pairs measured (diffuse, not predictive) and position groups (a real bias, removed by a
   group tilt held out). Analysis script: the job tmp repl_test.py; pairs code kept in projection/replacements.py.
2. DONE: position budgets per team (projection/groups.py; noisy: sd floor 8, half-life 10 games).
3. DONE: the solve (projection/solve.py) + engine.solve_allocation (alloc = "solve"); tests.
4. DONE: backtest on both windows (above); live since 2026-10-24.
5. DONE: the desk edits one number per player; the engine re-solves the team (balance / re-fill / keep-total removed).
6. Foul trouble: pre-game range, then the in-game update.

## The rotation skeleton (2026-10-24): the grid is the projection

The desk felt clunky because two minutes models were glued together: the solve decided how many minutes (books, history,
position budgets), the grid decided when but was proposed from stint history alone and never saw the solve, and the
engine rescaled each player's grid draw to the solved mean. Applying a grid pinned every row and threw the books away for
the whole team.

Reframed as the coach's slots (projection/skeleton.py):

- **Skeleton.** Every roster player's stint curve (rotation.curves: his on-floor share per minute over his last ten games)
  is a slot: "this stretch of the game belongs to this role".
- **Fill.** The solve still says how many minutes each available player gets. An available owner keeps his own slot. An
  absent owner's slot is **inherited**: the available players owed minutes beyond their own slot take it, same position
  group first (a big's slot only to bigs when any big is owed; guards and wings weighted 3:1 toward their group). When
  one heir is owed most of the slot and his own slot is no bigger, he **moves into it** and his old slot is vacated in
  turn (the backup big starts and plays the starter's stretches; the third big takes the backup's old ones). A player
  the solve wants well under his usual minutes keeps his strong stretches and loses his weak ones (his curve scaled by
  target / usual). A player with no history takes what he inherits, else a flat prior.
- **Schedule.** Minute by minute, five on the floor: each player's chance of being on from his (inherited) curve, pulled
  toward his target by the minutes he is still owed; a player who comes on stays at least three minutes and rests at
  least three when he goes off; breathers after fifteen straight; the named starters hold the tip. A balance pass then
  moves single boundary minutes between players (stints stretch and shrink, never fragment) until each row is within
  half a minute of its target where coherent stints allow (a second pass allows two-minute stints for a row still 1.5
  off). The grid's row sums ARE the projection's minutes; the solve's number is shown beside them as "tgt".
- **Locks.** A row the desk paints is locked: held exactly in the schedule, its total the player's minutes pin (a row
  painted empty scratches him). Apply saves the locked rows only; the others are re-filled around them by the next
  projection. Unlocking a row lifts its pin. The drawn-grid file carries "locked"; an old file without it locks every row.
- **Pool.** A player the schedule leaves off the floor has no minutes and leaves the projection (fpts None: out of the
  pool). The grid lists him under "off the floor tonight" with a + button that pins him at his history (at least 6 min).
- **Distribution.** Unchanged: the stint sampler jitters the grid's boundaries, hands a scratched player's stints to the
  best substitute, pulls the closers in garbage time, adds overtime. The grid's mean is its own, so the rescale is ~1.

Measured 2026-10-24 (scripts/nba/backtest_skeleton.py: walk-forward 2025-12-01..21, 246 team-games, oracle availability
and starters, no books so the solve anchors on history, 400 sims; players with 5+ games; the final build, with the
line-coverage rule below):

| | old: solve mean, proposed grid shape | new: skeleton grid is the projection |
|---|---|---|
| minutes MAE / bias | 4.55 / -0.02 | 4.57 / -0.09 |
| fpts MAE / bias | 7.72 / +0.18 | 7.73 / +0.15 |
| minutes 10-90 coverage (band) | 0.808 (15.4) | 0.796 (15.4) |
| fpts 10-90 coverage (band) | 0.783 (25.3) | 0.783 (25.3) |
| minutes MAE by actual minutes 28+ / 18-28 / 9-18 / <9 | 3.80 / 4.92 / 4.67 / 5.73 | 3.82 / 4.94 / 4.63 / 5.91 |

A tie on every measure: the grid can be the projection at no cost in accuracy, and the desk gets one object to edit.
The skeleton left 27 player-games off the floor (0.1 per team-game); all played, 4.7 minutes and 3.6 fpts on average,
four of them 10+ minutes (end-of-bench call-ups the oracle roster knew about). Switched live 2026-10-24: the live slate's
rotation_for returns skeleton inputs whenever config projection.minutes_model is "rotation".

**Lines (2026-10-24).** The schedule keeps a big and a guard on the floor in every minute (skeleton.cover_groups: a line
is enforced only when two or more of its players have targets adding to 46+, since one big cannot cover a game; the
first version enforced it for a lone big and pinned him for 48 minutes, minutes MAE 4.86). Swaps that would leave the
floor without the line are refused, the last places in a minute go to a missing line, balance moves cannot take the only
big off a minute, and a merged short stint goes to the same line. The desk shows the lines as **lanes** under each
grid: the five spots traced minute by minute with the holder's name on each stretch (rotation.spots_of; a gap means
nobody held the spot). Before the rule Philadelphia's centre lane ran Embiid, Bona, Walker, Love, Wade, Simons; after
it, Embiid and Bona alone, as in the game played the night before.

**Team view (2026-10-24).** Two grids with aligned minute columns: tonight's (paintable) over a played game (dropdown for
any game this season, fouls marked). Both panels list the starters first (tonight's by Rotowire, the game's by who was
on at the tip) closed by a dark rule, then the bench in one shared order (the saved projection's minutes); a player who
played the game but is out tonight is struck through in the game panel: those are the minutes to give away.

### Fouls (2026-10-24)

Personal fouls are parsed from the stored play-by-play (pbp.fouls_from_pbp -> nba_foul: player, elapsed seconds, period,
his personal count after the foul, the kind; technicals kept with personal = 0). Backfilled for all 659 games; the
nightly parses new games. The desk shows them where a reader asks "did fouls cut his minutes": a digit on the played
game's grid at the minute of each foul, a pf column, and in every player's recent-games table (pf, with a game marked
red when he had four or more fouls and played under 80% of his recent average). The foul-trouble sampler model and
blowout-conditioned stints are deliberately not built (2026-10-24): the slot framing was the priority.

### Spot colours (2026-10-24)

Every on-floor cell is coloured by which of the five lineup spots the player is filling (rotation.spots_of, mirrored in
the desk's JavaScript so the colours follow a repaint). The five on at the tip own the spots, guards first and bigs last;
when a player goes off, the player who comes on in his place inherits his spot: same position group first, then the
nearest position; a returning player reclaims his own spot when it is freed in the same minute. So the backup centre's
stretch wears the starting centre's colour, and the legend names each spot after its starter. Played games get the same
colouring, which shows how the coach actually filled each spot.

### Recorded substitutions (2026-10-24)

The play-by-play names every pair ("SUB: Eason FOR Smith Jr."); the stint table had dropped the pairing. Now
pbp.stints_from_pbp collects the resolved pairs and store_subs keeps them in nba_sub (35,836 over 659 games, backfilled;
the nightly parses new games; scripts/nba/pbp.py subs). What they are worth:

- Judged against the recorded pairs, the lanes' old inference (timing + position) named the right spot at 63% of the
  substitutions this season (836 of 1,332 over 50 team-games).
- The feed's pairing is only a role replacement when one player comes in for one. At single swaps 61% of pairs keep the
  position group; at multi-swaps 40%, and re-pairing by group alone would keep it in 202 more of 782: the scorer types
  the pairs in order, not by role (Philadelphia 2026-10-23 minute 8: "Brown for Embiid, Bona for LeBron, Maxey for
  Edgecombe"). So recorded pairs drive the lanes only at single swaps; a multi-swap keeps the exact set of ins and outs
  from the feed and pairs them by own spot, then position group, then the coach's habit within the group, then position.
- The coach's habits (pbp.sub_shares: share of a player's single-swap exits each teammate covered, recency weighted) go
  into the lanes as the within-group tie-break and into the heirs of a vacated slot (HABIT_W). As a bonus in the
  scheduler's swap choice they made the lines worse (Bona for Maxey, Simons for Embiid) and the backtest no better
  (fpts MAE 7.754 vs 7.732), so HABIT_BONUS is 0.
- Backtest with the final configuration (same window and harness, the habits passed as known before each game):
  minutes MAE 4.57 vs 4.55 old, fpts MAE 7.72 vs 7.70, minutes coverage 0.795 vs 0.808, fpts coverage 0.782 vs 0.786:
  a tie, as before. Philadelphia's centre lane on the played game is exact (Embiid, Bona only) and tonight's model lane
  matches it.

### The minute grid and the stint reconstruction (2026-10-24, "Allen comes in for Tyson with a one-minute offset")

Cleveland's data was exact (Allen in for Mobley, Tyson in for Allen, each at one recorded moment); the offsets were made
by the grid and by the reconstruction. Measured over this season's 50 team-games, before -> after:

| | before | after |
|---|---|---|
| minute columns without five on the floor (of 2,400) | 102 | 10 |
| lane gap-minutes | 186 | 7 |
| on-floor cells without a spot | 206 | 17 |
| recorded substitutions whose out and in change cells in different columns | 28 | 5 |
| player-games (season) with raw stint total off the box by >15% | 173 | 93 |

Three fixes:
1. **Cut at the nearest minute.** The played-game grid turned a cell on when the player had 30+ seconds of it, which put
   both players on at an exact half minute and let one swap straddle two columns. Every stint boundary now rounds to
   the nearest minute, so one substitution time moves the player going off and the player coming on in the same column.
2. **Name suffixes.** The feed writes "Reed" for Paul Reed Jr. and "Holland II" for Ron Holland; the resolver had no key
   for either, so Detroit lost 14 of 32 substitutions in one game (3-4 on the floor for 35 minutes). Suffix tokens are
   stripped from labels and box names on both sides.
3. **Shared initials.** Oklahoma City's Jalen, Jaylin and Kenrich Williams all appear as "J./K. Williams"; an action by
   "J. Williams" with neither on the floor was guessed, and the guess opened a phantom stint at the period start (seven
   on the floor for twelve minutes). The description's abbreviated first name ("Jal. Williams", "Jay. Williams") now
   breaks the tie, an unresolvable action name is not guessed, and the "came in at the break" rule only fires for a
   player who has not been on at all in that period. scripts/nba/pbp.py rebuild re-reconstructs every game from the
   stored raw feed (stints, fouls, pairs); all 659 games were rebuilt.

### Line alignment in the schedule (2026-10-24)

With the data exact, the model showed the same one-minute offset: the backup big entered at minute 8 and the starting
big left at 9 (the curves average games whose swap times differ by a minute), so for a minute the lanes handed the
centre's spot to a wing. skeleton.align_lines snaps such a pair into one minute: the two entrants of adjacent minutes
trade entry minutes when that pairs the big with the big, the floor count is unchanged, no line is left uncovered, no
stint drops under three minutes, and each player trades at most one minute. It runs before the balance pass (which
restores the totals the trade moved). Running it a second time after the balance cleaned a few more lanes but traded
totals for them: fpts MAE 7.756 against 7.703, so ALIGN_AFTER_BALANCE is False.

Backtest, final configuration (rebuilt stints, line coverage, pre-balance alignment, habits as tie-break only):
minutes MAE 4.55 vs 4.55 old, fpts MAE 7.72 vs 7.71, minutes coverage 0.796 vs 0.807, fpts coverage 0.779 vs 0.783.
Tonight's lanes: Philadelphia's centre lane Embiid/Bona only; Houston's Durant lane Durant/Bogdanovic only;
Cleveland's Mobley lane Mobley/Allen, Allen's own spot covered by Merrill and Tyson as in the game played.

### Rotation depth (2026-10-24): "we project 12 bench players, the coach plays 5"

Teams use 10 players with 5+ minutes (2025-26: mean 10.0, p10 9, p90 12); the skeleton gave Brooklyn 15 players minutes
because the solve hands every deep-bench man his expected minutes (play probability x minutes when he plays), honest on
average and wrong on any one night. groups.rotation_depth measures each team's depth (players with 5+ minutes over the
last 10 games, recency weighted, 8..12); skeleton.choose_rotation keeps that many by target plus every locked, pinned
or named-starter row; the kept players are re-solved through their anchors (the freed minutes land where the anchors
are loosest, not pro rata on the stars: a pro-rata spread put Philadelphia's starters at 40+); the rest get nothing and
leave the pool, listed under the grid with a + to add back. Config projection.depth_cut (true) and depth_extra (0).

Backtest, three December weeks, dressed-but-DNP players counted available (live conditions: nobody knows who sits):

| | old path | skeleton, no cut | cut at depth+1 | cut at depth |
|---|---|---|---|---|
| fpts MAE, all players with 5+ games | 7.54 | 7.76 | 8.08 | 8.34 |
| fpts bias | +0.09 | +0.07 | +0.56 | +1.09 |
| minutes MAE, actual 28+ | 4.53 | 4.57 | 4.28 | 3.98 |
| minutes MAE, actual 18-28 | 5.09 | 5.10 | 4.87 | 4.67 |
| minutes MAE, actual 9-18 | 4.72 | 4.66 | 4.40 | 4.61 |
| minutes MAE, actual <9 | 6.18 | 7.03 | 9.23 | 12.20 |
| players cut who played (per team-game) | - | 0.2 | 0.8 | 1.3 |

The trade is continuous: the deeper the cut, the better the rotation players (the ones a lineup rosters) and the worse
the fringe, who are projected at zero and play nine minutes when they do play; the kept players carry the minutes the
cut men actually take, a bias of about a minute each at full depth. Under oracle availability (every available player
played) the cut only costs (fpts MAE 7.92 vs 7.72), since there is no DNP uncertainty for it to resolve. Kept at the
measured depth for the grid and the pool (the user's framing: the grid is the coach's rotation); depth_extra 1 or
depth_cut false are a config edit away. The projection means of the kept players run high by that bias.
