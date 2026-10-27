# NFL slate chooser: one book, one bankroll, one world

*Plan written 2026-10-26. Status: phase 1 built the same day as the selectable method `slate` (greedy stays the default); unbacktested. Phases 2 and 3 not started.*

## The objective

Choose the entries and lineups across the whole slate to maximize

    mean over scenarios of  log( bankroll + sum over entries of (payout in that scenario - fee) )

in dollars, where a scenario is one way the slate plays out and the bankroll is the real
one, entered on the choose page. Only one scenario happens. The log makes the chooser care
about the distribution of the book's profit, not its mean, and with a real bankroll it is
the Kelly objective for a season played one slate at a time.

## What exists today (and why it falls short)

| Piece | Today | Gap |
|---|---|---|
| Worlds | One shared draw per slate since 2026-10-22: sim `w` is the same game in every contest (`SlateContext` / `SlateWorlds`). | None. This is what makes a slate-level book computable. |
| Per-lineup rows | `LineupSimulationResult.roi`: ROI per world. Monte Carlo path (live): a realized outcome per sim against a sampled field. Conditioned path (lab): the expected payout given the world. | MC rows are noisy (2000 realizations of lottery-shaped payoffs), hence `shrink_rows`. Conditioned rows hide the within-world distribution. |
| Chooser | `ivan.sports.nfl.choose.choose_lineup_ids`, one call per contest. Methods `greedy` (quality + trace distance) and `portfolio` (log utility of that contest's book, bankroll = 3x that contest's fees). | Each contest optimized alone; fake bankroll; on the conditioned rows the log is applied to an expectation (Jensen: a 2% shot at 500x reads as a steady +9). Own entries do not see each other. |
| Entries | `DKEntries(...).csv` fixes how many entries per contest; the chooser fills them. | Entry count and contest choice are not decisions. |
| Honest check | `cross_fit_book`: select on even worlds, score on odd. Replay harness + luck test. | Works per contest only. |

## Phase 1: slate-level portfolio on the existing rows (the piece that can be live first)

A third chooser method, `slate`, selectable like the others. Default stays `greedy`.

1. **Load the slate.** For every contest in the entries file, pull its rows and the latest
   `contest_sim_report`. Hard requirement: every contest reports `contest.draw == "slate"` and
   the same world count. Otherwise abort with a message; a misaligned world axis would make
   the sum across contests meaningless.
2. **Candidates.** `(contest, lineup)` pairs. Each carries its net dollars per world,
   `fee * roi[w]`, after the same empirical-Bayes shrinkage as today (per contest, so the
   pool it shrinks toward is the right one).
3. **Slots.** The entries file's count per contest. Each greedy step adds the candidate with
   the largest gain in `mean_w log(bankroll + net_w)` among contests with a slot left. A
   lineup already picked in a contest cannot be picked there again (DK rejects duplicates
   within an entry set); it can be picked in another contest.
4. **Bankroll in dollars.** One slate-level number on the choose page and in
   `weights.json` (`default.bankroll_dollars`), replacing the per-contest "x fees" knob for
   this method. Fees come from the entries file's `Entry Fee` column.
5. **Output.** The same per-contest `chosen_ids` the export already consumes
   (`choose_entries` slot assignment, upload CSV, `sim_rows`), plus a slate book summary:
   the book's dollar P&L distribution over worlds (p5 / p50 / p95, P(loss), P(lose half)),
   the per-contest split, and the slate cross-fit (select on even worlds, score on odd).
6. **Page.** The choose page shows the slate book above the per-contest sections; the
   per-contest viz keeps working from the per-contest subsets of the picks.
7. **Harness.** `compare_choosers.py` accepts `slate:bankroll_dollars=1000` so it runs next
   to `greedy` and `portfolio:1.2` on any archived slate.

Expect this: with a $1000 bankroll and $50 to $150 at risk, the log is nearly linear, so the
picks sit close to the shrunk-mean-ROI picks and the spreading effect is mild. That is the
correct answer if the sim's means are trusted; the shrinkage is what keeps noise out, and it
stays. A smaller bankroll figure is the honest way to ask for more risk aversion.

Not in phase 1: entry counts as decisions, own-entry ladder interaction, the Jensen fix.

### Finding from the first run (2026-10-25 KC @ SEA rows, 230 entries, $162 fees, $1000 bankroll)

The slate cross-fit scores both halves on raw rows, and the out-of-sample book ($+124)
beats the in-sample one ($+49) by about eight standard errors. That is the shrinkage, not
luck: `shrink_rows` weights each row by tau^2 / (tau^2 + se^2) with se from the row's own
sample variance, so a lineup that hit first place in the fit half has a large se, is shrunk
to the pool and is not picked, while the same kind of lineup that happened not to hit is
kept. The selection therefore prefers lineups that were unlucky in-sample, in-sample means
understate them, and the odd half shows their true level. The same mechanism acts on the
live choice (it prefers lineups that did not hit in the sim) and on greedy / portfolio.
Fix for phase 2: an se model that does not use the row's own realized variance (pooled by
structure, or the within-world distribution from the scenario tables), not the row's draws.

## Phase 2: scenarios with the within-world distribution (fixes Jensen)

A scenario becomes a `(world, draw)` pair rather than a world.

- Keep the conditioned world model (game latents, one draw per world) and its closed-form
  field tables (`conditioned.contest_tables`), so the field's noise stays integrated out.
- Per world, draw K residual score vectors for the candidate pool **jointly** from the
  residual covariance given the world, and read each lineup's payout off the contest's table
  at its realized score. Our entries that share players then win and lose together inside a
  world, which is what the log needs. K of 8 on 2000 worlds gives 16,000 scenarios with no
  per-lineup estimation noise to shrink.
- Store the scenario matrix as a per-contest artifact (`N x (W*K)` float32; ~10k lineups x
  16k scenarios is ~650 MB per contest, so store per contest and stream into the chooser, or
  reduce K / W to fit).
- The phase 1 chooser does not change: it reads scenarios instead of worlds.

Blocker: the conditioned method left the live path on 2026-10-19 because its count model
breaks when the field's lineups are deterministic given the world (the shared-lineup
dependence finding). Phase 2 depends on fixing or sidestepping that. Options to evaluate:
condition the field's count model on the same residual draw (the field also gets K draws per
world), or keep the MC field ranking and only replace our own entries' rows with joint draws.

## Phase 3: the book's shape as a decision

- Let the chooser leave a slot empty when its expected log contribution is negative, and
  report which contests it would drop.
- Choose entry counts per contest against DK's max-entry limits and the bankroll.
- Own-entry interaction: our entries take ladder spots from each other and duplicate each
  other within a contest; evaluate the book's payouts jointly, not lineup by lineup.
- Swap passes after the greedy (1-for-1 exchanges that raise the utility).

## Verification, before anything is the default

Per the standing rule: no live change without the backtest's field checks passing.

1. `compare_choosers.py` on the archived slates (2026-10-11 through 2026-10-25) with
   `greedy`, `portfolio:1.2`, `slate:bankroll_dollars=1000`; settle each book against the
   real fields.
2. The honest cross-fit gap per method (selection bias) and the luck test (`roi_draw`).
3. Repeat noise bands from the replay harness (`--repeats`), as on the models page.
4. The 20-repeat run used for the 10-19 sd fix is the reference: a method that is not
   profitable in more repeats than the current default does not ship.

## Files

- `src/ivan/sports/nfl/choose.py`, `src/ivan/core/portfolio.py`: the slate method and the
  dollar-bankroll utility (shared with MMA).
- `src/ivan/sports/nfl/lineup_export.py`: `choose_entries` gets a slate pre-pass; the
  per-contest loop consumes its picks.
- `src/ivan/web/pipeline_nfl.py`, `src/ivan/web/app.py`, `templates/choose.html`: method
  option, bankroll field, slate book panel.
- `scripts/nfl/compare_choosers.py`: the `slate:` config.
- Phase 2: `src/ivan/sports/nfl/conditioned.py` (joint residual draws), `simulate_contest.py`
  (scenario artifact).
