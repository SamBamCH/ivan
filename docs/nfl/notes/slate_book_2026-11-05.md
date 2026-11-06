# The slate as one book, for NFL and NBA

*Written 2026-11-05. Status: built on branch `slate-book` as the selectable method `slate` in both sports and the
desks' "Slate book" button; unbacktested in both. Supersedes the "what exists" and phase 1 sections of
[slate_chooser_plan.md](slate_chooser_plan.md); phases 2 and 3 of that plan stay open except where noted.*

## The ask

Think about the whole portfolio of lineups played across a slate, not contest by contest: get a diverse book
over the slate, and put each lineup in the contest where it fits. If single-entry contests want steadier,
cash-more-win-less lineups, playing them there means fewer chalky lineups are needed elsewhere. The only
inputs should be the entries file and the bankroll in dollars.

## Why the log book already is that, and what was missing

The 2026-10-26 slate method (NFL only) optimised mean over worlds of log(1 + book_w / bankroll) over
(contest, lineup) pairs. That objective does the portfolio work on its own:

- **Contest fit.** A lineup's row in the book is fee × ROI per world *in that contest*, so the same lineup
  is a different vector in a double-up and in a top-heavy GPP. A steady casher adds a flat positive vector
  in the double-up and a small one in the GPP; a tail lineup the other way round. Greedy on the marginal
  utility gain puts each where its gain is largest.
- **Diversity.** log is concave: once the book already wins in a set of worlds, another lineup that wins
  there adds little; a lineup that wins in uncovered worlds adds more. Spread across game scripts and
  across contests falls out, with no diversity knob.
- **One bankroll.** The bankroll is the only risk setting; everything else (which contests, how many
  entries, the fees) is read off the entries file.

Missing: the NBA had no shared world axis (each contest's sim drew its own worlds, so world w was a
different game per contest and the sum across contests was meaningless); the greedy had no second look
(an early pick can block a better one once the book is full, and a lineup can land in the wrong contest);
the engine lived in the NFL module.

## What was built (2026-11-05)

- `ivan.core.slate_book`: the sport-agnostic engine. `ContestCandidates` (one contest's lineups with their
  per-world ROI, fpts traces, cash rate, counts), `choose_book` (shrink per contest, stack as net dollars per
  world, `portfolio.pick_book` greedy, `portfolio.improve_book` exchange pass, the slate cross-fit, the
  per-contest split and `placements`), `describe_book` (the log lines). The NFL and NBA `choose.choose_slate`
  load their own rows, check the shared draw, call it.
- `portfolio.improve_book`: after the greedy, each pick in turn is scored out against the shortlist of its
  contest's candidates (the 300 with the largest marginal gain on the finished book, plus the picks) and
  replaced when a swap raises the utility; up to three passes or until none swaps. Costs about one greedy
  step per pick. The trace keeps the greedy walk and marks swapped slots (`swapped_from`, `swap_gain`); the
  book carries `greedy` (the book before) and `improve` (passes, swaps, log growth before and after).
- NBA sim: one draw per run (`simulate_contest.slate_draw`, seed `sim/slate`), every contest scores on it;
  reports say `contest.draw = "slate"`; `choose.shared_worlds` refuses anything else or two seeds. A
  contest simulated alone (`draw = "contest"`) cannot join a book.
- NBA chooser: method `slate`, `bankroll_dollars` knob (config `choose.bankroll_dollars`, the contests
  section's `$`, `scripts/nba/contest_weights.py`), `lineup_export.choose_entries` pre-pass like the NFL's,
  the slate book written under each of its contests in the choose json (the json's keys stay contest ids,
  which the desk's outputs and the slate backtest iterate).
- The one-input entry on both desks: **Slate book** beside Choose, one knob (bankroll $), running
  `scripts/<sport>/choose.py --method slate --bankroll-dollars B`. The per-contest knobs are not used by it.
- The NFL Choose page: exchange-pass and distinct/reused tiles, per-contest cash, P(loss) and "also
  elsewhere" columns, the lineups played in more than one contest.
- Tests: `tests/core/test_slate_book.py` (slots filled, no pair twice, the exchange pass never lowers the
  utility and undoes a forced bad pick, steady lineups go to the flat contest and tails to the top-heavy one
  on a two-contest pool, the world-count check, an empty contest), `tests/nba/test_nba_slate_book.py` (shared
  draw, the world-axis check, the run end to end, the knobs), `tests/nfl/test_slate_choose.py`.

## What the first NFL slate run (2026-10-25) already taught, still true

The shrinkage flaw stands: `shrink_rows` uses each row's own sample variance as its standard error, so a
lineup that hit first place in the sample is shrunk hardest and the selection prefers lineups that were
unlucky in-sample (the out-of-sample book beat the in-sample one by eight standard errors). The exchange
pass does not change this; it optimises the same shrunk rows. Phase 2 of the plan (an se model not built
from the row's own draws) is still the fix.

## Before either sport's default moves

Per the standing rule, no live change without the backtest's field checks passing.

- NFL: `scripts/nfl/compare_choosers.py --config greedy --config "portfolio:bankroll=1.2" --config
  "slate:bankroll_dollars=1000"` on the archived slates 2026-10-11 through 2026-10-25, settled against the
  real fields; the honest cross-fit gap per method; the luck test; repeat noise bands; the 20-repeat run of
  the 10-19 sd fix as the reference (a method that is profitable in fewer repeats than greedy does not
  ship).
- NBA: a sim over a held slate's entries file at its tip record, `choose.py --method slate`, the result
  through `slate_backtest` against the final exports; greedy on the same sim as the control. The slate's
  cross-fit (`book.cross_fit`) and the per-contest honest ROI are in the choose json for the scorecard.
- Both: watch `book.improve.swaps` and the log growth delta; if the exchange pass rarely swaps, the greedy
  was already at a local optimum and the pass can be made cheaper or dropped.

## Still open from the plan

- Phase 2: an se model independent of the row's own draws; the within-world scenario draw (the Jensen fix
  for conditioned rows). NFL only, since NBA's rows are realized MC draws.
- Phase 3: entry counts and contest choice as decisions (today the entries file fixes both); own-entry
  interaction inside a contest (our entries take ladder spots from each other and duplicate each other;
  the book values them independently); leaving a slot empty when its expected log contribution is negative.
- The greedy and portfolio per-contest methods still share no information across contests; the slate
  method is the only one that does.
