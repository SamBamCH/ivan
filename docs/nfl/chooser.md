# NFL: the chooser

`src/ivan/sports/nfl/choose.py` with the shared numerics in `ivan.core.chooser`, `ivan.core.portfolio`
and `ivan.core.slate_book`; `lineup_export.py` writes the files; `scripts/nfl/choose.py` is the command
line and the web UI's step 5 (**Choose**, **Slate book**) the forms. Defaults (`lineup_export.DEFAULT_WEIGHTS`, also
`web/nfl/pipeline.py`): `cash_pct` .5, `avg_roi_win` .5, `diversity_lambda` .1, `method` slate,
`bankroll` 3.0, `bankroll_dollars` blank. Weights are per contest (prompt, flags, `--weights-json`, or the
web page's remembered weights file); the slate method takes one bankroll for the whole slate.

## The bankroll

With no `bankroll_dollars` given, the bankroll is read from our results: `config/bankroll.json`'s `start` plus what our entries made (won − fees on the scored contests of every sport's results ledger, on the days from `since` on; `ivan.core.bankroll`, `scripts/bankroll.py` prints it). One DraftKings balance plays every sport, so every sport's results
count, and the figure moves with them: a winning week raises the next slate's bankroll, a losing one lowers it and the
book turns more cautious. A contest not yet scored moves nothing. On a consumer desk (the server), which holds no results and shows no money, the bankroll is `start` alone. After a deposit or a withdrawal set `start` to the
balance that day and `since` to that day. A figure that is given (`--bankroll-dollars`, a contest's knob, the desk's
Slate book knob) is used as it is; the choose log says which (`bankroll $X = $start start + $net from N scored
contest(s) (NBA +a, NFL −b)`, or `(given)`), and the slate book in the choose json carries it as `bankroll_from`. A
sport whose results cannot be read is named in that line and not counted. Results that have used up the start make the
run refuse until `config/bankroll.json` says what is being played from. A replay or backtest never reads the results
(`replay_adapter.REPLAY_WEIGHTS`: greedy, and $1,000 for a menu that asks for the slate method), so a past slate's book
does not depend on what came after it.

## First, shrinkage

Every candidate's per-world ROI row is pulled toward the pool by its own standard error
(`chooser.shrink_rows`, DerSimonian-Laird empirical Bayes), because sim ROI is mostly first-place
noise: the 2026-10-19 lab showed the chooser's claimed ROI was about three quarters selection on
sampling noise. Known flaw (2026-10-26): the se comes from the row's own draws, so a lineup that hit
first place in the sample is shrunk away and the selection prefers lineups that were unlucky
in-sample; an se model not built from the row's own draws is the fix (slate plan, phase 2).

## The three methods

| Method | Scores | Picks | When |
|---|---|---|---|
| **greedy** | each lineup alone: `cash_pct` × cash rate + `avg_roi_win` × mean ROI | one at a time by quality plus `diversity_lambda` × distance of its fpts trace from the picks so far; with a pool over 5000 it first drops lineups with mean ROI < 0 (which biases its shrinkage pool upward) | fast; the one the backtests were run with, and what a replay uses when its menu names no method |
| **portfolio** | the book: Σ_w log(1 + net_w / B), B = `bankroll` × the entries' fees | the lineup that most raises the book's log return per step; lineups that win where the book already wins add little, so picks spread across game scripts | built after the 2026-10-15 loss (the chooser stacked tail worlds); lost least on that slate |
| **slate** (default; unbacktested) | the whole slate as one book in **dollars**: mean over the shared worlds of log(1 + book_w / `bankroll_dollars`), the candidates (contest, lineup) pairs carrying fee × ROI per world, with the entries file's slot count per contest | greedy on the slate objective (`portfolio.pick_book`), each (contest, lineup) once, then an **exchange pass** (`portfolio.improve_book`: one-for-one swaps that raise the utility, so an early pick gives way to a lineup it blocked and a lineup moves to the contest where it does more) | requires every contest's sim report to carry `draw == slate` (one shared draw) and equal world counts, else it aborts; with bankroll ≫ fees the log is near linear and picks sit near the shrunk mean-ROI picks |

The slate method is the portfolio across the whole slate: a lineup's net vector differs per contest (its payout table),
so a steady casher is worth more in a flat-payout or single-entry contest and a tail lineup in a top-heavy one, and the
log's concavity makes a lineup that wins where the book already wins add little. The same lineup may be played in
several contests, never twice in one. Its only inputs are the entries file and the bankroll:

```bash
venv/bin/python scripts/nfl/choose.py --no-prompt [--entries f.csv]                       # the default: slate, the bankroll from our results
venv/bin/python scripts/nfl/choose.py --method slate --bankroll-dollars 1000 [--entries f.csv]
venv/bin/python scripts/bankroll.py [--start 1000] [--since 2026-11-01] [--sports nba,nfl]   # the bankroll a run would read
```

(the desk's **Slate book** button runs `--method slate`, with `--bankroll-dollars` only when its knob holds a figure). The run writes the slate book into the choose json under
each contest (`viz.slate`): the dollar distribution (mean, percentiles, P(loss), P(lose half), log growth, share of the
mean from the top 1% of worlds), the greedy book before the exchange pass and the swaps made, the quality-only book,
the slate cross-fit, the per-contest split (entries, fees, mean net, mean ROI, cash rate, P(loss), how many of its
lineups the book also plays elsewhere, honest ROI) and `placements` (every lineup and the contests it went to). The
engine is shared with NBA (`ivan.core.slate_book`). Plan: [notes/slate_chooser_plan.md](notes/slate_chooser_plan.md);
the shared build and what is still open: [notes/slate_book_2026-11-05.md](notes/slate_book_2026-11-05.md).

## Honest ROI: cross-fit and the luck test

`chooser.cross_fit_book` selects on the even worlds and scores on the odd, so the claimed ROI of the
picks is out-of-sample (`validate.honest_roi`); the choose json carries it per contest. The luck test
(`tests/core/test_luck.py`, `simroi.npz` with `roi_draw`) checks that the realized-draw book is
consistent with its claim. Quirk kept: for greedy and portfolio the in-sample half is scored on shrunk
rows and the out-of-sample half on raw rows, so their "selection bias" mixes shrinkage with selection;
the slate cross-fit (`slate_book._cross_fit`, greedy and exchange pass on the even worlds) scores both halves raw.

## Comparing choosers on a slate

```bash
venv/bin/python scripts/nfl/compare_choosers.py --config greedy --config "portfolio:bankroll=1.2" --config "slate:bankroll_dollars=1000" [--entries f.csv] [--artifacts dir] [--in-field]
```

runs several settings on a slate's existing simulation and settles each book against the real fields
(after the scrape). It is the gate before any method becomes the default, together with the backtest's
field checks and the luck test.

## Reading the Choose page

Per contest: every candidate on the cash % vs ROI plane with the weight direction drawn in; the fpts-trace
map (lineups that win and lose together sit close); the greedy walk (quality vs diversity bonus per
pick) or the book summary; the picks in order; and when the sim carries worlds, what to root for
(which team points make the book win).
