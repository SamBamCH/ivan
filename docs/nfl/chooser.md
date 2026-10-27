# NFL: the chooser

`src/ivan/sports/nfl/choose.py` with the shared numerics in `ivan.core.chooser` and
`ivan.core.portfolio`; `lineup_export.py` writes the files; `scripts/nfl/choose.py` is the command
line and the web UI's step 5 the form. Defaults (`lineup_export.DEFAULT_WEIGHTS`, also
`web/pipeline_nfl.py`): `cash_pct` .5, `avg_roi_win` .5, `diversity_lambda` .1, `method` greedy,
`bankroll` 3.0, `bankroll_dollars` 1000. Weights are per contest (prompt, flags, `--weights-json`, or the
web page's remembered weights file).

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
| **greedy** (default) | each lineup alone: `cash_pct` × cash rate + `avg_roi_win` × mean ROI | one at a time by quality plus `diversity_lambda` × distance of its fpts trace from the picks so far; with a pool over 5000 it first drops lineups with mean ROI < 0 (which biases its shrinkage pool upward) | the fast default; the one the backtests were run with |
| **portfolio** | the book: Σ_w log(1 + net_w / B), B = `bankroll` × the entries' fees | the lineup that most raises the book's log return per step; lineups that win where the book already wins add little, so picks spread across game scripts | built after the 2026-10-15 loss (the chooser stacked tail worlds); lost least on that slate |
| **slate** (2026-10-26, unbacktested) | the whole slate as one book in **dollars**: Σ_s log(1 + (book_s + net_s) / `bankroll_dollars`) over the shared worlds, (contest, lineup) pairs with per-contest slot counts | greedy on the slate objective (`portfolio.pick_book`), each (contest, lineup) once | requires every contest's sim report to carry `draw == slate` (one shared draw) and equal world counts, else it aborts; with bankroll ≫ fees the log is near linear and picks sit near the shrunk mean-ROI picks |

The slate run also writes a dollar book summary (mean, percentiles, P(loss), P(lose half), log growth,
share of the mean from the top 1% of worlds) and per-contest cross-fits; the Choose page shows "the
slate as one book" tiles. Plan and findings: [notes/slate_chooser_plan.md](notes/slate_chooser_plan.md).

## Honest ROI: cross-fit and the luck test

`chooser.cross_fit_book` selects on the even worlds and scores on the odd, so the claimed ROI of the
picks is out-of-sample (`validate.honest_roi`); the choose json carries it per contest. The luck test
(`tests/core/test_luck.py`, `simroi.npz` with `roi_draw`) checks that the realized-draw book is
consistent with its claim. Quirk kept: for greedy and portfolio the in-sample half is scored on shrunk
rows and the out-of-sample half on raw rows, so their "selection bias" mixes shrinkage with selection;
the slate cross-fit scores both halves raw.

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
