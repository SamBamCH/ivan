# The NBA backtest, redesigned: one date's data, one lock snapshot, the pre-lock models re-run (2026-11-05)

## What went wrong on 2026-11-04

The desk's backtest said slate 156078 lost $28.86 (-19 %); DraftKings paid $194.82 net (+126 %). Three of our
fourteen contests had a final standings export (the three largest public contests, scraped the next morning). The
other eleven had only the late swap's fetch, taken six minutes after lock, in which DraftKings prints LOCKED for
every player whose game has not tipped, ours included. The page rescored those one-to-four-player lineups from the
box scores against an equally partial field, ranked and paid them, and called the result final. Two code paths were
to blame, both fixed in the commit that adds this note:

- `slate_backtest` scored any export when box scores existed; it now scores only a final export, from DraftKings'
  own points, and lists the rest as unscored with their fees.
- the scrape read "our contests" from the pre-split shared entries path, gone since the one-directory-per-slate
  split (2026-11-03), so only the public contests were ever scraped final. It now reads the slate's own entries file.

On the same fourteen final exports the box-score rescore gave +$240.67 against DraftKings' +$194.82: a rank moved
across a payout rung by a point or two (a name matched loosely, a stat correction). DraftKings' points are the truth
for money; the box scores are the truth for the projection. They are not interchangeable.

## The rule

**A result is DraftKings' final export and the payout table. A backtest is the pre-lock models re-run on the pre-lock
data, compared to the actuals. Everything the page shows is computed at load from one date's directory.**

Three consequences:

1. Nothing is derived in the background and cached for the page to find. The nightly fetches data (box scores,
   stints, the scrape when it can run unattended); it does not pre-compute results. The derived contest folders,
   the nightly scorecards and the field backtest's cached model fields become page-load computations over the
   date's data, cached (if at all) by the raw file they came from.
2. The page for a date reads only that date. Props especially: a slate's page reads the odds rows posted for that
   slate's games (`projection.market.fresh_rows`), never the season's table.
3. "What we would have had before lock" is one snapshot, not a reconstruction. The models re-run on it, not on the
   slate database (which is post-lock state: slate 156078 was refreshed at 18:21, lock 18:00) and not on the day's
   edits directory (overwritten by the next slate of the day).

## The data, split by date

Everything a date's page needs lives under the slate's directory (`data/nba/slates/<dg>/`, the slate's `day` in
`slate.json` says which date), in parts that are written once and never rewritten. **The unit of truth is a game's
tip, not the slate's lock** (decided 2026-11-05): a 6:00 slate's 8:00 game is judged on what was known at 8:00, so
every tip gets its record, and a game's backtest reads the newest tip at or before its own (`tips.tip_for`).

```
data/nba/slates/<dg>/
  tips/<YYYYMMDD-HHMM>/       the record at each game's tip (ivan.sports.nba.backtest.tips; BUILT 2026-11-05, archive_at_lock writes it)
    meta.json                 the tip, when written, the teams tipping, the counts, where each part came from
    players.json              the pool as the desk showed it: salary, status, starter, minutes and points projected, p_out, ownership,
                              pub_fpts, the grid row (rotation_cells)
    odds.parquet              the props for the slate's players as stored at the tip (the odds table is replaced per player on every
                              refresh, so this is the only record; a replay cuts it to the game's tip with projection.market.fresh_rows)
    game_lines.json           spreads and totals for the slate's games
    news.json                 the slate's news seen by then (nba_slate_news x nba_news)
    edits/                    a copy of the day's edits (minutes_overrides.csv, rotations/, scenarios/) as they stood
  results/ledger.json         the money (ivan.sports.nba.backtest.results; BUILT 2026-11-05): per contest from the final exports, cached, complete
                              once every contest has one
  input/, output/, slate.sqlite, projection/   the live slate as today (the desk's working state; post-lock after the last refresh)
```

Still read from where they are, because they are dated or stamped already: the box scores and stints (`game_date`
before the slate's day), the entries file and the chooser's and late swap's upload files (`input/`, `output/`, stamped
before the tip), the contests and payouts (the slate's database, written before lock), the final standings exports
(raw `dk/standings/<id>/`, the results' input). The season-wide tables that are not dated (the odds table's "latest row
per player", `nba_market_rate`) are never read by a backtest.

The archive capture tables (`archived_slate` and its blobs, `nba_projection_capture` at_tip) hold the same players,
props and lines per tip since 2026-10-23 and are backfilled into `tips/` with `scripts/nba/tips.py backfill`; the timer
still writes them for now, and the per-capture copy of the odds table stops once every reader uses `tips/`.

## What is frozen at a tip, and what is not

| Frozen at the tip (in `tips/`) | Dated already (read by date) | Never pre-tip: the actuals |
|---|---|---|
| the pool and salaries, statuses, starters | box scores before the slate's day | the box score of the game |
| our minutes and points projection, p_out, ownership, the grid | stints before the slate's day | DK's %drafted, the standings |
| the props (every market, every book, with their update times) | our entries and upload files (stamped) | the payout received |
| the game lines | contests and payouts (written before lock) | |
| the news seen so far | | |
| the day's edits (pins, grids, scenarios) | | |

## Feasibility and cost (2026-11-05)

- **The money per page**: built. The first season pass parses every final export once (16 s for seven slates); after
  that a complete slate is one small JSON read and the season strip costs a millisecond. An incomplete slate is
  re-checked every ten minutes against the raw index (46 ms). The nightly warms the ledgers, so a page never pays the
  parse unless a scrape just landed. Past results do not change, so nothing is ever recomputed once complete.
- **The tip record**: built; a tip costs one parquet and a few JSON files (the props are the bulk, a few hundred KB).
- **The model re-run per page** (phase 4 below): the projection for a slate is seconds; the field and the sim are
  minutes, so they run once per (slate, commit) and are cached under the slate's directory, never recomputed on a
  page load. A date page sums its slates' cached stage results; the season page sums the days. The page shows "not
  run for this commit" rather than running the sim on load.

## The page: what runs at load

For a date, for a slate:

| Stage | Pre-lock input (lock/) | Model re-run | Actual | Shown |
|---|---|---|---|---|
| projection | players, odds, game_lines, edits, box scores before the day | `project_team` per team | box scores of the day | minutes and points, projected vs scored, per team; the grid over the game as played |
| ownership / field | the projection above, players | the two-group field, the softmax model | the final export's side table (%drafted) and lineups | ownership MAE, chalk concentration, duplication |
| sim | the field above, payouts | the contest sim | the final export's ranks | claimed ROI per contest vs the real distribution of our lineups' ranks |
| chooser | chosen.csv, the sim above | (the claim is recorded; the chooser is not re-run) | the final export's payouts | claimed vs realised ROI per entry and per contest |

Each stage runs with the code as it is now, so the page is also the per-component scoreboard over commits (the
direction set 2026-10-14: no knobs UI, a fixed benchmark replayed per commit): change the model, reload the page,
read the column. The projection re-run
is seconds; the field and sim are the heavy part and run once per slate per code version, cached by (slate, git
commit, parameters) under the slate's directory, never by the nightly.

What the page shows when something is missing is a sentence, not a number: "no final export for 4 contests
($47 in fees): scrape", "no lock snapshot (slate before 2026-11-06): the archive capture stands in".

## The scrape

Per played slate: every contest in `lock/entries.csv` plus the three largest public classic contests by size
whether we played them or not; a contest counts as scraped only when its export is final. It needs the logged-in
Chrome profile, so it stays a button and a script run after the last game; `status` says what is missing. The late
swap's fetch keeps writing to raw `dk/standings/` for the swap's own use and is never a result.

## Order of work

1. Done 2026-11-05: final-export-only results on the page; the scrape reads the slate's entries file; `--date` finds
   held slates by their day.
2. Done 2026-11-05: `tips/` written by `archive_at_lock` at every tip (players, props, lines, news, edits), backfilled
   from the captures; `results/ledger.json` per slate, the money on the home tile, the picker (season and per day) and
   the slate's backtest view, this machine only, identified by our DraftKings user name.
3. The projection re-run on a tip's record (replace `validation.replay_slate`, which drops props and the grid): per
   game, the newest tip at or before its own, on the slate's backtest page beside the "as shown" column, with the
   players' minutes and points MAE per team and per date. Seconds per slate; computed at page load and cached by
   (slate, commit).
4. The field and the sim on the tip record: the field backtest per slate on the page (ownership MAE, chalk, duplication
   against the final export), then the sim's claimed ROI against the real ranks of our lineups. Minutes per slate: run
   by a button or the nightly for the current commit, cached under the slate's directory; the page reads the cache.
5. Retire: the nightly scorecard files (the page computes them), `derive_pending` in the nightly (derive from the final
   export when the field backtest needs it), the per-capture odds blob once every reader uses `tips/`.
