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
`slate.json` says which date), in three parts that are written once and never rewritten:

```
data/nba/slates/<dg>/
  lock/                       the snapshot at lock (written by archive_at_lock at the first tip; read only after)
    players.parquet           the pool: id, name, team, opponent, position, salary, status, starter, dk_avg_fpts, gametime
    odds.parquet              the props posted for THIS slate's games, as of lock (fresh_rows), with posted-at times
    game_lines.parquet        spreads and totals for this slate's games, as of lock
    edits/                    the grids, overrides and scenarios as they stood (a copy, not the live day directory)
    entries.csv               DraftKings' entries file as uploaded (our contests, Entry IDs)
    chosen.csv                the lineup each entry held at lock (the last chooser upload before lock + swaps before lock)
    projection.parquet        what the desk projected at lock (the at_tip capture), for the "as shown" column
  results/                    DraftKings' word, written when scraped (final exports only; a mid-slate fetch stays raw)
    <contest id>.csv.gz       the final standings export
    payouts.json              the payout tables of the slate's contests
  input/, output/, slate.sqlite, projection/   the live slate as today (the desk's working state; post-lock)
```

History that is naturally dated stays where it is and is sliced by date: box scores (`nba_stats.game_date`), stints,
the news table (`first_seen`). The season-wide tables that are not dated (the odds table's "latest row per player",
`nba_market_rate`) are never read by a backtest.

The archive capture tables (`archived_slate` and its blobs) already hold players, the whole odds table and the game
lines per capture; `lock/` is that capture written to the slate's directory with the odds cut to the slate's games.
Once `lock/` exists for a slate the captures are a backup, and the per-capture copy of the 100,000-row odds table
stops (it is why a capture weighs what it does).

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

1. Done in this commit: final-export-only results on the page; the scrape reads the slate's entries file; `--date`
   finds held slates by their day.
2. `lock/`: `archive_at_lock` writes the snapshot to the slate's directory (players, odds for the slate's games,
   game lines, edits copy, entries, chosen lineups, projection). Backfill from the archive captures for the slates
   that have them (2026-10-23 on; odds cut by game at backfill).
3. `results/`: the scrape writes the final export and the payouts there; the page reads `results/` first, raw
   `dk/standings/` as a fallback for older slates. The derived contest folders are rebuilt from `results/` on demand
   (`standings.derive_contest` becomes a function of the page, not of the nightly).
4. The projection re-run on `lock/` (replace `validation.replay_slate`, which drops props and the grid), on the page
   beside the "as shown" column. Then the field (the field backtest per slate on the page), then the sim's claim
   against the real ranks.
5. Retire: the nightly scorecard files (the page computes them), `derive_pending` in the nightly, the per-capture
   odds blob once `lock/` is written at every lock.
