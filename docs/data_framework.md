# The data framework: store the raw inputs and outcomes of every slate

Design note, 2026-10-12. The aim: every slate we play (and, cheaply, every
slate we could have played) leaves behind the raw data that went into the
models before lock and the raw outcomes after it, kept forever, so any
model can be re-fitted or re-designed on the whole history and tested as
if it had been live.

## The principle

**Raw first, derived second.** Store every payload exactly as received,
with the time it was received, keyed by slate. Everything the models read
is derived from those payloads by code we can change and re-run. Nothing
in training may depend on a table that cannot be rebuilt from raw.

Today the pipeline keeps *parsed* rows (the ten DraftKings fields we use,
the odds rows after our parser, the showdown lineups after our parser) and
throws the payloads away. That is the root of "having to look things up
later": the parse is frozen at the moment of capture, and anything the
parser dropped is gone.

## What exists (audit of the live database, 2026-10-12)

| data | stored today | gap |
|---|---|---|
| DraftKings draftables (players, salaries, slots, status) | 10 fields per player in `nfl_player`; archived per ingest since 2025-10-09 (18 captures) | the payload itself (AvgPointsPerGame, news status, competition, swappability) is not kept; only ingest moments are captured, nothing at a fixed time before lock |
| Player props | parsed rows (player, market, book, side, point, price, last_update) for the current slate; history only inside captures (175k rows) | the API responses are not kept (event ids, book timestamps, markets we did not parse); no fixed pre-lock snapshot; line movement lost |
| Game lines | `game_odds`, 11k rows, 11 books, since 2025-09-21 | fine; same raw-payload caveat |
| Contests and payouts | 37,659 contests over 1,263 draft groups since 2025-08-24, payouts for 36,282 | fine (the lobby scrape runs regularly) |
| Our projections and ownership | `projection_snapshot`, 9k rows over 54 slate days; run artifacts since 2025-10-11 | the model version behind a live projection is not recorded |
| Box scores | nflverse 2022-2026, nightly | fine; a day or two of lag |
| Contest standings | showdown: 122 contests with per-player points and ownership, 113 with every lineup (822k); classic: 827 contests reduced to distribution bins | the standings export (every entry, user, lineup, points, rank, payout) is not kept; classic fields cannot be rebuilt at the player or lineup level; our own finishes never recorded |
| Settlement state | none | nothing says which slates are complete, so each backtest re-discovers the gaps |

## The layout

```
data/nfl/raw/                                    append-only, never edited; gzip json / csv
  dk/draftables/<draft_group>/<stamp>.json.gz    the full draftables payload, every fetch
  dk/lobby/<date>/<stamp>.json.gz                contests + payouts as listed
  dk/standings/<contest_id>.csv.gz               the standings export, as downloaded
  odds/<stamp>/events.json.gz                    the odds API: events, then one file per event
  odds/<stamp>/<event_id>.json.gz                  (player props, every book, every market asked for)
  odds/<stamp>/lines.json.gz                     game lines (h2h, spreads, totals)
  nflverse/<season>/week_<w>.parquet             box scores as fetched
```

One index table, `raw_artifact` (kind, key, stamp, path, bytes, sha256,
meta json), so the database knows what raw it has and any derived table can
be rebuilt by replaying the artifacts in order. Captures (`archived_slate`)
keep working as they do, but point at the raw artifacts instead of carrying
parsed rows; the replay restores a slate by re-parsing raw with today's
parser, which is exactly the point.

Derived tables stay where they are (players, odds, game lines, contest
players, lineups, field scores, entry results) but become rebuildable, and
the classic contests get the same player- and lineup-level tables the
showdowns have, built from the raw standings.

## The jobs

**Pre-lock capture.** Two moments per slate: at every ingest (exists) and at
a fixed time before lock (lock minus 30 minutes), scheduled from the
draftables' start times. The fixed one is what the replay should use by
default: same clock for every slate, line movement between ingest and lock
preserved as two artifacts.

**Post-lock settle.** Nightly, and on demand from the Backtest page. For
every slate whose lock passed more than six hours ago and that is not
settled:

1. standings for every contest we entered (from the chooser's outputs), plus
   the largest public contests of the draft group even when we did not play
   them, because the field models (exposures, dupes, score distributions)
   learn from public fields, not from ours;
2. box scores for the week once nflverse publishes them;
3. the derived tables, then the slate's status: settled, or waiting on box
   scores, or needs a DraftKings login (the one step that cannot run
   unattended for long; the job flags it instead of retrying).

A `slate_status` row per slate (captured at, settled at, what is missing)
is what the Backtest page's data step reads, instead of recomputing it.

**Model refresh.** The as-of model build runs weekly with today as the
cutoff, so a version always precedes the next slate; the replay never has to
fall back to live models for recent slates.

## What this buys

With raw pre-lock inputs and raw post-lock outcomes per slate, every model
in the pipeline is a function from the first to a prediction of the second,
and the replay can run any version of that function over the whole history:
a different way of turning props into a distribution, a different
ownership model, a different dupe model, each scored on the same slates
against the same actuals, with the sample growing by every slate played.
The overfitting worry resolves itself the same way: the test is always
"would this have worked on slates the fit never saw", on as many slates as
we have kept.

## What is kept and what is let go

The raw inputs and outcomes above, the derived tables and the model versions are kept for good: they
are what every future replay runs on. A backtest run's output is a function of them and of the code at
a commit, so a run is kept only as its record (manifests with the commit, the small csv, the report)
and its heavy files (repeats, database copies, validation csvs) are let go of 30 days after the run,
unless pinned. The per-slate run index reads only the record, so every build of a slate stays visible
after the sweep. Rules and the sweep: `scripts/nfl/retention.py`
([nfl/backtest-and-validation.md](nfl/backtest-and-validation.md)).

## Two numbers per backtest: the result and the confidence

A backtest answers two questions, and they must be shown together:

1. **The result.** Realized ROI, profit, cash rate, after tax, over the
   settled contests, with the spread across repeats and the sim's own claim
   beside it.
2. **The confidence.** How much the claim deserves to be believed. A sim can
   say +10,000,000% and mean nothing; the number only carries weight when
   every stage that produced it has been shown to track reality.

Confidence is not a new model; it is read off the validation at each stage,
on the same settled slates, each stage scored on what it alone predicts:

| stage | what it predicts | measured against | reliability reads as |
|---|---|---|---|
| projections | each player's score distribution | box scores / contest points | MAE against the salary line, bias, 10-90 coverage, sd ratio |
| ownership | each player's ownership | the field's actual ownership | MAE, rank correlation, chalk bias |
| field | the field's scores and exposures | standings | cash line and winning score inside the band, exposure error after raking |
| dupes | copies of each lineup | standings | top-lineup share ratio, our lineups' predicted vs actual copies |
| simulator | each lineup's ROI and cash % | placement in the real field | sim-vs-actual ROI correlation, cash % calibration, seed-to-seed reliability of the ranking |
| chooser | the entries it picks beat the pool | the pool placed in the real field | picks vs random draws from the pool, vs top-K by sim ROI |

The confidence shown next to a result is the weakest stage, named, with
the full table one click away, and the sim's ROI claim is printed with its
history of overstatement (sim ROI against realized ROI on every settled
window so far). A result whose chain has a failing stage is reported as
"not yet meaningful" rather than as a number, however large the number.

This is also the loop for improving the models: a change is accepted when
it raises a stage's reliability on the replayed history without lowering
another's, and the result column is read only once the chain holds.

## Status

- 2026-10-12: step 1 built. `ivan.core.raw` (store / load / find / latest / sync_index / summary; `raw_artifact`
  table) and the sinks in the fetchers: draftables, lobby, payouts, standings exports (DraftKings) and events, props,
  lines (the odds API), each keyed by sport and kept gzip under `data/<sport>/raw/<kind>/<key>/<stamp>`. The NFL ingest,
  contest scrape, lobby and payout scripts pass the sport and index the artifacts at the end.
- 2026-10-12: step 2 built. `ivan.sports.nfl.settle` + `scripts/nfl/settle.py` (assess -> fetch box scores -> scrape
  the played and largest public contests -> derive lineups.parquet / scores.npy from raw standings -> status rows in
  `slate_status`, ivan.core.slates); the overnight job runs it with --days 3 (scraping when IVAN_DK_USERNAME and a
  display exist, otherwise --no-scrape); the Backtest data step shows the status and has a Settle button.
- 2026-10-13: classic contests get player rows (step 4, first half): `settle.derive_classic_players` reads the export's player
  table (Player, Roster Position, %Drafted, FPTS), matches names to the slate's capture (id, team, salary, every roster slot
  incl. FLEX) and the projection snapshot of that lock, and writes them to the contest player table, so classic slates settle
  and validate at the player level and can train the ownership models. Still pending: lock-30 capture (3), our-finish
  recording is done by the scrape with --username, rebuild-from-raw tooling (5).

## Build order

1. Raw sinks: wrap the three fetchers (draftables, odds, standings download)
   so they write the raw artifact and index it before anything parses it.
   Small change, immediate value; nothing downstream moves.
2. Settle job and `slate_status`: the nightly loop above, with the Backtest
   page's data step reading the status instead of re-deriving it.
3. The fixed lock-minus-30 capture, scheduled from the draftables.
4. Classic contest player and lineup tables from raw standings; our own
   finishes by username.
5. Rebuild tooling: derive every table from raw, so a parser change is a
   re-run, not a migration.
