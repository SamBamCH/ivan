# NBA 2026-27: skeleton parity with NFL, then judge the odds model

Plan written 2026-10-18. Season opens Thursday 2026-10-22 (per Sam; not verified).
Scope: DraftKings classic only (the NBA code has never done showdown). Nothing
here runs before NFL lock work is done; phase 1 is the only thing that must be
live by Thursday.

## Where NBA stands today (audit)

Code (`src/ivan/sports/nba`, 5.7k lines, last commit 24c62d7):
- `projection/` is current and tested last season: minutes-anchored model,
  market rates from props (`market.py`, 12 markets: pts/reb/ast/3s/stl/blk +
  alternates), Gaussian-copula sampler, `scripts/nba/project.py`, `news.py`,
  `backtest_projection.py`.
- `pre_lock/` (generate 25k lineups -> sim -> choose with prompted weights) and
  `post_lock/` (late swap) are the pre-cleanup shape NFL had in September: logic
  in modules, but `scripts/nba/ingest.py` and `overnight.py` are top-level
  scripts that run on import, `sim.py` prompts for weights, there is no
  `--weights-json`, no run artifacts, no `contest_sim_report`.
- No `tests/nba`. `config/nba.json` has six keys.

Data (`data/nba/nba_history.sqlite`):
| table | rows | span |
|---|---|---|
| nba_stats (box scores, nba_api) | 19,647 | 2025-10-02 .. 2026-06-13 (full season + playoffs) |
| historical_contest_information | 11,426 | 2025-10-28 .. 2026-01-22 |
| payouts | 79,362 | |
| projection_snapshot (Stokastic) | 5,943 | 2025-11-05 .. 2026-01-20 |
| odds (player props) | 10,256 | one capture, 2025-11-30 |
| classic_contest_metadata | 377 | distribution bins only, no lineups |
| starting_lineups | 70 | no fetch time column |

So: a full season of box scores and a Stokastic yardstick for Nov-Jan, but
one props capture, no lineup-level fields, no raw payloads, no captures, no
slate status. Everything the NFL backtest machinery needs is missing.

What NFL has that NBA lacks, in dependency order:
1. `ivan.sports.nfl.ingest` (fetch_slate / store_slate / project_slate) with
   `core.raw` payload storage and `core.archive` captures.
2. Contest + payout fetch as pipeline steps (scripts bound to NFL models, but
   the table names are identical across sports).
3. Stage modules the web and the replay call in-process: `field_generation`,
   `simulate_contest`, `lineup_export` (+ `contest_sim_report`).
4. Run artifacts (`core.artifacts`) + validation checks (`validate.py`).
5. Web: `app.py` is hard-wired to NFL (`init_db("nfl")`, `NFL_DATA_DIR`,
   `pipeline_nfl.STEPS`); hub/models/validate/plan views carry NFL paths.
6. Settle (`settle.py`, `core.slates`), the nightly timer, standings scrape
   with raw export + derived `lineups.parquet` / `scores.npy`.
7. `replay_adapter.py` implementing `core.replay.SportAdapter`.

## Status 2026-10-19 (opener is Tuesday 2026-10-20, not Thursday)

Phase 1 is live: `ivan.sports.nba.ingest` (fetch/store/project with raw
payloads, archive captures and `nba_projection_capture` rows), `scripts/nba/
capture.py` on a 15-minute user timer with props at 360/180/90/45/15 minutes
before each slate's lock, `scripts/nba/nightly.py` for box scores, and the
news desk (`scripts/nba/serve.py`, port 8001) for the last hour before lock.
Opening-night slates 154575 (main, 14:00 CDT) and 154827 (night) were captured
the evening before with one props fetch (42 priced players at 20:24 the night
before). Decision for week one: capture and shadow-run; no entries.

Still open from the phases below: contests/payouts as pipeline steps (phase 1,
free), sim/choose parity and tests (phase 2), the NFL web app's sport
descriptor (phase 3; the desk is a separate small app for now), settle and the
replay adapter (phase 4).

## Status 2026-10-21: phase 2 landed, scripts folder trimmed

Opening day found the props fetch covering only the latest-locking slate's
teams (fixed: one fetch per tick for every due slate's teams; odds refresh
per player). `scripts/nba/{overnight,sim,swap}.py`, `cli.py` and the
pre_lock `slate/lineup/contest/payouts` modules are gone; their logic is now
the stage modules the NFL has, with the same shapes:

- `nba.field_generation.generate_field` (workers seeded from IVAN_SEED,
  dedupe, `clear_lineups`) <- `scripts/nba/generate_lineups.py N JOBS [--fresh]`
- `nba.sample_field` (the pickled projection-bin model, or projection-weighted;
  config `field.sampler`) and `nba.simulate_contest.simulate_entries` (MC per
  contest, results replace, sim artifact npz/json under `output/sim/<stamp>`,
  `contest_sim_report` row + `diagnostics/<contest>/report.json` with
  conservation checks) <- `scripts/nba/sim.py [--entries] [-n] [-c]`
- `nba.choose.choose_lineup_ids` (greedy / portfolio, ROI rows shrunk by se,
  cross-fit honest expectation; numerics shared with NFL in `ivan.core.chooser`)
  and `nba.lineup_export.run_choose` (DK upload CSV, exposures, lineups,
  choose json, simroi npz) <- `scripts/nba/choose.py [--no-prompt] [--weights-json]`
- `nba.contests` (lobby -> contests, contest detail -> payouts, raw kept);
  the capture tick stores the day's contests and payouts itself.
- `tests/nba/` smoke tests; `config/nba.json` gained generate / field / sim / choose.

`post_lock/` (late swap) is untouched and still importable but has no script;
porting it is a later phase. Not yet built: standings download after settle
(phase 4), game lines, the web descriptor (phase 3).

## Validation: projected vs actual is the bread and butter (2026-10-21)

Every capture now records why a player's minutes are what they are
(`baseline_min`, `hist_team`, `hist_games`, `hist_min`, `start_ewm`,
`play_ewm`, `fpm`, `usage_mult`, `market_fpts`). `ivan.sports.nba.validation`:

- scorecard: the last capture before lock joined to `nba_stats` by player;
  each fpts miss split into minutes x projected rate + actual minutes x rate
  gap; classes by starter label, priced, salary tier and history class
  (same team / new team / no history / pinned); DK's season average and
  the books' implied points as yardsticks; rule-based findings; one CSV +
  JSON per slate in `data/nba/diagnostics/scorecard/`. The nightly job
  writes the pending ones after the box scores; `scripts/nba/validate.py
  scorecard|history` by hand.
- review: the current pool per team with each player's minutes provenance
  (history team and games, last five games, play rate, baseline before the
  allocation, market-implied minutes) and flags: new team, no history,
  dnp-prone, above recent max, deep bench, market minutes gap. The desk
  shows the same flags; `scripts/nba/validate.py review [--team]` prints them.

Opening night (two slates, three games, n=78 rotation players): minutes bias
+0.1, MAE 3.1; fpts bias +2.5 (almost all rate, not minutes), MAE 6.5 vs DK
average 6.7. The misses were role changes the history could not see: Caleb
Love (22 min from Portland history, DNP for Philadelphia), Drummond (6 -> 24
for New York), and stale play rates. The review flags all three patterns
before lock. Candidate model change once a week of scorecards agrees: a
salary-informed minutes prior for players whose history is on another team
or whose play rate is stale, weighted down as this season's games arrive.

## Phase 1: capture from opening night (must be live Thursday)

The data captured from the first game day is the asset; the UI can wait.
This also answers the timing question directly.

- `ivan.sports.nba.ingest` mirroring NFL: `fetch_slate(draft_group)` stores
  raw `dk/draftables`, `odds/events`, `odds/props/<event>`, and the Rotowire
  lineups page as raw html (`rotowire/lineups`, free, every fetch);
  `store_slate` writes `nba_player`, `odds`, `starting_lineups` (add a
  `fetched_at` column) and an archive capture; `project_slate` runs the
  existing `LiveSlate` engine and writes `projections.<stamp>.csv` via
  `core.artifacts` plus the snapshot. `scripts/nba/ingest.py` becomes a thin
  argparse wrapper with `--draft-group` (today it prompts).
- Contests and payouts for NBA: generalise `get_contest_ids.py` /
  `get_payouts.py` behind a `--sport` flag (same tables; store the lobby
  payload as raw `dk/lobby`).
- `scripts/nba/capture.py`: on a game day, list the day's draft groups and
  locks, then take captures at fixed offsets before each lock. Suggested:
  draftables + Rotowire every 15 min from 10:00 local (free); props at
  lock-6h, -3h, -90m, -45m, -15m, -2m. Odds API cost per props capture is
  roughly events x markets (10 games x 12 markets = ~120 credits), so six
  captures a day on a 30-day month is ~20k credits: check the plan's quota
  before choosing the offsets. Run it from a user systemd timer like
  `ivan-nightly` (every 15 min, exits immediately on non-game days).
- Box scores: `fetch_boxscores` (nba_api) nightly for the slate's games.

Deliverable: by the first weekend, a raw archive with line-movement series,
starter-confirmation times and projections at every offset for every slate.

## Phase 2: pipeline parity

- Move `scripts/nba/{generate_lineups?,sim,choose}` logic into
  `nba.field_generation`, `nba.simulate_contest`, `nba.lineup_export` the way
  the 2026-10-14 cleanup did for NFL; `choose` takes `--weights-json
  --no-prompt`; sim writes `contest_sim_report` and the `sim/<stamp>` npz/json
  artifacts; choose writes `chosen_entries.*`.
- Field/ownership: keep NBA's `sample_field_with_projection_model`
  (`field_distribution_model.pkl`, trained on last season's bins) for now and
  mark it as the first component to score once lineup-level fields exist.
- `tests/nba` with the same smoke tests NFL has for export and choose.

## Phase 3: web UI

Recommend one process per sport, sport fixed at start-up:
`serve.py --sport nba --port 8001`. Extract a `Sport` descriptor (paths,
scripts dir, DB name, `Base`, working tables, player table, menu path, models
dir) and `pipeline_base.py` (Step/Field dataclasses, entries_summary, weights
handling); `pipeline_nfl.py` and a new `pipeline_nba.py` supply only STEPS,
`slate_status`, `players_frame`. Hub/models/validate/plan views read paths
from the descriptor instead of `ivan.sports.nfl.paths`. The single-worker job
runner then stays one per sport, which is what keeps SQLite happy. Second
systemd unit on the server (192.168.1.59) for port 8001.

## Phase 4: settle, replay, backtest

- `nba.settle`: standings scrape for played + largest public contests (raw
  `dk/standings` export, derived `lineups.parquet` + `scores.npy` like NFL
  classic), box scores via nba_api (same night), `slate_status` rows; add NBA
  to the nightly job.
- `NBAReplayAdapter` (restore players/odds/starters from the capture, project,
  generate, contests_for, entries_csv, sample_field, simulate, choose). The
  hub, /models and /validate pages then work through the descriptor.
- Classic-field dupe/ownership models need lineup-level standings, which the
  scrape in phase 4 starts collecting; last season's 377 contests are bins only.

## Phase 5: judge the odds model before paying for Stokastic

Questions, cheapest first, each read off the phase-1 archive:
1. Coverage: share of a slate's salary and projected fpts that priced players
   carry at lock-30m, and how that changes from lock-6h. Last season's single
   capture priced 71 players; the minutes model carries the rest.
2. Timing: line movement lock-3h -> lock-15m; when Rotowire confirms starters
   relative to lock; how often a scratch lands after lock (late-swap value).
3. Accuracy: fpts MAE vs box scores at each offset. Last season's walk-forward
   was 7.44 vs Stokastic 7.41 with box-score availability; the live number is
   the test. Free yardsticks this season: DK AvgPointsPerGame (in the
   draftables payload) and the closing-line projection as an upper bound.
4. Money: replay the captured slates with the lock-30m capture against actual
   standings, the same scorecard as NFL, once phases 2-4 exist.

Decision rule to agree on now: after ~15 slates, if the lock-30m MAE is within
0.1-0.2 of last season's Stokastic gap and the contest replays are not clearly
negative, stay odds-driven. Note that Stokastic itself tracked the market at
r=0.994, so buying it does not buy a different projection; the NBA edge, if
any, is in minutes/late news and the field model, not the mean projection.

## Pitfalls carried over

- `scripts/nba/overnight.py` runs on import and cleans `data/nba/diagnostics`;
  never call it with `--help`. Retire it into the settle/nightly modules.
- `core.archive.KINDS` is `(players, odds, game_lines)`; NBA needs starters in
  the capture (add a kind or keep starters in raw only and re-parse on restore).
- Config reads are dotted-path; the web app is not import-reachable from
  `serve.py` without the package install.
- Uncommitted NFL/MMA changes are in the tree; checkpoint commit before NBA
  edits start.
