# NBA

DraftKings NBA classic slates: the laptop fetches the slate and the books, maps every team (minutes on a
rotation grid, per-minute rates from the props), simulates each game on that map and reads every player's
projection off the simulated worlds, builds a field, simulates the contests you entered and picks the
lineups to upload. A second machine (the server) runs the same desk for someone else on the
projections the laptop publishes. The code is `src/ivan/sports/nba/`, one package per stage in the order the
process runs; the commands are `scripts/nba/`, the desk's pages `src/ivan/web/nba/`.

```
src/ivan/sports/nba/
  models.py  slates.py  ingest.py (fetch, store, project: the pipeline)  names.py  scoring.py  paths.py  utils.py
  data_fetch/   slate, contests, newsfeed, boxscores, pbp, live, standings, experience   what talks to the outside world
  data_clean/   boxscores, odds, pbp, standings                                    fetched payloads into the tables, read back
  projection/   market, history, solve, skeleton, rotation, rates, engine, live, worlds, store, record   layer 1 the mapping, layer 2 the World
  gamesim/      script, rotation, usage, profiles, world, draw                     the games simulated, n worlds at once
  field/        public_projection, public_field, model, generate, solver, optimal   the public's field; our candidate pool; the optimal per world
  contest/      simulate, choose, export                                            each contest simulated, lineups chosen
  lateswap/     swap, ingame, entry_parsing                                         after lock
  backtest/     validation, slate, results, readiness, field, tips, rerun           how it went, how right the model was
  slate/        autopilot, changes, mapping, scenarios, notes, considerations, job_time   the desk's working files
  desk/         the Desk object the pages render
```

Read in this order:

| Page                                                       | Question it answers                                                                                               |
|------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| [setup.md](setup.md)                                       | How do I get the NBA side running on a machine (database, env, timers, desk)?                                     |
| [game-day.md](game-day.md)                                 | What do I do on a slate day, in order, from loading the slate to the late swap?                                   |
| [projections.md](projections.md)                           | How is a player's projection built (props, minutes, the rotation grid, fantasy points, correlation)?              |
| [field-and-simulation.md](field-and-simulation.md)         | How are the field, the simulation and the chooser built, and what do they write?                                  |
| [overnight.md](overnight.md)                               | What runs on timers (nightly box scores, archive at tip, feed pull) and what do I run by hand (standings scrape)? Is yesterday ready to backtest (`readiness.py`)? |
| [validation-and-backtests.md](validation-and-backtests.md) | How do I check the model against what happened, refit it, and calibrate the field?                                |
| [data.md](data.md)                                         | What is in the database and on disk, and who writes each table and file?                                          |
| [../web/nba-desk.md](../web/nba-desk.md)                   | The desk page: cards, grids, pins, Publish, actions bar, exports, the consumer mode.                           |
| [notes/](notes/)                                           | Design notes and plans, dated; history of why things are the way they are.                                        |

## The flow in one picture

```
                 laptop (producer)                                     server (consumer)
                 ────────────────                                      ─────────────────
 DK lobby ─┐
 Rotowire ─┼─► desk Refresh / ingest.py ─► slates/<dg>/slate.sqlite (the slate's pool) ─┐
 odds-api ─┘      (store_slate)                               │
                                                              ▼
                       layer 1 the mapping (projection/engine, live) ─► minutes_proj, rates, the grid row
                       layer 2 the World on it (projection/worlds)   ─► fpts_mean/std, quantiles, the minutes band
                       desk pins + painted grids ─────────► days/<day>/edits/{minutes_overrides.csv, rotations/*.json}
                                                              │
                       every edit / Refresh writes the pool ─┼─◄── ssh: publish + rrsync (feed.pull_ssh) ◄── brother's desk: Refresh from laptop
                                                              │
                       generate_lineups.py  ─► lineup pool    │
                       sim.py               ─► lineup_simulation_results + contest_sim_report
                       choose.py            ─► chosen_entries.*.csv (DK upload)
                                                              │
 lock ───► archive_at_lock.py (timer) records the pool at each tip
 after lock ───► fetch_standings.py (a minute after a tip) ─► lateswap.py re-optimises the open slots of our entries
 next morning ───► nightly.py: box scores, scorecards, stints, standings derive, the reference feed bundle
                   standings.py scrape: the real fields (by hand, needs the DK Chrome login)
```

## Where to look when something is wrong

| Symptom                                   | Look at                                                                                                     |
|-------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| The desk shows one thing, the sim another | [game-day.md](game-day.md): only Publish writes the pool; sim/lateswap/generate from the desk publish first |
| A player's minutes look wrong             | [projections.md](projections.md) minutes section; `scripts/nba/validate.py review --team XXX`               |
| A player's market rate looks like another game's | [projections.md](projections.md) market rates; `scripts/nba/odds_games.py --player "name"` says which game each row is and why it counts |
| A starter or a headline has no history    | [data.md](data.md) player names: one key for every source (`names.py`); `scripts/nba/normalize_names.py` re-keys old rows |
| The brother's desk is stale               | [overnight.md](overnight.md) feed section; [../web/nba-desk.md](../web/nba-desk.md) consumer mode           |
| No box scores / scorecards this morning   | [overnight.md](overnight.md): `data/nba/logs/nightly.log`, `systemctl --user list-timers`                   |
| Field ownership looks off                 | `scripts/nba/backtest_field.py` (real fields vs the model, [validation-and-backtests.md](validation-and-backtests.md)); [field-and-simulation.md](field-and-simulation.md) field model |
| A past slate needs re-examining           | a past slate's page is its backtest; its Restore button loads it from its archive capture; [data.md](data.md) archive section  |
|                                           |                                                                                                             |
|                                           |
|                                           |

## Status (2026-10-27)

Live since opening night 2026-10-20. The desk is the only thing that fetches on a slate day (the
capture timer is installed but disabled since 2026-10-23). The consumer desk is deployed at
`https://ivan.prattflix.com/nba` behind a password. Not yet built: a replay adapter for NBA (the NFL
one exists), settle as a table of slate status, a decision rule for buying Stokastic, and the
field model's calibration on more than a handful of real contests. Plans and findings are in
[notes/plan_2026-27.md](notes/plan_2026-27.md), [notes/minutes_model_plan.md](notes/minutes_model_plan.md) and
[notes/game_sim_plan_2026-11-07.md](notes/game_sim_plan_2026-11-07.md) (the game simulation: possessions, fouls,
usage shares and outcomes as layers, each validated on the stored play-by-play before the live draw changes).
