# NBA

DraftKings NBA classic slates: the laptop fetches the slate and the books, projects every player as
minutes times per-minute rates, builds a field, simulates the contests you entered and picks the
lineups to upload. A second machine (the server) runs the same desk for someone else on the
projections the laptop publishes. Everything lives in `src/ivan/sports/nba/`, the commands in
`scripts/nba/`, the desk in `src/ivan/web/nba_desk.py`.

Read in this order:

| Page                                                       | Question it answers                                                                                               |
|------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| [setup.md](setup.md)                                       | How do I get the NBA side running on a machine (database, env, timers, desk)?                                     |
| [game-day.md](game-day.md)                                 | What do I do on a slate day, in order, from loading the slate to the late swap?                                   |
| [projections.md](projections.md)                           | How is a player's projection built (props, minutes, the rotation grid, fantasy points, correlation)?              |
| [field-and-simulation.md](field-and-simulation.md)         | How are the field, the simulation and the chooser built, and what do they write?                                  |
| [overnight.md](overnight.md)                               | What runs on timers (nightly box scores, archive at tip, feed pull) and what do I run by hand (standings scrape)? |
| [validation-and-backtests.md](validation-and-backtests.md) | How do I check the model against what happened, refit it, and calibrate the field?                                |
| [data.md](data.md)                                         | What is in the database and on disk, and who writes each table and file?                                          |
| [../web/nba-desk.md](../web/nba-desk.md)                   | The desk page: cards, grids, pins, Publish, actions bar, exports, the consumer mode.                           |
| [notes/](notes/)                                           | Design notes and plans, dated; history of why things are the way they are.                                        |

## The flow in one picture

```
                 laptop (producer)                                     server (consumer)
                 ────────────────                                      ─────────────────
 DK lobby ─┐
 Rotowire ─┼─► desk Refresh / ingest.py ─► nba_player (pool) ─┐
 odds-api ─┘      (store_slate)                               │
                                                              ▼
                       projection (projection/live.py) ─► minutes_proj, fpts_mean/std, quantiles
                       desk pins + painted grids ─────────► minutes_overrides.csv, rotations/*.json
                                                              │
                                           Publish ───────────┼─► feed.publish + push ───► data/feed ──► feed.pull (5 min)
                                                              │                                             └► brother's desk
                       generate_lineups.py  ─► lineup pool    │
                       sim.py               ─► lineup_simulation_results + contest_sim_report
                       choose.py            ─► chosen_entries.*.csv (DK upload)
                                                              │
 lock ───► archive_at_lock.py (timer) records the pool at each tip
 after lock ───► fetch_standings.py (a minute after a tip) ─► lateswap.py re-optimises the open slots of our entries
 next morning ───► nightly.py: box scores, scorecards, stints, standings derive, feed push
                   standings.py scrape: the real fields (by hand, needs the DK Chrome login)
```

## Where to look when something is wrong

| Symptom                                   | Look at                                                                                                     |
|-------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| The desk shows one thing, the sim another | [game-day.md](game-day.md): only Publish writes the pool; sim/lateswap/generate from the desk publish first |
| A player's minutes look wrong             | [projections.md](projections.md) minutes section; `scripts/nba/validate.py review --team XXX`               |
| The brother's desk is stale               | [overnight.md](overnight.md) feed section; [../web/nba-desk.md](../web/nba-desk.md) consumer mode           |
| No box scores / scorecards this morning   | [overnight.md](overnight.md): `data/nba/logs/nightly.log`, `systemctl --user list-timers`                   |
| Field ownership looks off                 | [field-and-simulation.md](field-and-simulation.md) field model; `scripts/nba/field_model.py check`          |
| A past slate needs re-examining           | the desk's slate picker restores a past slate from its archive capture; [data.md](data.md) archive section  |
|                                           |                                                                                                             |
|                                           |
|                                           |

## Status (2026-10-27)

Live since opening night 2026-10-20. The desk is the only thing that fetches on a slate day (the
capture timer is installed but disabled since 2026-10-23). The consumer desk is deployed at
`https://ivan.prattflix.com/nba` behind a password. Not yet built: a replay adapter for NBA (the NFL
one exists), settle as a table of slate status, a decision rule for buying Stokastic, and the
field model's calibration on more than a handful of real contests. Plans and findings are in
[notes/plan_2026-27.md](notes/plan_2026-27.md) and [notes/minutes_model_plan.md](notes/minutes_model_plan.md).
