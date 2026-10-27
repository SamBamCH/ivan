# NBA: a slate day, in order

The producer's day. Every step has a desk button and a script; the script is named so the desk's
job log makes sense. The one rule to remember: **the desk shows its own copy of the projection, the
pipeline reads the pool (`nba_player`), and only Publish moves the first into the second.**

## 1. Load the slate

Desk: the slate picker (top of the page) lists a day, then that day's classic draft groups from the
DraftKings lobby. Picking one fetches the pool and the Rotowire starters, stores them, projects and
runs the public side (ownership, the field's projection). A past day restores the slate from its
archive capture instead (no fetch, nothing recorded). Switching files the slate you leave (its pins and
grids) under `data/nba/input/edits/<draft group>/` and brings back the new slate's own.

Loading a slate also deletes the lineups, pool snapshots, sim results and contest reports of every
*other* draft group (`ingest.isolate_slate`), so the working tables hold one slate at a time.

Terminal:

```bash
venv/bin/python scripts/nba/ingest.py --list                              # today's classic draft groups
venv/bin/python scripts/nba/ingest.py --draft-group 155647                # DK pool + Rotowire, project (free)
venv/bin/python scripts/nba/ingest.py --draft-group 155647 --update-odds  # also the player props (credits)
```

`ingest.py` and the desk use the same transaction (`ingest.refresh_pool`), so the script can run
while the desk is up; the desk shows "new data · load".

## 2. Refresh through the day

Desk **Refresh**: `dk` re-fetches the pool, statuses and Rotowire (free); `props` also fetches the
player props (about 12 credits per game) and re-derives the market rates. Pins and locked grid rows
survive a refresh; statuses and props change underneath them. Fetch props once the books have posted
most of the slate (early evening for a 7pm lock) and again if the news moved.

The contests and payout tables for the day are stored by the sim when it meets a contest it does
not know (`contests.ensure_contest`); there is no separate contest step to run.

## 3. Work the teams

The board sorts team cards by how far they are from their norm; alerts are listed in
[../web/nba-desk.md](../web/nba-desk.md). Per team you can:

- **pin** a player's minutes (0 scratches him), his fppm, his 10-90 band, or his projection; blank unpins.
  Written at once to `data/nba/input/minutes_overrides.csv`; the team re-projects around the pins.
- **call a starter** (toggle) when Rotowire has not confirmed yet.
- **paint the rotation grid**: who is on the floor in each of the 48 minutes. The grid autosaves on every
  stroke to `data/nba/input/rotations/<TEAM>.json`; a **locked** row is held exactly by every later
  projection, **Re-fill** fills the unlocked rows around the locks, **Hold all** locks the grid as drawn,
  **Clear locks** drops the team's pins and grid. The grid's row sums are the minutes projection
  ([projections.md](projections.md)).
- **Map** a player the rotation left off back onto the grid (he starts at his history or 6 minutes).
- **late news form**: player + call (out / starts / bench / back) + what to run next (sim and choose
  before lock, late swap after, or publish only). One request pins, re-allocates, publishes and queues.
- **scenarios**: save the team's edits under a name ("Giannis out"); the scenario whose assumptions all
  hold activates itself when a refresh or a call changes the facts.
- **Check news** fetches Rotowire's news page (`ivan.sports.nba.newsfeed`); the auto box repeats it every
  5 minutes while the desk runs. The header counts the slate's open questions (Q/GTD/D/P players with no call).

Terminal equivalents: `scripts/nba/news.py "Player Name" [--accept] [--conditional]` for one scratch
(shows the team with and without him, asks minutes player by player, writes the pool for that team);
`scripts/nba/project.py [--team XXX] [--dry-run]` re-projects the whole pool from the overrides file and
the grids without recording a capture row; `scripts/nba/validate.py review --team XXX` prints the
minutes provenance and flags for a team.

## 4. Publish

**Publish** writes the desk's projection into the pool (`nba_player.minutes_proj`, `fpts_mean`,
`fpts_std`, `fpts_quantiles`), records an `nba_projection_capture` row per player (source "ours"),
writes a projections CSV artifact under `data/nba/output/projections/`, and runs `feed.publish` +
`feed.push` so the server's desk gets it within five minutes. About 10 to 20 seconds.

Generate, Sim and Late swap from the desk publish first (since 2026-10-25). Choose does not.
The scripts run from a terminal do **not** publish: run them after a Publish, or use `news.py` /
`project.py`, which write the pool themselves.

## 5. Generate, simulate, choose

Desk: the **pipeline** panel (fields: lineups, jobs, sims, field cap, fresh pool, and the chooser's
weights from `config/nba.json → choose`). **All** runs the three in a row. Jobs run one at a time
in the desk's job runner (`ivan.web.jobs`) with logs under `data/nba/jobs/`.

```bash
venv/bin/python scripts/nba/generate_lineups.py 10000 8 [--fresh] [--randomness 0.25] [--min-fpts 260]
venv/bin/python scripts/nba/sim.py [-n 1000] [-c 10000] [--entries data/nba/input/entries.csv]
venv/bin/python scripts/nba/choose.py --no-prompt [--method greedy|portfolio] [--weights-json ...]
```

What they do and write is in [field-and-simulation.md](field-and-simulation.md). The chooser's
upload file is `data/nba/output/chosen_entries.classic.<stamp>.csv`; the desk's exposures view and
the Downloads list read the newest run. The "fresh pool" box unticked keeps the existing pool, so a
second scenario's lineups can sit beside the first's before lock.

## 6. Lock

`scripts/nba/archive_at_lock.py` (timer, every 5 minutes) records the pool and the projections at
each game's tip, within 12 minutes after it, without touching anything live. The first tip is the
slate lock. These `at_tip` captures are what the scorecard and the backtests compare against.

## 7. After lock: the late swap

```bash
venv/bin/python scripts/nba/lateswap.py                    # standings export through the logged-in Chrome profile
venv/bin/python scripts/nba/lateswap.py --source raw       # the newest stored export per contest, no Chrome
venv/bin/python scripts/nba/lateswap.py --live             # started games score from the live box score
venv/bin/python scripts/nba/lateswap.py --contest 197015856 --candidates 20 --sims 2000
```

For every contest in the entries file it downloads the live standings (the field's locked players),
completes the field's open slots with the optimiser, re-optimises our entries' open slots in sampled
worlds and keeps the candidate with the best expected payout. Output:
`data/nba/output/lateswap.<stamp>.csv` (the entries export with the swapped lineups, ready to upload)
and `.json` (the desk's late swap view). The desk's late swap tab runs the same script and has an
upload box for standings CSVs exported by hand (`/lateswap/standings`), which `--source raw` then reads.

`--live` uses NBA's public live feed (`post_lock/live.py`) and the in-game update
(`post_lock/ingame.py`): banked points plus a draw of the rest of the game under the live score.

## 8. The next morning

`scripts/nba/nightly.py` runs on its timer (box scores, scorecards, stints, standings derive, feed
push). The real contest fields need you: `scripts/nba/standings.py scrape --days 3` (Chrome login).
Both in [overnight.md](overnight.md). Read the scorecard with `scripts/nba/validate.py history`.

## The two copies, for reference

| Action | Fetches | Writes the pool |
|---|---|---|
| paint, pin, starter call, Re-fill | no | no (desk copy + the override files) |
| Publish | no | yes |
| Refresh dk / props, pick a slate | yes | yes |
| Generate / Sim / Late swap / All from the desk | no | yes (publish first) |
| Choose from the desk | no | no |
| `news.py`, `project.py`, `ingest.py --project-only` | no | yes |
| archive at tip | no | no (records it) |

The history of how this was learned is in [notes/late_news_flow_2026-10-25.md](notes/late_news_flow_2026-10-25.md).
