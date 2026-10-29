# NBA: a slate day, in order

The producer's day. Every step has a desk button and a script; the script is named so the desk's
job log makes sense. The one rule to remember: **the desk shows its own copy of the projection, the
pipeline reads the pool (`nba_player`), and only Publish moves the first into the second.**

## 1. Load the slate

Desk: the slate picker (top of the page) lists a day, then that day's classic draft groups from the
DraftKings lobby. Picking one fetches the pool and the Rotowire starters, stores them, projects and
writes the public projection (the field's numbers). A past day restores the slate from its
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

Desk **Refresh**: `dk` re-fetches the pool, statuses, Rotowire's starters and its news (free); `props` also fetches the
player props (about 12 credits per game) and re-derives the market rates. Pins and locked grid rows
survive a refresh; statuses and props change underneath them. Fetch props once the books have posted
most of the slate (early evening for a 7pm lock) and again if the news moved.

A refresh takes about 10 seconds: the fetch, the market means (the props de-vigged in one
vectorised pass, `projection/market.py`), the pool, the projection and the public projection land
in one transaction. Nothing else runs: the public field and the ownership column are **not** rebuilt
by a refresh, a generate or an edit (since 2026-10-28). The sim builds the field it needs and writes
ownership from the largest contest's field; the pipeline panel's **Field + ownership** button
(`scripts/nba/field_ownership.py`, about 45 seconds at `field.ownership_entries` = 3,000) builds it
on demand when you want the column fresh before a sim. `ingest.py --field` does both in one go.

**Autopilot** (the header box; `scripts/nba/autopilot.py on|off|status`) is for a day you are not
working the desk but the other desk still plays. With it on, `ivan-nba-capture.timer` runs
`scripts/nba/capture.py` every 15 minutes: it refreshes the loaded slate (or the one that locks next)
through the same one-transaction path as Refresh, fetches the props when the slate crosses 360, 180,
90, 45 and 15 minutes before lock (about 12 credits per game each, at most 20 fetches a day), the
game lines with them, projects with the pins and grids on file, and publishes + pushes the feed. It
stands down while you are at the desk: an edit in the last 30 minutes not published since, or a
running job (`autopilot.desk_hold`). The switch is `data/nba/autopilot.json`; the last tick's result
shows next to the box and in the other desk's header. Off by default; turn it off when you play.
`scripts/nba/autopilot.py tick --force` runs one tick from a terminal whatever the switch says.

The contests and payout tables for the day are stored by the sim when it meets a contest it does
not know (`contests.ensure_contest`), and by the autopilot tick for its slate; there is no separate
contest step to run.

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
- **News** comes with every Refresh: the pool refresh (`ingest.run_ingest`, so `ingest.py` and the autopilot tick
  too) reads Rotowire's news page after the commit (`ivan.sports.nba.newsfeed`); the news auto box also reads it every
  5 minutes while the desk runs. The header counts the slate's open questions (Q/GTD/D/P players with no call).

Terminal equivalents: `scripts/nba/news.py "Player Name" [--accept] [--conditional]` for one scratch
(shows the team with and without him, asks minutes player by player, writes the pool for that team);
`scripts/nba/project.py [--team XXX] [--dry-run]` re-projects the whole pool from the overrides file and
the grids without recording a capture row; `scripts/nba/validate.py review --team XXX` prints the
minutes provenance and flags for a team.

## 4. Publish

**Publish** first applies any value still typed into a grid and not yet refigured (the page sends them along,
so nothing typed is lost to the reload), then writes the desk's projection into the pool (`nba_player.minutes_proj`, `fpts_mean`,
`fpts_std`, `fpts_quantiles`), records an `nba_projection_capture` row per player (source "ours"),
writes a projections CSV artifact under `data/nba/output/projections/`, and runs `feed.publish` +
`feed.push` (with a producer note: desk, autopilot or nightly, and the autopilot state) so the server's
desk gets it on its next pull: its timer every 5 minutes, or at once when its Refresh is pressed.
About 10 to 20 seconds. Terminal: `scripts/nba/project.py` then `scripts/nba/feed.py publish && scripts/nba/feed.py push`.

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

Two steps, apart: the **standings fetch** and the **swap**.

```bash
venv/bin/python scripts/nba/fetch_standings.py             # every contest's standings export, stored (Chrome login)
venv/bin/python scripts/nba/lateswap.py                    # the swap on the stored exports while they are current
venv/bin/python scripts/nba/lateswap.py --live             # started games score from the live box score
venv/bin/python scripts/nba/lateswap.py --source draftkings   # always download first (the old default)
venv/bin/python scripts/nba/lateswap.py --source raw       # the newest stored export per contest, never Chrome
venv/bin/python scripts/nba/lateswap.py --field each       # each contest against its own export (default: one shared)
venv/bin/python scripts/nba/lateswap.py --contest 197015856 --candidates 20 --sims 2000
```

**Fetch standings** (the desk's button on the late swap tab, or `fetch_standings.py`) downloads the
live standings export of every contest in the entries file through the logged-in Chrome profile and
keeps it under raw `dk/standings/<contest id>/`. The export shows which players are locked, and that
changes only at a tip: so press it a minute after a game tips, and the swap can be re-run as often as
the projections move until the next tip without fetching again. The desk lists each contest's
export with its time, green while no game has tipped since it was taken, amber ("before the last
tip") once one has.

**Run late swap** with standings "stored while current" (`--source auto`, the default): a contest's
stored export is used while it is current; when a game has tipped since it was taken the swap
downloads a fresh one (opening Chrome only then), and without a login it falls back to the stored
one with a warning in the log. "DraftKings live" always downloads; "stored export" never does.

The field: with **one for all contests** (`--field shared`, the default) the biggest contest's export
is completed once (its open slots filled by the optimiser in sampled worlds, its %Drafted as the
targets) and every contest is scored against a sample of it at its own size, its payout table scaled
to that; one fetch of one export is then enough. **Each contest's own** (`--field each`) completes
every contest's export as before. Then, per contest, our entries' open slots are re-optimised in
sampled worlds and the candidate with the best expected payout is kept. Output:
`data/nba/output/lateswap.<stamp>.csv` (the entries export with the swapped lineups, ready to upload)
and `.json` (the desk's late swap view, which names the export each field came from). The desk's
late swap tab runs the same script and has an upload box for standings CSVs exported by hand
(`/lateswap/standings`), which the stored sources then read.

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
| Fetch standings | yes (DraftKings standings) | no |
| `news.py`, `project.py`, `ingest.py --project-only` | no | yes |
| archive at tip | no | no (records it) |

The history of how this was learned is in [notes/late_news_flow_2026-10-25.md](notes/late_news_flow_2026-10-25.md).
