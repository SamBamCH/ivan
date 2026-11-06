# NBA: a slate day, in order

The producer's day. Every step has a desk button and a script; the script is named so the desk's
job log makes sense. The rules to remember: **every slate is its own directory** (`data/nba/slates/<dg>/`,
[data.md](data.md)), so the desk holds as many slates as you open and nothing you do on one touches another; and
**the pool is the projection**: every edit writes the slate's pool at once, the pipeline reads that pool.

## 1. Load the slate

Desk: the bar's day and slate dropdowns (a day opens its main slate, a slate its page;
[../web/README.md](../web/README.md)). Opening a slate this machine does not hold fetches its pool and the Rotowire
starters into its own directory, stores them, projects and writes the public projection (the field's numbers); every
other slate stays exactly as it was, and flipping back to one is instant. A past day's slate opens its backtest instead,
read only ([../web/nba-desk.md](../web/nba-desk.md)); its **Restore into the desk** button loads it from its archive
capture (no fetch, nothing recorded) into its own directory to re-work it. Each slate keeps its own lineups, sims,
contests and edits in its directory; nothing is filed away or cleared by opening another.

Terminal:

```bash
venv/bin/python scripts/nba/ingest.py --list                              # today's classic draft groups
venv/bin/python scripts/nba/ingest.py --draft-group 155647                # DK pool + Rotowire, project (free) into slates/155647/
venv/bin/python scripts/nba/ingest.py --draft-group 155647 --update-odds  # also the player props (credits)
```

`ingest.py` and the desk use the same transaction (`ingest.refresh_pool`) on the slate's own database, so the script can
run while the desk is up; the desk shows "new data · load". Every other script takes `--slate DG` (default: today's main
held slate, `slates.resolve`).

## 2. Refresh through the day

Desk **Refresh**: `dk` re-fetches the pool, statuses, Rotowire's starters and its news (free); `lines` also fetches
every game's spread and total (about 3 credits; `ingest.py --lines`), which the games strip and the minutes model's
blowout draw read: nothing else fetches them on a day worked by hand. `props` fetches the lines and the
player props (about 12 credits per game) and re-derives the market rates. Pins and locked grid rows
survive a refresh; statuses and props change underneath them. Fetch props once the books have posted
most of the slate (early evening for a 7pm lock) and again if the news moved.

A refresh takes about 10 seconds: the fetch, the market means (the props de-vigged in one
vectorised pass, `projection/market.py`), the pool, the projection and the public projection land
in one transaction. Nothing else runs: the public field and the ownership column are **not** rebuilt
by a refresh, a generate or an edit (since 2026-10-28). The sim builds the field it needs and writes
ownership from the largest contest's field; the actions bar's **Field + ownership** button
(`scripts/nba/field_ownership.py`, about 45 seconds at `field.ownership_entries` = 3,000) builds it
on demand when you want the column fresh before a sim. `ingest.py --field` does both in one go.

**Autopilot** (the header box; `scripts/nba/autopilot.py on|off|status`) is for a day you are not
working the desk but the other desk still plays. With it on, `ivan-nba-capture.timer` runs
`scripts/nba/capture.py` every 15 minutes: it refreshes **every slate this machine holds that has not locked**
(today's and tomorrow's, each in its own directory) through the same one-transaction path as Refresh, fetches a slate's
props when it crosses 360, 180, 90, 45 and 15 minutes before its lock (about 12 credits per game each, at most 20
fetches a day), the game lines with them, and projects with each slate's pins and grids, so the slates the server pulls
(every 30 minutes, and on its Refresh from laptop) are fresh and every slate's pre-lock record is kept. A slate nobody
opened is not refreshed. It stands down while a desk job is running (`autopilot.desk_hold`; an edit counts as published
the moment it is written, so edits never hold it). The switch is `data/nba/autopilot.json`; the last tick's result shows
next to the box and in the other desk's header. Off by default; turn it off when you play.
`scripts/nba/autopilot.py tick --force` runs one tick from a terminal whatever the switch says; `capture.py --only DG`
one slate.

The contests and payout tables for the day are stored by the sim when it meets a contest it does
not know (`contests.ensure_contest`), and by the autopilot tick for its slate; there is no separate
contest step to run.

## 3. Work the teams

The slate page reads; a team is worked on its own page (a chip in the games strip, a name in the news table or the
players view: `/nba/<day>/<slate>/<TEAM>/`). You open a team to map its minutes, when the slate page's **needs you**
list names it (changed since mapped, an open question or no lineup near the tip, a new headline), or when news breaks.
Everything else stays green and unopened.

**Every edit on a team page writes the team into the slate's pool at once** (`Desk.reproject` = `ingest.project_team`: the
team's `nba_player` rows and its record in the slate's projection store, `data/nba/slates/<dg>/projection/<TEAM>.pkl`,
about 0.3 s). A refresh of the slate running meanwhile does not close the page: it keeps reading, its edits wait. There is no publish step after an edit: the slate page, Generate, the sim, the late swap and the other desk
read what you see. Leaving the page loses nothing.

Start each team at its **special considerations** box (rest, starters out, a new five, players back, big
spread): the things the history-built rotation does not know. `scripts/nba/considerations.py` prints the same.
Alerts are listed in [../web/nba-desk.md](../web/nba-desk.md). Per team you can:

- **pin** a player's minutes (0 scratches him), his fppm, his 10-90 band, or his projection; blank unpins.
  Written at once to the day's `data/nba/days/<day>/edits/minutes_overrides.csv` (the slates of one day share their edits;
  the next day starts clean); the team re-projects around the pins.
- **call a starter** (toggle) when Rotowire has not confirmed yet.
- **a player ruled out voids the team's mapping** (`ivan.sports.nba.mapping`, inside every refresh and before every
  re-projection on the desk; `scripts/nba/mapping.py --slate DG` from a terminal). The mapping was made against a pool the
  grid's fingerprint recorded; when one of those players is OUT (DK's status or a scratch call here), the grid and the
  team's positive minutes pins go, the model initialises the new pool itself (the last game's rotation minus the outs,
  plus returners) and its rotation is on the grid; the team reads **unmapped** until you accept it (**Hold all**) or paint
  and hold. Nothing has to be cleared by hand first. The voided mapping is kept as a scenario assuming the player in
  ("Johnson in"), so a reversed call puts it straight back. Other changes (a starter, a status short of OUT, a player
  back) still only mark the team **changed** for you to look at.
- **paint the rotation grid**: who is on the floor in each of the 48 minutes. The grid autosaves on every
  stroke to `data/nba/days/<day>/edits/rotations/<TEAM>.json`; a **locked** row is held exactly by every later
  projection, **Re-fill** fills the unlocked rows around the locks, **Hold all** locks the grid as drawn,
  **Clear locks** drops the team's pins and grid. The grid's row sums are the minutes projection
  ([projections.md](projections.md)).
- **Map** a player the rotation left off back onto the grid (he starts at his history or 6 minutes).
- **late news form**: player + call (out / starts / bench / back) + what to run next (sim and choose
  before lock, late swap after, or the call only). One request pins, re-allocates, writes the pool and queues the
  jobs; the page is back in about a second.
- **scenarios**: save the team's edits under a name ("Giannis out"); the scenario whose assumptions all
  hold activates itself when a refresh or a call changes the facts.
- **News** comes with every Refresh: the pool refresh (`ingest.run_ingest`, so `ingest.py` and the autopilot tick
  too) reads Rotowire's news page after the commit (`ivan.sports.nba.newsfeed`); the news auto box also reads it every
  5 minutes while the desk runs. The header counts the slate's open questions (Q/GTD/D/P players with no call).
- **the news table** (on the slate page; `scripts/nba/newsfeed.py slate` in a terminal) is where to start:
  the slate's questionable players first, one row each with the model's minutes, your call and the last
  headline; then the players out tonight who played their team's last game (the rotations that move tonight);
  then, collapsed, the ones already out. Work it top down: every question gets a call (a pin, a starter call
  or the late news form, on the team's page) and every new out gets its team's rotation looked at. Columns and
  rules are in [../web/nba-desk.md](../web/nba-desk.md#the-news-table).

Terminal equivalents: `scripts/nba/news.py "Player Name" [--accept] [--conditional]` for one scratch
(shows the team with and without him, asks minutes player by player, writes the pool for that team);
`scripts/nba/project.py [--team XXX] [--dry-run]` re-projects the pool (or one team) from the overrides file and
the grids through the same `project_team` path, store record included, without a capture row;
`scripts/nba/validate.py review --team XXX` prints the minutes provenance and flags for a team.

## 4. The snapshot (the record a run is built on)

The pool is the projection: every edit, every refresh and `project.py` write it, and nothing is published from the
desk (the server pulls the slate, [../deployment.md](../deployment.md)). What remains of the old Publish is the
**snapshot**: `ingest.capture_pool` records the pool as it stands, without projecting anything, as an
`nba_projection_capture` row per player (source "ours", the rows the scorecard scores) and a projections CSV under
`data/nba/slates/<dg>/output/projections/`, from the projection store; any value still typed into a grid is applied
first. Under a second. It has no button: Generate, Sim, Late swap and 3–5 in a row take it before they start, so each
run has the record of what it was built on (Choose and Field + ownership do not; a refresh records its own projection
as it writes it). For a record between runs, `scripts/nba/snapshot.py --slate DG` from a terminal.

The actions bar's **chain** says what is current: `pool HH:MM › snapshot › field › lineups › sim › picks`, each
stage green when built after the pool's last write, amber when the pool moved since (run it again), red when the
pool moved 10+ minutes ago and the stage never followed.

When a refresh, a **Field + ownership** build or a sim finishes, the page reloads on the new pool: the desk's pool stamp
(`Desk._stamp`) covers the ownership column too, so the board's own column and the projections CSV export
(`/export/projections.csv`) carry the new numbers at once, whoever wrote them (a desk job, a script, the autopilot).

## 5. Generate, simulate, choose

**How long will it take?** Before a job starts the desk estimates it from this machine's past runs
(`ivan.sports.nba.job_time`, shown under the Generate and Sim knobs and in the job's status line) and checks the end against
the slate's lock less 5 minutes: a run that would end inside that margin is flagged red and starts with a **heads up** in
the flash; nothing is refused (2026-11-03: a sim too big for the time left missed lock; the refusal that followed was
dropped on 2026-11-05 when it held a 56 s sim two minutes before lock). From a terminal:

```bash
venv/bin/python scripts/nba/job_time.py fit                                                   # the model: seconds per unit per step
venv/bin/python scripts/nba/job_time.py estimate --step sim --sims 1000 --field-cap 10000 --contests 6 --slate DG
```


Desk: steps 3 to 5 of the **actions** bar, with the knobs in the **config** tile beside the games strip (lineups, jobs,
sims, field cap, fresh pool, and the chooser's weights from `config/nba.json → choose`); **3–5 in a row** runs the
three. The entries export goes in through the **contests** section, which flags contests of another slate (the sim
skips them) and takes each contest's own chooser knobs (`scripts/nba/contest_weights.py`; Choose passes them as
`--weights-json data/nba/slates/<dg>/input/contest_weights.json`; every job carries `--slate DG`). Jobs run
one at a time in the desk's job runner (`ivan.web.jobs`) with logs under `data/nba/jobs/`.

```bash
venv/bin/python scripts/nba/generate_lineups.py 10000 8 [--fresh] [--randomness 0.25] [--min-fpts 260]
venv/bin/python scripts/nba/sim.py --slate DG [-n 1000] [-c 10000] [--entries data/nba/slates/DG/input/entries.csv]
venv/bin/python scripts/nba/choose.py --no-prompt [--method greedy|portfolio] [--weights-json ...]
```

What they do and write is in [field-and-simulation.md](field-and-simulation.md). The chooser's
upload file is `data/nba/output/chosen_entries.classic.<stamp>.csv`; the desk's exposures section and
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
| paint + Apply, pin, starter call, Re-fill, a late-news call | no | **yes, that team** (the override files, then `ingest.project_team`: the team's rows and its projection store record) |
| `scripts/nba/snapshot.py` | no | no (records the pool: capture rows + the projections CSV) |
| Refresh dk / props, pick a slate | yes | yes |
| Generate / Sim / Late swap / All from the desk | no | no (a snapshot first, then the job reads the pool; the sim writes the ownership column) |
| Choose from the desk | no | no |
| Fetch standings | yes (DraftKings standings) | no |
| `news.py`, `project.py`, `ingest.py --project-only` | no | yes |
| archive at tip | no | no (records it) |

The history of how this was learned is in [notes/late_news_flow_2026-10-25.md](notes/late_news_flow_2026-10-25.md).
