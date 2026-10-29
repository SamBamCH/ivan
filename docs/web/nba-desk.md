# The NBA desk

`src/ivan/web/nba_desk.py` (FastAPI + Jinja + htmx, templates `src/ivan/web/templates/nba_desk.html`
and `_nba_*.html`). One page to manage a slate in the last minutes before lock, plus the pipeline
panel, the exposures view and the late swap. Served on its own by `scripts/nba/serve.py` (port 8001)
or mounted at `/nba` inside the NFL app (`ivan.web.app`), which is how the server runs it. Every link
carries the mount prefix (`root`). The flow of a day is [../nba/game-day.md](../nba/game-day.md); this
page is what is on the screen.

## Header

- **Slate picker**: a day, then that day's classic draft groups (from the DK lobby for today and
  tomorrow, from the archive for past days; the loaded slate is always listed). Picking a slate goes at once, and
  the ‹ › arrows step through every slate in time order. A slate of **today or tomorrow loads** into the desk (fetched,
  projected; the slate it replaces keeps its lineups, sims and edits for when it comes back). A slate of an
  **earlier day opens its backtest** (below), read only: nothing is loaded or deleted. The backtest's **Restore into
  the desk** button loads a past slate from its archive capture when it is to be re-worked.
  While a load or a refresh runs the board shows its progress; when it is done the whole page reloads, so the
  picker and the header counts show the new slate.
- **Refresh DK** (free) / **Refresh lines** (the game lines, spread and total, about 3 credits) / **Refresh props**
  (about 12 credits per game plus the lines; confirm dialog). Spreads show only once tonight's lines are fetched. The board shows the
  state until the new pool is projected (about 10 seconds). The public field and the ownership column are not
  rebuilt by a refresh: the sim does that, or the pipeline panel's **Field + ownership** button. On a consumer a
  single **Refresh** pulls the feed instead.
- **Publish**: applies any values still typed into the grids (not yet refigured) first, writes the projection into
  the pool, records a capture, publishes and pushes the feed. On a consumer (`ingest.consumer_publish`) the live
  bundle last pulled is applied again and only the slate's teams it edited are projected on top: the other teams
  are exactly as published, and clearing an edit then publishing gives the published numbers back. Generate, Sim
  and Late swap do the same before they queue.
- **news auto** box: reads Rotowire's news every 5 minutes while the desk runs. There is no news button: every
  Refresh (and a slate load) reads the news after the pool, and its message lists the new headlines on this slate.
- **autopilot** box (producer only; `scripts/nba/autopilot.py on|off|status`): the capture tick refreshes,
  fetches props at the offsets and publishes + pushes every 15 minutes while nobody is editing here; the
  last tick's time shows next to it ("held" when it stood down, "failed" in red). See
  [../nba/game-day.md](../nba/game-day.md#2-refresh-through-the-day). On a consumer the same spot shows
  the laptop's state instead: autopilot on/off, who published last (desk, autopilot, nightly) and when.
- **new data · load** appears when the pool changed on disk since the page was built (a terminal
  ingest or project, a feed pull); it reloads without fetching.
- The questions count: Q/GTD/D/P players with no call yet, by team.
- Views: **slate** (the board), **players**, **late swap**, **exposures**, **backtest**. Opening the desk with no view
  picks one from the loaded slate (`slate_backtest.mode_for`): before its first lock the slate, after it the late swap,
  and a slate of an earlier day its backtest.

## The board (view `slate`)

A games strip with tips and each game's line as "DET by 7.5 · 228" (the favourite, the points it is favoured by, the
total; tonight's game only, "no line" until the props refresh fetches the game lines), then team cards sorted by an
alert score (bad 10, warn 3, info 1):

| Level | Fires when |
|---|---|
| bad | a rotation player is OUT or scratched ("out, frees N minutes"); dnp-prone, deep bench projected like a rotation player, impossible band, distribution suspect |
| warn | a new starter or a benched regular; the market disagrees with history ("market ..."); a questionable rotation player; new team / above recent maximum / market implies other minutes / no history for a 10+ minute player; a 24+ minute player left with no minutes in the rotation; no Rotowire lineup or an unconfirmed one inside 75 minutes of tip |
| info | minutes far from history for a 12+ minute player; the available roster's history minutes far from 241; rotation depth notes; unpriced rotation players |

The flags come from `ivan.sports.nba.validation.review_team`, the same ones `scripts/nba/validate.py review` prints.

### The news table

Under the games strip, before the open team's card (`_nba_news.html`, `Desk.news_table` =
`ivan.sports.nba.newsfeed.slate_news`; terminal: `scripts/nba/newsfeed.py slate`). The slate's players whose
night is news, one row per player, in three lists:

1. **Questions**: DK status Q / GTD / D / P, not ruled out. Columns: player (bold when in the expected lineup;
   the name opens his team's card), team, opponent and tip, status, salary, usual minutes (the history state),
   the model's minutes now with your call ("no call" in amber until a pin or a starter call answers it), the
   latest headline on file.
2. **Out tonight, played last game**: DK OUT / O / IR / SUSP and on the floor in his team's last box score. These are
   the outs the rotation has not absorbed yet: the column shows the last game's date and his minutes in it.
3. **Already out** (collapsed): OUT and out of the team's last game too, or never in a box score ("did not play").

Each list is ordered earliest tip first, then the most usual minutes. The headline is the newest `nba_news`
row for the player within 7 days (its source link; hover for the text), with its age, the item count and
**new** when it arrived after the team was mapped (within 3 hours when the team is unmapped). "No news on
file" means nothing stored: Refresh reads the news. The team's last game is the newest box score on file, so
until the nightly brings last night's game the "last game" is the one before it.

## A team card (`/team/<TEAM>`)

Rows: starters first, then the players who are out, then the bench, minutes descending. Per player:
status, Rotowire starter mark, history minutes (and the last games), the model's minutes with its
10-90 band, fppm, **proj** (ours) and **pub** (the public projection the field plays to), DK FPPG,
salary, value, the rate source (market or history), the ownership, the market-implied minutes for
priced players, the flags. Inputs on the row:

- **minutes** pins (0 scratches); **fppm**, **low/high** band, **fpts** pin; blank unpins. Saved to
  `minutes_overrides.csv` at once; the team re-projects in place (`/team/<TEAM>/pin`).
- **starter** toggle (`/starter`).
- **Refigure** (`/refigure`): re-run the team's projection now with the current pins.
- **Clear** (`/clear`): drop the team's pins and grid.

**Since the last refresh** (top of the card): what the last Refresh moved for the team (`ivan.sports.nba.changes`,
returned by `ingest.run_ingest` and printed by `ingest.py`; a consumer's Refresh diffs its pull): statuses, players
into or out of Rotowire's five and a confirmed lineup, players added to or dropped from the pool, projections that
moved 3+ points, new headlines. The team's chip in the games strip gets a blue dot with the list on hover.

**Notes** (`/team/<TEAM>/note` = `scripts/nba/notes.py set|clear|list`, `ivan.sports.nba.notes`): the pencil beside a
name on tonight's grid asks for a note on that player (blank clears it); it turns amber when he has one, hovering shows
it, and the card lists the team's notes with a clear button. A note belongs to the slate it was written on (draft group
and day): it shows only while that slate is loaded, a past day restored from its archive included, and stays until
cleared. The slate's scorecard lists the notes that stood at lock beside how each player went
([../nba/validation-and-backtests.md](../nba/validation-and-backtests.md)). Each machine keeps its own (the table is not in the feed).

**Special considerations** (above the grids; `Desk.considerations` = `ivan.sports.nba.considerations`, terminal
`scripts/nba/considerations.py [--team XXX]`): what is unusual about tonight's game, warnings first. Rest (back to back,
3 games in 4 nights, 3+ days off; the opponent's back to back), a regular starter (started most games, 20+ usual
minutes) OUT or questionable, tonight's five against the last game's (who is new in it, who dropped out; a rookie
starting), a 15+ minute player back after sitting out the last game(s), a player whose last game was for another
team, a spread of 10+ (blowout risk) or no line yet, the opponent's regular starters out. From the files on hand: the
box scores lag until the nightly brings last night's game, so a back to back can read as a day of rest more before it
runs.

The **rotation grid** (`_nba_rotation.html`): players × 48 minutes. Paint cells to put a player on the
floor; the grid autosaves every stroke (`/rotation`). Row lock holds a row exactly; **Re-fill** fills the
unlocked rows around the locks (`/rotation/propose` with `source=propose`, or a past game's grid with
`game_id`); **Hold all** locks everything as drawn; **Clear locks**; **Map** adds a player the cut left off
(`/rotation/add`), **drop** removes one, **restore** undoes. Spot colours mark the slot each player
inherited; fouls are shown on historical views. The row sums are the minutes projection.
Rookies carry a purple **R** badge with their games played this season (`R4`) and a purple edge on the name, on
tonight's grid and on a past game's alike (`Desk.rookie_of`). A rookie is a player whose first NBA season is this one
in nba_api's all-players list (`ivan.sports.nba.experience`, fetched by the nightly or `scripts/nba/experience.py
fetch`), matched by name without suffixes, then by last name on his team; a name the list lacks is a rookie when no
NBA game is on file before this season.
A past game's grid (the dropdown) has two small rows above the minutes: each team's running score at the end of every
3 minutes, the team's own row green while it leads and red while it trails, the final and home/away beside the name,
and the score at each minute on hover (`Desk.score_strip` = `pbp.game_score`, the stored play-by-play, no fetch;
overtime folds into the last cell like the grid; no rows when the game's feed is not on disk).

The **late news form** (`/team/<TEAM>/news`): player, call (out / starts / bench / back), then
(sim + choose, late swap, publish only). **Scenarios**: save the team's edits under a name, activate,
delete (`/scenario/*`); the one whose assumptions hold activates itself when the facts move.

Each card lists its open questions with the latest news on file (headline, source link, first seen).

## Players (view `players`)

The whole pool as one sortable table: team, position, salary, status, minutes, proj, pub, ownership,
value, flags.

## Backtest (view `backtest`, `/?view=backtest&dg=<draft group>`, `_nba_backtest.html`)

A past slate, read only (`ivan.sports.nba.slate_backtest`, terminal `scripts/nba/backtest_slate.py --draft-group N |
--date YYYY-MM-DD`). Built in about two seconds from what is on disk; the desk's working tables are not touched.

- **Tiles**: net and ROI, winnings on fees, entries cashed, the sim's claim (the chooser's ROI for the same entries,
  fee-weighted; red when it claimed far more than happened).
- **Contests**: fee, field, our entries found, best rank, cashed, the cash line, won, net, ROI, the claim, where the
  points came from. Our entries are the Entry IDs in the upload files the desk wrote (`chosen_entries.classic.*.csv`,
  `lateswap.*.csv`); the field is the contest's newest standings export, every lineup rescored from the day's box
  scores (DraftKings' names matched to the box by name, by name without suffixes, then by a unique last name; 99.6
  to 100% of lineups equal DraftKings' final points on the finished exports checked), ranks with ties sharing the
  payouts of the places they cover. Before the box scores arrive the export's own points stand.
- **ROI through the night**: what our entries were expected to make at each point: the chooser's claim before lock,
  every late swap run's expected ROI for the entries as held and after its swaps (`lateswap.<stamp>.json`, the
  expected payouts it computed with the games started so far), then the final. Late news moves a slate after the first
  lock; this shows when the expectation fell.
- **By tip**: each tip's games judged on the projection before lock and on the newest capture taken by that tip (ours
  or the archive's at-tip capture, up to 12 minutes after it): minutes and points MAE and bias of the rotation
  players, and the players the late news moved most, with what they scored.
- **Fields**: per contest the entries, unique lineups, the share duplicated, the most copies, the winning score, our
  entries' mean copies; the most-owned players of the largest contest, DraftKings' ownership against our projected
  ownership at the first tip and our exposure.
- **Notes at lock**: the desk's player notes that stood at lock.
- **Players by team**: minutes and points projected (our last capture before lock) against played, the miss, our
  exposure, the field's ownership (DraftKings' %drafted in the largest contest), our projected ownership, the note.
  Below each team's table, the game as played over our grid: the grid drawn on the desk (filed per slate under
  `data/nba/input/edits/<draft group>/<TEAM>.json`), else the grid rows the archive kept at the tip (`rotation_cells`,
  kept from 2026-10-29 by `archive_at_lock.py`), each minute marked on the floor in both, played but not in ours, or in
  ours and sat; the share of the 240 player-minutes on the floor in both.
- **Our lineups**: every entry, best first: rank, points, payout, copies of it in that contest's field, and the sim's
  ROI for that lineup at choose time (chooser runs from 2026-10-29 keep it per lineup with its players, `by_lineup`;
  matched to the entry through the chooser's upload file, so a late-swapped entry shows its claim at choose time).

The plan this is phase 1 of: [../nba/notes/slate_views_plan_2026-10-29.md](../nba/notes/slate_views_plan_2026-10-29.md).

## Pipeline panel (`_nba_pipeline.html`, `/pipeline/<step>`)

Fields: lineups, jobs, sims, field cap, fresh pool, the chooser's weights (from `config/nba.json`),
and for the late swap: live scoring (`ls_live`) and jam. Buttons **Generate**, **Sim**, **Choose**,
**All**, **Late swap**, **Field + ownership**; generate, sim, late swap and all publish first. Generate
writes no ownership; the sim builds the public field once, at the largest size any contest in the run
needs, every contest slices it, and the largest contest's field becomes the ownership column.
**Field + ownership** (`/pipeline/field` = `scripts/nba/field_ownership.py`, about 45s) builds the field on
the stored public projection and writes ownership from it, on demand; nothing else does on a slate day. The desk warns
when a team's mapping is incomplete. Jobs run one at a time in `ivan.web.jobs.JobRunner`; status at
`/pipeline/status`, cancel at `/pipeline/cancel/<job id>`, logs under `data/nba/jobs/`. An entries
upload box (`/entries`) replaces `data/nba/input/entries.csv`.

## Exposures (view `exposures`, `/pipeline/exposures` toggles entries vs pool)

The newest choose run covering the loaded slate's contests, every chosen player joined to the pool
now: entries, exposure, the field's ownership and leverage, projection at pick vs ours now, minutes,
pub, status (OUT struck in red), value; team totals; sortable.

## Late swap (view `lateswap`, `_nba_lateswap.html`)

Two buttons. **Fetch standings** (`/lateswap/fetch`, job `fetch_standings` = `scripts/nba/fetch_standings.py`;
producer only) downloads every entries contest's standings export through the logged-in Chrome profile
and keeps it under raw `dk/standings/<contest id>/`: press it a minute after a tip. **Run late swap**
(`/pipeline/lateswap` = `scripts/nba/lateswap.py`) with: candidates, field cap, sims; **standings**
(`ls_source`: "stored while current" = `--source auto`, the stored export while no game has tipped since it
was taken, a download otherwise; "DraftKings live"; "stored export"); **field** (`ls_field`: "one for all
contests" = `--field shared`, the biggest contest's export completed once and sampled per contest; "each
contest's own"); live scoring (`ls_live`). Each contest's heading shows its export's time, green while
current, amber "before the last tip" once a game has tipped since.

Below, the report of the newest `lateswap.<stamp>.json`: per contest and entry, the original and the swapped
lineup with the expected payout of each, and the exposure table before and after. The upload box
(`/lateswap/standings`; the only way on a consumer or limited desk, which runs `--source raw`) stores
standings CSVs exported by hand under the same raw path.

## Exports and downloads

- `/export/projections.csv`: the pool's projection (name, team, position, salary, minutes, proj, sd, pub,
  ownership, status). Rebuilt when the desk is stale; sent with `Cache-Control: no-store`. The producer exports
  its desk view (edits included, published or not). A consumer exports the **pool**: the laptop's projection as
  published with the consumer's own edits in it as of its last Publish, which is also what Generate reads, so the
  file and the lineups agree; an edit typed but not published is not in the file. The desk's own 600-world
  re-simulation (what the card shows before Publish) differs by a few tenths and is never the file. The filename's
  stamp is the pool's projection time. `scripts/nba/consumer_check.py` compares the file with the producer's pool.
- `/export/lineups.csv`: the loaded slate's generated pool, one row per lineup, PG..UTIL as names, or DraftKings ids with `?ids=1`;
  when the loaded slate has no lineups the file is one explanatory line, never another slate's lineups (that happened
  on the server on 2026-10-27); `?draft_group=` exports a named slate explicitly. The filename carries the slate id.
- `/download/<name>`: files from `data/nba/output/` (the chooser's CSVs, the late swap CSV).

## Consumer mode (the server)

Set by `IVAN_FEED_URL` in `.env`. The desk never fetches: Refresh pulls the feed (the same pull the
`ivan-nba-pull.timer` makes every 5 minutes, at once), the slate shown is the one the producer
published, the producer's pins and grids arrive as `data/nba/input/published/` and show as green
placeholders; the consumer's own edits (amber) sit on top in its own files, a blank box falls back to
the published value, Clear locks drops only its own layer, and a pull re-projects only the teams it
edited. The card says "published" and counts "N of your own". No props button. Refresh can only
bring what the laptop published: when nothing is new it says so with the time of the laptop's last
publish and its autopilot state, and the freshness strip has a **published** entry (the bundle's
time; hover for the last pull). The header badge "laptop autopilot on/off · desk|autopilot|nightly
published HH:MM" reads the pull state (`data/feed-state/nba.json`, `feed.status`).

To try to break a consumer desk without touching his: `deploy/sandbox_server.sh up` runs a second consumer
on the server, fed by the same pushed directory, reached over an ssh tunnel at `http://127.0.0.1:8002/nba/`
(see [../deployment.md](../deployment.md#a-sandbox-consumer-on-the-server)). To test this desk from the
producer without a browser on the other machine:
`scripts/nba/consumer_check.py` (see [../deployment.md](../deployment.md)) logs in with a cookie minted on
the server, walks the pages and the CSV exports through the public hostname or the tailnet, presses
Refresh and compares what the desk shows with the bundle the laptop published.

## Trying a branch's desk before it is merged

A worktree has no venv, `.env` or `data/`. Run its desk on another port with the main checkout's interpreter and
database, and `PYTHONPATH=src` so the worktree's code wins over the editable install:

```bash
cd ~/ivan/.claude/worktrees/<branch>
IVAN_NBA_DB_URL=sqlite:////home/sam-pratt/ivan/data/nba/nba_history.sqlite PYTHONPATH=src \
  ~/ivan/venv/bin/python scripts/nba/serve.py --port 8002      # http://127.0.0.1:8002/
```

Views that read the raw archive (a past game's score strip reads the stored play-by-play) need the main checkout's:
link it once with `mkdir -p data/nba && ln -s ~/ivan/data/nba/raw data/nba/raw` in the worktree. The backtest view
reads our upload files: link `data/nba/output` the same way. Raw files and outputs the branch's desk then writes land
in the main folders, as the live desk's would. The filed edits (`data/nba/input`) stay the worktree's own, so the
backtest's overlay shows grids drawn on the live desk only after the merge.

It shows the live slate. Its pins, grids, jobs and autopilot switch live under the worktree's own `data/`, so the
live desk's edits are not in it and nothing typed here reaches the live desk; with no `.env` it never pushes the
feed. **Refresh, Publish, Generate and Sim still write the shared database** (the pool, the projections, the
lineups), so on a slate day look and click but leave those to the live desk, or point `IVAN_NBA_DB_URL` at a copy
of the sqlite file (2.7 GB) for a free hand. A Refresh also reads the news, which only adds headlines.

## Login and public access

`ivan.web.auth` puts a login page at `<root>/login` when `IVAN_DESK_PASSWORD_HASH` (or the plain
`IVAN_DESK_PASSWORD`) is set: pbkdf2 hash, a signed cookie (secret in `data/.desk_secret`), per-address
and global throttling of guesses. On the server the desk is also on the internet through a Cloudflare
tunnel that forwards `/nba` only (`deploy/setup_cloudflare_tunnel.sh`); the NFL app's middleware
answers 404 to any other path that arrives with Cloudflare's headers, and the desk refuses to serve
through the tunnel without a password. The tunnel can cache a `.csv` URL at the edge: the CSV routes
send no-store headers (fix of 2026-10-26, see the memory note on the desk CSV cache); a bypass-cache
rule for the hostname is the belt and braces.
