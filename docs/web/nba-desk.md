# The NBA desk

`src/ivan/web/nba_desk.py` (FastAPI + Jinja + htmx, templates `src/ivan/web/templates/nba_desk.html`
and `_nba_*.html`). One page to manage a slate in the last minutes before lock, plus the pipeline
panel, the exposures view and the late swap. Served on its own by `scripts/nba/serve.py` (port 8001)
or mounted at `/nba` inside the NFL app (`ivan.web.app`), which is how the server runs it. Every link
carries the mount prefix (`root`). The flow of a day is [../nba/game-day.md](../nba/game-day.md); this
page is what is on the screen.

## Header

- **Slate picker**: a day, then that day's classic draft groups (from the DK lobby for today and
  tomorrow, from the archive for past days; the loaded slate is always listed). **Load** switches.
- **Refresh DK** (free) / **Refresh props** (about 12 credits per game; confirm dialog). The board shows the
  state until the new pool is projected (about 10 seconds). The public field and the ownership column are not
  rebuilt by a refresh: the sim does that, or the pipeline panel's **Field + ownership** button. On a consumer a
  single **Refresh** pulls the feed instead.
- **Publish**: applies any values still typed into the grids (not yet refigured) first, writes the projection into
  the pool, records a capture, publishes and pushes the feed. On a consumer (`ingest.consumer_publish`) the live
  bundle last pulled is applied again and only the slate's teams it edited are projected on top: the other teams
  are exactly as published, and clearing an edit then publishing gives the published numbers back. Generate, Sim
  and Late swap do the same before they queue.
- **Check news** and the **auto** box (every 5 minutes while the desk runs).
- **autopilot** box (producer only; `scripts/nba/autopilot.py on|off|status`): the capture tick refreshes,
  fetches props at the offsets and publishes + pushes every 15 minutes while nobody is editing here; the
  last tick's time shows next to it ("held" when it stood down, "failed" in red). See
  [../nba/game-day.md](../nba/game-day.md#2-refresh-through-the-day). On a consumer the same spot shows
  the laptop's state instead: autopilot on/off, who published last (desk, autopilot, nightly) and when.
- **new data · load** appears when the pool changed on disk since the page was built (a terminal
  ingest or project, a feed pull); it reloads without fetching.
- The questions count: Q/GTD/D/P players with no call yet, by team.
- Views: **slate** (the board), **players**, **late swap**, **exposures**.

## The board (view `slate`)

A games strip with tips, then team cards sorted by an alert score (bad 10, warn 3, info 1):

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
file" means nothing stored: press **Check news**. The team's last game is the newest box score on file, so
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

The **rotation grid** (`_nba_rotation.html`): players × 48 minutes. Paint cells to put a player on the
floor; the grid autosaves every stroke (`/rotation`). Row lock holds a row exactly; **Re-fill** fills the
unlocked rows around the locks (`/rotation/propose` with `source=propose`, or a past game's grid with
`game_id`); **Hold all** locks everything as drawn; **Clear locks**; **Map** adds a player the cut left off
(`/rotation/add`), **drop** removes one, **restore** undoes. Spot colours mark the slot each player
inherited; fouls are shown on historical views. The row sums are the minutes projection.

The **late news form** (`/team/<TEAM>/news`): player, call (out / starts / bench / back), then
(sim + choose, late swap, publish only). **Scenarios**: save the team's edits under a name, activate,
delete (`/scenario/*`); the one whose assumptions hold activates itself when the facts move.

Each card lists its open questions with the latest news on file (headline, source link, first seen).

## Players (view `players`)

The whole pool as one sortable table: team, position, salary, status, minutes, proj, pub, ownership,
value, flags.

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

## Login and public access

`ivan.web.auth` puts a login page at `<root>/login` when `IVAN_DESK_PASSWORD_HASH` (or the plain
`IVAN_DESK_PASSWORD`) is set: pbkdf2 hash, a signed cookie (secret in `data/.desk_secret`), per-address
and global throttling of guesses. On the server the desk is also on the internet through a Cloudflare
tunnel that forwards `/nba` only (`deploy/setup_cloudflare_tunnel.sh`); the NFL app's middleware
answers 404 to any other path that arrives with Cloudflare's headers, and the desk refuses to serve
through the tunnel without a password. The tunnel can cache a `.csv` URL at the edge: the CSV routes
send no-store headers (fix of 2026-10-26, see the memory note on the desk CSV cache); a bypass-cache
rule for the hostname is the belt and braces.
