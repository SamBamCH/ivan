# NBA: how the data moves, and the fastest path for late news

Written 2026-10-25 after a late-swap scramble. The point: there are two copies of the projection, and only one of
them feeds the sim.

## The two copies

| copy | where | who writes it | who reads it |
|---|---|---|---|
| **the desk's view** | memory of the serve.py process (`desk.proj`) | every grid paint + Apply, every pin, every starter call, Refresh | the team cards, the grids, the players table |
| **the pool** | `nba_player` in `data/nba/nba_history.sqlite` (minutes_proj, fpts_mean, fpts_std, ownership) | **Publish**, Refresh, a slate switch, `ingest.py`, `project.py`, `news.py`, and the desk's Generate / All (they publish first) | generate, sim, choose, **late swap**, the archive at tip, the feed to the server |

Everything downstream reads the pool. The desk shows you its own copy. Until Publish runs, the sim and the late swap
use whatever the pool held last, however the grid looks.

Inputs that survive on disk no matter what (they are read every time anything projects):

- `data/nba/input/minutes_overrides.csv`: minute pins, fppm pins, bands, starter calls. Written the instant you edit
  a pin on the desk; a locked grid row writes its row-sum here on Apply.
- `data/nba/input/rotations/<TEAM>.json`: the painted grid with its locked rows. Written on Apply only. Locked rows are
  held exactly by every later projection; unlocked rows are re-filled around them.
- On a slate switch both are filed under `data/nba/input/edits/<old draft group>/`.

## What each action does

| action | fetches | writes the pool | notes |
|---|---|---|---|
| paint a row | no | no | browser only until Apply; lost on any re-render (a pin, a refresh, a reload) |
| Apply | no | no | saves the grid + locks, pins the locked rows' minutes, re-projects the desk's copy |
| pin a minute / fppm / band / starter | no | no | saved to the overrides file at once, desk copy re-projected |
| **Publish** | no | **yes** | full projection at 2000 sims, capture rows, projections CSV, feed push. ~10-20 s |
| Refresh (dk) | DK pool, statuses, Rotowire | yes | one transaction; pins and locks kept; then re-projects |
| Refresh (props) | the above + player props (credits) | yes | same |
| pick a slate | DK + Rotowire for the new group | yes | files the old slate's pins and grids away |
| Generate / All (desk) | no | yes, first | publishes, then runs generate_lineups.py |
| Sim (desk) | no | **no** | runs sim.py against the pool as it is |
| Late swap (desk) | no | **no** | runs lateswap.py against the pool as it is |
| `scripts/nba/news.py "Name"` | no | yes (that team) | scratch + pins by prompt, writes the pool for the team |
| `scripts/nba/project.py` | no | yes | the overrides file + grids, whole pool, no capture row |
| `scripts/nba/ingest.py --project-only` | no | yes | same, plus a capture row |
| archive at tip (timer) | no | no | records the pool as it stands at each tip |

## The late-news path, in order

A player is ruled in or out ten minutes before a tip:

1. Desk, that team's card: paint or pin. If you paint, click **Apply** and read the flash: "rotation applied, N rows
   locked". If you only pin, nothing to click.
2. **Publish.** This is the step that moves your minutes into the pool. Skip it and the sim runs on the old numbers.
3. Run the sim or the late swap, desk or terminal:
   `scripts/nba/sim.py` then `scripts/nba/choose.py --no-prompt` before lock;
   `scripts/nba/lateswap.py` (or `--source raw` with an uploaded standings export) after lock.

From the terminal with no desk at all: `scripts/nba/news.py "Player Name" --accept` (scratch him, re-allocate the
team, write the pool) then the sim or the swap. Two commands.

## What bit us on 2026-10-25

- The checkout had been left on a review branch, so the desk ran without the slate-switch housekeeping. Yesterday's
  Philadelphia grid stayed in `rotations/` and its three locked rows overrode tonight's pins until it was filed away by
  hand.
- Paint without Apply is lost on the next re-render. Nothing warns.
- Sim and late swap from the desk do not publish first, so the grid and the sim disagreed.
- The picker lists DraftKings' lobby, which drops a draft group near lock; the loaded slate was still fine but looked
  gone.

## Changed the same evening (2026-10-25)

1. Sim and late swap publish first, like Generate. Every job but choose starts from what the desk shows.
2. The grid saves itself on every stroke and every lock click. The Apply button is now "Re-fill" (re-fill the unlocked rows now).
3. "Hold all" on the team's grid locks every row as drawn; a refresh can then only change statuses and props. "Clear locks" undoes it.
4. The picker always lists the loaded slate, lobby or not.
5. A "late news" form on the team card: player + call (out / starts / bench / back) + what to run next (sim + choose before lock, late swap after, or publish only). One request: pin, re-allocate, publish, queue.
6. The grid shows **proj** (ours: lineups and the sim) next to **pub** (the public projection: the field and ownership). Our edits never move pub.

## Loading a slate (2026-10-25 evening)

The picker is a day, then that day's slates. A lobby slate (today, tomorrow) is fetched from DraftKings and projected,
as before. A past slate is restored from its archive capture (`ingest.restore_from_archive`: the last capture before
lock, else the last one; the pool as it was, its props and lines, then the projection with the model as it is now;
nothing fetched, no capture row written). Either way the pins and drawn grids of the slate you leave are filed under
`data/nba/input/edits/<draft group>/` and the slate you load gets its own back, so switching away and back loses
nothing; a failed switch puts the files back where they were. A restore of a full slate takes about a minute (the props).

## Questions and news (2026-10-25 evening)

A questionable player (DK status Q / GTD / D / P, not ruled out) is an open question until you make a call (a minutes
pin or a starter call). The header counts the slate's questions by team; "no questionable players" means good to go at
lock barring a surprise lineup. Each team card lists its questions with the player's usual minutes and the latest news
on file: Rotowire's NBA news page (`ivan.sports.nba.newsfeed`, `scripts/nba/newsfeed.py check|show`), fetched only by
the Check news button, the auto checkbox (every 5 minutes while the desk runs, off by default) or the script; each
headline links to the report it cites (the Underdog / Shams / beat tweet). Rotowire gives a date, not a time, so our
first-seen time is the clock. "new since mapped" marks news that arrived after the team's mapping was saved.

## Scenarios, one slate at a time, exposures (2026-10-26)

**One slate.** Loading a slate (fetch or restore) deletes the lineups, pool snapshots, sim results and contest reports
of every other draft group (`ingest.isolate_slate`); the loaded slate's own rows stay, so coming back to a slate finds
its work. The pool, props, captures, news and box scores are never touched.

**Scenarios** (`ivan.sports.nba.scenarios`, `data/nba/input/scenarios/<TEAM>/`): a team's edits (its pins + its drawn
grid) saved under a name with the assumption they encode, read off the calls: a scratch pin on a player DK does not
list as OUT = "out", a starter call = "starts"/"bench", a questionable player pinned or locked with minutes = "in".
"Save current as…" on the card; click a name to activate (the team's pins and grid are swapped in, the team
re-projects); × deletes. The pick is automatic: when the facts move (a refresh changed a status or the lineup, or a call
typed into the late-news form, which stays a fact until DK's own status for that player moves), the one scenario whose
assumptions all hold activates itself and the desk says so; two matches or none are reported, not acted on. The generate
panel's "fresh pool" box unticked adds a scenario's lineups to the existing pool, so both worlds' lineups sit in the
pool before lock and the switch at lock costs a publish and a sim.

**Exposures** (the exposures view): the newest choose run that covered this slate's contests, every chosen player joined
to the pool as it stands: entries, exposure, the field's ownership and the leverage against it, the projection at pick
and ours now with the delta, minutes, the public number, status (a chosen player now OUT is struck in red at the top
line), value; sortable by any column; team exposure totals.

## Two desks, one projection (2026-10-26)

Sam runs the minutes. His Publish writes the pool and pushes the live bundle, which now carries his edits too
(`minutes_overrides.csv`, `rotations/`). The brother's desk (the server, a consumer) takes the pool as published on
Refresh (`feed.pull`): no re-projection, the numbers are exactly Sam's. His edits land under
`data/nba/input/published/` as the base layer; anything the brother types goes into his own files on top (the desk writes
only its own layer), a blank box of his falls back to the published value, Clear locks drops his pins and grid for that
team, and a pull re-projects only the teams he edited (his edits on top of the published projection) and leaves the rest
as published. On his grid a published pin is a green placeholder; his own pins are the amber-bordered values; the card
says "published" and counts "N of your own". The server's 5-minute pull timer follows the same rule.

Verified 2026-10-26 on a separate checkout run as the consumer (the NFL app mounting the desk at /nba behind the login,
IVAN_FEED_URL set): login, the games strip and one card, no link without the /nba prefix, Refresh pulls the feed with the
pool equal to the published pool for all 241 players, published pins as green placeholders, Refigure keeps his value
local, Clear locks returns to the published, paint autosaves through the mount, Map, Check news, late news reloads the
page logged in on the same team, scenarios save and delete, every tab renders, the slate picker fills, a second Refresh
says "unchanged". Bugs found and fixed on the way: Refigure copied the published pins into his file; Map's fingerprint
was taken before the pins; flash messages were cleared by renders that never showed them; the late-news response consumed
its own message before the page reload.
