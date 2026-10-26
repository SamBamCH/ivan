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
